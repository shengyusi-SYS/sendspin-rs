//! Non-realtime canonical sync planning and a bounded atomic plan snapshot.
//! Readers never retry a concurrent publication; fallback retains actual phase.
use crate::audio::sync_correction::{
    CorrectionPlanner, CorrectionSchedule, EngageGate, SyncErrorFilter,
};
use crate::sync::ClockSync;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Default)]
struct PlanPayload {
    timeline: AtomicU64,
    word: AtomicU64,
}
impl PlanPayload {
    fn store(&self, timeline: u64, schedule: Option<CorrectionSchedule>) {
        let mut word = 0;
        match schedule {
            None => word |= 1 << 62,
            Some(schedule) => {
                assert!(!schedule.reanchor);
                assert!(schedule.insert_every_n_frames == 0 || schedule.drop_every_n_frames == 0);
                word |= u64::from(
                    schedule
                        .insert_every_n_frames
                        .max(schedule.drop_every_n_frames),
                );
                if schedule.insert_every_n_frames != 0 {
                    word |= 1 << 32;
                }
            }
        }
        self.timeline.store(timeline, Ordering::Relaxed);
        self.word.store(word, Ordering::Relaxed);
    }
    // The caller validates its publication revision after reading this payload.
    fn load(&self, timeline: u64) -> Option<Option<CorrectionSchedule>> {
        let tag = self.timeline.load(Ordering::Relaxed);
        let word = self.word.load(Ordering::Relaxed);
        if tag != timeline {
            return None;
        }
        if word & (1 << 62) != 0 {
            return Some(None);
        }
        let interval = word as u32;
        Some(Some(if word & (1 << 32) != 0 {
            CorrectionSchedule {
                insert_every_n_frames: interval,
                ..CorrectionSchedule::default()
            }
        } else {
            CorrectionSchedule {
                drop_every_n_frames: interval,
                ..CorrectionSchedule::default()
            }
        }))
    }
}
#[derive(Clone, Copy)]
pub(super) struct Measurement {
    pub(super) settled_seen: bool,
    pub(super) timeline: u64,
    pub(super) serial: u64,
    pub(super) source_cursor_us: i64,
    pub(super) presentation: Option<Instant>,
    pub(super) actual_schedule: CorrectionSchedule,
}

struct Feedback {
    sample_rate: u32,
    last_error: Option<(u64, i64, i64)>,
    settled_once: bool,
    timeline: Option<u64>,
    seen: Option<u64>,
    filter: SyncErrorFilter,
    gate: EngageGate,
}

impl Feedback {
    fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            last_error: None,
            settled_once: false,
            timeline: None,
            seen: None,
            filter: SyncErrorFilter::new(),
            gate: EngageGate::new(),
        }
    }

    fn compute_observation(
        &mut self,
        sync: &ClockSync,
        input: Measurement,
        delay_us: u64,
        timeline: u64,
    ) -> Option<Option<CorrectionSchedule>> {
        if input.timeline != timeline {
            return None;
        }
        if self.timeline != Some(timeline) {
            self.timeline = Some(timeline);
            self.seen = None;
            self.filter.reset();
            self.gate.reset();
        }
        match self.observe(sync, input, delay_us) {
            Ok(Some((_, schedule))) => Some(Some(schedule)),
            Err(()) => Some(None),
            _ => None,
        }
    }

    pub(super) fn observe(
        &mut self,
        sync: &ClockSync,
        input: Measurement,
        delay_us: u64,
    ) -> Result<Option<(i64, CorrectionSchedule)>, ()> {
        self.last_error = None;
        if self.seen.is_some_and(|seen| input.serial <= seen) {
            return Ok(None);
        }
        self.seen = Some(input.serial);
        // This is once per callback/worker lifetime in production, not once
        // per generation or every time the model regains availability.
        let settled = sync.is_settled();
        if input.settled_seen && !self.settled_once {
            self.settled_once = true;
            self.filter.reset();
            self.gate.reset();
        }
        let Some(expected) =
            sync.server_to_local_instant_with_latency(input.source_cursor_us, delay_us)
        else {
            if input.actual_schedule.is_correcting() {
                self.filter.reset();
                self.gate.reset();
                return Err(()); // Explicit plan invalidation, not fallback.
            }
            return Ok(None);
        };
        let Some(presentation) = input.presentation else {
            return Ok(None);
        };
        let raw = if presentation >= expected {
            presentation.duration_since(expected).as_micros() as i64
        } else {
            -(expected.duration_since(presentation).as_micros() as i64)
        };
        let filtered = self.filter.update(raw);
        self.last_error = Some((input.serial, raw, filtered));
        let correcting = input.actual_schedule.is_correcting();
        let planned = CorrectionPlanner::new().plan(filtered, self.sample_rate, correcting);
        let admitted = self.gate.admit(planned, correcting, self.filter.is_warm());
        Ok(Some((
            raw,
            if (settled && input.settled_seen) || admitted.reanchor {
                admitted
            } else {
                CorrectionSchedule::default()
            },
        )))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ModelValidity {
    Invalid,
    SameClock,
    Sampled(i64),
    SampledUnsettled(i64),
}
impl ModelValidity {
    pub(super) fn from_sync(sync: &ClockSync) -> Self {
        let health = sync.health();
        if !health.synchronized {
            Self::Invalid
        } else if let Some(last) = health.last_valid_t4_us {
            if health.settled {
                Self::Sampled(last)
            } else {
                Self::SampledUnsettled(last)
            }
        } else {
            Self::SameClock
        }
    }
    pub(super) fn allows(self, now: i64) -> bool {
        match self {
            Self::Invalid => false,
            Self::SameClock => true,
            Self::Sampled(last) | Self::SampledUnsettled(last) => {
                (0..=5_000_000).contains(&(i128::from(now) - i128::from(last)))
            }
        }
    }
}

#[derive(Default)]
pub(super) struct PlanMailbox {
    revision: AtomicU64,
    plan: PlanPayload,
    validity_kind: std::sync::atomic::AtomicU8,
    last_sample: std::sync::atomic::AtomicI64,
    acknowledged_reset: AtomicU64,
}
impl PlanMailbox {
    pub(super) fn publish_acknowledged(
        &self,
        timeline: u64,
        plan: Option<CorrectionSchedule>,
        validity: ModelValidity,
        reset_epoch: u64,
    ) {
        let revision = self.revision.load(Ordering::Relaxed);
        assert_eq!(revision % 2, 0, "one serialized publisher");
        let committed = revision.checked_add(2).unwrap();
        self.revision.store(revision + 1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        self.plan.store(timeline, plan);
        let (kind, last) = match validity {
            ModelValidity::Invalid => (0, 0),
            ModelValidity::SameClock => (1, 0),
            ModelValidity::Sampled(last) => (2, last),
            ModelValidity::SampledUnsettled(last) => (3, last),
        };
        self.last_sample.store(last, Ordering::Relaxed);
        self.validity_kind.store(kind, Ordering::Relaxed);
        self.acknowledged_reset
            .store(reset_epoch, Ordering::Relaxed);
        self.revision.store(committed, Ordering::Release);
    }
    pub(super) fn read(
        &self,
        timeline: u64,
    ) -> Option<(u64, Option<CorrectionSchedule>, ModelValidity, u64)> {
        let before = self.revision.load(Ordering::Acquire);
        if before == 0 || before % 2 != 0 {
            return None;
        }
        let plan = self.plan.load(timeline)?;
        let last = self.last_sample.load(Ordering::Relaxed);
        let kind = self.validity_kind.load(Ordering::Relaxed);
        let reset_epoch = self.acknowledged_reset.load(Ordering::Relaxed);
        // Pairs with the writer's fence before its atomic payload writes. No
        // non-atomic seqlock reads, callback retry, allocation or payload drop.
        std::sync::atomic::fence(Ordering::Acquire);
        if self.revision.load(Ordering::Relaxed) != before {
            return None;
        }
        let validity = match kind {
            0 => ModelValidity::Invalid,
            1 => ModelValidity::SameClock,
            2 => ModelValidity::Sampled(last),
            3 => ModelValidity::SampledUnsettled(last),
            _ => unreachable!(),
        };
        Some((before, plan, validity, reset_epoch))
    }
}

#[derive(Default)]
pub(super) struct PlanReader {
    pub(super) settled_seen: bool,
    cached: Option<(u64, u64, Option<CorrectionSchedule>, ModelValidity, u64)>,
    invalidated: Option<u64>,
    reset_epoch: u64,
}
impl PlanReader {
    pub(super) fn reset_epoch(&self) -> u64 {
        self.reset_epoch
    }
    pub(super) fn read(
        &mut self,
        mailbox: &PlanMailbox,
        timeline: u64,
        now: i64,
        actual_schedule: CorrectionSchedule,
    ) -> Option<Option<CorrectionSchedule>> {
        if let Some((revision, plan, validity, ack)) = mailbox.read(timeline) {
            // Model observation is independent of whether its plan has the
            // callback's reset acknowledgment. Keep the first observation sticky.
            if validity.allows(now)
                && matches!(
                    validity,
                    ModelValidity::SameClock | ModelValidity::Sampled(_)
                )
            {
                self.settled_seen = true;
            }
            if ack == self.reset_epoch {
                self.cached = Some((revision, timeline, plan, validity, ack));
            }
        }
        self.cached
            .and_then(|(revision, tag, plan, validity, ack)| {
                if tag != timeline || ack != self.reset_epoch {
                    return Some(None);
                }
                if self.invalidated == Some(revision) {
                    return Some(None);
                }
                if !validity.allows(now) {
                    self.invalidated = Some(revision);
                    // Fallback retains the applied schedule even when the cached
                    // candidate is default. Reset the history of what we clear.
                    if actual_schedule.is_correcting() {
                        self.reset_epoch = self.reset_epoch.checked_add(1).unwrap();
                    }
                    Some(None)
                } else {
                    Some(plan)
                }
            })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Observation {
    /// Captured with the device observation, never substituted at worker dequeue.
    pub(super) delay_us: u64,
    pub(super) latency_floor: Option<Duration>,
    pub(super) input: Measurement,
    pub(super) reset_epoch: u64,
}

pub(super) struct FeedbackWorker {
    feedback: Feedback,
    reset_epoch: u64,
}
impl FeedbackWorker {
    pub(super) fn new(sample_rate: u32) -> Self {
        Self {
            feedback: Feedback::new(sample_rate),
            reset_epoch: 0,
        }
    }

    /// Diagnostic values from the last accepted measured observation only.
    /// The serial binds them to the callback snapshot that the worker publishes.
    pub(super) fn last_error(&self) -> Option<(u64, i64, i64)> {
        self.feedback.last_error
    }
    pub(super) fn model_changed(&mut self, sync: &ClockSync, timeline: u64, plans: &PlanMailbox) {
        // Called by the same non-RT owner after canonical model mutations.
        // Availability alone cannot restore a plan or acknowledge a reader reset.
        let validity = ModelValidity::from_sync(sync);
        let (plan, ack) = plans
            .read(timeline)
            .map(|(_, plan, _, ack)| (plan, ack))
            .unwrap_or((None, self.reset_epoch));
        plans.publish_acknowledged(
            timeline,
            if validity == ModelValidity::Invalid {
                None
            } else {
                plan
            },
            validity,
            ack,
        );
    }

    pub(super) fn process(
        &mut self,
        sync: &ClockSync,
        input: Observation,
        timeline: u64,
        plans: &PlanMailbox,
    ) -> Option<Observation> {
        self.feedback.last_error = None;
        if input.input.timeline != timeline || input.reset_epoch < self.reset_epoch {
            return None;
        }
        let reset = input.reset_epoch > self.reset_epoch;
        if reset {
            self.reset_epoch = input.reset_epoch;
            self.feedback.filter.reset();
            self.feedback.gate.reset();
            self.feedback.seen = None;
        }
        let decision =
            self.feedback
                .compute_observation(sync, input.input, input.delay_us, timeline);
        if decision.is_some_and(|plan| plan.is_some_and(|schedule| schedule.reanchor)) {
            // The one non-RT controller consumes this request. It is not a
            // device cadence and must not enter the atomic schedule mailbox.
            return input.latency_floor.map(|_| input);
        }
        // A reset observed on fallback can acknowledge a default plan without
        // pretending to have a measured error or a prewarmed correction.
        if let Some(plan) =
            decision.or_else(|| reset.then_some(Some(CorrectionSchedule::default())))
        {
            plans.publish_acknowledged(
                timeline,
                plan,
                ModelValidity::from_sync(sync),
                self.reset_epoch,
            );
        }
        None
    }

    pub(super) fn reanchor_committed(
        &mut self,
        timeline: u64,
        sync: &ClockSync,
        plans: &PlanMailbox,
    ) {
        self.feedback.timeline = Some(timeline);
        self.feedback.seen = None;
        self.feedback.filter.reset();
        self.feedback.gate.reset();
        plans.publish_acknowledged(
            timeline,
            Some(CorrectionSchedule::default()),
            ModelValidity::from_sync(sync),
            self.reset_epoch,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::Clock;
    use std::sync::Arc;

    struct PinnedClock(Instant);
    impl Clock for PinnedClock {
        fn now_micros(&self) -> i64 {
            0
        }
        fn micros_to_instant(&self, us: i64) -> Option<Instant> {
            if us >= 0 {
                self.0.checked_add(Duration::from_micros(us as u64))
            } else {
                self.0.checked_sub(Duration::from_micros(us.unsigned_abs()))
            }
        }
    }

    #[test]
    fn renderer_realtime_feedback_uses_captured_delay_and_rejects_old_timeline() {
        let origin = Instant::now();
        let sync = ClockSync::new_same_clock(Arc::new(PinnedClock(origin)));
        let plans = PlanMailbox::default();
        let mut worker = FeedbackWorker::new(48_000);
        // Queue these device observations before processing them. No setter's
        // later delay value is an argument to processing: each pair stays intact.
        let observations: Vec<_> = (1..=180)
            .map(|serial| Observation {
                delay_us: 10_000,
                latency_floor: Some(Duration::from_millis(10)),
                reset_epoch: 0,
                input: Measurement {
                    settled_seen: true,
                    timeline: 7,
                    serial,
                    source_cursor_us: 40_000,
                    presentation: Some(origin + Duration::from_millis(30)),
                    actual_schedule: CorrectionSchedule::default(),
                },
            })
            .collect();
        for observation in &observations {
            assert!(worker.process(&sync, *observation, 7, &plans).is_none());
            assert_eq!(worker.last_error(), Some((observation.input.serial, 0, 0)));
        }
        assert_eq!(
            plans.read(7).unwrap().1,
            Some(CorrectionSchedule::default())
        );
        let revision = plans.read(7).unwrap().0;
        assert!(worker.process(&sync, observations[0], 8, &plans).is_none());
        assert_eq!(worker.last_error(), None);
        assert_eq!(plans.read(7).unwrap().0, revision);
        assert!(plans.read(8).is_none());
    }

    #[test]
    fn renderer_realtime_feedback_expired_plan_waits_for_matching_reset_ack() {
        let plans = PlanMailbox::default();
        let mut reader = PlanReader::default();
        let plan = CorrectionSchedule {
            drop_every_n_frames: 500,
            ..CorrectionSchedule::default()
        };
        plans.publish_acknowledged(0, Some(plan), ModelValidity::Sampled(0), 0);
        assert_eq!(reader.read(&plans, 0, 5_000_000, plan), Some(Some(plan)));
        assert_eq!(reader.read(&plans, 0, 5_000_001, plan), Some(None));
        let reset = reader.reset_epoch();
        assert_eq!(reset, 1);
        plans.publish_acknowledged(0, Some(plan), ModelValidity::Sampled(5_000_001), 0);
        assert_eq!(
            reader.read(&plans, 0, 5_000_001, CorrectionSchedule::default()),
            Some(None)
        );
        plans.publish_acknowledged(
            0,
            Some(CorrectionSchedule::default()),
            ModelValidity::Sampled(5_000_001),
            reset,
        );
        assert_eq!(
            reader.read(&plans, 0, 5_000_001, CorrectionSchedule::default()),
            Some(Some(CorrectionSchedule::default()))
        );
    }
}

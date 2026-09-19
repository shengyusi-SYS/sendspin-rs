//! Non-RT calculation probe. Transport, model revisions and reanchor commit
//! are intentionally not represented by this arithmetic/measurement test.
use super::*;
use crate::audio::sync_correction::{EngageGate, SyncErrorFilter};
use crate::sync::{Clock, ClockSync};
use std::time::{Duration, Instant};

// The standalone plan experiment and validated mailbox share the same wide
// payload. Each publisher uses only its own revision; validated publication
// protects plan, timeline, validity and reset acknowledgment together.
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
#[derive(Default)]
pub(super) struct PlanMailbox {
    revision: AtomicU64,
    payload: PlanPayload,
}
impl PlanMailbox {
    pub(super) fn publish(&self, timeline: u64, schedule: Option<CorrectionSchedule>) {
        let before = self.revision.load(Ordering::Relaxed);
        let after = before.checked_add(2).expect("scope plan revision");
        self.revision.store(before + 1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        self.payload.store(timeline, schedule);
        self.revision.store(after, Ordering::Release);
    }
    pub(super) fn read(&self, timeline: u64) -> Option<Option<CorrectionSchedule>> {
        let before = self.revision.load(Ordering::Acquire);
        if before == 0 || before % 2 != 0 {
            return None;
        }
        let plan = self.payload.load(timeline)?;
        std::sync::atomic::fence(Ordering::Acquire);
        (self.revision.load(Ordering::Relaxed) == before).then_some(plan)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Observation {
    pub(super) settled_seen: bool,
    pub(super) timeline: u64,
    pub(super) serial: u64,
    pub(super) source_cursor_us: i64,
    pub(super) presentation: Option<Instant>,
    pub(super) actual_schedule: CorrectionSchedule,
}

pub(super) struct Feedback {
    settled_once: bool,
    timeline: Option<u64>,
    seen: Option<u64>,
    filter: SyncErrorFilter,
    gate: EngageGate,
}

impl Feedback {
    pub(super) fn new() -> Self {
        Self {
            settled_once: false,
            timeline: None,
            seen: None,
            filter: SyncErrorFilter::new(),
            gate: EngageGate::new(),
        }
    }

    pub(super) fn publish_observation(
        &mut self,
        sync: &ClockSync,
        input: Observation,
        delay_us: u64,
        timeline: u64,
        mailbox: &PlanMailbox,
    ) {
        if let Some(plan) = self.compute_observation(sync, input, delay_us, timeline) {
            if !plan.is_some_and(|schedule| schedule.reanchor) {
                mailbox.publish(timeline, plan);
            }
        }
    }

    fn compute_observation(
        &mut self,
        sync: &ClockSync,
        input: Observation,
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
        input: Observation,
        delay_us: u64,
    ) -> Result<Option<(i64, CorrectionSchedule)>, ()> {
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
        let correcting = input.actual_schedule.is_correcting();
        let planned = CorrectionPlanner::new().plan(filtered, 1_000, correcting);
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

struct PinnedClock(Instant);
impl Clock for PinnedClock {
    fn now_micros(&self) -> i64 {
        0
    }
    fn micros_to_instant(&self, micros: i64) -> Option<Instant> {
        if micros >= 0 {
            self.0.checked_add(Duration::from_micros(micros as u64))
        } else {
            self.0
                .checked_sub(Duration::from_micros(micros.unsigned_abs()))
        }
    }
}

#[test]
fn realtime_handoff_probe_feedback_pairs_actual_cursor_and_deduplicates_measurements() {
    let origin = Instant::now();
    let sync = ClockSync::new_same_clock(Arc::new(PinnedClock(origin)));
    let mut feedback = Feedback {
        settled_once: false,
        timeline: None,
        seen: None,
        filter: SyncErrorFilter::new(),
        gate: EngageGate::new(),
    };
    let mut input = Observation {
        settled_seen: true,
        timeline: 0,
        serial: 1,
        source_cursor_us: 1_000_000,
        presentation: Some(origin + Duration::from_micros(990_000)),
        actual_schedule: CorrectionSchedule::default(),
    };
    // expected = source 1s - output delay 20ms; observed 990ms => +10ms.
    assert_eq!(
        feedback.observe(&sync, input, 20_000),
        Ok(Some((10_000, CorrectionSchedule::default())))
    );
    for _ in 0..200 {
        assert_eq!(feedback.observe(&sync, input, 20_000), Ok(None));
    }
    assert!(!feedback.filter.is_warm());
    input.serial = 2;
    input.presentation = None;
    assert_eq!(feedback.observe(&sync, input, 20_000), Ok(None));
    assert!(!feedback.filter.is_warm());
    // 100 distinct measured observations total must still be cold, regardless
    // of worker polls or the interleaved fallback observation.
    for serial in 3..=101 {
        input.serial = serial;
        input.presentation = Some(origin + Duration::from_micros(990_000));
        assert_eq!(
            feedback.observe(&sync, input, 20_000),
            Ok(Some((10_000, CorrectionSchedule::default())))
        );
    }
    assert!(!feedback.filter.is_warm());
    input.serial = 102;
    let (_, schedule) = feedback.observe(&sync, input, 20_000).unwrap().unwrap();
    assert!(feedback.filter.is_warm());
    assert_eq!(schedule.drop_every_n_frames, 500);
    input.serial = 103;
    input.presentation = None;
    assert_eq!(feedback.observe(&sync, input, 20_000), Ok(None));
    assert!(
        feedback.filter.is_warm(),
        "fallback retains valid measurement history"
    );
}

#[test]
fn realtime_handoff_probe_feedback_model_unavailable_clears_active_history_even_on_fallback() {
    let origin = Instant::now();
    let clock: Arc<dyn Clock> = Arc::new(PinnedClock(origin));
    let valid = ClockSync::new_same_clock(Arc::clone(&clock));
    let unavailable = ClockSync::new(clock);
    let mut feedback = Feedback {
        settled_once: false,
        timeline: None,
        seen: None,
        filter: SyncErrorFilter::new(),
        gate: EngageGate::new(),
    };
    let mut input = Observation {
        settled_seen: true,
        timeline: 0,
        serial: 0,
        source_cursor_us: 1_000_000,
        presentation: Some(origin + Duration::from_micros(990_000)),
        actual_schedule: CorrectionSchedule::default(),
    };
    for serial in 1..=101 {
        input.serial = serial;
        let (_, schedule) = feedback.observe(&valid, input, 20_000).unwrap().unwrap();
        input.actual_schedule = schedule;
    }
    assert!(input.actual_schedule.is_correcting());
    assert!(feedback.filter.is_warm());
    input.serial = 102;
    input.presentation = None;
    assert_eq!(
        feedback.observe(&unavailable, input, 20_000),
        Err(()),
        "unavailable model must invalidate an active plan, unlike timestamp fallback"
    );
    assert!(!feedback.filter.is_warm());
    input.serial = 103;
    input.actual_schedule = CorrectionSchedule::default();
    input.presentation = Some(origin + Duration::from_micros(990_000));
    assert_eq!(
        feedback.observe(&valid, input, 20_000),
        Ok(Some((10_000, CorrectionSchedule::default())))
    );
}

#[test]
fn realtime_handoff_probe_feedback_delayed_old_timeline_cannot_rewarm_or_publish_new_plan() {
    let origin = Instant::now();
    let sync = ClockSync::new_same_clock(Arc::new(PinnedClock(origin)));
    let plans = PlanMailbox::default();
    let mut feedback = Feedback::new();
    let mut input = Observation {
        settled_seen: true,
        timeline: 0,
        serial: 0,
        source_cursor_us: 1_000_000,
        presentation: Some(origin + Duration::from_micros(1_010_000)),
        actual_schedule: CorrectionSchedule::default(),
    };
    for serial in 1..=101 {
        input.serial = serial;
        feedback.publish_observation(&sync, input, 0, 0, &plans);
    }
    assert_eq!(plans.read(0).unwrap().unwrap().drop_every_n_frames, 500);
    input.serial = 102;
    feedback.publish_observation(&sync, input, 0, 1, &plans);
    assert_eq!(plans.read(1), None, "old observation must not be relabeled");
    input.timeline = 1;
    for serial in 103..=202 {
        input.serial = serial;
        feedback.publish_observation(&sync, input, 0, 1, &plans);
        assert_eq!(plans.read(1), Some(Some(CorrectionSchedule::default())));
    }
    assert!(!feedback.filter.is_warm());
    let mut delayed = input;
    delayed.timeline = 0;
    delayed.serial = 999;
    feedback.publish_observation(&sync, delayed, 0, 1, &plans);
    assert_eq!(feedback.seen, Some(202));
    assert!(!feedback.filter.is_warm());
    input.serial = 203;
    feedback.publish_observation(&sync, input, 0, 1, &plans);
    assert_eq!(plans.read(1).unwrap().unwrap().drop_every_n_frames, 500);
    assert_eq!(feedback.seen, Some(203));
}

#[test]
fn realtime_handoff_probe_feedback_bounded_transport_gap_preserves_measured_history() {
    let origin = Instant::now();
    let sync = ClockSync::new_same_clock(Arc::new(PinnedClock(origin)));
    let plans = PlanMailbox::default();
    let mut feedback = Feedback::new();
    let mut input = Observation {
        settled_seen: true,
        timeline: 0,
        serial: 0,
        source_cursor_us: 1_000_000,
        presentation: Some(origin + Duration::from_micros(1_010_000)),
        actual_schedule: CorrectionSchedule::default(),
    };
    for serial in 1..=101 {
        input.serial = serial;
        feedback.publish_observation(&sync, input, 0, 0, &plans);
    }
    input.actual_schedule = plans.read(0).unwrap().unwrap();
    assert!(input.actual_schedule.is_correcting());
    let (mut tx, mut rx) = RingBuffer::new(2);
    for serial in [102, 103] {
        input.serial = serial;
        assert!(tx.push(input).is_ok());
    }
    input.serial = 104;
    assert!(
        matches!(tx.push(input), Err(PushError::Full(_))),
        "one nonblocking attempt on full transport"
    );
    for _ in 0..2 {
        feedback.publish_observation(&sync, rx.pop().unwrap(), 0, 0, &plans);
    }
    assert!(rx.pop().is_err());
    // A transport gap is not a model or timeline invalidation. Fallback
    // skips measurement just as the production callback does.
    input.serial = 105;
    input.presentation = None;
    assert!(tx.push(input).is_ok());
    feedback.publish_observation(&sync, rx.pop().unwrap(), 0, 0, &plans);
    assert!(feedback.filter.is_warm());
    assert_eq!(plans.read(0).unwrap().unwrap().drop_every_n_frames, 500);
    // A second gap followed by a measurement must not require rewarming.
    input.serial = 107;
    input.presentation = Some(origin + Duration::from_micros(1_010_000));
    assert!(tx.push(input).is_ok());
    feedback.publish_observation(&sync, rx.pop().unwrap(), 0, 0, &plans);
    assert!(feedback.filter.is_warm());
    assert_eq!(plans.read(0).unwrap().unwrap().drop_every_n_frames, 500);

    // Gaps and duplicate/out-of-order deliveries do not invent measurements.
    let mut cold = Feedback::new();
    input.actual_schedule = CorrectionSchedule::default();
    for serial in (1..=199).step_by(2) {
        input.serial = serial;
        cold.publish_observation(&sync, input, 0, 0, &plans);
        cold.publish_observation(&sync, input, 0, 0, &plans);
        input.serial = serial - 1;
        cold.publish_observation(&sync, input, 0, 0, &plans);
    }
    assert!(!cold.filter.is_warm(), "only 100 distinct measured samples");
    assert_eq!(plans.read(0), Some(Some(CorrectionSchedule::default())));
    input.serial = 201;
    cold.publish_observation(&sync, input, 0, 0, &plans);
    assert!(cold.filter.is_warm());
    assert_eq!(plans.read(0).unwrap().unwrap().drop_every_n_frames, 500);
}

// Candidate validity publication: the callback receives endpoint NOW, not the
// future DAC presentation time. These are the existing ClockSync health rules.
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
pub(super) struct ValidatedPlanMailbox {
    revision: AtomicU64,
    plan: PlanPayload,
    validity_kind: std::sync::atomic::AtomicU8,
    last_sample: std::sync::atomic::AtomicI64,
    acknowledged_reset: AtomicU64,
}
impl ValidatedPlanMailbox {
    pub(super) fn publish(
        &self,
        timeline: u64,
        plan: Option<CorrectionSchedule>,
        validity: ModelValidity,
    ) {
        self.publish_after_open(timeline, plan, validity, 0, || {});
    }
    pub(super) fn publish_acknowledged(
        &self,
        timeline: u64,
        plan: Option<CorrectionSchedule>,
        validity: ModelValidity,
        reset_epoch: u64,
    ) {
        self.publish_after_open(timeline, plan, validity, reset_epoch, || {});
    }
    pub(super) fn publish_after_open(
        &self,
        timeline: u64,
        plan: Option<CorrectionSchedule>,
        validity: ModelValidity,
        reset_epoch: u64,
        after_open: impl FnOnce(),
    ) {
        let revision = self.revision.load(Ordering::Relaxed);
        assert_eq!(revision % 2, 0, "one serialized publisher");
        let committed = revision.checked_add(2).unwrap();
        self.revision.store(revision + 1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        after_open();
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
pub(super) struct ValidatedPlanReader {
    pub(super) settled_seen: bool,
    cached: Option<(u64, u64, Option<CorrectionSchedule>, ModelValidity, u64)>,
    invalidated: Option<u64>,
    reset_epoch: u64,
}
impl ValidatedPlanReader {
    pub(super) fn reset_epoch(&self) -> u64 {
        self.reset_epoch
    }
    pub(super) fn read(
        &mut self,
        mailbox: &ValidatedPlanMailbox,
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

#[test]
fn realtime_handoff_probe_feedback_validity_matches_canonical_clock_health() {
    use std::sync::atomic::AtomicI64;
    struct ControlledClock(Instant, AtomicI64);
    impl Clock for ControlledClock {
        fn now_micros(&self) -> i64 {
            self.1.load(Ordering::Acquire)
        }
        fn micros_to_instant(&self, us: i64) -> Option<Instant> {
            if us >= 0 {
                self.0.checked_add(Duration::from_micros(us as u64))
            } else {
                self.0.checked_sub(Duration::from_micros(us.unsigned_abs()))
            }
        }
    }
    let clock = Arc::new(ControlledClock(Instant::now(), AtomicI64::new(0)));
    let mut sync = ClockSync::new(clock.clone());
    assert_eq!(ModelValidity::from_sync(&sync), ModelValidity::Invalid);
    for i in 0..10 {
        let client = i * 1_000_000;
        clock.1.store(client + 2000, Ordering::Release);
        sync.update(client, client + 101_000, client + 101_000, client + 2000);
    }
    let validity = ModelValidity::from_sync(&sync);
    assert_eq!(validity, ModelValidity::Sampled(9_002_000));
    for now in [
        9_001_999,
        9_002_000,
        14_002_000,
        14_002_001,
        i64::MIN,
        i64::MAX,
    ] {
        clock.1.store(now, Ordering::Release);
        assert_eq!(validity.allows(now), sync.is_synchronized());
    }
    let same = ClockSync::new_same_clock(clock);
    let validity = ModelValidity::from_sync(&same);
    assert_eq!(validity, ModelValidity::SameClock);
    assert!(validity.allows(i64::MIN));
    assert!(validity.allows(i64::MAX));
}

#[test]
fn realtime_handoff_probe_feedback_late_result_cannot_acknowledge_reader_invalidation() {
    for timeline in [0, 1 << 40] {
        let plans = ValidatedPlanMailbox::default();
        let mut reader = ValidatedPlanReader::default();
        let drop_plan = CorrectionPlanner::new().plan(10_000, 1_000, false);
        plans.publish(timeline, Some(drop_plan), ModelValidity::Sampled(0));
        assert_eq!(
            reader.read(&plans, timeline, 5_000_000, CorrectionSchedule::default()),
            Some(Some(drop_plan))
        );
        assert_eq!(
            reader.read(&plans, timeline, 5_000_001, drop_plan),
            Some(None)
        );
        // A delayed result from before the reader's reset has a fresh publication
        // revision and fresh model, but has not reset the old filtering history.
        plans.publish(timeline, Some(drop_plan), ModelValidity::Sampled(5_000_001));
        assert_eq!(
            reader.read(&plans, timeline, 5_000_001, CorrectionSchedule::default()),
            Some(None)
        );
    }
}

#[derive(Clone, Copy)]
pub(super) struct ValidatedObservation {
    pub(super) latency_floor: Option<Duration>,
    pub(super) input: Observation,
    pub(super) reset_epoch: u64,
}

pub(super) struct ValidatedFeedback {
    feedback: Feedback,
    reset_epoch: u64,
}
impl ValidatedFeedback {
    pub(super) fn new() -> Self {
        Self {
            feedback: Feedback::new(),
            reset_epoch: 0,
        }
    }
    pub(super) fn model_changed(
        &mut self,
        sync: &ClockSync,
        timeline: u64,
        plans: &ValidatedPlanMailbox,
    ) {
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
        input: ValidatedObservation,
        delay_us: u64,
        timeline: u64,
        plans: &ValidatedPlanMailbox,
    ) -> Option<ValidatedObservation> {
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
        let decision = self
            .feedback
            .compute_observation(sync, input.input, delay_us, timeline);
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
        plans: &ValidatedPlanMailbox,
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

#[test]
fn realtime_handoff_probe_feedback_first_settled_discards_converging_history_on_fallback() {
    use std::sync::atomic::AtomicI64;
    struct EndpointClock(Instant, AtomicI64);
    impl Clock for EndpointClock {
        fn now_micros(&self) -> i64 {
            self.1.load(Ordering::Acquire)
        }
        fn micros_to_instant(&self, micros: i64) -> Option<Instant> {
            if micros >= 0 {
                self.0.checked_add(Duration::from_micros(micros as u64))
            } else {
                self.0
                    .checked_sub(Duration::from_micros(micros.unsigned_abs()))
            }
        }
    }
    let origin = Instant::now();
    let clock = Arc::new(EndpointClock(origin, AtomicI64::new(0)));
    let mut sync = ClockSync::new(clock.clone());
    for i in 0..9 {
        let client = i * 1_000_000;
        clock.1.store(client + 2000, Ordering::Release);
        sync.update(client, client + 101_000, client + 101_000, client + 2000);
    }
    assert!(sync.is_synchronized());
    assert!(!sync.is_settled());
    let plans = ValidatedPlanMailbox::default();
    let mut worker = ValidatedFeedback::new();
    let mut observation = ValidatedObservation {
        latency_floor: None,
        input: Observation {
            settled_seen: false,
            timeline: 0,
            serial: 0,
            source_cursor_us: 9_100_000,
            presentation: Some(origin + Duration::from_micros(9_010_000)),
            actual_schedule: CorrectionSchedule::default(),
        },
        reset_epoch: 0,
    };
    for serial in 1..=101 {
        observation.input.serial = serial;
        assert!(worker.process(&sync, observation, 0, 0, &plans).is_none());
    }
    assert!(worker.feedback.filter.is_warm());
    clock.1.store(9_002_000, Ordering::Release);
    sync.update(9_000_000, 9_101_000, 9_101_000, 9_002_000);
    assert!(sync.is_settled());
    // Production resets before testing timestamp provenance: even fallback
    // must discard measurements collected while the clock was converging.
    observation.input.settled_seen = true;
    observation.input.serial = 102;
    observation.input.presentation = None;
    assert!(worker.process(&sync, observation, 0, 0, &plans).is_none());
    assert!(!worker.feedback.filter.is_warm());
    observation.input.presentation = Some(origin + Duration::from_micros(9_010_000));
    let mut reader = ValidatedPlanReader::default();
    for serial in 103..=202 {
        observation.input.serial = serial;
        assert!(worker.process(&sync, observation, 0, 0, &plans).is_none());
        assert_eq!(
            reader.read(&plans, 0, clock.now_micros(), CorrectionSchedule::default()),
            Some(Some(CorrectionSchedule::default()))
        );
    }
    observation.input.serial = 203;
    assert!(worker.process(&sync, observation, 0, 0, &plans).is_none());
    assert_eq!(
        reader
            .read(&plans, 0, clock.now_micros(), CorrectionSchedule::default())
            .unwrap()
            .unwrap()
            .drop_every_n_frames,
        500
    );
}

#[test]
fn realtime_handoff_probe_feedback_queued_before_settled_does_not_prewarm() {
    // Same model, PCM coordinates and measurements, different processing order.
    // Measurements collected before settling must not prewarm the new window.
    let origin = Instant::now();
    let mut sync = ClockSync::new(Arc::new(PinnedClock(origin)));
    for i in 0..9 {
        let client = (i - 10) * 1_000_000;
        sync.update(client, client + 101_000, client + 101_000, client + 2000);
    }
    assert!(sync.is_synchronized());
    assert!(!sync.is_settled());
    assert_eq!(sync.server_to_client_micros(100_000), Some(0));
    let mut immediate = Feedback::new();
    let mut delayed = Feedback::new();
    let (mut tx, mut rx) = RingBuffer::new(2);
    for (serial, source_cursor_us, offset) in [(1, 100_000, 10_000), (2, 110_000, 20_000)] {
        let input = Observation {
            settled_seen: false,
            timeline: 0,
            serial,
            source_cursor_us,
            presentation: Some(origin + Duration::from_micros(offset)),
            actual_schedule: CorrectionSchedule::default(),
        };
        assert_eq!(
            immediate.observe(&sync, input, 0),
            Ok(Some((10_000, CorrectionSchedule::default())))
        );
        assert!(tx.push(input).is_ok());
    }
    sync.update(-1_000_000, -899_000, -899_000, -998_000);
    assert!(sync.is_settled());
    assert_eq!(sync.server_to_client_micros(100_000), Some(0));
    for _ in 0..2 {
        assert_eq!(
            delayed.observe(&sync, rx.pop().unwrap(), 0),
            Ok(Some((10_000, CorrectionSchedule::default())))
        );
    }
    let mut input = Observation {
        settled_seen: true,
        timeline: 0,
        serial: 3,
        source_cursor_us: 120_000,
        presentation: None,
        actual_schedule: CorrectionSchedule::default(),
    };
    assert_eq!(immediate.observe(&sync, input, 0), Ok(None));
    assert_eq!(delayed.observe(&sync, input, 0), Ok(None));
    let mut delayed_plan = CorrectionSchedule::default();
    for serial in 4..=102 {
        input.serial = serial;
        input.presentation = Some(origin + Duration::from_micros(30_000));
        assert_eq!(
            immediate.observe(&sync, input, 0),
            Ok(Some((10_000, CorrectionSchedule::default())))
        );
        delayed_plan = delayed.observe(&sync, input, 0).unwrap().unwrap().1;
    }
    assert_eq!(
        delayed_plan,
        CorrectionSchedule::default(),
        "queued pre-settle measurements must not count toward post-settle warmth"
    );
    assert!(!immediate.filter.is_warm());
    assert!(!delayed.filter.is_warm());
}

#[test]
fn realtime_handoff_probe_feedback_reset_ack_rewarms_real_filter_before_correction() {
    let origin = Instant::now();
    let sync = ClockSync::new_same_clock(Arc::new(PinnedClock(origin)));
    let plans = ValidatedPlanMailbox::default();
    let mut worker = ValidatedFeedback::new();
    let mut reader = ValidatedPlanReader::default();
    let observation = |serial, reset_epoch| ValidatedObservation {
        latency_floor: None,
        input: Observation {
            settled_seen: true,
            timeline: 0,
            serial,
            source_cursor_us: 1_000_000,
            presentation: Some(origin + Duration::from_micros(1_010_000)),
            actual_schedule: CorrectionSchedule::default(),
        },
        reset_epoch,
    };
    for serial in 1..=101 {
        let _ = worker.process(&sync, observation(serial, 0), 0, 0, &plans);
    }
    let drop_plan = CorrectionPlanner::new().plan(10_000, 1_000, false);
    assert_eq!(
        reader.read(&plans, 0, 0, CorrectionSchedule::default()),
        Some(Some(drop_plan))
    );
    assert!(worker.feedback.filter.is_warm());
    // Inject a sampled validity into the carrier to exercise the reader reset;
    // this fixture does not claim a same-clock mapping naturally expires.
    plans.publish(0, Some(drop_plan), ModelValidity::Sampled(0));
    assert_eq!(reader.read(&plans, 0, 5_000_001, drop_plan), Some(None));
    assert_eq!(reader.reset_epoch(), 1);
    let _ = worker.process(&sync, observation(102, 0), 0, 0, &plans);
    assert_eq!(
        reader.read(&plans, 0, 5_000_001, CorrectionSchedule::default()),
        Some(None),
        "late computation cannot acknowledge reset"
    );
    let _ = worker.process(&sync, observation(103, reader.reset_epoch()), 0, 0, &plans);
    assert!(!worker.feedback.filter.is_warm());
    assert_eq!(
        reader.read(&plans, 0, 5_000_001, CorrectionSchedule::default()),
        Some(Some(CorrectionSchedule::default()))
    );
    for serial in 104..=202 {
        let _ = worker.process(&sync, observation(serial, 1), 0, 0, &plans);
    }
    assert!(!worker.feedback.filter.is_warm());
    assert_eq!(
        reader.read(&plans, 0, 5_000_001, CorrectionSchedule::default()),
        Some(Some(CorrectionSchedule::default()))
    );
    let _ = worker.process(&sync, observation(203, 1), 0, 0, &plans);
    assert!(worker.feedback.filter.is_warm());
    assert_eq!(
        reader.read(&plans, 0, 5_000_001, CorrectionSchedule::default()),
        Some(Some(drop_plan))
    );
}

#[test]
fn realtime_handoff_probe_feedback_real_model_reset_invalidates_without_observation() {
    use std::sync::atomic::AtomicI64;
    struct ControlledClock(Instant, AtomicI64);
    impl Clock for ControlledClock {
        fn now_micros(&self) -> i64 {
            self.1.load(Ordering::Acquire)
        }
        fn micros_to_instant(&self, us: i64) -> Option<Instant> {
            if us >= 0 {
                self.0.checked_add(Duration::from_micros(us as u64))
            } else {
                self.0.checked_sub(Duration::from_micros(us.unsigned_abs()))
            }
        }
    }
    let clock = Arc::new(ControlledClock(Instant::now(), AtomicI64::new(0)));
    let mut sync = ClockSync::new(clock.clone());
    for i in 0..10 {
        let client = i * 1_000_000;
        clock.1.store(client + 2000, Ordering::Release);
        sync.update(client, client + 101_000, client + 101_000, client + 2000);
    }
    assert!(sync.is_synchronized());
    let plans = ValidatedPlanMailbox::default();
    let mut reader = ValidatedPlanReader::default();
    let mut worker = ValidatedFeedback::new();
    let drop_plan = CorrectionPlanner::new().plan(10_000, 1_000, false);
    plans.publish(0, Some(drop_plan), ModelValidity::from_sync(&sync));
    let now = clock.now_micros();
    assert_eq!(
        reader.read(&plans, 0, now, CorrectionSchedule::default()),
        Some(Some(drop_plan))
    );
    sync.reset();
    assert!(!sync.is_synchronized());
    // A canonical model reset need not coincide with an audio observation.
    worker.model_changed(&sync, 0, &plans);
    assert_eq!(reader.read(&plans, 0, now, drop_plan), Some(None));
    assert_eq!(reader.reset_epoch(), 1);
    // Re-establish real sampled synchronization. Model availability alone must
    // not acknowledge the callback's pending filter reset.
    for i in 10..20 {
        let client = i * 1_000_000;
        clock.1.store(client + 2000, Ordering::Release);
        sync.update(client, client + 101_000, client + 101_000, client + 2000);
    }
    assert!(sync.is_synchronized());
    worker.model_changed(&sync, 0, &plans);
    assert_eq!(
        reader.read(&plans, 0, clock.now_micros(), CorrectionSchedule::default()),
        Some(None)
    );
    let _ = worker.process(
        &sync,
        ValidatedObservation {
            latency_floor: None,
            input: Observation {
                settled_seen: true,
                timeline: 0,
                serial: 1,
                source_cursor_us: 100_000,
                presentation: None,
                actual_schedule: CorrectionSchedule::default(),
            },
            reset_epoch: reader.reset_epoch(),
        },
        0,
        0,
        &plans,
    );
    assert_eq!(
        reader.read(&plans, 0, clock.now_micros(), CorrectionSchedule::default()),
        Some(Some(CorrectionSchedule::default()))
    );
}

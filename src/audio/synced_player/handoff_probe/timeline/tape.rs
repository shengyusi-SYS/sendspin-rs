//! Alternative B representation: prepared PCM source frames plus a non-RT
//! schedule. Actual callback provenance gates only bounded frame selection.
//! No source queue, clock model, filter, allocation or final drop on the reader.

use super::*;

struct Tape {
    span: Span,
    source_base: u64,
    cursor_before: Option<i64>,
    skipped_tail: Option<Frame>,
}

fn prepare_tape(preparation: &mut Preparation, epoch: u64, source_base: u64, count: usize) -> Tape {
    let cursor_before = preparation
        .cursor
        .initialized
        .then_some(preparation.cursor.cursor_us);
    let retired_before = preparation.retired_through;
    let count = count.min(preparation.window_frames);
    let mut span = preparation.prepare(epoch, &vec![1; count]);
    // Missing output is a reader fact, not a source item to publish in advance.
    let valid = span
        .frames
        .iter()
        .take_while(|frame| frame.consumed == 1)
        .count();
    let retired_after_pcm = valid.checked_sub(1).map_or(retired_before, |index| {
        span.frames[index].position.retired_through
    });
    let skipped_tail = span
        .frames
        .get(valid)
        .copied()
        .filter(|frame| frame.position.retired_through != retired_after_pcm);
    let mut frames = span.frames.into_vec();
    frames.truncate(valid);
    span.frames = frames.into_boxed_slice();
    let mut pcm = span.pcm.into_vec();
    pcm.truncate(valid * span.channels);
    span.pcm = pcm.into_boxed_slice();
    Tape {
        span,
        source_base,
        cursor_before,
        skipped_tail,
    }
}

// One non-RT owner retains at most one source view and one unpublished tape.
// Callers provide replacement views lazily, after the old view has been dropped.
struct TapePreparation {
    source: Option<Preparation>,
    revision: Option<u64>,
    pending: Option<Tape>,
    pipe: PrepareSide<Tape>,
    epoch: u64,
    source_base: u64,
}

impl TapePreparation {
    fn reset(&mut self, epoch: u64, source_base: u64, snapshot: impl FnOnce() -> Preparation) {
        drop(self.pending.take());
        drop(self.source.take());
        self.epoch = epoch;
        self.source_base = source_base;
        self.source = Some(snapshot());
    }

    fn pump_from_ledger(
        &mut self,
        ledger: &mut super::super::admission::SourceLedger,
        gate: &ClaimGate,
        count: usize,
    ) -> Result<bool, ()> {
        let (view, revision, _) = ledger.preparation_version(gate)?;
        if self.source.is_none() || self.epoch != view.epoch() || self.revision != Some(revision) {
            // Release before cloning, including when the new observation must
            // retry. Published windows retain their own transport ownership.
            drop(self.pending.take());
            drop(self.source.take());
            let (view, revision, base, source) = ledger.preparation_snapshot(gate)?;
            self.source = Some(source);
            self.epoch = view.epoch();
            self.revision = Some(revision);
            self.source_base = base;
        }
        Ok(self.pump(count))
    }

    fn pump(&mut self, count: usize) -> bool {
        if self.pending.is_none() {
            let source = self.source.as_mut().unwrap();
            let tape = prepare_tape(source, self.epoch, self.source_base, count);
            if tape.span.frames.is_empty() && tape.skipped_tail.is_none() {
                return false;
            }
            self.source_base = self
                .source_base
                .checked_add(tape.span.frames.len() as u64)
                .unwrap();
            self.pending = Some(tape);
        }
        match self.pipe.publish(self.pending.take().unwrap()) {
            Ok(()) => true,
            Err(tape) => {
                self.pending = Some(tape);
                false
            }
        }
    }
}

struct TapeDevice {
    owner: Option<(crate::audio::RendererOwner, crate::audio::PlayerScope)>,
    pipe: DeviceSide<Tape>,
    checkpoint: Arc<PublishedCheckpoint>,
    last: Vec<i32>,
    position: Option<SourcePosition>,
    minimum_epoch: u64,
    sync_timeline: u64,
    feedback_serial: u64,
    feedback_origin: Option<i64>,
    latency_floor: Option<std::time::Duration>,
    schedule: CorrectionSchedule,
    insert_counter: u32,
    drop_counter: u32,
}

impl TapeDevice {
    fn new(pipe: DeviceSide<Tape>, checkpoint: Arc<PublishedCheckpoint>, channels: usize) -> Self {
        Self {
            owner: None,
            pipe,
            checkpoint,
            last: vec![0; channels],
            position: None,
            minimum_epoch: 0,
            sync_timeline: 0,
            feedback_serial: 0,
            feedback_origin: None,
            latency_floor: None,
            schedule: CorrectionSchedule::default(),
            insert_counter: 0,
            drop_counter: 0,
        }
    }

    fn retire(&mut self, visible: &mut usize) -> bool {
        self.pipe.pending_return = Some(self.pipe.ready.pop().unwrap());
        self.pipe.offset = 0;
        *visible -= 1;
        self.pipe.return_slot()
    }

    fn take_frame(
        &mut self,
        destination: Option<&mut [i32]>,
        visible: &mut usize,
        epoch: u64,
        current: &mut u64,
        consumed: &mut u64,
    ) -> bool {
        if self.pipe.pending_return.is_some() {
            return false;
        }
        while *visible > 0 {
            let Ok(slot) = self.pipe.ready.peek() else {
                return false;
            };
            let tape = &slot.prepared;
            let skip = consumed.saturating_sub(tape.source_base);
            self.pipe.offset = self.pipe.offset.max(
                usize::try_from(skip)
                    .unwrap_or(usize::MAX)
                    .min(tape.span.frames.len()),
            );
            let Some(frame) = tape
                .span
                .frames
                .get(self.pipe.offset)
                .or(tape.skipped_tail.as_ref())
            else {
                if !self.retire(visible) {
                    return false;
                }
                continue;
            };
            let mismatch = tape.span.epoch < self.minimum_epoch
                || tape.source_base + self.pipe.offset as u64 != *consumed
                || frame.before_current != *current
                || (*current != 0
                    && self
                        .position
                        .is_some_and(|position| position.index != frame.before_index))
                || (frame.requires_epoch && tape.span.epoch != epoch);
            if mismatch {
                if !self.retire(visible) {
                    return false;
                }
                continue;
            }
            // Capture from the first descriptor actually accepted by the
            // reader, before source consumption (including drop/skip steps).
            // Repeated prefixes use their predecessor's actual source cursor.
            if self.feedback_origin.is_none() {
                self.feedback_origin = self
                    .pipe
                    .offset
                    .checked_sub(1)
                    .and_then(|index| tape.span.frames.get(index))
                    .map(|previous| previous.position.cursor_us)
                    .or(tape.cursor_before);
            }
            if frame.consumed == 0 {
                self.position = Some(frame.position);
                *current = frame.after_current;
                if !self.retire(visible) {
                    return false;
                }
                continue;
            }
            if let Some(destination) = destination {
                let offset = self.pipe.offset * tape.span.channels;
                destination.copy_from_slice(&tape.span.pcm[offset..offset + tape.span.channels]);
            }
            self.position = Some(frame.position);
            *current = frame.after_current;
            *consumed = consumed.checked_add(1).unwrap();
            self.pipe.offset += 1;
            if self.pipe.offset == tape.span.frames.len() && tape.skipped_tail.is_none() {
                self.retire(visible);
            }
            return true;
        }
        false
    }

    fn reject_closed(&self, output: &mut [i32]) -> bool {
        if self
            .owner
            .as_ref()
            .is_some_and(|(owner, scope)| owner.needs_terminal_check(*scope))
        {
            output.fill(0);
            self.pipe.gate.end_callback();
            return true;
        }
        false
    }

    fn published_schedule(&mut self, plans: &super::feedback::PlanMailbox) -> CorrectionSchedule {
        let view = self.pipe.gate.view();
        let timeline = if view.current_revoked() || view.timeline_reanchored() {
            view.epoch()
        } else {
            self.sync_timeline
        };
        match plans.read(timeline) {
            Some(Some(schedule)) => schedule,
            Some(None) => {
                self.schedule = CorrectionSchedule::default();
                self.insert_counter = 0;
                self.drop_counter = 0;
                CorrectionSchedule::default()
            }
            None if timeline == self.sync_timeline => self.schedule,
            None => CorrectionSchedule::default(),
        }
    }

    fn render_scheduled(
        &mut self,
        output: &mut [i32],
        presentation_zone_us: Option<i64>,
        measured: bool,
        plans: &super::feedback::PlanMailbox,
        start: &impl super::super::start::StartDecision,
    ) -> Option<usize> {
        self.pipe.gate.begin_callback();
        self.render_scheduled_active(output, presentation_zone_us, measured, plans, start)
    }

    fn render_scheduled_active(
        &mut self,
        output: &mut [i32],
        presentation_zone_us: Option<i64>,
        measured: bool,
        plans: &super::feedback::PlanMailbox,
        start: &impl super::super::start::StartDecision,
    ) -> Option<usize> {
        use crate::audio::ScheduledStartOutcome;
        assert!(self.pipe.gate.view().callback_active());
        if self.reject_closed(output) {
            return None;
        }
        match start.decide(presentation_zone_us) {
            Ok(
                ScheduledStartOutcome::Unscheduled
                | ScheduledStartOutcome::Started { .. }
                | ScheduledStartOutcome::BoundaryWon { .. },
            ) => {}
            _ => {
                output.fill(0);
                self.pipe.gate.end_callback();
                return None;
            }
        }
        let wanted = self.published_schedule(plans);
        let missing = self.render_active(output, measured, wanted);
        self.pipe.gate.end_callback();
        Some(missing)
    }

    fn render_validated_plan(
        &mut self,
        output: &mut [i32],
        measured: bool,
        endpoint_now: i64,
        plans: &super::feedback::ValidatedPlanMailbox,
        reader: &mut super::feedback::ValidatedPlanReader,
    ) -> usize {
        self.pipe.gate.begin_callback();
        if self.reject_closed(output) {
            return output.len() / self.last.len();
        }
        let missing = self.render_validated_active(output, measured, endpoint_now, plans, reader);
        self.pipe.gate.end_callback();
        missing
    }

    fn render_validated_active(
        &mut self,
        output: &mut [i32],
        measured: bool,
        endpoint_now: i64,
        plans: &super::feedback::ValidatedPlanMailbox,
        reader: &mut super::feedback::ValidatedPlanReader,
    ) -> usize {
        let view = self.pipe.gate.view();
        let timeline = if view.current_revoked() || view.timeline_reanchored() {
            view.epoch()
        } else {
            self.sync_timeline
        };
        let wanted = match reader.read(plans, timeline, endpoint_now, self.schedule) {
            Some(Some(plan)) => plan,
            Some(None) => {
                self.schedule = CorrectionSchedule::default();
                self.insert_counter = 0;
                self.drop_counter = 0;
                self.schedule
            }
            None if timeline == self.sync_timeline => self.schedule,
            None => CorrectionSchedule::default(),
        };
        self.render_active(output, measured, wanted)
    }

    fn render_validated_feedback(
        &mut self,
        output: &mut [i32],
        captured_at: std::time::Instant,
        presentation: Option<std::time::Instant>,
        endpoint_now: i64,
        plans: &super::feedback::ValidatedPlanMailbox,
        reader: &mut super::feedback::ValidatedPlanReader,
        feedback: &mut rtrb::Producer<super::feedback::ValidatedObservation>,
    ) -> (usize, bool) {
        self.pipe.gate.begin_callback();
        if self.reject_closed(output) {
            return (output.len() / self.last.len(), false);
        }
        let presentation = presentation.filter(|instant| *instant >= captured_at);
        self.feedback_serial = self.feedback_serial.checked_add(1).unwrap();
        let missing = self.render_validated_active(
            output,
            presentation.is_some(),
            endpoint_now,
            plans,
            reader,
        );
        if let Some(instant) = presentation {
            let delta = instant.duration_since(captured_at);
            self.latency_floor = Some(self.latency_floor.map_or(delta, |floor| floor.min(delta)));
        }
        let observation = self.feedback_origin.map(|source_cursor_us| {
            super::feedback::ValidatedObservation {
                latency_floor: self.latency_floor,
                input: super::feedback::Observation {
                    settled_seen: reader.settled_seen,
                    timeline: self.sync_timeline,
                    serial: self.feedback_serial,
                    source_cursor_us,
                    presentation,
                    actual_schedule: self.schedule,
                },
                // Sticky until acknowledged: a full channel cannot consume the
                // reset request; every later observation carries it again.
                reset_epoch: reader.reset_epoch(),
            }
        });
        let published = observation.is_some_and(|value| feedback.push(value).is_ok());
        self.pipe.gate.end_callback();
        (missing, published)
    }

    fn render_feedback(
        &mut self,
        output: &mut [i32],
        presentation: Option<std::time::Instant>,
        plans: &super::feedback::PlanMailbox,
        feedback: &mut rtrb::Producer<super::feedback::Observation>,
    ) -> (usize, bool) {
        self.pipe.gate.begin_callback();
        if self.reject_closed(output) {
            return (output.len() / self.last.len(), false);
        }
        self.feedback_serial = self.feedback_serial.checked_add(1).unwrap();
        let wanted = self.published_schedule(plans);
        let missing = self.render_active(output, presentation.is_some(), wanted);
        let observation =
            self.feedback_origin
                .map(|source_cursor_us| super::feedback::Observation {
                    settled_seen: true,
                    timeline: self.sync_timeline,
                    serial: self.feedback_serial,
                    source_cursor_us,
                    presentation,
                    // Report the plan actually applied, including fallback
                    // retention and explicit invalidation, not its predecessor.
                    actual_schedule: self.schedule,
                });
        // Observation is Copy: a full bounded channel discards no payload lease
        // and cannot block, allocate, or turn readable PCM into silence.
        let published = observation.is_some_and(|value| feedback.push(value).is_ok());
        self.pipe.gate.end_callback();
        (missing, published)
    }

    fn render_published(
        &mut self,
        output: &mut [i32],
        measured: bool,
        plans: &super::feedback::PlanMailbox,
    ) -> usize {
        self.pipe.gate.begin_callback();
        if self.reject_closed(output) {
            return output.len() / self.last.len();
        }
        let wanted = self.published_schedule(plans);
        let missing = self.render_active(output, measured, wanted);
        self.pipe.gate.end_callback();
        missing
    }

    fn render(&mut self, output: &mut [i32], measured: bool, wanted: CorrectionSchedule) -> usize {
        self.pipe.gate.begin_callback();
        if self.reject_closed(output) {
            return output.len() / self.last.len();
        }
        let missing = self.render_active(output, measured, wanted);
        self.pipe.gate.end_callback();
        missing
    }

    fn render_active(
        &mut self,
        output: &mut [i32],
        measured: bool,
        wanted: CorrectionSchedule,
    ) -> usize {
        assert!(!wanted.reanchor, "reanchor is a source-control operation");
        let channels = self.last.len();
        assert_eq!(output.len() % channels, 0);
        let mut view = self.pipe.gate.view();
        let cleared = view.current_revoked();
        let reanchored = view.timeline_reanchored();
        self.feedback_origin = if cleared || reanchored {
            None
        } else {
            self.position.map(|position| position.cursor_us)
        };
        if cleared || reanchored {
            self.sync_timeline = view.epoch();
        }
        if reanchored {
            self.minimum_epoch = view.epoch();
            self.position = None;
            self.schedule = CorrectionSchedule::default();
            self.insert_counter = 0;
            self.drop_counter = 0;
            self.pipe
                .gate
                .control
                .fetch_and(!REANCHORED_TIMELINE, Ordering::AcqRel);
            view = self.pipe.gate.view();
        }
        if cleared {
            self.pipe.gate.finish_current();
            view = self.pipe.gate.view();
        }
        if cleared {
            self.latency_floor = None;
            self.position = None;
            self.last.fill(0);
            self.schedule = CorrectionSchedule::default();
            self.insert_counter = 0;
            self.drop_counter = 0;
        }
        // A fallback does not apply a new plan or advance existing cadence.
        if measured && wanted != self.schedule {
            self.schedule = wanted;
            self.insert_counter = wanted.insert_every_n_frames;
            self.drop_counter = wanted.drop_every_n_frames;
        }
        let mut visible = if self.pipe.return_slot() {
            self.pipe.ready.slots()
        } else {
            0
        };
        let mut current = view.current();
        let mut consumed = self.pipe.progress.load(Ordering::Acquire);
        let mut missing = 0;
        for destination in output.chunks_exact_mut(channels) {
            let mut step = 1;
            if measured {
                if self.schedule.drop_every_n_frames > 0 {
                    self.drop_counter = self.drop_counter.saturating_sub(1);
                    if self.drop_counter == 0 {
                        self.drop_counter = self.schedule.drop_every_n_frames;
                        step = 2;
                    }
                }
                if step != 2 && self.schedule.insert_every_n_frames > 0 {
                    self.insert_counter = self.insert_counter.saturating_sub(1);
                    if self.insert_counter == 0 {
                        self.insert_counter = self.schedule.insert_every_n_frames;
                        step = 0;
                    }
                }
            }
            if step == 0 {
                destination.copy_from_slice(&self.last);
                continue;
            }
            if step == 2 {
                self.take_frame(
                    None,
                    &mut visible,
                    view.epoch(),
                    &mut current,
                    &mut consumed,
                );
            }
            if self.take_frame(
                Some(destination),
                &mut visible,
                view.epoch(),
                &mut current,
                &mut consumed,
            ) {
                self.last.copy_from_slice(destination);
            } else if step == 2 {
                destination.copy_from_slice(&self.last);
            } else {
                destination.fill(0);
                missing += 1;
            }
        }
        self.pipe.progress.store(consumed, Ordering::Release);
        if let Some(position) = self.position {
            self.checkpoint.publish(
                Cadence {
                    schedule: self.schedule,
                    insert_counter: self.insert_counter,
                    drop_counter: self.drop_counter,
                },
                position,
                output.len() / channels,
                Some(&self.last),
            );
        }
        self.pipe.gate.publish_current(current);
        missing
    }
}

#[test]
fn realtime_handoff_probe_tape_actual_fallback_keeps_pcm_and_resumes_cadence() {
    for error in [10_000, -10_000] {
        let samples: Vec<i32> = (1..=503).collect();
        let mut preparation = Preparation::with_window_frames(vec![(1, source_pcm(&samples))], 512);
        let checkpoint = Arc::new(PublishedCheckpoint::new(1));
        let (mut producer, pipe) = pipe_for::<Tape>(1);
        let mut device = TapeDevice::new(pipe, checkpoint, 1);
        assert!(producer
            .publish(prepare_tape(&mut preparation, 0, 0, 503))
            .is_ok());
        let schedule = CorrectionPlanner::new().plan(error, 1_000, false);
        let mut prefix = [0; 499];
        assert_eq!(device.render(&mut prefix, true, schedule), 0);
        assert_eq!(prefix, std::array::from_fn(|i| i as i32 + 1));
        let mut fallback = [0];
        assert_eq!(device.render(&mut fallback, false, schedule), 0);
        assert_eq!(fallback, [500]);
        assert_eq!(device.pipe.progress.load(Ordering::Acquire), 500);
        let mut resumed = [0];
        assert_eq!(device.render(&mut resumed, true, schedule), 0);
        assert_eq!(resumed, if error > 0 { [502] } else { [500] });
        assert_eq!(
            device.pipe.progress.load(Ordering::Acquire),
            if error > 0 { 502 } else { 500 }
        );
        let actual = device
            .checkpoint
            .read(&device.pipe.gate, &device.pipe.progress)
            .unwrap();
        assert_eq!(
            actual
                .cadence
                .insert_counter
                .max(actual.cadence.drop_counter),
            500
        );
        assert_eq!(actual.output_frames, 501);
    }
}

#[test]
fn realtime_handoff_probe_tape_overlap_and_clear_use_actual_source_position() {
    use super::super::admission::SourceLedger;
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome};
    for first_source in [1, (1 << 14) - 1, 1 << 40] {
        let publication = Arc::new(PublishedCheckpoint::new(1));
        let (mut producer, pipe) = pipe_for::<Tape>(2);
        let mut ledger =
            SourceLedger::with_checkpoint(Arc::clone(&pipe.progress), Arc::clone(&publication));
        let mut device = TapeDevice::new(pipe, publication, 1);
        ledger.next_source_id = first_source;
        device
            .pipe
            .gate
            .control
            .store((1u64 << 40) << EPOCH_SHIFT, Ordering::Release);
        let schedule = CorrectionSchedule::default();
        for (timestamp, samples) in [(0, [1, 2, 3, 4]), (4_000, [10, 11, 12, 13])] {
            let mut source = source_pcm(&samples);
            source.timestamp = timestamp;
            assert!(matches!(
                ledger.try_enqueue(source, &device.pipe.gate, || {}),
                Ok(EnqueueOutcome::Accepted { .. })
            ));
        }
        let (view, mut preparation) = ledger.preparation(&device.pipe.gate).unwrap();
        assert!(producer
            .publish(prepare_tape(&mut preparation, view.epoch(), 0, 8))
            .is_ok());
        assert_eq!(device.render(&mut [0], true, schedule), 0);
        let mut replacement = source_pcm(&[50, 51, 52, 53]);
        replacement.timestamp = 4_000;
        assert_eq!(
            ledger
                .try_enqueue(replacement, &device.pipe.gate, || {})
                .ok(),
            Some(EnqueueOutcome::Accepted {
                queued_frames: 7,
                queued_buffers: 2
            })
        );
        let (view, mut preparation) = ledger.preparation(&device.pipe.gate).unwrap();
        assert!(producer
            .publish(prepare_tape(&mut preparation, view.epoch(), 1, 7))
            .is_ok());
        let mut rest = [0; 7];
        assert_eq!(device.render(&mut rest, false, schedule), 0);
        assert_eq!(rest, [2, 3, 4, 50, 51, 52, 53]);
        ledger.try_reconcile(&device.pipe.gate).unwrap();
        assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(8));
        assert_eq!(
            ledger
                .owner
                .capacity(ledger.scope)
                .unwrap()
                .current_frames(),
            0
        );
        assert_eq!(producer.reclaim(), 2);
        assert_eq!(
            ledger.try_clear(&device.pipe.gate),
            Ok(RendererOperationOutcome::Applied)
        );
        assert!(matches!(
            ledger.try_enqueue(source_pcm(&[9, 8]), &device.pipe.gate, || {}),
            Ok(EnqueueOutcome::Accepted { .. })
        ));
        let (view, mut preparation) = ledger.preparation(&device.pipe.gate).unwrap();
        assert!(producer
            .publish(prepare_tape(&mut preparation, view.epoch(), 8, 2))
            .is_ok());
        let mut fresh = [0; 2];
        assert_eq!(device.render(&mut fresh, false, schedule), 0);
        assert_eq!(fresh, [9, 8]);
        ledger.try_reconcile(&device.pipe.gate).unwrap();
        assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(10));
        assert_eq!(producer.reclaim(), 1);
    }
}

#[test]
fn realtime_handoff_probe_tape_terminal_joins_active_reader_before_ack_and_drop() {
    use crate::audio::{
        RendererOperationOutcome, RendererOwner, RendererQueueLimits, TerminalOutcome,
        TerminalState,
    };
    // The production owner owns this resource and runs its destructor outside
    // its lock. No simulated terminal state or fabricated acknowledgment.
    struct JoinedResource(Option<Box<dyn FnOnce() + Send>>);
    impl Drop for JoinedResource {
        fn drop(&mut self) {
            self.0.take().unwrap()();
        }
    }
    let owner = RendererOwner::new(RendererQueueLimits::new(32, 8, 16).unwrap());
    let scope = owner.mint_scope().unwrap();
    let (mut producer, pipe) = pipe_for::<Tape>(1);
    let actual = Arc::clone(&pipe.progress);
    let mut device = TapeDevice::new(pipe, Arc::new(PublishedCheckpoint::new(1)), 1);
    let (dropped_tx, dropped_rx) = mpsc::channel();
    let (published_tx, published_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel();
    let preparation = thread::spawn(move || {
        let mut cursor = Preparation::new(vec![(1, source_pcm(&[10, 20, 30, 40]))]);
        let mut tape = prepare_tape(&mut cursor, 0, 0, 4);
        tape.span.lifetime = Some(DropWitness(dropped_tx));
        assert!(producer.publish(tape).is_ok());
        published_tx.send(()).unwrap();
        stop_rx.recv().unwrap();
        producer
    });
    published_rx.recv().unwrap();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let reader = thread::spawn(move || {
        device.pipe.gate.begin_callback();
        entered_tx.send(()).unwrap();
        resume_rx.recv().unwrap();
        let mut output = [0; 3];
        assert_eq!(
            device.render_active(&mut output, false, CorrectionSchedule::default()),
            0
        );
        device.pipe.gate.end_callback();
        (device, output)
    });
    entered_rx.recv().unwrap();
    assert_eq!(actual.load(Ordering::Acquire), 0);
    assert!(dropped_rx.try_recv().is_err());
    let observed_owner = owner.clone();
    let (observed_tx, observed_rx) = mpsc::channel();
    assert_eq!(
        owner.attach_test_terminal_resource(
            scope,
            Box::new(JoinedResource(Some(Box::new(move || {
                assert!(matches!(
                    observed_owner.terminal_state(scope).unwrap(),
                    TerminalState::Finalizing { .. }
                ));
                stop_tx.send(()).unwrap();
                resume_tx.send(()).unwrap();
                let producer = preparation.join().unwrap();
                let (device, output) = reader.join().unwrap();
                drop(device);
                drop(producer);
                observed_tx.send(output).unwrap();
            }))))
        ),
        RendererOperationOutcome::Applied
    );
    let TerminalOutcome::Won(finalization) = owner.teardown(scope) else {
        panic!("first finalizer wins")
    };
    assert_eq!(observed_rx.recv().unwrap(), [10, 20, 30]);
    assert_eq!(actual.load(Ordering::Acquire), 3);
    assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
    assert!(finalization.ack.callback_stopped());
    assert!(finalization.ack.stream_released());
    assert_eq!(
        owner.teardown(scope),
        TerminalOutcome::AlreadyFinalized(finalization)
    );
    assert_eq!(actual.load(Ordering::Acquire), 3);
}

#[test]
fn realtime_handoff_probe_tape_terminal_discards_queued_feedback_without_next_callback() {
    use super::feedback::{ValidatedFeedback, ValidatedPlanMailbox, ValidatedPlanReader};
    use crate::audio::{
        RendererOperationOutcome, RendererOwner, RendererQueueLimits, TerminalOutcome,
        TerminalState,
    };
    use crate::sync::{Clock, ClockSync};
    use std::time::{Duration, Instant};
    struct FixedClock(Instant);
    impl Clock for FixedClock {
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
    struct JoinedResource(Option<Box<dyn FnOnce() + Send>>);
    impl Drop for JoinedResource {
        fn drop(&mut self) {
            self.0.take().unwrap()();
        }
    }
    let origin = Instant::now();
    let owner = RendererOwner::new(RendererQueueLimits::new(32, 8, 16).unwrap());
    let scope = owner.mint_scope().unwrap();
    let plans = Arc::new(ValidatedPlanMailbox::default());
    let worker_plans = Arc::clone(&plans);
    let (mut producer, pipe) = pipe_for::<Tape>(1);
    let actual = Arc::clone(&pipe.progress);
    let mut device = TapeDevice::new(pipe, Arc::new(PublishedCheckpoint::new(1)), 1);
    device.owner = Some((owner.clone(), scope));
    let (dropped_tx, dropped_rx) = mpsc::channel();
    let mut source = Preparation::new(vec![(1, source_pcm(&[1, 2, 3]))]);
    let mut tape = prepare_tape(&mut source, 0, 0, 3);
    tape.span.lifetime = Some(DropWitness(dropped_tx));
    assert!(producer.publish(tape).is_ok());
    let (mut tx, mut rx) = rtrb::RingBuffer::new(2);
    let (process_tx, process_rx) = mpsc::channel();
    let (processed_tx, processed_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let sync = ClockSync::new_same_clock(Arc::new(FixedClock(origin)));
        let mut feedback = ValidatedFeedback::new();
        process_rx.recv().unwrap();
        assert!(feedback
            .process(&sync, rx.pop().unwrap(), 0, 0, &worker_plans)
            .is_none());
        processed_tx.send(()).unwrap();
        // Preparation and feedback share the same resource lifetime. Stopping
        // does not wait for another PCM or observation to arrive.
        stop_rx.recv().unwrap();
        let discarded = rx.pop().unwrap();
        assert!(rx.pop().is_err());
        drop(feedback);
        drop(rx);
        (producer, discarded.input.serial)
    });
    let mut reader = ValidatedPlanReader::default();
    let mut output = [0];
    for frame in 0..2 {
        let timestamp = origin + Duration::from_micros(frame * 1000);
        assert_eq!(
            device.render_validated_feedback(
                &mut output,
                timestamp,
                Some(timestamp),
                frame as i64 * 1000,
                &plans,
                &mut reader,
                &mut tx
            ),
            (0, true)
        );
        assert_eq!(output, [frame as i32 + 1]);
        if frame == 0 {
            process_tx.send(()).unwrap();
            processed_rx.recv().unwrap();
        }
    }
    assert_eq!(actual.load(Ordering::Acquire), 2);
    let publication_before = plans.read(0).unwrap();
    let observed_owner = owner.clone();
    let (joined_tx, joined_rx) = mpsc::channel();
    assert_eq!(
        owner.attach_test_terminal_resource(
            scope,
            Box::new(JoinedResource(Some(Box::new(move || {
                assert!(matches!(
                    observed_owner.terminal_state(scope).unwrap(),
                    TerminalState::Finalizing { .. }
                ));
                stop_tx.send(()).unwrap();
                let (producer, discarded_serial) = worker.join().unwrap();
                // No callback is scheduled here or needed to stop the worker.
                drop(tx);
                drop(device);
                drop(producer);
                joined_tx.send(discarded_serial).unwrap();
            }))))
        ),
        RendererOperationOutcome::Applied
    );
    let TerminalOutcome::Won(finalization) = owner.teardown(scope) else {
        panic!("first finalizer wins")
    };
    assert_eq!(joined_rx.recv().unwrap(), 2);
    assert_eq!(
        plans.read(0).unwrap(),
        publication_before,
        "queued feedback was not published during shutdown"
    );
    assert_eq!(actual.load(Ordering::Acquire), 2);
    assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
    assert!(finalization.ack.callback_stopped());
    assert!(finalization.ack.stream_released());
    let replacement = owner.mint_scope().unwrap();
    assert_ne!(replacement, scope);
    let new_plans = ValidatedPlanMailbox::default();
    assert!(
        new_plans.read(0).is_none(),
        "successor owns a fresh publication channel"
    );
}

#[test]
fn realtime_handoff_probe_tape_settled_observation_survives_full_channel_and_model_loss() {
    use super::feedback::{ModelValidity, ValidatedPlanMailbox, ValidatedPlanReader};
    use std::time::{Duration, Instant};
    let origin = Instant::now();
    let (mut producer, pipe) = pipe_for::<Tape>(1);
    let mut device = TapeDevice::new(pipe, Arc::new(PublishedCheckpoint::new(1)), 1);
    let mut source = Preparation::new(vec![(1, source_pcm(&[1, 2, 3]))]);
    assert!(producer.publish(prepare_tape(&mut source, 0, 0, 3)).is_ok());
    let plans = ValidatedPlanMailbox::default();
    let mut reader = ValidatedPlanReader::default();
    let (mut tx, mut rx) = rtrb::RingBuffer::new(1);
    let mut output = [0];
    plans.publish(
        0,
        Some(CorrectionSchedule::default()),
        ModelValidity::SampledUnsettled(0),
    );
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            origin,
            Some(origin),
            0,
            &plans,
            &mut reader,
            &mut tx
        ),
        (0, true)
    );
    // The model can be observed even while the associated plan's ack is stale.
    plans.publish_acknowledged(
        0,
        Some(CorrectionSchedule::default()),
        ModelValidity::Sampled(0),
        99,
    );
    let now = origin + Duration::from_millis(1);
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            now,
            Some(now),
            1000,
            &plans,
            &mut reader,
            &mut tx
        ),
        (0, false)
    );
    assert_eq!(output, [2]);
    assert_eq!(
        reader.reset_epoch(),
        0,
        "model metadata must not acknowledge a reset"
    );
    plans.publish(0, None, ModelValidity::Invalid);
    let before_settled = rx.pop().unwrap();
    assert!(!before_settled.input.settled_seen);
    let now = origin + Duration::from_millis(2);
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            now,
            None,
            2000,
            &plans,
            &mut reader,
            &mut tx
        ),
        (0, true)
    );
    let recovered = rx.pop().unwrap();
    assert!(
        recovered.input.settled_seen,
        "lost first true observation is repeated on fallback"
    );
    assert_eq!(recovered.input.presentation, None);
    assert_eq!(recovered.input.serial, 3);
    assert_eq!(output, [3]);
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 3);
}

#[test]
fn realtime_handoff_probe_tape_plan_change_after_snapshot_reuses_pcm_without_replaying_source() {
    for channels in [1usize, 2] {
        let samples: Vec<i32> = (1..=1_010)
            .flat_map(|frame| {
                (0..channels).map(move |channel| if channel == 0 { frame } else { -frame })
            })
            .collect();
        let mut source = source_pcm(&samples);
        source.format.channels = (channels as u8).into();
        let mut canonical = Preparation::with_window_frames(vec![(1, source)], 512);
        let checkpoint = Arc::new(PublishedCheckpoint::new(channels));
        let (mut producer, pipe) = pipe_for::<Tape>(2);
        let mut device = TapeDevice::new(pipe, checkpoint, channels);
        let mut initial = canonical.fork_actual();
        assert!(producer
            .publish(prepare_tape(&mut initial, 0, 0, 510))
            .is_ok());
        let drop_plan = CorrectionPlanner::new().plan(10_000, 1_000, false);
        let insert_plan = CorrectionPlanner::new().plan(-10_000, 1_000, true);
        assert!(insert_plan.insert_every_n_frames > 0);
        let mut prefix = vec![0; 499 * channels];
        assert_eq!(device.render(&mut prefix, true, drop_plan), 0);
        assert_eq!(prefix, samples[..499 * channels]);
        let snapshot = device
            .checkpoint
            .read(&device.pipe.gate, &device.pipe.progress)
            .unwrap();
        canonical.apply_actual(&snapshot);
        let snapshot_epoch = snapshot.view.epoch();
        let snapshot_base = snapshot.source_frames;
        drop(snapshot);
        let mut rebuilding = canonical.fork_actual();
        // Preparation took its snapshot before the next callback. Both the
        // reader position and the correction plan change before publication.
        let mut fallback = vec![0; 2 * channels];
        assert_eq!(device.render(&mut fallback, false, insert_plan), 0);
        assert_eq!(fallback, samples[499 * channels..501 * channels]);
        assert_eq!(device.schedule, drop_plan);
        assert!(producer
            .publish(prepare_tape(
                &mut rebuilding,
                snapshot_epoch,
                snapshot_base,
                510
            ))
            .is_ok());
        let mut changed = vec![0; 500 * channels];
        assert_eq!(device.render(&mut changed, true, insert_plan), 0);
        let mut expected = samples[501 * channels..1_000 * channels].to_vec();
        expected.extend_from_slice(&samples[999 * channels..1_000 * channels]);
        assert_eq!(changed, expected);
        assert_eq!(device.pipe.progress.load(Ordering::Acquire), 1_000);
        assert_eq!(device.insert_counter, 500);
        assert_eq!(producer.reclaim(), 1);
        let mut next = vec![0; channels];
        assert_eq!(device.render(&mut next, true, insert_plan), 0);
        assert_eq!(next, samples[1_000 * channels..1_001 * channels]);
        assert_eq!(device.pipe.progress.load(Ordering::Acquire), 1_001);
    }
}

#[test]
fn realtime_handoff_probe_tape_clear_commit_invalidates_phase_before_checkpoint_cleanup() {
    for length in [499, 503] {
        let samples: Vec<i32> = (1..=length as i32).collect();
        let mut preparation = Preparation::with_window_frames(vec![(1, source_pcm(&samples))], 512);
        let (mut producer, pipe) = pipe_for::<Tape>(1);
        let mut device = TapeDevice::new(pipe, Arc::new(PublishedCheckpoint::new(1)), 1);
        assert!(producer
            .publish(prepare_tape(&mut preparation, 0, 0, length))
            .is_ok());
        let schedule = CorrectionPlanner::new().plan(-10_000, 1_000, false);
        assert_eq!(device.render(&mut [0; 499], true, schedule), 0);
        assert_eq!(device.insert_counter, 1);
        let view = device.pipe.gate.observe().unwrap();
        // Exact protocol interval inside SourceLedger::try_clear_with: the clear
        // CAS is visible but the non-RT checkpoint cleanup has not executed yet.
        device.pipe.gate.clear(view).unwrap();
        assert!(device.checkpoint.is_valid());
        let mut output = [-1];
        assert_eq!(device.render(&mut output, true, schedule), 1);
        assert_eq!(output, [0], "no replay of pre-clear last frame");
        assert!(
            device.position.is_none(),
            "no old position may be republished"
        );
        assert_eq!(device.pipe.progress.load(Ordering::Acquire), 499);
        device.checkpoint.invalidate_after_clear();
        assert_eq!(producer.reclaim(), 1);
    }
}

#[test]
fn realtime_handoff_probe_tape_paused_reclaim_drains_all_credits_then_resumes() {
    let (mut producer, pipe) = pipe_for::<Tape>(3);
    let mut device = TapeDevice::new(pipe, Arc::new(PublishedCheckpoint::new(1)), 1);
    let mut preparation = Preparation::new(vec![(1, source_pcm(&[1, 2, 3, 4, 5, 6, 7, 8]))]);
    let (dropped_tx, dropped_rx) = mpsc::channel();
    for base in [0, 2, 4] {
        let mut tape = prepare_tape(&mut preparation, 0, base, 2);
        tape.span.lifetime = Some(DropWitness(dropped_tx.clone()));
        assert!(producer.publish(tape).is_ok());
    }
    // Neither prepare nor reclaim runs during this render. All three credits
    // may retire; the seventh output frame is actual exhaustion, not contention.
    let mut output = [0; 7];
    assert_eq!(
        device.render(&mut output, false, CorrectionSchedule::default()),
        1
    );
    assert_eq!(output, [1, 2, 3, 4, 5, 6, 0]);
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 6);
    assert!(dropped_rx.try_recv().is_err());
    assert!(device.pipe.pending_return.is_none());
    assert_eq!(producer.retired.slots(), 3);
    let pending = prepare_tape(&mut preparation, 0, 6, 2);
    let pending = producer
        .publish(pending)
        .err()
        .expect("internal credits exhausted, payload returned");
    assert_eq!(producer.reclaim(), 3);
    for _ in 0..3 {
        assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
    }
    assert!(producer.publish(pending).is_ok());
    let mut resumed = [0; 2];
    assert_eq!(
        device.render(&mut resumed, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(resumed, [7, 8]);
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 8);
    assert_eq!(producer.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_tape_single_preparer_retains_pending_and_replaces_view_before_build() {
    let (pipe, reader) = pipe_for::<Tape>(1);
    let mut device = TapeDevice::new(reader, Arc::new(PublishedCheckpoint::new(1)), 1);
    let mut producer = TapePreparation {
        source: None,
        revision: None,
        pending: None,
        pipe,
        epoch: 0,
        source_base: 0,
    };
    producer.reset(0, 0, || {
        Preparation::with_window_frames(vec![(1, source_pcm(&[1, 2, 3, 4]))], 2)
    });
    assert!(
        producer.pump(usize::MAX),
        "request is clamped before temporary allocation"
    );
    assert!(!producer.pump(usize::MAX));
    let pending_pcm = producer.pending.as_ref().unwrap().span.pcm.as_ptr();
    for _ in 0..10 {
        assert!(!producer.pump(usize::MAX));
        assert_eq!(
            producer.pending.as_ref().unwrap().span.pcm.as_ptr(),
            pending_pcm
        );
        assert_eq!(
            producer.source_base, 4,
            "failed publication must not build further windows"
        );
    }
    let mut output = [0; 2];
    assert_eq!(
        device.render(&mut output, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(output, [1, 2]);
    assert_eq!(producer.pipe.reclaim(), 1);
    assert!(producer.pump(2));
    assert_eq!(
        device.render(&mut output, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(output, [3, 4]);
    assert_eq!(producer.pipe.reclaim(), 1);
    // Lazy snapshot creation sees the previous view's PCM already released.
    let old_pcm: Arc<[i32]> = Arc::from([9, 10]);
    let weak = Arc::downgrade(&old_pcm);
    let mut old = source_pcm(&[]);
    old.samples = old_pcm;
    producer.reset(0, 4, || Preparation::new(vec![(2, old)]));
    producer.reset(1, 4, || {
        assert!(weak.upgrade().is_none());
        Preparation::new(vec![(3, source_pcm(&[11, 12]))])
    });
    assert!(producer.pending.is_none());
    assert_eq!(producer.source_base, 4);
}

#[test]
fn realtime_handoff_probe_tape_preparer_uses_ledger_for_append_overlap_and_clear() {
    use super::super::admission::SourceLedger;
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome};
    let checkpoint = Arc::new(PublishedCheckpoint::new(1));
    let (pipe, reader) = pipe_for::<Tape>(2);
    let mut ledger =
        SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
    let mut device = TapeDevice::new(reader, checkpoint, 1);
    let mut producer = TapePreparation {
        source: None,
        revision: None,
        pending: None,
        pipe,
        epoch: 0,
        source_base: 0,
    };
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[1, 2, 3, 4]), &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 8),
        Ok(true)
    );
    let mut first = [0];
    assert_eq!(
        device.render(&mut first, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(first, [1]);
    let mut append = source_pcm(&[10, 11]);
    append.timestamp = 4_000;
    assert!(matches!(
        ledger.try_enqueue(append, &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 8),
        Ok(true)
    );
    let mut replacement = source_pcm(&[50, 51]);
    replacement.timestamp = 4_000;
    assert!(matches!(
        ledger.try_enqueue(replacement, &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 8),
        Ok(false)
    );
    assert!(producer.pending.is_some());
    let mut tail = [0; 3];
    assert_eq!(
        device.render(&mut tail, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(tail, [2, 3, 4]);
    assert_eq!(producer.pipe.reclaim(), 1);
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 8),
        Ok(true)
    );
    let mut replaced = [0; 2];
    assert_eq!(
        device.render(&mut replaced, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(replaced, [50, 51]);
    assert_eq!(producer.pipe.reclaim(), 2);
    assert_eq!(
        ledger.try_clear(&device.pipe.gate),
        Ok(RendererOperationOutcome::Applied)
    );
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[9, 8]), &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 8),
        Ok(true)
    );
    let mut fresh = [0; 2];
    assert_eq!(
        device.render(&mut fresh, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(fresh, [9, 8]);
    ledger.try_reconcile(&device.pipe.gate).unwrap();
    assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(8));
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        0
    );
}

#[test]
fn realtime_handoff_probe_tape_reanchor_rebuilds_current_time_without_clearing_current_pcm() {
    use super::super::admission::SourceLedger;
    use crate::audio::EnqueueOutcome;
    let checkpoint = Arc::new(PublishedCheckpoint::new(1));
    let (pipe, reader) = pipe_for::<Tape>(2);
    let mut ledger =
        SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
    let mut device = TapeDevice::new(reader, checkpoint, 1);
    let mut producer = TapePreparation {
        source: None,
        revision: None,
        pending: None,
        pipe,
        epoch: 0,
        source_base: 0,
    };
    for (timestamp, pcm) in [
        (0, [1, 2, 3, 4]),
        (10_000, [10, 11, 12, 13]),
        (20_000, [20, 21, 22, 23]),
    ] {
        let mut source = source_pcm(&pcm);
        source.timestamp = timestamp;
        assert!(matches!(
            ledger.try_enqueue(source, &device.pipe.gate, || {}),
            Ok(EnqueueOutcome::Accepted { .. })
        ));
    }
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 16),
        Ok(true)
    );
    let mut first = [0];
    assert_eq!(
        device.render(&mut first, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(first, [1]);
    assert_eq!(ledger.try_reanchor(&device.pipe.gate, 20_000), Ok(()));
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 16),
        Ok(true)
    );
    let mut tail = [0; 4];
    assert_eq!(
        device.render(&mut tail, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(tail, [2, 3, 4, 23]);
    let actual = device
        .checkpoint
        .read(&device.pipe.gate, &device.pipe.progress)
        .unwrap();
    assert_eq!(actual.position.cursor_us, 24_000);
    assert_eq!(actual.source_frames, 5);
    drop(actual);
    ledger.try_reconcile(&device.pipe.gate).unwrap();
    assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(5));
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        0
    );
    assert_eq!(producer.pipe.reclaim(), 2);
}

#[test]
fn realtime_handoff_probe_tape_reanchor_only_skipped_sources_releases_capacity_without_consumption()
{
    use super::super::admission::SourceLedger;
    use crate::audio::EnqueueOutcome;
    let checkpoint = Arc::new(PublishedCheckpoint::new(1));
    let (pipe, reader) = pipe_for::<Tape>(2);
    let mut ledger =
        SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
    let mut device = TapeDevice::new(reader, checkpoint, 1);
    let mut producer = TapePreparation {
        source: None,
        revision: None,
        pending: None,
        pipe,
        epoch: 0,
        source_base: 0,
    };
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[1, 2, 3, 4]), &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 8),
        Ok(true)
    );
    assert_eq!(ledger.try_reanchor(&device.pipe.gate, 20_000), Ok(()));
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 8),
        Ok(true),
        "publish skipped-source position even with no PCM"
    );
    let mut output = [-1; 2];
    assert_eq!(
        device.render(&mut output, false, CorrectionSchedule::default()),
        2
    );
    assert_eq!(output, [0, 0]);
    ledger.try_reconcile(&device.pipe.gate).unwrap();
    assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(0));
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        0
    );
    assert_eq!(producer.pipe.reclaim(), 2);
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 8),
        Ok(false),
        "no repeated empty publications"
    );
    let mut next = source_pcm(&[9, 8]);
    next.timestamp = 20_000;
    assert!(matches!(
        ledger.try_enqueue(next, &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    assert_eq!(
        producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 8),
        Ok(true)
    );
    assert_eq!(
        device.render(&mut output, false, CorrectionSchedule::default()),
        0
    );
    assert_eq!(output, [9, 8]);
    ledger.try_reconcile(&device.pipe.gate).unwrap();
    assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(2));
}

#[test]
fn realtime_handoff_probe_tape_published_plan_respects_timeline_and_model_invalidation() {
    let plans = super::feedback::PlanMailbox::default();
    let (mut producer, reader) = pipe_for::<Tape>(2);
    let mut device = TapeDevice::new(reader, Arc::new(PublishedCheckpoint::new(1)), 1);
    let mut source =
        Preparation::with_window_frames(vec![(1, source_pcm(&(1..=503).collect::<Vec<_>>()))], 512);
    assert!(producer
        .publish(prepare_tape(&mut source, 0, 0, 503))
        .is_ok());
    let correcting = CorrectionPlanner::new().plan(-10_000, 1_000, false);
    plans.publish(0, Some(correcting));
    assert_eq!(device.render_published(&mut [0; 499], true, &plans), 0);
    assert_eq!(device.insert_counter, 1);
    // Missing actual timestamp preserves phase; explicit model invalidation
    // must clear it even on this same fallback path.
    plans.publish(0, None);
    let mut frame = [0];
    assert_eq!(device.render_published(&mut frame, false, &plans), 0);
    assert_eq!(frame, [500]);
    assert_eq!(device.schedule, CorrectionSchedule::default());
    plans.publish(0, Some(correcting));
    let cleared = device
        .pipe
        .gate
        .clear(device.pipe.gate.observe().unwrap())
        .unwrap();
    let mut fresh = Preparation::new(vec![(2, source_pcm(&[9, 8]))]);
    assert!(producer
        .publish(prepare_tape(&mut fresh, cleared.epoch(), 500, 2))
        .is_ok());
    assert_eq!(device.render_published(&mut frame, true, &plans), 0);
    assert_eq!(frame, [9]);
    assert_eq!(
        device.schedule,
        CorrectionSchedule::default(),
        "old plan cannot reactivate after clear"
    );
    // Overlap preparation epochs do not by themselves reset the sync timeline.
    plans.publish(cleared.epoch(), Some(correcting));
    device
        .pipe
        .gate
        .replace(device.pipe.gate.observe().unwrap())
        .unwrap();
    assert_eq!(device.render_published(&mut frame, true, &plans), 0);
    assert_eq!(frame, [8]);
    assert_eq!(device.schedule, correcting);
    assert_eq!(producer.reclaim(), 2);
    let reanchored = device
        .pipe
        .gate
        .reanchor(device.pipe.gate.observe().unwrap())
        .unwrap();
    let mut later = Preparation::new(vec![(3, source_pcm(&[7]))]);
    assert!(producer
        .publish(prepare_tape(&mut later, reanchored.epoch(), 502, 1))
        .is_ok());
    assert_eq!(device.render_published(&mut frame, true, &plans), 0);
    assert_eq!(frame, [7]);
    assert_eq!(
        device.schedule,
        CorrectionSchedule::default(),
        "old plan cannot reactivate after reanchor"
    );
}

#[test]
fn realtime_handoff_probe_tape_cross_clock_feedback_drives_actual_correction_and_expiry() {
    use super::feedback::{Feedback, Observation, PlanMailbox};
    use crate::sync::{Clock, ClockSync};
    use std::sync::atomic::AtomicI64;
    use std::time::{Duration, Instant};
    struct EndpointClock {
        origin: Instant,
        now: AtomicI64,
    }
    impl Clock for EndpointClock {
        fn now_micros(&self) -> i64 {
            self.now.load(Ordering::Acquire)
        }
        fn micros_to_instant(&self, us: i64) -> Option<Instant> {
            if us >= 0 {
                self.origin.checked_add(Duration::from_micros(us as u64))
            } else {
                self.origin
                    .checked_sub(Duration::from_micros(us.unsigned_abs()))
            }
        }
    }
    let origin = Instant::now();
    let clock = Arc::new(EndpointClock {
        origin,
        now: AtomicI64::new(0),
    });
    let mut sync = ClockSync::new(clock.clone());
    // Exact synthetic server offset +100ms, symmetric 2ms RTT; real filter.
    for i in 0..10 {
        let client = i * 1_000_000;
        clock.now.store(client + 2_000, Ordering::Release);
        sync.update(client, client + 101_000, client + 101_000, client + 2_000);
    }
    assert!(sync.is_settled());
    assert_eq!(sync.server_to_client_micros(10_100_000), Some(10_000_000));
    let plans = PlanMailbox::default();
    let mut feedback = Feedback::new();
    let (mut producer, reader) = pipe_for::<Tape>(1);
    let mut device = TapeDevice::new(reader, Arc::new(PublishedCheckpoint::new(1)), 1);
    let mut source = source_pcm(&(1..=610).collect::<Vec<_>>());
    source.timestamp = 10_100_000;
    let mut preparation = Preparation::with_window_frames(vec![(1, source)], 1024);
    assert!(producer
        .publish(prepare_tape(&mut preparation, 0, 0, 610))
        .is_ok());
    for serial in 1..=101 {
        let observation = Observation {
            settled_seen: true,
            timeline: device.sync_timeline,
            serial,
            source_cursor_us: device
                .position
                .map_or(10_100_000, |position| position.cursor_us),
            presentation: Some(origin + Duration::from_micros(10_010_000 + (serial - 1) * 1_000)),
            actual_schedule: device.schedule,
        };
        let mut frame = [0];
        assert_eq!(device.render_published(&mut frame, true, &plans), 0);
        assert_eq!(frame, [serial as i32]);
        feedback.publish_observation(&sync, observation, 0, device.sync_timeline, &plans);
    }
    assert_eq!(plans.read(0).unwrap().unwrap().drop_every_n_frames, 500);
    let mut prefix = [0; 499];
    assert_eq!(device.render_published(&mut prefix, true, &plans), 0);
    assert_eq!(prefix, std::array::from_fn(|i| i as i32 + 102));
    let mut frame = [0];
    assert_eq!(device.render_published(&mut frame, true, &plans), 0);
    assert_eq!(frame, [602]);
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 602);
    clock.now.store(14_002_001, Ordering::Release);
    assert!(!sync.is_synchronized());
    feedback.publish_observation(
        &sync,
        Observation {
            settled_seen: true,
            timeline: device.sync_timeline,
            serial: 102,
            source_cursor_us: device.position.unwrap().cursor_us,
            presentation: None,
            actual_schedule: device.schedule,
        },
        0,
        device.sync_timeline,
        &plans,
    );
    assert_eq!(plans.read(0), Some(None));
    assert_eq!(device.render_published(&mut frame, false, &plans), 0);
    assert_eq!(frame, [603]);
    assert_eq!(device.schedule, CorrectionSchedule::default());
}

#[test]
fn realtime_handoff_probe_tape_reuses_owner_close_and_scope_fence_without_consuming_old_pcm() {
    use super::super::admission::SourceLedger;
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome};
    let checkpoint = Arc::new(PublishedCheckpoint::new(1));
    let (mut producer, reader) = pipe_for::<Tape>(1);
    let mut ledger =
        SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
    let mut device = TapeDevice::new(reader, checkpoint, 1);
    device.owner = Some((ledger.owner.clone(), ledger.scope));
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[1, 2]), &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    let (view, mut source) = ledger.preparation(&device.pipe.gate).unwrap();
    assert!(producer
        .publish(prepare_tape(&mut source, view.epoch(), 0, 2))
        .is_ok());
    assert_eq!(
        ledger.owner.close(ledger.scope),
        RendererOperationOutcome::Applied
    );
    let mut output = [-1; 2];
    assert_eq!(
        device.render(&mut output, false, CorrectionSchedule::default()),
        2
    );
    assert_eq!(output, [0, 0]);
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 0);
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[3]), &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::Closed)
    ));
    // Scope replacement is permitted by the real owner only after finalization.
    // This fixture has no attached device resource; it does not prove a stop.
    ledger.owner.teardown(ledger.scope);
    let replacement = ledger.owner.mint_scope().unwrap();
    assert_ne!(replacement, ledger.scope);
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[4]), &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::StaleScope)
    ));
    let plans = super::feedback::PlanMailbox::default();
    assert_eq!(device.render_published(&mut output, true, &plans), 2);
    assert_eq!(output, [0, 0]);
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 0);
}

#[test]
fn realtime_handoff_probe_tape_start_boundary_and_cancel_control_actual_pcm_consumption() {
    use super::super::start::StartAuthority;
    use crate::audio::ScheduledArmOutcome;
    for cancelled in [false, true] {
        let (mut producer, reader) = pipe_for::<Tape>(1);
        let mut device = TapeDevice::new(reader, Arc::new(PublishedCheckpoint::new(1)), 1);
        let mut source = Preparation::new(vec![(1, source_pcm(&[10, 20]))]);
        assert!(producer.publish(prepare_tape(&mut source, 0, 0, 2)).is_ok());
        let start = StartAuthority::default();
        assert_eq!(start.arm(100), ScheduledArmOutcome::Armed);
        let plans = super::feedback::PlanMailbox::default();
        let mut output = [-1];
        for presentation in [None, Some(99)] {
            assert_eq!(
                device.render_scheduled(&mut output, presentation, true, &plans, &start),
                None
            );
            assert_eq!(output, [0]);
            assert_eq!(device.pipe.progress.load(Ordering::Acquire), 0);
            assert_eq!(producer.reclaim(), 0);
        }
        if cancelled {
            start.cancel().unwrap();
        }
        assert_eq!(
            device.render_scheduled(&mut output, Some(100), true, &plans, &start),
            if cancelled { None } else { Some(0) }
        );
        assert_eq!(output, if cancelled { [0] } else { [10] });
        if !cancelled {
            assert_eq!(start.cancel(), Err(100));
        }
        assert_eq!(
            device.render_scheduled(&mut output, None, false, &plans, &start),
            if cancelled { None } else { Some(0) }
        );
        assert_eq!(output, if cancelled { [0] } else { [20] });
        assert_eq!(
            device.pipe.progress.load(Ordering::Acquire),
            if cancelled { 0 } else { 2 }
        );
        assert_eq!(producer.reclaim(), usize::from(!cancelled));
    }
}

#[test]
fn realtime_handoff_probe_separate_clear_and_start_can_freeze_a_revoked_boundary() {
    use super::super::{admission::SourceLedger, start::StartAuthority};
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome, ScheduledArmOutcome};
    let checkpoint = Arc::new(PublishedCheckpoint::new(1));
    let (mut producer, reader) = pipe_for::<Tape>(1);
    let mut ledger =
        SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
    let mut device = TapeDevice::new(reader, checkpoint, 1);
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[10, 20]), &device.pipe.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    let (view, mut source) = ledger.preparation(&device.pipe.gate).unwrap();
    assert!(producer
        .publish(prepare_tape(&mut source, view.epoch(), 0, 2))
        .is_ok());
    let start = StartAuthority::default();
    assert_eq!(start.arm(100), ScheduledArmOutcome::Armed);
    assert_eq!(
        ledger.owner.arm_scheduled_start(ledger.scope, 100),
        ScheduledArmOutcome::Armed
    );
    let gate = Arc::clone(&device.pipe.gate);
    let plans = super::feedback::PlanMailbox::default();
    let mut output = [-1];
    // Diagnostic, not compatibility GREEN: source clear has committed, but the
    // separate start reset has not. A callback can freeze the revoked boundary.
    assert_eq!(
        ledger.try_clear_with(&gate, || {
            assert_eq!(
                device.render_scheduled(&mut output, Some(100), true, &plans, &start),
                Some(1)
            );
            start.clear_unstarted();
        }),
        Ok(RendererOperationOutcome::Applied)
    );
    assert_eq!(output, [0]);
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 0);
    assert_eq!(
        start.arm(200),
        ScheduledArmOutcome::BoundaryAlreadyWon {
            start_at_zone_us: 100
        }
    );
    assert_eq!(
        ledger.owner.arm_scheduled_start(ledger.scope, 200),
        ScheduledArmOutcome::Armed
    );
}

#[test]
fn realtime_handoff_probe_gate_start_clear_and_rearm_share_pcm_revocation() {
    use super::super::{admission::SourceLedger, start::GateStartAuthority};
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome, ScheduledArmOutcome};
    for boundary_won in [false, true] {
        let checkpoint = Arc::new(PublishedCheckpoint::new(1));
        let (mut producer, reader) = pipe_for::<Tape>(2);
        let mut ledger =
            SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
        let mut device = TapeDevice::new(reader, checkpoint, 1);
        let gate = Arc::clone(&device.pipe.gate);
        let start = GateStartAuthority::new(Arc::clone(&gate));
        let plans = super::feedback::PlanMailbox::default();
        assert!(matches!(
            ledger.try_enqueue(source_pcm(&[10, 20]), &gate, || {}),
            Ok(EnqueueOutcome::Accepted { .. })
        ));
        let (view, mut source) = ledger.preparation(&gate).unwrap();
        assert!(producer
            .publish(prepare_tape(&mut source, view.epoch(), 0, 2))
            .is_ok());
        assert_eq!(start.arm(100), Ok(ScheduledArmOutcome::Armed));
        let mut output = [-1];
        if boundary_won {
            assert_eq!(
                device.render_scheduled(&mut output, Some(100), true, &plans, &start),
                Some(0)
            );
            assert_eq!(output, [10]);
        }
        assert_eq!(
            ledger.try_clear_with(&gate, || {
                // Same exact seam as the failing independent-state fixture: source
                // is revoked and the non-RT clear operation has not returned.
                assert_eq!(
                    device.render_scheduled(&mut output, Some(100), true, &plans, &start),
                    Some(1)
                );
                assert_eq!(output, [0]);
            }),
            Ok(RendererOperationOutcome::Applied)
        );
        let expected = if boundary_won {
            ScheduledArmOutcome::BoundaryAlreadyWon {
                start_at_zone_us: 100,
            }
        } else {
            ScheduledArmOutcome::Armed
        };
        assert_eq!(start.arm(200), Ok(expected));
        // Leave another clear notice pending across re-arm: the next callback
        // must not erase the newly armed boundary just because it sees a notice.
        assert_eq!(
            ledger.try_clear(&gate),
            Ok(RendererOperationOutcome::Applied)
        );
        assert_eq!(start.arm(200), Ok(expected));
        assert!(matches!(
            ledger.try_enqueue(source_pcm(&[30, 40]), &gate, || {}),
            Ok(EnqueueOutcome::Accepted { .. })
        ));
        let (view, mut source) = ledger.preparation(&gate).unwrap();
        let base = device.pipe.progress.load(Ordering::Acquire);
        assert!(producer
            .publish(prepare_tape(&mut source, view.epoch(), base, 2))
            .is_ok());
        assert_eq!(
            device.render_scheduled(&mut output, Some(100), true, &plans, &start),
            if boundary_won { Some(0) } else { None }
        );
        assert_eq!(output, if boundary_won { [30] } else { [0] });
        if !boundary_won {
            assert_eq!(
                device.render_scheduled(&mut output, Some(200), true, &plans, &start),
                Some(0)
            );
            assert_eq!(output, [30]);
        }
        assert_eq!(
            device.render_scheduled(&mut output, None, false, &plans, &start),
            Some(0)
        );
        assert_eq!(output, [40]);
        assert_eq!(device.pipe.progress.load(Ordering::Acquire), base + 2);
        assert_eq!(start.cancel(), Err(if boundary_won { 100 } else { 200 }));
    }
}

#[test]
fn realtime_handoff_probe_gate_start_real_finalizer_joins_before_pre_start_ack() {
    use super::super::start::GateStartAuthority;
    use crate::audio::{
        RendererOperationOutcome, RendererOwner, RendererQueueLimits, ScheduledArmOutcome,
        TerminalOutcome, TerminalState, TerminalWinner,
    };
    struct JoinedResource(Option<Box<dyn FnOnce() + Send>>);
    impl Drop for JoinedResource {
        fn drop(&mut self) {
            self.0.take().unwrap()();
        }
    }
    for boundary_won in [false, true] {
        let owner = RendererOwner::new(RendererQueueLimits::new(32, 8, 16).unwrap());
        let scope = owner.mint_scope().unwrap();
        let (mut producer, pipe) = pipe_for::<Tape>(1);
        let actual = Arc::clone(&pipe.progress);
        let start = Arc::new(GateStartAuthority::new(Arc::clone(&pipe.gate)));
        assert_eq!(start.arm(100), Ok(ScheduledArmOutcome::Armed));
        let mut device = TapeDevice::new(pipe, Arc::new(PublishedCheckpoint::new(1)), 1);
        let (published_tx, published_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::channel();
        let (dropped_tx, dropped_rx) = mpsc::channel();
        let mut source = Preparation::new(vec![(1, source_pcm(&[10, 20, 30]))]);
        let mut private = source.fork_actual();
        let preparation = thread::spawn(move || {
            let mut tape = prepare_tape(&mut private, 0, 0, 3);
            tape.span.lifetime = Some(DropWitness(dropped_tx));
            assert!(producer.publish(tape).is_ok());
            published_tx.send(()).unwrap();
            stop_rx.recv().unwrap();
            producer
        });
        published_rx.recv().unwrap();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let reader_start = Arc::clone(&start);
        let reader = thread::spawn(move || {
            let plans = super::feedback::PlanMailbox::default();
            if boundary_won {
                let mut first = [-1];
                assert_eq!(
                    device.render_scheduled(&mut first, Some(100), true, &plans, &*reader_start),
                    Some(0)
                );
                assert_eq!(first, [10]);
            }
            device.pipe.gate.begin_callback();
            entered_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
            let mut output = [-1];
            let result = device.render_scheduled_active(
                &mut output,
                Some(100),
                true,
                &plans,
                &*reader_start,
            );
            (device, output, result)
        });
        entered_rx.recv().unwrap();
        assert_eq!(actual.load(Ordering::Acquire), u64::from(boundary_won));
        let observed_owner = owner.clone();
        let (observed_tx, observed_rx) = mpsc::channel();
        assert_eq!(
            owner.attach_test_terminal_resource(
                scope,
                Box::new(JoinedResource(Some(Box::new(move || {
                    assert!(matches!(
                        observed_owner.terminal_state(scope).unwrap(),
                        TerminalState::Finalizing { .. }
                    ));
                    stop_tx.send(()).unwrap();
                    resume_tx.send(()).unwrap();
                    let producer = preparation.join().unwrap();
                    let (device, output, result) = reader.join().unwrap();
                    drop(device);
                    drop(producer);
                    observed_tx.send((output, result)).unwrap();
                }))))
            ),
            RendererOperationOutcome::Applied
        );
        let mut cleared = false;
        let outcome =
            owner.finalize_test_pre_start_with_authority(scope, &|| start.cancel(), || {
                source.reset();
                cleared = true;
            });
        let finalization = if boundary_won {
            assert_eq!(outcome, Err(100));
            assert!(!cleared);
            assert!(observed_rx.try_recv().is_err());
            assert!(dropped_rx.try_recv().is_err());
            assert_eq!(owner.terminal_state(scope), Ok(TerminalState::Open));
            let TerminalOutcome::Won(done) = owner.teardown(scope) else {
                panic!("teardown winner")
            };
            assert_eq!(done.winner, TerminalWinner::ExplicitTeardown);
            done
        } else {
            assert!(cleared);
            assert_eq!(source.cursor.queued_frames(1), 0);
            let Ok(TerminalOutcome::Won(done)) = outcome else {
                panic!("pre-start winner")
            };
            assert_eq!(done.winner, TerminalWinner::PreStartAbort);
            done
        };
        assert_eq!(
            observed_rx.recv().unwrap(),
            if boundary_won {
                ([20], Some(0))
            } else {
                ([0], None)
            }
        );
        assert_eq!(
            actual.load(Ordering::Acquire),
            if boundary_won { 2 } else { 0 }
        );
        assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
        assert!(finalization.ack.callback_stopped());
        assert!(finalization.ack.stream_released());
        assert_eq!(
            owner.finalize_test_pre_start_with_authority(
                scope,
                &|| panic!("do not claim twice"),
                || {}
            ),
            Ok(TerminalOutcome::AlreadyFinalized(finalization))
        );
    }
}

#[test]
fn realtime_handoff_probe_tape_actual_feedback_survives_paused_worker_and_full_channel() {
    use super::feedback::{Feedback, PlanMailbox};
    use crate::sync::{Clock, ClockSync};
    use std::time::{Duration, Instant};
    struct FixedClock(Instant);
    impl Clock for FixedClock {
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
    let origin = Instant::now();
    let plans = Arc::new(PlanMailbox::default());
    let (mut producer, reader) = pipe_for::<Tape>(1);
    let mut device = TapeDevice::new(reader, Arc::new(PublishedCheckpoint::new(1)), 1);
    let mut source = Preparation::new(vec![(1, source_pcm(&[1, 2, 3, 4, 5, 6]))]);
    assert!(producer.publish(prepare_tape(&mut source, 0, 0, 6)).is_ok());
    let (mut observation_tx, mut observation_rx) = rtrb::RingBuffer::new(2);
    let worker_plans = Arc::clone(&plans);
    let (resume_tx, resume_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let sync = ClockSync::new_same_clock(Arc::new(FixedClock(origin)));
        let mut feedback = Feedback::new();
        let mut seen = Vec::new();
        for count in [2, 1] {
            resume_rx.recv().unwrap();
            for _ in 0..count {
                let observation: super::feedback::Observation = observation_rx.pop().unwrap();
                seen.push((
                    observation.serial,
                    observation.source_cursor_us,
                    observation.presentation,
                ));
                feedback.publish_observation(&sync, observation, 0, 0, &worker_plans);
            }
            done_tx.send(()).unwrap();
        }
        seen
    });
    let mut output = [-1];
    // Worker is parked. Initial cursor is carried by the accepted descriptor;
    // observations beyond the two slots are dropped without interrupting PCM.
    for (sample, published) in [(1, true), (2, true), (3, false), (4, false)] {
        assert_eq!(
            device.render_feedback(
                &mut output,
                Some(origin + Duration::from_micros((sample - 1) * 1000)),
                &plans,
                &mut observation_tx
            ),
            (0, published)
        );
        assert_eq!(output, [sample as i32]);
    }
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 4);
    resume_tx.send(()).unwrap();
    done_rx.recv().unwrap();
    assert_eq!(
        device.render_feedback(&mut output, None, &plans, &mut observation_tx),
        (0, true)
    );
    assert_eq!(output, [5]);
    resume_tx.send(()).unwrap();
    done_rx.recv().unwrap();
    assert_eq!(
        worker.join().unwrap(),
        vec![
            (1, 0, Some(origin)),
            (2, 1000, Some(origin + Duration::from_micros(1000))),
            (5, 4000, None),
        ]
    );
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 5);
    // Missing observations and fallback preserve the last measured plan.
    assert_eq!(plans.read(0), Some(Some(CorrectionSchedule::default())));
}

#[test]
fn realtime_handoff_probe_tape_feedback_origin_tracks_initial_clear_and_reanchor() {
    use super::super::admission::SourceLedger;
    use super::feedback::PlanMailbox;
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome};
    let checkpoint = Arc::new(PublishedCheckpoint::new(1));
    let (mut producer, reader) = pipe_for::<Tape>(2);
    let mut ledger =
        SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
    let mut device = TapeDevice::new(reader, checkpoint, 1);
    let gate = Arc::clone(&device.pipe.gate);
    let plans = PlanMailbox::default();
    let (mut tx, mut rx) = rtrb::RingBuffer::new(1);
    let mut initial = source_pcm(&[10, 20]);
    initial.timestamp = 10_000;
    assert!(matches!(
        ledger.try_enqueue(initial, &gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    let (view, mut source) = ledger.preparation(&gate).unwrap();
    assert!(producer
        .publish(prepare_tape(&mut source, view.epoch(), 0, 2))
        .is_ok());
    let mut output = [-1];
    assert_eq!(
        device.render_feedback(&mut output, None, &plans, &mut tx),
        (0, true)
    );
    assert_eq!(output, [10]);
    let observation = rx.pop().unwrap();
    assert_eq!(
        (observation.timeline, observation.source_cursor_us),
        (view.epoch(), 10_000)
    );
    ledger.try_reanchor(&gate, 20_000).unwrap();
    let (view, mut source) = ledger.preparation(&gate).unwrap();
    assert!(producer
        .publish(prepare_tape(&mut source, view.epoch(), 1, 2))
        .is_ok());
    assert_eq!(
        device.render_feedback(&mut output, None, &plans, &mut tx),
        (0, true)
    );
    assert_eq!(
        output,
        [20],
        "reanchor preserves the partially consumed current"
    );
    let observation = rx.pop().unwrap();
    assert_eq!(
        (observation.timeline, observation.source_cursor_us),
        (view.epoch(), 20_000)
    );
    assert_eq!(producer.reclaim(), 2);
    assert_eq!(
        ledger.try_clear(&gate),
        Ok(RendererOperationOutcome::Applied)
    );
    let mut fresh = source_pcm(&[30, 40]);
    fresh.timestamp = 30_000;
    assert!(matches!(
        ledger.try_enqueue(fresh, &gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    let (view, mut source) = ledger.preparation(&gate).unwrap();
    assert!(producer
        .publish(prepare_tape(&mut source, view.epoch(), 2, 2))
        .is_ok());
    assert_eq!(
        device.render_feedback(&mut output, None, &plans, &mut tx),
        (0, true)
    );
    assert_eq!(output, [30]);
    let observation = rx.pop().unwrap();
    assert_eq!(
        (observation.timeline, observation.source_cursor_us),
        (view.epoch(), 30_000)
    );
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 3);
}

#[test]
fn realtime_handoff_probe_tape_feedback_reports_plan_actually_applied_in_same_callback() {
    use super::feedback::PlanMailbox;
    use std::time::{Duration, Instant};
    let (mut producer, reader) = pipe_for::<Tape>(1);
    let mut device = TapeDevice::new(reader, Arc::new(PublishedCheckpoint::new(1)), 1);
    let mut source =
        Preparation::with_window_frames(vec![(1, source_pcm(&(1..=510).collect::<Vec<_>>()))], 512);
    assert!(producer
        .publish(prepare_tape(&mut source, 0, 0, 510))
        .is_ok());
    let (mut tx, mut rx) = rtrb::RingBuffer::new(1);
    let plans = PlanMailbox::default();
    let drop_plan = CorrectionPlanner::new().plan(10_000, 1_000, false);
    let insert_plan = CorrectionPlanner::new().plan(-10_000, 1_000, true);
    assert_eq!(drop_plan.drop_every_n_frames, 500);
    assert_eq!(insert_plan.insert_every_n_frames, 500);
    let origin = Instant::now();
    plans.publish(0, Some(drop_plan));
    let mut prefix = [0; 499];
    assert_eq!(
        device.render_feedback(&mut prefix, Some(origin), &plans, &mut tx),
        (0, true)
    );
    assert_eq!(prefix, std::array::from_fn(|i| i as i32 + 1));
    let observation = rx.pop().unwrap();
    assert_eq!(observation.source_cursor_us, 0);
    assert_eq!(observation.presentation, Some(origin));
    assert_eq!(observation.actual_schedule, drop_plan);
    plans.publish(0, Some(insert_plan));
    let mut output = [0];
    // Fallback does not apply the newly published plan or advance correction.
    assert_eq!(
        device.render_feedback(&mut output, None, &plans, &mut tx),
        (0, true)
    );
    assert_eq!(output, [500]);
    let observation = rx.pop().unwrap();
    assert_eq!(observation.source_cursor_us, 499_000);
    assert_eq!(observation.presentation, None);
    assert_eq!(observation.actual_schedule, drop_plan);
    let presentation = Some(origin + Duration::from_micros(500_000));
    assert_eq!(
        device.render_feedback(&mut output, presentation, &plans, &mut tx),
        (0, true)
    );
    assert_eq!(output, [501]);
    let observation = rx.pop().unwrap();
    assert_eq!(observation.source_cursor_us, 500_000);
    assert_eq!(observation.presentation, presentation);
    assert_eq!(observation.actual_schedule, insert_plan);
    plans.publish(0, None);
    assert_eq!(
        device.render_feedback(&mut output, None, &plans, &mut tx),
        (0, true)
    );
    assert_eq!(output, [502]);
    let observation = rx.pop().unwrap();
    assert_eq!(observation.source_cursor_us, 501_000);
    assert_eq!(observation.actual_schedule, CorrectionSchedule::default());
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 502);
}

#[test]
fn realtime_handoff_probe_tape_old_plan_expires_while_publication_is_paused() {
    use super::feedback::{ModelValidity, ValidatedPlanMailbox, ValidatedPlanReader};
    let plans = Arc::new(ValidatedPlanMailbox::default());
    let drop_plan = CorrectionPlanner::new().plan(10_000, 1_000, false);
    let insert_plan = CorrectionPlanner::new().plan(-10_000, 1_000, true);
    plans.publish(0, Some(drop_plan), ModelValidity::Sampled(9_002_000));
    let (mut producer, pipe) = pipe_for::<Tape>(1);
    let mut device = TapeDevice::new(pipe, Arc::new(PublishedCheckpoint::new(1)), 1);
    let mut source =
        Preparation::with_window_frames(vec![(1, source_pcm(&(1..=510).collect::<Vec<_>>()))], 512);
    assert!(producer
        .publish(prepare_tape(&mut source, 0, 0, 510))
        .is_ok());
    let mut cache = ValidatedPlanReader::default();
    let mut prefix = [0; 499];
    assert_eq!(
        device.render_validated_plan(&mut prefix, true, 14_002_000, &plans, &mut cache),
        0
    );
    assert_eq!(prefix, std::array::from_fn(|i| i as i32 + 1));
    assert_eq!(device.drop_counter, 1);
    let (opened_tx, opened_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let publisher = Arc::clone(&plans);
    let worker = thread::spawn(move || {
        publisher.publish_after_open(
            0,
            Some(insert_plan),
            ModelValidity::Sampled(14_002_001),
            0,
            || {
                opened_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
            },
        )
    });
    opened_rx.recv().unwrap();
    let mut output = [0];
    let missing = device.render_validated_plan(&mut output, true, 14_002_001, &plans, &mut cache);
    // Release/join before assertions so an assertion failure cannot strand it.
    resume_tx.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(missing, 0);
    assert_eq!(output, [500], "expired drop must not discard sample 500");
    assert_eq!(device.schedule, CorrectionSchedule::default());
    plans.publish_acknowledged(
        0,
        Some(insert_plan),
        ModelValidity::Sampled(14_002_001),
        cache.reset_epoch(),
    );
    assert_eq!(
        device.render_validated_plan(&mut output, true, 14_002_001, &plans, &mut cache),
        0
    );
    assert_eq!(output, [501]);
    assert_eq!(device.schedule, insert_plan);
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 501);
    assert_eq!(
        device.render_validated_plan(&mut output, false, 14_002_000, &plans, &mut cache),
        0
    );
    assert_eq!(output, [502]);
    assert_eq!(
        device.schedule,
        CorrectionSchedule::default(),
        "backwards endpoint invalidates even on fallback"
    );
    assert_eq!(
        device.render_validated_plan(&mut output, true, 14_002_001, &plans, &mut cache),
        0
    );
    assert_eq!(output, [503]);
    assert_eq!(
        device.schedule,
        CorrectionSchedule::default(),
        "invalidated publication cannot revive without a new plan"
    );
    plans.publish_acknowledged(
        0,
        Some(insert_plan),
        ModelValidity::Sampled(14_002_001),
        cache.reset_epoch(),
    );
    assert_eq!(
        device.render_validated_plan(&mut output, true, 14_002_001, &plans, &mut cache),
        0
    );
    assert_eq!(output, [504]);
    assert_eq!(
        device.schedule, insert_plan,
        "a new complete publication can restore correction"
    );
}

#[test]
fn realtime_handoff_probe_tape_fallback_cached_default_cannot_hide_actual_plan_reset() {
    use super::feedback::{ModelValidity, ValidatedPlanMailbox, ValidatedPlanReader};
    let plans = ValidatedPlanMailbox::default();
    let mut cache = ValidatedPlanReader::default();
    let drop_plan = CorrectionPlanner::new().plan(10_000, 1_000, false);
    plans.publish(0, Some(drop_plan), ModelValidity::Sampled(0));
    let (mut producer, pipe) = pipe_for::<Tape>(1);
    let mut device = TapeDevice::new(pipe, Arc::new(PublishedCheckpoint::new(1)), 1);
    let mut source = Preparation::new(vec![(1, source_pcm(&[1, 2, 3, 4]))]);
    assert!(producer.publish(prepare_tape(&mut source, 0, 0, 4)).is_ok());
    let mut output = [0];
    assert_eq!(
        device.render_validated_plan(&mut output, true, 0, &plans, &mut cache),
        0
    );
    assert_eq!(output, [1]);
    plans.publish(
        0,
        Some(CorrectionSchedule::default()),
        ModelValidity::Sampled(0),
    );
    assert_eq!(
        device.render_validated_plan(&mut output, false, 1, &plans, &mut cache),
        0
    );
    assert_eq!(output, [2]);
    assert_eq!(device.schedule, drop_plan);
    assert_eq!(
        device.render_validated_plan(&mut output, false, 5_000_001, &plans, &mut cache),
        0
    );
    assert_eq!(output, [3]);
    assert_eq!(device.schedule, CorrectionSchedule::default());
    assert_eq!(
        cache.reset_epoch(),
        1,
        "the applied drop was cleared despite a cached default"
    );
    plans.publish(0, Some(drop_plan), ModelValidity::Sampled(5_000_001));
    assert_eq!(
        device.render_validated_plan(&mut output, true, 5_000_001, &plans, &mut cache),
        0
    );
    assert_eq!(output, [4]);
    assert_eq!(device.schedule, CorrectionSchedule::default());
}

#[test]
fn realtime_handoff_probe_tape_validated_feedback_retries_reset_after_full_channel() {
    use super::feedback::{
        ModelValidity, Observation, ValidatedFeedback, ValidatedObservation, ValidatedPlanMailbox,
        ValidatedPlanReader,
    };
    use crate::sync::{Clock, ClockSync};
    use std::time::{Duration, Instant};
    struct FixedClock(Instant);
    impl Clock for FixedClock {
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
    let origin = Instant::now();
    let plans = Arc::new(ValidatedPlanMailbox::default());
    let (mut tx, mut rx) = rtrb::RingBuffer::new(2);
    let (ready_tx, ready_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let worker_plans = Arc::clone(&plans);
    let worker = thread::spawn(move || {
        let sync = ClockSync::new_same_clock(Arc::new(FixedClock(origin)));
        let mut feedback = ValidatedFeedback::new();
        // Warm the real filter. Carrier expiry is injected below; a same-clock
        // model does not naturally expire. This isolates transport/reset order.
        for serial in 1..=101 {
            let _ = feedback.process(
                &sync,
                ValidatedObservation {
                    latency_floor: None,
                    input: Observation {
                        settled_seen: true,
                        timeline: 0,
                        serial,
                        source_cursor_us: 0,
                        presentation: Some(origin + Duration::from_micros(10_000)),
                        actual_schedule: CorrectionSchedule::default(),
                    },
                    reset_epoch: 0,
                },
                0,
                0,
                &worker_plans,
            );
        }
        ready_tx.send(()).unwrap();
        let mut seen = Vec::new();
        for count in [2, 1] {
            resume_rx.recv().unwrap();
            for _ in 0..count {
                let observation: ValidatedObservation = rx.pop().unwrap();
                seen.push((
                    observation.input.serial,
                    observation.reset_epoch,
                    observation.input.source_cursor_us,
                    observation.input.actual_schedule,
                ));
                let _ = feedback.process(&sync, observation, 0, 0, &worker_plans);
            }
            done_tx.send(()).unwrap();
        }
        seen
    });
    ready_rx.recv().unwrap();
    let (mut producer, pipe_reader) = pipe_for::<Tape>(1);
    let mut device = TapeDevice::new(pipe_reader, Arc::new(PublishedCheckpoint::new(1)), 1);
    device.feedback_serial = 101;
    let mut source = Preparation::new(vec![(1, source_pcm(&[1, 2, 3, 4, 5]))]);
    assert!(producer.publish(prepare_tape(&mut source, 0, 0, 5)).is_ok());
    let mut reader = ValidatedPlanReader::default();
    let drop_plan = CorrectionPlanner::new().plan(10_000, 1_000, false);
    plans.publish(0, Some(drop_plan), ModelValidity::Sampled(0));
    let mut output = [-1];
    for sample in 1..=2 {
        assert_eq!(
            device.render_validated_feedback(
                &mut output,
                origin + Duration::from_micros((sample - 1) * 1000),
                Some(origin + Duration::from_micros(10_000 + (sample - 1) * 1000)),
                0,
                &plans,
                &mut reader,
                &mut tx
            ),
            (0, true)
        );
        assert_eq!(output, [sample as i32]);
    }
    assert_eq!(device.schedule, drop_plan);
    // Expiry occurs while the worker is parked and both slots contain epoch 0.
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            origin + Duration::from_millis(3),
            None,
            5_000_001,
            &plans,
            &mut reader,
            &mut tx
        ),
        (0, false)
    );
    assert_eq!(output, [3]);
    assert_eq!(reader.reset_epoch(), 1);
    assert_eq!(device.schedule, CorrectionSchedule::default());
    resume_tx.send(()).unwrap();
    done_rx.recv().unwrap();
    // The worker publishes fresh same-clock results from the queued old epoch.
    // They must not restore correction, and the next fallback retries reset 1.
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            origin + Duration::from_millis(3),
            None,
            5_000_001,
            &plans,
            &mut reader,
            &mut tx
        ),
        (0, true)
    );
    assert_eq!(output, [4]);
    assert_eq!(device.schedule, CorrectionSchedule::default());
    resume_tx.send(()).unwrap();
    done_rx.recv().unwrap();
    let seen = worker.join().unwrap();
    assert_eq!(
        seen,
        vec![
            (102, 0, 0, drop_plan),
            (103, 0, 1000, drop_plan),
            (105, 1, 3000, CorrectionSchedule::default()),
        ]
    );
    assert_eq!(
        reader.read(&plans, 0, 5_000_001, device.schedule),
        Some(Some(CorrectionSchedule::default())),
        "worker acknowledged actual reset"
    );
    assert_eq!(
        device.render_validated_plan(&mut output, true, 5_000_001, &plans, &mut reader),
        0
    );
    assert_eq!(output, [5]);
    assert_eq!(device.pipe.progress.load(Ordering::Acquire), 5);
}

#[test]
fn realtime_handoff_probe_tape_delayed_reanchor_needs_time_not_consumption_delta() {
    use super::super::admission::SourceLedger;
    use super::feedback::{ValidatedFeedback, ValidatedPlanMailbox, ValidatedPlanReader};
    use crate::audio::EnqueueOutcome;
    use crate::sync::{Clock, ClockSync};
    use std::sync::atomic::AtomicI64;
    use std::time::{Duration, Instant};
    struct FixedClock(Instant, AtomicI64);
    impl Clock for FixedClock {
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
        fn instant_to_micros(&self, instant: Instant) -> i64 {
            if instant >= self.0 {
                instant.duration_since(self.0).as_micros() as i64
            } else {
                -(self.0.duration_since(instant).as_micros() as i64)
            }
        }
    }
    for preparation_delay_us in [0, 20_000, 600_000] {
        let capture = Instant::now();
        let clock = Arc::new(FixedClock(capture, AtomicI64::new(0)));
        let sync = ClockSync::new_same_clock(clock.clone());
        let mut feedback = ValidatedFeedback::new();
        let plans = ValidatedPlanMailbox::default();
        let mut cache = ValidatedPlanReader::default();
        let (mut tx, mut rx) = rtrb::RingBuffer::new(1);
        // A previously observed delta floor of zero differs from this wake's 600ms.
        // This is the production target formula, not cursor + filtered error.
        let target_at = |instant| {
            sync.client_to_server_micros(sync.instant_to_client_micros(instant))
                .unwrap()
        };
        assert_eq!(target_at(capture), 0);
        let later_boundary = capture + Duration::from_millis(20);
        assert_eq!(target_at(later_boundary), 20_000);
        let checkpoint = Arc::new(PublishedCheckpoint::new(1));
        let (pipe, reader) = pipe_for::<Tape>(2);
        let mut ledger =
            SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
        let mut device = TapeDevice::new(reader, checkpoint, 1);
        let mut producer = TapePreparation {
            source: None,
            revision: None,
            pending: None,
            pipe,
            epoch: 0,
            source_base: 0,
        };
        for (timestamp, pcm) in [
            (0, [1, 2, 3, 4]),
            (10_000, [10, 11, 12, 13]),
            (20_000, [20, 21, 22, 23]),
        ] {
            let mut source = source_pcm(&pcm);
            source.timestamp = timestamp;
            assert!(matches!(
                ledger.try_enqueue(source, &device.pipe.gate, || {}),
                Ok(EnqueueOutcome::Accepted { .. })
            ));
        }
        assert_eq!(
            producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 16),
            Ok(true)
        );
        let mut sample = [0];
        assert_eq!(
            device.render_validated_feedback(
                &mut sample,
                capture,
                Some(capture + Duration::from_millis(1)),
                0,
                &plans,
                &mut cache,
                &mut tx
            ),
            (0, true)
        );
        assert_eq!(sample, [1]);
        // The worker does not receive this earlier sample, but the device floor
        // must survive and accompany its next measurement.
        let _ = rx.pop().unwrap();
        assert_eq!(
            device.render_validated_feedback(
                &mut sample,
                capture,
                Some(capture + Duration::from_millis(600)),
                0,
                &plans,
                &mut cache,
                &mut tx
            ),
            (0, true)
        );
        assert_eq!(sample, [2]);
        let request = feedback
            .process(&sync, rx.pop().unwrap(), 2_000, 0, &plans)
            .expect("real callback observation must reach non-RT reanchor control");
        assert_eq!(request.latency_floor, Some(Duration::from_millis(1)));
        // The measured control delay is 20ms, but only two source frames advanced.
        // Neither the old absolute target nor source-count extrapolation is current.
        let consumed = device.pipe.progress.load(Ordering::Acquire);
        assert_eq!(consumed, 2);
        assert_ne!(
            target_at(capture) + consumed as i64 * 1000,
            target_at(later_boundary)
        );
        let gate = Arc::clone(&device.pipe.gate);
        let original_epoch = gate.view().epoch();
        clock.1.store(7_000, Ordering::Release);
        let compute_target = || {
            // The paired device latency floor is 1ms and configured delay is 2ms.
            let client = sync
                .clock()
                .now_micros()
                .checked_add(i64::try_from(request.latency_floor.unwrap().as_micros()).unwrap())?
                .checked_add(2_000)?;
            sync.client_to_server_micros(client)
        };
        assert_eq!(
            ledger.try_reanchor_with(&gate, || {
                let target = compute_target();
                assert_eq!(target, Some(10_000));
                // A callback advances after reconciliation, before the control CAS.
                let mut advanced = [0];
                assert_eq!(
                    device.render(&mut advanced, false, CorrectionSchedule::default()),
                    0
                );
                assert_eq!(advanced, [3]);
                target
            }),
            Err(())
        );
        assert_eq!(gate.view().epoch(), original_epoch);
        assert_eq!(
            ledger.preparation(&gate).unwrap().1.cursor.cursor_us,
            3_000,
            "failed control did not commit its target"
        );
        clock.1.store(17_000, Ordering::Release);
        assert_eq!(ledger.try_reanchor_with(&gate, compute_target), Ok(20_000));
        let timeline = gate.view().epoch();
        feedback.reanchor_committed(timeline, &sync, &plans);
        assert!(
            feedback
                .process(&sync, request, 2_000, timeline, &plans)
                .is_none(),
            "old triggering observation cannot issue another command on the new timeline"
        );
        clock
            .1
            .store(17_000 + preparation_delay_us, Ordering::Release);

        assert_eq!(
            producer.pump_from_ledger(&mut ledger, &device.pipe.gate, 16),
            Ok(true)
        );
        let mut tail = [0; 4];
        let resumed_capture =
            capture + Duration::from_micros((17_000 + preparation_delay_us) as u64);
        assert_eq!(
            device.render_validated_feedback(
                &mut tail,
                resumed_capture,
                Some(resumed_capture + Duration::from_millis(1)),
                clock.now_micros(),
                &plans,
                &mut cache,
                &mut tx
            ),
            (0, true)
        );
        let observed = rx.pop().unwrap();
        assert_eq!(observed.input.source_cursor_us, 20_000);
        let expected = sync
            .server_to_local_instant_with_latency(20_000, 2_000)
            .unwrap();
        assert_eq!(
            observed
                .input
                .presentation
                .unwrap()
                .duration_since(expected)
                .as_micros(),
            preparation_delay_us as u128
        );
        let next_request = feedback.process(&sync, observed, 2_000, timeline, &plans);
        assert_eq!(
            next_request.is_some(),
            preparation_delay_us >= 500_000,
            "post-commit preparation delay remains a real error, not hidden by consumption"
        );
        assert_eq!(tail, [4, 21, 22, 23]);
        ledger.try_reconcile(&device.pipe.gate).unwrap();
        assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(7));
        assert_eq!(
            ledger
                .owner
                .capacity(ledger.scope)
                .unwrap()
                .current_frames(),
            0
        );
    }
}

#[test]
fn realtime_handoff_probe_tape_latency_floor_survives_full_feedback_and_reanchor_but_not_clear() {
    use super::super::admission::SourceLedger;
    use super::feedback::{ValidatedPlanMailbox, ValidatedPlanReader};
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome};
    use std::time::{Duration, Instant};
    let checkpoint = Arc::new(PublishedCheckpoint::new(1));
    let (pipe, reader) = pipe_for::<Tape>(3);
    let mut ledger =
        SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
    let mut device = TapeDevice::new(reader, checkpoint, 1);
    let gate = Arc::clone(&device.pipe.gate);
    let mut producer = TapePreparation {
        source: None,
        revision: None,
        pending: None,
        pipe,
        epoch: 0,
        source_base: 0,
    };
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[1, 2, 3, 4, 5]), &gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    assert_eq!(producer.pump_from_ledger(&mut ledger, &gate, 16), Ok(true));
    let plans = ValidatedPlanMailbox::default();
    let mut cache = ValidatedPlanReader::default();
    let (mut tx, mut rx) = rtrb::RingBuffer::new(1);
    let capture = Instant::now();
    let mut output = [-1];
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            capture,
            Some(capture + Duration::from_millis(10)),
            0,
            &plans,
            &mut cache,
            &mut tx
        ),
        (0, true)
    );
    assert_eq!(output, [1]);
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            capture,
            Some(capture + Duration::from_millis(2)),
            0,
            &plans,
            &mut cache,
            &mut tx
        ),
        (0, false)
    );
    assert_eq!(output, [2]);
    assert_eq!(
        rx.pop().unwrap().latency_floor,
        Some(Duration::from_millis(10))
    );
    // The best measured delta was dropped from transport, but the callback
    // retains it; fallback publishes the floor without fabricating a sample.
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            capture,
            None,
            0,
            &plans,
            &mut cache,
            &mut tx
        ),
        (0, true)
    );
    assert_eq!(output, [3]);
    let fallback = rx.pop().unwrap();
    assert!(fallback.input.presentation.is_none());
    assert_eq!(fallback.latency_floor, Some(Duration::from_millis(2)));
    assert_eq!(ledger.try_reanchor(&gate, 3_000), Ok(()));
    assert_eq!(producer.pump_from_ledger(&mut ledger, &gate, 16), Ok(true));
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            capture,
            None,
            0,
            &plans,
            &mut cache,
            &mut tx
        ),
        (0, true)
    );
    assert_eq!(output, [4]);
    assert_eq!(
        rx.pop().unwrap().latency_floor,
        Some(Duration::from_millis(2))
    );
    assert_eq!(
        ledger.try_clear(&gate),
        Ok(RendererOperationOutcome::Applied)
    );
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[9]), &gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    assert_eq!(producer.pump_from_ledger(&mut ledger, &gate, 16), Ok(true));
    assert_eq!(
        device.render_validated_feedback(
            &mut output,
            capture,
            None,
            0,
            &plans,
            &mut cache,
            &mut tx
        ),
        (0, true)
    );
    assert_eq!(output, [9]);
    assert_eq!(rx.pop().unwrap().latency_floor, None);
}

#[test]
fn realtime_handoff_probe_tape_feedback_compares_production_after_started_offset() {
    use super::super::admission::SourceLedger;
    use super::feedback::{ValidatedFeedback, ValidatedPlanMailbox, ValidatedPlanReader};
    use crate::audio::EnqueueOutcome;
    use crate::sync::{Clock, ClockSync};
    use std::sync::atomic::AtomicI64;
    use std::time::{Duration, Instant};
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
        fn instant_to_micros(&self, instant: Instant) -> i64 {
            if instant >= self.0 {
                instant.duration_since(self.0).as_micros() as i64
            } else {
                -(self.0.duration_since(instant).as_micros() as i64)
            }
        }
    }
    for initial_delay_us in [20_000, 600_000] {
        let mut baseline = crate::audio::synced_player::callback_tests::Harness::streaming_mono();
        let origin = Instant::now();
        let clock = Arc::new(ControlledClock(origin, AtomicI64::new(initial_delay_us)));
        let sync = ClockSync::new_same_clock(clock.clone());
        let checkpoint = Arc::new(PublishedCheckpoint::new(1));
        let (pipe, reader) = pipe_for::<Tape>(4);
        let mut ledger =
            SourceLedger::with_checkpoint(Arc::clone(&reader.progress), Arc::clone(&checkpoint));
        let mut device = TapeDevice::new(reader, checkpoint, 1);
        let gate = Arc::clone(&device.pipe.gate);
        let mut producer = TapePreparation {
            source: None,
            revision: None,
            pending: None,
            pipe,
            epoch: 0,
            source_base: 0,
        };
        let plans = ValidatedPlanMailbox::default();
        let mut cache = ValidatedPlanReader::default();
        let mut worker = ValidatedFeedback::new();
        let (mut tx, mut rx) = rtrb::RingBuffer::new(1);
        let mut timeline = 0;
        let mut source_frame = 0i64;
        let mut last_sample = 0;
        let mut missing_total = 0;
        let mut reanchors = 0;
        let mut final_error = None;
        let mut first_deadband_tick = None;
        let mut baseline_deadband_tick = None;
        let mut baseline_final = None;
        let mut differing_samples = 0;
        let mut baseline_last_sample = 0;
        // Deterministic 1kHz PCM, ten frames per callback. A 30-second
        // observation initially exposed a large residual after source catch-up.
        // Observe 120 simulated seconds: the existing 0.2% speed cap needs
        // at least 67.5 further seconds to remove that 135ms residual.
        // This horizon is not a product recovery SLA.
        for tick in 0..12200 {
            let now = tick * 10_000 + if tick >= 200 { initial_delay_us } else { 0 };
            clock.1.store(now, Ordering::Release);
            ledger.try_reconcile(&gate).unwrap();
            for _ in 0..2 {
                if ledger
                    .owner
                    .capacity(ledger.scope)
                    .unwrap()
                    .current_frames()
                    > 16
                    || baseline.streaming_queued_frames() > 16
                {
                    break;
                }
                let samples: Vec<i32> = (source_frame + 1..=source_frame + 16)
                    .map(|x| x as i32)
                    .collect();
                let mut buffer = source_pcm(&samples);
                buffer.timestamp = source_frame * 1000;
                assert!(matches!(
                    ledger.try_enqueue(buffer, &gate, || {}),
                    Ok(EnqueueOutcome::Accepted { .. })
                ));
                baseline.streaming_enqueue(source_frame);
                source_frame += 16;
            }
            producer.pipe.reclaim();
            for _ in 0..4 {
                if !producer.pump_from_ledger(&mut ledger, &gate, 16).unwrap() {
                    break;
                }
            }
            if tick == 0 {
                let (startup_pcm, startup_diagnostics) = baseline.streaming_render(0);
                assert!(startup_pcm.iter().all(|sample| *sample == 0));
                assert_eq!(startup_diagnostics.startup_reanchors, 1);
                assert_eq!(baseline.streaming_cursor_us(), 10_000);
                // Normalize to the source position selected by the real startup
                // handoff. This test compares post-start correction, not startup.
                ledger
                    .try_reanchor(&gate, baseline.streaming_cursor_us())
                    .unwrap();
                timeline = gate.view().epoch();
                worker.reanchor_committed(timeline, &sync, &plans);
                continue;
            }
            let captured = origin + Duration::from_micros(now as u64);
            let mut output = [0; 10];
            let (missing, published) = device.render_validated_feedback(
                &mut output,
                captured,
                Some(captured),
                now,
                &plans,
                &mut cache,
                &mut tx,
            );
            missing_total += missing;
            let (baseline_pcm, baseline_diagnostics) = baseline.streaming_render(now as u64);
            if tick < 200 {
                assert_eq!(
                    output.as_slice(),
                    baseline_pcm.as_slice(),
                    "both paths start normally before the offset"
                );
            } else {
                differing_samples += output
                    .iter()
                    .zip(&baseline_pcm)
                    .filter(|(a, b)| a != b)
                    .count();
                if baseline_diagnostics
                    .raw_error_us
                    .is_some_and(|e| e.abs() <= 1500)
                    && baseline_deadband_tick.is_none()
                {
                    baseline_deadband_tick = Some(tick - 200);
                }
            }
            for sample in baseline_pcm.into_iter().filter(|x| *x != 0) {
                assert!(sample >= baseline_last_sample);
                baseline_last_sample = sample;
            }
            baseline_final = Some(baseline_diagnostics);
            for sample in output.into_iter().filter(|x| *x != 0) {
                assert!(
                    sample >= last_sample,
                    "no replay of earlier source positions"
                );
                last_sample = sample;
            }
            if published {
                let observation = rx.pop().unwrap();
                let error = now - observation.input.source_cursor_us;
                final_error = Some(error);
                if tick >= 200 && error.abs() <= 1500 && first_deadband_tick.is_none() {
                    first_deadband_tick = Some(tick - 200);
                }
                if let Some(request) = worker.process(&sync, observation, 0, timeline, &plans) {
                    let target = ledger
                        .try_reanchor_with(&gate, || {
                            let floor = i64::try_from(request.latency_floor?.as_micros()).ok()?;
                            sync.client_to_server_micros(
                                sync.clock().now_micros().checked_add(floor)?,
                            )
                        })
                        .unwrap();
                    assert_eq!(target, now);
                    timeline = gate.view().epoch();
                    worker.reanchor_committed(timeline, &sync, &plans);
                    reanchors += 1;
                }
            }
        }
        assert!(
            final_error.unwrap().abs() <= 1500,
            "delay={initial_delay_us}, error={final_error:?}"
        );
        assert!(!device.schedule.is_correcting());
        assert!(first_deadband_tick.is_some());
        assert_eq!(reanchors, if initial_delay_us == 600_000 { 1 } else { 0 });
        ledger.try_reconcile(&gate).unwrap();
        assert_eq!(
            ledger.owner.consumed_frames(ledger.scope).unwrap(),
            device.pipe.progress.load(Ordering::Acquire)
        );
        let baseline_final = baseline_final.unwrap();
        assert!(baseline_final.raw_error_us.unwrap().abs() <= 1500);
        assert!(baseline_deadband_tick.is_some());
        eprintln!("baseline delay_us={initial_delay_us} first_deadband_ms={} final_error_us={} underrun_frames={} reanchors={} differing_samples={differing_samples}", baseline_deadband_tick.unwrap()*10, baseline_final.raw_error_us.unwrap(), baseline_final.underrun_frames, baseline_final.correction_reanchors);
        eprintln!("delay_us={initial_delay_us} first_deadband_ms={} final_error_us={} missing_frames={missing_total} reanchors={reanchors}", first_deadband_tick.unwrap()*10, final_error.unwrap());
    }
}

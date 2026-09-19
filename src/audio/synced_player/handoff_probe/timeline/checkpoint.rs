//! Fixed-size device publication for the cadence and last-frame experiment.
//! The existing gate validates coherent non-RT reads; no callback retry or lock.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicUsize};

#[derive(Clone, Copy)]
pub(crate) struct Cadence {
    pub schedule: CorrectionSchedule,
    pub insert_counter: u32,
    pub drop_counter: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SourcePosition {
    pub retired_through: u64,
    pub current: u64,
    pub index: usize,
    pub cursor_us: i64,
    pub cursor_remainder: i64,
}

pub(crate) struct Snapshot<'a> {
    pub view: Observation<'a>,
    pub cadence: Cadence,
    pub last: Vec<i32>,
    pub source_frames: u64,
    pub output_frames: u64,
    pub position: SourcePosition,
}

pub(crate) struct PublishedCheckpoint {
    valid: AtomicBool,
    output_frames: AtomicU64,
    retired_through: AtomicU64,
    index: AtomicUsize,
    cursor_us: AtomicI64,
    cursor_remainder: AtomicI64,
    schedule: AtomicU64,
    counters: AtomicU64,
    last: Box<[AtomicI32]>,
}

impl PublishedCheckpoint {
    pub fn new(channels: usize) -> Self {
        Self {
            valid: AtomicBool::new(false),
            output_frames: AtomicU64::new(0),
            retired_through: AtomicU64::new(0),
            index: AtomicUsize::new(0),
            cursor_us: AtomicI64::new(0),
            cursor_remainder: AtomicI64::new(0),
            schedule: AtomicU64::new(0),
            counters: AtomicU64::new(0),
            last: (0..channels).map(|_| AtomicI32::new(0)).collect(),
        }
    }

    pub fn publish(
        &self,
        cadence: Cadence,
        position: SourcePosition,
        visited: usize,
        last: Option<&[i32]>,
    ) {
        self.output_frames
            .fetch_add(visited as u64, Ordering::Relaxed);
        self.retired_through
            .store(u64::from(position.retired_through), Ordering::Relaxed);
        self.index.store(position.index, Ordering::Relaxed);
        self.cursor_us.store(position.cursor_us, Ordering::Relaxed);
        self.cursor_remainder
            .store(position.cursor_remainder, Ordering::Relaxed);
        self.schedule.store(
            (u64::from(cadence.schedule.insert_every_n_frames) << 32)
                | u64::from(cadence.schedule.drop_every_n_frames),
            Ordering::Relaxed,
        );
        self.counters.store(
            (u64::from(cadence.insert_counter) << 32) | u64::from(cadence.drop_counter),
            Ordering::Relaxed,
        );
        if let Some(last) = last {
            assert_eq!(last.len(), self.last.len());
            for (target, sample) in self.last.iter().zip(last) {
                target.store(*sample, Ordering::Relaxed);
            }
        }
        self.valid.store(true, Ordering::Release);
    }

    pub fn output_frames(&self) -> u64 {
        self.output_frames.load(Ordering::Relaxed)
    }

    pub fn source_index(&self) -> usize {
        self.index.load(Ordering::Relaxed)
    }

    pub fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }

    pub fn read<'a>(&self, gate: &'a ClaimGate, progress: &AtomicU64) -> Option<Snapshot<'a>> {
        let view = gate.observe()?;
        self.read_observed(view, progress)
    }

    pub fn read_observed<'a>(
        &self,
        view: Observation<'a>,
        progress: &AtomicU64,
    ) -> Option<Snapshot<'a>> {
        let gate = view.gate;
        if view.callback_active() || !self.valid.load(Ordering::Acquire) {
            return None;
        }
        let schedule = self.schedule.load(Ordering::Relaxed);
        let counters = self.counters.load(Ordering::Relaxed);
        let current = view.current();
        let snapshot = Snapshot {
            view,
            cadence: Cadence {
                schedule: CorrectionSchedule {
                    insert_every_n_frames: (schedule >> 32) as u32,
                    drop_every_n_frames: schedule as u32,
                    reanchor: false,
                },
                insert_counter: (counters >> 32) as u32,
                drop_counter: counters as u32,
            },
            last: self
                .last
                .iter()
                .map(|sample| sample.load(Ordering::Relaxed))
                .collect(),
            source_frames: progress.load(Ordering::Acquire),
            output_frames: self.output_frames(),
            position: SourcePosition {
                retired_through: self.retired_through.load(Ordering::Relaxed) as u64,
                current,
                index: self.index.load(Ordering::Relaxed),
                cursor_us: self.cursor_us.load(Ordering::Relaxed),
                cursor_remainder: self.cursor_remainder.load(Ordering::Relaxed),
            },
        };
        (self.valid.load(Ordering::Acquire) && gate.validate(&snapshot.view)).then_some(snapshot)
    }

    pub fn invalidate_for_reanchor(&self) {
        self.valid.store(false, Ordering::Release);
    }

    // Called by the serialized source owner after clear revokes the old output
    // and before new source publication. Old windows cannot publish again.
    pub fn invalidate_after_clear(&self) {
        self.valid.store(false, Ordering::Release);
        self.output_frames.store(0, Ordering::Relaxed);
        self.index.store(0, Ordering::Relaxed);
        for sample in &self.last {
            sample.store(0, Ordering::Relaxed);
        }
    }
}

#[test]
fn realtime_handoff_probe_checkpoint_clear_invalidates_phase_without_resetting_consumption() {
    use super::super::admission::SourceLedger;
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome};
    let publication = Arc::new(PublishedCheckpoint::new(1));
    let (mut producer, mut device) = pipe_for::<Span>(2);
    let mut ledger = SourceLedger::new(Arc::clone(&device.progress));
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[10, 20, 30]), &device.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    let schedule = CorrectionPlanner::new().plan(-10_000, 1_000, false);
    let mut old = Preparation::new(vec![(1, source_pcm(&[10, 20, 30]))]);
    old.checkpoint = Some(Arc::clone(&publication));
    assert!(producer
        .publish(old.prepare_scheduled(0, 3, schedule, true))
        .is_ok());
    let mut prefix = [0; 2];
    assert_eq!(device.render_span(&mut prefix, 1), 0);
    assert_eq!(prefix, [10, 20]);
    assert_eq!(
        publication
            .read(&device.gate, &device.progress)
            .unwrap()
            .cadence
            .insert_counter,
        498
    );
    assert_eq!(
        ledger.try_clear_with(&device.gate, || publication.invalidate_after_clear()),
        Ok(RendererOperationOutcome::Applied)
    );
    assert!(publication.read(&device.gate, &device.progress).is_none());
    assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(2));
    assert!(matches!(
        ledger.try_enqueue(source_pcm(&[50, 60]), &device.gate, || {}),
        Ok(EnqueueOutcome::Accepted { .. })
    ));
    let mut fresh = Preparation::new(vec![(2, source_pcm(&[50, 60]))]);
    fresh.checkpoint = Some(Arc::clone(&publication));
    assert!(producer
        .publish(fresh.prepare_scheduled(device.gate.view().epoch(), 1, schedule, true))
        .is_ok());
    let mut output = [0];
    assert_eq!(device.render_span(&mut output, 1), 0);
    assert_eq!(output, [50]);
    let actual = publication.read(&device.gate, &device.progress).unwrap();
    assert_eq!(actual.source_frames, 3);
    assert_eq!(actual.last, [50]);
    assert_eq!(actual.cadence.insert_counter, 499);
    ledger.reconcile();
    assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(3));
    assert_eq!(producer.reclaim(), 2);
}

#[test]
fn realtime_handoff_probe_checkpoint_restores_insert_phase_from_actual_output() {
    for output_frames in [499, 500] {
        let samples: Vec<i32> = (1..=1_002).collect();
        let publication = Arc::new(PublishedCheckpoint::new(1));
        let mut preparation = Preparation::with_window_frames(vec![(1, source_pcm(&samples))], 512);
        preparation.checkpoint = Some(Arc::clone(&publication));
        let schedule = CorrectionPlanner::new().plan(-10_000, 1_000, false);
        let span = preparation.prepare_scheduled(0, output_frames, schedule, true);
        // Speculation goes beyond the published window and must not become truth.
        let abandoned = preparation.prepare_scheduled(0, 20, schedule, true);
        let (mut producer, mut device) = pipe_for::<Span>(1);
        assert!(producer.publish(span).is_ok());
        assert_eq!(device.render_span(&mut vec![0; output_frames], 1), 0);
        let observation_gate = Arc::clone(&device.gate);
        let actual = publication
            .read(&observation_gate, &device.progress)
            .unwrap();
        assert_eq!(actual.source_frames, 499);
        assert_eq!(actual.last, [499]);
        assert_eq!(producer.reclaim(), 1);
        drop(abandoned);

        // Restore the retained source view directly from the actual position.
        let mut restored = Preparation::with_window_frames(vec![(1, source_pcm(&samples))], 512);
        restored.apply_actual(&actual);
        assert!(device.gate.replace(actual.view).is_ok());
        restored.checkpoint = Some(Arc::clone(&publication));
        let next = restored.prepare_scheduled(device.gate.view().epoch(), 2, schedule, true);
        assert!(producer.publish(next).is_ok());
        let mut output = [0; 2];
        assert_eq!(device.render_span(&mut output, 1), 0);
        assert_eq!(
            output,
            if output_frames == 499 {
                [499, 500]
            } else {
                [500, 501]
            }
        );
        assert_eq!(producer.reclaim(), 1);
    }
}

#[test]
fn realtime_handoff_probe_checkpoint_reanchor_reconciles_skipped_sources() {
    let sources = || {
        [
            (1, 0, vec![1, 2, 3, 4]),
            (2, 10_000, vec![10, 11, 12, 13]),
            (3, 20_000, vec![20, 21, 22, 23]),
        ]
        .into_iter()
        .map(|(id, timestamp, samples)| {
            let mut source = source_pcm(&samples);
            source.timestamp = timestamp;
            (id, source)
        })
        .collect()
    };
    let mut preparation = Preparation::new(sources());
    let mut retained = Preparation::new(sources());
    let publication = Arc::new(PublishedCheckpoint::new(1));
    preparation.checkpoint = Some(Arc::clone(&publication));
    let (mut producer, mut device) = pipe_for::<Span>(2);
    let schedule = CorrectionSchedule::default();
    assert!(producer
        .publish(preparation.prepare_scheduled(0, 1, schedule, true))
        .is_ok());
    let mut first = [0];
    assert_eq!(device.render_span(&mut first, 1), 0);
    assert_eq!(first, [1]);
    retained.reconcile_position(
        publication
            .read(&device.gate, &device.progress)
            .unwrap()
            .position,
    );
    assert_eq!(retained.cursor.index, 1);
    assert_eq!(producer.reclaim(), 1);

    // Same cursor-only operation as the production reanchor; current survives.
    preparation.cursor.cursor_us = 20_000;
    preparation.cursor.cursor_remainder = 0;
    assert!(producer
        .publish(preparation.prepare_scheduled(0, 4, schedule, true))
        .is_ok());
    let mut prefix = [0; 2];
    assert_eq!(device.render_span(&mut prefix, 1), 0);
    assert_eq!(prefix, [2, 3]);
    let actual = publication.read(&device.gate, &device.progress).unwrap();
    assert_eq!(actual.source_frames, 3);
    retained.reconcile_position(actual.position);
    assert_eq!(retained.current(), 1);
    assert_eq!(retained.cursor.index, 3);
    assert_eq!(retained.cursor.cursor_us, 22_000);
    drop(actual);

    let mut suffix = [0; 2];
    assert_eq!(device.render_span(&mut suffix, 1), 0);
    assert_eq!(suffix, [4, 23]);
    let actual = publication.read(&device.gate, &device.progress).unwrap();
    assert_eq!(actual.source_frames, 5);
    assert_eq!(actual.position.retired_through, 3);
    retained.reconcile_position(actual.position);
    assert!(retained.cursor.current.is_none());
    assert!(retained.cursor.queue.is_empty());
    assert_eq!(retained.cursor.pending_frames, 0);
    assert_eq!(retained.cursor.cursor_us, 24_000);
    assert_eq!(producer.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_checkpoint_reanchor_releases_actual_admission_capacity() {
    use super::super::admission::SourceLedger;
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome};
    let publication = Arc::new(PublishedCheckpoint::new(1));
    let (mut producer, mut device) = pipe_for::<Span>(2);
    let mut ledger =
        SourceLedger::with_checkpoint(Arc::clone(&device.progress), Arc::clone(&publication));
    let source_at = |timestamp, samples: &[i32]| {
        let mut source = source_pcm(samples);
        source.timestamp = timestamp;
        source
    };
    for (timestamp, samples) in [
        (0, [1, 2, 3, 4]),
        (10_000, [10, 11, 12, 13]),
        (20_000, [20, 21, 22, 23]),
    ] {
        assert!(matches!(
            ledger.try_enqueue(source_at(timestamp, &samples), &device.gate, || {}),
            Ok(EnqueueOutcome::Accepted { .. })
        ));
    }
    let schedule = CorrectionSchedule::default();
    let (view, mut preparation) = ledger.preparation(&device.gate).unwrap();
    assert!(producer
        .publish(preparation.prepare_scheduled(view.epoch(), 1, schedule, true))
        .is_ok());
    let mut first = [0];
    assert_eq!(device.render_span(&mut first, 1), 0);
    assert_eq!(first, [1]);
    assert_eq!(producer.reclaim(), 1);
    // Rebuild from the ledger's actual checkpoint, never replay consumed frames.
    let (view, mut preparation) = ledger.preparation(&device.gate).unwrap();
    preparation.cursor.cursor_us = 20_000;
    preparation.cursor.cursor_remainder = 0;
    assert!(producer
        .publish(preparation.prepare_scheduled(view.epoch(), 4, schedule, true))
        .is_ok());
    let mut prefix = [0; 2];
    assert_eq!(device.render_span(&mut prefix, 1), 0);
    assert_eq!(prefix, [2, 3]);
    // Append while an already prepared window still has a valid suffix.
    assert_eq!(
        ledger
            .try_enqueue(source_at(24_000, &[100; 16]), &device.gate, || {})
            .ok(),
        Some(EnqueueOutcome::Accepted {
            queued_frames: 25,
            queued_buffers: 4
        })
    );
    let mut suffix = [0; 2];
    assert_eq!(device.render_span(&mut suffix, 1), 0);
    assert_eq!(suffix, [4, 23]);
    assert_eq!(producer.reclaim(), 1);
    // The skipped blocks must release capacity even though consumption is only 5.
    assert_eq!(
        ledger
            .try_enqueue(source_at(40_000, &[200; 16]), &device.gate, || {})
            .ok(),
        Some(EnqueueOutcome::Accepted {
            queued_frames: 32,
            queued_buffers: 2
        })
    );
    assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(5));
    assert!(matches!(
        ledger.try_enqueue(source_at(56_000, &[300]), &device.gate, || {}),
        Ok(EnqueueOutcome::Full { .. })
    ));
    let (view, mut preparation) = ledger.preparation(&device.gate).unwrap();
    assert!(producer
        .publish(preparation.prepare_scheduled(view.epoch(), 2, schedule, true))
        .is_ok());
    let mut resumed = [0];
    assert_eq!(device.render_span(&mut resumed, 1), 0);
    assert_eq!(resumed, [100]);
    assert_eq!(
        ledger.try_clear(&device.gate),
        Ok(RendererOperationOutcome::Applied)
    );
    assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(6));
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        0
    );
    assert!(!publication.is_valid());
    assert_eq!(
        ledger
            .try_enqueue(source_at(0, &[9, 8]), &device.gate, || {})
            .ok(),
        Some(EnqueueOutcome::Accepted {
            queued_frames: 2,
            queued_buffers: 1
        })
    );
    let (view, mut preparation) = ledger.preparation(&device.gate).unwrap();
    assert!(producer
        .publish(preparation.prepare_scheduled(view.epoch(), 2, schedule, true))
        .is_ok());
    let mut fresh = [0; 2];
    assert_eq!(device.render_span(&mut fresh, 1), 0);
    assert_eq!(fresh, [9, 8]);
    ledger.try_reconcile(&device.gate).unwrap();
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
}

#[test]
fn realtime_handoff_probe_checkpoint_overlap_rebuild_retains_only_unplayed_output() {
    use super::super::admission::SourceLedger;
    use crate::audio::EnqueueOutcome;
    for old_window_frames in [3, 8] {
        for advance_before_publish in [0, 2] {
            let publication = Arc::new(PublishedCheckpoint::new(1));
            let (mut producer, mut device) = pipe_for::<Span>(2);
            let mut ledger =
                SourceLedger::with_checkpoint(Arc::clone(&device.progress), publication);
            assert!(matches!(
                ledger.try_enqueue(source_pcm(&[1, 2, 3, 4]), &device.gate, || {}),
                Ok(EnqueueOutcome::Accepted { .. })
            ));
            let mut pending = source_pcm(&[10, 11, 12, 13]);
            pending.timestamp = 4_000;
            assert!(matches!(
                ledger.try_enqueue(pending, &device.gate, || {}),
                Ok(EnqueueOutcome::Accepted { .. })
            ));
            let schedule = CorrectionSchedule::default();
            let (view, mut old) = ledger.preparation(&device.gate).unwrap();
            assert!(producer
                .publish(old.prepare_scheduled(view.epoch(), old_window_frames, schedule, true))
                .is_ok());
            let mut prefix = [0];
            assert_eq!(device.render_span(&mut prefix, 1), 0);
            assert_eq!(prefix, [1]);
            let mut replacement = source_pcm(&[50, 51, 52, 53]);
            replacement.timestamp = 4_000;
            assert_eq!(
                ledger.try_enqueue(replacement, &device.gate, || {}).ok(),
                Some(EnqueueOutcome::Accepted {
                    queued_frames: 7,
                    queued_buffers: 2
                })
            );
            // Prepare before the device drains the preserved current. That current is
            // represented in both windows; it must be output once, not lose the suffix.
            let (view, mut fresh) = ledger.preparation(&device.gate).unwrap();
            let rebuilt = fresh.prepare_scheduled(view.epoch(), 7, schedule, true);
            let mut rest = [0; 7];
            // The device may advance while the worker is holding the unpublished view.
            assert_eq!(
                device.render_span(&mut rest[..advance_before_publish], 1),
                0
            );
            assert!(producer.publish(rebuilt).is_ok());
            let missing = device.render_span(&mut rest[advance_before_publish..], 1);
            assert_eq!(rest, [2, 3, 4, 50, 51, 52, 53]);
            assert_eq!(missing, 0);
            ledger.try_reconcile(&device.gate).unwrap();
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
        }
    }
}

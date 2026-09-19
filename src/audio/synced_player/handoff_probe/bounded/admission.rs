//! Non-RT reconciliation experiment using the actual queue and RendererOwner.
//! Serialized callers reconcile actual source progress. Transaction tests use
//! callback boundaries to validate admission; this is not production integration.

use super::timeline::{checkpoint::PublishedCheckpoint, Preparation};
use super::*;
use crate::audio::{
    EnqueueOutcome, PlayerScope, RendererOperationOutcome, RendererOwner, RendererQueueLimits,
};

pub(super) struct SourceLedger {
    source: Preparation,
    pub(super) next_source_id: u64,
    publication: Option<Arc<PublishedCheckpoint>>,
    pub(super) owner: RendererOwner,
    pub(super) scope: PlayerScope,
    progress: Arc<AtomicU64>,
    settled: u64,
    limits: RendererQueueLimits,
}

impl SourceLedger {
    pub(super) fn new(progress: Arc<AtomicU64>) -> Self {
        let limits = RendererQueueLimits::new(32, 8, 16).unwrap();
        let owner = RendererOwner::new(limits);
        let scope = owner.mint_scope().unwrap();
        Self {
            source: Preparation::empty(1, 1_000, 16),
            next_source_id: 1,
            publication: None,
            owner,
            scope,
            progress,
            settled: 0,
            limits,
        }
    }

    pub(super) fn with_checkpoint(
        progress: Arc<AtomicU64>,
        publication: Arc<PublishedCheckpoint>,
    ) -> Self {
        let mut this = Self::new(progress);
        this.source.attach_checkpoint(Arc::clone(&publication));
        this.publication = Some(publication);
        this
    }

    pub(super) fn try_reconcile<'a>(&mut self, gate: &'a ClaimGate) -> Result<Observation<'a>, ()> {
        let view = gate.observe().ok_or(())?;
        if view.callback_active() {
            return Err(());
        }
        let Some(publication) = &self.publication else {
            self.reconcile();
            return Ok(view);
        };
        if publication.is_valid() {
            let actual = publication.read_observed(view, &self.progress).ok_or(())?;
            self.source.apply_actual(&actual);
            let delta =
                usize::try_from(actual.source_frames.checked_sub(self.settled).unwrap()).unwrap();
            let mut permit = self.owner.try_callback_permit(self.scope).ok_or(())?;
            // A zero consumed delta can still retire stale source blocks.
            permit.record_actual_progress(
                delta,
                None,
                self.source.cursor.queued_frames(1),
                self.source.cursor.buffer_count(),
            );
            self.settled = actual.source_frames;
            return Ok(actual.view);
        }
        // Before any output, or after clear, no position exists. Do not mistake
        // an incoherent read during callback activity for this empty case.
        if publication.is_valid()
            || self.progress.load(Ordering::Acquire) != self.settled
            || !gate.validate(&view)
        {
            return Err(());
        }
        Ok(view)
    }

    pub(super) fn preparation_version(&mut self, gate: &ClaimGate) -> Result<(View, u64, u64), ()> {
        let view = self.try_reconcile(gate)?;
        Ok((view.view, self.next_source_id, self.settled))
    }

    pub(super) fn preparation(&mut self, gate: &ClaimGate) -> Result<(View, Preparation), ()> {
        let view = self.try_reconcile(gate)?;
        let source = self.source.fork_actual();
        if !gate.validate(&view) {
            return Err(());
        }
        Ok((view.view, source))
    }

    pub(super) fn preparation_snapshot(
        &mut self,
        gate: &ClaimGate,
    ) -> Result<(View, u64, u64, Preparation), ()> {
        let observation = self.try_reconcile(gate)?;
        let source = self.source.fork_actual();
        if !gate.validate(&observation) {
            return Err(());
        }
        Ok((observation.view, self.next_source_id, self.settled, source))
    }

    pub(super) fn reconcile(&mut self) {
        assert!(
            self.publication.is_none(),
            "position-aware ledger requires a validated checkpoint"
        );
        let actual = self.progress.load(Ordering::Acquire);
        let delta = usize::try_from(actual.checked_sub(self.settled).unwrap()).unwrap();
        if delta == 0 {
            return;
        }
        for _ in 0..delta {
            assert!(self.source.cursor.consume_next_frame(1, 1_000, None));
        }
        // Reuse the real owner accounting on the non-RT side only. Production
        // consumed_frames readers must eventually read the canonical publisher
        // directly; this bridge does not claim that integration is already done.
        let mut permit = self.owner.try_callback_permit(self.scope).unwrap();
        permit.record_actual_progress(
            delta,
            None,
            self.source.cursor.queued_frames(1),
            self.source.cursor.buffer_count(),
        );
        self.settled = actual;
    }

    fn enqueue(&mut self, buffer: AudioBuffer) -> EnqueueOutcome {
        self.reconcile();
        let frames = buffer.samples.len();
        let id = self.next_source_id;
        self.next_source_id = id.checked_add(1).unwrap();
        self.owner.enqueue_with_actual(self.scope, frames, || {
            self.source.push(id, buffer);
            (
                self.source.cursor.queued_frames(1),
                self.source.cursor.buffer_count(),
            )
        })
    }

    // Err retains the input for a non-RT retry, never a public Full result.
    // Reconciliation before validation is safe: it only retires PCM that the
    // device has already consumed. A failed attempt has not enqueued the input.
    pub(super) fn try_enqueue(
        &mut self,
        buffer: AudioBuffer,
        gate: &ClaimGate,
        after_accounting: impl FnOnce(),
    ) -> Result<EnqueueOutcome, AudioBuffer> {
        if self.owner.needs_terminal_check(self.scope) {
            // The atomics are only a hint: obtain the exact public outcome
            // from the real owner, without touching the source queue.
            return Ok(self
                .owner
                .enqueue_with_actual(self.scope, buffer.samples.len(), || {
                    unreachable!("closed/stale execution cannot reopen the same scope")
                }));
        }
        let Ok(view) = self.try_reconcile(gate) else {
            return Err(buffer);
        };
        let frames = buffer.samples.len();
        let queued_frames = self.source.cursor.queued_frames(1);
        let queued_buffers = self.source.cursor.buffer_count();
        // Mirror only the gross quota predicate to select validation vs revoke;
        // the real owner still computes the public result below. Fixed mono
        // fixtures have already passed the player-format validation boundary.
        let fits = frames > 0
            && frames <= self.limits.max_chunk_frames()
            && queued_frames
                .checked_add(frames)
                .is_some_and(|n| n <= self.limits.hard_frames())
            && queued_buffers
                .checked_add(1)
                .is_some_and(|n| n <= self.limits.hard_buffers());
        // Match the real queue's fast append boundary. A pure suffix does not
        // invalidate any already prepared prefix; reordering/replacement does.
        let appends = self.source.cursor.queue.is_empty()
            || (self
                .source
                .cursor
                .queue
                .back()
                .is_some_and(|tail| tail.timestamp <= buffer.timestamp)
                && self
                    .source
                    .cursor
                    .pending_end_upper_bound
                    .is_some_and(|end| i128::from(buffer.timestamp) >= end));
        after_accounting();
        let committed = if fits && !appends {
            gate.replace(view).is_ok()
        } else {
            gate.validate(&view)
        };
        if !committed {
            return Err(buffer);
        }
        // Do not reload device progress after the linearization point: the
        // public counts belong to this transaction's validated observation.
        let id = self.next_source_id;
        let outcome = self.owner.enqueue_with_actual(self.scope, frames, || {
            self.source.push(id, buffer);
            (
                self.source.cursor.queued_frames(1),
                self.source.cursor.buffer_count(),
            )
        });
        match outcome {
            EnqueueOutcome::Accepted { .. } => {
                assert!(fits);
                self.next_source_id = id.checked_add(1).unwrap();
            }
            EnqueueOutcome::Full { .. } => assert!(!fits),
            EnqueueOutcome::Closed
            | EnqueueOutcome::StaleScope
            | EnqueueOutcome::FormatMismatch
            | EnqueueOutcome::InvalidBuffer => {}
        }
        Ok(outcome)
    }

    pub(super) fn try_reanchor(&mut self, gate: &ClaimGate, cursor_us: i64) -> Result<(), ()> {
        self.try_reanchor_with(gate, || Some(cursor_us)).map(|_| ())
    }

    pub(super) fn try_reanchor_with(
        &mut self,
        gate: &ClaimGate,
        target_now: impl FnOnce() -> Option<i64>,
    ) -> Result<i64, ()> {
        let view = self.try_reconcile(gate)?;
        // Canonical conversion belongs to this attempt, after actual progress
        // reconciliation. A failed CAS does not retain the computed target.
        let cursor_us = target_now().ok_or(())?;
        gate.reanchor(view).map_err(|_| ())?;
        self.source.cursor.cursor_us = cursor_us;
        self.source.cursor.cursor_remainder = 0;
        if let Some(publication) = &self.publication {
            publication.invalidate_for_reanchor();
        }
        Ok(cursor_us)
    }

    pub(super) fn try_clear(&mut self, gate: &ClaimGate) -> Result<RendererOperationOutcome, ()> {
        self.try_clear_with(gate, || {})
    }

    pub(super) fn try_clear_with(
        &mut self,
        gate: &ClaimGate,
        invalidate_preparation: impl FnOnce(),
    ) -> Result<RendererOperationOutcome, ()> {
        let view = self.try_reconcile(gate)?;
        gate.clear(view).map_err(|_| ())?;
        // New source publication is serialized with this non-RT ledger. After
        // the clear commit a callback can only retire the revoked old output;
        // it cannot consume new source until the subsequent enqueue/publication.
        Ok(self.owner.clear_with_actual(self.scope, || {
            self.source.reset();
            if let Some(publication) = &self.publication {
                publication.invalidate_after_clear();
            }
            invalidate_preparation();
        }))
    }
}

#[test]
fn realtime_handoff_probe_admission_clear_preserves_consumption_and_reopens_empty_queue() {
    let (mut prepare, mut device) = pipe(3);
    let mut ledger = SourceLedger::new(Arc::clone(&device.progress));
    let first = source_pcm(&[1, 2, 3, 4]);
    let mut pending = source_pcm(&[11, 12, 13, 14]);
    pending.timestamp = 4_000;
    assert!(prepare
        .publish(PreparedSource::prepare(&first, 1, 0, &[1; 4]))
        .is_ok());
    assert!(prepare
        .publish(PreparedSource::prepare(&pending, 2, 0, &[1; 4]))
        .is_ok());
    ledger.enqueue(first);
    ledger.enqueue(pending);
    device.gate.begin_callback();
    let mut prefix = [0];
    assert_eq!(device.render_active(&mut prefix), 0);
    assert_eq!(prefix, [1]);
    assert_eq!(ledger.try_clear(&device.gate), Err(()));
    device.gate.end_callback();
    assert_eq!(
        ledger.try_clear(&device.gate),
        Ok(RendererOperationOutcome::Applied)
    );
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        0
    );
    assert_eq!(ledger.owner.consumed_frames(ledger.scope).unwrap(), 1);
    // Clear need not wait for another device read to expose its empty queue.
    let replacement = source_pcm(&[30, 31]);
    let prepared = PreparedSource::prepare(&replacement, 3, device.gate.view().epoch(), &[1; 2]);
    assert_eq!(
        ledger.try_enqueue(replacement, &device.gate, || {}).ok(),
        Some(EnqueueOutcome::Accepted {
            queued_frames: 2,
            queued_buffers: 1,
        })
    );
    assert!(prepare.publish(prepared).is_ok());
    let mut output = [0; 3];
    assert_eq!(device.render(&mut output), 1);
    assert_eq!(output, [30, 31, 0]);
    ledger.reconcile();
    assert_eq!(ledger.owner.consumed_frames(ledger.scope).unwrap(), 3);
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        0
    );
    assert_eq!(prepare.reclaim(), 3);
}

#[test]
fn realtime_handoff_probe_admission_append_keeps_unclaimed_prepared_prefix() {
    let (mut prepare, mut device) = pipe(1);
    let mut ledger = SourceLedger::new(Arc::clone(&device.progress));
    ledger.enqueue(source_pcm(&[1, 2]));
    assert!(prepare
        .publish(PreparedSource::prepare(&source_pcm(&[1, 2]), 1, 0, &[1; 2]))
        .is_ok());
    let mut suffix = source_pcm(&[3, 4]);
    suffix.timestamp = 2_000;
    assert_eq!(
        ledger.try_enqueue(suffix, &device.gate, || {}).ok(),
        Some(EnqueueOutcome::Accepted {
            queued_frames: 4,
            queued_buffers: 2,
        })
    );
    let mut output = [0; 2];
    assert_eq!(device.render(&mut output), 0);
    assert_eq!(output, [1, 2]);
    assert_eq!(prepare.reclaim(), 1);
    ledger.reconcile();
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        2
    );
}

#[test]
fn realtime_handoff_probe_admission_retries_obsolete_full_after_partial_callback() {
    let (mut prepare, mut device) = pipe(1);
    let mut ledger = SourceLedger::new(Arc::clone(&device.progress));
    ledger.enqueue(source_pcm(&[10; 16]));
    let mut pending = source_pcm(&[20; 16]);
    pending.timestamp = 16_000;
    ledger.enqueue(pending);
    assert!(prepare
        .publish(PreparedSource::prepare(
            &source_pcm(&[10; 16]),
            1,
            0,
            &[1; 16]
        ))
        .is_ok());
    assert_eq!(device.render(&mut [0]), 0);
    let mut incoming = source_pcm(&[30; 8]);
    incoming.timestamp = 32_000;
    let gate = Arc::clone(&device.gate);
    let retry = ledger.try_enqueue(incoming, &gate, || {
        assert_eq!(device.render(&mut [0; 8]), 0);
    });
    assert!(retry.is_err(), "obsolete Full is an internal retry");
    let outcome = ledger.try_enqueue(retry.unwrap_err(), &gate, || {});
    assert_eq!(
        outcome.ok().expect("stable admission commits"),
        EnqueueOutcome::Accepted {
            queued_frames: 31,
            queued_buffers: 3,
        }
    );
    let epoch = gate.view().epoch();
    let mut excessive = source_pcm(&[40; 2]);
    excessive.timestamp = 40_000;
    assert_eq!(
        ledger
            .try_enqueue(excessive, &gate, || {})
            .ok()
            .expect("stable rejection commits"),
        EnqueueOutcome::Full {
            queued_frames: 31,
            queued_buffers: 3,
        }
    );
    assert_eq!(gate.view().epoch(), epoch, "Full does not revoke output");
    let mut tail = [0; 7];
    assert_eq!(device.render(&mut tail), 0);
    assert_eq!(tail, [10; 7]);
    assert_eq!(prepare.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_admission_waits_for_first_frame_before_overlap_commit() {
    let (mut prepare, mut device) = pipe(1);
    let mut ledger = SourceLedger::new(Arc::clone(&device.progress));
    ledger.enqueue(source_pcm(&[1, 2, 3, 4]));
    assert!(prepare
        .publish(PreparedSource::prepare(
            &source_pcm(&[1, 2, 3, 4]),
            1,
            0,
            &[1; 4]
        ))
        .is_ok());
    device.gate.begin_callback();
    assert!(device.gate.claim(0, 1));
    device.claimed = true; // Pause at the real reader's successful-claim boundary.
    let mut replacement = source_pcm(&[20, 21, 22, 23]);
    replacement.timestamp = 2_000;
    let retry = ledger.try_enqueue(replacement, &device.gate, || {});
    assert!(retry.is_err());
    let mut first = [0];
    assert_eq!(device.render_active(&mut first), 0);
    assert_eq!(first, [1]);
    device.gate.end_callback();
    assert_eq!(
        ledger
            .try_enqueue(retry.unwrap_err(), &device.gate, || {})
            .ok()
            .expect("completed read permits admission"),
        EnqueueOutcome::Accepted {
            queued_frames: 7,
            queued_buffers: 2,
        }
    );
    let mut tail = [0; 3];
    assert_eq!(device.render(&mut tail), 0);
    assert_eq!(tail, [2, 3, 4]);
    ledger.reconcile();
    let mut remaining = Vec::new();
    let mut sample = [0];
    while ledger
        .source
        .cursor
        .consume_next_frame(1, 1_000, Some(&mut sample))
    {
        remaining.push(sample[0]);
    }
    assert_eq!(remaining, [22, 23]);
    assert_eq!(prepare.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_admission_overlapping_next_source_outputs_only_live_suffix() {
    let (mut prepare, mut device) = pipe(2);
    let mut ledger = SourceLedger::new(Arc::clone(&device.progress));
    let first = source_pcm(&[1, 2, 3, 4]);
    let first_prepared = PreparedSource::prepare(&first, 1, 0, &[1; 4]);
    ledger.enqueue(first);
    assert!(prepare.publish(first_prepared).is_ok());
    assert_eq!(device.render(&mut [0]), 0);
    let mut next = source_pcm(&[20, 21, 22, 23]);
    next.timestamp = 2_000;
    let next_prepared =
        PreparedSource::prepare_from_cursor(&next, 2, device.gate.view().epoch(), &[1; 2], 4_000);
    assert_eq!(
        ledger.try_enqueue(next, &device.gate, || {}).ok(),
        Some(EnqueueOutcome::Accepted {
            queued_frames: 7,
            queued_buffers: 2,
        })
    );
    assert!(prepare.publish(next_prepared).is_ok());
    let mut output = [0; 6];
    assert_eq!(device.render(&mut output), 1);
    assert_eq!(output, [2, 3, 4, 22, 23, 0]);
    assert_eq!(device.progress.load(Ordering::Acquire), 6);
    ledger.reconcile();
    assert_eq!(ledger.owner.consumed_frames(ledger.scope).unwrap(), 6);
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        0
    );
    assert_eq!(prepare.reclaim(), 2);
}

#[test]
fn realtime_handoff_probe_admission_uses_partial_source_progress_before_full_check() {
    let (mut prepare, mut device) = pipe(2);
    let mut ledger = SourceLedger::new(Arc::clone(&device.progress));
    for first in [1, 20] {
        let samples: Vec<_> = (first..first + 16).collect();
        assert_eq!(
            ledger.enqueue(source_pcm(&samples)),
            EnqueueOutcome::Accepted {
                queued_frames: 16,
                queued_buffers: 1,
            }
        );
    }
    let data = PreparedSource::prepare(&ledger.source.cursor.queue[0].buffer, 1, 0, &[1; 16]);
    assert!(prepare.publish(data).is_ok());
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        16
    );
    assert_eq!(ledger.owner.consumed_frames(ledger.scope).unwrap(), 0);
    let mut first = [0; 8];
    assert_eq!(device.render(&mut first), 0);
    assert_eq!(first, [20, 21, 22, 23, 24, 25, 26, 27]);
    let mut next = source_pcm(&[100; 16]);
    next.timestamp = 16_000;
    assert_eq!(
        ledger.enqueue(next),
        EnqueueOutcome::Accepted {
            queued_frames: 24,
            queued_buffers: 2,
        }
    );
    assert_eq!(ledger.owner.consumed_frames(ledger.scope).unwrap(), 8);
    let mut excessive = source_pcm(&[200; 9]);
    excessive.timestamp = 32_000;
    assert_eq!(
        ledger.enqueue(excessive),
        EnqueueOutcome::Full {
            queued_frames: 24,
            queued_buffers: 2,
        }
    );
    let mut rest = [0; 8];
    assert_eq!(device.render(&mut rest), 0);
    assert_eq!(rest, [28, 29, 30, 31, 32, 33, 34, 35]);
    ledger.reconcile();
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        16
    );
    assert_eq!(ledger.owner.consumed_frames(ledger.scope).unwrap(), 16);
    ledger.reconcile();
    assert_eq!(ledger.owner.consumed_frames(ledger.scope).unwrap(), 16);
    assert_eq!(prepare.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_admission_correction_progress_releases_only_consumed_source() {
    let (mut prepare, mut device) = pipe(1);
    let mut ledger = SourceLedger::new(Arc::clone(&device.progress));
    assert!(matches!(
        ledger.enqueue(source_pcm(&[10, 20, 30, 40])),
        EnqueueOutcome::Accepted { .. }
    ));
    let data = PreparedSource::prepare(&ledger.source.cursor.queue[0].buffer, 1, 0, &[1, 0, 2, 1]);
    assert!(prepare.publish(data).is_ok());
    let mut output = [0; 2];
    assert_eq!(device.render(&mut output), 0);
    ledger.reconcile();
    assert_eq!(output, [10, 10]);
    assert_eq!(ledger.owner.consumed_frames(ledger.scope).unwrap(), 1);
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        3
    );
    assert_eq!(device.render(&mut output), 0);
    ledger.reconcile();
    assert_eq!(output, [30, 40]);
    assert_eq!(ledger.owner.consumed_frames(ledger.scope).unwrap(), 4);
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        0
    );
    assert_eq!(prepare.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_admission_close_during_accounting_returns_owner_outcome() {
    for replace_scope in [false, true] {
        let gate = ClaimGate::default();
        let mut ledger = SourceLedger::with_checkpoint(
            Arc::new(AtomicU64::new(0)),
            Arc::new(PublishedCheckpoint::new(1)),
        );
        let owner = ledger.owner.clone();
        let scope = ledger.scope;
        let id_before = ledger.next_source_id;
        let result = ledger.try_enqueue(source_pcm(&[1]), &gate, || {
            if replace_scope {
                owner.teardown(scope);
                owner.mint_scope().unwrap();
            } else {
                assert_eq!(owner.close(scope), RendererOperationOutcome::Applied);
            }
        });
        assert!(
            matches!(result, Ok(outcome) if outcome == if replace_scope { EnqueueOutcome::StaleScope } else { EnqueueOutcome::Closed })
        );
        assert_eq!(ledger.next_source_id, id_before);
        assert_eq!(ledger.source.cursor.queued_frames(1), 0);
        assert_eq!(ledger.source.cursor.buffer_count(), 0);
    }
}

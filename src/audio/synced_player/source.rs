//! Serialized non-RT source transactions over actual device checkpoints.
//! Preparation never advances public consumption or decides enqueue outcomes.

use super::ingress::{ControlGate, Observation, PublishedCheckpoint};
use super::{AudioBuffer, AudioBufferLifetime, PlaybackQueue};
use crate::audio::{
    EnqueueOutcome, PlayerScope, RendererOperationOutcome, RendererOwner, RendererQueueLimits,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

pub(super) struct Publication {
    pub control: Arc<ControlGate>,
    pub checkpoint: Arc<PublishedCheckpoint>,
    pub consumed: Arc<AtomicU64>,
    pub limits: RendererQueueLimits,
    pub channels: usize,
}

impl Publication {
    pub(super) fn new(
        owner: &RendererOwner,
        scope: PlayerScope,
        channels: usize,
        limits: RendererQueueLimits,
    ) -> Result<Arc<Self>, RendererOperationOutcome> {
        Ok(Arc::new(Self {
            control: owner.control_gate(scope)?,
            checkpoint: Arc::new(PublishedCheckpoint::new()),
            consumed: owner.consumption_counter(scope)?,
            limits,
            channels,
        }))
    }
}

type PendingInput = (AudioBuffer, Option<Arc<dyn AudioBufferLifetime>>);

impl PlaybackQueue {
    pub(super) fn attach_publication(&mut self, publication: Arc<Publication>) {
        assert!(self.publication.is_none());
        self.settled_consumed = publication.consumed.load(Ordering::Acquire);
        self.publication = Some(publication);
    }

    /// Failed validation may retain reconciliation of already-consumed PCM, but
    /// never installs the pending input. Caller holds the canonical queue mutex.
    pub(super) fn reconcile_actual<'a>(
        &mut self,
        publication: &'a Publication,
        owner: &RendererOwner,
        scope: PlayerScope,
    ) -> Result<Observation<'a>, ()> {
        if owner.needs_terminal_check(scope) {
            return Err(());
        }
        let view = publication.control.observe().ok_or(())?;
        if publication.checkpoint.is_valid() {
            let actual = publication
                .checkpoint
                .read_observed(view, &publication.consumed)
                .ok_or(())?;
            self.reconcile_source(actual.position);
            self.settled_consumed = actual.source_frames;
            let result = owner.reconcile_queue(
                scope,
                self.queued_frames(publication.channels),
                self.buffer_count(),
                None,
            );
            if result != RendererOperationOutcome::Applied {
                return Err(());
            }
            return Ok(actual.view);
        }
        if publication.consumed.load(Ordering::Acquire) != self.settled_consumed
            || !publication.control.validate(&view)
        {
            return Err(());
        }
        Ok(view)
    }

    /// Input has passed format/timestamp validation. Err is an internal non-RT
    /// retry with input ownership preserved, never a new public Full result.
    pub(super) fn try_enqueue_prepared(
        &mut self,
        publication: &Publication,
        owner: &RendererOwner,
        scope: PlayerScope,
        input: PendingInput,
        frames: usize,
    ) -> Result<EnqueueOutcome, PendingInput> {
        if owner.needs_terminal_check(scope) {
            return Ok(owner.enqueue_with_actual(scope, frames, || {
                unreachable!("closed/stale scope cannot reopen")
            }));
        }
        let Ok(view) = self.reconcile_actual(publication, owner, scope) else {
            return Err(input);
        };
        let fits = frames > 0
            && frames <= publication.limits.max_chunk_frames()
            && self
                .queued_frames(publication.channels)
                .checked_add(frames)
                .is_some_and(|n| n <= publication.limits.hard_frames())
            && self
                .buffer_count()
                .checked_add(1)
                .is_some_and(|n| n <= publication.limits.hard_buffers());
        let buffer = &input.0;
        let appends = self.queue.is_empty()
            || (self
                .queue
                .back()
                .is_some_and(|tail| tail.timestamp <= buffer.timestamp)
                && self
                    .pending_end_upper_bound
                    .is_some_and(|end| i128::from(buffer.timestamp) >= end));
        #[cfg(test)]
        tests::before_admission_validation();
        let valid = if fits && !appends {
            publication.control.replace(view).is_ok()
        } else {
            publication.control.validate(&view)
        };
        if !valid {
            return Err(input);
        }
        let (buffer, lifetime) = input;
        Ok(owner.enqueue_with_actual(scope, frames, || {
            self.push_with_lifetime(buffer, lifetime);
            self.enqueue_count += 1;
            (
                self.queued_frames(publication.channels),
                self.buffer_count(),
            )
        }))
    }

    /// Build a preparation view from a validated actual position. Reuse the
    /// preallocated destination container; no per-frame source identity scan.
    pub(super) fn preparation_snapshot(
        &mut self,
        publication: &Publication,
        owner: &RendererOwner,
        scope: PlayerScope,
        destination: &mut Self,
    ) -> Result<(u64, u64, u64), ()> {
        let view = self.reconcile_actual(publication, owner, scope)?;
        self.copy_source_into(destination);
        if !publication.control.validate(&view) {
            destination.clear();
            return Err(());
        }
        Ok((view.epoch(), self.next_source_id, self.settled_consumed))
    }

    /// Preserve the original first-playable selection for explicit reanchor.
    /// A missing target leaves force_reanchor pending, without invalidating PCM.
    pub(super) fn try_reanchor_prepared(
        &mut self,
        publication: &Publication,
        owner: &RendererOwner,
        scope: PlayerScope,
        target_now: impl FnOnce() -> Option<i64>,
        explicit: bool,
    ) -> Result<Option<i64>, ()> {
        let view = self.reconcile_actual(publication, owner, scope)?;
        let Some(target) = target_now() else {
            return Ok(None);
        };
        let target = if explicit {
            let Some(target) = self.first_playable_cursor_at_or_after(target) else {
                return Ok(None);
            };
            target
        } else {
            target
        };
        let committed = publication.control.reanchor(view).map_err(|_| ())?;
        self.cursor_us = target;
        self.cursor_remainder = 0;
        self.force_reanchor = false;
        publication.checkpoint.invalidate_before(committed.epoch());
        Ok(Some(target))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{reader::Reader, transport};
    use super::*;
    use crate::audio::sync_correction::CorrectionSchedule;
    use crate::audio::{AudioFormat, Codec};
    use std::cell::RefCell;
    use std::sync::mpsc;

    thread_local! {
        // Test harness only: stop a real admission after its capacity decision,
        // before the original validation. No hook exists in production builds.
        static ADMISSION_GATE: RefCell<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>> =
            const { RefCell::new(None) };
    }

    pub(super) fn before_admission_validation() {
        let gate = ADMISSION_GATE.with(|slot| slot.borrow_mut().take());
        if let Some((entered, proceed)) = gate {
            entered.send(()).unwrap();
            proceed.recv().unwrap();
        }
    }

    #[test]
    fn renderer_realtime_source_retries_stale_full_and_accepted_after_actual_progress() {
        // Both an initially Full decision and an initially fitting decision must
        // be revalidated, preserving the same input through the internal retry.
        for (initial, consumed_now, expected_tail) in [
            (vec![1, 2, 3, 4], 2, vec![3, 4, 9, 10]),
            (vec![1, 2], 1, vec![2, 9, 10]),
        ] {
            let limits = RendererQueueLimits::new(4, 3, 4).unwrap();
            let owner = RendererOwner::new(limits);
            let scope = owner.mint_scope().unwrap();
            let publication = Publication::new(&owner, scope, 1, limits).unwrap();
            let mut canonical = PlaybackQueue::new();
            canonical.attach_publication(publication.clone());
            assert!(matches!(
                canonical.try_enqueue_prepared(
                    &publication,
                    &owner,
                    scope,
                    input(0, &initial),
                    initial.len()
                ),
                Ok(EnqueueOutcome::Accepted { .. })
            ));
            let mut private = PlaybackQueue::new();
            private.queue.reserve(limits.hard_buffers());
            let (epoch, _, base) = canonical
                .preparation_snapshot(&publication, &owner, scope, &mut private)
                .unwrap();
            let (mut producer, mut device) = transport::pipe(2, 4, 1).unwrap();
            private.prepare_window(producer.builder_mut(), epoch, base, 1, 1_000);
            assert!(producer.publish());
            let mut reader = Reader::new(1);
            let pending = input(initial.len() as i64 * 1_000, &[9, 10]);
            let original_samples = Arc::clone(&pending.0.samples);
            let (entered_tx, entered_rx) = mpsc::channel();
            let (proceed_tx, proceed_rx) = mpsc::channel();
            let (attempt, prefix) = std::thread::scope(|threads| {
                let canonical = &mut canonical;
                let publication = &publication;
                let owner = &owner;
                let admission = threads.spawn(move || {
                    ADMISSION_GATE.with(|slot| {
                        *slot.borrow_mut() = Some((entered_tx, proceed_rx));
                    });
                    canonical.try_enqueue_prepared(publication, owner, scope, pending, 2)
                });
                entered_rx.recv().unwrap();
                let mut prefix = vec![0; consumed_now];
                reader.render(
                    &mut device,
                    &publication.control.begin_callback(),
                    &publication.checkpoint,
                    &publication.consumed,
                    &mut prefix,
                    false,
                    Some(CorrectionSchedule::default()),
                );
                // Release the admission before assertions, so a failure cannot
                // strand a scoped thread at the controlled interleaving.
                proceed_tx.send(()).unwrap();
                (admission.join().unwrap(), prefix)
            });
            assert_eq!(prefix, initial[..consumed_now]);
            let pending = match attempt {
                Err(pending) => pending,
                Ok(outcome) => panic!("stale decision escaped as {outcome:?}"),
            };
            assert!(Arc::ptr_eq(&pending.0.samples, &original_samples));
            assert_eq!(owner.consumed_frames(scope), Ok(consumed_now as u64));
            let outcome = canonical
                .try_enqueue_prepared(&publication, &owner, scope, pending, 2)
                .ok()
                .unwrap();
            assert!(
                matches!(outcome, EnqueueOutcome::Accepted { queued_frames, .. }
                if queued_frames == expected_tail.len())
            );
            let (epoch, _, base) = canonical
                .preparation_snapshot(&publication, &owner, scope, &mut private)
                .unwrap();
            private.prepare_window(producer.builder_mut(), epoch, base, 1, 1_000);
            assert!(producer.publish());
            let mut tail = vec![0; expected_tail.len()];
            reader.render(
                &mut device,
                &publication.control.begin_callback(),
                &publication.checkpoint,
                &publication.consumed,
                &mut tail,
                false,
                Some(CorrectionSchedule::default()),
            );
            assert_eq!(tail, expected_tail, "accepted input remains consumable");
            assert_eq!(owner.consumed_frames(scope), Ok((initial.len() + 2) as u64));
        }
    }

    fn input(timestamp: i64, samples: &[i32]) -> PendingInput {
        (
            AudioBuffer {
                timestamp,
                samples: Arc::from(samples),
                format: AudioFormat {
                    codec: Codec::Pcm,
                    sample_rate: 1_000,
                    channels: 1,
                    bit_depth: 24,
                    codec_header: None,
                },
            },
            None,
        )
    }

    #[test]
    fn renderer_realtime_source_admission_settles_actual_capacity_and_clear_keeps_counter() {
        let limits = RendererQueueLimits::new(4, 3, 4).unwrap();
        let owner = RendererOwner::new(limits);
        let scope = owner.mint_scope().unwrap();
        let publication = Publication::new(&owner, scope, 1, limits).unwrap();
        let mut canonical = PlaybackQueue::new();
        canonical.attach_publication(publication.clone());
        let accepted = canonical
            .try_enqueue_prepared(&publication, &owner, scope, input(0, &[1, 2, 3, 4]), 4)
            .ok()
            .unwrap();
        assert!(matches!(
            accepted,
            EnqueueOutcome::Accepted {
                queued_frames: 4,
                ..
            }
        ));
        let mut private = PlaybackQueue::new();
        private.queue.reserve(limits.hard_buffers());
        let (epoch, _, base) = canonical
            .preparation_snapshot(&publication, &owner, scope, &mut private)
            .unwrap();
        let (mut producer, mut device) = transport::pipe(2, 2, 1).unwrap();
        for base in [base, base + 2] {
            private.prepare_window(producer.builder_mut(), epoch, base, 1, 1_000);
            assert!(producer.publish());
        }
        let mut reader = Reader::new(1);
        let mut output = [0; 2];
        reader.render(
            &mut device,
            &publication.control.begin_callback(),
            &publication.checkpoint,
            &publication.consumed,
            &mut output,
            false,
            Some(CorrectionSchedule::default()),
        );
        assert_eq!(output, [1, 2]);
        assert_eq!(owner.consumed_frames(scope), Ok(2));
        // Admission must see two actually consumed frames even when no worker
        // has independently refreshed the old capacity snapshot.
        let accepted = canonical
            .try_enqueue_prepared(&publication, &owner, scope, input(4_000, &[5, 6]), 2)
            .ok()
            .unwrap();
        assert!(matches!(
            accepted,
            EnqueueOutcome::Accepted {
                queued_frames: 4,
                ..
            }
        ));
        assert_eq!(owner.consumed_frames(scope), Ok(2));
        assert_eq!(
            owner.clear_with_actual(scope, || canonical.clear()),
            RendererOperationOutcome::Applied
        );
        assert_eq!(owner.consumed_frames(scope), Ok(2));
        let accepted = canonical
            .try_enqueue_prepared(&publication, &owner, scope, input(10_000, &[9, 10]), 2)
            .ok()
            .unwrap();
        assert!(matches!(
            accepted,
            EnqueueOutcome::Accepted {
                queued_frames: 2,
                ..
            }
        ));
        let (epoch, _, base) = canonical
            .preparation_snapshot(&publication, &owner, scope, &mut private)
            .unwrap();
        assert_eq!(base, 2);
        assert_eq!(producer.reclaim(), 1);
        private.prepare_window(producer.builder_mut(), epoch, base, 1, 1_000);
        assert!(producer.publish());
        reader.render(
            &mut device,
            &publication.control.begin_callback(),
            &publication.checkpoint,
            &publication.consumed,
            &mut output,
            false,
            Some(CorrectionSchedule::default()),
        );
        assert_eq!(output, [9, 10], "revoked old tail must not resume");
        assert_eq!(owner.consumed_frames(scope), Ok(4));
        drop(
            canonical
                .reconcile_actual(&publication, &owner, scope)
                .unwrap(),
        );
        assert_eq!(owner.capacity(scope).unwrap().current_frames(), 0);
        assert_eq!(owner.capacity(scope).unwrap().high_water_frames(), 4);
        owner.teardown_with_actual(scope, || canonical.clear());
        assert!(
            canonical
                .reconcile_actual(&publication, &owner, scope)
                .is_err(),
            "a stopped worker must not reconcile a pre-terminal checkpoint into a cleared source"
        );
        assert_eq!(owner.consumed_frames(scope), Ok(4));
    }

    #[test]
    fn renderer_realtime_source_explicit_reanchor_retains_pending_until_playable_target() {
        let limits = RendererQueueLimits::new(4, 3, 4).unwrap();
        let owner = RendererOwner::new(limits);
        let scope = owner.mint_scope().unwrap();
        let publication = Publication::new(&owner, scope, 1, limits).unwrap();
        let mut source = PlaybackQueue::new();
        source.attach_publication(publication.clone());
        source
            .try_enqueue_prepared(&publication, &owner, scope, input(100_000, &[1, 2]), 2)
            .ok()
            .unwrap();
        source
            .try_enqueue_prepared(&publication, &owner, scope, input(300_000, &[3, 4]), 2)
            .ok()
            .unwrap();
        let epoch = publication.control.view().epoch();
        assert_eq!(
            source.try_reanchor_prepared(&publication, &owner, scope, || None, true),
            Ok(None)
        );
        assert_eq!(
            source.try_reanchor_prepared(&publication, &owner, scope, || Some(400_000), true),
            Ok(None)
        );
        assert!(source.force_reanchor);
        assert_eq!(publication.control.view().epoch(), epoch);
        assert_eq!(
            source.try_reanchor_prepared(&publication, &owner, scope, || Some(200_000), true),
            Ok(Some(300_000))
        );
        assert!(!source.force_reanchor);
        assert_eq!(source.cursor_us, 300_000);
        assert_eq!(publication.control.view().epoch(), epoch + 1);
        assert_eq!(owner.consumed_frames(scope), Ok(0));
    }
}

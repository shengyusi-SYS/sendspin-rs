//! Fixed-size actual device position; all writes occur within ACTIVE.
//! The existing gate validates coherent non-RT reads; no callback retry or lock.

use super::{CallbackGuard, ControlGate, Observation};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize};
use std::sync::atomic::{AtomicU64, Ordering};

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
    pub source_frames: u64,
    pub position: SourcePosition,
}

pub(crate) struct PublishedCheckpoint {
    valid: AtomicBool,
    epoch: AtomicU64,
    invalid_before: AtomicU64,
    retired_through: AtomicU64,
    index: AtomicUsize,
    cursor_us: AtomicI64,
    cursor_remainder: AtomicI64,
}

impl PublishedCheckpoint {
    pub(crate) fn new() -> Self {
        Self {
            valid: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            invalid_before: AtomicU64::new(0),
            retired_through: AtomicU64::new(0),
            index: AtomicUsize::new(0),
            cursor_us: AtomicI64::new(0),
            cursor_remainder: AtomicI64::new(0),
        }
    }

    pub(crate) fn publish(&self, callback: &CallbackGuard<'_>, position: SourcePosition) {
        callback.publish_current(position.current);
        self.retired_through
            .store(position.retired_through, Ordering::Relaxed);
        self.index.store(position.index, Ordering::Relaxed);
        self.cursor_us.store(position.cursor_us, Ordering::Relaxed);
        self.cursor_remainder
            .store(position.cursor_remainder, Ordering::Relaxed);
        self.epoch.store(callback.view().epoch(), Ordering::Relaxed);
        self.valid.store(true, Ordering::Release);
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
            && self.epoch.load(Ordering::Relaxed) >= self.invalid_before.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) fn read<'a>(
        &self,
        gate: &'a ControlGate,
        progress: &AtomicU64,
    ) -> Option<Snapshot<'a>> {
        let view = gate.observe()?;
        self.read_observed(view, progress)
    }

    pub(crate) fn read_observed<'a>(
        &self,
        view: Observation<'a>,
        progress: &AtomicU64,
    ) -> Option<Snapshot<'a>> {
        let gate = view.gate;
        if view.callback_active()
            || !self.valid.load(Ordering::Acquire)
            || self.epoch.load(Ordering::Relaxed) < self.invalid_before.load(Ordering::Acquire)
        {
            return None;
        }
        let current = view.current();
        let snapshot = Snapshot {
            view,
            source_frames: progress.load(Ordering::Acquire),
            position: SourcePosition {
                retired_through: self.retired_through.load(Ordering::Relaxed),
                current,
                index: self.index.load(Ordering::Relaxed),
                cursor_us: self.cursor_us.load(Ordering::Relaxed),
                cursor_remainder: self.cursor_remainder.load(Ordering::Relaxed),
            },
        };
        (self.valid.load(Ordering::Acquire) && gate.validate(&snapshot.view)).then_some(snapshot)
    }
    /// Call under owner serialization after successful clear/reanchor and before
    /// publishing new windows. Only the callback writes actual position.
    pub(crate) fn invalidate_before(&self, epoch: u64) {
        self.invalid_before.store(epoch, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_realtime_checkpoint_replace_preserves_actual_clear_rejects_old_position() {
        let gate = ControlGate::default();
        let checkpoint = PublishedCheckpoint::new();
        let progress = AtomicU64::new(7);
        let position = SourcePosition {
            retired_through: 17,
            current: 1 << 40,
            index: 7,
            cursor_us: -123,
            cursor_remainder: 11,
        };
        {
            let callback = gate.begin_callback();
            checkpoint.publish(&callback, position);
        }
        let actual = checkpoint.read(&gate, &progress).unwrap();
        assert_eq!(actual.position, position);
        assert_eq!(actual.source_frames, 7);
        gate.replace(actual.view).unwrap();
        assert!(checkpoint.read(&gate, &progress).is_some());
        let cleared = gate.clear(gate.observe().unwrap()).unwrap();
        checkpoint.invalidate_before(cleared.epoch());
        assert!(checkpoint.read(&gate, &progress).is_none());
        assert_eq!(progress.load(Ordering::Relaxed), 7);
        {
            let callback = gate.begin_callback();
            callback.finish_current();
            let fresh = SourcePosition {
                current: 0,
                index: 0,
                ..position
            };
            checkpoint.publish(&callback, fresh);
        }
        let fresh = checkpoint.read(&gate, &progress).unwrap();
        assert_eq!(fresh.position.current, 0);
        assert_eq!(fresh.source_frames, 7);
    }
}

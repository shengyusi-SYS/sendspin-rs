//! Scope-local control and actual-position publication for realtime consumption.
//! The existing renderer owner serializes non-RT controls and validates scopes.
//! A callback never acquires or waits for the observation lease.

use crate::audio::{ScheduledArmOutcome, ScheduledStartOutcome, StartState};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

mod checkpoint;
pub(crate) use checkpoint::{PublishedCheckpoint, SourcePosition};

// Low flags and a 56-bit control version share the clear/start CAS.
// Source identity is a separate u64 checkpoint written by the active reader.
// These are scope-local engineering counters, not a proof of infinite lifetime.
const EPOCH_SHIFT: u32 = 8;
const LOWER_FIELDS: u64 = 255;
const CALLBACK_ACTIVE: u64 = 1;
const CALLBACK_DIRTY: u64 = 2;
const REVOKED_CURRENT: u64 = 4;
const REANCHORED_TIMELINE: u64 = 8;
const START_ARMED: u64 = 16;
const START_WON: u64 = 32;
const START_CLOSED: u64 = 64;
// Only the callback already active at the first close may still consume.
// Later silent callbacks must not postpone an already stable terminal snapshot.
const CLOSED_CONSUMPTION_PENDING: u64 = 128;

fn closed_control(control: u64) -> u64 {
    let pending = if control & START_CLOSED == 0 && control & CALLBACK_ACTIVE != 0 {
        CLOSED_CONSUMPTION_PENDING
    } else {
        0
    };
    control | START_CLOSED | pending
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct View(u64, u64);
impl View {
    pub(crate) fn epoch(self) -> u64 {
        self.0 >> EPOCH_SHIFT
    }
    pub(crate) fn callback_active(self) -> bool {
        self.0 & CALLBACK_ACTIVE != 0
    }
    pub(crate) fn current(self) -> u64 {
        self.1
    }
    pub(crate) fn timeline_reanchored(self) -> bool {
        self.0 & REANCHORED_TIMELINE != 0
    }
    pub(crate) fn current_revoked(self) -> bool {
        self.0 & REVOKED_CURRENT != 0
    }
}

#[derive(Default, Debug)]
pub(crate) struct ControlGate {
    control: AtomicU64,
    current: AtomicU64,
    boundary: AtomicI64,
    // Lease for a serialized non-RT observation transaction. The
    // callback never accesses this lease. It prevents a second observation
    // from clearing DIRTY while an older transaction can still commit.
    observing: std::sync::atomic::AtomicBool,
}
#[derive(Debug)]
pub(crate) struct Observation<'a> {
    gate: &'a ControlGate,
    view: View,
}
impl std::ops::Deref for Observation<'_> {
    type Target = View;
    fn deref(&self) -> &View {
        &self.view
    }
}
impl Drop for Observation<'_> {
    fn drop(&mut self) {
        self.gate.observing.store(false, Ordering::Release);
    }
}
impl ControlGate {
    // Device/control-state inspection, not a non-RT coherent checkpoint.
    pub(crate) fn view(&self) -> View {
        View(
            self.control.load(Ordering::Acquire),
            self.current.load(Ordering::Relaxed),
        )
    }
    pub(crate) fn observe(&self) -> Option<Observation<'_>> {
        self.observing
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()?;
        let before = self.control.load(Ordering::Acquire);
        if before & CALLBACK_ACTIVE != 0
            || self
                .control
                .compare_exchange(
                    before,
                    before & !CALLBACK_DIRTY,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
        {
            self.observing.store(false, Ordering::Release);
            return None;
        }
        let observation = Observation {
            gate: self,
            view: View(
                before & !CALLBACK_DIRTY,
                self.current.load(Ordering::Relaxed),
            ),
        };
        Some(observation)
    }
    pub(crate) fn begin_callback(&self) -> CallbackGuard<'_> {
        let previous = self
            .control
            .fetch_or(CALLBACK_ACTIVE | CALLBACK_DIRTY, Ordering::AcqRel);
        assert_eq!(previous & CALLBACK_ACTIVE, 0, "single callback writer");
        CallbackGuard { gate: self }
    }
    fn end_callback(&self) {
        assert!(self.view().callback_active());
        self.control.fetch_and(
            !(CALLBACK_ACTIVE | CLOSED_CONSUMPTION_PENDING),
            Ordering::Release,
        );
    }
    fn publish_current(&self, current: u64) {
        assert!(
            self.view().callback_active(),
            "checkpoint publication requires ACTIVE"
        );
        self.current.store(current, Ordering::Relaxed);
    }
    fn claim(&self, published_epoch: u64, source: u64) -> bool {
        assert_ne!(source, 0);
        let empty = self.view();
        assert!(empty.callback_active(), "source claim requires ACTIVE");
        if empty.epoch() != published_epoch || empty.current() != 0 {
            return false;
        }
        if self
            .control
            .compare_exchange(
                empty.0,
                (empty.0 & !REVOKED_CURRENT) | CALLBACK_DIRTY,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        self.publish_current(source);
        true
    }
    pub(crate) fn validate(&self, observation: &Observation<'_>) -> bool {
        assert!(std::ptr::eq(self, observation.gate));
        self.control
            .compare_exchange(
                observation.0,
                observation.0,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
    pub(crate) fn replace(&self, observation: Observation<'_>) -> Result<View, View> {
        self.commit(observation, 0, false)
    }
    pub(crate) fn reanchor(&self, observation: Observation<'_>) -> Result<View, View> {
        self.commit(observation, REANCHORED_TIMELINE, false)
    }
    pub(crate) fn clear(&self, observation: Observation<'_>) -> Result<View, View> {
        self.commit(observation, REVOKED_CURRENT, true)
    }
    fn commit(&self, observation: Observation<'_>, notice: u64, clear: bool) -> Result<View, View> {
        assert!(std::ptr::eq(self, observation.gate));
        let version = observation
            .epoch()
            .checked_add(1)
            .expect("scope control version");
        assert!(
            version < (1 << 56),
            "scope control version exhausted; versions must never be reused"
        );
        let mut flags = (observation.0 & LOWER_FIELDS) | notice;
        if clear && flags & START_WON == 0 {
            flags &= !START_ARMED;
        }
        let next = (version << EPOCH_SHIFT) | flags;
        self.control
            .compare_exchange(observation.0, next, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| View(next, observation.current()))
            .map_err(|_| self.view())
    }
    fn finish_current(&self) {
        assert!(
            self.view().callback_active(),
            "source release requires ACTIVE"
        );
        self.publish_current(0);
        self.control
            .fetch_and(!(REVOKED_CURRENT | REANCHORED_TIMELINE), Ordering::AcqRel);
    }
}

impl ControlGate {
    /// Serialized owner operation; interference is retried only by the owner.
    pub(crate) fn try_arm(&self, boundary: i64) -> Result<ScheduledArmOutcome, ()> {
        let view = self.view();
        if view.0 & START_CLOSED != 0 {
            return Ok(ScheduledArmOutcome::Closed);
        }
        if view.callback_active() {
            return Err(());
        }
        let old = self.boundary.load(Ordering::Relaxed);
        if view.0 & START_WON != 0 {
            return Ok(ScheduledArmOutcome::BoundaryAlreadyWon {
                start_at_zone_us: old,
            });
        }
        if view.0 & START_ARMED != 0 {
            return Ok(ScheduledArmOutcome::AlreadyArmed {
                start_at_zone_us: old,
            });
        }
        self.boundary.store(boundary, Ordering::Relaxed);
        self.control
            .compare_exchange(
                view.0,
                view.0 | START_ARMED,
                Ordering::Release,
                Ordering::Acquire,
            )
            .map(|_| ScheduledArmOutcome::Armed)
            .map_err(|_| ())
    }

    /// Read while holding the same owner serialization used by arm and clear.
    /// Closing admission does not erase the publicly observable start phase.
    pub(crate) fn start_state(&self) -> StartState {
        let view = self.view();
        let boundary = self.boundary.load(Ordering::Relaxed);
        if view.0 & START_WON != 0 {
            StartState::BoundaryWon {
                start_at_zone_us: boundary,
            }
        } else if view.0 & START_ARMED != 0 {
            StartState::Armed {
                start_at_zone_us: boundary,
            }
        } else {
            StartState::Idle
        }
    }

    pub(crate) fn close(&self) {
        // Non-RT retry pairs closure with the exact callback active at closure.
        // Callback completion clears pending in its existing Release operation.
        let _ = self
            .control
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |control| {
                Some(closed_control(control))
            });
    }

    pub(crate) fn terminal_consumption_pending(&self) -> bool {
        self.control.load(Ordering::Acquire) & CLOSED_CONSUMPTION_PENDING != 0
    }

    /// Non-RT cancellation claim, never a stream-stopped acknowledgment.
    /// The original finalizer still stops/joins resources before publishing Ack.
    pub(crate) fn cancel_before_start(&self) -> Result<(), i64> {
        loop {
            let view = self.view();
            if view.0 & START_WON != 0 {
                return Err(self.boundary.load(Ordering::Relaxed));
            }
            let next = closed_control(view.0) & !START_ARMED;
            if self
                .control
                .compare_exchange(view.0, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }
}

/// Borrowed callback boundary: destruction performs one atomic operation only.
/// Keep the owning Arc alive outside the device invocation so this guard can
/// never perform the last owner/lifetime destruction on the realtime thread.
pub(crate) struct CallbackGuard<'a> {
    gate: &'a ControlGate,
}

impl CallbackGuard<'_> {
    pub(crate) fn view(&self) -> View {
        self.gate.view()
    }
    pub(crate) fn claim(&self, epoch: u64, source: u64) -> bool {
        self.gate.claim(epoch, source)
    }
    pub(crate) fn publish_current(&self, source: u64) {
        self.gate.publish_current(source);
    }
    pub(crate) fn finish_current(&self) {
        self.gate.finish_current();
    }

    pub(crate) fn acknowledge_reanchor(&self) {
        self.gate
            .control
            .fetch_and(!REANCHORED_TIMELINE, Ordering::AcqRel);
    }

    /// ACTIVE prevents arm/clear/reanchor from changing the paired boundary.
    /// Only close/cancel can defeat the single start CAS; that is closed output,
    /// not generic contention silence or an invitation to retry on the device.
    pub(crate) fn decide_start(&self, presentation: Option<i64>) -> ScheduledStartOutcome {
        let view = self.view();
        if view.0 & START_CLOSED != 0 {
            return ScheduledStartOutcome::Closed;
        }
        let boundary = self.gate.boundary.load(Ordering::Relaxed);
        if view.0 & START_WON != 0 {
            return ScheduledStartOutcome::Started {
                start_at_zone_us: boundary,
            };
        }
        if view.0 & START_ARMED == 0 {
            return ScheduledStartOutcome::Unscheduled;
        }
        if presentation.is_none_or(|at| at < boundary) {
            return ScheduledStartOutcome::Waiting {
                start_at_zone_us: boundary,
            };
        }
        match self.gate.control.compare_exchange(
            view.0,
            (view.0 & !START_ARMED) | START_WON,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => ScheduledStartOutcome::BoundaryWon {
                start_at_zone_us: boundary,
            },
            Err(actual) => {
                debug_assert_ne!(
                    actual & START_CLOSED,
                    0,
                    "only close/cancel may interrupt active start"
                );
                ScheduledStartOutcome::Closed
            }
        }
    }
}

impl Drop for CallbackGuard<'_> {
    fn drop(&mut self) {
        self.gate.end_callback();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_realtime_control_close_preserves_phase_cancel_clears_only_unwon() {
        for boundary in [i64::MIN, 0, i64::MAX] {
            let gate = ControlGate::default();
            assert_eq!(gate.try_arm(boundary), Ok(ScheduledArmOutcome::Armed));
            gate.close();
            assert_eq!(
                gate.start_state(),
                StartState::Armed {
                    start_at_zone_us: boundary
                }
            );
            assert_eq!(gate.cancel_before_start(), Ok(()));
            assert_eq!(gate.start_state(), StartState::Idle);

            let gate = ControlGate::default();
            assert_eq!(gate.try_arm(boundary), Ok(ScheduledArmOutcome::Armed));
            let callback = gate.begin_callback();
            assert_eq!(
                callback.decide_start(Some(boundary)),
                ScheduledStartOutcome::BoundaryWon {
                    start_at_zone_us: boundary
                }
            );
            assert_eq!(gate.cancel_before_start(), Err(boundary));
            drop(callback);
            gate.close();
            assert_eq!(
                gate.start_state(),
                StartState::BoundaryWon {
                    start_at_zone_us: boundary
                }
            );
        }
    }

    #[test]
    fn renderer_realtime_control_observer_never_blocks_callback_or_revives_old_transaction() {
        let gate = ControlGate::default();
        gate.try_arm(10).unwrap();
        let old = gate.observe().unwrap();
        {
            let callback = gate.begin_callback();
            assert!(gate.observe().is_none());
            assert_eq!(
                callback.decide_start(Some(10)),
                ScheduledStartOutcome::BoundaryWon {
                    start_at_zone_us: 10
                }
            );
            callback.publish_current(1 << 40);
        }
        assert!(!gate.validate(&old));
        assert!(gate.clear(old).is_err());
        let fresh = gate.observe().unwrap();
        assert_eq!(fresh.current(), 1 << 40);
        gate.clear(fresh).unwrap();
        assert_eq!(
            gate.start_state(),
            StartState::BoundaryWon {
                start_at_zone_us: 10
            }
        );
    }
}

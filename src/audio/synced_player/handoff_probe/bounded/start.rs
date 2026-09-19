//! Start-authority candidate, not an additional production authority.
//! Non-RT arm/clear calls must be serialized by the existing owner. Callback
//! evaluation makes one CAS attempt; source-control integration remains pending.
use super::*;
use crate::audio::{ScheduledArmOutcome, ScheduledStartOutcome};
use std::sync::atomic::AtomicI64;

const ARMED: u64 = 1;
const WON: u64 = 2;
const CLOSED: u64 = 4;
const KIND: u64 = 3;
const VERSION_STEP: u64 = 8;

#[derive(Default)]
pub(super) struct StartAuthority {
    state: AtomicU64,
    boundary: AtomicI64,
}

impl StartAuthority {
    pub(super) fn arm(&self, boundary: i64) -> ScheduledArmOutcome {
        let state = self.state.load(Ordering::Acquire);
        if state & CLOSED != 0 {
            return ScheduledArmOutcome::Closed;
        }
        let existing = self.boundary.load(Ordering::Relaxed);
        match state & KIND {
            ARMED => {
                return ScheduledArmOutcome::AlreadyArmed {
                    start_at_zone_us: existing,
                }
            }
            WON => {
                return ScheduledArmOutcome::BoundaryAlreadyWon {
                    start_at_zone_us: existing,
                }
            }
            _ => {}
        }
        self.boundary.store(boundary, Ordering::Relaxed);
        let next = state.checked_add(VERSION_STEP).unwrap() | ARMED;
        match self
            .state
            .compare_exchange(state, next, Ordering::Release, Ordering::Acquire)
        {
            Ok(_) => ScheduledArmOutcome::Armed,
            Err(actual) => {
                assert_ne!(actual & CLOSED, 0);
                ScheduledArmOutcome::Closed
            }
        }
    }

    pub(super) fn evaluate(&self, presentation: Option<i64>) -> Result<ScheduledStartOutcome, ()> {
        self.evaluate_after_read(presentation, || {})
    }

    fn evaluate_after_read(
        &self,
        presentation: Option<i64>,
        after_read: impl FnOnce(),
    ) -> Result<ScheduledStartOutcome, ()> {
        let state = self.state.load(Ordering::Acquire);
        if state & CLOSED != 0 {
            return Ok(ScheduledStartOutcome::Closed);
        }
        let boundary = self.boundary.load(Ordering::Relaxed);
        match state & KIND {
            0 => Ok(ScheduledStartOutcome::Unscheduled),
            WON => Ok(ScheduledStartOutcome::Started {
                start_at_zone_us: boundary,
            }),
            ARMED => {
                after_read();
                if presentation.is_none_or(|at| at < boundary) {
                    // Validate paired boundary before returning a frozen value.
                    return (self.state.load(Ordering::Acquire) == state)
                        .then_some(ScheduledStartOutcome::Waiting {
                            start_at_zone_us: boundary,
                        })
                        .ok_or(());
                }
                self.state
                    .compare_exchange(
                        state,
                        (state & !KIND) | WON,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .map(|_| ScheduledStartOutcome::BoundaryWon {
                        start_at_zone_us: boundary,
                    })
                    .map_err(|_| ())
            }
            _ => unreachable!(),
        }
    }

    // Internal claim only; it is never a stream-stopped acknowledgment.
    pub(super) fn cancel(&self) -> Result<(), i64> {
        loop {
            let state = self.state.load(Ordering::Acquire);
            if state & KIND == WON {
                return Err(self.boundary.load(Ordering::Relaxed));
            }
            if self
                .state
                .compare_exchange(state, state | CLOSED, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    pub(super) fn clear_unstarted(&self) {
        loop {
            let state = self.state.load(Ordering::Acquire);
            if state & KIND == WON || state & CLOSED != 0 {
                return;
            }
            let next = (state & !KIND).checked_add(VERSION_STEP).unwrap();
            if self
                .state
                .compare_exchange(state, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }
}

#[test]
fn realtime_handoff_probe_start_authority_matches_real_owner_boundaries_and_clear() {
    use crate::audio::{RendererOperationOutcome, RendererOwner, RendererQueueLimits};
    for boundary in [i64::MIN, -5, 0, i64::MAX] {
        let authority = StartAuthority::default();
        let owner = RendererOwner::new(RendererQueueLimits::new(32, 8, 16).unwrap());
        let scope = owner.mint_scope().unwrap();
        assert_eq!(
            authority.arm(boundary),
            owner.arm_scheduled_start(scope, boundary)
        );
        assert_eq!(authority.arm(123), owner.arm_scheduled_start(scope, 123));
        for presentation in [None, boundary.checked_sub(1), Some(boundary), None] {
            let expected = owner
                .try_callback_permit(scope)
                .unwrap()
                .scheduled_start(presentation);
            assert_eq!(authority.evaluate(presentation), Ok(expected));
        }
        assert_eq!(authority.cancel(), Err(boundary));
        authority.clear_unstarted();
        assert_eq!(owner.clear(scope), RendererOperationOutcome::Applied);
        assert_eq!(authority.arm(123), owner.arm_scheduled_start(scope, 123));
    }
    let authority = StartAuthority::default();
    assert_eq!(authority.arm(100), ScheduledArmOutcome::Armed);
    authority.clear_unstarted();
    assert_eq!(
        authority.evaluate(None),
        Ok(ScheduledStartOutcome::Unscheduled)
    );
    assert_eq!(authority.arm(200), ScheduledArmOutcome::Armed);
    assert_eq!(
        authority.evaluate(Some(100)),
        Ok(ScheduledStartOutcome::Waiting {
            start_at_zone_us: 200
        })
    );
}

#[test]
fn realtime_handoff_probe_start_cancel_and_boundary_have_one_winner() {
    for cancel_first in [false, true] {
        let authority = Arc::new(StartAuthority::default());
        authority.arm(100);
        let other = Arc::clone(&authority);
        let (done_tx, done_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            if cancel_first {
                other.cancel().unwrap();
                done_tx.send(()).unwrap();
            } else {
                assert_eq!(
                    other.evaluate(Some(100)),
                    Ok(ScheduledStartOutcome::BoundaryWon {
                        start_at_zone_us: 100
                    })
                );
                done_tx.send(()).unwrap();
            }
        });
        done_rx.recv().unwrap();
        if cancel_first {
            assert_eq!(
                authority.evaluate(Some(100)),
                Ok(ScheduledStartOutcome::Closed)
            );
        } else {
            assert_eq!(authority.cancel(), Err(100));
        }
        worker.join().unwrap();
    }
}

#[test]
fn realtime_handoff_probe_start_old_boundary_cannot_win_after_clear_and_rearm() {
    let authority = StartAuthority::default();
    authority.arm(100);
    assert_eq!(
        authority.evaluate_after_read(Some(100), || {
            authority.clear_unstarted();
            assert_eq!(authority.arm(200), ScheduledArmOutcome::Armed);
        }),
        Err(())
    );
    assert_eq!(
        authority.evaluate(Some(100)),
        Ok(ScheduledStartOutcome::Waiting {
            start_at_zone_us: 200
        })
    );
    assert_eq!(
        authority.evaluate(Some(200)),
        Ok(ScheduledStartOutcome::BoundaryWon {
            start_at_zone_us: 200
        })
    );
}

// The historical independent authority above remains a differential fixture.
// This candidate stores phase only in ClaimGate; there is no shadow phase.
pub(super) trait StartDecision {
    fn decide(&self, presentation: Option<i64>) -> Result<ScheduledStartOutcome, ()>;
}

impl StartDecision for StartAuthority {
    fn decide(&self, presentation: Option<i64>) -> Result<ScheduledStartOutcome, ()> {
        self.evaluate(presentation)
    }
}

pub(super) struct GateStartAuthority {
    gate: Arc<ClaimGate>,
    boundary: AtomicI64,
}

impl GateStartAuthority {
    pub(super) fn new(gate: Arc<ClaimGate>) -> Self {
        Self {
            gate,
            boundary: AtomicI64::new(0),
        }
    }

    // Serialized non-RT owner operation. Retry belongs to that caller; an
    // in-flight callback never waits for this arm operation to finish.
    pub(super) fn arm(&self, boundary: i64) -> Result<ScheduledArmOutcome, ()> {
        let view = self.gate.view();
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
        self.gate
            .control
            .compare_exchange(
                view.0,
                view.0 | START_ARMED,
                Ordering::Release,
                Ordering::Acquire,
            )
            .map(|_| ScheduledArmOutcome::Armed)
            .map_err(|_| ())
    }

    // Non-RT caller holds the same owner serialization used by arm/clear.
    // Boundary storage cannot change under that serialization; callback may
    // change Armed to Won, so the phase load is the snapshot linearization point.
    // No second phase, preparation backfill, or callback completion is needed.
    pub(super) fn snapshot(&self) -> crate::audio::StartState {
        use crate::audio::StartState;
        let view = self.gate.view();
        let boundary = self.boundary.load(Ordering::Relaxed);
        if view.0 & START_WON != 0 {
            StartState::BoundaryWon {
                start_at_zone_us: boundary,
            }
        } else if view.0 & START_CLOSED != 0 || view.0 & START_ARMED == 0 {
            StartState::Idle
        } else {
            StartState::Armed {
                start_at_zone_us: boundary,
            }
        }
    }

    // Claim only. Resource stop/join and the public Ack remain the owner's job.
    pub(super) fn cancel(&self) -> Result<(), i64> {
        loop {
            let view = self.gate.view();
            if view.0 & START_WON != 0 {
                return Err(self.boundary.load(Ordering::Relaxed));
            }
            if self
                .gate
                .control
                .compare_exchange(
                    view.0,
                    view.0 | START_CLOSED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
    }
}

impl StartDecision for GateStartAuthority {
    fn decide(&self, presentation: Option<i64>) -> Result<ScheduledStartOutcome, ()> {
        self.decide_after_read(presentation, || {})
    }
}
impl GateStartAuthority {
    pub(super) fn decide_after_read(
        &self,
        presentation: Option<i64>,
        after_read: impl FnOnce(),
    ) -> Result<ScheduledStartOutcome, ()> {
        let view = self.gate.view();
        after_read();
        assert!(view.callback_active());
        if view.0 & START_CLOSED != 0 {
            return Ok(ScheduledStartOutcome::Closed);
        }
        let boundary = self.boundary.load(Ordering::Relaxed);
        if view.0 & START_WON != 0 {
            return Ok(ScheduledStartOutcome::Started {
                start_at_zone_us: boundary,
            });
        }
        if view.0 & START_ARMED == 0 {
            return Ok(ScheduledStartOutcome::Unscheduled);
        }
        if presentation.is_none_or(|at| at < boundary) {
            return Ok(ScheduledStartOutcome::Waiting {
                start_at_zone_us: boundary,
            });
        }
        // ACTIVE excludes arm/clear/reanchor. Only cancellation can defeat this
        // single attempt; the caller then emits legitimate cancelled silence.
        self.gate
            .control
            .compare_exchange(
                view.0,
                (view.0 & !START_ARMED) | START_WON,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ScheduledStartOutcome::BoundaryWon {
                start_at_zone_us: boundary,
            })
            .map_err(|_| ())
    }
}

#[test]
fn realtime_handoff_probe_gate_start_preserves_full_boundary_and_cancel_winner() {
    for boundary in [i64::MIN, -5, 0, i64::MAX] {
        for cancel_first in [false, true] {
            let gate = Arc::new(ClaimGate::default());
            let start = GateStartAuthority::new(Arc::clone(&gate));
            assert_eq!(start.arm(boundary), Ok(ScheduledArmOutcome::Armed));
            gate.begin_callback();
            // A control arm cannot modify boundary storage while it is read.
            assert_eq!(start.arm(123), Err(()));
            if cancel_first {
                start.cancel().unwrap();
                assert_eq!(
                    start.decide(Some(boundary)),
                    Ok(ScheduledStartOutcome::Closed)
                );
            } else {
                assert_eq!(
                    start.decide(None),
                    Ok(ScheduledStartOutcome::Waiting {
                        start_at_zone_us: boundary
                    })
                );
                assert_eq!(
                    start.decide(Some(boundary)),
                    Ok(ScheduledStartOutcome::BoundaryWon {
                        start_at_zone_us: boundary
                    })
                );
                assert_eq!(start.cancel(), Err(boundary));
                assert_eq!(
                    start.decide(None),
                    Ok(ScheduledStartOutcome::Started {
                        start_at_zone_us: boundary
                    })
                );
            }
            gate.end_callback();
            gate.clear(gate.observe().unwrap()).unwrap();
            assert_eq!(
                start.arm(123),
                Ok(if cancel_first {
                    ScheduledArmOutcome::Closed
                } else {
                    ScheduledArmOutcome::BoundaryAlreadyWon {
                        start_at_zone_us: boundary,
                    }
                })
            );
        }
    }
}

#[test]
fn realtime_handoff_probe_gate_start_snapshot_matches_owner_without_callback_backfill() {
    use crate::audio::{RendererOwner, RendererQueueLimits, StartState};
    for boundary in [i64::MIN, -1, 0, i64::MAX] {
        let gate = Arc::new(ClaimGate::default());
        let start = GateStartAuthority::new(Arc::clone(&gate));
        let owner = RendererOwner::new(RendererQueueLimits::new(32, 8, 16).unwrap());
        let scope = owner.mint_scope().unwrap();
        assert_eq!(start.snapshot(), owner.start_state(scope).unwrap());
        assert_eq!(
            start.arm(boundary),
            Ok(owner.arm_scheduled_start(scope, boundary))
        );
        assert_eq!(start.snapshot(), owner.start_state(scope).unwrap());
        gate.begin_callback();
        assert_eq!(
            start.decide(None),
            Ok(owner
                .try_callback_permit(scope)
                .unwrap()
                .scheduled_start(None))
        );
        assert_eq!(
            start.snapshot(),
            StartState::Armed {
                start_at_zone_us: boundary
            }
        );
        assert_eq!(
            start.decide(Some(boundary)),
            Ok(owner
                .try_callback_permit(scope)
                .unwrap()
                .scheduled_start(Some(boundary)))
        );
        // Read before candidate callback completion and before any preparation
        // reconciliation: the unique published authority already reports Won.
        assert_eq!(start.snapshot(), owner.start_state(scope).unwrap());
        assert_eq!(
            start.snapshot(),
            StartState::BoundaryWon {
                start_at_zone_us: boundary
            }
        );
        gate.end_callback();
        gate.clear(gate.observe().unwrap()).unwrap();
        owner.clear(scope);
        assert_eq!(start.snapshot(), owner.start_state(scope).unwrap());
    }
    let gate = Arc::new(ClaimGate::default());
    let start = GateStartAuthority::new(Arc::clone(&gate));
    start.arm(100).unwrap();
    gate.clear(gate.observe().unwrap()).unwrap();
    assert_eq!(start.snapshot(), StartState::Idle);
    start.arm(200).unwrap();
    assert_eq!(
        start.snapshot(),
        StartState::Armed {
            start_at_zone_us: 200
        }
    );
    start.cancel().unwrap();
    assert_eq!(
        start.snapshot(),
        StartState::Idle,
        "internal CLOSED is not a new public phase"
    );
}

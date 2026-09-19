//! Width and interference regressions for the same ClaimGate used by Tape.
//! No parallel representation or second control authority lives here.
use super::start::{GateStartAuthority, StartDecision};
use super::*;
use crate::audio::{ScheduledArmOutcome, ScheduledStartOutcome};

#[test]
fn realtime_handoff_probe_wide_gate_rejects_complete_callback_aba_and_partial_position() {
    let gate = ClaimGate::default();
    let stale = gate.observe().unwrap();
    gate.begin_callback();
    gate.publish_current(0);
    gate.end_callback();
    assert!(!gate.validate(&stale));
    // A second reader cannot clear DIRTY and revive this still-live transaction.
    assert!(gate.observe().is_none());
    assert!(gate.replace(stale).is_err());
    let before = gate.observe().unwrap();
    gate.begin_callback();
    gate.publish_current(1 << 40);
    assert!(!gate.validate(&before));
    assert!(gate.replace(before).is_err());
    assert!(gate.observe().is_none());
    gate.end_callback();
    let after = gate.observe().unwrap();
    assert_eq!(after.current(), 1 << 40);
    assert!(gate.replace(after).is_ok());
    gate.begin_callback();
    gate.publish_current(1 << 14);
    gate.end_callback();
    let after = gate.observe().unwrap();
    assert_eq!(after.current(), 1 << 14);
    assert!(!after.current_revoked());
    assert!(!after.timeline_reanchored());
}

#[test]
fn realtime_handoff_probe_wide_gate_clear_keeps_start_winner_in_same_commit() {
    for win in [false, true] {
        let gate = Arc::new(ClaimGate::default());
        let start = GateStartAuthority::new(Arc::clone(&gate));
        assert_eq!(start.arm(i64::MAX), Ok(ScheduledArmOutcome::Armed));
        let old = gate.observe().unwrap();
        gate.begin_callback();
        if win {
            assert_eq!(
                start.decide(Some(i64::MAX)),
                Ok(ScheduledStartOutcome::BoundaryWon {
                    start_at_zone_us: i64::MAX
                })
            );
        }
        gate.publish_current(1 << 40);
        gate.end_callback();
        assert!(gate.clear(old).is_err());
        gate.clear(gate.observe().unwrap()).unwrap();
        let view = gate.view();
        assert_eq!(view.0 & START_ARMED, 0);
        assert_eq!(view.0 & START_WON != 0, win);
        assert!(view.current_revoked());
        assert_eq!(view.current(), 1 << 40);
        assert_eq!(view.epoch(), 1);
    }
}

#[test]
fn realtime_handoff_probe_wide_gate_observer_cannot_defeat_inflight_start() {
    let gate = Arc::new(ClaimGate::default());
    let start = GateStartAuthority::new(Arc::clone(&gate));
    assert_eq!(start.arm(123), Ok(ScheduledArmOutcome::Armed));
    gate.begin_callback();
    assert_eq!(
        start.decide_after_read(Some(123), || {
            assert!(gate.observe().is_none());
        }),
        Ok(ScheduledStartOutcome::BoundaryWon {
            start_at_zone_us: 123
        })
    );
    gate.end_callback();
}

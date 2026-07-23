use sendspin::audio::{
    EnqueueOutcome, PlayerScope, PreStartAbortOutcome, RendererFault, RendererOperationOutcome,
    RendererOwner, RendererQueueLimits, RendererTerminal, ScheduledArmOutcome, TerminalOutcome,
};

fn owner() -> RendererOwner {
    RendererOwner::new(RendererQueueLimits::new(8, 2, 6).expect("valid limits"))
}

fn scope(owner: &RendererOwner) -> PlayerScope {
    owner.mint_scope().expect("scope available")
}

#[test]
fn bounded_enqueue_rejects_before_mutation() {
    let owner = owner();
    let scope = scope(&owner);

    assert_eq!(
        owner.enqueue(scope, 4),
        EnqueueOutcome::Accepted {
            queued_frames: 4,
            queued_buffers: 1,
        }
    );
    let before = owner.capacity(scope).expect("current scope");

    assert_eq!(
        owner.enqueue(scope, 7),
        EnqueueOutcome::Full {
            queued_frames: 4,
            queued_buffers: 1,
        }
    );
    assert_eq!(owner.capacity(scope).expect("current scope"), before);

    assert_eq!(
        owner.enqueue(scope, 5),
        EnqueueOutcome::Full {
            queued_frames: 4,
            queued_buffers: 1,
        }
    );
    assert_eq!(owner.capacity(scope).expect("current scope"), before);

    assert_eq!(
        owner.enqueue(scope, 4),
        EnqueueOutcome::Accepted {
            queued_frames: 8,
            queued_buffers: 2,
        }
    );
    let full = owner.capacity(scope).expect("current scope");
    assert_eq!(
        owner.enqueue(scope, 1),
        EnqueueOutcome::Full {
            queued_frames: 8,
            queued_buffers: 2,
        }
    );
    assert_eq!(owner.capacity(scope).expect("current scope"), full);
}

#[test]
fn checked_frame_add_overflow_rejects_before_mutation() {
    let owner = RendererOwner::new(
        RendererQueueLimits::new(usize::MAX, 3, usize::MAX).expect("valid wide limits"),
    );
    let scope = scope(&owner);
    assert!(matches!(
        owner.enqueue(scope, usize::MAX - 1),
        EnqueueOutcome::Accepted { .. }
    ));
    let before = owner.capacity(scope).unwrap();
    assert!(matches!(
        owner.enqueue(scope, 2),
        EnqueueOutcome::Full { .. }
    ));
    assert_eq!(owner.capacity(scope).unwrap(), before);
}

#[test]
fn buffer_limit_rejects_while_frame_capacity_remains() {
    let owner = RendererOwner::new(RendererQueueLimits::new(100, 2, 50).unwrap());
    let scope = scope(&owner);
    assert!(matches!(
        owner.enqueue(scope, 1),
        EnqueueOutcome::Accepted { .. }
    ));
    assert!(matches!(
        owner.enqueue(scope, 1),
        EnqueueOutcome::Accepted { .. }
    ));
    let before = owner.capacity(scope).unwrap();
    assert_eq!(before.current_frames(), 2);
    assert_eq!(before.current_buffers(), 2);
    assert!(matches!(
        owner.enqueue(scope, 1),
        EnqueueOutcome::Full { .. }
    ));
    assert_eq!(owner.capacity(scope).unwrap(), before);
}

#[test]
fn close_and_stale_scope_fence_enqueue_and_consume() {
    let owner = owner();
    let first = scope(&owner);
    assert!(matches!(
        owner.enqueue(first, 4),
        EnqueueOutcome::Accepted { .. }
    ));
    assert_eq!(owner.close(first), RendererOperationOutcome::Applied);
    assert_eq!(owner.enqueue(first, 1), EnqueueOutcome::Closed);
    assert_eq!(owner.consume(first, 1), RendererOperationOutcome::Closed);

    let _ = owner.teardown(first);

    let second = scope(&owner);
    let second_before = owner.health(second).unwrap();
    assert_eq!(owner.enqueue(first, 1), EnqueueOutcome::StaleScope);
    assert_eq!(
        owner.consume(first, 1),
        RendererOperationOutcome::StaleScope
    );
    assert_eq!(
        owner.health(first),
        Err(RendererOperationOutcome::StaleScope)
    );
    assert_eq!(owner.clear(first), RendererOperationOutcome::StaleScope);
    assert_eq!(owner.close(first), RendererOperationOutcome::StaleScope);
    assert_eq!(
        owner.fault(first, RendererFault::OutputInvalidated),
        RendererOperationOutcome::StaleScope
    );
    assert!(matches!(owner.teardown(first), TerminalOutcome::StaleScope));
    assert_eq!(owner.health(second).unwrap(), second_before);
    assert!(matches!(
        owner.enqueue(second, 2),
        EnqueueOutcome::Accepted { .. }
    ));
}

#[test]
fn health_and_capacity_are_owner_generated_and_complete() {
    let owner = owner();
    let scope = scope(&owner);
    assert!(matches!(
        owner.enqueue(scope, 6),
        EnqueueOutcome::Accepted { .. }
    ));
    assert_eq!(
        owner.record_callback(scope, 2, 3, Some(55_000)),
        RendererOperationOutcome::Applied
    );

    let capacity = owner.capacity(scope).expect("current scope");
    assert_eq!(capacity.hard_frames(), 8);
    assert_eq!(capacity.hard_buffers(), 2);
    assert_eq!(capacity.current_frames(), 4);
    assert_eq!(capacity.current_buffers(), 1);
    assert_eq!(capacity.high_water_frames(), 6);
    assert_eq!(capacity.high_water_buffers(), 1);

    let health = owner.health(scope).expect("current scope");
    assert_eq!(health.scope(), scope);
    assert_eq!(health.queued_frames(), 4);
    assert_eq!(health.queued_buffers(), 1);
    assert_eq!(health.limits(), RendererQueueLimits::new(8, 2, 6).unwrap());
    assert_eq!(health.consumed_frames(), 2);
    assert_eq!(health.callback_count(), 1);
    assert_eq!(health.underrun_frames(), 3);
    assert_eq!(health.last_presentation_boundary_zone_us(), Some(55_000));
    assert_eq!(health.fault(), None);
    assert_eq!(health.terminal(), None);
}

#[test]
fn clear_fault_and_terminal_snapshots_are_typed() {
    let owner = owner();
    let scope = scope(&owner);
    assert!(matches!(
        owner.enqueue(scope, 4),
        EnqueueOutcome::Accepted { .. }
    ));
    assert_eq!(owner.clear(scope), RendererOperationOutcome::Applied);
    let cleared = owner.capacity(scope).unwrap();
    assert_eq!(
        (cleared.current_frames(), cleared.current_buffers()),
        (0, 0)
    );
    assert_eq!(
        owner.fault(scope, RendererFault::CallbackFailed),
        RendererOperationOutcome::Applied
    );
    assert_eq!(owner.enqueue(scope, 1), EnqueueOutcome::Closed);

    let finalized = owner.teardown(scope);
    let TerminalOutcome::Won(finalization) = finalized else {
        panic!("first finalizer must win: {finalized:?}");
    };
    assert!(!finalization.ack.stream_released());
    assert!(!finalization.ack.callback_stopped());
    assert!(matches!(
        owner.terminal_state(scope).unwrap(),
        sendspin::audio::TerminalState::Finalized { .. }
    ));
    assert_eq!(
        owner.health(scope).unwrap().terminal(),
        Some(RendererTerminal::Faulted(RendererFault::CallbackFailed))
    );
}

#[test]
fn output_contract_open_fault_is_visible_and_closes_acceptance() {
    let fault_owner = owner();
    let fault_scope = scope(&fault_owner);
    assert_eq!(
        fault_owner.fault(fault_scope, RendererFault::OutputContractRejected),
        RendererOperationOutcome::Applied
    );
    assert_eq!(fault_owner.enqueue(fault_scope, 1), EnqueueOutcome::Closed);
    assert_eq!(
        fault_owner.health(fault_scope).unwrap().fault(),
        Some(RendererFault::OutputContractRejected)
    );
    assert_eq!(
        fault_owner.fault(fault_scope, RendererFault::CallbackFailed),
        RendererOperationOutcome::Closed
    );
    assert_eq!(
        fault_owner.health(fault_scope).unwrap().fault(),
        Some(RendererFault::OutputContractRejected)
    );

    let closed_owner = owner();
    let closed_scope = scope(&closed_owner);
    assert_eq!(
        closed_owner.close(closed_scope),
        RendererOperationOutcome::Applied
    );
    assert_eq!(
        closed_owner.fault(closed_scope, RendererFault::CallbackFailed),
        RendererOperationOutcome::Closed
    );
    assert_eq!(closed_owner.health(closed_scope).unwrap().fault(), None);
}

#[test]
fn pre_start_abort_clears_armed_accounting() {
    let owner = owner();
    let abort_scope = scope(&owner);
    assert_eq!(
        owner.arm_scheduled_start(abort_scope, 20_000),
        ScheduledArmOutcome::Armed
    );
    assert!(matches!(
        owner.enqueue(abort_scope, 4),
        EnqueueOutcome::Accepted { .. }
    ));

    let PreStartAbortOutcome::Won(finalization) = owner.abort_before_start(abort_scope) else {
        panic!("pre-start abort must win on its independent scope")
    };
    assert_eq!(
        finalization.winner,
        sendspin::audio::TerminalWinner::PreStartAbort
    );
    let capacity = owner.capacity(abort_scope).unwrap();
    assert_eq!(capacity.current_frames(), 0);
    assert_eq!(capacity.current_buffers(), 0);
}

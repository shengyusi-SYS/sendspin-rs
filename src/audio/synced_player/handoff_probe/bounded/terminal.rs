//! Compatibility experiment: real RendererOwner finalization owns a joining
//! reader resource. The candidate getter reads the device's scoped publisher.
//! This does not replace the production getter or emulate a CPAL stream.

use super::timeline::{Preparation, Span};
use super::*;
use crate::audio::{
    PlayerScope, RendererOperationOutcome, RendererOwner, RendererQueueLimits, TerminalOutcome,
    TerminalState,
};

struct ScopedConsumption {
    owner: RendererOwner,
    scope: PlayerScope,
    actual: Arc<AtomicU64>,
}

impl ScopedConsumption {
    fn read(&self, requested: PlayerScope) -> Result<u64, RendererOperationOutcome> {
        if requested != self.scope {
            return Err(RendererOperationOutcome::StaleScope);
        }
        // Only reuse scope validation; the returned legacy tally is deliberately
        // not the candidate's consumption source. No test-only owner setter.
        self.owner.consumed_frames(requested)?;
        Ok(self.actual.load(Ordering::Acquire))
    }
}

struct JoiningReader {
    owner: RendererOwner,
    scope: PlayerScope,
    resume: mpsc::Sender<()>,
    reader: Option<thread::JoinHandle<(DeviceSide<Span>, [i32; 3])>>,
    stop_preparation: mpsc::Sender<()>,
    preparation: Option<thread::JoinHandle<PrepareSide<Span>>>,
    observed: mpsc::Sender<(TerminalState, [i32; 3])>,
}

impl Drop for JoiningReader {
    fn drop(&mut self) {
        let state = self.owner.terminal_state(self.scope).unwrap();
        self.stop_preparation.send(()).unwrap();
        self.resume.send(()).unwrap();
        let prepare = self.preparation.take().unwrap().join().unwrap();
        let (device, output) = self.reader.take().unwrap().join().unwrap();
        drop(device); // Reader has stopped; remaining payloads die on finalizer.
        drop(prepare);
        self.observed.send((state, output)).unwrap();
    }
}

#[test]
fn realtime_handoff_probe_terminal_join_precedes_final_source_count_and_ack() {
    let (mut prepare, mut device) = pipe_for::<Span>(1);
    let owner = RendererOwner::new(RendererQueueLimits::new(32, 8, 16).unwrap());
    let scope = owner.mint_scope().unwrap();
    let consumption = ScopedConsumption {
        owner: owner.clone(),
        scope,
        actual: Arc::clone(&device.progress),
    };
    let (published_tx, published_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel();
    let (dropped_tx, dropped_rx) = mpsc::channel();
    let preparation = thread::spawn(move || {
        let mut second = source_pcm(&[30, 40]);
        second.timestamp = 2_000;
        let mut cursor = Preparation::new(vec![(1, source_pcm(&[10, 20])), (2, second)]);
        let mut span = cursor.prepare(0, &[1, 0, 2, 1]);
        span.lifetime = Some(DropWitness(dropped_tx));
        assert!(prepare.publish(span).is_ok());
        published_tx.send(()).unwrap();
        stop_rx.recv().unwrap();
        prepare
    });
    published_rx.recv().unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let (observed_tx, observed_rx) = mpsc::channel();
    let reader = thread::spawn(move || {
        device.gate.begin_callback();
        ready_tx.send(()).unwrap();
        resume_rx.recv().unwrap();
        let mut output = [0; 3];
        assert_eq!(device.render_span_active(&mut output, 1), 0);
        device.gate.end_callback();
        (device, output)
    });
    ready_rx.recv().unwrap();
    assert_eq!(
        owner.attach_test_terminal_resource(
            scope,
            Box::new(JoiningReader {
                owner: owner.clone(),
                scope,
                resume: resume_tx,
                reader: Some(reader),
                stop_preparation: stop_tx,
                preparation: Some(preparation),
                observed: observed_tx,
            })
        ),
        RendererOperationOutcome::Applied
    );
    assert_eq!(consumption.read(scope), Ok(0));
    let outcome = owner.teardown(scope);
    let (during_drop, output) = observed_rx.recv().unwrap();
    assert!(matches!(during_drop, TerminalState::Finalizing { .. }));
    assert_eq!(output, [10, 10, 30]);
    let TerminalOutcome::Won(finalization) = outcome else {
        panic!("first finalizer wins")
    };
    assert!(finalization.ack.callback_stopped());
    assert!(finalization.ack.stream_released());
    assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
    assert_eq!(consumption.read(scope), Ok(3));
    assert_eq!(
        owner.consumed_frames(scope),
        Ok(0),
        "legacy tally is not the device publisher"
    );
    assert_eq!(
        owner.teardown(scope),
        TerminalOutcome::AlreadyFinalized(finalization)
    );
    assert_eq!(consumption.read(scope), Ok(3));
    let replacement = owner.mint_scope().unwrap();
    assert_eq!(
        consumption.read(scope),
        Err(RendererOperationOutcome::StaleScope)
    );
    assert_eq!(
        consumption.read(replacement),
        Err(RendererOperationOutcome::StaleScope)
    );
}

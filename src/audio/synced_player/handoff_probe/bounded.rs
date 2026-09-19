//! Actual SPSC transport experiment. Credits bound queued/current/retired objects;
//! frame/byte bounds and canonical admission accounting are not yet implemented.

use super::*;
use rtrb::{Consumer, Producer, PushError, RingBuffer};

#[path = "bounded/admission.rs"]
mod admission;

#[path = "bounded/terminal.rs"]
mod terminal;

#[path = "bounded/start.rs"]
mod start;

#[path = "wide_gate.rs"]
mod wide_gate;

#[path = "timeline.rs"]
mod timeline;

struct Credit;

struct Slot<T = PreparedSource> {
    credit: Credit,
    prepared: T,
}

struct PrepareSide<T = PreparedSource> {
    free: Vec<Credit>,
    ready: Producer<Slot<T>>,
    retired: Consumer<Slot<T>>,
    limit: usize,
}

struct DeviceSide<T = PreparedSource> {
    ready: Consumer<Slot<T>>,
    retired: Producer<Slot<T>>,
    gate: Arc<ClaimGate>,
    progress: Arc<AtomicU64>,
    offset: usize,
    claimed: bool,
    pending_return: Option<Slot<T>>,
}

fn pipe(limit: usize) -> (PrepareSide, DeviceSide) {
    pipe_for(limit)
}

fn pipe_for<T>(limit: usize) -> (PrepareSide<T>, DeviceSide<T>) {
    let (ready_tx, ready_rx) = RingBuffer::new(limit);
    let (retired_tx, retired_rx) = RingBuffer::new(limit);
    (
        PrepareSide {
            free: (0..limit).map(|_| Credit).collect(),
            ready: ready_tx,
            retired: retired_rx,
            limit,
        },
        DeviceSide {
            ready: ready_rx,
            retired: retired_tx,
            gate: Arc::new(ClaimGate::default()),
            progress: Arc::new(AtomicU64::new(0)),
            offset: 0,
            claimed: false,
            pending_return: None,
        },
    )
}

impl<T> PrepareSide<T> {
    // Preparation backpressure, NOT the public source enqueue outcome.
    fn publish(&mut self, prepared: T) -> Result<(), T> {
        let Some(credit) = self.free.pop() else {
            return Err(prepared);
        };
        match self.ready.push(Slot { credit, prepared }) {
            Ok(()) => Ok(()),
            Err(PushError::Full(slot)) => {
                self.free.push(slot.credit);
                Err(slot.prepared)
            }
        }
    }

    fn reclaim(&mut self) -> usize {
        let mut reclaimed = 0;
        for _ in 0..self.limit {
            let Ok(slot) = self.retired.pop() else { break };
            let Slot { credit, prepared } = slot;
            drop(prepared); // Non-RT owner executes final payload/lifetime drop.
            self.free.push(credit);
            reclaimed += 1;
        }
        reclaimed
    }
}

impl<T> DeviceSide<T> {
    fn return_slot(&mut self) -> bool {
        let Some(slot) = self.pending_return.take() else {
            return true;
        };
        match self.retired.push(slot) {
            Ok(()) => true,
            Err(PushError::Full(slot)) => {
                // Preserve ownership even if the credit invariant is broken.
                // Never drop the returned payload in the device callback.
                self.pending_return = Some(slot);
                false
            }
        }
    }
}

impl DeviceSide {
    fn render(&mut self, output: &mut [i32]) -> usize {
        self.gate.begin_callback();
        let missing = self.render_active(output);
        self.gate.end_callback();
        missing
    }

    fn render_active(&mut self, output: &mut [i32]) -> usize {
        output.fill(0);
        if !self.return_slot() || output.is_empty() {
            return output.len();
        }
        // Do not drain an unbounded stream of concurrently published work.
        let visible = self.ready.slots();
        let mut written = 0;
        for _ in 0..visible {
            if written == output.len() {
                break;
            }
            let Ok(slot) = self.ready.peek() else { break };
            let mut reader = PreparedConsumer {
                gate: &self.gate,
                prepared: &slot.prepared,
                progress: &self.progress,
                offset: self.offset,
                claimed: self.claimed,
            };
            let remaining = output.len() - written;
            let missing = reader.render_active(&mut output[written..]);
            written += remaining - missing;
            self.offset = reader.offset;
            self.claimed = reader.claimed;
            let done = self.offset == slot.prepared.frames.len();
            let revoked = !self.claimed && slot.prepared.epoch != self.gate.view().epoch();
            if !done && !revoked {
                break;
            }
            // The only consumer just peeked this entry; publication cannot remove it.
            self.pending_return = self.ready.pop().ok();
            self.offset = 0;
            self.claimed = false;
            if !self.return_slot() {
                break;
            }
        }
        output.len() - written
    }
}

#[test]
fn realtime_handoff_probe_partial_callback_invalidates_admission_snapshot() {
    let (mut prepare, mut device) = pipe(1);
    let prepared = PreparedSource::prepare(&source_pcm(&[1, 2, 3, 4]), 1, 0, &[1; 4]);
    assert!(prepare.publish(prepared).is_ok());
    let mut output = [0];
    assert_eq!(device.render(&mut output), 0);
    assert_eq!(output, [1]);
    let observation_gate = Arc::clone(&device.gate);
    let before_partial_consumption = observation_gate.observe().unwrap();
    assert_eq!(device.render(&mut output), 0);
    assert_eq!(output, [2]);
    assert!(device.gate.replace(before_partial_consumption).is_err());
    assert!(device.gate.replace(device.gate.observe().unwrap()).is_ok());
    let mut tail = [0; 2];
    assert_eq!(device.render(&mut tail), 0);
    assert_eq!(tail, [3, 4]);
    assert_eq!(prepare.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_inflight_read_prevents_clear_commit_without_blocking_pcm() {
    let (mut prepare, mut device) = pipe(1);
    let prepared = PreparedSource::prepare(&source_pcm(&[10, 20]), 1, 0, &[1; 2]);
    assert!(prepare.publish(prepared).is_ok());
    let before = device.gate.observe().unwrap();
    device.gate.begin_callback();
    assert!(device.gate.clear(before).is_err());
    assert!(device.gate.observe().is_none());
    assert!(device.gate.observe().is_none());
    let mut output = [0];
    assert_eq!(device.render_active(&mut output), 0);
    assert_eq!(output, [10]);
    device.gate.end_callback();
    assert!(device.gate.clear(device.gate.observe().unwrap()).is_ok());
    assert_eq!(device.render(&mut output), 1);
    assert_eq!(output, [0]);
    assert_eq!(device.progress.load(Ordering::Acquire), 1);
    assert_eq!(prepare.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_bounded_retirement_and_resume_do_not_drop_on_device() {
    let (mut prepare, mut device) = pipe(2);
    let (dropped_tx, dropped_rx) = mpsc::channel();
    for (id, samples) in [(1, [11, 12]), (2, [21, 22])] {
        let mut data = PreparedSource::prepare(&source_pcm(&samples), id, 0, &[1, 1]);
        data.lifetime = Some(DropWitness(dropped_tx.clone()));
        assert!(prepare.publish(data).is_ok());
    }
    let device_thread = thread::spawn(move || {
        let mut output = [0; 5];
        let missing = device.render(&mut output);
        (device, output, missing)
    });
    let (mut device, output, missing) = device_thread.join().unwrap();
    assert_eq!(output, [11, 12, 21, 22, 0]);
    assert_eq!(missing, 1);
    assert_eq!(device.progress.load(Ordering::Acquire), 4);
    assert!(device.pending_return.is_none());
    assert!(dropped_rx.try_recv().is_err());
    assert_eq!(prepare.retired.slots(), 2);
    assert!(prepare.free.is_empty());
    assert_eq!(prepare.reclaim(), 2);
    assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
    assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
    assert_eq!(prepare.free.len(), 2);
    let next = PreparedSource::prepare(&source_pcm(&[31, 32]), 3, 0, &[1, 1]);
    assert!(prepare.publish(next).is_ok());
    let mut resumed = [0; 2];
    assert_eq!(device.render(&mut resumed), 0);
    assert_eq!(resumed, [31, 32]);
    assert_eq!(device.progress.load(Ordering::Acquire), 6);
    assert_eq!(prepare.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_bounded_revoked_slot_does_not_block_new_pcm() {
    let (mut prepare, mut device) = pipe(2);
    let old = PreparedSource::prepare(&source_pcm(&[1, 2]), 1, 0, &[1, 1]);
    assert!(prepare.publish(old).is_ok());
    device.gate.replace(device.gate.observe().unwrap()).unwrap();
    let replacement = PreparedSource::prepare(&source_pcm(&[20, 21]), 2, 1, &[1, 1]);
    assert!(prepare.publish(replacement).is_ok());
    let mut output = [0; 2];
    assert_eq!(device.render(&mut output), 0);
    assert_eq!(output, [20, 21]);
    assert_eq!(device.progress.load(Ordering::Acquire), 2);
    assert_eq!(prepare.reclaim(), 2);
}

#[test]
fn realtime_handoff_probe_bounded_clear_discards_current_tail_and_pending() {
    let (mut prepare, mut device) = pipe(3);
    for (id, samples) in [(1, [1, 2, 3]), (2, [11, 12, 13])] {
        let data = PreparedSource::prepare(&source_pcm(&samples), id, 0, &[1, 1, 1]);
        assert!(prepare.publish(data).is_ok());
    }
    let mut first = [0];
    assert_eq!(device.render(&mut first), 0);
    assert_eq!(first, [1]);
    device.gate.clear(device.gate.observe().unwrap()).unwrap();
    let new = PreparedSource::prepare(&source_pcm(&[30, 31]), 3, 1, &[1, 1]);
    assert!(prepare.publish(new).is_ok());
    let mut output = [0; 2];
    assert_eq!(device.render(&mut output), 0);
    assert_eq!(output, [30, 31]);
    assert_eq!(device.progress.load(Ordering::Acquire), 3);
    assert_eq!(prepare.reclaim(), 3);
    assert_eq!(device.gate.view().current(), 0);
    assert!(!device.gate.view().current_revoked());
}

#[test]
fn realtime_handoff_probe_bounded_quiescent_drop_needs_no_future_render() {
    let (mut prepare, mut device) = pipe(2);
    let (dropped_tx, dropped_rx) = mpsc::channel();
    for id in [1, 2] {
        let mut data = PreparedSource::prepare(&source_pcm(&[1, 2]), id, 0, &[1, 1]);
        data.lifetime = Some(DropWitness(dropped_tx.clone()));
        assert!(prepare.publish(data).is_ok());
    }
    let device_thread = thread::spawn(move || {
        let mut one = [0];
        device.render(&mut one);
        device
    });
    let device = device_thread.join().unwrap(); // Proves this experiment's reader stopped.
    assert_eq!(device.progress.load(Ordering::Acquire), 1);
    device.gate.clear(device.gate.observe().unwrap()).unwrap();
    assert!(dropped_rx.try_recv().is_err());
    // Owner disposes both endpoints after joining the reader. No callback is
    // invoked to drain invalid PCM, and this is not a CPAL TerminalAck proof.
    drop(device);
    drop(prepare);
    assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
    assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
}

#[test]
fn realtime_handoff_probe_pcm_drains_while_source_queue_owner_is_paused() {
    let mut source_queue = PlaybackQueue::new();
    source_queue.push(source_pcm(&[11, 12, 13, 14]));
    let prepared = PreparedSource::prepare(&source_queue.queue[0].buffer, 1, 0, &[1, 1, 1, 1]);
    assert_eq!(source_queue.queued_frames(1), 4);
    let source_queue = Arc::new(parking_lot::Mutex::new(source_queue));
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let producer_queue = Arc::clone(&source_queue);
    let producer = thread::spawn(move || {
        let _held = producer_queue.lock();
        locked_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    locked_rx.recv().unwrap();
    let (mut prepare, mut device) = pipe(1);
    assert!(prepare.publish(prepared).is_ok());
    let mut output = [0; 6];
    let missing = device.render(&mut output);
    release_tx.send(()).unwrap();
    producer.join().unwrap();
    assert_eq!(missing, 2);
    assert_eq!(output, [11, 12, 13, 14, 0, 0]);
    assert_eq!(device.progress.load(Ordering::Acquire), 4);
    // The integration must later reconcile this from actual progress; preparation
    // and rendering this immutable view have not silently consumed the source queue.
    assert_eq!(source_queue.lock().queued_frames(1), 4);
}

#[test]
fn realtime_handoff_probe_clear_empty_then_claim_preserves_new_source_tail() {
    let (mut prepare, mut device) = pipe(1);
    let cleared = device.gate.clear(device.gate.observe().unwrap()).unwrap();
    let prepared = PreparedSource::prepare(&source_pcm(&[10, 20]), 1, cleared.epoch(), &[1, 1]);
    assert!(prepare.publish(prepared).is_ok());
    let mut output = [0];
    assert_eq!(device.render(&mut output), 0);
    assert_eq!(output, [10]);
    assert_eq!(device.render(&mut output), 0);
    assert_eq!(output, [20]);
    assert_eq!(device.progress.load(Ordering::Acquire), 2);
}

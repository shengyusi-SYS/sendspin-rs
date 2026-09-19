//! B-candidate feasibility probe, not a production callback implementation.
//! The single device writer publishes wide source identity under ACTIVE. A
//! serialized non-RT observation stays dirty after any intervening callback;
//! clear, reanchor and start decisions share one wide-version control word.
//! Historical single-source helpers and the current Tape use this same gate.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

use super::{AudioBuffer, AudioFormat, PlaybackQueue};
use crate::audio::Codec;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct View(u64, u64);
impl View {
    fn epoch(self) -> u64 {
        self.0 >> EPOCH_SHIFT
    }
    fn callback_active(self) -> bool {
        self.0 & CALLBACK_ACTIVE != 0
    }
    fn current(self) -> u64 {
        self.1
    }
    fn timeline_reanchored(self) -> bool {
        self.0 & REANCHORED_TIMELINE != 0
    }
    fn current_revoked(self) -> bool {
        self.0 & REVOKED_CURRENT != 0
    }
}

#[derive(Default, Debug)]
struct ClaimGate {
    control: AtomicU64,
    current: AtomicU64,
    // Fixture representation of the existing serialized non-RT owner. The
    // callback never accesses this lease. It prevents a second observation
    // from clearing DIRTY while an older transaction can still commit.
    observing: std::sync::atomic::AtomicBool,
}
#[derive(Debug)]
struct Observation<'a> {
    gate: &'a ClaimGate,
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
impl ClaimGate {
    // Device/control-state inspection, not a non-RT coherent checkpoint.
    fn view(&self) -> View {
        View(
            self.control.load(Ordering::Acquire),
            self.current.load(Ordering::Relaxed),
        )
    }
    fn observe(&self) -> Option<Observation<'_>> {
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
    fn begin_callback(&self) {
        let previous = self
            .control
            .fetch_or(CALLBACK_ACTIVE | CALLBACK_DIRTY, Ordering::AcqRel);
        assert_eq!(previous & CALLBACK_ACTIVE, 0, "single callback writer");
    }
    fn end_callback(&self) {
        assert!(self.view().callback_active());
        self.control.fetch_and(!CALLBACK_ACTIVE, Ordering::Release);
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
    fn validate(&self, observation: &Observation<'_>) -> bool {
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
    fn replace(&self, observation: Observation<'_>) -> Result<View, View> {
        self.commit(observation, 0, false)
    }
    fn reanchor(&self, observation: Observation<'_>) -> Result<View, View> {
        self.commit(observation, REANCHORED_TIMELINE, false)
    }
    fn clear(&self, observation: Observation<'_>) -> Result<View, View> {
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
            "full control-version reuse is outside this probe"
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

struct PreparedFrame {
    sample: i32,
    source_frames: u64,
}

struct PreparedSource {
    epoch: u64,
    source: u64,
    frames: Vec<PreparedFrame>,
    lifetime: Option<DropWitness>,
}

struct DropWitness(mpsc::Sender<thread::ThreadId>);

impl Drop for DropWitness {
    fn drop(&mut self) {
        let _ = self.0.send(thread::current().id());
    }
}

impl PreparedSource {
    // Non-RT preparation uses the real queue reader on a private source view.
    // The explicit step trace isolates transport/progress from correction policy:
    // 0 repeats, 1 copies, and 2 discards one source frame then copies the next.
    // This probe covers a complete mono source; it does not move the real renderer.
    fn prepare(source: &AudioBuffer, id: u64, epoch: u64, steps: &[usize]) -> Self {
        Self::prepare_from_cursor(source, id, epoch, steps, source.timestamp)
    }

    fn prepare_from_cursor(
        source: &AudioBuffer,
        id: u64,
        epoch: u64,
        steps: &[usize],
        cursor_us: i64,
    ) -> Self {
        assert_eq!(source.format.channels, 1);
        assert!(steps.first().is_some_and(|step| *step != 0));
        let mut cursor = PlaybackQueue::new();
        cursor.push(AudioBuffer {
            timestamp: source.timestamp,
            samples: Arc::clone(&source.samples),
            format: source.format.clone(),
        });
        // The preceding source's predicted end is a private preparation cursor,
        // not a committed consumption count. Reuse the actual queue's stale
        // prefix behavior instead of slicing PCM with a second timestamp rule.
        cursor.cursor_us = cursor_us;
        let mut last = [0];
        let mut frames = Vec::with_capacity(steps.len());
        for &step in steps {
            // A complete-source window ends on the final source consumption.
            // Repeats after source exhaustion belong to a later preparation span;
            // cross-source correction spans are not implemented by this probe.
            assert_ne!(cursor.queued_frames(1), 0);
            for _ in 1..step {
                assert!(cursor.consume_next_frame(1, source.format.sample_rate, None));
            }
            if step != 0 {
                assert!(cursor.consume_next_frame(1, source.format.sample_rate, Some(&mut last),));
            }
            frames.push(PreparedFrame {
                sample: last[0],
                source_frames: step as u64,
            });
        }
        assert_eq!(cursor.queued_frames(1), 0);
        Self {
            epoch,
            source: id,
            frames,
            lifetime: None,
        }
    }
}

#[path = "handoff_probe/bounded.rs"]
mod bounded;

// Borrows preallocated immutable PCM. Destruction remains with the non-RT owner.
// Historical single-source payload reader used only inside DeviceSide's ACTIVE
// interval. The bounded owner retains the transport and final destruction.
struct PreparedConsumer<'a> {
    gate: &'a ClaimGate,
    prepared: &'a PreparedSource,
    progress: &'a AtomicU64,
    offset: usize,
    claimed: bool,
}

impl<'a> PreparedConsumer<'a> {
    fn render_active(&mut self, output: &mut [i32]) -> usize {
        assert!(self.gate.view().callback_active(), "reader requires ACTIVE");
        output.fill(0);
        if output.is_empty() {
            return 0;
        }
        if self.offset == self.prepared.frames.len() {
            return output.len();
        }
        if self.claimed && self.gate.view().current_revoked() {
            self.gate.finish_current();
            self.offset = self.prepared.frames.len();
            self.claimed = false;
            return output.len();
        }
        if !self.claimed {
            if !self.gate.claim(self.prepared.epoch, self.prepared.source) {
                return output.len();
            }
            self.claimed = true;
        }
        let readable = output.len().min(self.prepared.frames.len() - self.offset);
        let mut consumed = 0;
        for (destination, frame) in output[..readable]
            .iter_mut()
            .zip(&self.prepared.frames[self.offset..self.offset + readable])
        {
            *destination = frame.sample;
            consumed += frame.source_frames;
        }
        self.offset += readable;
        self.progress.fetch_add(consumed, Ordering::Release);
        if self.offset == self.prepared.frames.len() {
            self.gate.finish_current();
        }
        output.len() - readable
    }
}

fn source_pcm(samples: &[i32]) -> AudioBuffer {
    AudioBuffer {
        timestamp: 0,
        samples: Arc::from(samples),
        format: AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 1_000,
            channels: 1,
            bit_depth: 32,
            codec_header: None,
        },
    }
}

#[test]
fn realtime_handoff_probe_reanchor_baseline_keeps_current_then_skips_stale_pending() {
    let mut queue = PlaybackQueue::new();
    queue.push(source_pcm(&[1, 2, 3, 4]));
    let mut middle = source_pcm(&[10, 11, 12, 13]);
    middle.timestamp = 10_000;
    queue.push(middle);
    let mut last = source_pcm(&[20, 21, 22, 23]);
    last.timestamp = 20_000;
    queue.push(last);
    let mut sample = [0];
    assert!(queue.consume_next_frame(1, 1_000, Some(&mut sample)));
    assert_eq!(sample, [1]);
    // The production CorrectionReanchor closure changes only these cursor fields.
    queue.cursor_us = 20_000;
    queue.cursor_remainder = 0;
    let mut output = Vec::new();
    while queue.consume_next_frame(1, 1_000, Some(&mut sample)) {
        output.push(sample[0]);
    }
    assert_eq!(output, [2, 3, 4, 23]);
    assert_eq!(queue.queued_frames(1), 0);
}

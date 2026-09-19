//! Preparation-span experiment. A corrected output frame may consume two source
//! blocks; a source block is therefore not a sufficient transport unit.
//! Explicit correction steps still isolate representation from clock policy.

use super::*;
use crate::audio::sync_correction::{CorrectionPlanner, CorrectionSchedule};
use crate::audio::AudioBufferLifetime;
use std::sync::Weak;

#[path = "timeline/checkpoint.rs"]
pub(super) mod checkpoint;
use checkpoint::{Cadence, PublishedCheckpoint, SourcePosition};

#[path = "timeline/tape.rs"]
mod tape;

#[path = "timeline/feedback.rs"]
mod feedback;

#[derive(Clone, Copy)]
struct Frame {
    output_index: u64,
    before_index: usize,
    consumed: u64,
    before_current: u64,
    after_current: u64,
    requires_epoch: bool,
    missing: bool,
    cadence: Option<Cadence>,
    position: SourcePosition,
}

pub(super) struct Span {
    epoch: u64,
    channels: usize,
    frames: Box<[Frame]>,
    pcm: Box<[i32]>,
    pub(super) lifetime: Option<DropWitness>,
    checkpoint: Option<Arc<PublishedCheckpoint>>,
}

pub(super) struct Preparation {
    pub(super) cursor: PlaybackQueue,
    retired_through: u64,
    output_frames: u64,
    sources: Vec<(u64, Weak<dyn AudioBufferLifetime>)>,
    last: Vec<i32>,
    channels: usize,
    rate: u32,
    window_frames: usize,
    schedule: CorrectionSchedule,
    insert_counter: u32,
    drop_counter: u32,
    checkpoint: Option<Arc<PublishedCheckpoint>>,
}

impl Preparation {
    pub(super) fn new(sources: Vec<(u64, AudioBuffer)>) -> Self {
        // Fixture default, not a new public queue or chunk limit.
        Self::with_window_frames(sources, 16)
    }

    fn with_window_frames(sources: Vec<(u64, AudioBuffer)>, window_frames: usize) -> Self {
        let mut this = Self::empty(
            usize::from(sources[0].1.format.channels),
            sources[0].1.format.sample_rate,
            window_frames,
        );
        for (id, source) in sources {
            this.push(id, source);
        }
        this
    }

    pub(super) fn empty(channels: usize, rate: u32, window_frames: usize) -> Self {
        assert!(window_frames > 0 && channels > 0);
        let samples = window_frames
            .checked_mul(channels)
            .expect("window sample bound");
        std::alloc::Layout::array::<i32>(samples).expect("window PCM layout");
        std::alloc::Layout::array::<Frame>(window_frames).expect("window metadata layout");
        Self {
            cursor: PlaybackQueue::new(),
            retired_through: 0,
            output_frames: 0,
            sources: Vec::new(),
            last: vec![0; channels],
            channels,
            rate,
            window_frames,
            schedule: CorrectionSchedule::default(),
            insert_counter: 0,
            drop_counter: 0,
            checkpoint: None,
        }
    }

    pub(super) fn push(&mut self, id: u64, source: AudioBuffer) {
        assert_eq!(usize::from(source.format.channels), self.channels);
        assert_eq!(source.format.sample_rate, self.rate);
        // Use the real queue entry's lifetime attachment as a probe-only tag.
        // Production should carry a source ID explicitly, without repurposing
        // the caller's lifetime lease or equating identity with PCM allocation.
        let marker: Arc<dyn AudioBufferLifetime> = Arc::new(());
        self.sources.push((id, Arc::downgrade(&marker)));
        self.cursor.push_with_lifetime(source, Some(marker));
        self.prune_sources();
    }

    pub(super) fn reset(&mut self) {
        self.cursor.clear();
        self.retired_through = 0;
        self.output_frames = 0;
        self.sources.clear();
        self.last.fill(0);
        self.schedule = CorrectionSchedule::default();
        self.insert_counter = 0;
        self.drop_counter = 0;
    }

    pub(super) fn attach_checkpoint(&mut self, checkpoint: Arc<PublishedCheckpoint>) {
        self.checkpoint = Some(checkpoint);
    }

    pub(super) fn apply_actual(&mut self, actual: &checkpoint::Snapshot) {
        self.reconcile_position(actual.position);
        self.output_frames = actual.output_frames;
        self.last.copy_from_slice(&actual.last);
        self.schedule = actual.cadence.schedule;
        self.insert_counter = actual.cadence.insert_counter;
        self.drop_counter = actual.cadence.drop_counter;
    }

    pub(super) fn fork_actual(&self) -> Self {
        // Probe-only explicit copy of the real queue view; PCM allocations and
        // source identity leases remain shared, never copied sample by sample.
        let copy = |entry: &super::super::super::QueuedAudioBuffer| {
            super::super::super::QueuedAudioBuffer {
                source_id: entry.source_id,
                buffer: AudioBuffer {
                    timestamp: entry.timestamp,
                    samples: Arc::clone(&entry.samples),
                    format: entry.format.clone(),
                },
                _lifetime: entry._lifetime.clone(),
            }
        };
        Self {
            cursor: PlaybackQueue {
                publication: None,
                settled_consumed: self.cursor.settled_consumed,
                next_source_id: self.cursor.next_source_id,
                retired_through: self.cursor.retired_through,
                queue: self.cursor.queue.iter().map(copy).collect(),
                pending_frames: self.cursor.pending_frames,
                pending_end_upper_bound: self.cursor.pending_end_upper_bound,
                current: self.cursor.current.as_ref().map(copy),
                index: self.cursor.index,
                cursor_us: self.cursor.cursor_us,
                cursor_remainder: self.cursor.cursor_remainder,
                initialized: self.cursor.initialized,
                generation: self.cursor.generation,
                force_reanchor: self.cursor.force_reanchor,
                enqueue_count: self.cursor.enqueue_count,
            },
            retired_through: self.retired_through,
            output_frames: self.output_frames,
            sources: self.sources.clone(),
            last: self.last.clone(),
            channels: self.channels,
            rate: self.rate,
            window_frames: self.window_frames,
            schedule: self.schedule,
            insert_counter: self.insert_counter,
            drop_counter: self.drop_counter,
            checkpoint: self.checkpoint.clone(),
        }
    }

    fn prune_sources(&mut self) {
        self.sources
            .retain(|(_, marker)| marker.strong_count() != 0);
    }

    fn source_id(&self, source: &super::super::super::QueuedAudioBuffer) -> u64 {
        self.sources
            .iter()
            .find(|(_, marker)| marker.ptr_eq(&Arc::downgrade(source._lifetime.as_ref().unwrap())))
            .unwrap()
            .0
    }

    fn current(&self) -> u64 {
        self.cursor
            .current
            .as_ref()
            .map_or(0, |source| self.source_id(source))
    }

    // Probe-only observation of the real queue. Production should emit explicit
    // source transitions rather than allocate/scan this bounded fixture list.
    fn ordered_ids(&self) -> Vec<u64> {
        self.cursor
            .current
            .iter()
            .chain(self.cursor.queue.iter())
            .map(|source| self.source_id(source))
            .collect()
    }

    fn position_after(&mut self, before: &[u64]) -> SourcePosition {
        let first = self
            .cursor
            .current
            .as_ref()
            .or_else(|| self.cursor.queue.front())
            .map(|source| self.source_id(source));
        let removed = first.map_or(before.len(), |id| {
            before.iter().position(|old| *old == id).unwrap()
        });
        if removed > 0 {
            self.retired_through = before[removed - 1];
        }
        SourcePosition {
            retired_through: self.retired_through,
            current: self.current(),
            index: self.cursor.index,
            cursor_us: self.cursor.cursor_us,
            cursor_remainder: self.cursor.cursor_remainder,
        }
    }

    // Apply an actual checkpoint to an independently retained real source view.
    // No consumption replay: skipped stale blocks retire by exact identity.
    fn reconcile_position(&mut self, position: SourcePosition) {
        if position.retired_through != self.retired_through {
            loop {
                let removed = self
                    .cursor
                    .current
                    .take()
                    .or_else(|| self.cursor.pop_pending())
                    .expect("checkpoint retirement belongs to retained source view");
                if self.source_id(&removed) == position.retired_through {
                    break;
                }
            }
            self.retired_through = position.retired_through;
        }
        if position.current != 0 && self.cursor.current.is_none() {
            self.cursor.current = self.cursor.pop_pending();
        }
        assert_eq!(self.current(), position.current);
        self.cursor.index = position.index;
        self.cursor.cursor_us = position.cursor_us;
        self.cursor.cursor_remainder = position.cursor_remainder;
        self.prune_sources();
    }

    fn prepare_scheduled(
        &mut self,
        epoch: u64,
        requested_frames: usize,
        schedule: CorrectionSchedule,
        measured_timestamp: bool,
    ) -> Span {
        assert!(
            !schedule.reanchor,
            "reanchor is a separate control operation"
        );
        if schedule != self.schedule {
            self.schedule = schedule;
            self.insert_counter = schedule.insert_every_n_frames;
            self.drop_counter = schedule.drop_every_n_frames;
        }
        let count = requested_frames.min(self.window_frames);
        let mut steps = Vec::with_capacity(count);
        let mut checkpoints = Vec::with_capacity(count);
        for _ in 0..count {
            let mut step = 1;
            if measured_timestamp {
                if schedule.drop_every_n_frames > 0 {
                    self.drop_counter = self.drop_counter.saturating_sub(1);
                    if self.drop_counter == 0 {
                        self.drop_counter = schedule.drop_every_n_frames;
                        step = 2;
                    }
                }
                if step != 2 && schedule.insert_every_n_frames > 0 {
                    self.insert_counter = self.insert_counter.saturating_sub(1);
                    if self.insert_counter == 0 {
                        self.insert_counter = schedule.insert_every_n_frames;
                        step = 0;
                    }
                }
            }
            steps.push(step);
            checkpoints.push(Cadence {
                schedule,
                insert_counter: self.insert_counter,
                drop_counter: self.drop_counter,
            });
        }
        let mut span = self.prepare(epoch, &steps);
        for (frame, cadence) in span.frames.iter_mut().zip(checkpoints) {
            frame.cadence = Some(cadence);
        }
        span
    }

    pub(super) fn prepare(&mut self, epoch: u64, steps: &[usize]) -> Span {
        let count = steps.len().min(self.window_frames);
        let mut frames = Vec::with_capacity(count);
        let mut pcm = Vec::with_capacity(count.checked_mul(self.channels).unwrap());
        for &step in &steps[..count] {
            assert!(step <= 2);
            let before_ids = self.ordered_ids();
            let before_current = self.current();
            let before_index = self.cursor.index;
            let remaining = self.cursor.current.as_ref().map_or(0, |current| {
                (current.samples.len() - self.cursor.index) / self.channels
            });
            let mut consumed = 0;
            let mut missing = false;
            // Mirror the renderer's discrete output operations while delegating
            // source selection, timestamps, skipping and cursor to the real queue.
            if step == 2
                && self
                    .cursor
                    .consume_next_frame(self.channels, self.rate, None)
            {
                consumed += 1;
            }
            if step != 0 {
                if self
                    .cursor
                    .consume_next_frame(self.channels, self.rate, Some(&mut self.last))
                {
                    consumed += 1;
                } else if step == 1 {
                    missing = true;
                }
                // Existing drop branch repeats last if no following frame exists.
            }
            if missing {
                pcm.resize(pcm.len() + self.channels, 0);
            } else {
                pcm.extend_from_slice(&self.last);
            }
            let position = self.position_after(&before_ids);
            frames.push(Frame {
                output_index: self.output_frames,
                before_index,
                consumed,
                before_current,
                after_current: self.current(),
                requires_epoch: before_current == 0 || step > remaining,
                missing,
                cadence: None,
                position,
            });
            self.output_frames = self.output_frames.checked_add(1).unwrap();
        }
        self.prune_sources();
        Span {
            epoch,
            channels: self.channels,
            frames: frames.into_boxed_slice(),
            pcm: pcm.into_boxed_slice(),
            lifetime: None,
            checkpoint: self.checkpoint.clone(),
        }
    }
}

struct Read {
    visited: usize,
    missing: usize,
    invalidated: bool,
}

fn render_active(
    span: &Span,
    offset: &mut usize,
    output: &mut [i32],
    gate: &ClaimGate,
    progress: &AtomicU64,
) -> Read {
    assert_eq!(output.len() % span.channels, 0);
    assert!(gate.view().callback_active());
    output.fill(0);
    let mut visited = 0;
    let mut missing = 0;
    let mut view = gate.view();
    if view.current_revoked() {
        gate.finish_current();
        view = gate.view();
    }
    let mut current = view.current();
    let mut consumed = 0;
    let mut invalidated = false;
    let mut last_pcm = None;
    let mut cadence = None;
    let mut position = None;
    let mut output_index = span
        .checkpoint
        .as_ref()
        .map(|publication| publication.output_frames());
    let mut source_index = span
        .checkpoint
        .as_ref()
        .map_or(0, |publication| publication.source_index());
    for destination in output.chunks_exact_mut(span.channels) {
        // A rebuilt window can repeat the still-valid old current prefix. Skip
        // only already emitted output ordinals, without consuming source again.
        if let (Some(actual), Some(frame)) = (output_index, span.frames.get(*offset)) {
            let duplicate = actual.saturating_sub(frame.output_index);
            let skip = usize::try_from(duplicate)
                .unwrap_or(usize::MAX)
                .min(span.frames.len() - *offset);
            *offset += skip;
        }
        let Some(frame) = span.frames.get(*offset) else {
            break;
        };
        if output_index.is_some_and(|actual| frame.output_index != actual)
            || (output_index.is_some() && current != 0 && source_index != frame.before_index)
            || current != frame.before_current
            || (frame.requires_epoch && view.epoch() != span.epoch)
        {
            invalidated = true;
            break;
        }
        let start = *offset * span.channels;
        destination.copy_from_slice(&span.pcm[start..start + span.channels]);
        if !frame.missing {
            last_pcm = Some(start);
        }
        cadence = frame.cadence;
        position = Some(frame.position);
        source_index = frame.position.index;
        if let Some(index) = &mut output_index {
            *index = index.checked_add(1).unwrap();
        }
        consumed += frame.consumed;
        current = frame.after_current;
        *offset += 1;
        visited += 1;
        missing += usize::from(frame.missing);
    }
    // Non-RT cannot commit during ACTIVE. One callback-end checkpoint replaces
    // per-frame atomic writes; all frame checks above use local immutable data.
    progress.fetch_add(consumed, Ordering::Release);
    if let (Some(publication), Some(cadence), Some(position)) =
        (&span.checkpoint, cadence, position)
    {
        publication.publish(
            cadence,
            position,
            visited,
            last_pcm.map(|start| &span.pcm[start..start + span.channels]),
        );
    }
    gate.publish_current(current);
    Read {
        visited,
        missing,
        invalidated,
    }
}

impl DeviceSide<Span> {
    fn render_span(&mut self, output: &mut [i32], channels: usize) -> usize {
        self.gate.begin_callback();
        let missing = self.render_span_active(output, channels);
        self.gate.end_callback();
        missing
    }

    pub(super) fn render_span_active(&mut self, output: &mut [i32], channels: usize) -> usize {
        assert_eq!(output.len() % channels, 0);
        assert!(self.gate.view().callback_active());
        output.fill(0);
        let total = output.len() / channels;
        let mut written = 0;
        let mut missing = 0;
        if self.return_slot() {
            let visible = self.ready.slots();
            for _ in 0..visible {
                if written == total {
                    break;
                }
                let Ok(slot) = self.ready.peek() else { break };
                assert_eq!(slot.prepared.channels, channels);
                let read = render_active(
                    &slot.prepared,
                    &mut self.offset,
                    &mut output[written * channels..],
                    &self.gate,
                    &self.progress,
                );
                written += read.visited;
                missing += read.missing;
                if self.offset != slot.prepared.frames.len() && !read.invalidated {
                    break;
                }
                self.pending_return = self.ready.pop().ok();
                self.offset = 0;
                if !self.return_slot() {
                    break;
                }
            }
        }
        missing + total - written
    }
}

fn sources(channels: u8) -> Vec<(u64, AudioBuffer)> {
    let samples: [Vec<i32>; 2] = if channels == 1 {
        [vec![10, 20], vec![30, 40]]
    } else {
        [vec![10, 110, 20, 120], vec![30, 130, 40, 140]]
    };
    samples
        .into_iter()
        .enumerate()
        .map(|(i, samples)| {
            let mut source = source_pcm(&samples);
            source.timestamp = i as i64 * 2_000;
            source.format.channels = channels.into();
            ((i + 1) as u64, source)
        })
        .collect()
}

#[test]
fn realtime_handoff_probe_timeline_correction_crosses_source_boundary_in_mono_and_stereo() {
    for channels in [1, 2] {
        let mut preparation = Preparation::new(sources(channels));
        let span = preparation.prepare(0, &[1, 2, 1]);
        let (mut producer, mut device) = pipe_for::<Span>(1);
        assert!(producer.publish(span).is_ok());
        let mut prefix = vec![0; usize::from(channels)];
        assert_eq!(device.render_span(&mut prefix, channels.into()), 0);
        assert_eq!(
            prefix,
            if channels == 1 {
                vec![10]
            } else {
                vec![10, 110]
            }
        );
        assert_eq!(device.progress.load(Ordering::Acquire), 1);
        let mut rest = vec![0; usize::from(channels) * 2];
        assert_eq!(device.render_span(&mut rest, channels.into()), 0);
        assert_eq!(
            rest,
            if channels == 1 {
                vec![30, 40]
            } else {
                vec![30, 130, 40, 140]
            }
        );
        assert_eq!(device.progress.load(Ordering::Acquire), 4);
        assert_eq!(device.gate.view().current(), 0);
        assert_eq!(producer.reclaim(), 1);
    }
}

#[test]
fn realtime_handoff_probe_timeline_replacement_replans_cross_source_frame_without_losing_current() {
    let mut old = Preparation::new(sources(1));
    let old_span = old.prepare(0, &[1, 2, 1]);
    let (mut producer, mut device) = pipe_for::<Span>(2);
    assert!(producer.publish(old_span).is_ok());
    assert_eq!(device.render_span(&mut [0], 1), 0);
    assert!(device.gate.replace(device.gate.observe().unwrap()).is_ok());
    let mut stale = [99; 2];
    assert_eq!(device.render_span(&mut stale, 1), 2);
    assert_eq!(stale, [0, 0]);
    assert_eq!(device.progress.load(Ordering::Acquire), 1);
    // Reconstruct the private preparation cursor at actual consumption, then
    // apply the real queue's pending replacement. No advance of current's tail.
    let mut fresh = Preparation::new(sources(1));
    fresh.prepare(0, &[1]);
    let mut replacement = source_pcm(&[50, 60]);
    replacement.timestamp = 2_000;
    fresh.push(3, replacement);
    let new_span = fresh.prepare(device.gate.view().epoch(), &[2, 1]);
    assert!(producer.publish(new_span).is_ok());
    let mut output = [0; 2];
    assert_eq!(device.render_span(&mut output, 1), 0);
    assert_eq!(output, [50, 60]);
    assert_eq!(device.progress.load(Ordering::Acquire), 4);
    assert_eq!(producer.reclaim(), 2);
}

#[test]
fn realtime_handoff_probe_timeline_repeat_between_sources_does_not_consume_next_source() {
    let mut preparation = Preparation::new(sources(1));
    let span = preparation.prepare(0, &[1, 1, 0, 1, 1]);
    let (mut producer, mut device) = pipe_for::<Span>(1);
    assert!(producer.publish(span).is_ok());
    let mut prefix = [0; 3];
    assert_eq!(device.render_span(&mut prefix, 1), 0);
    assert_eq!(prefix, [10, 20, 20]);
    assert_eq!(device.progress.load(Ordering::Acquire), 2);
    let mut suffix = [0; 2];
    assert_eq!(device.render_span(&mut suffix, 1), 0);
    assert_eq!(suffix, [30, 40]);
    assert_eq!(device.progress.load(Ordering::Acquire), 4);
    assert_eq!(producer.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_timeline_window_bound_splits_preparation_without_losing_source_progress()
{
    let mut preparation = Preparation::with_window_frames(sources(2), 2);
    let first = preparation.prepare(0, &[1, 2, 1]);
    assert_eq!(first.frames.len(), 2);
    assert_eq!(&*first.pcm, [10, 110, 30, 130]);
    let second = preparation.prepare(0, &[1]);
    let (mut producer, mut device) = pipe_for::<Span>(2);
    assert!(producer.publish(first).is_ok());
    assert!(producer.publish(second).is_ok());
    let mut output = [0; 8];
    assert_eq!(device.render_span(&mut output, 2), 1);
    assert_eq!(output, [10, 110, 30, 130, 40, 140, 0, 0]);
    assert_eq!(device.progress.load(Ordering::Acquire), 4);
    assert_eq!(producer.reclaim(), 2);
}

#[test]
fn realtime_handoff_probe_timeline_source_identity_does_not_retain_consumed_pcm() {
    let source = source_pcm(&[10, 20]);
    let released = Arc::downgrade(&source.samples);
    let mut preparation = Preparation::new(vec![(1, source)]);
    let span = preparation.prepare(0, &[1, 1]);
    assert!(
        released.upgrade().is_none(),
        "private preparation no longer needs the source allocation"
    );
    assert_eq!(&*span.pcm, [10, 20]);
}

#[test]
fn realtime_handoff_probe_timeline_real_schedule_preserves_cadence_across_windows() {
    for error in [10_000, -10_000] {
        let samples: Vec<i32> = (1..=503).collect();
        let mut preparation = Preparation::with_window_frames(vec![(1, source_pcm(&samples))], 128);
        let schedule = CorrectionPlanner::new().plan(error, 1_000, false);
        let (mut producer, mut device) = pipe_for::<Span>(4);
        let mut remaining = 501;
        while remaining > 0 {
            let span = preparation.prepare_scheduled(0, remaining, schedule, true);
            remaining -= span.frames.len();
            assert!(producer.publish(span).is_ok());
        }
        let mut output = [0; 501];
        assert_eq!(device.render_span(&mut output[..129], 1), 0);
        assert_eq!(device.render_span(&mut output[129..257], 1), 0);
        assert_eq!(device.render_span(&mut output[257..], 1), 0);
        assert_eq!(&output[..499], &(1..=499).collect::<Vec<i32>>());
        if error > 0 {
            assert_eq!(&output[499..], [501, 502]);
            assert_eq!(device.progress.load(Ordering::Acquire), 502);
        } else {
            assert_eq!(&output[499..], [499, 500]);
            assert_eq!(device.progress.load(Ordering::Acquire), 500);
        }
        assert_eq!(producer.reclaim(), 4);
    }
}

#[test]
fn realtime_handoff_probe_timeline_source_identity_distinguishes_reused_sample_allocation() {
    let first = source_pcm(&[10, 20]);
    let second = AudioBuffer {
        timestamp: 2_000,
        samples: Arc::clone(&first.samples),
        format: first.format.clone(),
    };
    let resumed = AudioBuffer {
        timestamp: second.timestamp,
        samples: Arc::clone(&second.samples),
        format: second.format.clone(),
    };
    let mut preparation = Preparation::new(vec![(1, first), (2, second)]);
    let span = preparation.prepare(0, &[1, 1, 1]);
    let (mut producer, mut device) = pipe_for::<Span>(1);
    assert!(producer.publish(span).is_ok());
    let mut prefix = [0; 3];
    assert_eq!(device.render_span(&mut prefix, 1), 0);
    assert_eq!(prefix, [10, 20, 10]);
    assert_eq!(producer.reclaim(), 1);
    // A renewed view retains the second source's identity despite shared PCM.
    let mut fresh = Preparation::new(vec![(2, resumed)]);
    fresh.prepare(0, &[1]);
    assert!(producer.publish(fresh.prepare(0, &[1])).is_ok());
    let mut suffix = [0];
    assert_eq!(device.render_span(&mut suffix, 1), 0);
    assert_eq!(suffix, [20]);
    assert_eq!(device.progress.load(Ordering::Acquire), 4);
    assert_eq!(producer.reclaim(), 1);
}

#[test]
fn realtime_handoff_probe_timeline_real_schedule_pauses_cadence_without_measured_timestamp() {
    for error in [10_000, -10_000] {
        let samples: Vec<i32> = (1..=503).collect();
        let mut preparation = Preparation::with_window_frames(vec![(1, source_pcm(&samples))], 512);
        let schedule = CorrectionPlanner::new().plan(error, 1_000, false);
        let (mut producer, mut device) = pipe_for::<Span>(3);
        for (count, measured) in [(499, true), (1, false), (1, true)] {
            assert!(producer
                .publish(preparation.prepare_scheduled(0, count, schedule, measured))
                .is_ok());
        }
        let mut output = [0; 501];
        assert_eq!(device.render_span(&mut output, 1), 0);
        assert_eq!(&output[..500], &(1..=500).collect::<Vec<i32>>());
        if error > 0 {
            assert_eq!(output[500], 502);
            assert_eq!(device.progress.load(Ordering::Acquire), 502);
        } else {
            assert_eq!(output[500], 500);
            assert_eq!(device.progress.load(Ordering::Acquire), 500);
        }
        assert_eq!(producer.reclaim(), 3);
    }
}

#[test]
fn realtime_handoff_probe_timeline_clear_reconciles_admission_and_reclaims_off_device() {
    use super::admission::SourceLedger;
    use crate::audio::{EnqueueOutcome, RendererOperationOutcome};

    let (mut producer, mut device) = pipe_for::<Span>(2);
    let gate = Arc::clone(&device.gate);
    let mut ledger = SourceLedger::new(Arc::clone(&device.progress));
    let input = sources(1);
    for (_, source) in &input {
        let copied = AudioBuffer {
            timestamp: source.timestamp,
            samples: Arc::clone(&source.samples),
            format: source.format.clone(),
        };
        assert!(matches!(
            ledger.try_enqueue(copied, &gate, || {}),
            Ok(EnqueueOutcome::Accepted { .. })
        ));
    }
    let (dropped_tx, dropped_rx) = mpsc::channel();
    let mut preparation = Preparation::new(input);
    let mut span = preparation.prepare(gate.view().epoch(), &[1, 2, 1]);
    span.lifetime = Some(DropWitness(dropped_tx.clone()));
    assert!(producer.publish(span).is_ok());
    let (ready_tx, ready_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut prefix = [0];
        let missing_prefix = device.render_span(&mut prefix, 1);
        ready_tx.send((prefix, missing_prefix)).unwrap();
        resume_rx.recv().unwrap();
        let mut output = [0; 3];
        let missing = device.render_span(&mut output, 1);
        (device, output, missing)
    });
    let (prefix, missing_prefix) = ready_rx.recv().unwrap();
    let clear = ledger.try_clear(&gate);
    let empty = ledger
        .owner
        .capacity(ledger.scope)
        .unwrap()
        .current_frames();
    let new_source = source_pcm(&[50, 60]);
    let admitted = ledger.try_enqueue(source_pcm(&[50, 60]), &gate, || {});
    let mut fresh = Preparation::new(vec![(3, new_source)]);
    let mut next = fresh.prepare(gate.view().epoch(), &[1, 1]);
    next.lifetime = Some(DropWitness(dropped_tx));
    let published = producer.publish(next).is_ok();
    resume_tx.send(()).unwrap();
    let (_device, output, missing) = reader.join().unwrap();
    // Assert only after releasing and joining the controlled reader.
    assert_eq!((prefix, missing_prefix), ([10], 0));
    assert_eq!(clear, Ok(RendererOperationOutcome::Applied));
    assert_eq!(empty, 0);
    assert!(matches!(
        admitted,
        Ok(EnqueueOutcome::Accepted {
            queued_frames: 2,
            queued_buffers: 1
        })
    ));
    assert!(published);
    assert_eq!((output, missing), ([50, 60, 0], 1));
    assert!(dropped_rx.try_recv().is_err(), "collector has not run");
    ledger.reconcile();
    assert_eq!(ledger.owner.consumed_frames(ledger.scope), Ok(3));
    assert_eq!(
        ledger
            .owner
            .capacity(ledger.scope)
            .unwrap()
            .current_frames(),
        0
    );
    assert_eq!(producer.reclaim(), 2);
    for _ in 0..2 {
        assert_eq!(dropped_rx.recv().unwrap(), thread::current().id());
    }
}

// Diagnostic counterexample, not a passing compatibility acceptance test.
#[test]
fn realtime_handoff_probe_final_pcm_cannot_defer_actual_fallback_decision() {
    for error in [10_000, -10_000] {
        let samples: Vec<i32> = (1..=503).collect();
        let mut preparation = Preparation::with_window_frames(vec![(1, source_pcm(&samples))], 512);
        let schedule = CorrectionPlanner::new().plan(error, 1_000, false);
        let (mut producer, mut device) = pipe_for::<Span>(2);
        assert!(producer
            .publish(preparation.prepare_scheduled(0, 499, schedule, true))
            .is_ok());
        // These final samples are computed before the future callback supplies
        // its actual provenance. The copier has no remaining branch to select.
        assert!(producer
            .publish(preparation.prepare_scheduled(0, 1, schedule, true))
            .is_ok());
        let mut prefix = [0; 499];
        assert_eq!(device.render_span(&mut prefix, 1), 0);
        assert_eq!(prefix, std::array::from_fn(|i| i as i32 + 1));
        assert_eq!(device.progress.load(Ordering::Acquire), 499);
        let mut future_fallback_output = [0];
        assert_eq!(device.render_span(&mut future_fallback_output, 1), 0);
        let source_delta = device.progress.load(Ordering::Acquire) - 499;
        // Existing fallback contract: output source frame 500, consume one,
        // retain the correction counter for a later measured callback.
        let required_fallback = ([500], 1);
        assert_eq!(
            (future_fallback_output, source_delta),
            if error > 0 { ([501], 2) } else { ([499], 0) }
        );
        assert_ne!(
            (future_fallback_output, source_delta),
            required_fallback,
            "a fixed corrected path cannot honor the later fallback branch"
        );
        assert_eq!(producer.reclaim(), 2);
    }
}

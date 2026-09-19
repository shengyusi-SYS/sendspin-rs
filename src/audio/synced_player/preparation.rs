//! Non-realtime source view and fixed-capacity uncorrected PCM windows.
//! Source identities are independent of caller lifetime attachments.

use super::ingress::SourcePosition;
use super::PlaybackQueue;

#[derive(Clone, Copy, Debug)]
pub(super) struct Frame {
    pub before_current: u64,
    pub before_index: usize,
    pub requires_epoch: bool,
    pub position: SourcePosition,
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            before_current: 0,
            before_index: 0,
            requires_epoch: true,
            position: SourcePosition {
                retired_through: 0,
                current: 0,
                index: 0,
                cursor_us: 0,
                cursor_remainder: 0,
            },
        }
    }
}

pub(super) struct Window {
    pub epoch: u64,
    pub source_base: u64,
    pub cursor_before: Option<i64>,
    pub frames: Vec<Frame>,
    pub pcm: Vec<i32>,
    pub valid: usize,
    pub skipped_tail: Option<Frame>,
}

impl Window {
    /// Construction only. The worker refills these allocations without resizing.
    pub(super) fn new(frames: usize, channels: usize) -> Option<Self> {
        if frames == 0 || channels == 0 {
            return None;
        }
        let samples = frames.checked_mul(channels)?;
        std::alloc::Layout::array::<Frame>(frames).ok()?;
        std::alloc::Layout::array::<i32>(samples).ok()?;
        Some(Self {
            epoch: 0,
            source_base: 0,
            cursor_before: None,
            frames: vec![Frame::default(); frames],
            pcm: vec![0; samples],
            valid: 0,
            skipped_tail: None,
        })
    }
}

impl PlaybackQueue {
    fn source_position(&self) -> SourcePosition {
        SourcePosition {
            retired_through: self.retired_through,
            current: self.current.as_ref().map_or(0, |entry| entry.source_id),
            index: self.index,
            cursor_us: self.cursor_us,
            cursor_remainder: self.cursor_remainder,
        }
    }

    /// Both views are non-RT owned. The caller validates the observation before
    /// publishing any window produced from this snapshot.
    pub(super) fn copy_source_into(&self, destination: &mut Self) {
        assert!(destination.queue.capacity() >= self.queue.len());
        destination.queue.clear();
        destination.current = None;
        destination.queue.extend(self.queue.iter().cloned());
        destination.current = self.current.clone();
        destination.next_source_id = self.next_source_id;
        destination.retired_through = self.retired_through;
        destination.pending_frames = self.pending_frames;
        destination.pending_end_upper_bound = self.pending_end_upper_bound;
        destination.index = self.index;
        destination.cursor_us = self.cursor_us;
        destination.cursor_remainder = self.cursor_remainder;
        destination.initialized = self.initialized;
        destination.generation = self.generation;
        destination.force_reanchor = self.force_reanchor;
        destination.enqueue_count = self.enqueue_count;
    }

    /// Reconcile an actual device checkpoint, without replaying consumed PCM.
    /// Invoke only after validating that it belongs to this retained source view.
    pub(super) fn reconcile_source(&mut self, position: SourcePosition) {
        if position.retired_through != self.retired_through {
            loop {
                let retired = self
                    .current
                    .take()
                    .or_else(|| self.pop_pending())
                    .expect("actual retirement belongs to retained source view");
                if retired.source_id == position.retired_through {
                    break;
                }
            }
            self.retired_through = position.retired_through;
        }
        if position.current != 0 && self.current.is_none() {
            self.current = self.pop_pending();
        }
        assert_eq!(
            self.current.as_ref().map_or(0, |entry| entry.source_id),
            position.current
        );
        self.index = position.index;
        self.cursor_us = position.cursor_us;
        self.cursor_remainder = position.cursor_remainder;
    }

    pub(super) fn prepare_window(
        &mut self,
        window: &mut Window,
        epoch: u64,
        source_base: u64,
        channels: usize,
        sample_rate: u32,
    ) {
        assert_eq!(window.pcm.len(), window.frames.len() * channels);
        window.epoch = epoch;
        window.source_base = source_base;
        window.cursor_before = self.initialized.then_some(self.cursor_us);
        window.valid = 0;
        window.skipped_tail = None;
        for index in 0..window.frames.len() {
            let before = self.source_position();
            let remaining = self.current.as_ref().map_or(0, |entry| {
                entry.samples.len().saturating_sub(self.index) / channels
            });
            let consumed = self.consume_next_frame(
                channels,
                sample_rate,
                Some(&mut window.pcm[index * channels..(index + 1) * channels]),
            );
            let frame = Frame {
                before_current: before.current,
                before_index: before.index,
                requires_epoch: remaining == 0,
                position: self.source_position(),
            };
            if !consumed {
                if frame.position.retired_through != before.retired_through {
                    window.skipped_tail = Some(frame);
                }
                break;
            }
            window.frames[index] = frame;
            window.valid += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioBuffer, AudioFormat, Codec};
    use std::sync::Arc;

    fn source(timestamp: i64, samples: &[i32]) -> AudioBuffer {
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
        }
    }

    #[test]
    fn renderer_realtime_preparation_reconciles_timestamp_order_without_pcm_replay() {
        let mut canonical = PlaybackQueue::new();
        canonical.push(source(0, &[1, 2]));
        canonical.push(source(4_000, &[5, 6]));
        canonical.push(source(2_000, &[3, 4]));
        let mut private = PlaybackQueue::new();
        private.queue.reserve(3);
        canonical.copy_source_into(&mut private);
        let mut window = Window::new(4, 1).unwrap();
        let pcm_allocation = window.pcm.as_ptr();
        let metadata_allocation = window.frames.as_ptr();
        private.prepare_window(&mut window, 7, 0, 1, 1_000);
        assert_eq!(window.valid, 4);
        assert_eq!(&window.pcm[..4], &[1, 2, 3, 4]);
        assert_eq!(
            canonical.queued_frames(1),
            6,
            "preparation is not consumption"
        );
        canonical.reconcile_source(window.frames[3].position);
        assert_eq!(canonical.queued_frames(1), 2);
        assert_eq!(canonical.cursor_us, 4_000);
        canonical.copy_source_into(&mut private);
        private.prepare_window(&mut window, 8, 4, 1, 1_000);
        assert_eq!(window.valid, 2);
        assert_eq!(&window.pcm[..2], &[5, 6]);
        assert_eq!(window.pcm.as_ptr(), pcm_allocation);
        assert_eq!(window.frames.as_ptr(), metadata_allocation);
        assert_eq!(window.source_base, 4);
        canonical.reconcile_source(window.frames[1].position);
        assert_eq!(canonical.queued_frames(1), 0);
        canonical.clear();
        canonical.push(source(0, &[9]));
        assert_eq!(canonical.queue.front().unwrap().source_id, 4);
    }

    #[test]
    fn renderer_realtime_preparation_publishes_stale_retirement_without_fake_consumption() {
        let mut canonical = PlaybackQueue::new();
        canonical.push(source(0, &[1, 2]));
        canonical.cursor_us = 10_000;
        let mut private = PlaybackQueue::new();
        private.queue.reserve(1);
        canonical.copy_source_into(&mut private);
        let mut window = Window::new(2, 1).unwrap();
        private.prepare_window(&mut window, 1, 0, 1, 1_000);
        assert_eq!(window.valid, 0);
        let skipped = window.skipped_tail.unwrap();
        assert_eq!(skipped.position.retired_through, 1);
        assert_eq!(skipped.position.cursor_us, 10_000);
        canonical.reconcile_source(skipped.position);
        assert_eq!(canonical.queued_frames(1), 0);
        assert_eq!(canonical.cursor_us, 10_000);
    }
}

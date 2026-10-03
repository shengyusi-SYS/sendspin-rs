//! Device-owned finite frame selection over prepared, uncorrected PCM.
//! No source queue, clock model, payload allocation or owner mutex is used here.

use super::ingress::{CallbackGuard, PublishedCheckpoint, SourcePosition};
use super::transport::DeviceSide;
use crate::audio::sync_correction::CorrectionSchedule;
use std::sync::atomic::{AtomicU64, Ordering};

/// How an existing source frame contributes to the device output.
#[derive(Clone, Copy)]
enum FrameRead {
    Copy,
    Blend,
    PeekBlend,
    RetireOnly,
    DiscardBefore(i64),
}

#[derive(Default, Debug)]
pub(super) struct Rendered {
    pub consumed: u64,
    pub missing: usize,
    pub inserted: usize,
    pub dropped: usize,
    pub source_cursor_us: Option<i64>,
    pub first_output_frame: Option<usize>,
}

pub(super) struct Reader {
    last: Vec<i32>,
    position: Option<SourcePosition>,
    minimum_epoch: u64,
    timeline: u64,
    schedule: CorrectionSchedule,
    insert_counter: u32,
    drop_counter: u32,
    origin: Option<i64>,
    presentation: Option<(i64, u32, usize)>,
    waiting: bool,
    timing_required: bool,
    prefix_ready: bool,
}

impl Reader {
    pub(super) fn set_presentation(
        &mut self,
        source_now_us: Option<i64>,
        rate: u32,
        required: bool,
    ) {
        self.presentation = source_now_us.map(|now| (now, rate, 0));
        self.timing_required = required;
    }

    pub(super) fn last_capacity(&self) -> usize {
        self.last.capacity()
    }

    pub(super) fn new(channels: usize) -> Self {
        assert_ne!(channels, 0);
        Self {
            last: vec![0; channels],
            position: None,
            minimum_epoch: 0,
            timeline: 0,
            schedule: CorrectionSchedule::default(),
            insert_counter: 0,
            drop_counter: 0,
            origin: None,
            presentation: None,
            waiting: false,
            timing_required: false,
            prefix_ready: false,
        }
    }

    pub(super) fn timeline(&self, callback: &CallbackGuard<'_>) -> u64 {
        let view = callback.view();
        if view.current_revoked() || view.timeline_reanchored() {
            view.epoch()
        } else {
            self.timeline
        }
    }

    pub(super) fn schedule(&self) -> CorrectionSchedule {
        self.schedule
    }

    fn reset_cadence(&mut self) {
        self.schedule = CorrectionSchedule::default();
        self.insert_counter = 0;
        self.drop_counter = 0;
    }

    fn take_frame(
        &mut self,
        pipe: &mut DeviceSide,
        destination: &mut [i32],
        read: FrameRead,
        visible: &mut usize,
        epoch: u64,
        current: &mut u64,
        consumed: &mut u64,
    ) -> bool {
        while *visible > 0 {
            let offset = pipe.offset();
            let Some(window) = pipe.peek() else {
                return false;
            };
            let skip = usize::try_from(consumed.saturating_sub(window.source_base))
                .unwrap_or(usize::MAX)
                .min(window.valid);
            if skip > offset {
                pipe.advance(skip - offset);
                continue;
            }
            let Some(frame) = window
                .frames
                .get(offset)
                .filter(|_| offset < window.valid)
                .or(window.skipped_tail.as_ref())
                .copied()
            else {
                *visible -= 1;
                if !pipe.retire() {
                    return false;
                }
                continue;
            };
            let mismatch = window.epoch < self.minimum_epoch
                || window.source_base.checked_add(offset as u64) != Some(*consumed)
                || frame.before_current != *current
                || (*current != 0 && self.position.is_some_and(|p| p.index != frame.before_index))
                || (frame.requires_epoch && window.epoch != epoch);
            if mismatch {
                *visible -= 1;
                if !pipe.retire() {
                    return false;
                }
                continue;
            }
            if offset == window.valid {
                if matches!(read, FrameRead::PeekBlend) {
                    return false;
                }
                self.position = Some(frame.position);
                *current = frame.position.current;
                *visible -= 1;
                if !pipe.retire() {
                    return false;
                }
                continue;
            }
            if matches!(read, FrameRead::RetireOnly) {
                return false;
            }
            if let FrameRead::DiscardBefore(target_us) = read {
                // Compare the exact source-frame end, not the number of frames
                // since the old anchor. A future suffix must remain untouched.
                let end = frame.position;
                if end.cursor_us > target_us
                    || (end.cursor_us == target_us && end.cursor_remainder != 0)
                {
                    self.prefix_ready = true;
                    return false;
                }
            }
            // Ordinary contiguous PCM stays under the soft correction planner.
            // Only a discontinuity waits for a new source time; integer source
            // timestamps may differ by one microsecond at a chunk boundary.
            let discontinuous = self.position.is_none_or(|position| {
                frame.source_start_us > position.cursor_us.saturating_add(1)
            });
            if discontinuous
                && (self.presentation.is_none() && self.timing_required
                    || self.presentation.is_some_and(|(now, rate, index)| {
                        i128::from(frame.source_start_us)
                            > i128::from(now) + index as i128 * 1_000_000 / i128::from(rate)
                    }))
            {
                self.waiting = true;
                return false;
            }
            if self.origin.is_none() && !matches!(read, FrameRead::PeekBlend) {
                self.origin = Some(frame.source_start_us);
            }
            let channels = self.last.len();
            let pcm = &window.pcm[offset * channels..(offset + 1) * channels];
            match read {
                FrameRead::Copy | FrameRead::DiscardBefore(_) => destination.copy_from_slice(pcm),
                FrameRead::RetireOnly => unreachable!("retirement does not copy PCM"),
                FrameRead::Blend | FrameRead::PeekBlend => {
                    for (out, sample) in destination.iter_mut().zip(pcm) {
                        // Widen before addition: full-scale same-sign samples must
                        // neither overflow nor lose a bit before averaging.
                        *out = ((i64::from(*out) + i64::from(*sample)) / 2) as i32;
                    }
                }
            }
            if matches!(read, FrameRead::PeekBlend) {
                // The next real output owns this frame and its source cursor.
                return true;
            }
            self.position = Some(frame.position);
            *current = frame.position.current;
            *consumed = consumed.checked_add(1).expect("scope consumed frame count");
            let done = offset + 1 == window.valid && window.skipped_tail.is_none();
            pipe.advance(1);
            if done {
                *visible -= 1;
                // A failed return retains ownership inside the transport. The
                // frame just copied is still valid; later takes see no work.
                pipe.retire();
            }
            return true;
        }
        false
    }

    /// Caller owns ACTIVE for this entire invocation and gates start/closed
    /// before entering. One finite ready budget is shared by all output frames.
    pub(super) fn render(
        &mut self,
        pipe: &mut DeviceSide,
        callback: &CallbackGuard<'_>,
        checkpoint: &PublishedCheckpoint,
        progress: &AtomicU64,
        output: &mut [i32],
        measured: bool,
        wanted: Option<CorrectionSchedule>,
    ) -> Rendered {
        let mut visible = if pipe.return_pending() {
            pipe.visible()
        } else {
            0
        };
        self.render_with_budget(
            pipe,
            callback,
            checkpoint,
            progress,
            output,
            measured,
            wanted,
            &mut visible,
        )
    }

    /// Accept a prepared all-expired tail without requiring an audible frame.
    pub(super) fn retire_tail(
        &mut self,
        pipe: &mut DeviceSide,
        callback: &CallbackGuard<'_>,
        checkpoint: &PublishedCheckpoint,
        progress: &AtomicU64,
        visible: &mut usize,
    ) {
        self.render_with_budget(
            pipe,
            callback,
            checkpoint,
            progress,
            &mut [],
            false,
            Some(self.schedule),
            visible,
        );
        let mut current = callback.view().current();
        let mut consumed = progress.load(Ordering::Acquire);
        self.take_frame(
            pipe,
            &mut [],
            FrameRead::RetireOnly,
            visible,
            callback.view().epoch(),
            &mut current,
            &mut consumed,
        );
        if let Some(position) = self.position {
            checkpoint.publish(callback, position);
        }
        callback.publish_current(current);
    }

    /// Discard only the late startup prefix, using the same finite ready budget
    /// as rendering. True means the next source frame reached the target; false
    /// means the finite ready budget ended before that could be established.
    /// Consumption and checkpoints remain owned by this reader.
    pub(super) fn discard_start_prefix(
        &mut self,
        pipe: &mut DeviceSide,
        callback: &CallbackGuard<'_>,
        checkpoint: &PublishedCheckpoint,
        progress: &AtomicU64,
        scratch: &mut [i32],
        target_us: i64,
        visible: &mut usize,
    ) -> bool {
        // Apply clear/reanchor before looking at the new source timeline.
        self.render_with_budget(
            pipe,
            callback,
            checkpoint,
            progress,
            &mut [],
            false,
            Some(self.schedule),
            visible,
        );
        let mut current = callback.view().current();
        let before = progress.load(Ordering::Acquire);
        let mut consumed = before;
        self.prefix_ready = false;
        while self.take_frame(
            pipe,
            scratch,
            FrameRead::DiscardBefore(target_us),
            visible,
            callback.view().epoch(),
            &mut current,
            &mut consumed,
        ) {}
        progress.store(consumed, Ordering::Release);
        if let Some(position) = self.position {
            checkpoint.publish(callback, position);
        }
        callback.publish_current(current);
        self.prefix_ready
    }

    /// Share one callback-entry budget across fixed scratch blocks. Refilling
    /// the ready ring concurrently cannot extend this device invocation's work.
    pub(super) fn render_with_budget(
        &mut self,
        pipe: &mut DeviceSide,
        callback: &CallbackGuard<'_>,
        checkpoint: &PublishedCheckpoint,
        progress: &AtomicU64,
        output: &mut [i32],
        measured: bool,
        wanted: Option<CorrectionSchedule>,
        visible: &mut usize,
    ) -> Rendered {
        let channels = self.last.len();
        assert_eq!(output.len() % channels, 0);
        let view = callback.view();
        let cleared = view.current_revoked();
        let reanchored = view.timeline_reanchored();
        self.origin = None;
        if cleared || reanchored {
            self.timeline = view.epoch();
            self.position = None;
            self.reset_cadence();
        }
        if reanchored {
            self.minimum_epoch = view.epoch();
            callback.acknowledge_reanchor();
        }
        if cleared {
            callback.finish_current();
            self.last.fill(0);
        }
        match wanted {
            None => self.reset_cadence(),
            Some(plan) if measured && plan != self.schedule => {
                assert!(!plan.reanchor, "reanchor belongs to source control");
                self.schedule = plan;
                self.insert_counter = plan.insert_every_n_frames;
                self.drop_counter = plan.drop_every_n_frames;
            }
            _ => {}
        }
        let mut current = callback.view().current();
        let before = progress.load(Ordering::Acquire);
        let mut consumed = before;
        let mut rendered = Rendered::default();
        for (index, destination) in output.chunks_exact_mut(channels).enumerate() {
            if index > 0 {
                if let Some((_, _, frame)) = &mut self.presentation {
                    *frame += 1;
                }
            }
            self.waiting = false;
            let mut step = 1;
            if measured {
                if self.schedule.drop_every_n_frames > 0 {
                    self.drop_counter = self.drop_counter.saturating_sub(1);
                    if self.drop_counter == 0 {
                        self.drop_counter = self.schedule.drop_every_n_frames;
                        step = 2;
                    }
                }
                if step != 2 && self.schedule.insert_every_n_frames > 0 {
                    self.insert_counter = self.insert_counter.saturating_sub(1);
                    if self.insert_counter == 0 {
                        self.insert_counter = self.schedule.insert_every_n_frames;
                        step = 0;
                    }
                }
            }
            if step == 0 {
                destination.copy_from_slice(&self.last);
                // Interpolate between the previous source frame and the next one.
                // If lookahead is unavailable, retain the existing repeat fallback.
                self.take_frame(
                    pipe,
                    destination,
                    FrameRead::PeekBlend,
                    visible,
                    view.epoch(),
                    &mut current,
                    &mut consumed,
                );
                if self.waiting {
                    destination.fill(0);
                    rendered.missing += 1;
                } else {
                    rendered.inserted += 1;
                }
                continue;
            }
            let dropped = step == 2
                && self.take_frame(
                    pipe,
                    destination,
                    FrameRead::Copy,
                    visible,
                    view.epoch(),
                    &mut current,
                    &mut consumed,
                );
            if self.take_frame(
                pipe,
                destination,
                if dropped {
                    FrameRead::Blend
                } else {
                    FrameRead::Copy
                },
                visible,
                view.epoch(),
                &mut current,
                &mut consumed,
            ) {
                if dropped {
                    rendered.dropped += 1;
                }
                self.last.copy_from_slice(destination);
            } else if dropped {
                // The first frame is real even when the second falls across a
                // future gap. Preserve it instead of replacing it with silence.
                self.last.copy_from_slice(destination);
            } else if step == 2 && !self.waiting {
                destination.copy_from_slice(&self.last);
            } else {
                destination.fill(0);
                rendered.missing += 1;
            }
            if rendered.first_output_frame.is_none() && self.origin.is_some() {
                rendered.first_output_frame = Some(index);
            }
        }
        progress.store(consumed, Ordering::Release);
        if let Some(position) = self.position {
            checkpoint.publish(callback, position);
        }
        callback.publish_current(current);
        if !output.is_empty() {
            if let Some((_, _, frame)) = &mut self.presentation {
                *frame += 1;
            }
        }
        rendered.consumed = consumed - before;
        rendered.source_cursor_us = self.origin;
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::sync_correction::CorrectionPlanner;
    use crate::audio::synced_player::{ingress::ControlGate, transport, PlaybackQueue};
    use crate::audio::{AudioBuffer, AudioFormat, Codec};
    use std::sync::Arc;

    #[test]
    fn renderer_realtime_reader_fixed_blocks_share_one_ready_budget() {
        let (mut producer, mut device) = transport::pipe(1, 1, 1).unwrap();
        let mut source = PlaybackQueue::new();
        source.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from([11, 22]),
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: 1_000,
                channels: 1,
                bit_depth: 24,
                codec_header: None,
            },
        });
        source.prepare_window(producer.builder_mut(), 0, 0, 1, 1_000);
        assert!(producer.publish());
        let gate = ControlGate::default();
        let checkpoint = PublishedCheckpoint::new();
        let progress = AtomicU64::new(0);
        let mut reader = Reader::new(1);
        let callback = gate.begin_callback();
        assert!(device.return_pending());
        let mut visible = device.visible();
        let mut output = [0];
        reader.render_with_budget(
            &mut device,
            &callback,
            &checkpoint,
            &progress,
            &mut output,
            false,
            Some(CorrectionSchedule::default()),
            &mut visible,
        );
        assert_eq!(output, [11]);
        assert_eq!(visible, 0);
        assert_eq!(producer.reclaim(), 1);
        source.prepare_window(producer.builder_mut(), 0, 1, 1, 1_000);
        assert!(producer.publish());
        reader.render_with_budget(
            &mut device,
            &callback,
            &checkpoint,
            &progress,
            &mut output,
            false,
            Some(CorrectionSchedule::default()),
            &mut visible,
        );
        assert_eq!(
            output,
            [0],
            "a scratch block cannot replenish the callback's work budget"
        );
        assert_eq!(progress.load(Ordering::Acquire), 1);
        drop(callback);
        reader.render(
            &mut device,
            &gate.begin_callback(),
            &checkpoint,
            &progress,
            &mut output,
            false,
            Some(CorrectionSchedule::default()),
        );
        assert_eq!(output, [22]);
        assert_eq!(progress.load(Ordering::Acquire), 2);
    }

    #[test]
    fn renderer_realtime_reader_fallback_preserves_pcm_and_actual_cadence() {
        for error in [10_000, -10_000] {
            let (mut producer, mut device) = transport::pipe(1, 503, 1).unwrap();
            let mut source = PlaybackQueue::new();
            source.push(AudioBuffer {
                timestamp: 0,
                samples: Arc::from((1..=503).collect::<Vec<i32>>()),
                format: AudioFormat {
                    codec: Codec::Pcm,
                    sample_rate: 1_000,
                    channels: 1,
                    bit_depth: 24,
                    codec_header: None,
                },
            });
            source.prepare_window(producer.builder_mut(), 0, 0, 1, 1_000);
            assert!(producer.publish());
            let gate = ControlGate::default();
            let checkpoint = PublishedCheckpoint::new();
            let progress = AtomicU64::new(0);
            let mut reader = Reader::new(1);
            let plan = CorrectionPlanner::new().plan(error, 1_000, false);
            let mut prefix = [0; 499];
            reader.render(
                &mut device,
                &gate.begin_callback(),
                &checkpoint,
                &progress,
                &mut prefix,
                true,
                Some(plan),
            );
            assert_eq!(prefix, std::array::from_fn(|i| i as i32 + 1));
            let mut single = [0];
            reader.render(
                &mut device,
                &gate.begin_callback(),
                &checkpoint,
                &progress,
                &mut single,
                false,
                Some(CorrectionSchedule::default()),
            );
            assert_eq!(single, [500]);
            assert_eq!(progress.load(Ordering::Acquire), 500);
            reader.render(
                &mut device,
                &gate.begin_callback(),
                &checkpoint,
                &progress,
                &mut single,
                true,
                Some(plan),
            );
            assert_eq!(single, if error > 0 { [501] } else { [500] });
            assert_eq!(
                progress.load(Ordering::Acquire),
                if error > 0 { 502 } else { 500 }
            );
            // Reset notification is sufficient even before non-RT checkpoint
            // invalidation; no old last frame may survive to repeat on clear.
            gate.clear(gate.observe().unwrap()).unwrap();
            let cleared = reader.render(
                &mut device,
                &gate.begin_callback(),
                &checkpoint,
                &progress,
                &mut single,
                false,
                Some(plan),
            );
            assert_eq!(single, [0]);
            assert_eq!(cleared.missing, 1);
            assert_eq!(cleared.consumed, 0);
            assert_eq!(producer.reclaim(), 1);
        }
    }
}

#[cfg(test)]
mod smoothing_tests {
    use super::*;
    use crate::audio::synced_player::{ingress::ControlGate, transport, PlaybackQueue};
    use crate::audio::{AudioBuffer, AudioFormat, Codec};
    use std::sync::Arc;

    fn render_ramp(plan: CorrectionSchedule, chunks: &[usize]) -> (Vec<i32>, u64, usize, usize) {
        render_samples(
            plan,
            chunks,
            (0..32).flat_map(|i| [i * 1000, -i * 1000]).collect(),
            32,
        )
    }

    fn render_samples(
        plan: CorrectionSchedule,
        chunks: &[usize],
        samples: Vec<i32>,
        window_frames: usize,
    ) -> (Vec<i32>, u64, usize, usize) {
        let frames = samples.len() / 2;
        let windows = frames.div_ceil(window_frames);
        let (mut producer, mut device) = transport::pipe(windows, window_frames, 2).unwrap();
        let mut source = PlaybackQueue::new();
        source.push(AudioBuffer {
            timestamp: 0,
            samples: samples.into(),
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: 1000,
                channels: 2,
                bit_depth: 32,
                codec_header: None,
            },
        });
        for base in (0..frames).step_by(window_frames) {
            source.prepare_window(producer.builder_mut(), 0, base as u64, 2, 1000);
            assert!(producer.publish());
        }
        let gate = ControlGate::default();
        let checkpoint = PublishedCheckpoint::new();
        let progress = AtomicU64::new(0);
        let mut reader = Reader::new(2);
        let mut pcm = Vec::new();
        let (mut inserted, mut dropped) = (0, 0);
        for &frames in chunks {
            let mut block = vec![0; frames * 2];
            let result = reader.render(
                &mut device,
                &gate.begin_callback(),
                &checkpoint,
                &progress,
                &mut block,
                true,
                Some(plan),
            );
            assert_eq!(result.missing, 0);
            let snapshot = checkpoint.read(&gate, &progress).unwrap();
            assert_eq!(
                snapshot.position.cursor_us,
                snapshot.source_frames as i64 * 1000
            );
            inserted += result.inserted;
            dropped += result.dropped;
            pcm.extend(block);
        }
        (pcm, progress.load(Ordering::Acquire), inserted, dropped)
    }

    #[test]
    fn correction_smooths_stereo_drop_without_changing_consumption() {
        let plan = CorrectionSchedule {
            drop_every_n_frames: 4,
            ..Default::default()
        };
        let (pcm, consumed, inserted, dropped) = render_ramp(plan, &[3, 1, 4]);
        assert_eq!(
            &pcm[..10],
            &[0, 0, 1000, -1000, 2000, -2000, 3500, -3500, 5000, -5000]
        );
        assert_eq!((consumed, inserted, dropped), (10, 0, 2));
        assert_eq!(pcm, render_ramp(plan, &[8]).0);
    }

    #[test]
    fn correction_interpolates_insert_without_consuming_lookahead() {
        let plan = CorrectionSchedule {
            insert_every_n_frames: 4,
            ..Default::default()
        };
        let (pcm, consumed, inserted, dropped) = render_ramp(plan, &[3, 1, 4]);
        assert_eq!(
            &pcm[..10],
            &[0, 0, 1000, -1000, 2000, -2000, 2500, -2500, 3000, -3000]
        );
        assert_eq!((consumed, inserted, dropped), (6, 2, 0));
        assert_eq!(pcm, render_ramp(plan, &[8]).0);
    }

    #[test]
    fn correction_preserves_window_boundary_and_full_scale_channels() {
        for plan in [
            CorrectionSchedule {
                insert_every_n_frames: 4,
                ..Default::default()
            },
            CorrectionSchedule {
                drop_every_n_frames: 4,
                ..Default::default()
            },
        ] {
            // Correction falls on/across a prepared-window boundary.
            let samples: Vec<_> = (0..32).flat_map(|_| [i32::MAX, i32::MIN]).collect();
            let reference = render_samples(plan, &[8], samples.clone(), 32);
            for window_frames in [1, 3, 4] {
                assert_eq!(
                    render_samples(plan, &[3, 1, 4], samples.clone(), window_frames),
                    reference
                );
            }
            assert_eq!(reference.0, vec![i32::MAX, i32::MIN].repeat(8));
        }
    }

    #[test]
    fn uncorrected_pcm_is_bit_exact() {
        let samples: Vec<_> = (0..32).flat_map(|i| [i * 1000, -i * 1000]).collect();
        let result = render_samples(CorrectionSchedule::default(), &[3, 5], samples.clone(), 3);
        assert_eq!(result, (samples[..16].to_vec(), 8, 0, 0));
    }

    #[test]
    fn insertion_without_lookahead_repeats_last_without_consumption() {
        let plan = CorrectionSchedule {
            insert_every_n_frames: 4,
            ..Default::default()
        };
        let result = render_samples(plan, &[3, 1], vec![100, -100, 200, -200, 300, -300], 3);
        assert_eq!(
            result,
            (vec![100, -100, 200, -200, 300, -300, 300, -300], 3, 1, 0)
        );
    }

    #[test]
    fn production_cadence_halves_ramp_correction_corner() {
        // A ramp has zero second difference away from corrections. Compare
        // its correction corners against the previous hard skip/repeat policy.
        // This is a PCM property, not an assertion about perceived loudness.
        let samples: Vec<_> = (0..2100).flat_map(|i| [i * 1000, -i * 1000]).collect();
        for dropping in [false, true] {
            let plan = CorrectionSchedule {
                insert_every_n_frames: if dropping { 0 } else { 500 },
                drop_every_n_frames: if dropping { 500 } else { 0 },
                ..Default::default()
            };
            let actual = render_samples(plan, &[2000], samples.clone(), 2100);
            let mut source = 0;
            let mut last = 0;
            let hard: Vec<i32> = (1..=2000)
                .map(|frame| {
                    if frame % 500 == 0 {
                        if !dropping {
                            return last;
                        }
                        source += 1;
                    }
                    last = source * 1000;
                    source += 1;
                    last
                })
                .collect();
            let peak_corner = |mono: &[i32]| {
                mono.windows(3)
                    .map(|w| (i64::from(w[2]) - 2 * i64::from(w[1]) + i64::from(w[0])).abs())
                    .max()
                    .unwrap()
            };
            let smooth: Vec<_> = actual.0.iter().step_by(2).copied().collect();
            assert_eq!(peak_corner(&hard), 1000);
            assert_eq!(peak_corner(&smooth), 500);
            assert_eq!(actual.1, source as u64);
            assert_eq!((actual.2, actual.3), if dropping { (0, 4) } else { (4, 0) });
        }
    }
}

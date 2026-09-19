//! Device-owned finite frame selection over prepared, uncorrected PCM.
//! No source queue, clock model, payload allocation or owner mutex is used here.

use super::ingress::{CallbackGuard, PublishedCheckpoint, SourcePosition};
use super::transport::DeviceSide;
use crate::audio::sync_correction::CorrectionSchedule;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default, Debug)]
pub(super) struct Rendered {
    pub consumed: u64,
    pub missing: usize,
    pub inserted: usize,
    pub dropped: usize,
    pub source_cursor_us: Option<i64>,
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
}

impl Reader {
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
        destination: Option<&mut [i32]>,
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
            if self.origin.is_none() {
                self.origin = offset
                    .checked_sub(1)
                    .map(|i| window.frames[i].position.cursor_us)
                    .or(window.cursor_before);
            }
            if offset == window.valid {
                self.position = Some(frame.position);
                *current = frame.position.current;
                *visible -= 1;
                if !pipe.retire() {
                    return false;
                }
                continue;
            }
            if let Some(destination) = destination {
                let channels = self.last.len();
                destination
                    .copy_from_slice(&window.pcm[offset * channels..(offset + 1) * channels]);
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
        self.origin = if cleared || reanchored {
            None
        } else {
            self.position.map(|position| position.cursor_us)
        };
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
        for destination in output.chunks_exact_mut(channels) {
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
                rendered.inserted += 1;
                continue;
            }
            if step == 2
                && self.take_frame(
                    pipe,
                    None,
                    visible,
                    view.epoch(),
                    &mut current,
                    &mut consumed,
                )
            {
                rendered.dropped += 1;
            }
            if self.take_frame(
                pipe,
                Some(destination),
                visible,
                view.epoch(),
                &mut current,
                &mut consumed,
            ) {
                self.last.copy_from_slice(destination);
            } else if step == 2 {
                destination.copy_from_slice(&self.last);
            } else {
                destination.fill(0);
                rendered.missing += 1;
            }
        }
        progress.store(consumed, Ordering::Release);
        if let Some(position) = self.position {
            checkpoint.publish(callback, position);
        }
        callback.publish_current(current);
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
            assert_eq!(single, if error > 0 { [502] } else { [500] });
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

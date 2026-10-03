//! In-memory tests of the production data callback. No stream, device, sleep,
//! network or global clock is used. PCM encodes a generated ascending frame ID
//! in both channels; power-of-two scaling preserves exact f32 sample values.
//! This hermetic callback/queue/renderer contract test belongs to default T2:
//! planner-only tests cannot detect bad input admission or actual frame consumption.

use super::*;
use crate::audio::Codec;
use crate::sync::Clock;
use cpal::{OutputStreamTimestamp, OutputTimestampSource, StreamInstant};

struct CallbackClock(Instant);

impl Clock for CallbackClock {
    fn now_micros(&self) -> i64 {
        0
    }

    fn micros_to_instant(&self, micros: i64) -> Option<Instant> {
        self.0
            .checked_add(Duration::from_micros(micros.try_into().ok()?))
    }

    fn instant_to_micros(&self, instant: Instant) -> i64 {
        instant.duration_since(self.0).as_micros() as i64
    }
}

type DataCallback = Box<
    dyn FnMut(
            &mut [f32],
            OutputStreamTimestamp,
            OutputTimestampSource,
            Option<cpal::OutputTimestampDiagnostics>,
            Instant,
        ) + Send,
>;

pub(super) struct Harness {
    callback: DataCallback,
    worker: runtime::Worker,
    origin: Instant,
    now_us: u64,
    frames: usize,
    diagnostics: SyncDiagnosticsReader,
    owner: RendererOwner,
    scope: PlayerScope,
    queue: Arc<Mutex<PlaybackQueue>>,
    static_delay_us: Arc<AtomicU64>,
}

impl Harness {
    fn new(sample_rate: u32) -> Self {
        Self::with_channels(sample_rate, 2)
    }

    fn with_channels(sample_rate: u32, channels: u8) -> Self {
        Self::configured(
            sample_rate,
            channels,
            RendererQueueLimits::new(sample_rate as usize * 12, 4, sample_rate as usize * 12)
                .unwrap(),
            true,
        )
    }

    fn configured(
        sample_rate: u32,
        channels: u8,
        limits: RendererQueueLimits,
        prefill: bool,
    ) -> Self {
        Self::configured_with_hook(sample_rate, channels, limits, prefill, None)
    }

    fn configured_with_hook(
        sample_rate: u32,
        channels: u8,
        limits: RendererQueueLimits,
        prefill: bool,
        process_callback: Option<ProcessCallback>,
    ) -> Self {
        Self::configured_with_capacity(
            sample_rate,
            channels,
            limits,
            prefill,
            process_callback,
            Some(sample_rate as usize / 100),
        )
    }

    fn configured_with_capacity(
        sample_rate: u32,
        channels: u8,
        limits: RendererQueueLimits,
        prefill: bool,
        process_callback: Option<ProcessCallback>,
        max_callback_frames: Option<usize>,
    ) -> Self {
        let origin = Instant::now(); // Arbitrary epoch; every later instant is injected.
        let clock = Arc::new(Mutex::new(ClockSync::new_same_clock(Arc::new(
            CallbackClock(origin),
        ))));
        let owner = RendererOwner::new(limits);
        let scope = owner.mint_scope().unwrap();
        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate,
            channels,
            bit_depth: 32,
            codec_header: None,
        };
        let chunk_frames = sample_rate as usize * 3;
        for chunk in 0..if prefill { 4 } else { 0 } {
            assert!(matches!(
                owner.enqueue_with_actual(scope, chunk_frames, || {
                    let mut q = queue.lock();
                    q.push(AudioBuffer {
                        timestamp: 1_000_000 + chunk as i64 * 3_000_000,
                        samples: (chunk * chunk_frames + 1..=(chunk + 1) * chunk_frames)
                            .flat_map(|frame| {
                                std::iter::repeat_n(frame as i32 * 1024, channels as usize)
                            })
                            .collect::<Vec<_>>()
                            .into(),
                        format: format.clone(),
                    });
                    (q.queued_frames(channels as usize), q.buffer_count())
                }),
                EnqueueOutcome::Accepted { .. }
            ));
        }
        assert_eq!(
            owner.arm_scheduled_start(scope, if prefill { 1_000_000 } else { 0 }),
            ScheduledArmOutcome::Armed
        );
        let diagnostics = SyncDiagnosticsReader::new(clock.clone());
        let static_delay_us = Arc::new(AtomicU64::new(0));
        let (mut callback, worker) = make_output_callback::<f32>(
            queue.clone(),
            clock,
            format,
            CallbackConfig {
                max_callback_frames,
                gain_control: GainControl::new(100, false),
                process_callback,
                static_delay_us: static_delay_us.clone(),
            },
            owner.clone(),
            scope,
            diagnostics.clone(),
        )
        .unwrap();
        Self {
            callback: Box::new(move |data, timestamp, source, evidence, instant| {
                render_output_channels(data, channels == 1, |input| {
                    callback(input, timestamp, source, evidence, instant)
                });
            }),
            worker,
            origin,
            now_us: if prefill { 823_000 } else { 0 }, // Existing prefills start at 1000ms.
            frames: sample_rate as usize / 100,
            diagnostics,
            owner,
            scope,
            queue,
            static_delay_us,
        }
    }

    pub(super) fn streaming_mono() -> Self {
        Self::configured(
            1_000,
            1,
            RendererQueueLimits::new(32, 8, 16).unwrap(),
            false,
        )
    }

    pub(super) fn streaming_queued_frames(&self) -> usize {
        self.owner.capacity(self.scope).unwrap().current_frames()
    }

    pub(super) fn streaming_cursor_us(&self) -> i64 {
        self.queue.lock().cursor_us
    }

    pub(super) fn streaming_enqueue(&self, first_frame: i64) {
        let frames = 16;
        assert!(matches!(
            self.owner.enqueue_with_actual(self.scope, frames, || {
                let mut queue = self.queue.lock();
                queue.push(AudioBuffer {
                    timestamp: first_frame * 1000,
                    samples: (first_frame + 1..=first_frame + 16)
                        .map(|x| x as i32 * 1024)
                        .collect::<Vec<_>>()
                        .into(),
                    format: AudioFormat {
                        codec: Codec::Pcm,
                        sample_rate: 1_000,
                        channels: 1,
                        bit_depth: 32,
                        codec_header: None,
                    },
                });
                (queue.queued_frames(1), queue.buffer_count())
            }),
            EnqueueOutcome::Accepted { .. }
        ));
    }

    pub(super) fn streaming_render(&mut self, now_us: u64) -> (Vec<i32>, SyncDiagnosticsSnapshot) {
        self.now_us = now_us;
        let (samples, _) = self.render(OutputTimestampSource::DevicePresentation, 0);
        // Existing mono adaptation duplicates each normalized f32 sample.
        let ids = samples
            .into_iter()
            .step_by(2)
            .map(|x| (x * 2_097_152.0).round() as i32)
            .collect();
        (ids, self.diagnostics.snapshot())
    }

    fn render(&mut self, source: OutputTimestampSource, latency_us: u64) -> (Vec<f32>, u64) {
        self.render_with_diagnostics(source, latency_us, None)
    }

    fn render_with_diagnostics(
        &mut self,
        source: OutputTimestampSource,
        latency_us: u64,
        evidence: Option<cpal::OutputTimestampDiagnostics>,
    ) -> (Vec<f32>, u64) {
        let captured_at = self.origin + Duration::from_micros(self.now_us);
        self.worker.step(captured_at);
        let before = self.owner.health(self.scope).unwrap().consumed_frames();
        assert_eq!(self.owner.consumed_frames(self.scope).unwrap(), before);
        let mut data = vec![0.0; self.frames * 2];
        let callback = StreamInstant::from_nanos(self.now_us * 1_000);
        (self.callback)(
            &mut data,
            OutputStreamTimestamp {
                callback,
                playback: callback + Duration::from_micros(latency_us),
            },
            source,
            evidence,
            self.origin + Duration::from_micros(self.now_us),
        );
        self.worker.step(captured_at);
        self.now_us += 10_000;
        let after = self.owner.health(self.scope).unwrap().consumed_frames();
        assert_eq!(self.owner.consumed_frames(self.scope).unwrap(), after);
        (data, after - before)
    }

    fn warm(&mut self, source: OutputTimestampSource) {
        let (first, _) = self.render(source, 167_000);
        assert!(first.iter().all(|s| *s == 0.0));
        for _ in 0..200 {
            let (data, consumed) = self.render(source, 167_000);
            assert!(data.iter().all(|s| *s > 0.0));
            assert_eq!(consumed, self.frames as u64);
        }
        assert_eq!(self.diagnostics.snapshot().inserted_frames, 0);
        assert_eq!(self.diagnostics.snapshot().dropped_frames, 0);
    }
}

#[test]
fn callback_capacity_512_frames_keeps_continuous_pcm() {
    assert_callback_capacity(&[512], Some(512));
}

#[test]
fn callback_capacity_2048_frames_keeps_continuous_pcm() {
    assert_callback_capacity(&[2048], Some(2048));
}

#[test]
fn callback_capacity_period_changes_keep_continuous_pcm() {
    assert_callback_capacity(&[512, 2048, 4096, 512], Some(4096));
}

#[test]
fn callback_capacity_unknown_backend_uses_source_horizon() {
    assert_callback_capacity(&[8192], None);
}

fn capacity_harness(max_callback_frames: Option<usize>) -> Harness {
    Harness::configured_with_capacity(
        44_100,
        2,
        RendererQueueLimits::new(529_200, 4, 529_200).unwrap(),
        true,
        None,
        max_callback_frames,
    )
}

// T2: exercise actual preparation, partial-window ownership and callback reads.
// An arithmetic capacity test cannot detect entry-budget or retirement errors.
fn assert_callback_capacity(requests: &[usize], capacity: Option<usize>) {
    let mut h = capacity_harness(capacity);
    h.warm(OutputTimestampSource::DevicePresentation);
    let start_us = h.now_us;
    let mut total = 0;
    for period in 0..48 {
        let frames = requests[period % requests.len()];
        h.frames = frames;
        h.now_us = start_us + total as u64 * 1_000_000 / 44_100;
        let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert_eq!(
            consumed, frames as u64,
            "callback {period}, {frames} frames"
        );
        for (offset, stereo) in data.chunks_exact(2).enumerate() {
            let expected = (88_201 + total + offset) as f32 / 2_097_152.0;
            assert_eq!(
                stereo,
                [expected, expected],
                "callback {period}, frame {offset}"
            );
        }
        assert_eq!(h.owner.health(h.scope).unwrap().underrun_frames(), 0);
        total += frames;
    }
}

#[test]
fn callback_capacity_late_prefix_and_large_output_share_one_budget() {
    let mut h = capacity_harness(Some(2048));
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    h.frames = 2048;
    h.now_us = 873_000; // 40 ms late: discard 1764 source frames first.
    let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert_eq!(consumed, 3812);
    for (offset, stereo) in data.chunks_exact(2).enumerate() {
        let expected = (1765 + offset) as f32 / 2_097_152.0;
        assert_eq!(stereo, [expected, expected]);
    }
    assert_eq!(h.owner.health(h.scope).unwrap().underrun_frames(), 0);
}

#[test]
fn callback_capacity_active_drop_correction_has_extra_source_frames() {
    let mut h = capacity_harness(Some(2048));
    h.warm(OutputTimestampSource::DevicePresentation);
    h.frames = 2048;
    let start_us = h.now_us;
    // Preserve the canonical observation-count engagement gate before asserting
    // actual correction; ninety large periods are not enough to engage it.
    for period in 0..200 {
        h.now_us = start_us + period * 2048 * 1_000_000 / 44_100;
        let (data, _) = h.render(OutputTimestampSource::DevicePresentation, 267_000);
        assert!(data.iter().all(|sample| *sample > 0.0));
        assert_eq!(h.owner.health(h.scope).unwrap().underrun_frames(), 0);
    }
    assert!(h.diagnostics.snapshot().dropped_frames > 0);
}

#[test]
fn callback_continues_after_contended_diagnostic_read_is_skipped() {
    let mut h = Harness::new(44_100);
    h.warm(OutputTimestampSource::DevicePresentation);
    let owner = h.owner.clone();
    let scope = h.scope;
    let permit = owner.try_callback_permit(scope).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = {
        let owner = owner.clone();
        std::thread::spawn(move || {
            tx.send((owner.needs_terminal_check(scope), owner.try_health(scope)))
                .unwrap();
        })
    };
    // The channel observes completion while playback state is unavailable.
    // Timeout is only a failure guard; release and join even on regression.
    let observed = rx.recv_timeout(Duration::from_secs(1));
    drop(permit);
    reader.join().unwrap();
    let (needs_check, health) = observed.expect("diagnostics must not wait for playback state");
    assert!(!needs_check);
    assert_eq!(health.unwrap(), None);
    let before = h.diagnostics.snapshot();
    let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert!(data.iter().all(|sample| *sample > 0.0));
    assert_eq!(consumed, 441);
    assert_eq!(
        h.diagnostics.snapshot().access_silence_frames,
        before.access_silence_frames
    );
    assert!(owner.try_health(scope).unwrap().is_some());
    assert_eq!(owner.close(scope), RendererOperationOutcome::Applied);
    assert!(owner.needs_terminal_check(scope));
    assert!(owner.health(scope).unwrap().terminal().is_some());
}

#[test]
fn callback_realtime_renderer_contention_preserves_prepared_pcm_and_consumption() {
    assert_prepared_playback_survives_contention(true);
}

#[test]
fn callback_realtime_source_contention_preserves_prepared_pcm_and_consumption() {
    assert_prepared_playback_survives_contention(false);
}

fn assert_prepared_playback_survives_contention(block_renderer: bool) {
    let mut h = Harness::new(44_100);
    h.warm(OutputTimestampSource::DevicePresentation);
    let before = h.diagnostics.snapshot();
    let consumed_before = h.owner.consumed_frames(h.scope).unwrap();
    let owner = h.owner.clone();
    let queue = h.queue.clone();
    // Acquiring the guard establishes the contention before calling the actual
    // production callback. No thread scheduling or elapsed-time assertion is used.
    let permit = block_renderer.then(|| owner.try_callback_permit(h.scope).unwrap());
    let queue_guard = (!block_renderer).then(|| queue.lock());
    let callback = StreamInstant::from_nanos(h.now_us * 1_000);
    let mut data = vec![0.0; 882];
    (h.callback)(
        &mut data,
        OutputStreamTimestamp {
            callback,
            playback: callback + Duration::from_micros(167_000),
        },
        OutputTimestampSource::DevicePresentation,
        Some(cpal::OutputTimestampDiagnostics {
            output_xrun_count: Some(3),
            output_buffer_size_frames: Some(1024),
            ..Default::default()
        }),
        h.origin + Duration::from_micros(h.now_us),
    );
    h.now_us += 10_000;
    drop(queue_guard);
    drop(permit);
    h.worker
        .step(h.origin + Duration::from_micros(h.now_us - 10_000));

    // Warm-up consumed 200 periods of 441 generated frames. Both channels
    // must carry the next source IDs, not merely a nonzero/repeated last frame.
    let ids: Vec<i32> = data
        .chunks_exact(2)
        .map(|frame| {
            assert_eq!(frame[0], frame[1]);
            (frame[0] * 2_097_152.0).round() as i32
        })
        .collect();
    assert_eq!(ids, (88_201..=88_641).collect::<Vec<_>>());
    let health = h.owner.health(h.scope).unwrap();
    assert_eq!(health.consumed_frames() - consumed_before, 441);
    assert_eq!(health.underrun_frames(), 0);
    let snapshot = h.diagnostics.snapshot();
    assert_eq!(snapshot.requested_frames - before.requested_frames, 441);
    assert_eq!(snapshot.silent_frames, before.silent_frames);
    assert_eq!(snapshot.access_silence_frames, before.access_silence_frames);
    assert_eq!(snapshot.underrun_frames, before.underrun_frames);
    assert_eq!(snapshot.output_xrun_count, Some(3));
    assert_eq!(snapshot.output_buffer_size_frames, Some(1024));

    let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert_eq!(consumed, 441);
    assert_eq!((data[0] * 2_097_152.0).round() as i32, 88_642);
    assert_eq!(h.diagnostics.snapshot().output_xrun_count, None);
}

#[test]
fn callback_valid_and_unspecified_timestamps_start_and_remain_aligned() {
    for rate in [44_100, 96_000] {
        for source in [
            OutputTimestampSource::DevicePresentation,
            OutputTimestampSource::Unspecified,
        ] {
            let mut h = Harness::new(rate);
            h.warm(source);
            let mut next_frame = rate as usize * 2 + 1;
            for _ in 0..200 {
                let (data, consumed) = h.render(source, 167_000);
                assert_eq!(consumed, h.frames as u64);
                for stereo in data.chunks_exact(2) {
                    let expected = next_frame as f32 / 2_097_152.0;
                    assert_eq!(stereo, [expected, expected]);
                    next_frame += 1;
                }
            }
        }
    }
}

#[test]
fn callback_fallback_does_not_create_correction_or_poison_recovery() {
    for rate in [44_100, 96_000] {
        for period in [200, 20] {
            let mut h = Harness::new(rate);
            h.warm(OutputTimestampSource::DevicePresentation);
            for tick in 0..300 {
                let fallback = tick % period == 0;
                let (source, delay) = if fallback {
                    (OutputTimestampSource::MonotonicFallback, 0)
                } else {
                    (OutputTimestampSource::DevicePresentation, 167_000)
                };
                let (data, consumed) = h.render(source, delay);
                assert!(data.iter().all(|s| *s > 0.0));
                assert_eq!(
                    consumed, h.frames as u64,
                    "false correction at {rate} Hz, tick {tick}"
                );
                let diagnostic = h.diagnostics.snapshot();
                if fallback {
                    assert_eq!(
                        diagnostic.raw_error_us, None,
                        "fallback is not a device measurement"
                    );
                    assert_eq!(diagnostic.filtered_error_us, None);
                }
                assert_eq!(diagnostic.correction_reanchors, 0);
            }
        }
    }
}

#[test]
fn callback_fallback_suspends_active_correction_without_rewarming() {
    for rate in [44_100, 96_000] {
        let mut h = Harness::new(rate);
        h.warm(OutputTimestampSource::DevicePresentation);
        for _ in 0..200 {
            h.render(OutputTimestampSource::DevicePresentation, 267_000);
        }
        assert!(h.diagnostics.snapshot().drop_every > 0);
        for _ in 0..10 {
            let (data, consumed) = h.render(OutputTimestampSource::MonotonicFallback, 0);
            assert!(data.iter().all(|s| *s > 0.0));
            assert_eq!(consumed, h.frames as u64);
            let diagnostic = h.diagnostics.snapshot();
            assert_eq!((diagnostic.insert_every, diagnostic.drop_every), (0, 0));
            h.render(OutputTimestampSource::DevicePresentation, 267_000);
            assert!(
                h.diagnostics.snapshot().drop_every > 0,
                "valid history must survive fallback"
            );
            assert_eq!(h.diagnostics.snapshot().insert_every, 0);
        }
    }
}

#[test]
fn callback_fallback_only_start_respects_future_boundary_and_remains_audible() {
    for rate in [44_100, 96_000] {
        let mut h = Harness::new(rate);
        for _ in 0..60 {
            let callback_us = h.now_us;
            let (data, _) = h.render(OutputTimestampSource::MonotonicFallback, 0);
            if callback_us + 10_000 < 1_000_000 {
                assert!(
                    data.iter().all(|s| *s == 0.0),
                    "output before scheduled boundary"
                );
            }
            if callback_us >= 1_010_000 {
                assert!(
                    data.iter().all(|s| *s > 0.0),
                    "fallback must not prevent startup"
                );
            }
            assert_eq!(h.diagnostics.snapshot().raw_error_us, None);
        }
        // The first device measurement follows startup based only on the
        // one-buffer estimate. Its positive latency exposes real lateness;
        // correction must warm from these measurements and consume faster,
        // while output remains continuous throughout the transition.
        let mut extra_consumed = 0;
        for _ in 0..200 {
            let (audio, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
            assert!(audio.iter().all(|sample| *sample > 0.0));
            assert!(h
                .diagnostics
                .snapshot()
                .raw_error_us
                .is_some_and(|error| error > 0));
            assert!(consumed >= h.frames as u64);
            extra_consumed += consumed - h.frames as u64;
            assert_eq!(h.diagnostics.snapshot().insert_every, 0);
            assert_eq!(h.diagnostics.snapshot().correction_reanchors, 0);
        }
        assert!(
            extra_consumed > 0,
            "valid observations must enable actual correction"
        );
    }
}

#[test]
fn callback_actual_speed_stays_within_half_percent_over_150ms() {
    for rate in [44_100, 96_000] {
        for latency_us in [67_000, 267_000] {
            let mut h = Harness::new(rate);
            h.warm(OutputTimestampSource::DevicePresentation);
            let mut window = VecDeque::new();
            let window_frames = rate as usize * 150 / 1000;
            for tick in 0..450 {
                // Change the target during an active episode as well as holding it steady.
                let latency = if tick >= 250 {
                    334_000 - latency_us
                } else {
                    latency_us
                };
                let (data, _) = h.render(OutputTimestampSource::DevicePresentation, latency);
                // Decode actual output PCM, independently of the planner and its
                // diagnostic counters. Check every frame-offset 150ms window,
                // including windows crossing callback and schedule boundaries.
                for stereo in data.chunks_exact(2) {
                    assert_eq!(stereo[0], stereo[1]);
                    assert!(stereo[0] > 0.0);
                    let frame_id = (stereo[0] * 2_097_152.0).round() as i64;
                    window.push_back(frame_id);
                    if window.len() > window_frames + 1 {
                        window.pop_front();
                    }
                    if window.len() == window_frames + 1 {
                        let progress = frame_id - window.front().unwrap();
                        let change = (progress - window_frames as i64).unsigned_abs();
                        assert!(
                            change * 200 <= window_frames as u64,
                            "speed exceeds ±0.5% at {rate} Hz: {change} changed frames / {window_frames}"
                        );
                    }
                }
            }
            assert!(h.diagnostics.snapshot().inserted_frames > 0);
            assert!(h.diagnostics.snapshot().dropped_frames > 0);
            assert_eq!(h.diagnostics.snapshot().correction_reanchors, 0);
        }
    }
}

#[test]
fn callback_retains_fallback_evidence_after_valid_observation() {
    use cpal::OutputTimestampFallbackReason::*;
    let mut h = Harness::new(44_100);
    h.warm(OutputTimestampSource::DevicePresentation);
    for reason in [
        Unavailable,
        Unsupported,
        Invalid,
        NonMonotonic,
        ClockDomainMismatch,
    ] {
        let evidence = cpal::OutputTimestampDiagnostics {
            fallback_reason: Some(reason),
            query_error_code: (reason == Unavailable).then_some(-899),
            projected_step_ns: (reason == NonMonotonic).then_some(-1_000_000),
            ..Default::default()
        };
        let (_, consumed) =
            h.render_with_diagnostics(OutputTimestampSource::MonotonicFallback, 0, Some(evidence));
        assert_eq!(consumed, h.frames as u64);
        let event_callback = h.diagnostics.snapshot().callbacks;
        h.render(OutputTimestampSource::DevicePresentation, 167_000);
        let snapshot = h.diagnostics.snapshot();
        assert_eq!(snapshot.last_timestamp_fallback, Some(evidence));
        assert_eq!(snapshot.last_timestamp_fallback_callback, event_callback);
    }
    let snapshot = h.diagnostics.snapshot();
    assert_eq!(
        [
            snapshot.fallback_unavailable,
            snapshot.fallback_unsupported,
            snapshot.fallback_invalid,
            snapshot.fallback_non_monotonic,
            snapshot.fallback_clock_domain_mismatch
        ],
        [1; 5]
    );
}

#[test]
fn callback_fallback_cannot_poison_explicit_reanchor_latency_floor() {
    for rate in [44_100, 96_000] {
        let mut h = Harness::new(rate);
        h.warm(OutputTimestampSource::DevicePresentation);
        h.render(OutputTimestampSource::MonotonicFallback, 0);
        set_device_delay_state(
            &h.queue,
            &h.static_delay_us,
            DeviceDelayMs::new(10.0).unwrap(),
        );
        let (audio, _) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert!(audio.iter().all(|sample| *sample > 0.0));
        let snapshot = h.diagnostics.snapshot();
        assert_eq!(snapshot.startup_reanchors, 2);
        assert_eq!(snapshot.raw_error_us, Some(0));
        assert_eq!(snapshot.correction_reanchors, 0);
    }
}

#[test]
fn mono_output_preserves_stereo_frame_timing_and_consumption() {
    let mut mono = Harness::with_channels(48_000, 1);
    let mut stereo = Harness::new(48_000);
    for _ in 0..220 {
        let (actual, consumed) = mono.render(OutputTimestampSource::DevicePresentation, 167_000);
        let (expected, expected_consumed) =
            stereo.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert_eq!(actual, expected);
        assert_eq!(consumed, expected_consumed);
    }
    assert_eq!(mono.diagnostics.snapshot().inserted_frames, 0);
    assert_eq!(mono.diagnostics.snapshot().dropped_frames, 0);
}

#[test]
fn callback_fixed_scratch_hook_processes_every_sample_after_gain() {
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    let origin = Instant::now();
    let clock = Arc::new(Mutex::new(ClockSync::new_same_clock(Arc::new(
        CallbackClock(origin),
    ))));
    let limits = RendererQueueLimits::new(128, 4, 128).unwrap();
    let owner = RendererOwner::new(limits);
    let scope = owner.mint_scope().unwrap();
    let format = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 1_000,
        channels: 1,
        bit_depth: 32,
        codec_header: None,
    };
    let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
    assert!(matches!(
        owner.enqueue_with_actual(scope, 128, || {
            let mut queue = queue.lock();
            queue.push(AudioBuffer {
                timestamp: 1_000,
                samples: vec![1 << 30; 128].into(),
                format: format.clone(),
            });
            (queue.queued_frames(1), queue.buffer_count())
        }),
        EnqueueOutcome::Accepted { .. }
    ));
    let gain = GainControl::new(100, false);
    gain.set_linear_gain(0.5);
    let playing = Arc::new(AtomicBool::new(false));
    let observed = Arc::new(AtomicUsize::new(0));
    let hook_playing = Arc::clone(&playing);
    let hook_observed = Arc::clone(&observed);
    let config = CallbackConfig {
        max_callback_frames: Some(10),
        gain_control: gain,
        static_delay_us: Arc::new(AtomicU64::new(0)),
        process_callback: Some(Box::new(move |samples| {
            let expected = if hook_playing.load(Ordering::Relaxed) {
                0.25
            } else {
                0.0
            };
            for sample in samples.iter_mut() {
                assert_eq!(*sample, expected);
                *sample += 0.125;
            }
            hook_observed.fetch_add(samples.len(), Ordering::Relaxed);
        })),
    };
    let diagnostics = SyncDiagnosticsReader::new(Arc::clone(&clock));
    let (mut callback, mut worker) = make_output_callback::<f32>(
        queue,
        clock,
        format,
        config,
        owner.clone(),
        scope,
        diagnostics,
    )
    .unwrap();
    let timestamp = |micros: u64| OutputStreamTimestamp {
        callback: StreamInstant::from_nanos(micros * 1_000),
        playback: StreamInstant::from_nanos(micros * 1_000),
    };
    worker.step(origin);
    let mut silence = [0.0];
    callback(
        &mut silence,
        timestamp(0),
        OutputTimestampSource::DevicePresentation,
        None,
        origin,
    );
    assert_eq!(silence, [0.125]);
    worker.step(origin);
    playing.store(true, Ordering::Relaxed);
    let now = origin + Duration::from_millis(1);
    worker.step(now);
    // K=B=20 at 1 kHz: this single device buffer crosses three scratch blocks.
    let mut output = [0.0; 45];
    callback(
        &mut output,
        timestamp(1_000),
        OutputTimestampSource::DevicePresentation,
        None,
        now,
    );
    assert_eq!(output, [0.375; 45]);
    assert_eq!(observed.load(Ordering::Relaxed), 46);
    assert_eq!(owner.consumed_frames(scope).unwrap(), 45);
}

#[test]
fn callback_repeated_delay_updates_keep_pending_pcm_and_observation_paired() {
    let mut h = Harness::new(1_000);
    h.warm(OutputTimestampSource::DevicePresentation);
    let now = h.origin + Duration::from_micros(h.now_us);
    // No device consumption between setters: even with all credits initially
    // free this exhausts the fixed handoff allowance without adding windows.
    for delay in 1..=5 {
        set_device_delay_state(
            &h.queue,
            &h.static_delay_us,
            DeviceDelayMs::new(delay as f64).unwrap(),
        );
        h.worker.step(now);
        if delay == 1 {
            assert_eq!(h.queue.lock().cursor_us, h.now_us as i64 + 167_000 + 1_000);
        }
    }
    let first_anchor = h.now_us as i64 + 167_000 + 1_000;
    let latest_delay = 5_000;
    let latest_target = h.now_us as i64 + 167_000 + latest_delay;
    assert!(h.queue.lock().force_reanchor);
    let before = h.owner.consumed_frames(h.scope).unwrap();
    let (pending_audio, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert!(pending_audio.iter().all(|sample| *sample > 0.0));
    assert!(
        consumed > 0,
        "pending delay must not prevent credit reclamation"
    );
    assert!(h.owner.consumed_frames(h.scope).unwrap() > before);
    // This callback still used the first committed delay. Pairing its old PCM
    // with the latest requested delay would produce a spurious nonzero error.
    assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
    assert!(
        !h.queue.lock().force_reanchor,
        "reclaimed credit applies latest delay"
    );
    assert_eq!(consumed, 10);
    // Reanchor changes the time cursor without rewinding the current PCM index.
    // The next anchor therefore clamps to actual cursor progress, not the frame
    // ID encoded in PCM (which intentionally retains its original source index).
    let latest_anchor = (first_anchor + consumed as i64 * 1_000).max(latest_target);
    assert_eq!(h.queue.lock().cursor_us, latest_anchor);
    let callback_us = h.now_us;
    let (audio, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert!(audio.iter().all(|sample| *sample > 0.0));
    assert!(consumed > 0);
    let expected_error = callback_us as i64 + 167_000 + latest_delay - latest_anchor;
    // The first callback on the new anchor now consumes its late prefix before
    // emitting PCM, so this handoff no longer leaves a 4ms correction tail.
    assert_eq!(consumed, 10 + expected_error as u64 / 1_000);
    assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
}

#[test]
fn renderer_realtime_worker_join_finishes_before_terminal_ack() {
    use crate::audio::player_contract::TerminalState;
    use std::sync::{atomic::AtomicBool, mpsc};
    use std::thread;

    struct WorkerClock {
        origin: Instant,
        entered: mpsc::Sender<Option<String>>,
        release: Mutex<mpsc::Receiver<()>>,
        dropped: Arc<AtomicBool>,
    }
    impl Clock for WorkerClock {
        fn now_micros(&self) -> i64 {
            0
        }
        fn micros_to_instant(&self, micros: i64) -> Option<Instant> {
            self.origin
                .checked_add(Duration::from_micros(micros.try_into().ok()?))
        }
        fn instant_to_micros(&self, instant: Instant) -> i64 {
            instant.duration_since(self.origin).as_micros() as i64
        }
    }
    impl Drop for WorkerClock {
        fn drop(&mut self) {
            self.entered
                .send(thread::current().name().map(str::to_owned))
                .unwrap();
            self.release.get_mut().recv().unwrap();
            self.dropped.store(true, Ordering::Release);
        }
    }
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let dropped = Arc::new(AtomicBool::new(false));
    let clock = Arc::new(Mutex::new(ClockSync::new_same_clock(Arc::new(
        WorkerClock {
            origin: Instant::now(),
            entered: entered_tx,
            release: Mutex::new(release_rx),
            dropped: Arc::clone(&dropped),
        },
    ))));
    let owner = RendererOwner::new(RendererQueueLimits::new(128, 4, 128).unwrap());
    let scope = owner.mint_scope().unwrap();
    let format = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 1_000,
        channels: 1,
        bit_depth: 32,
        codec_header: None,
    };
    let diagnostics = SyncDiagnosticsReader::new(Arc::clone(&clock));
    let config = CallbackConfig {
        max_callback_frames: Some(10),
        gain_control: GainControl::new(100, false),
        process_callback: None,
        static_delay_us: Arc::new(AtomicU64::new(0)),
    };
    let (device, worker) = runtime::build(
        Arc::new(Mutex::new(PlaybackQueue::new())),
        clock,
        &format,
        &config,
        owner.clone(),
        scope,
        diagnostics,
    )
    .unwrap();
    // No stream is installed: this test isolates the real preparation thread's
    // join boundary, and does not claim to exercise CPAL callback stopping.
    drop(device);
    owner
        .attach_preparation_resource(scope, Box::new(worker.spawn().unwrap()))
        .unwrap();
    let (ack_tx, ack_rx) = mpsc::channel();
    let finalizer = {
        let owner = owner.clone();
        thread::spawn(move || ack_tx.send(owner.teardown(scope)).unwrap())
    };
    let drop_thread = entered_rx.recv().unwrap();
    let before_release = owner.terminal_state(scope);
    let premature_ack = ack_rx.try_recv();
    let premature_drop = dropped.load(Ordering::Acquire);
    // Release before asserting, so an assertion cannot strand the worker.
    release_tx.send(()).unwrap();
    finalizer.join().unwrap();
    let outcome = ack_rx.recv().unwrap();
    assert_eq!(drop_thread.as_deref(), Some("sendspin-prepare"));
    assert!(matches!(
        before_release,
        Ok(TerminalState::Finalizing { .. })
    ));
    assert_eq!(premature_ack, Err(mpsc::TryRecvError::Empty));
    assert!(!premature_drop);
    assert!(dropped.load(Ordering::Acquire));
    let TerminalOutcome::Won(finalization) = outcome else {
        panic!("single terminal caller must win");
    };
    assert!(!finalization.ack.stream_released());
    assert!(!finalization.ack.callback_stopped());
}

// T2: actual callback + reader + fixed transport + source reconciliation. A
// reader-only test cannot detect preparation accidentally advancing public
// consumption, or a callback pulling canonical PCM after prepared PCM runs out.
#[test]
fn callback_worker_pause_drains_prepared_pcm_then_resumes_without_skipping() {
    let mut h = Harness::new(1_000);
    let (silence, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert_eq!(silence, vec![0.0; 20]);
    assert_eq!(consumed, 0);
    // This production configuration starts with three 20-frame windows; 60 is
    // this fixture's prepared coverage, not a product minimum-buffer promise.
    // Requesting 75 crosses that finite supply while canonical PCM has 12 s.
    // Deliberately invoke the real callback without either worker step.
    let mut render_paused = |frames: usize| {
        let now = h.origin + Duration::from_micros(h.now_us);
        let callback = StreamInstant::from_nanos(h.now_us * 1_000);
        let mut output = vec![-1.0; frames * 2];
        (h.callback)(
            &mut output,
            OutputStreamTimestamp {
                callback,
                playback: callback + Duration::from_micros(167_000),
            },
            OutputTimestampSource::DevicePresentation,
            None,
            now,
        );
        h.now_us += frames as u64 * 1_000;
        output
    };
    let first = render_paused(75);
    let after_first = h.owner.health(h.scope).unwrap();
    let exhausted = render_paused(10);
    drop(render_paused);
    let expected: Vec<_> = (1..=60)
        .flat_map(|frame| [frame as f32 / 2_097_152.0; 2])
        .chain(std::iter::repeat_n(0.0, 30))
        .collect();
    assert_eq!(first, expected);
    assert_eq!(after_first.consumed_frames(), 60);
    assert_eq!(after_first.underrun_frames(), 15);
    assert_eq!(exhausted, vec![0.0; 20]);
    assert_eq!(h.owner.consumed_frames(h.scope), Ok(60));
    let health = h.owner.health(h.scope).unwrap();
    assert_eq!(health.consumed_frames(), 60);
    assert_eq!(health.underrun_frames(), 25);
    assert!(h.queue.lock().queued_frames(2) > 1_000);
    // The worker settles exactly the actual 60 frames, not all prepared frames
    // or the 85 requested frames. Recovery must resume the next source frame.
    let (resumed, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    let expected: Vec<_> = (61..=70)
        .flat_map(|frame| [frame as f32 / 2_097_152.0; 2])
        .collect();
    assert_eq!(resumed, expected);
    assert_eq!(consumed, 10);
    assert_eq!(h.owner.consumed_frames(h.scope), Ok(70));
    assert_eq!(h.owner.health(h.scope).unwrap().underrun_frames(), 25);
    assert_eq!(h.diagnostics.snapshot().underrun_frames, 25);
}

// T2: the hook is only a controlled scheduling boundary inside the real
// callback's ACTIVE extent. The host replacement owns/join-stops that actual
// callback thread; it does not implement consumption or terminal decisions.
#[test]
fn callback_inflight_consumption_remains_final_while_owner_joins_resource() {
    use crate::audio::player_contract::TerminalState;
    use std::sync::{atomic::AtomicBool, mpsc};
    use std::thread::{self, JoinHandle};

    struct CallbackHost {
        thread: Option<JoinHandle<Vec<f32>>>,
        dropping: mpsc::Sender<()>,
        output: mpsc::Sender<Vec<f32>>,
    }
    impl Drop for CallbackHost {
        fn drop(&mut self) {
            self.dropping.send(()).unwrap();
            let output = self.thread.take().unwrap().join().unwrap();
            self.output.send(output).unwrap();
        }
    }
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let armed = Arc::new(AtomicBool::new(false));
    let hook_armed = Arc::clone(&armed);
    let mut h = Harness::configured_with_hook(
        1_000,
        2,
        RendererQueueLimits::new(12_000, 4, 12_000).unwrap(),
        true,
        Some(Box::new(move |_| {
            if hook_armed.swap(false, Ordering::AcqRel) {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            }
        })),
    );
    let (silence, _) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert_eq!(silence, vec![0.0; 20]);
    armed.store(true, Ordering::Release);
    let exited = Arc::new(AtomicBool::new(false));
    let callback_exited = Arc::clone(&exited);
    let mut callback = h.callback;
    let captured_at = h.origin + Duration::from_micros(h.now_us);
    let instant = StreamInstant::from_nanos(h.now_us * 1_000);
    let callback_thread = thread::spawn(move || {
        let mut output = vec![-1.0; 20];
        callback(
            &mut output,
            OutputStreamTimestamp {
                callback: instant,
                playback: instant + Duration::from_micros(167_000),
            },
            OutputTimestampSource::DevicePresentation,
            None,
            captured_at,
        );
        drop(callback);
        callback_exited.store(true, Ordering::Release);
        output
    });
    entered_rx.recv().unwrap();
    let active_consumed = h.owner.consumed_frames(h.scope);
    let (dropping_tx, dropping_rx) = mpsc::channel();
    let (output_tx, output_rx) = mpsc::channel();
    assert_eq!(
        h.owner.attach_test_terminal_resource(
            h.scope,
            Box::new(CallbackHost {
                thread: Some(callback_thread),
                dropping: dropping_tx,
                output: output_tx,
            })
        ),
        RendererOperationOutcome::Applied
    );
    let (ack_tx, ack_rx) = mpsc::channel();
    let finalizer = {
        let owner = h.owner.clone();
        let scope = h.scope;
        thread::spawn(move || ack_tx.send(owner.teardown(scope)).unwrap())
    };
    // Resource Drop proves the real owner has closed its gate and claimed the
    // finalizer. It is now joining a real callback which remains inside Reader's
    // enclosing ACTIVE guard, after publishing its actual frame count.
    dropping_rx.recv().unwrap();
    let phase = h.owner.terminal_state(h.scope);
    let closing_consumed = h.owner.consumed_frames(h.scope);
    let closing_health = h.owner.health(h.scope).unwrap();
    let premature_ack = ack_rx.try_recv();
    let premature_exit = exited.load(Ordering::Acquire);
    release_tx.send(()).unwrap();
    finalizer.join().unwrap();
    let outcome = ack_rx.recv().unwrap();
    let output = output_rx.recv().unwrap();
    assert_eq!(active_consumed, Ok(10));
    assert_eq!(closing_consumed, Ok(10));
    assert_eq!(closing_health.consumed_frames(), 10);
    assert!(matches!(phase, Ok(TerminalState::Finalizing { .. })));
    assert_eq!(premature_ack, Err(mpsc::TryRecvError::Empty));
    assert!(!premature_exit);
    assert!(exited.load(Ordering::Acquire));
    let expected: Vec<_> = (1..=10)
        .flat_map(|frame| [frame as f32 / 2_097_152.0; 2])
        .collect();
    assert_eq!(output, expected);
    assert_eq!(h.owner.consumed_frames(h.scope), Ok(10));
    assert_eq!(h.owner.health(h.scope).unwrap().consumed_frames(), 10);
    let TerminalOutcome::Won(finalization) = outcome else {
        panic!("single finalizer must win");
    };
    assert!(finalization.ack.stream_released());
    assert!(finalization.ack.callback_stopped());
    assert_eq!(
        h.owner.teardown(h.scope),
        TerminalOutcome::AlreadyFinalized(finalization)
    );
    assert_eq!(h.owner.consumed_frames(h.scope), Ok(10));
}

// T2: close/fault publication must compose with the complete real callback,
// not merely one Reader scratch block. The process hook pauses after the first
// 20 of 45 frames while ACTIVE still covers two remaining scratch blocks.
#[test]
fn media_timeline_close_and_fault_freeze_boundary_after_inflight_callback() {
    use crate::audio::player_contract::RendererTerminal;
    use std::sync::{atomic::AtomicBool, mpsc};
    use std::thread;

    for fault in [false, true] {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let armed = Arc::new(AtomicBool::new(false));
        let hook_armed = Arc::clone(&armed);
        let mut h = Harness::configured_with_hook(
            1_000,
            2,
            RendererQueueLimits::new(12_000, 4, 12_000).unwrap(),
            true,
            Some(Box::new(move |_| {
                if hook_armed.swap(false, Ordering::AcqRel) {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                }
            })),
        );
        let (silence, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert_eq!(silence, vec![0.0; 20]);
        assert_eq!(consumed, 0);
        armed.store(true, Ordering::Release);
        let mut callback = h.callback;
        let captured_at = h.origin + Duration::from_micros(h.now_us);
        let instant = StreamInstant::from_nanos(h.now_us * 1_000);
        let in_flight = thread::spawn(move || {
            let mut output = vec![-1.0; 90];
            callback(
                &mut output,
                OutputStreamTimestamp {
                    callback: instant,
                    playback: instant + Duration::from_micros(167_000),
                },
                OutputTimestampSource::DevicePresentation,
                None,
                captured_at,
            );
            (callback, output)
        });
        entered_rx.recv().unwrap();
        let before_close = h.owner.consumed_frames(h.scope);
        let outcome = if fault {
            h.owner.fault(h.scope, RendererFault::CallbackFailed)
        } else {
            h.owner.close(h.scope)
        };
        let closing = h.owner.health(h.scope).unwrap();
        let needs_terminal_check = h.owner.needs_terminal_check(h.scope);
        // Complete the controlled callback before assertions so a failing RED
        // cannot leave a callback thread parked behind the test's own channel.
        release_tx.send(()).unwrap();
        let (mut callback, output) = in_flight.join().unwrap();
        assert_eq!(outcome, RendererOperationOutcome::Applied);
        assert_eq!(before_close, Ok(20));
        assert!(needs_terminal_check);
        assert_eq!(closing.consumed_frames(), 20);
        assert_eq!(closing.terminal(), None, "fault={fault}");
        assert_eq!(closing.fault(), None, "fault={fault}");
        let expected: Vec<_> = (1..=45)
            .flat_map(|frame| [frame as f32 / 2_097_152.0; 2])
            .collect();
        assert_eq!(output, expected);
        let finished = h.owner.health(h.scope).unwrap();
        assert_eq!(finished.consumed_frames(), 45);
        assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_045_000)));
        assert_eq!(h.owner.consumed_frames(h.scope), Ok(45));
        if fault {
            assert_eq!(finished.fault(), Some(RendererFault::CallbackFailed));
        } else {
            assert_eq!(finished.terminal(), Some(RendererTerminal::Closed));
        }
        armed.store(true, Ordering::Release);
        let silent_in_flight = thread::spawn(move || {
            let mut after_close = vec![-1.0; 20];
            callback(
                &mut after_close,
                OutputStreamTimestamp {
                    callback: instant + Duration::from_millis(45),
                    playback: instant + Duration::from_millis(212),
                },
                OutputTimestampSource::DevicePresentation,
                None,
                captured_at + Duration::from_millis(45),
            );
            after_close
        });
        entered_rx.recv().unwrap();
        // A later silent callback is ACTIVE too, but must never hide a terminal
        // result which has already become observable with stable consumption.
        let silent_health = h.owner.health(h.scope).unwrap();
        release_tx.send(()).unwrap();
        let after_close = silent_in_flight.join().unwrap();
        assert_eq!(silent_health.consumed_frames(), 45);
        assert_eq!(silent_health.terminal(), finished.terminal());
        assert_eq!(silent_health.fault(), finished.fault());
        assert_eq!(after_close, vec![0.0; 20]);
        assert_eq!(h.owner.consumed_frames(h.scope), Ok(45));
        let stable = h.owner.health(h.scope).unwrap();
        assert_eq!(stable.consumed_frames(), 45);
        assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_045_000)));
        assert_eq!(stable.terminal(), finished.terminal());
        assert_eq!(stable.fault(), finished.fault());
    }
}

// T2: continuous public admission + production worker + actual data callback.
// Separate deterministic clocks exercise append/prepare/render interleavings;
// no native stream, real sleep or OS scheduling assumption is involved.
// A one-second source lead is replenished every 20ms. Exercise a 40ms prefill
// and a split 10+30ms burst without an intervening preparation turn, as well
// as ordinary 10ms service. This does not model arbitrary Pulse negotiation.
#[test]
fn callback_streaming_append_preserves_continuous_pcm() {
    for rate in [44_100_u32, 48_000] {
        for append_phase_ms in [0, 7] {
            for requests_ms in [&[10_u64][..], &[40][..], &[10, 30][..]] {
                let chunk_frames = rate as usize / 50;
                let request_period_ms: u64 = requests_ms.iter().sum();
                let mut h = Harness::configured(
                    rate,
                    2,
                    RendererQueueLimits::new(rate as usize * 2, 128, chunk_frames).unwrap(),
                    false,
                );
                let enqueue = |h: &Harness, chunk: usize| {
                    let buffer = AudioBuffer {
                        timestamp: 1_000_000 + chunk as i64 * 20_000,
                        samples: (chunk * chunk_frames + 1..=(chunk + 1) * chunk_frames)
                            .flat_map(|frame| [frame as i32 * 1024; 2])
                            .collect::<Vec<_>>()
                            .into(),
                        format: AudioFormat {
                            codec: Codec::Pcm,
                            sample_rate: rate,
                            channels: 2,
                            bit_depth: 32,
                            codec_header: None,
                        },
                    };
                    let mut queue = h.queue.lock();
                    let publication = queue.publication.clone().unwrap();
                    let outcome = queue
                        .try_enqueue_prepared(
                            &publication,
                            &h.owner,
                            h.scope,
                            (buffer, None),
                            chunk_frames,
                        )
                        .unwrap_or_else(|_| panic!("serialized fixture admission must validate"));
                    assert!(matches!(outcome, EnqueueOutcome::Accepted { .. }));
                };
                for chunk in 0..50 {
                    enqueue(&h, chunk);
                }
                // Give the real startup path its initial device-latency observation.
                h.now_us = 823_000;
                let (startup, consumed) =
                    h.render(OutputTimestampSource::DevicePresentation, 167_000);
                assert!(startup.iter().all(|sample| *sample == 0.0));
                assert_eq!(consumed, 0);
                let mut next_chunk = 50;
                let mut expected_frame = 1;
                for ms in 0..2_000_u64 {
                    if ms % 20 == append_phase_ms {
                        enqueue(&h, next_chunk);
                        next_chunk += 1;
                    }
                    let now = h.origin + Duration::from_micros(833_000 + ms * 1_000);
                    h.worker.step(now);
                    if ms % request_period_ms != 0 {
                        continue;
                    }
                    let mut presentation_offset_us = 0;
                    for &request_ms in requests_ms {
                        let callback_frames = rate as usize * request_ms as usize / 1_000;
                        let timestamp = StreamInstant::from_nanos((833_000 + ms * 1_000) * 1_000);
                        let mut output = vec![0.0; callback_frames * 2];
                        (h.callback)(
                            &mut output,
                            OutputStreamTimestamp {
                                callback: timestamp,
                                playback: timestamp
                                    + Duration::from_micros(167_000 + presentation_offset_us),
                            },
                            OutputTimestampSource::DevicePresentation,
                            None,
                            now,
                        );
                        for frame in output.chunks_exact(2) {
                            let expected = [expected_frame as f32 / 2_097_152.0; 2];
                            assert_eq!(
                        frame, expected,
                        "rate={rate} phase={append_phase_ms} requests={requests_ms:?} ms={ms} frame={expected_frame}"
                    );
                            expected_frame += 1;
                        }
                        presentation_offset_us += request_ms * 1_000;
                    }
                }
                h.worker.step(h.origin + Duration::from_micros(2_833_000));
                assert_eq!(h.owner.consumed_frames(h.scope), Ok(rate as u64 * 2));
                assert_eq!(h.diagnostics.snapshot().underrun_frames, 0);
            }
        }
    }
}

// A real replacement invalidates future windows, even after append refreshes.
// Keep the original current chunk's tail, then play only the replacement PCM.
#[test]
fn callback_streaming_replacement_rebuilds_future_pcm() {
    let mut h = Harness::configured(
        1_000,
        2,
        RendererQueueLimits::new(2_000, 128, 200).unwrap(),
        false,
    );
    let enqueue = |h: &Harness, time_ms: i64, first: i32, frames: usize| {
        let buffer = AudioBuffer {
            timestamp: time_ms * 1_000,
            samples: (first..first + frames as i32)
                .flat_map(|frame| [frame * 1024; 2])
                .collect::<Vec<_>>()
                .into(),
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: 1_000,
                channels: 2,
                bit_depth: 32,
                codec_header: None,
            },
        };
        let mut queue = h.queue.lock();
        let publication = queue.publication.clone().unwrap();
        assert!(matches!(
            queue
                .try_enqueue_prepared(&publication, &h.owner, h.scope, (buffer, None), frames,)
                .ok()
                .unwrap(),
            EnqueueOutcome::Accepted { .. }
        ));
    };
    for chunk in 0..10 {
        enqueue(&h, 1_000 + chunk * 20, chunk as i32 * 20 + 1, 20);
    }
    h.now_us = 823_000;
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    let mut actual = Vec::new();
    let mut collect = |samples: Vec<f32>| {
        actual.extend(samples.chunks_exact(2).filter_map(|frame| {
            assert_eq!(frame[0], frame[1]);
            (frame[0] != 0.0).then_some((frame[0] * 2_097_152.0).round() as i32)
        }));
    };
    collect(h.render(OutputTimestampSource::MonotonicFallback, 0).0);
    enqueue(&h, 1_200, 201, 20); // Refresh an unchanged preparation timeline.
    h.worker.step(h.origin + Duration::from_micros(h.now_us));
    // Retain current 1..20; replace every pending original chunk by 1001..1200.
    enqueue(&h, 1_020, 1_001, 200);
    for _ in 0..25 {
        collect(h.render(OutputTimestampSource::MonotonicFallback, 0).0);
    }
    drop(collect);
    let expected: Vec<_> = (1..=20).chain(1_001..=1_200).collect();
    assert_eq!(actual, expected);
    assert_eq!(h.owner.consumed_frames(h.scope), Ok(220));
}

#[test]
fn callback_start_alignment_splits_first_block_at_target_frame() {
    let mut h = Harness::new(1_000);
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    h.now_us = 828_000; // This block presents at 995ms; source starts at 1000ms.
    let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    let ids: Vec<_> = data
        .iter()
        .step_by(2)
        .map(|x| (x * 2_097_152.0).round() as i32)
        .collect();
    assert_eq!(ids, [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]);
    assert_eq!(consumed, 5);
    assert_eq!(h.queue.lock().cursor_us, 1_005_000);
    assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
}

#[test]
fn callback_start_alignment_skips_only_late_source_prefix() {
    let mut h = Harness::new(1_000);
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    h.now_us = 848_000; // This block presents at 1015ms, 15ms after source start.
    let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    let ids: Vec<_> = data
        .iter()
        .step_by(2)
        .map(|x| (x * 2_097_152.0).round() as i32)
        .collect();
    assert_eq!(ids, (16..=25).collect::<Vec<_>>());
    assert_eq!(consumed, 25);
    assert_eq!(h.queue.lock().cursor_us, 1_025_000);
    assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
}

#[test]
fn callback_start_alignment_handles_audio_rates_without_correction_tail() {
    for rate in [44_100, 48_000] {
        for offset_us in [-19_000i64, -5_000, 0, 15_000] {
            let mut h = Harness::new(rate);
            h.render(OutputTimestampSource::DevicePresentation, 167_000);
            h.frames = rate as usize / 50;
            h.now_us = (833_000 + offset_us) as u64;
            let start_us = h.now_us;
            let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
            let silent =
                ((-offset_us).max(0) as u64 * u64::from(rate)).div_ceil(1_000_000) as usize;
            let skipped = (offset_us.max(0) as u64 * u64::from(rate) / 1_000_000) as usize;
            let ids: Vec<_> = data
                .iter()
                .step_by(2)
                .map(|x| (x * 2_097_152.0).round() as usize)
                .collect();
            assert!(ids[..silent].iter().all(|id| *id == 0));
            assert_eq!(
                ids[silent..],
                (skipped + 1..=skipped + h.frames - silent).collect::<Vec<_>>()
            );
            assert_eq!(consumed as usize, skipped + h.frames - silent);
            assert!(
                h.diagnostics.snapshot().raw_error_us.unwrap().abs()
                    <= (1_000_000 / rate + 1) as i64
            );
            // Run past the correction filter's warm-up; no delayed correction
            // may appear merely because the first callback crossed the start.
            for i in 1..=120 {
                h.now_us = start_us + i * 20_000;
                let (_, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
                assert_eq!(consumed, h.frames as u64);
            }
            let diagnostic = h.diagnostics.snapshot();
            assert_eq!(
                (diagnostic.inserted_frames, diagnostic.dropped_frames),
                (0, 0)
            );
            assert_eq!(diagnostic.underrun_frames, 0);
        }
    }
}

#[test]
fn callback_start_alignment_after_clear_uses_new_source_only() {
    let mut h = Harness::new(1_000);
    h.warm(OutputTimestampSource::DevicePresentation);
    {
        let mut queue = h.queue.lock();
        assert_eq!(
            h.owner.clear_with_actual(h.scope, || queue.clear()),
            RendererOperationOutcome::Applied
        );
        assert!(matches!(
            h.owner.enqueue_with_actual(h.scope, 100, || {
                queue.push(AudioBuffer {
                    timestamp: 4_000_000,
                    samples: (10_001..=10_100)
                        .flat_map(|id| [id * 1024; 2])
                        .collect::<Vec<_>>()
                        .into(),
                    format: AudioFormat {
                        codec: Codec::Pcm,
                        sample_rate: 1_000,
                        channels: 2,
                        bit_depth: 32,
                        codec_header: None,
                    },
                });
                (queue.queued_frames(2), queue.buffer_count())
            }),
            EnqueueOutcome::Accepted { .. }
        ));
    }
    h.now_us = 3_823_000;
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    h.now_us = 3_828_000;
    let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    let ids: Vec<_> = data
        .iter()
        .step_by(2)
        .map(|x| (x * 2_097_152.0).round() as i32)
        .collect();
    assert_eq!(ids, [0, 0, 0, 0, 0, 10_001, 10_002, 10_003, 10_004, 10_005]);
    assert_eq!(consumed, 5);
    assert_eq!(h.queue.lock().cursor_us, 4_005_000);
    assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
}

#[test]
fn callback_start_alignment_large_lateness_keeps_finite_window_budget() {
    let mut h = Harness::new(1_000);
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    h.now_us = 1_033_000; // 200ms late, more than all prepared windows.
    let (first, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert!(first.iter().all(|s| *s == 0.0));
    assert!(consumed > 0 && consumed <= 80);
    let mut caught_up = false;
    for _ in 0..10 {
        let presentation = h.now_us + 167_000;
        let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert!(consumed <= 80);
        if data[0] != 0.0 {
            assert_eq!(
                (data[0] * 2_097_152.0).round() as u64,
                (presentation - 1_000_000) / 1000 + 1
            );
            assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
            caught_up = true;
            break;
        }
    }
    assert!(caught_up);
}

#[test]
fn callback_start_alignment_silence_spans_scratch_blocks() {
    let mut h = Harness::new(1_000);
    h.now_us = 773_000;
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    h.frames = 60; // Three scratch blocks; start falls in the third.
    let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    let ids: Vec<_> = data
        .iter()
        .step_by(2)
        .map(|x| (x * 2_097_152.0).round() as i32)
        .collect();
    assert_eq!(&ids[..50], &[0; 50]);
    assert_eq!(&ids[50..], &(1..=10).collect::<Vec<_>>());
    assert_eq!(consumed, 10);
    assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
}

#[test]
fn callback_start_alignment_variable_blocks_preserve_source_continuity() {
    let mut h = Harness::new(44_100);
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    let start_us = 838_000; // First real block is 5ms late.
    let mut output_frames = 0u64;
    let mut last_id = 220; // floor(5ms * 44100).
    for i in 0..160 {
        h.frames = if i % 2 == 0 { 40 } else { 842 };
        h.now_us = start_us + output_frames * 1_000_000 / 44_100;
        let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        let ids: Vec<_> = data
            .iter()
            .step_by(2)
            .map(|x| (x * 2_097_152.0).round() as i32)
            .collect();
        assert_eq!(
            ids,
            (last_id + 1..=last_id + h.frames as i32).collect::<Vec<_>>()
        );
        assert_eq!(consumed, h.frames as u64 + if i == 0 { 220 } else { 0 });
        last_id += h.frames as i32;
        output_frames += h.frames as u64;
    }
    let snapshot = h.diagnostics.snapshot();
    assert_eq!(
        (
            snapshot.inserted_frames,
            snapshot.dropped_frames,
            snapshot.underrun_frames
        ),
        (0, 0, 0)
    );
    assert!(snapshot.raw_error_us.unwrap().abs() <= 23);
}

#[test]
fn callback_start_alignment_retries_when_prefix_exhausts_ready_pcm() {
    let mut h = Harness::new(1_000);
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    h.now_us = 833_000 + 60_000;
    let (data, _) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert!(data.iter().all(|s| *s == 0.0));
    let presentation = h.now_us + 167_000;
    let (data, _) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    assert_eq!(
        (data[0] * 2_097_152.0).round() as u64,
        (presentation - 1_000_000) / 1000 + 1
    );
    assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
}

#[test]
fn media_timeline_future_gap_waits_for_source_time() {
    assert_media_gap(68, 12, 0);
}

#[test]
fn media_timeline_on_time_gap_preserves_source_time() {
    assert_media_gap(80, 0, 0);
}

#[test]
fn media_timeline_gap_inside_callback_pairs_first_real_presentation() {
    assert_media_gap(68, 12, 5_000);
}

fn assert_media_gap(resume_tick: i64, lead_ticks: i64, shift_us: i64) {
    let rate = 48_000;
    let mut h = Harness::configured(
        rate,
        2,
        RendererQueueLimits::new(96_000, 128, 960).unwrap(),
        false,
    );
    let enqueue = |h: &Harness, chunk: i64| {
        let buffer = AudioBuffer {
            timestamp: 1_000_000 + chunk * 20_000 + if chunk >= 40 { shift_us } else { 0 },
            samples: (chunk * 960 + 1..=(chunk + 1) * 960)
                .flat_map(|frame| [frame as i32 * 1024; 2])
                .collect::<Vec<_>>()
                .into(),
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: rate,
                channels: 2,
                bit_depth: 32,
                codec_header: None,
            },
        };
        let mut queue = h.queue.lock();
        let publication = queue.publication.clone().unwrap();
        let outcome = queue
            .try_enqueue_prepared(&publication, &h.owner, h.scope, (buffer, None), 960)
            .unwrap_or_else(|_| panic!("admission rejected"));
        assert!(matches!(outcome, EnqueueOutcome::Accepted { .. }));
    };
    for chunk in 0..10 {
        enqueue(&h, chunk);
    }
    h.now_us = 823_000;
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    for tick in 0..82 {
        if tick >= resume_tick && tick % 2 == 0 {
            enqueue(&h, (tick + lead_ticks) / 2);
        }
        let (output, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        if (20..80).contains(&tick) {
            assert!(
                output.iter().all(|sample| *sample == 0.0),
                "future source played at tick {tick}"
            );
            assert_eq!(consumed, 0);
            assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_200_000)));
        }
        if tick == 80 {
            let silent_frames = (shift_us * 48_000 / 1_000_000) as usize;
            assert!(output[..silent_frames * 2]
                .iter()
                .all(|sample| *sample == 0.0));
            assert_eq!(
                (output[silent_frames * 2] * 2_097_152.0).round() as i32,
                38_401
            );
            assert_eq!(
                h.owner.consumed_frames(h.scope),
                Ok(10_080 - silent_frames as u64)
            );
            assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
            assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_810_000)));
        }
    }
}

#[test]
fn media_timeline_44100_short_tail_and_retirement_only_preserve_exact_end() {
    for retire in [false, true] {
        let mut h = Harness::configured(
            44_100,
            2,
            RendererQueueLimits::new(2_002, 4, 1_001).unwrap(),
            false,
        );
        {
            let mut queue = h.queue.lock();
            let publication = queue.publication.clone().unwrap();
            let buffer = AudioBuffer {
                timestamp: 1_000_000,
                samples: (1..=1001)
                    .flat_map(|frame| [frame * 1024; 2])
                    .collect::<Vec<_>>()
                    .into(),
                format: AudioFormat {
                    codec: Codec::Pcm,
                    sample_rate: 44_100,
                    channels: 2,
                    bit_depth: 32,
                    codec_header: None,
                },
            };
            assert!(matches!(
                queue
                    .try_enqueue_prepared(&publication, &h.owner, h.scope, (buffer, None), 1001)
                    .ok()
                    .unwrap(),
                EnqueueOutcome::Accepted { .. }
            ));
        }
        h.now_us = 823_000;
        h.render(OutputTimestampSource::DevicePresentation, 167_000);
        for _ in 0..2 {
            h.render(OutputTimestampSource::DevicePresentation, 167_000);
        }
        assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_020_000)));
        if retire {
            let mut queue = h.queue.lock();
            let publication = queue.publication.clone().unwrap();
            queue
                .try_reanchor_prepared(&publication, &h.owner, h.scope, || Some(1_030_000), false)
                .unwrap();
            assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_020_000)));
        }
        let (audio, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        if retire {
            assert!(audio.iter().all(|sample| *sample == 0.0));
            assert_eq!(consumed, 0);
        } else {
            assert_eq!(consumed, 119);
            assert_eq!((audio[0] * 2_097_152.0).round() as i32, 883);
            assert_eq!((audio[236] * 2_097_152.0).round() as i32, 1001);
            assert!(audio[238..].iter().all(|sample| *sample == 0.0));
        }
        assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_022_698)));
        assert_eq!(
            h.owner
                .clear_with_actual(h.scope, || h.queue.lock().clear()),
            RendererOperationOutcome::Applied
        );
        assert!(h
            .render(OutputTimestampSource::DevicePresentation, 167_000)
            .0
            .iter()
            .all(|sample| *sample == 0.0));
        assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_022_698)));
    }
}

#[test]
fn media_timeline_entirely_expired_initial_tail_is_disposed_without_audio() {
    let mut h = Harness::configured(44_100, 2, RendererQueueLimits::new(4, 2, 2).unwrap(), false);
    {
        let mut queue = h.queue.lock();
        let publication = queue.publication.clone().unwrap();
        let buffer = AudioBuffer {
            timestamp: 1_000_000,
            samples: Arc::from([1024, 1024, 2048, 2048]),
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: 44_100,
                channels: 2,
                bit_depth: 32,
                codec_header: None,
            },
        };
        assert!(matches!(
            queue
                .try_enqueue_prepared(&publication, &h.owner, h.scope, (buffer, None), 2)
                .ok()
                .unwrap(),
            EnqueueOutcome::Accepted { .. }
        ));
    }
    h.now_us = 2_000_000;
    for _ in 0..2 {
        let (audio, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert!(audio.iter().all(|sample| *sample == 0.0));
        assert_eq!(consumed, 0);
    }
    assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_000_045)));
    assert_eq!(h.owner.capacity(h.scope).unwrap().current_frames(), 0);
}

// T2: prepared queue -> actual callback startup discard must stop at source time,
// including a discontinuity after an explicit control reanchor.
#[test]
fn media_timeline_late_start_waits_at_future_source_after_gap() {
    for reanchor in [false, true] {
        assert_late_start_gap(reanchor, 2_000_000);
    }
}

#[test]
fn media_timeline_late_start_reaches_suffix_inside_same_callback() {
    assert_late_start_gap(false, 1_035_000);
}

fn assert_late_start_gap(reanchor: bool, suffix_start: i64) {
    let mut h = Harness::configured(
        1_000,
        2,
        RendererQueueLimits::new(100, 4, 40).unwrap(),
        false,
    );
    for (timestamp, first_id, frames) in [(1_000_000, 1, 20), (suffix_start, 1001, 40)] {
        let mut queue = h.queue.lock();
        let publication = queue.publication.clone().unwrap();
        let buffer = AudioBuffer {
            timestamp,
            samples: (first_id..first_id + frames as i32)
                .flat_map(|id| [id * 1024; 2])
                .collect::<Vec<_>>()
                .into(),
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: 1_000,
                channels: 2,
                bit_depth: 32,
                codec_header: None,
            },
        };
        assert!(matches!(
            queue
                .try_enqueue_prepared(&publication, &h.owner, h.scope, (buffer, None), frames)
                .ok()
                .unwrap(),
            EnqueueOutcome::Accepted { .. }
        ));
    }
    h.now_us = 823_000; // Warm preparation with presentation 990ms, before source.
    h.render(OutputTimestampSource::DevicePresentation, 167_000);
    if reanchor {
        h.now_us = 833_000;
        let (_, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert_eq!(consumed, 10);
        let mut queue = h.queue.lock();
        let publication = queue.publication.clone().unwrap();
        queue
            .try_reanchor_prepared(&publication, &h.owner, h.scope, || Some(1_010_000), false)
            .unwrap();
        drop(queue);
        // A pre-target callback returns revoked windows so the real worker can
        // prepare the reanchored source before the intentionally late callback.
        h.frames = 1;
        h.now_us = 834_000;
        let (silent, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert!(silent.iter().all(|sample| *sample == 0.0));
        assert_eq!(consumed, 0);
        h.frames = 10;
    }
    h.now_us = 863_000; // First (or reanchored) audible presentation is 1030ms.
    let (audio, _) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
    let ids: Vec<_> = audio
        .chunks_exact(2)
        .map(|frame| (frame[0] * 2_097_152.0).round() as i32)
        .collect();
    if suffix_start == 2_000_000 {
        assert_eq!(
            ids,
            vec![0; 10],
            "future suffix must survive late-start discard; reanchor={reanchor}"
        );
        assert_eq!(h.owner.consumed_frames(h.scope), Ok(20));
        assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_020_000)));
        h.now_us = 1_833_000; // presentation 2000ms: untouched first future frame.
        let (audio, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert_eq!((audio[0] * 2_097_152.0).round() as i32, 1001);
        assert_eq!(consumed, 10);
        assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(2_010_000)));
    } else {
        assert_eq!(ids, vec![0, 0, 0, 0, 0, 1001, 1002, 1003, 1004, 1005]);
        assert_eq!(h.owner.consumed_frames(h.scope), Ok(25));
        assert_eq!(h.owner.media_boundary_us(h.scope), Ok(Some(1_040_000)));
        assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(0));
    }
}

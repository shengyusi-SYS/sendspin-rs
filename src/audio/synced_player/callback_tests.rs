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
                        assert!(change * 200 <= window_frames as u64,
                            "speed exceeds ±0.5% at {rate} Hz: {change} changed frames / {window_frames}");
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
    for delay in 1..=runtime::WINDOWS + 1 {
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
    let latest_delay = (runtime::WINDOWS as i64 + 1) * 1_000;
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
    assert_eq!(h.diagnostics.snapshot().raw_error_us, Some(expected_error));
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
fn callback_close_and_fault_publish_health_only_after_actual_consumption_stabilizes() {
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
        assert_eq!(stable.terminal(), finished.terminal());
        assert_eq!(stable.fault(), finished.fault());
    }
}

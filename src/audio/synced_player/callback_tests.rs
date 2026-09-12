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
    ),
>;

struct Harness {
    callback: DataCallback,
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
        let origin = Instant::now(); // Arbitrary epoch; every later instant is injected.
        let clock = Arc::new(Mutex::new(ClockSync::new_same_clock(Arc::new(
            CallbackClock(origin),
        ))));
        let owner = RendererOwner::new(
            RendererQueueLimits::new(sample_rate as usize * 12, 4, sample_rate as usize * 12)
                .unwrap(),
        );
        let scope = owner.mint_scope().unwrap();
        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate,
            channels: 2,
            bit_depth: 32,
            codec_header: None,
        };
        let frame_count = sample_rate as usize * 12;
        assert!(matches!(
            owner.enqueue_with_actual(scope, frame_count, || {
                let mut q = queue.lock();
                q.push(AudioBuffer {
                    timestamp: 1_000_000,
                    samples: (1..=frame_count)
                        .flat_map(|frame| [frame as i32 * 1024; 2])
                        .collect::<Vec<_>>()
                        .into(),
                    format: format.clone(),
                });
                (q.queued_frames(2), q.buffer_count())
            }),
            EnqueueOutcome::Accepted { .. }
        ));
        assert_eq!(
            owner.arm_scheduled_start(scope, 1_000_000),
            ScheduledArmOutcome::Armed
        );
        let diagnostics = SyncDiagnosticsReader::new(clock.clone());
        let static_delay_us = Arc::new(AtomicU64::new(0));
        let callback = make_output_callback::<f32>(
            queue.clone(),
            clock,
            format,
            CallbackConfig {
                gain_control: GainControl::new(100, false),
                process_callback: None,
                static_delay_us: static_delay_us.clone(),
            },
            owner.clone(),
            scope,
            diagnostics.clone(),
        );
        Self {
            callback: Box::new(callback),
            origin,
            now_us: 823_000, // First valid presentation is 990ms; start is 1000ms.
            frames: sample_rate as usize / 100,
            diagnostics,
            owner,
            scope,
            queue,
            static_delay_us,
        }
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
fn callback_access_failure_reports_silence_and_retains_evidence_after_recovery() {
    for block_renderer in [true, false] {
        let mut h = Harness::new(44_100);
        h.warm(OutputTimestampSource::DevicePresentation);
        let before = h.diagnostics.snapshot();
        let consumed_before = h.owner.health(h.scope).unwrap().consumed_frames();
        let owner = h.owner.clone();
        let queue = h.queue.clone();
        let permit = block_renderer.then(|| owner.try_callback_permit(h.scope).unwrap());
        let queue_guard = (!block_renderer).then(|| queue.lock());
        let callback = StreamInstant::from_nanos(h.now_us * 1_000);
        let mut data = vec![1.0; 882];
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

        assert!(data.iter().all(|sample| *sample == 0.0));
        let health = h.owner.health(h.scope).unwrap();
        assert_eq!(health.consumed_frames(), consumed_before);
        assert_eq!(health.underrun_frames(), 441);
        let snapshot = h.diagnostics.snapshot();
        assert_eq!(snapshot.requested_frames - before.requested_frames, 441);
        assert_eq!(snapshot.silent_callbacks - before.silent_callbacks, 1);
        assert_eq!(snapshot.silent_frames - before.silent_frames, 441);
        assert_eq!(snapshot.access_silence_frames, 441);
        assert_eq!(snapshot.renderer_access_misses, u64::from(block_renderer));
        assert_eq!(snapshot.queue_lock_misses, u64::from(!block_renderer));
        assert_eq!(snapshot.last_access_miss_callback, before.callbacks + 1);
        assert_eq!(snapshot.last_access_miss_phase, Some("timing_snapshot"));
        assert_eq!(snapshot.underrun_frames, 0); // This queue never ran dry.
        assert_eq!(snapshot.output_xrun_count, Some(3));
        assert_eq!(snapshot.output_buffer_size_frames, Some(1024));
        assert_eq!(snapshot.raw_error_us, None);

        let (data, consumed) = h.render(OutputTimestampSource::DevicePresentation, 167_000);
        assert!(data.iter().all(|sample| *sample > 0.0));
        assert_eq!(consumed, 441);
        let recovered = h.diagnostics.snapshot();
        assert_eq!(recovered.access_silence_frames, 441);
        assert_eq!(recovered.last_access_miss_callback, snapshot.callbacks);
        assert_eq!(recovered.last_access_miss_phase, Some("timing_snapshot"));
        assert_eq!(recovered.silent_frames, snapshot.silent_frames);
        assert_eq!(recovered.output_xrun_count, None); // Unsupported is not zero.
        assert_eq!(recovered.output_buffer_size_frames, None);
    }
}

#[test]
fn callback_valid_and_unspecified_timestamps_start_and_remain_aligned() {
    for rate in [44_100, 96_000] {
        for source in [
            OutputTimestampSource::DevicePresentation,
            OutputTimestampSource::Unspecified,
        ] {
            Harness::new(rate).warm(source);
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

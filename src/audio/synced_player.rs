// ABOUTME: Synced audio player with drift correction
// ABOUTME: Uses DAC callback timestamps to drop/insert frames for alignment

use crate::audio::gain::{GainControl, GainRamp};
use crate::audio::player_contract::{
    EnqueueOutcome, OpenError, OutputBackendError, PlayerScope, PreStartAbortOutcome,
    RendererCallbackPermit, RendererCapacitySnapshot, RendererFault, RendererHealthSnapshot,
    RendererOperationOutcome, RendererOwner, RendererQueueLimits, ScheduledArmOutcome,
    ScheduledStartOutcome, StartState, TerminalOutcome,
};
use crate::audio::sync_correction::{
    CorrectionPlanner, CorrectionSchedule, EngageGate, SyncErrorFilter,
};
use crate::audio::{AudioBuffer, AudioFormat, SyncDiagnosticsReader, SyncDiagnosticsSnapshot};
use crate::error::Error;
use crate::log_sampling::should_log_sample;
use crate::sync::ClockSync;
use cpal::traits::DeviceTrait;
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use cpal::{Sample, I24};
use parking_lot::{Mutex, MutexGuard};
use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Callback for post-processing audio samples before output.
///
/// Receives `&mut [f32]` (interleaved, after gain is applied).
///
/// The callback is invoked on **every** audio callback, including during
/// pre-start silence when the buffer is all zeros. This allows consumers
/// (e.g. VU meters) to observe the silence rather than missing callbacks.
///
/// # Thread Safety
///
/// This closure runs on the **audio callback thread**. It must:
/// - Not block (no locks, I/O, or sleeping)
/// - Not allocate (no `Vec::push`, `Box::new`, etc.)
/// - Not panic (would abort the audio thread)
///
/// # Why `Box<dyn>`?
///
/// Using dynamic dispatch (`Box<dyn FnMut>`) keeps `SyncedPlayer` a concrete,
/// non-generic type. This simplifies storage, trait object compatibility, and
/// downstream usage at the cost of one vtable indirect call per audio callback
/// (~1 ns vs the ~200 us callback budget).
pub type ProcessCallback = Box<dyn FnMut(&mut [f32]) + Send + 'static>;

/// Maximum static delay in milliseconds. The Sendspin protocol defines
/// `static_delay_ms` over 0–5000; values outside that range are rejected.
pub const MAX_STATIC_DELAY_MS: u16 = 5000;

/// A validated device-output delay in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceDelayMs(f64);

/// Typed rejection for an invalid device-output delay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DeviceDelayError {
    /// The supplied value was NaN or infinite.
    #[error("device delay must be finite")]
    NonFinite,
    /// The supplied value was outside the inclusive 0..=5000ms range.
    #[error("device delay must be in 0..=5000ms")]
    OutOfRange,
}

impl DeviceDelayMs {
    /// Validate a millisecond delay without clamping.
    pub fn new(value: f64) -> Result<Self, DeviceDelayError> {
        if !value.is_finite() {
            return Err(DeviceDelayError::NonFinite);
        }
        if !(0.0..=f64::from(MAX_STATIC_DELAY_MS)).contains(&value) {
            return Err(DeviceDelayError::OutOfRange);
        }
        Ok(Self(value))
    }

    /// Validated millisecond value.
    pub fn get(self) -> f64 {
        self.0
    }

    fn as_micros(self) -> u64 {
        (self.0 * 1_000.0).round() as u64
    }
}

/// Marker returned when a delay update requires a one-shot presentation reanchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReanchorRequired;

/// Configuration for [`SyncedPlayer`] construction.
pub struct SyncedPlayerConfig {
    /// Explicitly selected, leased audio output device.
    device: Device,
    /// Initial playback volume, 0-100.
    pub volume: u8,
    /// Initial mute state.
    pub muted: bool,
    /// Optional fixed cpal buffer size in frames. When `None`, Windows
    /// requests a 40ms endpoint buffer (see `WINDOWS_DEFAULT_BUFFER_MS` for
    /// the rationale) and other platforms use the cpal device default.
    pub buffer_size: Option<u32>,
}

impl SyncedPlayerConfig {
    /// Create a config for an explicitly selected output device.
    ///
    /// The library never queries or falls back to the platform default device.
    ///
    /// ```compile_fail
    /// use sendspin::audio::SyncedPlayerConfig;
    /// let _ = SyncedPlayerConfig::new();
    /// ```
    pub fn new(device: Device) -> Self {
        Self {
            device,
            volume: 100,
            muted: false,
            buffer_size: None,
        }
    }
}

/// Endpoint buffer requested on Windows when the caller does not override
/// [`SyncedPlayerConfig::buffer_size`].
///
/// WASAPI's shared-mode default allocates two engine periods (20ms at the
/// common 10ms period), so one late wake of the event-loop thread drains the
/// buffer completely and the next slip starves the mixer (silence insertion —
/// a real timeline displacement). Requesting ~40ms deepens the endpoint
/// queue only — the engine still wakes us every period — leaving roughly
/// three periods queued at each wake, so the mixer reliably survives two
/// consecutive late wakes (a third lands exactly at empty). That margin
/// matters even with MMCSS: the thread
/// shares the real-time class with every other pro-audio client and will
/// sometimes lose its slot.
///
/// Depth does not affect inter-device sync: the sync error is measured
/// against padding-compensated presentation timestamps, so queued depth
/// cancels out of the alignment math. The cost is bounded reaction latency —
/// a track-skip flush, volume ramp, or drift correction reaches the speaker
/// only after the already-queued frames drain.
const WINDOWS_DEFAULT_BUFFER_MS: u32 = 40;

const DEFAULT_RENDERER_FRAMES: usize = 96_000;
const DEFAULT_RENDERER_BUFFERS: usize = 64;

fn default_renderer_limits(sample_rate: u32) -> Result<RendererQueueLimits, Error> {
    let max_chunk_frames = usize::try_from(sample_rate)
        .map_err(|_| Error::Output("sample rate is not representable".to_string()))?
        .min(DEFAULT_RENDERER_FRAMES);
    RendererQueueLimits::new(
        DEFAULT_RENDERER_FRAMES,
        DEFAULT_RENDERER_BUFFERS,
        max_chunk_frames,
    )
    .map_err(|_| Error::Output("invalid renderer queue limits".to_string()))
}

fn validate_output_format(format: &AudioFormat) -> Result<(), OpenError> {
    if format.channels == 0 {
        return Err(OpenError::UnsupportedFormat);
    }
    if format.sample_rate == 0 {
        return Err(OpenError::UnsupportedFormat);
    }
    Ok(())
}

fn validate_enqueue_buffer(
    player_format: &AudioFormat,
    buffer: &AudioBuffer,
) -> Result<(usize, i64), EnqueueOutcome> {
    if buffer.format != *player_format {
        return Err(EnqueueOutcome::FormatMismatch);
    }
    let channels = usize::from(player_format.channels);
    if channels == 0
        || player_format.sample_rate == 0
        || !buffer.samples.len().is_multiple_of(channels)
    {
        return Err(EnqueueOutcome::InvalidBuffer);
    }
    let frames = buffer.samples.len() / channels;
    let duration_numerator = (frames as u128)
        .checked_mul(1_000_000)
        .and_then(|value| value.checked_add(u128::from(player_format.sample_rate) / 2))
        .ok_or(EnqueueOutcome::InvalidBuffer)?;
    let duration_us = duration_numerator / u128::from(player_format.sample_rate);
    let duration_us = i64::try_from(duration_us).map_err(|_| EnqueueOutcome::InvalidBuffer)?;
    buffer
        .timestamp
        .checked_add(duration_us)
        .ok_or(EnqueueOutcome::InvalidBuffer)?;
    Ok((frames, duration_us))
}

fn preflight_device_output_format(device: &Device, format: &AudioFormat) -> Result<(), OpenError> {
    let default_config = device
        .default_output_config()
        .map_err(|error| OpenError::Backend(OutputBackendError::new(error.to_string())))?;
    if !matches!(
        default_config.sample_format(),
        SampleFormat::F32
            | SampleFormat::F64
            | SampleFormat::I8
            | SampleFormat::I16
            | SampleFormat::I24
            | SampleFormat::I32
            | SampleFormat::I64
            | SampleFormat::U8
            | SampleFormat::U16
            | SampleFormat::U32
            | SampleFormat::U64
    ) {
        return Err(OpenError::UnsupportedFormat);
    }
    let supported = device
        .supported_output_configs()
        .map_err(|error| OpenError::Backend(OutputBackendError::new(error.to_string())))?
        .any(|range| {
            range.channels() == u16::from(format.channels)
                && range.sample_format() == default_config.sample_format()
                && range.min_sample_rate() <= format.sample_rate
                && format.sample_rate <= range.max_sample_rate()
        });
    if !supported {
        return Err(OpenError::UnsupportedFormat);
    }
    Ok(())
}

/// Frames for [`WINDOWS_DEFAULT_BUFFER_MS`] at `sample_rate`.
const fn windows_default_buffer_frames(sample_rate: u32) -> u32 {
    sample_rate * WINDOWS_DEFAULT_BUFFER_MS / 1000
}

/// Endpoint buffer request when the caller does not specify one.
fn default_buffer_size(sample_rate: u32) -> cpal::BufferSize {
    if cfg!(target_os = "windows") {
        cpal::BufferSize::Fixed(windows_default_buffer_frames(sample_rate))
    } else {
        cpal::BufferSize::Default
    }
}

/// Opaque owner token retained for exactly as long as an enqueued buffer.
pub trait AudioBufferLifetime: Send + Sync {}

impl<T: Send + Sync> AudioBufferLifetime for T {}

struct QueuedAudioBuffer {
    buffer: AudioBuffer,
    _lifetime: Option<Arc<dyn AudioBufferLifetime>>,
}

impl Deref for QueuedAudioBuffer {
    type Target = AudioBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

struct PlaybackQueue {
    queue: VecDeque<QueuedAudioBuffer>,
    current: Option<QueuedAudioBuffer>,
    index: usize,
    /// Current playback position in **server-time microseconds**. Periodically
    /// reanchored to the server's clock during clock-sync correction, so this
    /// represents "what server timestamp is playing right now", not how much
    /// audio content has been consumed.
    cursor_us: i64,
    cursor_remainder: i64,
    initialized: bool,
    generation: u64,
    force_reanchor: bool,
    /// Buffers enqueued in this generation; the sampling key for the enqueue
    /// trace line. Reset by `clear()`.
    enqueue_count: u64,
}

fn buffer_end_zone_us(buffer: &AudioBuffer) -> i128 {
    let channels = i128::from(buffer.format.channels.max(1));
    let frames = buffer.samples.len() as i128 / channels;
    let rate = i128::from(buffer.format.sample_rate.max(1));
    let duration_us = (frames * 1_000_000 + rate / 2) / rate;
    i128::from(buffer.timestamp) + duration_us
}

impl PlaybackQueue {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            current: None,
            index: 0,
            cursor_us: 0,
            cursor_remainder: 0,
            initialized: false,
            generation: 0,
            force_reanchor: true,
            enqueue_count: 0,
        }
    }

    fn clear(&mut self) {
        self.queue.clear();
        self.current = None;
        self.index = 0;
        self.cursor_us = 0;
        self.cursor_remainder = 0;
        self.initialized = false;
        self.generation = self.generation.wrapping_add(1);
        self.force_reanchor = true;
        self.enqueue_count = 0;
    }

    #[cfg(test)]
    fn push(&mut self, buffer: AudioBuffer) {
        self.push_with_lifetime(buffer, None);
    }

    fn push_with_lifetime(
        &mut self,
        buffer: AudioBuffer,
        lifetime: Option<Arc<dyn AudioBufferLifetime>>,
    ) {
        // Initialize the cursor from the first enqueued buffer so the audio
        // callback can see a valid cursor_us before it starts reading. Without
        // this, the callback's pre-start gate can't evaluate timestamps and
        // outputs silence indefinitely. Use the minimum timestamp seen so far
        // since buffers may arrive out of order.
        if !self.initialized {
            self.cursor_us = buffer.timestamp;
            self.cursor_remainder = 0;
            self.initialized = true;
        }

        // When the server rebases its timeline backward (e.g. after event loop
        // starvation), new chunks arrive with timestamps that overlap chunks
        // already in the queue. Remove all overlapping buffers to prevent
        // duplicate audio that causes audible stuttering (~500ms of audio
        // played twice). The server will send fresh buffers for any gaps.
        // This works for any chunk size (the sendspin spec allows arbitrarily
        // small chunks), unlike the previous fixed-threshold approach.
        //
        // The overlap must be at least one frame to count. A sub-frame
        // "overlap" cannot contain a duplicated sample — it is timestamp
        // rounding, not a rebase: at rates where chunks are not a whole
        // number of microseconds (44.1kHz: 1102 frames = 24988.66µs), the
        // server's floor-based timestamp grid steps 24988µs while we measure
        // the buffer as 24989µs, landing a third of all chunks 1µs "inside"
        // their predecessor. Evicting on those phantom overlaps silently
        // discarded ~34% of all 44.1kHz audio (heard as continuous popping).
        let rate = i64::from(buffer.format.sample_rate.max(1));
        let frame_us = (1_000_000 + rate - 1) / rate;
        let new_end = buffer_end_zone_us(&buffer);
        self.queue.retain(|b| {
            let existing_end = buffer_end_zone_us(b);
            let overlap_us =
                new_end.min(existing_end) - i128::from(buffer.timestamp.max(b.timestamp));
            overlap_us < i128::from(frame_us)
        });

        let pos = self
            .queue
            .iter()
            .position(|b| b.timestamp > buffer.timestamp);
        if let Some(pos) = pos {
            self.queue.insert(
                pos,
                QueuedAudioBuffer {
                    buffer,
                    _lifetime: lifetime,
                },
            );
        } else {
            self.queue.push_back(QueuedAudioBuffer {
                buffer,
                _lifetime: lifetime,
            });
        }
    }

    fn consume_next_frame(
        &mut self,
        channels: usize,
        sample_rate: u32,
        destination: Option<&mut [i32]>,
    ) -> bool {
        let needs_buffer = match self.current {
            None => true,
            Some(ref c) => self.index + channels > c.samples.len(),
        };
        if needs_buffer {
            // Drop stale buffers that are entirely before the cursor.
            if self.initialized {
                while let Some(front) = self.queue.front() {
                    if buffer_end_zone_us(front) < i128::from(self.cursor_us) {
                        let _ = self.queue.pop_front();
                        continue;
                    }
                    break;
                }
            }

            // Pop buffers until we find one with remaining samples past the
            // cursor, or the queue is empty.
            loop {
                self.current = self.queue.pop_front();
                self.index = 0;

                // Skip past samples that are behind the cursor. This handles
                // buffers that partially overlap with the current playback
                // position, e.g. from backward timestamp jumps during server
                // timeline rebases. Without this, playing from the start of
                // such a buffer repeats audio the cursor has already passed,
                // causing an audible stutter.
                if self.initialized {
                    if let Some(ref current) = self.current {
                        if current.timestamp < self.cursor_us {
                            let skip_us = self.cursor_us - current.timestamp;
                            let skip_frames =
                                (skip_us.saturating_mul(sample_rate as i64) / 1_000_000) as usize;
                            if skip_frames > 0 {
                                self.index = skip_frames
                                    .saturating_mul(channels)
                                    .min(current.samples.len());
                            }
                        }
                    }
                }

                // If the skip consumed the entire buffer (or left fewer
                // samples than one frame), discard it and try the next one.
                match self.current {
                    Some(ref c) if self.index + channels > c.samples.len() => {
                        self.current = None;
                        if self.queue.is_empty() {
                            break;
                        }
                        continue;
                    }
                    _ => break,
                }
            }
        }

        if !self.initialized {
            if let Some(current) = self.current.as_ref() {
                self.cursor_us = current.timestamp;
                self.cursor_remainder = 0;
                self.initialized = true;
            }
        }

        // Bail before advancing cursor/index when the queue is empty.
        // Without this the cursor races ahead during underruns, causing
        // the stale-buffer-dropping logic to discard valid buffers when
        // audio resumes.
        if self.current.is_none() {
            return false;
        }

        let start = self.index;
        let end = self.index + channels;
        if let Some(destination) = destination {
            destination.copy_from_slice(
                &self.current.as_ref().expect("checked current").samples[start..end],
            );
        }
        self.index = end;
        self.advance_cursor(sample_rate);
        if self
            .current
            .as_ref()
            .is_some_and(|current| self.index >= current.samples.len())
        {
            self.current = None;
            self.index = 0;
        }
        true
    }

    #[cfg(test)]
    fn next_frame(&mut self, channels: usize, sample_rate: u32) -> Option<Vec<i32>> {
        let mut frame = vec![0; channels];
        self.consume_next_frame(channels, sample_rate, Some(&mut frame))
            .then_some(frame)
    }

    fn advance_cursor(&mut self, sample_rate: u32) {
        self.cursor_remainder += 1_000_000;
        let advance = self.cursor_remainder / sample_rate as i64;
        self.cursor_remainder %= sample_rate as i64;
        self.cursor_us += advance;
    }

    fn first_playable_cursor_at_or_after(&self, server_time_us: i64) -> Option<i64> {
        if let Some(buffer) = self.current.as_ref() {
            let remaining_start = buffer.timestamp.max(self.cursor_us);
            if buffer_end_zone_us(buffer) > i128::from(server_time_us.max(remaining_start)) {
                return Some(remaining_start.max(server_time_us));
            }
        }

        for buffer in &self.queue {
            if buffer_end_zone_us(buffer) > i128::from(server_time_us) {
                return Some(buffer.timestamp.max(server_time_us));
            }
        }

        None
    }

    fn queued_frames(&self, channels: usize) -> usize {
        let current_frames = self.current.as_ref().map_or(0, |current| {
            current.samples.len().saturating_sub(self.index) / channels
        });
        let queued_frames = self
            .queue
            .iter()
            .map(|buffer| buffer.samples.len() / channels)
            .sum::<usize>();
        current_frames + queued_frames
    }

    fn queued_duration_us(&self, channels: usize, sample_rate: u32) -> u64 {
        self.queued_frames(channels) as u64 * 1_000_000 / sample_rate.max(1) as u64
    }

    fn buffer_count(&self) -> usize {
        self.queue.len() + usize::from(self.current.is_some())
    }
}

/// Microseconds as fractional milliseconds, for log formatting.
fn us_to_ms(us: u64) -> f64 {
    us as f64 / 1000.0
}

fn canonical_presentation_zone_us(
    device_presentation_zone_us: Option<i64>,
    static_delay_us: u64,
) -> Option<i64> {
    let delay_us = i64::try_from(static_delay_us).ok()?;
    device_presentation_zone_us?.checked_add(delay_us)
}

/// Queue depth below which the edge-triggered "queue low" debug line fires.
const QUEUE_LOW_WATER_US: u64 = 100_000;

/// Queue depth required to log recovery after a low-queue warning. Kept above
/// [`QUEUE_LOW_WATER_US`] so a queue hovering at one boundary cannot flood the
/// log with low/recovered pairs.
const QUEUE_RECOVERED_WATER_US: u64 = 200_000;

/// Diagnostic counters for the audio callback. Logging-only: playback
/// decisions never read these.
///
/// `callbacks` counts for the lifetime of the stream and anchors every log
/// line to one timeline. The rest are per-generation — reset whenever the
/// playback queue generation changes — so each stream start reports its own
/// startup behavior.
#[derive(Default)]
struct CallbackStats {
    /// Data callbacks since the stream was built. Never reset.
    callbacks: u64,
    /// Callbacks that emitted silence (pre-start gate, reanchor wait, early).
    silent_callbacks: u64,
    /// Callbacks that skipped sync because the clock lock was contended.
    sync_lock_misses: u64,
    /// Frames filled with silence because the queue ran dry.
    underrun_frames: u64,
    /// Callbacks that had at least one underrun frame.
    underrun_callbacks: u64,
    /// Length of the current run of underrun callbacks (0 while healthy).
    consecutive_underrun_callbacks: u64,
    /// Schedule updates within the current correction episode; the sampling
    /// key for the correction trace line. Reset when correction disengages.
    correction_updates: u64,
    /// Corrections the planner requested during clock warm-up that were
    /// suppressed; the sampling key for the warm-up trace line.
    warmup_suppressed_corrections: u64,
    /// Corrections the planner requested that the engage gate suppressed
    /// while awaiting a sustained error; the sampling key for its trace line.
    gate_suppressed_corrections: u64,
    /// Correction episodes started (idle -> correcting transitions, including
    /// reanchor engagements). Mirrors the "Sync correction engaged" debug
    /// line 1:1 so the generation summary can answer whether the corrector
    /// ever fired, even when debug logging was off during playback.
    correction_engagements: u64,
    /// Whether the queue was below [`QUEUE_LOW_WATER_US`] at the last render.
    /// Drives the edge-triggered low/recovered debug lines.
    queue_low: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CallbackTimestampObservation {
    timestamp: cpal::OutputStreamTimestamp,
    source: cpal::OutputTimestampSource,
    monotonic_violation: bool,
}

fn classify_output_timestamp(
    timestamp: cpal::OutputStreamTimestamp,
    source: cpal::OutputTimestampSource,
    previous_playback: Option<cpal::StreamInstant>,
) -> CallbackTimestampObservation {
    CallbackTimestampObservation {
        timestamp,
        source,
        monotonic_violation: previous_playback
            .is_some_and(|previous| timestamp.playback <= previous),
    }
}

impl CallbackStats {
    /// Reset per-generation counters, keeping the lifetime callback count.
    fn reset_for_generation(&mut self) {
        *self = Self {
            callbacks: self.callbacks,
            ..Self::default()
        };
    }
}

/// Bundles gain and post-processing parameters for the data callback.
struct CallbackConfig {
    gain_control: GainControl,
    process_callback: Option<ProcessCallback>,
    static_delay_us: Arc<AtomicU64>,
}

struct CallbackOutputs {
    error: Arc<Mutex<Option<String>>>,
    renderer: RendererOwner,
    scope: PlayerScope,
}

#[derive(Clone, Copy)]
enum CallbackQueuePhase {
    TimingSnapshot,
    StartupReanchor,
    CorrectionReanchor,
    Render,
}

impl CallbackQueuePhase {
    fn label(self) -> &'static str {
        match self {
            Self::TimingSnapshot => "timing_snapshot",
            Self::StartupReanchor => "startup_reanchor",
            Self::CorrectionReanchor => "correction_reanchor",
            Self::Render => "render",
        }
    }
}

enum StartupReanchorOutcome {
    Applied(i64),
    NoPlayable,
    Stale,
}

fn try_callback_queue<T>(
    renderer: &RendererOwner,
    scope: PlayerScope,
    queue: &Mutex<PlaybackQueue>,
    silent_frames: usize,
    phase: CallbackQueuePhase,
    observation: &mut SyncDiagnosticsSnapshot,
    operation: impl FnOnce(&mut PlaybackQueue, &mut RendererCallbackPermit<'_>) -> T,
) -> Option<T> {
    let (mut permit, mut queue) =
        try_callback_queue_guards(renderer, scope, queue, silent_frames, phase, observation)?;
    Some(operation(&mut queue, &mut permit))
}

fn try_callback_queue_guards<'a>(
    renderer: &'a RendererOwner,
    scope: PlayerScope,
    queue: &'a Mutex<PlaybackQueue>,
    silent_frames: usize,
    phase: CallbackQueuePhase,
    observation: &mut SyncDiagnosticsSnapshot,
) -> Option<(RendererCallbackPermit<'a>, MutexGuard<'a, PlaybackQueue>)> {
    let mut record_failure = |renderer_unavailable: bool| {
        let counter = if renderer_unavailable {
            &mut observation.renderer_access_misses
        } else {
            &mut observation.queue_lock_misses
        };
        *counter = counter.saturating_add(1);
        observation.access_silence_frames = observation
            .access_silence_frames
            .saturating_add(silent_frames as u64);
        observation.last_access_miss_callback = observation.callbacks;
        observation.last_access_miss_phase = Some(phase.label());
        renderer.record_callback_underrun(silent_frames as u64);
    };
    let Some(permit) = renderer.try_callback_permit(scope) else {
        record_failure(true);
        return None;
    };
    let Some(queue) = queue.try_lock() else {
        drop(permit);
        record_failure(false);
        return None;
    };
    Some((permit, queue))
}

fn abort_before_start_with_queue<F>(
    owner: &RendererOwner,
    scope: PlayerScope,
    queue: &Mutex<PlaybackQueue>,
    after_queue_lock: F,
) -> PreStartAbortOutcome
where
    F: FnOnce(),
{
    let mut queue = queue.lock();
    after_queue_lock();
    owner.abort_before_start_with_actual(scope, || queue.clear())
}

fn teardown_with_queue(
    owner: &RendererOwner,
    scope: PlayerScope,
    queue: &Mutex<PlaybackQueue>,
) -> TerminalOutcome {
    let mut queue = queue.lock();
    let outcome = owner.teardown_with_actual(scope, || queue.clear());
    drop(queue);
    match outcome {
        crate::audio::player_contract::ActualTeardownOutcome::Complete(outcome) => outcome,
        crate::audio::player_contract::ActualTeardownOutcome::FinalizationPending => {
            owner.wait_for_claimed_terminal(scope)
        }
    }
}

fn set_device_delay_state(
    queue: &Mutex<PlaybackQueue>,
    static_delay_us: &AtomicU64,
    delay_ms: DeviceDelayMs,
) {
    let mut queue = queue.lock();
    static_delay_us.store(delay_ms.as_micros(), Ordering::Relaxed);
    if queue.initialized {
        queue.force_reanchor = true;
    }
}

/// Synced audio output with drift correction.
pub struct SyncedPlayer {
    diagnostics: SyncDiagnosticsReader,
    format: AudioFormat,
    queue: Arc<Mutex<PlaybackQueue>>,
    renderer: RendererOwner,
    scope: PlayerScope,
    /// Last error from the audio stream callback, if any.
    last_error: Arc<Mutex<Option<String>>>,
    gain: GainControl,
    /// Shared with the audio callback. See [`SyncedPlayer::set_static_delay`].
    static_delay_us: Arc<AtomicU64>,
}

impl SyncedPlayer {
    /// Create a new synced player using the provided clock sync and optional device.
    ///
    /// The player starts at `volume` (0-100) and `muted` state. These are
    /// applied immediately — the first audio callback uses the correct gain
    /// with no ramp from a default value.
    /// The `buffer_size` overrides the endpoint buffer request. If not set,
    /// Windows requests 40ms of margin (see `WINDOWS_DEFAULT_BUFFER_MS` for
    /// the rationale) and other platforms use the cpal device default.
    pub fn new(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
    ) -> Result<Self, OpenError> {
        Self::build(format, clock_sync, config, None, None)
    }

    /// Create a player bound to a caller-owned renderer scope.
    ///
    /// This is intended for route/lease authorities that must use the same
    /// scope for device validation, callback fencing, capacity evidence, and
    /// terminal acknowledgement.
    pub fn new_with_renderer(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
        renderer: RendererOwner,
        scope: PlayerScope,
    ) -> Result<Self, OpenError> {
        Self::build(format, clock_sync, config, None, Some((renderer, scope)))
    }

    /// Create a player with a process callback for post-gain audio processing.
    ///
    /// The callback receives samples **after** gain/mute processing has been
    /// applied. See [`ProcessCallback`] for thread-safety requirements.
    ///
    /// # Example (requires physical audio hardware to run)
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use parking_lot::Mutex;
    /// # use cpal::traits::HostTrait;
    /// # use sendspin::audio::{AudioFormat, Codec, SyncedPlayer, SyncedPlayerConfig};
    /// # use sendspin::sync::ClockSync;
    /// # use sendspin::DefaultClock;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let format = AudioFormat {
    ///     codec: Codec::Pcm,
    ///     sample_rate: 48_000,
    ///     channels: 2,
    ///     bit_depth: 24,
    ///     codec_header: None,
    /// };
    /// let clock_sync = Arc::new(Mutex::new(ClockSync::new(Arc::new(DefaultClock::new()))));
    /// let device = cpal::default_host()
    ///     .default_output_device()
    ///     .ok_or_else(|| std::io::Error::other("no output device available"))?;
    /// let player = SyncedPlayer::with_process_callback(
    ///     format,
    ///     clock_sync,
    ///     SyncedPlayerConfig::new(device),
    ///     Box::new(|data| { /* e.g. feed a VU meter or visualizer */ }),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_process_callback(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
        callback: ProcessCallback,
    ) -> Result<Self, OpenError> {
        Self::build(format, clock_sync, config, Some(callback), None)
    }

    fn build(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
        process_callback: Option<ProcessCallback>,
        renderer_scope: Option<(RendererOwner, PlayerScope)>,
    ) -> Result<Self, OpenError> {
        validate_output_format(&format)?;
        let device = config.device;
        preflight_device_output_format(&device, &format)?;

        let stream_config = StreamConfig {
            channels: format.channels as u16,
            sample_rate: cpal::SampleRate::from(format.sample_rate),
            buffer_size: match config.buffer_size {
                Some(frames) => cpal::BufferSize::Fixed(frames),
                None => default_buffer_size(format.sample_rate),
            },
        };

        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        let queue_clone = Arc::clone(&queue);
        let format_clone = format.clone();
        let last_error = Arc::new(Mutex::new(None));
        let gain = GainControl::new(config.volume, config.muted);
        let static_delay_us = Arc::new(AtomicU64::new(0));
        let (renderer, scope) = match renderer_scope {
            Some((renderer, scope)) => {
                renderer
                    .capacity(scope)
                    .map_err(|_| OpenError::StaleGeneration)?;
                (renderer, scope)
            }
            None => {
                let renderer =
                    RendererOwner::new(default_renderer_limits(format.sample_rate).map_err(
                        |error| OpenError::Backend(OutputBackendError::new(error.to_string())),
                    )?);
                let scope = renderer.mint_scope()?;
                (renderer, scope)
            }
        };

        let cb_config = CallbackConfig {
            gain_control: gain.clone(),
            process_callback,
            static_delay_us: Arc::clone(&static_delay_us),
        };
        let callback_outputs = CallbackOutputs {
            error: Arc::clone(&last_error),
            renderer: renderer.clone(),
            scope,
        };

        let diagnostics = SyncDiagnosticsReader::new(Arc::clone(&clock_sync));
        let stream = Self::build_stream(
            &device,
            &stream_config,
            queue_clone,
            Arc::clone(&clock_sync),
            format_clone,
            cb_config,
            callback_outputs,
            diagnostics.clone(),
        )
        .map_err(|error| OpenError::Backend(OutputBackendError::new(error.to_string())))?;
        if renderer.attach_output_stream(scope, stream) != RendererOperationOutcome::Applied {
            return Err(OpenError::Backend(OutputBackendError::new(
                "renderer rejected opened stream ownership",
            )));
        }
        match renderer.start_output_stream(scope) {
            Ok(RendererOperationOutcome::Applied) => {}
            Ok(_) => {
                let _ = renderer.teardown(scope);
                return Err(OpenError::Backend(OutputBackendError::new(
                    "renderer rejected leased output stream startup",
                )));
            }
            Err(error) => {
                let _ = renderer.teardown(scope);
                return Err(OpenError::Backend(error));
            }
        }
        log::info!(
            "SyncedPlayer started: {} channels, {} Hz, {}-bit",
            format.channels,
            format.sample_rate,
            format.bit_depth,
        );

        Ok(Self {
            diagnostics,
            format,
            queue,
            renderer,
            scope,
            last_error,
            gain,
            static_delay_us,
        })
    }

    /// Enqueue a decoded buffer for playback.
    ///
    /// Scheduling uses `buffer.timestamp` (server time in microseconds) for
    /// drift-corrected playback.
    pub fn enqueue(&self, buffer: AudioBuffer) -> EnqueueOutcome {
        self.enqueue_with_optional_lifetime(buffer, None)
    }

    /// Enqueue a buffer while retaining an opaque owner token until the
    /// buffer is consumed, cleared, rejected, or the player is dropped.
    pub fn enqueue_with_lifetime(
        &self,
        buffer: AudioBuffer,
        lifetime: Arc<dyn AudioBufferLifetime>,
    ) -> EnqueueOutcome {
        self.enqueue_with_optional_lifetime(buffer, Some(lifetime))
    }

    fn enqueue_with_optional_lifetime(
        &self,
        buffer: AudioBuffer,
        lifetime: Option<Arc<dyn AudioBufferLifetime>>,
    ) -> EnqueueOutcome {
        let channels = self.format.channels as usize;
        let (frames, buffer_duration_us) = match validate_enqueue_buffer(&self.format, &buffer) {
            Ok(validated) => validated,
            Err(outcome) => return outcome,
        };
        let buffer_timestamp = buffer.timestamp;
        let sample_rate = self.format.sample_rate;
        let mut queue = self.queue.lock();
        let outcome = self.renderer.enqueue_with_actual(self.scope, frames, || {
            queue.push_with_lifetime(buffer, lifetime);
            queue.enqueue_count += 1;
            (queue.queued_frames(channels), queue.buffer_count())
        });
        drop(queue);
        if !matches!(outcome, EnqueueOutcome::Accepted { .. }) {
            return outcome;
        }

        // Snapshot log fields under the lock but log after dropping it: the
        // audio callback contends on this lock, and logging can block on I/O.
        // The O(buffers) depth walk runs only for sampled, trace-enabled
        // enqueues.
        let trace_fields = {
            let queue = self.queue.lock();
            if log::log_enabled!(log::Level::Trace) && should_log_sample(queue.enqueue_count) {
                Some((
                    queue.enqueue_count,
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                    queue.cursor_us,
                    queue.generation,
                ))
            } else {
                None
            }
        };

        if let Some((enqueue_count, queued_us, buffers, cursor_us, generation)) = trace_fields {
            log::trace!(
                "Audio buffer enqueued: enqueue={}, ts={}µs, duration={:.1}ms, queued={:.1}ms, buffers={}, cursor={}µs, generation={}",
                enqueue_count,
                buffer_timestamp,
                buffer_duration_us as f64 / 1000.0,
                us_to_ms(queued_us),
                buffers,
                cursor_us,
                generation,
            );
        }
        outcome
    }

    /// Clear queued audio and reset playback state.
    pub fn clear(&self) {
        let channels = self.format.channels as usize;
        let sample_rate = self.format.sample_rate;

        // Snapshot log fields under the lock but log after dropping it: the
        // audio callback contends on this lock, and logging can block on I/O.
        let debug_fields = {
            let mut queue = self.queue.lock();
            let fields = log::log_enabled!(log::Level::Debug).then(|| {
                (
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                    queue.cursor_us,
                    queue.generation,
                )
            });
            if self
                .renderer
                .clear_with_actual(self.scope, || queue.clear())
                != RendererOperationOutcome::Applied
            {
                return;
            }
            fields
        };

        if let Some((queued_us, buffers, cursor_us, generation)) = debug_fields {
            log::debug!(
                "Cleared playback queue: queued={:.1}ms, buffers={}, cursor={}µs, generation={}",
                us_to_ms(queued_us),
                buffers,
                cursor_us,
                generation,
            );
        }
    }

    /// Release the underlying audio output device.
    ///
    /// Consuming the player drops its active `cpal::Stream`, allowing another
    /// local source to open the same device. Use this before advertising
    /// `state: "external_source"`.
    ///
    /// The returned [`GainControl`] preserves Sendspin's software volume/mute
    /// state across the handoff. It does not reflect hardware or OS mixer
    /// changes made by the external source.
    pub fn release_audio_device(self) -> GainControl {
        let gain = self.gain.clone();
        let _ = self.teardown();
        gain
    }

    /// Current opaque renderer scope.
    pub fn scope(&self) -> PlayerScope {
        self.scope
    }

    /// Read bounded renderer capacity from the scope-fenced owner.
    pub fn renderer_capacity(&self) -> Result<RendererCapacitySnapshot, RendererOperationOutcome> {
        self.renderer.capacity(self.scope)
    }

    /// Read complete typed renderer health from the scope-fenced owner.
    pub fn renderer_health(&self) -> Result<RendererHealthSnapshot, RendererOperationOutcome> {
        self.renderer.health(self.scope)
    }

    /// Close queue acceptance and callback consumption without dropping the stream.
    pub fn close(&self) -> RendererOperationOutcome {
        self.renderer.close(self.scope)
    }

    /// Arm an explicit Zone presentation timestamp for this scope.
    pub fn arm_scheduled_start(&self, start_at_zone_us: i64) -> ScheduledArmOutcome {
        self.renderer
            .arm_scheduled_start(self.scope, start_at_zone_us)
    }

    /// Read the scheduled presentation state for this scope.
    pub fn start_state(&self) -> Result<StartState, RendererOperationOutcome> {
        self.renderer.start_state(self.scope)
    }

    /// Abort only while the scheduled boundary has not won.
    pub fn abort_before_start(&self) -> PreStartAbortOutcome {
        abort_before_start_with_queue(&self.renderer, self.scope, &self.queue, || {})
    }

    /// Drop the stream through the sole shared terminal finalizer.
    pub fn teardown(&self) -> TerminalOutcome {
        teardown_with_queue(&self.renderer, self.scope, &self.queue)
    }

    /// Return the configured audio format.
    pub fn format(&self) -> &AudioFormat {
        &self.format
    }

    /// Check if the audio stream has encountered an error.
    ///
    /// Returns the error message if one occurred, clearing it in the process.
    pub fn take_error(&self) -> Option<String> {
        self.last_error.lock().take()
    }

    /// Check if the audio stream has an error without clearing it.
    pub fn has_error(&self) -> bool {
        self.last_error.lock().is_some()
    }

    /// Get a reference to the volume/mute control.
    ///
    /// Call `.clone()` if you need an owned handle to share across threads
    /// (cloning is cheap — single `Arc` increment, no data copy).
    pub fn gain_control(&self) -> &GainControl {
        &self.gain
    }

    // -- Volume/mute convenience methods --
    //
    // These promote the most common operations for ergonomics in simple
    // use-cases. For full control, use `gain_control()` directly.

    /// Current volume as 0-100.
    pub fn volume(&self) -> u8 {
        self.gain.volume()
    }

    /// Whether playback is currently muted.
    pub fn is_muted(&self) -> bool {
        self.gain.is_muted()
    }

    /// Set playback volume (0-100).
    pub fn set_volume(&self, volume: u8) {
        self.gain.set_volume(volume);
    }

    /// Set mute state.
    pub fn set_mute(&self, muted: bool) {
        self.gain.set_mute(muted);
    }

    /// Set a validated device playback delay.
    ///
    /// Compensates for external speaker/amplifier latency: the server pre-sends
    /// audio by this amount, so the player shifts each sample's emission earlier
    /// by the same delay to keep alignment correct. Takes effect on the next
    /// audio callback.
    ///
    /// A delay change is an intentional local timing offset, not clock drift.
    /// Request a one-shot reanchor so the audio callback either skips forward or
    /// waits for the new target time instead of feeding the delay delta through
    /// pitch-shifting drift correction.
    pub fn set_device_delay(&self, delay_ms: DeviceDelayMs) -> ReanchorRequired {
        set_device_delay_state(&self.queue, &self.static_delay_us, delay_ms);
        ReanchorRequired
    }

    /// Validate and set a protocol static delay without clamping.
    pub fn set_static_delay(&self, delay_ms: u16) -> Result<ReanchorRequired, DeviceDelayError> {
        DeviceDelayMs::new(f64::from(delay_ms)).map(|delay| self.set_device_delay(delay))
    }

    /// Observe sync on a host-owned, non-audio thread. No logging is performed
    /// by this observer; dropping the player still closes the stream normally.
    pub fn sync_diagnostics(&self) -> SyncDiagnosticsReader {
        self.diagnostics.clone()
    }

    /// Current static delay in milliseconds.
    pub fn static_delay_ms(&self) -> u16 {
        (self.static_delay_us.load(Ordering::Relaxed) / 1_000) as u16
    }

    /// Current validated device delay in fractional milliseconds.
    pub fn device_delay_ms(&self) -> f64 {
        self.static_delay_us.load(Ordering::Relaxed) as f64 / 1_000.0
    }

    fn build_stream(
        device: &Device,
        config: &StreamConfig,
        queue: Arc<Mutex<PlaybackQueue>>,
        clock_sync: Arc<Mutex<ClockSync>>,
        format: AudioFormat,
        cb_config: CallbackConfig,
        outputs: CallbackOutputs,
        diagnostics: SyncDiagnosticsReader,
    ) -> Result<Stream, Error> {
        let CallbackOutputs {
            error,
            renderer,
            scope,
        } = outputs;
        let device_config = device
            .default_output_config()
            .map_err(|e| Error::Output(e.to_string()))?;
        let mut stream_config = device_config.config();
        stream_config.buffer_size = config.buffer_size;
        stream_config.channels = format.channels.into();
        stream_config.sample_rate = format.sample_rate;

        macro_rules! output_stream {
            ($sample:ty) => {{
                let renderer_for_error = renderer.clone();
                let mut callback = make_output_callback::<$sample>(
                    queue,
                    clock_sync,
                    format,
                    cb_config,
                    renderer,
                    scope,
                    diagnostics,
                );
                device
                    .build_output_stream(
                        stream_config,
                        move |data: &mut [$sample], info: &cpal::OutputCallbackInfo| {
                            callback(
                                data,
                                info.timestamp(),
                                info.timestamp_source(),
                                info.timestamp_diagnostics(),
                                Instant::now(),
                            );
                        },
                        move |err| {
                            // cpal reports a refused real-time promotion as
                            // RealtimeDenied ("Audio will still play"); playback
                            // continues at normal priority. Warn without storing:
                            // take_error()/has_error() signal fatal stream
                            // failures, and a consumer must not tear down a
                            // working stream over a scheduling downgrade.
                            if err.kind() == cpal::ErrorKind::RealtimeDenied {
                                log::warn!(
                                    "Audio thread priority promotion failed (non-fatal): {err}"
                                );
                                return;
                            }
                            log::error!("Audio stream error: {err}");
                            *error.lock() = Some(err.to_string());
                            let _ = renderer_for_error.fault(scope, RendererFault::CallbackFailed);
                        },
                        None,
                    )
                    .map_err(|e| Error::Output(e.to_string()))
            }};
        }

        log::debug!(
            "Using output device: {}, config: {:?}",
            device
                .id()
                .map(|id| format!("{:?}", id))
                .map_err(|e| Error::Output(e.to_string()))?,
            device_config,
        );
        match device_config.sample_format() {
            SampleFormat::F32 => output_stream!(f32),
            SampleFormat::F64 => output_stream!(f64),
            SampleFormat::I8 => output_stream!(i8),
            SampleFormat::I16 => output_stream!(i16),
            SampleFormat::I24 => output_stream!(I24),
            SampleFormat::I32 => output_stream!(i32),
            SampleFormat::I64 => output_stream!(i64),
            SampleFormat::U8 => output_stream!(u8),
            SampleFormat::U16 => output_stream!(u16),
            SampleFormat::U32 => output_stream!(u32),
            SampleFormat::U64 => output_stream!(u64),
            _ => Err(Error::Output(format!(
                "Unsupported sample format: {:?}",
                device_config.sample_format()
            ))),
        }
    }
}

/// Canonical data callback, constructed independently of opening the native stream.
/// The stream wrapper supplies one paired timestamp/source and one local capture instant.
fn make_output_callback<T: cpal::SizedSample + cpal::FromSample<f32>>(
    queue: Arc<Mutex<PlaybackQueue>>,
    clock_sync: Arc<Mutex<ClockSync>>,
    format: AudioFormat,
    mut cb_config: CallbackConfig,
    renderer_for_data: RendererOwner,
    scope: PlayerScope,
    diagnostics: SyncDiagnosticsReader,
) -> impl FnMut(
    &mut [T],
    cpal::OutputStreamTimestamp,
    cpal::OutputTimestampSource,
    Option<cpal::OutputTimestampDiagnostics>,
    Instant,
) + Send {
    let channels = format.channels as usize;
    let sample_rate = format.sample_rate;
    let planner = CorrectionPlanner::new();
    let mut error_filter = SyncErrorFilter::new();
    let mut engage_gate = EngageGate::new();
    let mut last_frame = vec![i32::EQUILIBRIUM; channels];
    let mut schedule = CorrectionSchedule::default();
    let mut insert_counter = 0u32;
    let mut drop_counter = 0u32;
    let mut started = false;
    let mut handoff_warned = false;
    let mut sync_settle_logged = false;
    let mut last_callback_instant: Option<Instant> = None;
    let mut last_playback_timestamp: Option<cpal::StreamInstant> = None;
    let mut last_playback_delta_us: Option<u64> = None;
    // Running minimum of measured presentation latency, reset per
    // generation. Reanchors anchor against this floor rather than one
    // wake's reading: padding noise is one-sided (see SyncErrorFilter),
    // so a single sample may run a whole period high, and anchoring to it
    // bakes that period into the timeline until corrections audibly
    // unwind it. A stale floor after a latency-regime shift costs at most
    // one period of realignment — no worse than the shift itself.
    let mut min_playback_delta: Option<Duration> = None;
    let mut last_measured_playback_delta: Option<Duration> = None;
    let mut last_generation = 0u64;
    let mut stats = CallbackStats::default();
    let mut observation = SyncDiagnosticsSnapshot {
        sample_rate,
        ..Default::default()
    };
    let mut previous_callback_at: Option<Instant> = None;
    let initial_gain = cb_config.gain_control.gain();
    let mut gain_ramp = GainRamp::new(sample_rate, initial_gain);
    let mut f32_buffer = Vec::<f32>::new();
    move |data: &mut [T], timestamp, timestamp_source, timestamp_diagnostics, captured_at| {
        observation.callbacks = observation.callbacks.saturating_add(1);
        observation.requested_frames = observation
            .requested_frames
            .saturating_add((data.len() / channels) as u64);
        observation.captured_at = Some(captured_at);
        observation.output_xrun_count = timestamp_diagnostics.and_then(|d| d.output_xrun_count);
        observation.output_buffer_size_frames =
            timestamp_diagnostics.and_then(|d| d.output_buffer_size_frames);
        if timestamp_source == cpal::OutputTimestampSource::MonotonicFallback {
            if let Some(evidence) = timestamp_diagnostics {
                if let Some(reason) = evidence.fallback_reason {
                    use cpal::OutputTimestampFallbackReason::*;
                    let count = match reason {
                        Unavailable => &mut observation.fallback_unavailable,
                        Unsupported => &mut observation.fallback_unsupported,
                        Invalid => &mut observation.fallback_invalid,
                        NonMonotonic => &mut observation.fallback_non_monotonic,
                        ClockDomainMismatch => &mut observation.fallback_clock_domain_mismatch,
                    };
                    *count = count.saturating_add(1);
                    observation.last_timestamp_fallback_callback = observation.callbacks;
                    observation.last_timestamp_fallback = Some(evidence);
                }
            }
        }
        observation.callback_frames = data.len() / channels;
        observation.raw_error_us = None;
        observation.insert_every = 0;
        observation.drop_every = 0;
        observation.filtered_error_us = None;
        observation.playback_delay_us = timestamp
            .playback
            .duration_since(timestamp.callback)
            .as_micros() as u64;
        if let Some(previous) = previous_callback_at {
            observation.max_callback_gap_us = observation
                .max_callback_gap_us
                .max(captured_at.duration_since(previous).as_micros() as u64);
        }
        previous_callback_at = Some(captured_at);
        // Keep all early returns inside the render operation, so
        // silence and contention still publish a fresh observation.
        let mut callback_silenced = false;
        let mut render_callback = || {
            let mut process_output = |data: &mut [T], buffer: &mut Vec<f32>| {
                if let Some(ref mut cb) = cb_config.process_callback {
                    cb(buffer);
                }

                for (dst, &sample) in data.iter_mut().zip(buffer.iter()) {
                    *dst = <T>::from_sample(sample);
                }
            };

            // Advance the gain ramp even while silent so the first real
            // audio resumes at the target gain with no fade-in.
            let mut emit_silence = |data: &mut [T]| {
                callback_silenced = true;
                let target = cb_config.gain_control.gain();
                gain_ramp.advance(data.len() / channels, target);
                f32_buffer.clear();
                f32_buffer.resize(data.len(), 0.0);
                process_output(data, &mut f32_buffer);
            };

            // Snapshot the level checks once per callback. At info level
            // these two loads are the only per-callback logging cost.
            let debug_logging = log::log_enabled!(log::Level::Debug);
            let trace_logging = log::log_enabled!(log::Level::Trace);

            let timestamp_observation =
                classify_output_timestamp(timestamp, timestamp_source, last_playback_timestamp);
            if !renderer_for_data.try_callback_telemetry(
                scope,
                timestamp_observation.source,
                timestamp_observation.monotonic_violation,
            ) {
                emit_silence(data);
                return;
            }
            last_playback_timestamp = Some(timestamp_observation.timestamp.playback);

            stats.callbacks += 1;
            let frames = data.len() / channels;

            // Read queue timing state together. The generation is
            // rechecked before consuming force_reanchor so a clear()
            // racing with this callback cannot clear the next startup's
            // one-shot handoff.
            let Some((
                start_state,
                generation,
                cursor_us,
                force_reanchor,
                delay_us,
                queued_us,
                queued_buffers,
            )) = try_callback_queue(
                &renderer_for_data,
                scope,
                &queue,
                frames,
                CallbackQueuePhase::TimingSnapshot,
                &mut observation,
                |queue, permit| {
                    let cursor = if queue.initialized {
                        Some(queue.cursor_us)
                    } else {
                        None
                    };
                    // Queue depth costs a walk over every queued buffer,
                    // so only measure it when a log line below can print
                    // it.
                    let (queued_us, queued_buffers) = if debug_logging {
                        (
                            queue.queued_duration_us(channels, sample_rate),
                            queue.buffer_count(),
                        )
                    } else {
                        (0, 0)
                    };
                    (
                        permit.start_state(),
                        queue.generation,
                        cursor,
                        queue.force_reanchor,
                        cb_config.static_delay_us.load(Ordering::Relaxed),
                        queued_us,
                        queued_buffers,
                    )
                },
            )
            else {
                emit_silence(data);
                return;
            };
            // This first branch is an advisory snapshot only. Public
            // arm/clear operations may legally change the state before
            // the render permit is acquired below, so the render permit
            // remains the sole authority for the transition decision.
            match start_state {
                StartState::Idle | StartState::Armed { .. } | StartState::BoundaryWon { .. } => {}
            }
            observation.generation = generation;
            if generation != last_generation {
                log::debug!(
            "Playback queue generation changed: {} -> {}, queued={:.1}ms, buffers={}, callbacks={}, silent_callbacks={}, underrun_callbacks={}, underrun_frames={}, sync_lock_misses={}, correction_engagements={}",
            last_generation,
            generation,
            us_to_ms(queued_us),
            queued_buffers,
            stats.callbacks,
            stats.silent_callbacks,
            stats.underrun_callbacks,
            stats.underrun_frames,
            stats.sync_lock_misses,
            stats.correction_engagements,
        );
                last_generation = generation;
                started = false;
                schedule = CorrectionSchedule::default();
                insert_counter = 0;
                drop_counter = 0;
                error_filter.reset();
                engage_gate.reset();
                min_playback_delta = None;
                last_measured_playback_delta = None;
                stats.reset_for_generation();
                for sample in last_frame.iter_mut() {
                    *sample = i32::EQUILIBRIUM;
                }
                handoff_warned = false;
            }

            let callback_instant = captured_at;
            let ts = timestamp_observation.timestamp;
            let measured_playback_delta = (timestamp_source
                != cpal::OutputTimestampSource::MonotonicFallback
                && ts.playback >= ts.callback)
                .then(|| ts.playback.duration_since(ts.callback));
            if let Some(delta) = measured_playback_delta {
                min_playback_delta =
                    Some(min_playback_delta.map_or(delta, |floor| floor.min(delta)));
                last_measured_playback_delta = Some(delta);
            }
            // Scheduling must keep advancing even without a device measurement. Reuse
            // only a latency duration, never an old absolute presentation instant.
            // The initial one-buffer estimate is advisory and cannot train correction.
            let callback_period = Duration::from_secs_f64(frames as f64 / sample_rate as f64);
            let playback_delta = measured_playback_delta
                .or(last_measured_playback_delta)
                .unwrap_or(callback_period);
            let playback_instant = callback_instant + playback_delta;
            let mut presentation_zone_us = None;

            // Both values are normally steady, so a step in either
            // explains a sync-error step: a callback gap means this
            // thread stalled; a playback-delta shift means the OS
            // moved the presentation timeline.
            let playback_delta_us = playback_delta.as_micros() as u64;
            if let Some(last) = last_callback_instant {
                let gap_us = callback_instant.duration_since(last).as_micros() as u64;
                let period_us = frames as u64 * 1_000_000 / u64::from(sample_rate.max(1));
                if gap_us >= 2 * period_us {
                    log::debug!(
                "Audio callback gap: {:.1}ms since previous (period ~{:.1}ms), callback={}, generation={}",
                us_to_ms(gap_us),
                us_to_ms(period_us),
                stats.callbacks,
                generation,
            );
                }
            }
            last_callback_instant = Some(callback_instant);
            if let Some(last) = last_playback_delta_us {
                if playback_delta_us.abs_diff(last) > 1_000 {
                    log::debug!(
                "Output timeline shifted: playback_delta {:.1}ms -> {:.1}ms, callback={}, generation={}",
                us_to_ms(last),
                us_to_ms(playback_delta_us),
                stats.callbacks,
                generation,
            );
                }
            }
            last_playback_delta_us = Some(playback_delta_us);

            // try_lock: skip sync if contended rather than blocking
            // the audio thread. force_reanchor is sticky in the
            // queue, so it will be retried on the next callback.
            let sync = clock_sync.try_lock();
            if cursor_us.is_some() && sync.is_none() {
                // Count lock contention only once playback has an
                // initialized cursor; before that there is no timeline
                // position to synchronize yet.
                stats.sync_lock_misses += 1;
                observation.sync_lock_misses = observation.sync_lock_misses.saturating_add(1);
                if trace_logging && should_log_sample(stats.sync_lock_misses) {
                    log::trace!(
                "Audio callback skipped sync: clock lock contended, callback={}, sync_lock_miss={}, queued={:.1}ms, buffers={}, started={}",
                stats.callbacks,
                stats.sync_lock_misses,
                us_to_ms(queued_us),
                queued_buffers,
                started,
            );
                }
            }
            if let (Some(cursor_us), Some(sync)) = (cursor_us, sync) {
                presentation_zone_us = canonical_presentation_zone_us(
                    sync.client_to_server_micros(sync.instant_to_client_micros(playback_instant)),
                    delay_us,
                );
                // Emit each sample `delay` earlier so downstream
                // (amp/speaker) latency lands it on time. The reanchor
                // below adds the same delay in the local→server
                // direction; the two signs must stay in step or the
                // planner chases a phantom error every callback.
                let mut effective_cursor_us = cursor_us;
                let sync_settled = sync.is_settled();
                if sync_settled && !sync_settle_logged {
                    sync_settle_logged = true;
                    // Warm-up measurements track the converging clock
                    // estimate, not playback; start the filter fresh.
                    error_filter.reset();
                    engage_gate.reset();
                    log::debug!(
                "Clock sync settled; corrections enabled: callback={}, suppressed_during_warmup={}, generation={}",
                stats.callbacks,
                stats.warmup_suppressed_corrections,
                generation,
            );
                }

                if force_reanchor {
                    let mut reanchor_applied = false;
                    // Startup/explicit handoff can use the scheduling estimate when
                    // no device measurement exists. It never updates the measured floor.
                    let anchor_instant =
                        callback_instant + min_playback_delta.unwrap_or(playback_delta);
                    let handoff_instant = if started {
                        anchor_instant
                    } else {
                        // Startup handoff: anchor to `+ handoff_delta` (this buffer's
                        // end = the next callback's start) so the next start gate sees
                        // `expected ≈ playback_instant`. Playing now would misalign the
                        // cursor by one buffer, so we stay silent for this one period.
                        let handoff_delta =
                            Duration::from_secs_f64(frames as f64 / sample_rate as f64);
                        anchor_instant + handoff_delta
                    };
                    let client_micros =
                        sync.instant_to_client_micros(handoff_instant) + delay_us as i64;
                    if let Some(server_time) = sync.client_to_server_micros(client_micros) {
                        let Some(outcome) = try_callback_queue(
                            &renderer_for_data,
                            scope,
                            &queue,
                            frames,
                            CallbackQueuePhase::StartupReanchor,
                            &mut observation,
                            |queue, _permit| {
                                if queue.generation != generation || !queue.initialized {
                                    return StartupReanchorOutcome::Stale;
                                }
                                let Some(cursor_us) =
                                    queue.first_playable_cursor_at_or_after(server_time)
                                else {
                                    return StartupReanchorOutcome::NoPlayable;
                                };
                                queue.cursor_us = cursor_us;
                                queue.cursor_remainder = 0;
                                queue.force_reanchor = false;
                                StartupReanchorOutcome::Applied(cursor_us)
                            },
                        ) else {
                            emit_silence(data);
                            return;
                        };
                        match outcome {
                            StartupReanchorOutcome::Applied(cursor_us) => {
                                observation.startup_reanchors =
                                    observation.startup_reanchors.saturating_add(1);
                                effective_cursor_us = cursor_us;
                                reanchor_applied = true;
                                schedule = CorrectionSchedule::default();
                                insert_counter = 0;
                                drop_counter = 0;
                                // The cursor just jumped (e.g. a
                                // static-delay change, which does not
                                // bump the generation); prior
                                // measurements describe the old
                                // timeline.
                                error_filter.reset();
                                engage_gate.reset();
                                log::debug!(
                            "Sync reanchor applied: cursor reset to server_time={cursor_us}µs"
                        );
                            }
                            StartupReanchorOutcome::NoPlayable if !handoff_warned => {
                                handoff_warned = true;
                                log::warn!(
                                    "Sync reanchor: no playable buffer at or after \
                             server_time={server_time}µs — staying silent"
                                );
                            }
                            StartupReanchorOutcome::NoPlayable | StartupReanchorOutcome::Stale => {}
                        }
                    }

                    if !reanchor_applied || !started {
                        stats.silent_callbacks += 1;
                        if trace_logging && should_log_sample(stats.silent_callbacks) {
                            log::trace!(
                        "Audio callback silent during reanchor: callback={}, silent_callback={}, reanchor_applied={}, started={}, queued={:.1}ms, buffers={}, generation={}",
                        stats.callbacks,
                        stats.silent_callbacks,
                        reanchor_applied,
                        started,
                        us_to_ms(queued_us),
                        queued_buffers,
                        generation,
                    );
                        }
                        emit_silence(data);
                        return;
                    }
                }

                if let Some(expected_instant) =
                    sync.server_to_local_instant_with_latency(effective_cursor_us, delay_us)
                {
                    // Pre-start only: hold silence until the cursor's
                    // scheduled instant. After start, "early" readings
                    // are jitter — injecting silence here caused real
                    // dropouts (audible blips); the planner handles
                    // sustained earliness instead.
                    let early_window = Duration::from_millis(1);
                    if !started && playback_instant + early_window < expected_instant {
                        stats.silent_callbacks += 1;
                        if trace_logging && should_log_sample(stats.silent_callbacks) {
                            let early_us = expected_instant
                                .duration_since(playback_instant)
                                .as_micros() as u64;
                            log::trace!(
                        "Audio callback early; emitting silence: callback={}, silent_callback={}, early={:.1}ms, cursor={}µs, queued={:.1}ms, buffers={}, generation={}",
                        stats.callbacks,
                        stats.silent_callbacks,
                        us_to_ms(early_us),
                        effective_cursor_us,
                        us_to_ms(queued_us),
                        queued_buffers,
                        generation,
                    );
                        }
                        emit_silence(data);
                        return;
                    }
                    if !started {
                        started = true;
                        log::debug!(
                    "Audio playback started: callback={}, cursor={}µs, queued={:.1}ms, buffers={}, silent_callbacks_before_start={}, sync_lock_misses={}",
                    stats.callbacks,
                    effective_cursor_us,
                    us_to_ms(queued_us),
                    queued_buffers,
                    stats.silent_callbacks,
                    stats.sync_lock_misses,
                );
                    }

                    if measured_playback_delta.is_some() {
                        let raw_error_us = if playback_instant >= expected_instant {
                            playback_instant
                                .duration_since(expected_instant)
                                .as_micros() as i64
                        } else {
                            -(expected_instant
                                .duration_since(playback_instant)
                                .as_micros() as i64)
                        };
                        // A single reading can sit a whole engine period
                        // above true alignment while the FIFO plays
                        // gaplessly (see SyncErrorFilter); plan against
                        // the window floor, never one wake's snapshot.
                        let error_us = error_filter.update(raw_error_us);
                        observation.raw_error_us = Some(raw_error_us);
                        observation.filtered_error_us = Some(error_us);
                        let planned_schedule =
                            planner.plan(error_us, sample_rate, schedule.is_correcting());
                        // Corrections mutate audible frames: engage only
                        // on sustained evidence over a warm filter (see
                        // EngageGate).
                        let gated_schedule = engage_gate.admit(
                            planned_schedule,
                            schedule.is_correcting(),
                            error_filter.is_warm(),
                        );
                        if gated_schedule != planned_schedule {
                            stats.gate_suppressed_corrections += 1;
                            if trace_logging && should_log_sample(stats.gate_suppressed_corrections)
                            {
                                log::trace!(
                            "Sync correction awaiting sustained error: callback={}, suppressed={}, error={:.3}ms, raw_error={:.3}ms, generation={}",
                            stats.callbacks,
                            stats.gate_suppressed_corrections,
                            error_us as f64 / 1000.0,
                            raw_error_us as f64 / 1000.0,
                            generation,
                        );
                            }
                        }
                        let planned_schedule = gated_schedule;
                        // While the sync estimate is still converging,
                        // measured error is mostly movement of the
                        // estimate itself; correcting for it chases
                        // filter noise audibly. Trust the server's audio
                        // until settled, honoring only gross reanchors.
                        let new_schedule = if sync_settled || planned_schedule.reanchor {
                            planned_schedule
                        } else {
                            if planned_schedule.is_correcting() {
                                stats.warmup_suppressed_corrections += 1;
                                if trace_logging
                                    && should_log_sample(stats.warmup_suppressed_corrections)
                                {
                                    log::trace!(
                                "Sync correction suppressed during clock warm-up: callback={}, suppressed={}, error={:.3}ms, raw_error={:.3}ms, generation={}",
                                stats.callbacks,
                                stats.warmup_suppressed_corrections,
                                error_us as f64 / 1000.0,
                                raw_error_us as f64 / 1000.0,
                                generation,
                            );
                                }
                            }
                            CorrectionSchedule::default()
                        };
                        if new_schedule != schedule {
                            if new_schedule.is_correcting() != schedule.is_correcting() {
                                if new_schedule.is_correcting() {
                                    stats.correction_engagements += 1;
                                    log::debug!(
                                "Sync correction engaged: error={:.3}ms, raw_error={:.3}ms, insert_every={}, drop_every={}, reanchor={}, callback={}, generation={}",
                                error_us as f64 / 1000.0,
                                raw_error_us as f64 / 1000.0,
                                new_schedule.insert_every_n_frames,
                                new_schedule.drop_every_n_frames,
                                new_schedule.reanchor,
                                stats.callbacks,
                                generation,
                            );
                                } else {
                                    // The floor lags rises, so error= may
                                    // read worse than raw_error= here;
                                    // expected, not a bug.
                                    log::debug!(
                                "Sync correction disengaged: error={:.3}ms, raw_error={:.3}ms, callback={}, generation={}",
                                error_us as f64 / 1000.0,
                                raw_error_us as f64 / 1000.0,
                                stats.callbacks,
                                generation,
                            );
                                }
                            }
                            if new_schedule.is_correcting() {
                                // The cadence is re-planned as the error
                                // converges, which can change the schedule
                                // on every callback. Sample the updates so
                                // each correction episode logs its first
                                // few adjustments and then a heartbeat;
                                // engage/disengage transitions are logged
                                // at debug above and reanchor execution is
                                // logged where it is applied below.
                                stats.correction_updates += 1;
                                if trace_logging && should_log_sample(stats.correction_updates) {
                                    log::trace!(
                                "Sync correction updated: callback={}, correction_update={}, error={:.3}ms, raw_error={:.3}ms, insert_every={}, drop_every={}, reanchor={}, queued={:.1}ms, generation={}",
                                stats.callbacks,
                                stats.correction_updates,
                                error_us as f64 / 1000.0,
                                raw_error_us as f64 / 1000.0,
                                new_schedule.insert_every_n_frames,
                                new_schedule.drop_every_n_frames,
                                new_schedule.reanchor,
                                us_to_ms(queued_us),
                                generation,
                            );
                                }
                            } else {
                                stats.correction_updates = 0;
                            }
                            schedule = new_schedule;
                            insert_counter = schedule.insert_every_n_frames;
                            drop_counter = schedule.drop_every_n_frames;
                        }

                        if schedule.reanchor {
                            // Mirror of the start-gate subtraction: audio
                            // emitted now is heard `delay_us` later, so
                            // anchor the cursor to that hear-instant —
                            // derived from the delta floor, not this
                            // wake's reading (see min_playback_delta).
                            let anchor_instant =
                                callback_instant + min_playback_delta.unwrap_or(playback_delta);
                            let client_micros =
                                sync.instant_to_client_micros(anchor_instant) + delay_us as i64;
                            if let Some(server_time) = sync.client_to_server_micros(client_micros) {
                                if try_callback_queue(
                                    &renderer_for_data,
                                    scope,
                                    &queue,
                                    frames,
                                    CallbackQueuePhase::CorrectionReanchor,
                                    &mut observation,
                                    |queue, _permit| {
                                        queue.cursor_us = server_time;
                                        queue.cursor_remainder = 0;
                                    },
                                )
                                .is_none()
                                {
                                    emit_silence(data);
                                    return;
                                }
                                observation.correction_reanchors =
                                    observation.correction_reanchors.saturating_add(1);
                                observation.last_reanchor_error_us = Some(error_us);
                                log::debug!(
                            "Sync reanchor applied: cursor reset to server_time={server_time}µs"
                        );
                            }
                            schedule = CorrectionSchedule::default();
                            insert_counter = 0;
                            drop_counter = 0;
                            stats.correction_updates = 0;
                            // The cursor just jumped; prior measurements
                            // describe the old timeline.
                            error_filter.reset();
                            engage_gate.reset();
                        }
                    }
                } else if schedule.is_correcting() {
                    // Conversions went dark (the implausible-drift
                    // safety net): stop correcting rather than
                    // resample blind on the stale cadence.
                    log::debug!(
                "Sync conversions unavailable; clearing correction schedule: callback={}, generation={}",
                stats.callbacks,
                generation,
            );
                    schedule = CorrectionSchedule::default();
                    insert_counter = 0;
                    drop_counter = 0;
                    stats.correction_updates = 0;
                    error_filter.reset();
                    engage_gate.reset();
                }
            }

            // If playback hasn't started yet (clock sync not converged,
            // lock contention, or pre-start gate active), output silence.
            // Audio data stays in the ring buffer for when sync converges
            // and reanchor positions the cursor correctly.
            if !started {
                stats.silent_callbacks += 1;
                if trace_logging && should_log_sample(stats.silent_callbacks) {
                    log::trace!(
                "Audio callback silent before start: callback={}, silent_callback={}, cursor_present={}, queued={:.1}ms, buffers={}, generation={}",
                stats.callbacks,
                stats.silent_callbacks,
                cursor_us.is_some(),
                us_to_ms(queued_us),
                queued_buffers,
                generation,
            );
                }
                emit_silence(data);
                return;
            }

            let Some((mut renderer_permit, mut queue_guard)) = try_callback_queue_guards(
                &renderer_for_data,
                scope,
                &queue,
                frames,
                CallbackQueuePhase::Render,
                &mut observation,
            ) else {
                emit_silence(data);
                return;
            };
            match renderer_permit.scheduled_start(presentation_zone_us) {
                ScheduledStartOutcome::Waiting { .. } => {
                    drop(queue_guard);
                    drop(renderer_permit);
                    stats.silent_callbacks += 1;
                    emit_silence(data);
                    return;
                }
                ScheduledStartOutcome::BoundaryWon { start_at_zone_us } => {
                    log::debug!(
                "Scheduled presentation boundary won: start_at_zone_us={start_at_zone_us}, callback={}, generation={}",
                stats.callbacks,
                generation,
            );
                }
                ScheduledStartOutcome::Unscheduled | ScheduledStartOutcome::Started { .. } => {}
                ScheduledStartOutcome::Closed | ScheduledStartOutcome::StaleScope => {
                    debug_assert!(false, "callback permit guarantees an open current scope");
                    drop(queue_guard);
                    drop(renderer_permit);
                    renderer_for_data.record_callback_underrun(frames as u64);
                    emit_silence(data);
                    return;
                }
            }
            f32_buffer.resize(data.len(), 0.0);

            let applied_schedule = if measured_playback_delta.is_some() {
                schedule
            } else {
                CorrectionSchedule::default()
            };
            observation.insert_every = applied_schedule.insert_every_n_frames;
            observation.drop_every = applied_schedule.drop_every_n_frames;
            let (callback_underrun_frames, queued_after_us, buffers_after) = {
                let queue = &mut *queue_guard;
                let mut missing_frames = 0u64;
                let mut consumed_frames = 0usize;
                let mut out_index = 0;

                for _ in 0..frames {
                    if applied_schedule.drop_every_n_frames > 0 {
                        drop_counter = drop_counter.saturating_sub(1);
                        if drop_counter == 0 {
                            // Discard one frame to catch up
                            if queue.consume_next_frame(channels, sample_rate, None) {
                                observation.dropped_frames =
                                    observation.dropped_frames.saturating_add(1);
                                consumed_frames += 1;
                            }
                            drop_counter = applied_schedule.drop_every_n_frames;
                            // Get and output the next frame (don't repeat last_frame)
                            if queue.consume_next_frame(
                                channels,
                                sample_rate,
                                Some(&mut last_frame),
                            ) {
                                consumed_frames += 1;
                                for sample in &last_frame {
                                    f32_buffer[out_index] = f32::from_sample(*sample);
                                    out_index += 1;
                                }
                            } else {
                                for sample in &last_frame {
                                    f32_buffer[out_index] = f32::from_sample(*sample);
                                    out_index += 1;
                                }
                            }
                            continue;
                        }
                    }

                    if applied_schedule.insert_every_n_frames > 0 {
                        insert_counter = insert_counter.saturating_sub(1);
                        if insert_counter == 0 {
                            insert_counter = applied_schedule.insert_every_n_frames;
                            observation.inserted_frames =
                                observation.inserted_frames.saturating_add(1);
                            for sample in &last_frame {
                                f32_buffer[out_index] = f32::from_sample(*sample);
                                out_index += 1;
                            }
                            continue;
                        }
                    }

                    if queue.consume_next_frame(channels, sample_rate, Some(&mut last_frame)) {
                        consumed_frames += 1;
                        for sample in &last_frame {
                            f32_buffer[out_index] = f32::from_sample(*sample);
                            out_index += 1;
                        }
                    } else {
                        missing_frames += 1;
                        for _ in 0..channels {
                            f32_buffer[out_index] = 0.0;
                            out_index += 1;
                        }
                    }
                }

                let buffers_after = queue.buffer_count();
                let queued_after_us = if debug_logging {
                    queue.queued_duration_us(channels, sample_rate)
                } else {
                    0
                };
                renderer_permit.record_actual_progress(
                    consumed_frames,
                    None,
                    queue.queued_frames(channels),
                    buffers_after,
                );
                (missing_frames, queued_after_us, buffers_after)
            };
            drop(queue_guard);
            drop(renderer_permit);
            renderer_for_data.record_callback_underrun(callback_underrun_frames);
            observation.underrun_frames = observation
                .underrun_frames
                .saturating_add(callback_underrun_frames);

            let recovered =
                callback_underrun_frames == 0 && stats.consecutive_underrun_callbacks > 0;
            if callback_underrun_frames > 0 {
                stats.underrun_frames += callback_underrun_frames;
                stats.underrun_callbacks += 1;
                stats.consecutive_underrun_callbacks += 1;

                // Per-generation totals reset on stream changes, so
                // every stream logs its first few underruns at debug.
                // That is intentional: startup underruns after a
                // clear/track change are the main diagnostic.
                if debug_logging
                    && (should_log_sample(stats.underrun_callbacks)
                        || should_log_sample(stats.consecutive_underrun_callbacks))
                {
                    log::debug!(
                "Audio underrun: callback={}, missing_frames={} ({:.1}ms), queued_before={:.1}ms, queued_after={:.1}ms, buffers_after={}, cursor={:?}µs, generation={}, underrun_frames={}, underrun_callbacks={}, consecutive_underrun_callbacks={}",
                stats.callbacks,
                callback_underrun_frames,
                callback_underrun_frames as f64 * 1000.0 / sample_rate as f64,
                us_to_ms(queued_us),
                us_to_ms(queued_after_us),
                buffers_after,
                cursor_us,
                generation,
                stats.underrun_frames,
                stats.underrun_callbacks,
                stats.consecutive_underrun_callbacks,
            );
                }
            } else if recovered {
                let underrun_run = stats.consecutive_underrun_callbacks;
                stats.consecutive_underrun_callbacks = 0;
                log::debug!(
            "Audio underrun recovered: callback={}, consecutive_underrun_callbacks={}, underrun_frames={}, queued_after={:.1}ms, buffers_after={}, generation={}",
            stats.callbacks,
            underrun_run,
            stats.underrun_frames,
            us_to_ms(queued_after_us),
            buffers_after,
            generation,
        );
            }

            // Edge-triggered low-queue warnings with hysteresis, so a
            // queue hovering at one boundary cannot flood the log.
            if debug_logging {
                if !stats.queue_low && queued_after_us < QUEUE_LOW_WATER_US {
                    stats.queue_low = true;
                    log::debug!(
                "Playback queue low: queued={:.1}ms, buffers={}, callback={}, underrun_frames={}, generation={}",
                us_to_ms(queued_after_us),
                buffers_after,
                stats.callbacks,
                stats.underrun_frames,
                generation,
            );
                } else if stats.queue_low && queued_after_us >= QUEUE_RECOVERED_WATER_US {
                    stats.queue_low = false;
                    log::debug!(
                "Playback queue recovered: queued={:.1}ms, buffers={}, callback={}, generation={}",
                us_to_ms(queued_after_us),
                buffers_after,
                stats.callbacks,
                generation,
            );
                }
            }

            // Apply gain with per-frame ramping
            let target = cb_config.gain_control.gain();
            gain_ramp.apply(&mut f32_buffer, channels, target);

            process_output(data, &mut f32_buffer);

            // One sampled health line per rendered callback, emitted
            // after gain and the user process callback so it describes
            // the audio actually delivered.
            if trace_logging
                && callback_underrun_frames == 0
                && !recovered
                && should_log_sample(stats.callbacks)
            {
                let peak_abs = f32_buffer
                    .iter()
                    .map(|sample| sample.abs())
                    .fold(0.0, f32::max);
                log::trace!(
            "Audio callback rendered: callback={}, frames={}, queued_before={:.1}ms, queued_after={:.1}ms, buffers_after={}, peak_abs={:.6}, generation={}",
            stats.callbacks,
            frames,
            us_to_ms(queued_us),
            us_to_ms(queued_after_us),
            buffers_after,
            peak_abs,
            generation,
        );
            }
        };
        render_callback();
        if callback_silenced {
            observation.silent_callbacks = observation.silent_callbacks.saturating_add(1);
            observation.silent_frames = observation
                .silent_frames
                .saturating_add(observation.callback_frames as u64);
        }
        diagnostics.publish(observation);
    }
}

impl Drop for SyncedPlayer {
    fn drop(&mut self) {
        let _ = self.teardown();
    }
}

#[cfg(test)]
mod tests {
    // Note: SyncedPlayer's convenience methods (volume, is_muted, set_volume,
    // set_mute, gain_control) delegate to GainControl which is thoroughly tested
    // in gain.rs. Canonical data callback tests are in callback_tests.

    use super::{
        abort_before_start_with_queue, canonical_presentation_zone_us, classify_output_timestamp,
        default_renderer_limits, set_device_delay_state, teardown_with_queue, try_callback_queue,
        validate_enqueue_buffer, validate_output_format, windows_default_buffer_frames,
        CallbackQueuePhase, DeviceDelayError, DeviceDelayMs, PlaybackQueue,
        SyncDiagnosticsSnapshot, MAX_STATIC_DELAY_MS,
    };
    use crate::audio::{
        AudioBuffer, AudioFormat, Codec, EnqueueOutcome, PlayerScope, PreStartAbortOutcome,
        RendererFault, RendererOperationOutcome, RendererOwner, RendererQueueLimits,
        ScheduledArmOutcome, ScheduledStartOutcome, StartState, TerminalOutcome,
    };
    use cpal::Sample;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{mpsc, Arc};
    use std::thread;

    /// Standard test format: 48kHz stereo 24-bit PCM.
    fn test_format() -> AudioFormat {
        AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        }
    }

    /// Mono variant of [`test_format`].
    fn test_format_mono() -> AudioFormat {
        AudioFormat {
            channels: 1,
            ..test_format()
        }
    }

    fn renderer_harness() -> (RendererOwner, PlayerScope, Arc<Mutex<PlaybackQueue>>) {
        let owner = RendererOwner::new(RendererQueueLimits::new(32, 8, 16).unwrap());
        let scope = owner.mint_scope().unwrap();
        (owner, scope, Arc::new(Mutex::new(PlaybackQueue::new())))
    }

    #[test]
    fn timestamp_provenance_callback_observation_uses_value_and_source_pair() {
        let first = cpal::OutputStreamTimestamp {
            callback: cpal::StreamInstant::from_nanos(10),
            playback: cpal::StreamInstant::from_nanos(20),
        };
        let equal = cpal::OutputStreamTimestamp {
            callback: cpal::StreamInstant::from_nanos(11),
            playback: cpal::StreamInstant::from_nanos(20),
        };
        let earlier = cpal::OutputStreamTimestamp {
            callback: cpal::StreamInstant::from_nanos(12),
            playback: cpal::StreamInstant::from_nanos(19),
        };
        let later = cpal::OutputStreamTimestamp {
            callback: cpal::StreamInstant::from_nanos(13),
            playback: cpal::StreamInstant::from_nanos(21),
        };

        let first_observation =
            classify_output_timestamp(first, cpal::OutputTimestampSource::DevicePresentation, None);
        assert_eq!(first_observation.timestamp, first);
        assert_eq!(
            first_observation.source,
            cpal::OutputTimestampSource::DevicePresentation
        );
        assert!(!first_observation.monotonic_violation);

        for (timestamp, source, expected_violation) in [
            (equal, cpal::OutputTimestampSource::MonotonicFallback, true),
            (earlier, cpal::OutputTimestampSource::Unspecified, true),
            (
                later,
                cpal::OutputTimestampSource::DevicePresentation,
                false,
            ),
        ] {
            let observation = classify_output_timestamp(timestamp, source, Some(first.playback));
            assert_eq!(observation.timestamp, timestamp);
            assert_eq!(observation.source, source);
            assert_eq!(observation.monotonic_violation, expected_violation);
        }
    }

    #[test]
    fn timestamp_provenance_survives_clear_and_reanchor_on_same_stream() {
        let (owner, scope, queue) = renderer_harness();
        assert!(owner.try_callback_telemetry(
            scope,
            cpal::OutputTimestampSource::DevicePresentation,
            false,
        ));
        let before = owner.health(scope).unwrap().output_timestamps();

        assert_eq!(owner.clear(scope), RendererOperationOutcome::Applied);
        let delay_us = AtomicU64::new(0);
        set_device_delay_state(&queue, &delay_us, DeviceDelayMs::new(25.0).unwrap());
        assert!(queue.lock().force_reanchor);

        assert_eq!(owner.health(scope).unwrap().output_timestamps(), before);
    }

    fn mono_buffer(timestamp: i64, frames: usize) -> AudioBuffer {
        AudioBuffer {
            timestamp,
            samples: Arc::from(vec![0i32; frames]),
            format: test_format_mono(),
        }
    }

    fn enqueue_harness(
        owner: &RendererOwner,
        scope: PlayerScope,
        queue: &Mutex<PlaybackQueue>,
        buffer: AudioBuffer,
    ) -> EnqueueOutcome {
        let frames = buffer.samples.len();
        let mut queue = queue.lock();
        owner.enqueue_with_actual(scope, frames, || {
            queue.push(buffer);
            (queue.queued_frames(1), queue.buffer_count())
        })
    }

    #[test]
    fn renderer_callback_commit_cannot_overwrite_later_enqueue() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));

        let mut permit = owner.try_callback_permit(scope).unwrap();
        let mut queue_guard = queue.try_lock().unwrap();
        assert!(queue_guard.next_frame(1, 48_000).is_some());
        let queued_frames = queue_guard.queued_frames(1);
        let queued_buffers = queue_guard.buffer_count();

        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let enqueue = {
            let owner = owner.clone();
            let queue = Arc::clone(&queue);
            thread::spawn(move || {
                started_tx.send(()).unwrap();
                let outcome = enqueue_harness(&owner, scope, &queue, mono_buffer(10_000, 2));
                done_tx.send(()).unwrap();
                outcome
            })
        };
        started_rx.recv().unwrap();
        assert_eq!(done_rx.try_recv(), Err(mpsc::TryRecvError::Empty));
        permit.record_actual_progress(1, None, queued_frames, queued_buffers);
        drop(queue_guard);
        drop(permit);
        assert!(matches!(
            enqueue.join().unwrap(),
            EnqueueOutcome::Accepted { .. }
        ));

        let queue = queue.lock();
        let capacity = owner.capacity(scope).unwrap();
        assert_eq!(capacity.current_frames(), queue.queued_frames(1));
        assert_eq!(capacity.current_buffers(), queue.buffer_count());
        assert_eq!(capacity.current_frames(), 5);
    }

    #[test]
    fn close_after_heartbeat_fences_callback_consumption() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        assert!(owner.try_callback_heartbeat(scope));
        assert_eq!(owner.close(scope), RendererOperationOutcome::Applied);
        assert!(owner.try_callback_permit(scope).is_none());

        let queue = queue.lock();
        assert_eq!(queue.queued_frames(1), 4);
        let health = owner.health(scope).unwrap();
        assert_eq!(health.queued_frames(), 4);
        assert_eq!(health.consumed_frames(), 0);
    }

    #[test]
    fn open_scope_queue_contention_silence_is_observable() {
        let (owner, scope, queue) = renderer_harness();
        let mut observation = SyncDiagnosticsSnapshot::default();
        let queue_guard = queue.lock();
        for phase in [
            CallbackQueuePhase::TimingSnapshot,
            CallbackQueuePhase::StartupReanchor,
            CallbackQueuePhase::CorrectionReanchor,
            CallbackQueuePhase::Render,
        ] {
            observation.callbacks += 1;
            assert!(owner.try_callback_heartbeat(scope));
            assert!(try_callback_queue(
                &owner,
                scope,
                &queue,
                6,
                phase,
                &mut observation,
                |_, _| { unreachable!("contended queue must not run callback operation") }
            )
            .is_none());
        }
        drop(queue_guard);

        let health = owner.health(scope).unwrap();
        assert_eq!(health.callback_count(), 4);
        assert_eq!(health.underrun_frames(), 24);
        assert_eq!(observation.queue_lock_misses, 4);
        assert_eq!(observation.renderer_access_misses, 0);
        assert_eq!(observation.access_silence_frames, 24);
        assert_eq!(observation.last_access_miss_callback, 4);
        assert_eq!(observation.last_access_miss_phase, Some("render"));
        assert_eq!(health.consumed_frames(), 0);
    }

    #[test]
    fn scheduled_start_production_queue_waits_wins_and_reuses_frozen_boundary() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        assert_eq!(
            owner.arm_scheduled_start(scope, 100),
            ScheduledArmOutcome::Armed
        );
        assert_eq!(
            owner.arm_scheduled_start(scope, 200),
            ScheduledArmOutcome::AlreadyArmed {
                start_at_zone_us: 100,
            }
        );

        assert!(owner.try_callback_heartbeat(scope));
        let waiting = try_callback_queue(
            &owner,
            scope,
            &queue,
            1,
            CallbackQueuePhase::Render,
            &mut SyncDiagnosticsSnapshot::default(),
            |queue, permit| {
                let decision = permit.scheduled_start(Some(99));
                (decision, queue.queued_frames(1), queue.buffer_count())
            },
        )
        .unwrap();
        assert_eq!(
            waiting,
            (
                ScheduledStartOutcome::Waiting {
                    start_at_zone_us: 100,
                },
                4,
                1,
            )
        );
        let waiting_health = owner.health(scope).unwrap();
        assert_eq!(waiting_health.callback_count(), 1);
        assert_eq!(waiting_health.consumed_frames(), 0);
        assert_eq!(waiting_health.last_presentation_boundary_zone_us(), None);

        assert!(owner.try_callback_heartbeat(scope));
        let won = try_callback_queue(
            &owner,
            scope,
            &queue,
            1,
            CallbackQueuePhase::Render,
            &mut SyncDiagnosticsSnapshot::default(),
            |queue, permit| {
                let decision = permit.scheduled_start(Some(100));
                assert!(queue.next_frame(1, 48_000).is_some());
                permit.record_actual_progress(
                    1,
                    None,
                    queue.queued_frames(1),
                    queue.buffer_count(),
                );
                decision
            },
        )
        .unwrap();
        assert_eq!(
            won,
            ScheduledStartOutcome::BoundaryWon {
                start_at_zone_us: 100,
            }
        );

        assert!(owner.try_callback_heartbeat(scope));
        let started = try_callback_queue(
            &owner,
            scope,
            &queue,
            1,
            CallbackQueuePhase::Render,
            &mut SyncDiagnosticsSnapshot::default(),
            |queue, permit| {
                let decision = permit.scheduled_start(None);
                assert!(queue.next_frame(1, 48_000).is_some());
                permit.record_actual_progress(
                    1,
                    None,
                    queue.queued_frames(1),
                    queue.buffer_count(),
                );
                decision
            },
        )
        .unwrap();
        assert_eq!(
            started,
            ScheduledStartOutcome::Started {
                start_at_zone_us: 100,
            }
        );
        let health = owner.health(scope).unwrap();
        assert_eq!(health.callback_count(), 3);
        assert_eq!(health.consumed_frames(), 2);
        assert_eq!(health.last_presentation_boundary_zone_us(), Some(100));

        {
            let mut queue = queue.lock();
            assert_eq!(
                owner.clear_with_actual(scope, || queue.clear()),
                RendererOperationOutcome::Applied
            );
            assert_eq!(queue.queued_frames(1), 0);
            assert_eq!(queue.buffer_count(), 0);
        }
        assert_eq!(
            owner.start_state(scope).unwrap(),
            crate::audio::StartState::BoundaryWon {
                start_at_zone_us: 100,
            }
        );
        assert_eq!(
            owner
                .health(scope)
                .unwrap()
                .last_presentation_boundary_zone_us(),
            Some(100)
        );
        assert_eq!(
            owner.arm_scheduled_start(scope, 200),
            ScheduledArmOutcome::BoundaryAlreadyWon {
                start_at_zone_us: 100,
            }
        );
    }

    #[test]
    fn scheduled_start_honors_positive_static_delay() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        assert_eq!(
            owner.arm_scheduled_start(scope, 100_000),
            ScheduledArmOutcome::Armed
        );

        let before = canonical_presentation_zone_us(Some(94_999), 5_000);
        let waiting = try_callback_queue(
            &owner,
            scope,
            &queue,
            1,
            CallbackQueuePhase::Render,
            &mut SyncDiagnosticsSnapshot::default(),
            |queue, permit| {
                let outcome = permit.scheduled_start(before);
                (outcome, queue.queued_frames(1), queue.buffer_count())
            },
        )
        .unwrap();
        assert_eq!(
            waiting,
            (
                ScheduledStartOutcome::Waiting {
                    start_at_zone_us: 100_000,
                },
                4,
                1,
            )
        );

        let at_boundary = canonical_presentation_zone_us(Some(95_000), 5_000);
        let won = try_callback_queue(
            &owner,
            scope,
            &queue,
            1,
            CallbackQueuePhase::Render,
            &mut SyncDiagnosticsSnapshot::default(),
            |queue, permit| {
                let outcome = permit.scheduled_start(at_boundary);
                assert!(queue.next_frame(1, 48_000).is_some());
                permit.record_actual_progress(
                    1,
                    None,
                    queue.queued_frames(1),
                    queue.buffer_count(),
                );
                outcome
            },
        )
        .unwrap();
        assert_eq!(
            won,
            ScheduledStartOutcome::BoundaryWon {
                start_at_zone_us: 100_000,
            }
        );
        assert_eq!(owner.health(scope).unwrap().consumed_frames(), 1);
        assert_eq!(canonical_presentation_zone_us(None, 5_000), None);
        assert_eq!(canonical_presentation_zone_us(Some(i64::MAX), 1), None);
        assert_eq!(canonical_presentation_zone_us(Some(0), u64::MAX), None);
    }

    #[test]
    fn scheduled_start_timing_snapshot_changes_are_advisory() {
        let (owner, idle_scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, idle_scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        let idle_snapshot = try_callback_queue(
            &owner,
            idle_scope,
            &queue,
            1,
            CallbackQueuePhase::TimingSnapshot,
            &mut SyncDiagnosticsSnapshot::default(),
            |_queue, permit| permit.start_state(),
        )
        .unwrap();
        assert_eq!(idle_snapshot, StartState::Idle);
        assert_eq!(
            owner.arm_scheduled_start(idle_scope, 100),
            ScheduledArmOutcome::Armed
        );
        let armed_after_idle_snapshot = try_callback_queue(
            &owner,
            idle_scope,
            &queue,
            1,
            CallbackQueuePhase::Render,
            &mut SyncDiagnosticsSnapshot::default(),
            |_queue, permit| permit.scheduled_start(Some(100)),
        )
        .unwrap();
        assert_eq!(
            armed_after_idle_snapshot,
            ScheduledStartOutcome::BoundaryWon {
                start_at_zone_us: 100,
            }
        );

        assert!(owner.teardown(idle_scope).finalization().is_some());
        queue.lock().clear();
        let armed_scope = owner.mint_scope().expect("first scope released");
        assert!(matches!(
            enqueue_harness(&owner, armed_scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        assert_eq!(
            owner.arm_scheduled_start(armed_scope, 200),
            ScheduledArmOutcome::Armed
        );
        let armed_snapshot = try_callback_queue(
            &owner,
            armed_scope,
            &queue,
            1,
            CallbackQueuePhase::TimingSnapshot,
            &mut SyncDiagnosticsSnapshot::default(),
            |_queue, permit| permit.start_state(),
        )
        .unwrap();
        assert_eq!(
            armed_snapshot,
            StartState::Armed {
                start_at_zone_us: 200,
            }
        );
        {
            let mut actual_queue = queue.lock();
            assert_eq!(
                owner.clear_with_actual(armed_scope, || actual_queue.clear()),
                RendererOperationOutcome::Applied
            );
        }
        let idle_after_armed_snapshot = try_callback_queue(
            &owner,
            armed_scope,
            &queue,
            1,
            CallbackQueuePhase::Render,
            &mut SyncDiagnosticsSnapshot::default(),
            |_queue, permit| permit.scheduled_start(Some(200)),
        )
        .unwrap();
        assert_eq!(
            idle_after_armed_snapshot,
            ScheduledStartOutcome::Unscheduled
        );
    }

    #[test]
    fn pre_start_abort_clears_actual_queue_and_fences_callback() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        assert_eq!(
            owner.arm_scheduled_start(scope, 100),
            ScheduledArmOutcome::Armed
        );

        let outcome = {
            let mut queue = queue.lock();
            owner.abort_before_start_with_actual(scope, || queue.clear())
        };
        assert!(matches!(outcome, PreStartAbortOutcome::Won(_)));
        assert!(owner.try_callback_permit(scope).is_none());
        let queue = queue.lock();
        assert_eq!(queue.queued_frames(1), 0);
        assert_eq!(queue.buffer_count(), 0);
        let capacity = owner.capacity(scope).unwrap();
        assert_eq!(capacity.current_frames(), 0);
        assert_eq!(capacity.current_buffers(), 0);
    }

    #[test]
    fn scheduled_start_callback_and_actual_abort_linearize() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        assert_eq!(
            owner.arm_scheduled_start(scope, 100),
            ScheduledArmOutcome::Armed
        );
        let mut permit = owner.try_callback_permit(scope).unwrap();
        let queue_guard = queue.lock();
        let (started_tx, started_rx) = mpsc::channel();
        let abort = {
            let owner = owner.clone();
            let queue = Arc::clone(&queue);
            thread::spawn(move || {
                started_tx.send(()).unwrap();
                abort_before_start_with_queue(&owner, scope, &queue, || {})
            })
        };
        started_rx.recv().unwrap();
        assert_eq!(
            permit.scheduled_start(Some(100)),
            ScheduledStartOutcome::BoundaryWon {
                start_at_zone_us: 100,
            }
        );
        drop(queue_guard);
        drop(permit);

        assert_eq!(
            abort.join().unwrap(),
            PreStartAbortOutcome::BoundaryAlreadyWon {
                start_at_zone_us: 100,
            }
        );
        assert!(matches!(
            owner.terminal_state(scope).unwrap(),
            crate::audio::TerminalState::Open
        ));
        assert_eq!(queue.lock().queued_frames(1), 4);
        assert_eq!(owner.capacity(scope).unwrap().current_frames(), 4);

        assert!(owner.teardown(scope).finalization().is_some());
        queue.lock().clear();
        let abort_scope = owner.mint_scope().expect("first scope released");
        assert!(matches!(
            enqueue_harness(&owner, abort_scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        assert_eq!(
            owner.arm_scheduled_start(abort_scope, 200),
            ScheduledArmOutcome::Armed
        );

        let permit = owner.try_callback_permit(abort_scope).unwrap();
        let (queue_locked_tx, queue_locked_rx) = mpsc::channel();
        let abort = {
            let owner = owner.clone();
            let queue = Arc::clone(&queue);
            thread::spawn(move || {
                abort_before_start_with_queue(&owner, abort_scope, &queue, || {
                    queue_locked_tx.send(()).unwrap();
                })
            })
        };
        queue_locked_rx.recv().unwrap();
        assert!(queue.try_lock().is_none());
        drop(permit);

        let PreStartAbortOutcome::Won(finalization) = abort.join().unwrap() else {
            panic!("actual queue abort must win before the callback boundary")
        };
        assert_eq!(
            finalization.winner,
            crate::audio::TerminalWinner::PreStartAbort
        );
        assert!(owner.try_callback_permit(abort_scope).is_none());
        assert_eq!(queue.lock().queued_frames(1), 0);
        let capacity = owner.capacity(abort_scope).unwrap();
        assert_eq!(capacity.current_frames(), 0);
        assert_eq!(capacity.current_buffers(), 0);
    }

    #[test]
    fn clear_and_enqueue_share_queue_owner_linearization() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        let (locked_tx, locked_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let clear = {
            let owner = owner.clone();
            let queue = Arc::clone(&queue);
            thread::spawn(move || {
                let mut queue = queue.lock();
                locked_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                owner.clear_with_actual(scope, || queue.clear())
            })
        };
        locked_rx.recv().unwrap();
        let enqueue = {
            let owner = owner.clone();
            let queue = Arc::clone(&queue);
            thread::spawn(move || enqueue_harness(&owner, scope, &queue, mono_buffer(20_000, 3)))
        };
        release_tx.send(()).unwrap();
        assert_eq!(clear.join().unwrap(), RendererOperationOutcome::Applied);
        assert!(matches!(
            enqueue.join().unwrap(),
            EnqueueOutcome::Accepted { .. }
        ));

        let queue = queue.lock();
        let capacity = owner.capacity(scope).unwrap();
        assert_eq!(capacity.current_frames(), queue.queued_frames(1));
        assert_eq!(capacity.current_buffers(), queue.buffer_count());
        assert_eq!(
            (capacity.current_frames(), capacity.current_buffers()),
            (3, 1)
        );
    }

    #[test]
    fn clear_after_inflight_callback_clears_queue_and_owner_together() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));

        let mut permit = owner.try_callback_permit(scope).unwrap();
        let mut queue_guard = queue.try_lock().unwrap();
        assert!(queue_guard.next_frame(1, 48_000).is_some());
        let queued_frames = queue_guard.queued_frames(1);
        let queued_buffers = queue_guard.buffer_count();

        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let clear = {
            let owner = owner.clone();
            let queue = Arc::clone(&queue);
            thread::spawn(move || {
                started_tx.send(()).unwrap();
                let mut queue = queue.lock();
                let outcome = owner.clear_with_actual(scope, || queue.clear());
                done_tx.send(()).unwrap();
                outcome
            })
        };
        started_rx.recv().unwrap();
        assert_eq!(done_rx.try_recv(), Err(mpsc::TryRecvError::Empty));

        permit.record_actual_progress(1, None, queued_frames, queued_buffers);
        drop(queue_guard);
        drop(permit);
        assert_eq!(clear.join().unwrap(), RendererOperationOutcome::Applied);

        let queue = queue.lock();
        let capacity = owner.capacity(scope).unwrap();
        assert_eq!(queue.queued_frames(1), 0);
        assert_eq!(queue.buffer_count(), 0);
        assert_eq!(capacity.current_frames(), 0);
        assert_eq!(capacity.current_buffers(), 0);
    }

    #[test]
    fn test_output_preflight_rejects_zero_channels_without_device_access() {
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 0,
            bit_depth: 24,
            codec_header: None,
        };
        assert_eq!(
            validate_output_format(&format),
            Err(crate::audio::OpenError::UnsupportedFormat)
        );
    }

    #[test]
    fn test_windows_default_buffer_frames_is_40ms() {
        assert_eq!(windows_default_buffer_frames(44_100), 1_764);
        assert_eq!(windows_default_buffer_frames(48_000), 1_920);
        assert_eq!(windows_default_buffer_frames(192_000), 7_680);
    }

    #[test]
    fn device_delay_validates_without_clamping() {
        assert_eq!(DeviceDelayMs::new(0.0).unwrap().as_micros(), 0);
        assert_eq!(DeviceDelayMs::new(100.25).unwrap().as_micros(), 100_250);
        assert_eq!(
            DeviceDelayMs::new(f64::from(MAX_STATIC_DELAY_MS))
                .unwrap()
                .as_micros(),
            MAX_STATIC_DELAY_MS as u64 * 1_000
        );
        assert_eq!(DeviceDelayMs::new(-0.1), Err(DeviceDelayError::OutOfRange));
        assert_eq!(
            DeviceDelayMs::new(5000.1),
            Err(DeviceDelayError::OutOfRange)
        );
        assert_eq!(
            DeviceDelayMs::new(f64::NAN),
            Err(DeviceDelayError::NonFinite)
        );
        assert_eq!(
            DeviceDelayMs::new(f64::INFINITY),
            Err(DeviceDelayError::NonFinite)
        );
    }

    #[test]
    fn device_delay_value_and_reanchor_are_published_under_one_queue_lock() {
        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        queue.lock().initialized = true;
        queue.lock().force_reanchor = false;
        let delay_us = Arc::new(AtomicU64::new(0));
        let held = queue.lock();
        let update = {
            let queue = Arc::clone(&queue);
            let delay_us = Arc::clone(&delay_us);
            thread::spawn(move || {
                set_device_delay_state(&queue, &delay_us, DeviceDelayMs::new(25.0).unwrap())
            })
        };
        thread::yield_now();
        assert_eq!(delay_us.load(Ordering::Relaxed), 0);
        drop(held);
        update.join().unwrap();
        let snapshot = queue.lock();
        assert!(snapshot.force_reanchor);
        assert_eq!(delay_us.load(Ordering::Relaxed), 25_000);
    }

    #[test]
    fn default_renderer_capacity_uses_gate0_hard_limits() {
        let limits = default_renderer_limits(48_000).unwrap();
        assert_eq!(limits.hard_frames(), 96_000);
        assert_eq!(limits.hard_buffers(), 64);
        assert_eq!(limits.max_chunk_frames(), 48_000);

        let high_rate = default_renderer_limits(192_000).unwrap();
        assert_eq!(high_rate.hard_frames(), 96_000);
        assert_eq!(high_rate.hard_buffers(), 64);
        assert_eq!(high_rate.max_chunk_frames(), 96_000);
    }

    #[test]
    fn test_queue_clear_bumps_generation() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 96];
        queue.push(AudioBuffer {
            timestamp: 1234,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        let before = queue.generation;
        queue.clear();
        assert_ne!(queue.generation, before);
        assert!(queue.queue.is_empty());
        assert!(!queue.initialized);
    }

    #[test]
    fn capacity_lifetime_token_follows_queued_buffer_until_clear() {
        struct DropCounter(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut queue = PlaybackQueue::new();
        queue.push_with_lifetime(
            AudioBuffer {
                timestamp: 1234,
                samples: Arc::from(vec![i32::EQUILIBRIUM; 96].into_boxed_slice()),
                format: test_format(),
            },
            Some(Arc::new(DropCounter(Arc::clone(&drops)))),
        );
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        queue.clear();
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn capacity_lifetime_releases_on_last_frame_and_teardown() {
        struct DropCounter(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut queue = PlaybackQueue::new();
        queue.push_with_lifetime(
            AudioBuffer {
                timestamp: 0,
                samples: Arc::from(vec![1].into_boxed_slice()),
                format: AudioFormat {
                    codec: Codec::Pcm,
                    sample_rate: 48_000,
                    channels: 1,
                    bit_depth: 32,
                    codec_header: None,
                },
            },
            Some(Arc::new(DropCounter(Arc::clone(&drops)))),
        );
        assert!(queue.consume_next_frame(1, 48_000, None));
        assert_eq!(queue.buffer_count(), 0);
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);

        let owner = RendererOwner::new(RendererQueueLimits::new(8, 2, 4).unwrap());
        let scope = owner.mint_scope().unwrap();
        let queue = Mutex::new(PlaybackQueue::new());
        {
            let mut queue = queue.lock();
            let outcome = owner.enqueue_with_actual(scope, 1, || {
                queue.push_with_lifetime(
                    mono_buffer(1_000, 1),
                    Some(Arc::new(DropCounter(Arc::clone(&drops)))),
                );
                (queue.queued_frames(1), queue.buffer_count())
            });
            assert!(matches!(outcome, EnqueueOutcome::Accepted { .. }));
        }
        let before = owner.capacity(scope).unwrap();
        assert_eq!(before.current_frames(), 1);
        assert_eq!(before.current_buffers(), 1);
        let _ = teardown_with_queue(&owner, scope, &queue);
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(queue.lock().buffer_count(), 0);
        let after = owner.capacity(scope).unwrap();
        assert_eq!(after.current_frames(), 0);
        assert_eq!(after.current_buffers(), 0);
    }

    #[test]
    fn terminal_loser_reconciles_actual_queue_after_fault_or_retained_owner_teardown() {
        struct DropCounter(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        for mark_fault in [false, true] {
            let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let owner = RendererOwner::new(RendererQueueLimits::new(8, 2, 4).unwrap());
            let scope = owner.mint_scope().unwrap();
            let queue = Mutex::new(PlaybackQueue::new());
            {
                let mut queue = queue.lock();
                assert!(matches!(
                    owner.enqueue_with_actual(scope, 1, || {
                        queue.push_with_lifetime(
                            mono_buffer(1_000, 1),
                            Some(Arc::new(DropCounter(Arc::clone(&drops)))),
                        );
                        (queue.queued_frames(1), queue.buffer_count())
                    }),
                    EnqueueOutcome::Accepted { .. }
                ));
            }
            if mark_fault {
                assert_eq!(
                    owner.fault(scope, RendererFault::CallbackFailed),
                    RendererOperationOutcome::Applied
                );
            }

            let _ = owner.teardown(scope);
            let _ = teardown_with_queue(&owner, scope, &queue);
            let _ = teardown_with_queue(&owner, scope, &queue);

            assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert_eq!(queue.lock().buffer_count(), 0);
            let capacity = owner.capacity(scope).unwrap();
            assert_eq!(capacity.current_frames(), 0);
            assert_eq!(capacity.current_buffers(), 0);
        }
    }

    #[test]
    fn concurrent_terminal_requests_release_actual_queue_once() {
        struct DropCounter(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let owner = Arc::new(RendererOwner::new(
            RendererQueueLimits::new(8, 2, 4).unwrap(),
        ));
        let scope = owner.mint_scope().unwrap();
        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        {
            let mut queue = queue.lock();
            assert!(matches!(
                owner.enqueue_with_actual(scope, 1, || {
                    queue.push_with_lifetime(
                        mono_buffer(1_000, 1),
                        Some(Arc::new(DropCounter(Arc::clone(&drops)))),
                    );
                    (queue.queued_frames(1), queue.buffer_count())
                }),
                EnqueueOutcome::Accepted { .. }
            ));
        }
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let retained_owner = Arc::clone(&owner);
        let retained_barrier = Arc::clone(&barrier);
        let retained = std::thread::spawn(move || {
            retained_barrier.wait();
            retained_owner.teardown(scope)
        });
        let player_owner = Arc::clone(&owner);
        let player_queue = Arc::clone(&queue);
        let player_barrier = Arc::clone(&barrier);
        let player = std::thread::spawn(move || {
            player_barrier.wait();
            teardown_with_queue(&player_owner, scope, &player_queue)
        });
        barrier.wait();
        retained.join().unwrap();
        player.join().unwrap();
        let _ = teardown_with_queue(&owner, scope, &queue);

        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(queue.lock().buffer_count(), 0);
        let capacity = owner.capacity(scope).unwrap();
        assert_eq!(capacity.current_frames(), 0);
        assert_eq!(capacity.current_buffers(), 0);
    }

    #[test]
    fn pending_actual_teardown_claim_blocks_scope_rotation_until_stable_ack() {
        struct DropCounter(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
        struct BlockingDrop {
            started: mpsc::Sender<()>,
            release: mpsc::Receiver<()>,
        }
        impl Drop for BlockingDrop {
            fn drop(&mut self) {
                self.started.send(()).unwrap();
                self.release.recv().unwrap();
            }
        }

        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let owner = Arc::new(RendererOwner::new(
            RendererQueueLimits::new(8, 2, 4).unwrap(),
        ));
        let scope = owner.mint_scope().unwrap();
        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        {
            let mut queue = queue.lock();
            assert!(matches!(
                owner.enqueue_with_actual(scope, 1, || {
                    queue.push_with_lifetime(
                        mono_buffer(1_000, 1),
                        Some(Arc::new(DropCounter(Arc::clone(&drops)))),
                    );
                    (queue.queued_frames(1), queue.buffer_count())
                }),
                EnqueueOutcome::Accepted { .. }
            ));
        }
        let (drop_started_tx, drop_started_rx) = mpsc::channel();
        let (release_drop_tx, release_drop_rx) = mpsc::channel();
        assert_eq!(
            owner.attach_test_terminal_resource(
                scope,
                Box::new(BlockingDrop {
                    started: drop_started_tx,
                    release: release_drop_rx,
                }),
            ),
            RendererOperationOutcome::Applied
        );

        let winner_owner = Arc::clone(&owner);
        let winner = std::thread::spawn(move || winner_owner.teardown(scope));
        drop_started_rx.recv().unwrap();
        let loser_owner = Arc::clone(&owner);
        let loser_queue = Arc::clone(&queue);
        let loser =
            std::thread::spawn(move || teardown_with_queue(&loser_owner, scope, &loser_queue));
        while queue.lock().buffer_count() != 0 {
            std::thread::yield_now();
        }

        let mint_owner = Arc::clone(&owner);
        let (minted_tx, minted_rx) = mpsc::channel();
        let mint = std::thread::spawn(move || {
            let next = mint_owner.mint_scope().unwrap();
            minted_tx.send(next).unwrap();
        });
        assert!(minted_rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());

        release_drop_tx.send(()).unwrap();
        let TerminalOutcome::Won(won) = winner.join().unwrap() else {
            panic!("retained owner must win terminal finalization")
        };
        let TerminalOutcome::Lost(observed) = loser.join().unwrap() else {
            panic!("claimed actual-aware loser must observe the stable final ack")
        };
        assert_eq!(won, observed);
        let next_scope = minted_rx.recv().unwrap();
        mint.join().unwrap();
        assert_ne!(scope, next_scope);
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn enqueue_validation_rejects_mismatch_malformed_and_timeline_overflow() {
        let format = test_format();
        let valid = AudioBuffer {
            timestamp: 0,
            samples: Arc::from(vec![0; 4].into_boxed_slice()),
            format: format.clone(),
        };
        assert_eq!(validate_enqueue_buffer(&format, &valid), Ok((2, 42)));

        let hard_limit = AudioBuffer {
            timestamp: 0,
            samples: Arc::from(vec![0; 96_000 * 2].into_boxed_slice()),
            format: format.clone(),
        };
        assert_eq!(
            validate_enqueue_buffer(&format, &hard_limit),
            Ok((96_000, 2_000_000))
        );

        let mismatch = AudioBuffer {
            format: AudioFormat {
                sample_rate: 44_100,
                ..format.clone()
            },
            ..valid
        };
        assert_eq!(
            validate_enqueue_buffer(&format, &mismatch),
            Err(EnqueueOutcome::FormatMismatch)
        );
        let malformed = AudioBuffer {
            timestamp: 0,
            samples: Arc::from(vec![0; 3].into_boxed_slice()),
            format: format.clone(),
        };
        assert_eq!(
            validate_enqueue_buffer(&format, &malformed),
            Err(EnqueueOutcome::InvalidBuffer)
        );
        let overflow = AudioBuffer {
            timestamp: i64::MAX,
            samples: Arc::from(vec![0; 2].into_boxed_slice()),
            format,
        };
        assert_eq!(
            validate_enqueue_buffer(&overflow.format, &overflow),
            Err(EnqueueOutcome::InvalidBuffer)
        );
    }

    #[test]
    fn test_queue_drops_stale_buffers() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Use distinct sample values so we can verify which buffer was returned.
        // 4800 stereo frames at 48kHz = 100ms per buffer.
        // With cursor at 150ms, the first buffer (ts=0, ends at 100ms) is stale.
        let stale_samples: Vec<i32> = (0..4800 * 2).map(|_| 111).collect();
        let fresh_samples: Vec<i32> = (0..4800 * 2).map(|_| 222).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(stale_samples.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 200_000,
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        queue.cursor_us = 150_000;
        queue.initialized = true;

        // Copy the frame data so we can release the mutable borrow on queue.
        let frame_data: Vec<i32> = queue
            .next_frame(2, 48_000)
            .expect("expected a frame")
            .to_vec();
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);
        // Verify we got the fresh buffer's data, not the stale one
        assert_eq!(frame_data[0], 222);
        assert_eq!(frame_data[1], 222);
    }

    #[test]
    fn test_queue_push_sorts_by_timestamp() {
        let mut queue = PlaybackQueue::new();
        let format = test_format_mono();

        // Push out of order: 300, 100, 200
        for ts in [300_000i64, 100_000, 200_000] {
            let samples = vec![ts as i32; 48]; // 1ms of mono
            queue.push(AudioBuffer {
                timestamp: ts,
                samples: Arc::from(samples.into_boxed_slice()),
                format: format.clone(),
            });
        }

        // Reset cursor so stale-buffer-dropping doesn't interfere with
        // the sort-order verification (in real usage, sync reanchor sets
        // cursor before playback begins).
        queue.cursor_us = 0;

        // Drain and verify sorted order
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 100_000);

        // Exhaust the first buffer (48 frames)
        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        // Next frame should come from the second buffer
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);

        // Exhaust the second buffer
        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        // Next frame should come from the third buffer
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 300_000);
    }

    #[test]
    fn test_queue_cursor_advances_correctly() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // 480 stereo frames = 10ms at 48kHz
        let num_frames = 480;
        let samples = vec![i32::EQUILIBRIUM; num_frames * 2];
        let start_ts = 1_000_000i64; // 1 second
        queue.push(AudioBuffer {
            timestamp: start_ts,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        // Consume all frames
        for _ in 0..num_frames {
            let _ = queue.next_frame(2, 48_000);
        }

        // 480 frames at 48kHz = 10,000us = 10ms
        let expected_end = start_ts + 10_000;
        assert_eq!(
            queue.cursor_us,
            expected_end,
            "cursor should advance by exactly 10ms (10000us), got delta={}",
            queue.cursor_us - start_ts
        );
    }

    #[test]
    fn test_cursor_does_not_advance_during_underrun() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Push one buffer to initialize the cursor
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo
        let start_ts = 1_000_000i64;
        queue.push(AudioBuffer {
            timestamp: start_ts,
            samples: Arc::from(samples.into_boxed_slice()),
            format: format.clone(),
        });

        // Consume all frames
        for _ in 0..480 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        let cursor_after_drain = queue.cursor_us;

        // Queue is now empty. Calling next_frame should return None
        // and NOT advance the cursor.
        for _ in 0..1000 {
            assert!(queue.next_frame(2, 48_000).is_none());
        }
        assert_eq!(
            queue.cursor_us,
            cursor_after_drain,
            "cursor must not advance during underrun; advanced by {}us",
            queue.cursor_us - cursor_after_drain
        );

        // Push a new buffer after the underrun. It should NOT be
        // dropped as stale — the cursor hasn't raced ahead.
        let fresh_samples: Vec<i32> = (0..480 * 2).map(|_| 999).collect();
        queue.push(AudioBuffer {
            timestamp: cursor_after_drain, // starts right where we left off
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("buffer should not be dropped as stale");
        assert_eq!(frame[0], 999, "should get the fresh buffer, not stale data");
    }

    #[test]
    fn test_push_initializes_cursor_from_first_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        assert!(!queue.initialized);
        assert_eq!(queue.cursor_us, 0);

        let samples = vec![i32::EQUILIBRIUM; 96];
        queue.push(AudioBuffer {
            timestamp: 500_000,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert!(queue.initialized);
        assert_eq!(queue.cursor_us, 500_000);
        assert_eq!(queue.cursor_remainder, 0);
    }

    #[test]
    fn test_push_does_not_regress_cursor_after_init() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 96];

        // First buffer at 500ms — initializes cursor
        queue.push(AudioBuffer {
            timestamp: 500_000,
            samples: Arc::from(samples.clone().into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.cursor_us, 500_000);

        // Consume a frame so cursor advances past init
        let _ = queue.next_frame(2, 48_000);
        let cursor_after_consume = queue.cursor_us;
        assert!(cursor_after_consume > 500_000);

        // Push an earlier buffer — cursor must NOT regress
        queue.push(AudioBuffer {
            timestamp: 200_000,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });
        assert_eq!(
            queue.cursor_us, cursor_after_consume,
            "cursor must not regress after playback has started"
        );
    }

    #[test]
    fn test_first_playable_cursor_skips_stale_audio() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.first_playable_cursor_at_or_after(5_000), Some(5_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(10_000), None);
    }

    #[test]
    fn test_first_playable_cursor_waits_for_future_audio() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 20_000,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.first_playable_cursor_at_or_after(5_000), Some(20_000));
    }

    #[test]
    fn test_first_playable_cursor_uses_current_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        // Pull one frame so the buffer moves from `queue` into `current`,
        // exercising the `current` branch of first_playable_cursor_at_or_after.
        let _ = queue.next_frame(2, 48_000);
        assert!(queue.current.is_some());
        assert!(queue.queue.is_empty());

        assert_eq!(queue.first_playable_cursor_at_or_after(5_000), Some(5_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(10_000), None);
    }

    #[test]
    fn test_first_playable_cursor_does_not_rewind_current_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        for _ in 0..240 {
            let _ = queue.next_frame(2, 48_000);
        }
        assert_eq!(queue.cursor_us, 5_000);

        assert_eq!(queue.first_playable_cursor_at_or_after(1_000), Some(5_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(6_000), Some(6_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(10_000), None);
    }

    #[test]
    fn test_next_frame_skips_into_overlapping_buffer() {
        // Simulates a backward timestamp jump from a server timeline rebase.
        // Buffer A is 50ms (2400 frames stereo at 48kHz). After consuming A
        // the cursor is at 50ms. Buffer B arrives at 25ms — the skip logic
        // should jump 25ms (1200 frames) into B.
        //
        // B is pushed AFTER A is consumed so dedup doesn't apply (A is no
        // longer in the queue).
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let buf_a: Vec<i32> = (0..2400 * 2).map(|_| 111).collect();
        let buf_b: Vec<i32> = (0..2400 * 2)
            .map(|i| {
                // First half (1200 frames) = 222, second half = 333
                if i < 2400 {
                    222
                } else {
                    333
                }
            })
            .collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(buf_a.into_boxed_slice()),
            format: format.clone(),
        });

        // Consume all of buffer A (2400 frames). Cursor advances to 50000µs.
        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        assert_eq!(queue.cursor_us, 50_000);

        // Push B after A is consumed — no dedup, tests skip logic only.
        // push() no longer regresses cursor_us after init, so cursor stays
        // at 50000 and the skip logic activates naturally.
        queue.push(AudioBuffer {
            timestamp: 25_000,
            samples: Arc::from(buf_b.into_boxed_slice()),
            format,
        });

        // Buffer B starts at 25ms but cursor is at 50ms, so 25ms (1200 frames)
        // should be skipped. First returned frame should be Sample(333).
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get a frame from buffer B");
        assert_eq!(
            frame[0], 333,
            "expected skip into second half of buffer B (past the overlap), \
             got first half — backward-timestamped audio was replayed"
        );
    }

    #[test]
    fn test_next_frame_no_skip_when_buffer_starts_at_or_after_cursor() {
        // Verify that the skip logic doesn't activate for normal (non-overlapping)
        // buffers — only for buffers that start before the cursor.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // 2400 frames = 50ms per buffer. Adjacent, non-overlapping.
        let samples_a: Vec<i32> = (0..2400 * 2).map(|_| 111).collect();
        let samples_b: Vec<i32> = (0..2400 * 2).map(|_| 222).collect();

        // Two consecutive, non-overlapping buffers.
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 50_000, // starts exactly where A ends
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Consume all of buffer A.
        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }

        // Buffer B starts at cursor (50000µs) — no skip should occur.
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get first frame of buffer B");
        assert_eq!(
            frame[0], 222,
            "buffer B should play from the start (no skip needed)"
        );
    }

    #[test]
    fn test_push_dedup_replaces_overlapping_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Buffer A: 10ms at ts=0 (480 stereo frames)
        let samples_a: Vec<i32> = (0..480 * 2).map(|_| 111).collect();
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 1);

        // Buffer B: 10ms at ts=5000 (5ms) — overlaps A's range [0, 10000)
        let samples_b: Vec<i32> = (0..480 * 2).map(|_| 222).collect();
        queue.push(AudioBuffer {
            timestamp: 5_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Should replace A, not add a second entry
        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].samples[0], 222);
    }

    #[test]
    fn test_push_dedup_no_false_positive_small_chunks() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Two adjacent 5ms chunks (240 stereo frames each).
        // Chunk A: [0, 5000), Chunk B: [5000, 10000) — no overlap.
        let samples_a: Vec<i32> = (0..240 * 2).map(|_| 111).collect();
        let samples_b: Vec<i32> = (0..240 * 2).map(|_| 222).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 5_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Both should be kept — they're adjacent, not overlapping
        assert_eq!(queue.queue.len(), 2);
        assert_eq!(queue.queue[0].timestamp, 0);
        assert_eq!(queue.queue[1].timestamp, 5_000);
    }

    #[test]
    fn test_push_keeps_chunks_on_44_1k_floor_timestamp_grid() {
        // Regression for the field "continuous popping" report: aiosendspin's
        // 25ms chunks at 44.1kHz are 1102 frames = 24988.66µs, and its
        // floor-based timestamp grid advances 24988µs for a third of chunks
        // while `duration_us` rounds every chunk to 24989µs. Those chunks
        // start 1µs "inside" their predecessor; the phantom overlap must not
        // evict anything (pre-fix it discarded ~34% of all queued audio).
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 44_100,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };

        // Mirror the server's residue arithmetic:
        // delta, residue = divmod(residue + frames * 1e6, rate).
        let mut ts = 282_697_405_880_i64; // first chunk ts from the field trace
        let mut residue = 0_i64;
        for _ in 0..12 {
            let samples: Vec<i32> = vec![0; 1102 * 2];
            queue.push(AudioBuffer {
                timestamp: ts,
                samples: Arc::from(samples.into_boxed_slice()),
                format: format.clone(),
            });
            residue += 1102 * 1_000_000;
            ts += residue / 44_100;
            residue %= 44_100;
        }

        assert_eq!(
            queue.queue.len(),
            12,
            "sub-frame timestamp jitter must never evict queued audio"
        );
    }

    #[test]
    fn test_push_keeps_floor_grid_chunks_at_all_supported_rates() {
        // The 44.1k regression swept across the whole rate family. 25ms
        // chunks are floor(rate/40) frames; the server grid advances by
        // floor-based (divmod) deltas. The mismatch between that grid and
        // our round-to-nearest duration_us is at most 1µs, which must stay
        // below the one-frame eviction threshold at every rate. (Only 44.1k
        // and 22.05k actually exhibit the mismatch; the rest divide 25ms
        // evenly and are exact.)
        for &rate in &[
            8_000_u32, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
            176_400, 192_000,
        ] {
            let mut queue = PlaybackQueue::new();
            let format = AudioFormat {
                codec: Codec::Pcm,
                sample_rate: rate,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            };
            let frames = (rate / 40) as usize;
            let mut ts = 1_000_000_i64;
            let mut residue = 0_i64;
            for _ in 0..16 {
                queue.push(AudioBuffer {
                    timestamp: ts,
                    samples: Arc::from(vec![0_i32; frames * 2].into_boxed_slice()),
                    format: format.clone(),
                });
                residue += frames as i64 * 1_000_000;
                ts += residue / i64::from(rate);
                residue %= i64::from(rate);
            }
            assert_eq!(
                queue.queue.len(),
                16,
                "rate {rate}: floor-grid timestamp jitter must not evict chunks"
            );
        }
    }

    #[test]
    fn test_push_dedup_still_evicts_one_frame_overlap_at_44_1k() {
        // Counterpart to the floor-grid regression: an overlap of a full
        // frame or more is real duplicate audio and must still dedup.
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 44_100,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };

        let samples_a: Vec<i32> = vec![111; 1102 * 2];
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        // Chunk A spans [0, 24989). A chunk rebased 1ms back into it
        // overlaps by ~1ms ≫ one frame (23µs): evict A.
        let samples_b: Vec<i32> = vec![222; 1102 * 2];
        queue.push(AudioBuffer {
            timestamp: 23_989,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].timestamp, 23_989);
    }

    #[test]
    fn test_push_dedup_removes_all_overlapping() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Buffer A: 10ms at ts=0 — range [0, 10000)
        let samples_a: Vec<i32> = (0..480 * 2).map(|_| 111).collect();
        // Buffer B: 10ms at ts=12000 — range [12000, 22000). No overlap with A.
        let samples_b: Vec<i32> = (0..480 * 2).map(|_| 222).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 12_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 2);

        // Buffer C: 20ms at ts=9000 — range [9000, 29000).
        // Overlaps both A (9000 < 10000 && 0 < 29000) and B (9000 < 22000 && 12000 < 29000).
        // Both stale buffers should be removed — the server will send fresh
        // data for any gaps. Keeping either would cause duplicate audio.
        let samples_c: Vec<i32> = (0..960 * 2).map(|_| 333).collect();
        queue.push(AudioBuffer {
            timestamp: 9_000,
            samples: Arc::from(samples_c.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].timestamp, 9_000);
        assert_eq!(queue.queue[0].samples[0], 333);
    }

    #[test]
    fn test_push_keeps_adjacent_buffers_inserted_in_reverse_order() {
        // When buffer B is pushed *after* buffer A and they abut at the
        // boundary (B.ts == A.end), the existing dedup check keeps both
        // because `b.timestamp < new_end` is false at the boundary. The
        // symmetric case — pushing the *earlier* buffer second — must also
        // keep both: when we push the earlier buffer A and retain() walks
        // B, we see `B.ts == new_end` (A.end). The dedup has to treat the
        // boundary as *not* overlapping, otherwise we'd evict a buffer that
        // simply abuts — a common pattern when chunks arrive out-of-order.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Two 5ms adjacent chunks: A at [0, 5000), B at [5000, 10000).
        let samples_a: Vec<i32> = (0..240 * 2).map(|_| 111).collect();
        let samples_b: Vec<i32> = (0..240 * 2).map(|_| 222).collect();

        // Push the *later* buffer (B) first …
        queue.push(AudioBuffer {
            timestamp: 5_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format: format.clone(),
        });
        // … then push the *earlier* buffer (A). A.end == B.ts (boundary case).
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format,
        });

        assert_eq!(
            queue.queue.len(),
            2,
            "adjacent buffers pushed in reverse order should both be kept"
        );
        // Should also be sorted: A first, then B.
        assert_eq!(queue.queue[0].timestamp, 0);
        assert_eq!(queue.queue[1].timestamp, 5_000);
    }

    #[test]
    fn test_next_frame_returns_final_frame_when_skip_lands_at_last_frame() {
        // When the cursor skip lands `self.index` at exactly
        // `samples.len() - channels`, there is still one playable frame at
        // the tail of the buffer. The "is this buffer exhausted?" check in
        // the outer match is `self.index + channels > c.samples.len()` —
        // strictly greater — so `index == samples.len() - channels` falls
        // through to the frame-return path. A non-strict comparison here
        // would silently drop the last frame of every buffer whose skip
        // landed on the final-frame boundary.
        //
        // Setup: 48-frame stereo buffer at ts=0, cursor at 980 µs.
        //   skip_us     = 980 - 0                          = 980
        //   skip_frames = 980 * 48_000 / 1_000_000         = 47
        //   self.index  = 47 * 2                           = 94
        //   samples.len = 48 * 2                           = 96
        //   94 + 2 == 96 — keep buffer, return last frame.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Distinctive sample values so we can assert *which* frame was returned.
        let samples: Vec<i32> = (0..48 * 2).map(|_| 111).collect();

        queue.initialized = true;
        queue.cursor_us = 980;
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("last frame should be returned, not discarded");
        // Final stereo frame — indices 94 and 95 of the flat sample array.
        assert_eq!(frame, &[111, 111]);
    }

    #[test]
    fn test_skip_past_entire_buffer_does_not_panic() {
        // When the cursor is far ahead of a buffer, the skip logic can set
        // self.index past the buffer's sample count. next_frame must not
        // panic; it should discard the buffer and return the next one.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // 1ms buffer (48 stereo frames) at ts=49000. Duration = 1000µs,
        // so it ends at 50000 which is NOT < cursor (50000), surviving
        // the stale-drop. But the skip logic sees ts=49000 < cursor=50000
        // and tries to skip 1ms (48 frames) — exactly the buffer length.
        let short_samples: Vec<i32> = (0..48 * 2).map(|_| 111).collect();
        // Buffer that starts at cursor: 10ms at ts=50000
        let ahead_samples: Vec<i32> = (0..480 * 2).map(|_| 222).collect();

        queue.initialized = true;
        queue.cursor_us = 50_000;
        queue.push(AudioBuffer {
            timestamp: 49_000,
            samples: Arc::from(short_samples.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 50_000,
            samples: Arc::from(ahead_samples.into_boxed_slice()),
            format,
        });

        // The skip tries to skip 1ms (48 frames) into a 48-frame buffer —
        // index lands at the end. Must not panic; should discard the short
        // buffer and return from the next one (ts=50000).
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should return a frame from the next buffer, not panic");
        assert_eq!(frame[0], 222, "expected frame from the ahead buffer");
    }
}

#[cfg(test)]
mod callback_tests;

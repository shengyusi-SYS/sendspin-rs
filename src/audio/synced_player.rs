// ABOUTME: Synced audio player with drift correction
// ABOUTME: Uses DAC callback timestamps to drop/insert frames for alignment

use crate::audio::gain::{GainControl, GainRamp};
#[cfg(test)]
use crate::audio::player_contract::RendererCallbackPermit;
use crate::audio::player_contract::{
    EnqueueOutcome, OpenError, OutputBackendError, PlayerScope, PreStartAbortOutcome,
    RendererCapacitySnapshot, RendererFault, RendererHealthSnapshot, RendererOperationOutcome,
    RendererOwner, RendererQueueLimits, ScheduledArmOutcome, ScheduledStartOutcome, StartState,
    TerminalOutcome,
};
use crate::audio::{AudioBuffer, AudioFormat, SyncDiagnosticsReader, SyncDiagnosticsSnapshot};
use crate::error::Error;
use crate::log_sampling::should_log_sample;
use crate::sync::ClockSync;
use cpal::traits::DeviceTrait;
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use cpal::{Sample, I24};
use parking_lot::Mutex;
#[cfg(test)]
use parking_lot::MutexGuard;
use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod allocation;
mod feedback;
mod preparation;
mod reader;
mod source;
mod transport;

pub(crate) mod ingress;
mod runtime;

/// Callback for post-processing audio samples before output.
///
/// Receives `&mut [f32]` (interleaved, after gain is applied).
///
/// The callback is invoked on **every** audio callback, including during
/// pre-start silence when the buffer is all zeros. Long backend buffers may be
/// delivered in frame-aligned chunks from fixed scratch storage; each sample is
/// processed exactly once after gain. Consumers must not assume backend block
/// boundaries or exactly one invocation per backend callback.
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

fn select_output_channels(
    input: u16,
    supported: impl Iterator<Item = u16>,
) -> Result<u16, OpenError> {
    let mut exact = false;
    let mut stereo = false;
    for channels in supported {
        exact |= channels == input;
        stereo |= channels == 2;
    }
    if exact {
        Ok(input)
    } else if input == 1 && stereo {
        Ok(2)
    } else {
        Err(OpenError::UnsupportedFormat)
    }
}

// Render in the input frame domain, then map into the physical output buffer.
fn render_output_channels<T: cpal::Sample>(
    data: &mut [T],
    mono_to_stereo: bool,
    render: impl FnOnce(&mut [T]),
) {
    if !mono_to_stereo {
        render(data);
        return;
    }
    let frames = data.len() / 2;
    render(&mut data[..frames]);
    // Expand backwards so unread mono samples are never overwritten. No
    // allocation or extra renderer/queue access occurs on the audio thread.
    for frame in (0..frames).rev() {
        let sample = data[frame];
        data[2 * frame] = sample;
        data[2 * frame + 1] = sample;
    }
    if data.len() % 2 != 0 {
        *data.last_mut().unwrap() = T::EQUILIBRIUM;
    }
}

fn preflight_device_output_format(device: &Device, format: &AudioFormat) -> Result<u16, OpenError> {
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
        .map_err(|error| OpenError::Backend(OutputBackendError::new(error.to_string())))?;
    select_output_channels(
        u16::from(format.channels),
        supported
            .filter(|range| {
                range.sample_format() == default_config.sample_format()
                    && range.min_sample_rate() <= format.sample_rate
                    && format.sample_rate <= range.max_sample_rate()
            })
            .map(|range| range.channels()),
    )
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
    source_id: u64,
    buffer: AudioBuffer,
    _lifetime: Option<Arc<dyn AudioBufferLifetime>>,
}

impl Clone for QueuedAudioBuffer {
    fn clone(&self) -> Self {
        Self {
            source_id: self.source_id,
            buffer: AudioBuffer {
                timestamp: self.timestamp,
                samples: Arc::clone(&self.samples),
                format: self.format.clone(),
            },
            _lifetime: self._lifetime.clone(),
        }
    }
}

impl Deref for QueuedAudioBuffer {
    type Target = AudioBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

struct PlaybackQueue {
    publication: Option<Arc<source::Publication>>,
    settled_consumed: u64,
    next_source_id: u64,
    retired_through: u64,
    queue: VecDeque<QueuedAudioBuffer>,
    /// Complete frames in `queue`; the current buffer is counted from its index.
    pending_frames: usize,
    /// Conservative maximum end of pending buffers, used only for fast append.
    pending_end_upper_bound: Option<i128>,
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
            publication: None,
            settled_consumed: 0,
            next_source_id: 1,
            retired_through: 0,
            queue: VecDeque::new(),
            pending_frames: 0,
            pending_end_upper_bound: None,
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
        if let Some(publication) = &self.publication {
            publication
                .checkpoint
                .invalidate_before(publication.control.view().epoch());
            self.settled_consumed = publication.consumed.load(Ordering::Acquire);
        }
        self.retired_through = 0;
        self.queue.clear();
        self.pending_frames = 0;
        self.pending_end_upper_bound = None;
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
        let source_id = self.next_source_id;
        self.next_source_id = source_id
            .checked_add(1)
            .expect("scope source identity exhausted");
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

        // Enqueue validation guarantees complete frames in the fixed player format.
        let frames = buffer.samples.len() / usize::from(buffer.format.channels);
        let new_end = buffer_end_zone_us(&buffer);
        let can_append = self.queue.is_empty()
            || (self
                .queue
                .back()
                .is_some_and(|tail| tail.timestamp <= buffer.timestamp)
                && self
                    .pending_end_upper_bound
                    .is_some_and(|end| i128::from(buffer.timestamp) >= end));
        if can_append {
            self.pending_frames += frames;
            self.pending_end_upper_bound = Some(
                self.pending_end_upper_bound
                    .map_or(new_end, |end| end.max(new_end)),
            );
            self.queue.push_back(QueuedAudioBuffer {
                source_id,
                buffer,
                _lifetime: lifetime,
            });
            return;
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
        let mut retained_frames = 0;
        let mut retained_end: Option<i128> = None;
        self.queue.retain(|b| {
            let existing_end = buffer_end_zone_us(b);
            let overlap_us =
                new_end.min(existing_end) - i128::from(buffer.timestamp.max(b.timestamp));
            let keep = overlap_us < i128::from(frame_us);
            if keep {
                retained_frames += b.samples.len() / usize::from(b.format.channels);
                retained_end = Some(retained_end.map_or(existing_end, |end| end.max(existing_end)));
            }
            keep
        });
        self.pending_frames = retained_frames + frames;
        self.pending_end_upper_bound = Some(retained_end.map_or(new_end, |end| end.max(new_end)));

        let pos = self
            .queue
            .iter()
            .position(|b| b.timestamp > buffer.timestamp);
        if let Some(pos) = pos {
            self.queue.insert(
                pos,
                QueuedAudioBuffer {
                    source_id,
                    buffer,
                    _lifetime: lifetime,
                },
            );
        } else {
            self.queue.push_back(QueuedAudioBuffer {
                source_id,
                buffer,
                _lifetime: lifetime,
            });
        }
    }

    fn pop_pending(&mut self) -> Option<QueuedAudioBuffer> {
        let buffer = self.queue.pop_front()?;
        self.pending_frames -= buffer.samples.len() / usize::from(buffer.format.channels);
        if self.queue.is_empty() {
            self.pending_end_upper_bound = None;
        }
        Some(buffer)
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
                        self.retired_through =
                            self.pop_pending().expect("checked pending").source_id;
                        continue;
                    }
                    break;
                }
            }

            // Pop buffers until we find one with remaining samples past the
            // cursor, or the queue is empty.
            loop {
                if let Some(previous) = self.current.take() {
                    self.retired_through = previous.source_id;
                }
                self.current = self.pop_pending();
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
                        self.retired_through = c.source_id;
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
            self.retired_through = self.current.as_ref().expect("checked current").source_id;
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
        current_frames + self.pending_frames
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

#[cfg(test)]
fn canonical_presentation_zone_us(
    device_presentation_zone_us: Option<i64>,
    static_delay_us: u64,
) -> Option<i64> {
    let delay_us = i64::try_from(static_delay_us).ok()?;
    device_presentation_zone_us?.checked_add(delay_us)
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
#[cfg(test)]
enum CallbackQueuePhase {
    TimingSnapshot,
    StartupReanchor,
    CorrectionReanchor,
    Render,
}

#[cfg(test)]
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

#[cfg(test)]
enum StartupReanchorOutcome {
    Applied(i64),
    NoPlayable,
    Stale,
}

#[cfg(test)]
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

#[cfg(test)]
fn record_callback_access_failure(
    renderer: &RendererOwner,
    silent_frames: usize,
    phase: CallbackQueuePhase,
    renderer_unavailable: bool,
    observation: &mut SyncDiagnosticsSnapshot,
) {
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
}

#[cfg(test)]
fn try_callback_timing_queue<T>(
    renderer: &RendererOwner,
    queue: &Mutex<PlaybackQueue>,
    silent_frames: usize,
    observation: &mut SyncDiagnosticsSnapshot,
    operation: impl FnOnce(&PlaybackQueue) -> T,
) -> Option<T> {
    let Some(queue) = queue.try_lock() else {
        record_callback_access_failure(
            renderer,
            silent_frames,
            CallbackQueuePhase::TimingSnapshot,
            false,
            observation,
        );
        return None;
    };
    Some(operation(&queue))
}

#[cfg(test)]
fn try_callback_queue_guards<'a>(
    renderer: &'a RendererOwner,
    scope: PlayerScope,
    queue: &'a Mutex<PlaybackQueue>,
    silent_frames: usize,
    phase: CallbackQueuePhase,
    observation: &mut SyncDiagnosticsSnapshot,
) -> Option<(RendererCallbackPermit<'a>, MutexGuard<'a, PlaybackQueue>)> {
    let Some(permit) = renderer.try_callback_permit(scope) else {
        record_callback_access_failure(renderer, silent_frames, phase, true, observation);
        return None;
    };
    let Some(queue) = queue.try_lock() else {
        drop(permit);
        record_callback_access_failure(renderer, silent_frames, phase, false, observation);
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
        let output_channels = preflight_device_output_format(&device, &format)?;

        let stream_config = StreamConfig {
            channels: output_channels,
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
            let _ = renderer.teardown(scope);
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
        let outcome = if let Some(publication) = queue.publication.clone() {
            let mut pending = (buffer, lifetime);
            loop {
                match queue.try_enqueue_prepared(
                    &publication,
                    &self.renderer,
                    self.scope,
                    pending,
                    frames,
                ) {
                    Ok(outcome) => break outcome,
                    Err(input) => {
                        pending = input;
                        std::thread::yield_now();
                    }
                }
            }
        } else {
            self.renderer.enqueue_with_actual(self.scope, frames, || {
                queue.push_with_lifetime(buffer, lifetime);
                queue.enqueue_count += 1;
                (queue.queued_frames(channels), queue.buffer_count())
            })
        };
        if !matches!(outcome, EnqueueOutcome::Accepted { .. }) {
            return outcome;
        }

        // Snapshot log fields under the lock but log after dropping it: the
        // audio callback contends on this lock, and logging can block on I/O.
        // Capture log fields only for sampled, trace-enabled enqueues.
        let trace_fields = {
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
        drop(queue);

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

    /// Read committed consumption without waiting for timestamp telemetry.
    pub fn consumed_frames(&self) -> Result<u64, RendererOperationOutcome> {
        self.renderer.consumed_frames(self.scope)
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
    /// by the same delay to keep alignment correct. Publishes a delay update and
    /// requests a reanchor; returning does not mean the audible transition has
    /// completed. Until preparation can apply the new target, playback retains
    /// the previous effective delay.
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
        stream_config.channels = config.channels;
        let mono_to_stereo = format.channels == 1 && config.channels == 2;
        stream_config.sample_rate = format.sample_rate;

        macro_rules! output_stream {
            ($sample:ty) => {{
                let renderer_for_error = renderer.clone();
                let renderer_for_preparation = renderer.clone();
                let (mut callback, worker) = make_output_callback::<$sample>(
                    queue,
                    clock_sync,
                    format,
                    cb_config,
                    renderer,
                    scope,
                    diagnostics,
                )?;
                let resource = worker.spawn()?;
                renderer_for_preparation
                    .attach_preparation_resource(scope, Box::new(resource))
                    .map_err(|_| Error::Output("renderer rejected preparation ownership".into()))?;
                let result = device
                    .build_output_stream(
                        stream_config,
                        move |data: &mut [$sample], info: &cpal::OutputCallbackInfo| {
                            render_output_channels(data, mono_to_stereo, |input| {
                                callback(
                                    input,
                                    info.timestamp(),
                                    info.timestamp_source(),
                                    info.timestamp_diagnostics(),
                                    Instant::now(),
                                );
                            });
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
                    .map_err(|e| Error::Output(e.to_string()));
                if result.is_err() {
                    let _ = renderer_for_preparation.teardown(scope);
                }
                result
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
) -> Result<
    (
        impl FnMut(
                &mut [T],
                cpal::OutputStreamTimestamp,
                cpal::OutputTimestampSource,
                Option<cpal::OutputTimestampDiagnostics>,
                Instant,
            ) + Send,
        runtime::Worker,
    ),
    Error,
> {
    let (mut device, worker) = runtime::build(
        queue,
        clock_sync,
        &format,
        &cb_config,
        renderer_for_data.clone(),
        scope,
        diagnostics,
    )?;
    Ok((
        move |data: &mut [T],
              timestamp: cpal::OutputStreamTimestamp,
              source: cpal::OutputTimestampSource,
              evidence: Option<cpal::OutputTimestampDiagnostics>,
              captured_at: Instant| {
            device.render(
                data,
                timestamp,
                source,
                evidence,
                captured_at,
                &mut cb_config,
                &renderer_for_data,
                scope,
            );
        },
        worker,
    ))
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
        try_callback_timing_queue, validate_enqueue_buffer, validate_output_format,
        windows_default_buffer_frames, CallbackQueuePhase, DeviceDelayError, DeviceDelayMs,
        PlaybackQueue, SyncDiagnosticsSnapshot, MAX_STATIC_DELAY_MS,
    };
    use crate::audio::{
        AudioBuffer, AudioFormat, Codec, EnqueueOutcome, PlayerScope, PreStartAbortOutcome,
        RendererFault, RendererOperationOutcome, RendererOwner, RendererQueueLimits,
        ScheduledArmOutcome, ScheduledStartOutcome, TerminalOutcome,
    };
    use cpal::Sample;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{mpsc, Arc};
    use std::thread;

    #[test]
    fn mono_output_prefers_native_and_falls_back_only_to_stereo() {
        assert_eq!(
            super::select_output_channels(1, [2, 1].into_iter()).unwrap(),
            1
        );
        assert_eq!(
            super::select_output_channels(1, [2].into_iter()).unwrap(),
            2
        );
        assert!(super::select_output_channels(1, [6].into_iter()).is_err());
        assert!(super::select_output_channels(2, [1].into_iter()).is_err());
        assert_eq!(
            super::select_output_channels(2, [2].into_iter()).unwrap(),
            2
        );
    }

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
    fn timing_snapshot_reads_queue_during_renderer_contention() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        let permit = owner.try_callback_permit(scope).unwrap();
        let mut observation = SyncDiagnosticsSnapshot::default();
        let snapshot = try_callback_timing_queue(&owner, &queue, 1, &mut observation, |queue| {
            (queue.cursor_us, queue.generation)
        });
        drop(permit);
        assert_eq!(snapshot, Some((0, 0)));
        assert_eq!(observation.access_silence_frames, 0);
    }

    #[test]
    fn scheduled_start_timing_snapshot_changes_are_advisory() {
        let (owner, idle_scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, idle_scope, &queue, mono_buffer(0, 4)),
            EnqueueOutcome::Accepted { .. }
        ));
        let idle_snapshot = try_callback_timing_queue(
            &owner,
            &queue,
            1,
            &mut SyncDiagnosticsSnapshot::default(),
            |queue| (queue.cursor_us, queue.generation),
        )
        .unwrap();
        assert_eq!(idle_snapshot, (0, 0));
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
        let armed_snapshot = try_callback_timing_queue(
            &owner,
            &queue,
            1,
            &mut SyncDiagnosticsSnapshot::default(),
            |queue| (queue.cursor_us, queue.generation),
        )
        .unwrap();
        assert_eq!(armed_snapshot, (0, 1));
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
    fn realtime_handoff_admission_replacement_releases_capacity_before_callback() {
        // Characterize the real queue/owner admission boundary before any callback exists.
        // Counting both handoff entries until a later callback processes the replacement
        // would incorrectly reject the third enqueue, which the product treats as an error.
        let (owner, scope, queue) = renderer_harness();
        for timestamp in [0, 0] {
            assert_eq!(
                enqueue_harness(&owner, scope, &queue, mono_buffer(timestamp, 16)),
                EnqueueOutcome::Accepted {
                    queued_frames: 16,
                    queued_buffers: 1,
                }
            );
        }
        assert_eq!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(1_000_000, 16)),
            EnqueueOutcome::Accepted {
                queued_frames: 32,
                queued_buffers: 2,
            }
        );
        assert_eq!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(2_000_000, 1)),
            EnqueueOutcome::Full {
                queued_frames: 32,
                queued_buffers: 2,
            }
        );
    }

    #[test]
    fn realtime_handoff_early_source_read_changes_replacement_and_capacity() {
        // Compare actual device consumption with the proposed reuse of the same
        // destructive read for preparation. No new handoff algorithm is modelled.
        // Preparation must not inherit these irreversible queue side effects.
        for (read_before_replacement, expected_pcm, expected_queued) in [
            (false, vec![20, 21, 22, 23], 4),
            (true, vec![1, 2, 3, 4, 22, 23], 7),
        ] {
            let format = AudioFormat {
                sample_rate: 1_000,
                ..test_format_mono()
            };
            let mut queue = PlaybackQueue::new();
            queue.push(AudioBuffer {
                timestamp: 0,
                samples: Arc::from([1, 2, 3, 4]),
                format: format.clone(),
            });
            let mut output = Vec::new();
            if read_before_replacement {
                output.extend(queue.next_frame(1, 1_000).unwrap());
                assert_eq!(queue.queued_frames(1), 3);
            }
            queue.push(AudioBuffer {
                timestamp: 2_000,
                samples: Arc::from([20, 21, 22, 23]),
                format,
            });
            assert_eq!(queue.queued_frames(1), expected_queued);
            while let Some(frame) = queue.next_frame(1, 1_000) {
                output.extend(frame);
            }
            assert_eq!(output, expected_pcm);
            assert_eq!(queue.queued_frames(1), 0);
        }
    }

    #[test]
    fn realtime_handoff_capacity_remaining_frames_excludes_consumed_current_prefix() {
        let (owner, scope, queue) = renderer_harness();
        assert!(matches!(
            enqueue_harness(&owner, scope, &queue, mono_buffer(0, 16)),
            EnqueueOutcome::Accepted { .. }
        ));
        {
            let mut permit = owner.try_callback_permit(scope).unwrap();
            let mut actual = queue.lock();
            for _ in 0..15 {
                assert!(actual.consume_next_frame(1, 48_000, None));
            }
            permit.record_actual_progress(15, None, actual.queued_frames(1), actual.buffer_count());
        }
        for (timestamp, frames) in [(1_000_000, 16), (2_000_000, 15)] {
            assert!(matches!(
                enqueue_harness(&owner, scope, &queue, mono_buffer(timestamp, frames)),
                EnqueueOutcome::Accepted { .. }
            ));
        }
        assert_eq!(owner.capacity(scope).unwrap().current_frames(), 32);
        let actual = queue.lock();
        // The current Arc still owns its consumed prefix. This is a payload
        // length observation, not an allocator/RSS or caller-owned memory metric.
        let held_samples = actual.current.as_ref().unwrap().samples.len()
            + actual.queue.iter().map(|b| b.samples.len()).sum::<usize>();
        assert_eq!(held_samples, 47);
        assert_eq!(actual.queued_frames(1), 32);
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
    fn queue_capacity_tracks_pending_current_and_clear() {
        let buffer = |timestamp, samples: &[i32]| AudioBuffer {
            timestamp,
            samples: Arc::from(samples),
            format: AudioFormat {
                sample_rate: 1_000,
                ..test_format_mono()
            },
        };
        let mut queue = PlaybackQueue::new();
        queue.push(buffer(0, &[1, 2, 3, 4]));
        queue.push(buffer(4_000, &[5, 6, 7, 8]));
        assert_eq!(queue.queued_frames(1), 8);
        for sample in [1, 2, 3] {
            assert_eq!(queue.next_frame(1, 1_000), Some(vec![sample]));
        }
        assert_eq!(queue.queued_frames(1), 5);
        for sample in [4, 5] {
            assert_eq!(queue.next_frame(1, 1_000), Some(vec![sample]));
        }
        assert_eq!(queue.queued_frames(1), 3);
        queue.clear();
        assert_eq!(queue.queued_frames(1), 0);
        queue.push(buffer(10_000, &[9, 10]));
        assert_eq!(queue.queued_frames(1), 2);
        for sample in [9, 10] {
            assert_eq!(queue.next_frame(1, 1_000), Some(vec![sample]));
        }
        assert_eq!(queue.queued_frames(1), 0);
    }

    #[test]
    fn queue_capacity_and_pcm_follow_replacement_and_cursor() {
        // Expected PCM comes from the input ranges, independently of queue accounting.
        let cases: &[(&str, &[(i64, &[i32])], i64, usize, &[i32])] = &[
            (
                "out of order",
                &[(4_000, &[5, 6, 7, 8]), (0, &[1, 2, 3, 4])],
                0,
                8,
                &[1, 2, 3, 4, 5, 6, 7, 8],
            ),
            (
                "overlap replacement",
                &[(0, &[1, 2, 3, 4]), (2_000, &[20, 21, 22, 23])],
                0,
                4,
                &[20, 21, 22, 23],
            ),
            (
                "stale pending",
                &[(0, &[1, 2]), (4_000, &[5, 6, 7])],
                3_000,
                5,
                &[5, 6, 7],
            ),
            (
                "skip into current",
                &[(0, &[1, 2, 3, 4]), (4_000, &[5, 6])],
                2_000,
                6,
                &[3, 4, 5, 6],
            ),
        ];
        for &(name, buffers, cursor_us, before_frames, expected) in cases {
            let mut queue = PlaybackQueue::new();
            for &(timestamp, samples) in buffers {
                queue.push(AudioBuffer {
                    timestamp,
                    samples: Arc::from(samples),
                    format: AudioFormat {
                        sample_rate: 1_000,
                        ..test_format_mono()
                    },
                });
            }
            queue.cursor_us = cursor_us;
            assert_eq!(queue.queued_frames(1), before_frames, "{name}");
            for (index, &sample) in expected.iter().enumerate() {
                assert_eq!(queue.next_frame(1, 1_000), Some(vec![sample]), "{name}");
                assert_eq!(queue.queued_frames(1), expected.len() - index - 1, "{name}");
            }
            assert_eq!(queue.next_frame(1, 1_000), None, "{name}");
            assert_eq!(queue.queued_frames(1), 0, "{name}");
        }
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

// Temporary protocol experiment; not wired into the production player.
#[cfg(test)]
#[path = "synced_player/handoff_probe.rs"]
mod handoff_probe;

//! Read-only sync observations. Audio callbacks only publish with try_lock;
//! hosts sample and format them on a non-audio thread.

use crate::sync::{ClockHealthSnapshot, ClockSync};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

/// Last completed callback observation. Counters are cumulative for the output
/// stream, including across queue clears. Errors are absent when that callback
/// did not calculate a sync error. This is diagnostic data, never control input.
#[derive(Clone, Copy, Debug, Default)]
pub struct SyncDiagnosticsSnapshot {
    /// Capture time; None before the first published callback.
    pub captured_at: Option<Instant>,
    /// All callbacks, including silent and contended callbacks.
    pub callbacks: u64,
    /// All frames requested by callbacks, including startup and silence.
    pub requested_frames: u64,
    /// Callbacks which emitted a whole buffer of silence instead of rendering.
    /// Includes intentional pre-start silence; excludes partial queue starvation.
    pub silent_callbacks: u64,
    /// Frames in those whole-buffer silent callbacks.
    pub silent_frames: u64,
    /// Renderer permit unavailable: contention or a closed/stale scope.
    pub renderer_access_misses: u64,
    /// Playback queue try_lock failures after acquiring a renderer permit.
    pub queue_lock_misses: u64,
    /// Silent frames caused by either queue-access guard failing.
    pub access_silence_frames: u64,
    /// Sequence of the most recent queue-access failure, retained across recovery.
    pub last_access_miss_callback: u64,
    /// Static phase label of that failure, absent before the first failure.
    pub last_access_miss_phase: Option<&'static str>,
    /// Backend underrun count before the latest callback, None when unavailable.
    /// Independent from application silence and queue-starvation counters.
    pub output_xrun_count: Option<u32>,
    /// Backend output buffer size observed before the latest callback.
    pub output_buffer_size_frames: Option<u32>,
    /// Playback queue generation observed by this callback.
    pub generation: u64,
    /// Configured output sample rate.
    pub sample_rate: u32,
    /// Frames requested in this callback.
    pub callback_frames: usize,
    /// Largest callback gap since the output stream was created.
    pub max_callback_gap_us: u64,
    /// Backend's presentation time minus callback time, before filtering.
    pub playback_delay_us: u64,
    /// Raw error in this callback, absent when correction was not evaluated.
    pub raw_error_us: Option<i64>,
    /// Filtered error for that same callback.
    pub filtered_error_us: Option<i64>,
    /// Cumulative timestamp query/rejection counts, including silent callbacks.
    pub fallback_unavailable: u64,
    /// Queries unsupported by the host.
    pub fallback_unsupported: u64,
    /// Invalid or already elapsed presentation projections.
    pub fallback_invalid: u64,
    /// Projections which did not advance past the last accepted presentation.
    pub fallback_non_monotonic: u64,
    /// Queries from an incompatible clock domain.
    pub fallback_clock_domain_mismatch: u64,
    /// Callback sequence of the latest detailed fallback, retained across valid callbacks.
    pub last_timestamp_fallback_callback: u64,
    /// Latest fallback evidence, retained so a slow logger does not miss short events.
    pub last_timestamp_fallback: Option<cpal::OutputTimestampDiagnostics>,
    /// Applied repeat-frame cadence for this callback, zero when inactive.
    pub insert_every: u32,
    /// Applied drop-frame cadence for this callback, zero when inactive.
    pub drop_every: u32,
    /// Frames actually repeated by the corrector.
    pub inserted_frames: u64,
    /// Frames actually discarded by the corrector.
    pub dropped_frames: u64,
    /// Startup or explicit delay-change reanchors.
    pub startup_reanchors: u64,
    /// Automatic large-error reanchors, separately from startup.
    pub correction_reanchors: u64,
    /// Filtered error which caused the most recent automatic reanchor.
    pub last_reanchor_error_us: Option<i64>,
    /// Callbacks unable to acquire the clock model.
    pub sync_lock_misses: u64,
    /// Frames filled with silence on queue starvation.
    pub underrun_frames: u64,
}

/// Cloneable observer. It does not own a stream or prolong its lifecycle.
#[derive(Clone)]
pub struct SyncDiagnosticsReader {
    callback: Arc<Mutex<SyncDiagnosticsSnapshot>>,
    clock: Arc<Mutex<ClockSync>>,
}

/// Clock model sampled off the audio thread, independently of the callback.
#[derive(Clone, Copy, Debug)]
pub struct SyncClockDiagnostics {
    /// Freshness and probe counters at observation time.
    pub health: ClockHealthSnapshot,
    /// Filter estimate at its last accepted sample (not an acoustic offset).
    pub offset_us: f64,
    /// Estimated rate difference in parts per million.
    pub estimated_drift_ppm: f64,
    /// Rate used in conversions, zero when SNR gated, None when unavailable.
    pub applied_drift_ppm: Option<f64>,
}

impl SyncDiagnosticsReader {
    pub(crate) fn new(clock: Arc<Mutex<ClockSync>>) -> Self {
        Self {
            callback: Arc::new(Mutex::new(SyncDiagnosticsSnapshot::default())),
            clock,
        }
    }

    /// Read a coherent last callback snapshot. Call only off the audio thread.
    pub fn snapshot(&self) -> SyncDiagnosticsSnapshot {
        *self.callback.lock()
    }

    /// Read the current model independently; it need not match callback time.
    pub fn clock_snapshot(&self) -> SyncClockDiagnostics {
        let clock = self.clock.lock();
        let (offset_us, estimated_drift_ppm, applied_drift_ppm) = clock.diagnostic_estimate();
        let health = clock.health();
        SyncClockDiagnostics {
            health,
            offset_us,
            estimated_drift_ppm,
            applied_drift_ppm: health.synchronized.then_some(applied_drift_ppm).flatten(),
        }
    }

    pub(crate) fn publish(&self, observation: SyncDiagnosticsSnapshot) {
        // A reader must never delay audio. The next callback retries with all
        // cumulative counters, so a missed publication does not lose events.
        if let Some(mut latest) = self.callback.try_lock() {
            *latest = observation;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::DefaultClock;

    #[test]
    fn sync_diagnostics_contended_reader_preserves_audio_and_cumulative_events() {
        let reader = SyncDiagnosticsReader::new(Arc::new(Mutex::new(ClockSync::new_same_clock(
            Arc::new(DefaultClock::new()),
        ))));
        let snapshot = SyncDiagnosticsSnapshot {
            callbacks: 7,
            correction_reanchors: 2,
            access_silence_frames: 441,
            last_access_miss_callback: 7,
            last_access_miss_phase: Some("render"),
            last_reanchor_error_us: Some(510_000),
            ..Default::default()
        };
        let held = reader.callback.lock();
        reader.publish(snapshot); // Would deadlock if publication acquired a blocking lock.
        assert_eq!(held.callbacks, 0);
        drop(held);
        reader.publish(SyncDiagnosticsSnapshot {
            callbacks: 8,
            ..snapshot
        });
        assert_eq!(reader.snapshot().correction_reanchors, 2);
        assert_eq!(reader.snapshot().access_silence_frames, 441);
        assert_eq!(reader.snapshot().last_access_miss_callback, 7);
        assert_eq!(reader.snapshot().last_access_miss_phase, Some("render"));
        assert_eq!(reader.snapshot().last_reanchor_error_us, Some(510_000));
        reader.publish(SyncDiagnosticsSnapshot {
            callbacks: 9,
            raw_error_us: None,
            ..snapshot
        });
        assert_eq!(reader.snapshot().raw_error_us, None);
        let model = reader.clock_snapshot();
        assert_eq!(model.applied_drift_ppm, Some(0.0));
        assert!(model.health.synchronized);
        let cold = SyncDiagnosticsReader::new(Arc::new(Mutex::new(ClockSync::new(Arc::new(
            DefaultClock::new(),
        )))));
        assert_eq!(cold.clock_snapshot().applied_drift_ppm, None);
    }
}

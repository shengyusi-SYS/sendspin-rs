//! Hermetic driver for the production queue, admission, worker and f32 callback.
//! No device lease, platform stream, network or worker thread is created. The caller
//! owns the driver and advances its synthetic clock. This is not a product backend.
use super::*;
use crate::sync::Clock;
use cpal::{OutputStreamTimestamp, OutputTimestampSource, StreamInstant};

struct TestClock(Instant);
impl Clock for TestClock {
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

type Callback = Box<
    dyn FnMut(
            &mut [f32],
            OutputStreamTimestamp,
            OutputTimestampSource,
            Option<cpal::OutputTimestampDiagnostics>,
            Instant,
        ) + Send,
>;

/// The player is shared independently of the driver: enqueue and callback do not
/// acquire a harness-wide lock. Drop the driver and player to release all resources.
pub struct HeadlessPlayerHarness {
    player: Arc<SyncedPlayer>,
    callback: Callback,
    worker: runtime::Worker,
    origin: Instant,
}

impl HeadlessPlayerHarness {
    pub fn new(format: AudioFormat, limits: RendererQueueLimits) -> Result<Self, OpenError> {
        validate_output_format(&format)?;
        let origin = Instant::now();
        let clock = Arc::new(Mutex::new(ClockSync::new_same_clock(Arc::new(TestClock(
            origin,
        )))));
        let renderer = RendererOwner::new(limits);
        let scope = renderer.mint_scope()?;
        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        let gain = GainControl::new(100, false);
        let static_delay_us = Arc::new(AtomicU64::new(0));
        let diagnostics = SyncDiagnosticsReader::new(clock.clone());
        let (callback, worker) = make_output_callback::<f32>(
            queue.clone(),
            clock,
            format.clone(),
            CallbackConfig {
                max_callback_frames: None,
                gain_control: gain.clone(),
                process_callback: None,
                static_delay_us: static_delay_us.clone(),
            },
            renderer.clone(),
            scope,
            diagnostics.clone(),
        )
        .map_err(|error| OpenError::Backend(OutputBackendError::new(error.to_string())))?;
        let player = Arc::new(SyncedPlayer {
            diagnostics,
            format,
            queue,
            renderer,
            scope,
            last_error: Arc::new(Mutex::new(None)),
            gain,
            static_delay_us,
        });
        Ok(Self {
            player,
            callback: Box::new(callback),
            worker,
            origin,
        })
    }

    pub fn player(&self) -> Arc<SyncedPlayer> {
        self.player.clone()
    }

    /// `now_us` is the callback time; `latency_us` supplies trusted device
    /// presentation timing, matching the existing production callback fixture.
    pub fn render_at(&mut self, now_us: u64, latency_us: u64, frames: usize) -> Vec<f32> {
        let captured_at = self.origin + Duration::from_micros(now_us);
        self.worker.step(captured_at);
        let mut output = vec![0.0; frames * usize::from(self.player.format.channels)];
        let callback = StreamInstant::from_nanos(now_us * 1_000);
        (self.callback)(
            &mut output,
            OutputStreamTimestamp {
                callback,
                playback: callback + Duration::from_micros(latency_us),
            },
            OutputTimestampSource::DevicePresentation,
            None,
            captured_at,
        );
        self.worker.step(captured_at);
        output
    }
}

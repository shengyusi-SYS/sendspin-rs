//! Production preparation executor and device-owned rendering state.
//! ClockSync, source reconciliation, filtering and diagnostics stay on the worker.

use super::allocation::Budget;
use super::feedback::{FeedbackWorker, Measurement, Observation, PlanMailbox, PlanReader};
use super::reader::Reader;
use super::source::Publication;
use super::transport::{self, DeviceSide, PrepareSide};
use super::*;
use crate::sync::Clock;
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::atomic::{fence, AtomicBool, AtomicI64, AtomicU32, AtomicU8};
use std::thread::{self, JoinHandle};

pub(super) const WINDOWS: usize = 4;
pub(super) const OBSERVATIONS: usize = 64;

// The payload is all atomic; the outer revision validates one bounded read.
// Store full Instant magnitudes instead of reducing i64 scheduled boundaries.
#[derive(Default)]
struct AtomicInstant {
    kind: AtomicU8,
    seconds: AtomicU64,
    nanos: AtomicU32,
}
impl AtomicInstant {
    fn store(&self, origin: Instant, value: Option<Instant>) {
        let (kind, duration) = match value {
            None => (0, Duration::ZERO),
            Some(value) if value >= origin => (1, value.duration_since(origin)),
            Some(value) => (2, origin.duration_since(value)),
        };
        self.seconds.store(duration.as_secs(), Ordering::Relaxed);
        self.nanos.store(duration.subsec_nanos(), Ordering::Relaxed);
        self.kind.store(kind, Ordering::Relaxed);
    }
    fn load(&self, origin: Instant) -> Option<Instant> {
        let duration = Duration::new(
            self.seconds.load(Ordering::Relaxed),
            self.nanos.load(Ordering::Relaxed),
        );
        match self.kind.load(Ordering::Relaxed) {
            1 => origin.checked_add(duration),
            2 => origin.checked_sub(duration),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Timing {
    epoch: u64,
    generation: u64,
    delay_us: u64,
    expected: Option<Instant>,
    boundary: i64,
    boundary_deadline: Option<Instant>,
    valid_until: Option<Instant>,
    valid: bool,
    pending: bool,
}
pub(super) struct TimingMailbox {
    origin: Instant,
    revision: AtomicU64,
    epoch: AtomicU64,
    generation: AtomicU64,
    delay_us: AtomicU64,
    expected: AtomicInstant,
    boundary: AtomicI64,
    boundary_deadline: AtomicInstant,
    valid_until: AtomicInstant,
    flags: AtomicU8,
}
impl TimingMailbox {
    fn new() -> Self {
        Self {
            origin: Instant::now(),
            revision: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            delay_us: AtomicU64::new(0),
            expected: AtomicInstant::default(),
            boundary: AtomicI64::new(0),
            boundary_deadline: AtomicInstant::default(),
            valid_until: AtomicInstant::default(),
            flags: AtomicU8::new(0),
        }
    }
    fn publish(&self, value: Timing) {
        let revision = self.revision.load(Ordering::Relaxed);
        let next = revision
            .checked_add(2)
            .expect("scope timing publication version");
        self.revision.store(revision + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        self.epoch.store(value.epoch, Ordering::Relaxed);
        self.generation.store(value.generation, Ordering::Relaxed);
        self.delay_us.store(value.delay_us, Ordering::Relaxed);
        self.expected.store(self.origin, value.expected);
        self.boundary.store(value.boundary, Ordering::Relaxed);
        self.boundary_deadline
            .store(self.origin, value.boundary_deadline);
        self.valid_until.store(self.origin, value.valid_until);
        self.flags.store(
            u8::from(value.valid) | (u8::from(value.pending) << 1),
            Ordering::Relaxed,
        );
        self.revision.store(next, Ordering::Release);
    }
    fn read(&self) -> Option<Timing> {
        let before = self.revision.load(Ordering::Acquire);
        if before == 0 || before % 2 != 0 {
            return None;
        }
        let flags = self.flags.load(Ordering::Relaxed);
        let value = Timing {
            epoch: self.epoch.load(Ordering::Relaxed),
            generation: self.generation.load(Ordering::Relaxed),
            delay_us: self.delay_us.load(Ordering::Relaxed),
            expected: self.expected.load(self.origin),
            boundary: self.boundary.load(Ordering::Relaxed),
            boundary_deadline: self.boundary_deadline.load(self.origin),
            valid_until: self.valid_until.load(self.origin),
            valid: flags & 1 != 0,
            pending: flags & 2 != 0,
        };
        fence(Ordering::Acquire);
        (self.revision.load(Ordering::Relaxed) == before).then_some(value)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Event {
    diagnostics: SyncDiagnosticsSnapshot,
    feedback: Option<Observation>,
    playback_delta: Duration,
    latency_floor: Option<Duration>,
    started: bool,
}

pub(super) struct Device {
    publication: Arc<Publication>,
    pipe: DeviceSide,
    reader: Reader,
    clock: Arc<dyn Clock>,
    timing: Arc<TimingMailbox>,
    cached_timing: Option<Timing>,
    plans: Arc<PlanMailbox>,
    plan_reader: PlanReader,
    events: Producer<Event>,
    pcm: Vec<i32>,
    float: Vec<f32>,
    started: bool,
    minimum_epoch: u64,
    last_generation: Option<u64>,
    min_latency: Option<Duration>,
    last_latency: Option<Duration>,
    last_timestamp: Option<cpal::StreamInstant>,
    observation: SyncDiagnosticsSnapshot,
    gain: GainRamp,
}

pub(super) struct Worker {
    queue: Arc<Mutex<PlaybackQueue>>,
    sync: Arc<Mutex<ClockSync>>,
    owner: RendererOwner,
    scope: PlayerScope,
    publication: Arc<Publication>,
    private: PlaybackQueue,
    producer: PrepareSide,
    sample_rate: u32,
    channels: usize,
    delay: Arc<AtomicU64>,
    applied_delay_us: u64,
    timing: Arc<TimingMailbox>,
    plans: Arc<PlanMailbox>,
    feedback: FeedbackWorker,
    events: Consumer<Event>,
    diagnostics: SyncDiagnosticsReader,
    latest: Option<Event>,
    snapshot: Option<(u64, u64)>,
    prepared_base: u64,
    pending_window: bool,
    timeline: u64,
    generation: u64,
    startup_reanchors: u64,
    correction_reanchors: u64,
    last_reanchor_error: Option<i64>,
    pending_reanchor: Option<Observation>,
    // Requested and actual checked player-owned allocation totals, excluding
    // variable codec headers and caller/backend/thread-owned opaque resources.
    allocation_bytes: [usize; 2],
}

pub(super) fn build(
    queue: Arc<Mutex<PlaybackQueue>>,
    sync: Arc<Mutex<ClockSync>>,
    format: &AudioFormat,
    config: &CallbackConfig,
    owner: RendererOwner,
    scope: PlayerScope,
    diagnostics: SyncDiagnosticsReader,
) -> Result<(Device, Worker), Error> {
    let limits = owner
        .health(scope)
        .map_err(|_| Error::Output("stale renderer scope".into()))?
        .limits();
    let channels = usize::from(format.channels);
    let frames = (format.sample_rate as usize / 50)
        .max(1)
        .min(limits.max_chunk_frames());
    let samples = frames
        .checked_mul(channels)
        .ok_or_else(|| Error::Output("PCM workspace size overflow".into()))?;
    let preflight = allocation_budget(
        limits,
        channels,
        frames,
        limits.hard_buffers(),
        limits.hard_buffers(),
        None,
        OBSERVATIONS,
        channels,
        [samples; 2],
    )
    .ok_or_else(|| Error::Output("player allocation size overflow".into()))?;
    let (producer, pipe) = transport::pipe(WINDOWS, frames, channels)
        .ok_or_else(|| Error::Output("invalid PCM transport dimensions".into()))?;
    let publication = Publication::new(&owner, scope, channels, limits)
        .map_err(|_| Error::Output("stale renderer publication".into()))?;
    {
        let mut queue = queue.lock();
        let reserve = limits.hard_buffers().saturating_sub(queue.queue.len());
        queue.queue.reserve(reserve);
        queue.attach_publication(Arc::clone(&publication));
    }
    let mut private = PlaybackQueue::new();
    private.queue.reserve(limits.hard_buffers());
    let timing = Arc::new(TimingMailbox::new());
    let plans = Arc::new(PlanMailbox::default());
    let (events_tx, events_rx) = RingBuffer::new(OBSERVATIONS);
    let clock = sync.lock().clock();
    let device = Device {
        publication: Arc::clone(&publication),
        pipe,
        reader: Reader::new(channels),
        clock,
        timing: Arc::clone(&timing),
        cached_timing: None,
        plans: Arc::clone(&plans),
        plan_reader: PlanReader::default(),
        events: events_tx,
        pcm: vec![0; samples],
        float: vec![0.0; samples],
        started: false,
        minimum_epoch: 0,
        last_generation: None,
        min_latency: None,
        last_latency: None,
        last_timestamp: None,
        observation: SyncDiagnosticsSnapshot {
            sample_rate: format.sample_rate,
            ..Default::default()
        },
        gain: GainRamp::new(format.sample_rate, config.gain_control.gain()),
    };
    let mut worker = Worker {
        queue,
        sync,
        owner,
        scope,
        publication,
        private,
        producer,
        sample_rate: format.sample_rate,
        channels,
        delay: Arc::clone(&config.static_delay_us),
        applied_delay_us: config.static_delay_us.load(Ordering::Relaxed),
        timing,
        plans,
        feedback: FeedbackWorker::new(format.sample_rate),
        events: events_rx,
        diagnostics,
        latest: None,
        snapshot: None,
        prepared_base: 0,
        pending_window: false,
        timeline: 0,
        generation: 0,
        startup_reanchors: 0,
        correction_reanchors: 0,
        last_reanchor_error: None,
        pending_reanchor: None,
        allocation_bytes: [preflight.bytes(), 0],
    };
    let canonical = worker.queue.lock();
    let actual = allocation_budget(
        limits,
        channels,
        frames,
        canonical.queue.capacity(),
        worker.private.queue.capacity(),
        Some(worker.producer.allocation()),
        device.events.buffer().capacity(),
        device.reader.last_capacity(),
        [device.pcm.capacity(), device.float.capacity()],
    )
    .ok_or_else(|| Error::Output("player reserved allocation size overflow".into()))?;
    drop(canonical);
    worker.allocation_bytes[1] = actual.bytes();
    Ok((device, worker))
}

/// Inline roots include pipe endpoints and the private queue exactly once.
/// Separate allocations are counted using requested or observed capacities.
#[allow(clippy::too_many_arguments)]
fn allocation_budget(
    limits: RendererQueueLimits,
    channels: usize,
    frames: usize,
    canonical_capacity: usize,
    private_capacity: usize,
    transport: Option<transport::Allocation>,
    feedback_capacity: usize,
    last_capacity: usize,
    scratch_capacities: [usize; 2],
) -> Option<Budget> {
    let mut budget = Budget::sources(limits, channels, canonical_capacity, private_capacity)?;
    budget.add_transport(WINDOWS, frames, channels, transport)?;
    budget.add_feedback::<Event>(feedback_capacity)?;
    budget.add_inline::<Worker>()?;
    budget.add_inline::<Device>()?;
    budget.add_inline::<CallbackConfig>()?;
    budget.add_inline::<RendererOwner>()?;
    budget.add_inline::<PlayerScope>()?;
    budget.add_inline::<PreparationResource>()?;
    budget.add_inline::<SyncedPlayer>()?;
    // The native data wrapper captures mono adaptation; the error callback
    // captures the fields of CallbackOutputs. Their host storage is opaque.
    budget.add_inline::<bool>()?;
    budget.add_inline::<CallbackOutputs>()?;
    budget.add_arc::<Mutex<PlaybackQueue>>()?;
    budget.add_arc::<Mutex<Option<String>>>()?;
    budget.add_arc_layout(RendererOwner::shared_layout())?;
    budget.add_arc_layout(GainControl::shared_layout())?;
    budget.add_arc::<Publication>()?;
    budget.add_arc::<ingress::ControlGate>()?;
    budget.add_arc::<ingress::PublishedCheckpoint>()?;
    budget.add_arc::<PlanMailbox>()?;
    budget.add_arc::<TimingMailbox>()?;
    budget.add_arc::<AtomicBool>()?; // worker stop
    budget.add_arc::<AtomicU64>()?; // actual consumption
    budget.add_arc::<AtomicU64>()?; // dynamic delay
    budget.add_arc::<Mutex<SyncDiagnosticsSnapshot>>()?;
    budget.add_array::<i32>(last_capacity)?;
    budget.add_array::<i32>(scratch_capacities[0])?;
    budget.add_array::<f32>(scratch_capacities[1])?;
    Some(budget)
}

pub(super) struct PreparationResource {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl Drop for PreparationResource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.thread.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

impl Worker {
    pub(super) fn spawn(mut self) -> Result<PreparationResource, Error> {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("sendspin-prepare".into())
            .spawn(move || {
                while !worker_stop.load(Ordering::Acquire) {
                    self.step(Instant::now());
                    thread::park_timeout(Duration::from_millis(1));
                }
            })
            .map_err(|error| Error::Output(error.to_string()))?;
        Ok(PreparationResource {
            stop,
            thread: Some(handle),
        })
    }

    /// One bounded executor turn. Tests call this same production executor with
    /// their captured Instant; the live thread supplies the actual current time.
    pub(super) fn step(&mut self, now: Instant) {
        self.producer.reclaim();
        let mut timing_update = None;
        let mut use_reserved_credit = false;
        let sync_arc = Arc::clone(&self.sync);
        let sync = sync_arc.try_lock();
        let mut latest_diagnostics = None;
        let visible = self.events.slots();
        for _ in 0..visible {
            let Ok(mut event) = self.events.pop() else {
                break;
            };
            if event.diagnostics.generation != self.generation {
                // A delayed event can still report telemetry, but cannot supply
                // the new generation's startup latency or correction history.
                latest_diagnostics = Some(event.diagnostics);
                continue;
            }
            if let (Some(sync), Some(input)) = (sync.as_deref(), event.feedback) {
                if let Some(request) =
                    self.feedback
                        .process(sync, input, self.timeline, &self.plans)
                {
                    self.pending_reanchor = Some(request);
                }
                if let Some((serial, raw, filtered)) = self.feedback.last_error() {
                    if serial == event.diagnostics.callbacks {
                        event.diagnostics.raw_error_us = Some(raw);
                        event.diagnostics.filtered_error_us = Some(filtered);
                    }
                }
            }
            latest_diagnostics = Some(event.diagnostics);
            self.latest = Some(event);
        }
        if !self.owner.needs_terminal_check(self.scope) {
            if let Some(sync) = sync.as_deref() {
                let queue_arc = Arc::clone(&self.queue);
                if let Some(mut queue) = queue_arc.try_lock() {
                    if queue.generation != self.generation {
                        self.generation = queue.generation;
                        self.latest = None;
                        self.timeline = self.publication.control.view().epoch();
                        self.pending_reanchor = None;
                        self.snapshot = None;
                        self.pending_window = false;
                        self.private.clear();
                    }
                    let delay = self.delay.load(Ordering::Relaxed);
                    if queue.initialized {
                        let explicit = queue.force_reanchor;
                        let gross = self
                            .pending_reanchor
                            .filter(|event| event.input.timeline == self.timeline);
                        if (explicit || gross.is_some()) && self.producer.available_credits() > 0 {
                            if let Some(event) = self.latest {
                                let anchor_delta =
                                    event.latency_floor.unwrap_or(event.playback_delta);
                                let handoff = if explicit && !event.started {
                                    Duration::from_secs_f64(
                                        event.diagnostics.callback_frames as f64
                                            / f64::from(self.sample_rate),
                                    )
                                } else {
                                    Duration::ZERO
                                };
                                let outcome = queue.try_reanchor_prepared(
                                    &self.publication,
                                    &self.owner,
                                    self.scope,
                                    || {
                                        let anchor =
                                            now.checked_add(anchor_delta)?.checked_add(handoff)?;
                                        let client = sync.instant_to_client_micros(anchor);
                                        let client =
                                            i64::try_from(i128::from(client) + i128::from(delay))
                                                .ok()?;
                                        sync.client_to_server_micros(client)
                                    },
                                    explicit,
                                );
                                if let Ok(Some(_)) = outcome {
                                    self.applied_delay_us = delay;
                                    use_reserved_credit = true;
                                    self.timeline = self.publication.control.view().epoch();
                                    self.snapshot = None;
                                    self.pending_window = false;
                                    self.feedback.reanchor_committed(
                                        self.timeline,
                                        sync,
                                        &self.plans,
                                    );
                                    if explicit {
                                        self.startup_reanchors =
                                            self.startup_reanchors.saturating_add(1);
                                    } else {
                                        self.correction_reanchors =
                                            self.correction_reanchors.saturating_add(1);
                                        self.last_reanchor_error = self
                                            .feedback
                                            .last_error()
                                            .map(|(_, _, filtered)| filtered);
                                    }
                                    self.pending_reanchor = None;
                                }
                            }
                        }
                        if !queue.force_reanchor || self.snapshot.is_some() {
                            let key = (
                                self.publication.control.view().epoch(),
                                queue.next_source_id,
                            );
                            if self.snapshot != Some(key) {
                                // Appending does not invalidate prepared PCM. Keep
                                // the worker horizon instead of refilling the ring
                                // with a duplicate of its still-valid prefix.
                                let append = self.snapshot.is_some_and(|(epoch, _)| epoch == key.0);
                                let refreshed = append
                                    && queue
                                        .reconcile_actual(
                                            &self.publication,
                                            &self.owner,
                                            self.scope,
                                        )
                                        .is_ok_and(|view| {
                                            if self.prepared_base < queue.settled_consumed
                                                || !self.publication.control.validate(&view)
                                            {
                                                return false;
                                            }
                                            queue.refresh_preparation_into(&mut self.private);
                                            true
                                        });
                                if refreshed {
                                    self.snapshot = Some(key);
                                } else if let Ok((epoch, revision, base)) = queue
                                    .preparation_snapshot(
                                        &self.publication,
                                        &self.owner,
                                        self.scope,
                                        &mut self.private,
                                    )
                                {
                                    self.snapshot = Some((epoch, revision));
                                    self.prepared_base = base;
                                    self.pending_window = false;
                                }
                            } else {
                                let _ = queue.reconcile_actual(
                                    &self.publication,
                                    &self.owner,
                                    self.scope,
                                );
                            }
                        }
                    }
                    let state = self.owner.start_state(self.scope).ok();
                    let boundary = match state {
                        Some(
                            StartState::Armed { start_at_zone_us }
                            | StartState::BoundaryWon { start_at_zone_us },
                        ) => start_at_zone_us,
                        _ => 0,
                    };
                    let health = sync.health();
                    let valid_until = health
                        .last_valid_t4_us
                        .and_then(|last| last.checked_add(5_000_000))
                        .and_then(|last| sync.clock().micros_to_instant(last));
                    let delay = self.applied_delay_us;
                    timing_update = Some(Timing {
                        epoch: self.publication.control.view().epoch(),
                        generation: queue.generation,
                        delay_us: delay,
                        expected: queue
                            .initialized
                            .then(|| {
                                sync.server_to_local_instant_with_latency(queue.cursor_us, delay)
                            })
                            .flatten(),
                        boundary,
                        boundary_deadline: match i64::try_from(
                            i128::from(boundary) - i128::from(delay),
                        ) {
                            Ok(zone) => sync.server_to_local_instant(zone),
                            // Presentation-zone addition cannot be below i64::MIN;
                            // this boundary is already due at any valid observation.
                            Err(_) => Some(now),
                        },
                        valid: health.synchronized
                            && (health.last_valid_t4_us.is_none() || valid_until.is_some()),
                        valid_until,
                        // A deferred setter does not invalidate the previously
                        // prepared timeline. Initial startup still needs an anchor.
                        pending: !queue.initialized || self.snapshot.is_none(),
                    });
                    self.feedback
                        .model_changed(sync, self.timeline, &self.plans);
                };
            }
        }
        if self.snapshot.is_some() && !self.owner.needs_terminal_check(self.scope) {
            for _ in 0..WINDOWS {
                // Keep one of the existing N credits for a control handoff.
                // A successful reanchor may use it for its first new window.
                if self.producer.available_credits() <= usize::from(!use_reserved_credit) {
                    break;
                }
                if !self.pending_window {
                    let epoch = self.snapshot.expect("preparation snapshot installed").0;
                    self.private.prepare_window(
                        self.producer.builder_mut(),
                        epoch,
                        self.prepared_base,
                        self.channels,
                        self.sample_rate,
                    );
                    let builder = self.producer.builder_mut();
                    if builder.valid == 0 && builder.skipped_tail.is_none() {
                        break;
                    }
                    self.prepared_base = self
                        .prepared_base
                        .checked_add(builder.valid as u64)
                        .expect("scope prepared source progress");
                    self.pending_window = true;
                }
                if !self.producer.publish() {
                    break;
                }
                self.pending_window = false;
                use_reserved_credit = false;
            }
        }
        if let Some(timing) = timing_update {
            // This ordering narrows the handoff gap; the source gate commit and
            // ring publication are not one atomic audible-switch operation.
            self.timing.publish(timing);
        }
        if let Some(mut observation) = latest_diagnostics {
            observation.startup_reanchors = self.startup_reanchors;
            observation.correction_reanchors = self.correction_reanchors;
            observation.last_reanchor_error_us = self.last_reanchor_error;
            self.diagnostics.publish(observation);
        }
    }
}

impl Device {
    pub(super) fn render<T: cpal::SizedSample + cpal::FromSample<f32>>(
        &mut self,
        data: &mut [T],
        timestamp: cpal::OutputStreamTimestamp,
        timestamp_source: cpal::OutputTimestampSource,
        evidence: Option<cpal::OutputTimestampDiagnostics>,
        captured_at: Instant,
        config: &mut CallbackConfig,
        owner: &RendererOwner,
        scope: PlayerScope,
    ) {
        let channels = self.publication.channels;
        let frames = data.len() / channels;
        let sample_rate = self.observation.sample_rate;
        let previous_capture = self.observation.captured_at;
        let observation = &mut self.observation;
        observation.callbacks = observation.callbacks.saturating_add(1);
        observation.requested_frames = observation.requested_frames.saturating_add(frames as u64);
        observation.captured_at = Some(captured_at);
        observation.callback_frames = frames;
        observation.output_xrun_count = evidence.and_then(|value| value.output_xrun_count);
        observation.output_buffer_size_frames =
            evidence.and_then(|value| value.output_buffer_size_frames);
        observation.raw_error_us = None;
        observation.filtered_error_us = None;
        observation.insert_every = 0;
        observation.drop_every = 0;
        observation.playback_delay_us = timestamp
            .playback
            .duration_since(timestamp.callback)
            .as_micros() as u64;
        if let Some(previous) = previous_capture {
            observation.max_callback_gap_us = observation
                .max_callback_gap_us
                .max(captured_at.saturating_duration_since(previous).as_micros() as u64);
        }
        if timestamp_source == cpal::OutputTimestampSource::MonotonicFallback {
            if let Some(evidence) = evidence {
                if let Some(reason) = evidence.fallback_reason {
                    use cpal::OutputTimestampFallbackReason::*;
                    let counter = match reason {
                        Unavailable => &mut observation.fallback_unavailable,
                        Unsupported => &mut observation.fallback_unsupported,
                        Invalid => &mut observation.fallback_invalid,
                        NonMonotonic => &mut observation.fallback_non_monotonic,
                        ClockDomainMismatch => &mut observation.fallback_clock_domain_mismatch,
                    };
                    *counter = counter.saturating_add(1);
                    observation.last_timestamp_fallback_callback = observation.callbacks;
                    observation.last_timestamp_fallback = Some(evidence);
                }
            }
        }
        let timestamp = classify_output_timestamp(timestamp, timestamp_source, self.last_timestamp);
        let open =
            owner.try_callback_telemetry(scope, timestamp.source, timestamp.monotonic_violation);
        self.last_timestamp = Some(timestamp.timestamp.playback);
        let callback = self.publication.control.begin_callback();
        let view = callback.view();
        if view.current_revoked() && view.epoch() > self.minimum_epoch {
            self.minimum_epoch = view.epoch();
            self.started = false;
            self.min_latency = None;
            self.last_latency = None;
        }
        if let Some(timing) = self.timing.read() {
            self.cached_timing = Some(timing);
        }
        if let Some(timing) = self.cached_timing {
            if self.last_generation != Some(timing.generation) {
                self.last_generation = Some(timing.generation);
                self.started = false;
                self.min_latency = None;
                self.last_latency = None;
            }
            observation.generation = timing.generation;
        }
        let measured = (timestamp.source != cpal::OutputTimestampSource::MonotonicFallback
            && timestamp.timestamp.playback >= timestamp.timestamp.callback)
            .then(|| {
                timestamp
                    .timestamp
                    .playback
                    .duration_since(timestamp.timestamp.callback)
            });
        if let Some(measured) = measured {
            self.min_latency = Some(
                self.min_latency
                    .map_or(measured, |floor| floor.min(measured)),
            );
            self.last_latency = Some(measured);
        }
        let period = Duration::from_secs_f64(frames as f64 / f64::from(sample_rate));
        let playback_delta = measured.or(self.last_latency).unwrap_or(period);
        let playback = captured_at
            .checked_add(playback_delta)
            .unwrap_or(captured_at);
        let timing = self.cached_timing;
        let delay_us = timing.map_or_else(
            || config.static_delay_us.load(Ordering::Relaxed),
            |timing| timing.delay_us,
        );
        let timing_valid = timing.is_some_and(|timing| {
            timing.valid && timing.valid_until.is_none_or(|until| captured_at <= until)
        });
        let ready =
            timing.is_some_and(|timing| !timing.pending && timing.epoch >= self.minimum_epoch);
        if !self.started && ready && timing_valid {
            if timing
                .and_then(|timing| timing.expected)
                .is_some_and(|expected| {
                    playback
                        .checked_add(Duration::from_millis(1))
                        .is_some_and(|time| time >= expected)
                })
            {
                self.started = true;
            }
        }
        let mut audible = open && self.started && ready;
        if audible {
            let presentation = match callback.decide_start(None) {
                ScheduledStartOutcome::Waiting { start_at_zone_us } => timing
                    .filter(|timing| timing_valid && timing.boundary == start_at_zone_us)
                    .and_then(|timing| timing.boundary_deadline)
                    .filter(|deadline| playback >= *deadline)
                    .map(|_| start_at_zone_us),
                _ => None,
            };
            audible = matches!(
                callback.decide_start(presentation),
                ScheduledStartOutcome::Unscheduled
                    | ScheduledStartOutcome::Started { .. }
                    | ScheduledStartOutcome::BoundaryWon { .. }
            );
        }
        let timeline = self.reader.timeline(&callback);
        let actual = self.reader.schedule();
        let wanted = self
            .plan_reader
            .read(
                &self.plans,
                timeline,
                self.clock.instant_to_micros(captured_at),
                actual,
            )
            .unwrap_or(Some(actual));
        let mut visible = if self.pipe.return_pending() {
            self.pipe.visible()
        } else {
            0
        };
        if !audible && (view.current_revoked() || view.timeline_reanchored()) {
            while visible > 0
                && self
                    .pipe
                    .peek()
                    .is_some_and(|window| window.epoch < view.epoch())
            {
                visible -= 1;
                if !self.pipe.retire() {
                    break;
                }
            }
        }
        let mut origin = None;
        let mut missing = 0u64;
        let target_gain = config.gain_control.gain();
        for output in data.chunks_mut(self.float.len()) {
            let length = output.len();
            let usable = length / channels * channels;
            let float = &mut self.float[..length];
            if audible {
                let rendered = self.reader.render_with_budget(
                    &mut self.pipe,
                    &callback,
                    &self.publication.checkpoint,
                    &self.publication.consumed,
                    &mut self.pcm[..usable],
                    measured.is_some(),
                    wanted,
                    &mut visible,
                );
                if origin.is_none() {
                    origin = rendered.source_cursor_us;
                }
                missing = missing.saturating_add(rendered.missing as u64);
                observation.inserted_frames = observation
                    .inserted_frames
                    .saturating_add(rendered.inserted as u64);
                observation.dropped_frames = observation
                    .dropped_frames
                    .saturating_add(rendered.dropped as u64);
                for (out, sample) in float[..usable].iter_mut().zip(&self.pcm[..usable]) {
                    *out = f32::from_sample(*sample);
                }
                float[usable..].fill(0.0);
                self.gain.apply(float, channels, target_gain);
            } else {
                self.gain.advance(usable / channels, target_gain);
                float.fill(0.0);
            }
            if let Some(process) = &mut config.process_callback {
                process(float);
            }
            for (destination, sample) in output.iter_mut().zip(float.iter().copied()) {
                *destination = T::from_sample(sample);
            }
        }
        if data.is_empty() {
            if let Some(process) = &mut config.process_callback {
                process(&mut []);
            }
        }
        if audible {
            let applied = self.reader.schedule();
            if measured.is_some() {
                observation.insert_every = applied.insert_every_n_frames;
                observation.drop_every = applied.drop_every_n_frames;
            }
            owner.record_callback_underrun(missing);
            observation.underrun_frames = observation.underrun_frames.saturating_add(missing);
        } else {
            observation.silent_callbacks = observation.silent_callbacks.saturating_add(1);
            observation.silent_frames = observation.silent_frames.saturating_add(frames as u64);
        }
        let feedback = origin.map(|source_cursor_us| Observation {
            delay_us,
            latency_floor: self.min_latency,
            reset_epoch: self.plan_reader.reset_epoch(),
            input: Measurement {
                settled_seen: self.plan_reader.settled_seen,
                timeline,
                serial: observation.callbacks,
                source_cursor_us,
                presentation: measured.map(|_| playback),
                actual_schedule: self.reader.schedule(),
            },
        });
        drop(callback);
        let _ = self.events.push(Event {
            diagnostics: *observation,
            feedback,
            playback_delta,
            latency_floor: self.min_latency,
            started: self.started,
        });
    }
}

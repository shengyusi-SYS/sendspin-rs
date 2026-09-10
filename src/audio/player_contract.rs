// ABOUTME: Scope-fenced bounded renderer and shared terminal finalization contracts
// ABOUTME: Keeps queue accounting, health, and teardown races typed and observable

use cpal::traits::StreamTrait;
use cpal::{OutputTimestampSource, Stream};
use parking_lot::{Condvar, Mutex, MutexGuard};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const CALLBACK_CLOSED: u64 = 1 << 63;
const CALLBACK_COUNT_MASK: u64 = !CALLBACK_CLOSED;
const TIMESTAMP_TELEMETRY_CLOSED: u64 = 1 << 63;
const TIMESTAMP_TELEMETRY_CLOSING: u64 = 1 << 62;
const TIMESTAMP_TELEMETRY_FLAGS: u64 = TIMESTAMP_TELEMETRY_CLOSED | TIMESTAMP_TELEMETRY_CLOSING;
// Low bits form a single-writer sequence: even is stable, odd is committing.
// CPAL invokes one data callback at a time for a stream. A concurrent attempt
// fails instead of blocking the real-time thread.
const TIMESTAMP_TELEMETRY_VERSION_MASK: u64 = !TIMESTAMP_TELEMETRY_FLAGS;

/// Opaque identity for one opened renderer lifetime.
///
/// Scopes can only be obtained from [`RendererOwner::mint_scope`].
///
/// ```compile_fail
/// use sendspin::audio::PlayerScope;
/// let _ = PlayerScope { id: 1 };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlayerScope {
    id: u64,
}

/// Failure to create another renderer scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ScopeMintError {
    /// The monotonically increasing scope identifier has been exhausted.
    #[error("renderer scope identifiers exhausted")]
    Exhausted,
}

/// Backend failure while preflighting, opening, or starting an output stream.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct OutputBackendError {
    message: String,
}

impl OutputBackendError {
    /// Create a stable backend diagnostic without exposing backend-specific error types.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Stable diagnostic message supplied by the output backend.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Typed failure to open one explicitly leased output renderer.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OpenError {
    /// Renderer scope allocation failed.
    #[error("renderer scope allocation failed: {0}")]
    Scope(#[from] ScopeMintError),
    /// The caller's route generation is no longer current.
    #[error("output route generation is stale")]
    StaleGeneration,
    /// The leased device does not match the requested canonical identity.
    #[error("output identity mismatch")]
    IdentityMismatch,
    /// The requested output lease is not available.
    #[error("output lease unavailable")]
    LeaseUnavailable,
    /// The explicit device cannot satisfy the requested stream format.
    #[error("unsupported output format")]
    UnsupportedFormat,
    /// The selected output backend rejected stream creation or startup.
    #[error("output backend failed: {0}")]
    Backend(OutputBackendError),
}

/// Invalid bounded-queue configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererQueueLimitsError {
    /// Every limit must be non-zero and the per-chunk limit must fit the queue.
    Invalid,
}

/// Immutable hard limits for one renderer queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RendererQueueLimits {
    hard_frames: usize,
    hard_buffers: usize,
    max_chunk_frames: usize,
}

impl RendererQueueLimits {
    /// Validate and construct queue limits.
    pub fn new(
        hard_frames: usize,
        hard_buffers: usize,
        max_chunk_frames: usize,
    ) -> Result<Self, RendererQueueLimitsError> {
        if hard_frames == 0
            || hard_buffers == 0
            || max_chunk_frames == 0
            || max_chunk_frames > hard_frames
        {
            return Err(RendererQueueLimitsError::Invalid);
        }
        Ok(Self {
            hard_frames,
            hard_buffers,
            max_chunk_frames,
        })
    }

    /// Maximum queued frames.
    pub fn hard_frames(self) -> usize {
        self.hard_frames
    }

    /// Maximum queued buffers.
    pub fn hard_buffers(self) -> usize {
        self.hard_buffers
    }

    /// Maximum frames accepted in one enqueue.
    pub fn max_chunk_frames(self) -> usize {
        self.max_chunk_frames
    }
}

/// Typed result of a renderer enqueue attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnqueueOutcome {
    /// The chunk was accepted and the returned counts include it.
    Accepted {
        /// Frames now queued.
        queued_frames: usize,
        /// Buffers now queued.
        queued_buffers: usize,
    },
    /// A hard frame, buffer, per-chunk, or checked-arithmetic limit rejected it.
    Full {
        /// Frames queued before the rejected attempt.
        queued_frames: usize,
        /// Buffers queued before the rejected attempt.
        queued_buffers: usize,
    },
    /// The buffer sample rate or channel layout does not match this player.
    FormatMismatch,
    /// The buffer is not whole-frame or its duration/timestamp arithmetic overflows.
    InvalidBuffer,
    /// This scope no longer accepts or consumes audio.
    Closed,
    /// The supplied scope is not the current renderer lifetime.
    StaleScope,
}

/// Typed result shared by non-enqueue renderer operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererOperationOutcome {
    /// The operation mutated the current open scope.
    Applied,
    /// The current scope is closed or finalizing/finalized.
    Closed,
    /// The supplied scope is not current.
    StaleScope,
}

/// Scheduled presentation state for one renderer scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartState {
    /// No explicit presentation boundary is armed.
    Idle,
    /// Callbacks must remain silent until this Zone timestamp.
    Armed {
        /// Frozen requested presentation boundary.
        start_at_zone_us: i64,
    },
    /// One callback won the boundary and froze the timestamp for all successors.
    BoundaryWon {
        /// Frozen presentation boundary shared by every later callback.
        start_at_zone_us: i64,
    },
}

/// Result of evaluating one callback against the scheduled boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduledStartOutcome {
    /// This scope has no explicit schedule and may use its existing start logic.
    Unscheduled,
    /// The boundary is still in the future; this callback must remain silent.
    Waiting {
        /// Frozen requested presentation boundary.
        start_at_zone_us: i64,
    },
    /// This callback atomically won the boundary.
    BoundaryWon {
        /// Frozen presentation boundary.
        start_at_zone_us: i64,
    },
    /// A prior callback won; this callback reuses the frozen boundary.
    Started {
        /// Frozen presentation boundary.
        start_at_zone_us: i64,
    },
    /// The scope is closed or finalizing/finalized.
    Closed,
    /// The supplied scope is not current.
    StaleScope,
}

/// Typed result of arming an explicit scheduled boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduledArmOutcome {
    /// The idle scope was armed.
    Armed,
    /// The scope already has an armed boundary.
    AlreadyArmed {
        /// Existing frozen request.
        start_at_zone_us: i64,
    },
    /// A callback already won, so this scope can never be re-armed.
    BoundaryAlreadyWon {
        /// Boundary frozen by the winner.
        start_at_zone_us: i64,
    },
    /// The scope is closed or finalizing/finalized.
    Closed,
    /// The supplied scope is stale.
    StaleScope,
}

/// Faults that close renderer acceptance while preserving finalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererFault {
    /// The selected output disappeared or was invalidated.
    OutputInvalidated,
    /// The audio callback failed.
    CallbackFailed,
    /// A trusted presentation timestamp was unavailable.
    TimestampUnavailable,
    /// The output backend rejected the frozen output contract.
    OutputContractRejected,
}

/// Renderer-visible terminal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererTerminal {
    /// Acceptance and consumption were explicitly closed.
    Closed,
    /// The stream was explicitly torn down.
    TornDown,
    /// Finalization followed a renderer fault.
    Faulted(RendererFault),
}

/// Winner of the single terminal finalizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalWinner {
    /// Normal explicit teardown.
    ExplicitTeardown,
    /// Abort won before scheduled presentation began.
    PreStartAbort,
}

/// Proof published only after the owned stream has been released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalAck {
    stream_released: bool,
    callback_stopped: bool,
}

impl TerminalAck {
    /// Whether the finalizer released the stream resource.
    pub fn stream_released(self) -> bool {
        self.stream_released
    }

    /// Whether callbacks are stopped at ack publication time.
    pub fn callback_stopped(self) -> bool {
        self.callback_stopped
    }
}

/// Shared terminal state machine snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalState {
    /// No terminal request has won.
    Open,
    /// One request is dropping the owned stream.
    Finalizing {
        /// The sole finalizer winner.
        winner: TerminalWinner,
    },
    /// The stream has been dropped and the final ack is stable.
    Finalized {
        /// The sole finalizer winner.
        winner: TerminalWinner,
        /// Shared final acknowledgement.
        ack: TerminalAck,
    },
}

/// Stable terminal result observed by winner and losers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalFinalization {
    /// The request that won the finalizer.
    pub winner: TerminalWinner,
    /// Ack published after stream release.
    pub ack: TerminalAck,
}

/// Typed outcome of a terminal request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalOutcome {
    /// This caller won and dropped the scope-owned terminal resource.
    Won(TerminalFinalization),
    /// This caller raced with the winner and waited for its ack.
    Lost(TerminalFinalization),
    /// Finalization had already completed before this request.
    AlreadyFinalized(TerminalFinalization),
    /// The supplied scope is stale.
    StaleScope,
}

/// Typed result of an abort that is only legal before `BoundaryWon`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreStartAbortOutcome {
    /// This abort won the shared terminal finalizer.
    Won(TerminalFinalization),
    /// Another terminal request won and this abort observed its ack.
    Lost(TerminalFinalization),
    /// Finalization was already complete.
    AlreadyFinalized(TerminalFinalization),
    /// A callback already won the scheduled boundary, so abort cannot mutate.
    BoundaryAlreadyWon {
        /// Frozen boundary that defeated this abort.
        start_at_zone_us: i64,
    },
    /// The supplied scope is stale.
    StaleScope,
}

impl PreStartAbortOutcome {
    /// Return the shared finalization when this request entered the terminal path.
    pub fn finalization(self) -> Option<TerminalFinalization> {
        match self {
            Self::Won(value) | Self::Lost(value) | Self::AlreadyFinalized(value) => Some(value),
            Self::BoundaryAlreadyWon { .. } | Self::StaleScope => None,
        }
    }
}

impl TerminalOutcome {
    /// Return the shared finalization, if this was a current-scope request.
    pub fn finalization(self) -> Option<TerminalFinalization> {
        match self {
            Self::Won(value) | Self::Lost(value) | Self::AlreadyFinalized(value) => Some(value),
            Self::StaleScope => None,
        }
    }
}

/// Owner-generated capacity snapshot.
///
/// ```compile_fail
/// use sendspin::audio::RendererCapacitySnapshot;
/// let _ = RendererCapacitySnapshot { hard_frames: 1, hard_buffers: 1,
///     current_frames: 0, current_buffers: 0, high_water_frames: 0,
///     high_water_buffers: 0 };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RendererCapacitySnapshot {
    hard_frames: usize,
    hard_buffers: usize,
    current_frames: usize,
    current_buffers: usize,
    high_water_frames: usize,
    high_water_buffers: usize,
}

impl RendererCapacitySnapshot {
    /// Hard frame limit.
    pub fn hard_frames(self) -> usize {
        self.hard_frames
    }
    /// Hard buffer limit.
    pub fn hard_buffers(self) -> usize {
        self.hard_buffers
    }
    /// Frames currently queued.
    pub fn current_frames(self) -> usize {
        self.current_frames
    }
    /// Buffers currently queued.
    pub fn current_buffers(self) -> usize {
        self.current_buffers
    }
    /// Highest frame depth observed in this scope.
    pub fn high_water_frames(self) -> usize {
        self.high_water_frames
    }
    /// Highest buffer depth observed in this scope.
    pub fn high_water_buffers(self) -> usize {
        self.high_water_buffers
    }
}

/// Owner-generated same-callback output timestamp evidence.
///
/// ```compile_fail
/// use sendspin::audio::OutputTimestampEvidenceSnapshot;
/// let _ = OutputTimestampEvidenceSnapshot {
///     device_presentation: 0, monotonic_fallback: 0,
///     unspecified: 0, monotonic_violations: 0,
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputTimestampEvidenceSnapshot {
    device_presentation: u64,
    monotonic_fallback: u64,
    unspecified: u64,
    monotonic_violations: u64,
}

impl OutputTimestampEvidenceSnapshot {
    /// Callbacks whose playback timestamp came from the device presentation timeline.
    pub fn device_presentation(self) -> u64 {
        self.device_presentation
    }

    /// Callbacks whose host fell back to a monotonic callback timestamp.
    pub fn monotonic_fallback(self) -> u64 {
        self.monotonic_fallback
    }

    /// Callbacks whose host does not expose stable timestamp provenance.
    pub fn unspecified(self) -> u64 {
        self.unspecified
    }

    /// Playback timestamps that did not strictly advance within the opened stream.
    pub fn monotonic_violations(self) -> u64 {
        self.monotonic_violations
    }

    /// Callbacks that carried a timestamp value/source pair from the real output callback.
    pub fn provenance_callback_count(self) -> u64 {
        self.device_presentation
            .saturating_add(self.monotonic_fallback)
            .saturating_add(self.unspecified)
    }
}

/// Owner-generated renderer health snapshot.
///
/// ```compile_fail
/// use sendspin::audio::RendererHealthSnapshot;
/// let _ = RendererHealthSnapshot {
///     scope: panic!(), queued_frames: 0, queued_buffers: 0, limits: panic!(),
///     consumed_frames: 0, callback_count: 0, underrun_frames: 0,
///     output_timestamps: panic!(),
///     last_presentation_boundary_zone_us: None, fault: None, terminal: None,
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RendererHealthSnapshot {
    scope: PlayerScope,
    queued_frames: usize,
    queued_buffers: usize,
    limits: RendererQueueLimits,
    consumed_frames: u64,
    callback_count: u64,
    underrun_frames: u64,
    output_timestamps: OutputTimestampEvidenceSnapshot,
    last_presentation_boundary_zone_us: Option<i64>,
    fault: Option<RendererFault>,
    terminal: Option<RendererTerminal>,
}

impl RendererHealthSnapshot {
    /// Current renderer scope.
    pub fn scope(self) -> PlayerScope {
        self.scope
    }
    /// Frames currently queued.
    pub fn queued_frames(self) -> usize {
        self.queued_frames
    }
    /// Buffers currently queued.
    pub fn queued_buffers(self) -> usize {
        self.queued_buffers
    }
    /// Frozen queue limits.
    pub fn limits(self) -> RendererQueueLimits {
        self.limits
    }
    /// Frames consumed by callbacks.
    pub fn consumed_frames(self) -> u64 {
        self.consumed_frames
    }
    /// Callback heartbeat count.
    pub fn callback_count(self) -> u64 {
        self.callback_count
    }
    /// Frames emitted as presentation underrun silence.
    pub fn underrun_frames(self) -> u64 {
        self.underrun_frames
    }
    /// Same-callback output timestamp provenance and monotonicity evidence.
    pub fn output_timestamps(self) -> OutputTimestampEvidenceSnapshot {
        self.output_timestamps
    }
    /// Last trusted scheduled presentation boundary.
    pub fn last_presentation_boundary_zone_us(self) -> Option<i64> {
        self.last_presentation_boundary_zone_us
    }
    /// Current renderer fault, if any.
    pub fn fault(self) -> Option<RendererFault> {
        self.fault
    }
    /// Current renderer terminal state, if any.
    pub fn terminal(self) -> Option<RendererTerminal> {
        self.terminal
    }
}

#[derive(Debug)]
struct ScopeState {
    scope: PlayerScope,
    queued_frames: usize,
    queued_buffers: usize,
    high_water_frames: usize,
    high_water_buffers: usize,
    consumed_frames: u64,
    last_boundary: Option<i64>,
    fault: Option<RendererFault>,
    terminal: Option<RendererTerminal>,
    accepting: bool,
    terminal_state: TerminalState,
    terminal_waiters: usize,
    start_state: StartState,
}

impl ScopeState {
    fn new(scope: PlayerScope) -> Self {
        Self {
            scope,
            queued_frames: 0,
            queued_buffers: 0,
            high_water_frames: 0,
            high_water_buffers: 0,
            consumed_frames: 0,
            last_boundary: None,
            fault: None,
            terminal: None,
            accepting: true,
            terminal_state: TerminalState::Open,
            terminal_waiters: 0,
            start_state: StartState::Idle,
        }
    }

    fn scheduled_start(&mut self, presentation_zone_us: Option<i64>) -> ScheduledStartOutcome {
        match self.start_state {
            StartState::Idle => ScheduledStartOutcome::Unscheduled,
            StartState::Armed { start_at_zone_us } => {
                if presentation_zone_us.is_none_or(|now| now < start_at_zone_us) {
                    return ScheduledStartOutcome::Waiting { start_at_zone_us };
                }
                self.start_state = StartState::BoundaryWon { start_at_zone_us };
                self.last_boundary = Some(start_at_zone_us);
                ScheduledStartOutcome::BoundaryWon { start_at_zone_us }
            }
            StartState::BoundaryWon { start_at_zone_us } => {
                ScheduledStartOutcome::Started { start_at_zone_us }
            }
        }
    }

    fn record_trusted_boundary(&mut self, boundary_zone_us: i64) {
        match self.start_state {
            StartState::Idle => self.last_boundary = Some(boundary_zone_us),
            StartState::Armed { .. } => {}
            StartState::BoundaryWon { start_at_zone_us } => {
                self.last_boundary = Some(start_at_zone_us);
            }
        }
    }

    fn consume_frames(&mut self, requested: usize) -> usize {
        let consumed = requested.min(self.queued_frames);
        self.queued_frames -= consumed;
        if self.queued_frames == 0 {
            self.queued_buffers = 0;
        }
        self.consumed_frames = self.consumed_frames.saturating_add(consumed as u64);
        consumed
    }

    fn clear_queued_state(&mut self) {
        self.queued_frames = 0;
        self.queued_buffers = 0;
        if !matches!(self.start_state, StartState::BoundaryWon { .. }) {
            self.last_boundary = None;
            self.start_state = StartState::Idle;
        }
    }
}

struct OwnerState {
    next_scope: u64,
    current: Option<ScopeState>,
    terminal_resource: Option<(PlayerScope, OwnedTerminalResource)>,
}

enum OwnedTerminalResource {
    Stream(Stream),
    #[cfg(test)]
    Test(Box<dyn Send + 'static>),
}

struct Shared {
    state: Mutex<OwnerState>,
    finalized: Condvar,
    active_scope: AtomicU64,
    callback_state: AtomicU64,
    timestamp_telemetry_state: AtomicU64,
    underrun_frames: AtomicU64,
    device_presentation_timestamps: AtomicU64,
    monotonic_fallback_timestamps: AtomicU64,
    unspecified_timestamps: AtomicU64,
    timestamp_monotonic_violations: AtomicU64,
    #[cfg(test)]
    timestamp_snapshot_retries: AtomicU64,
}

enum FinalizeRequestOutcome {
    Terminal(TerminalOutcome),
    TerminalPending,
    BoundaryAlreadyWon { start_at_zone_us: i64 },
}

pub(crate) enum ActualTeardownOutcome {
    Complete(TerminalOutcome),
    FinalizationPending,
}

/// Scope-fenced owner of bounded renderer accounting and terminal state.
#[derive(Clone)]
pub struct RendererOwner {
    limits: RendererQueueLimits,
    shared: Arc<Shared>,
}

impl RendererOwner {
    /// Create an owner with validated hard limits.
    pub fn new(limits: RendererQueueLimits) -> Self {
        Self {
            limits,
            shared: Arc::new(Shared {
                state: Mutex::new(OwnerState {
                    next_scope: 0,
                    current: None,
                    terminal_resource: None,
                }),
                finalized: Condvar::new(),
                active_scope: AtomicU64::new(0),
                callback_state: AtomicU64::new(CALLBACK_CLOSED),
                timestamp_telemetry_state: AtomicU64::new(TIMESTAMP_TELEMETRY_CLOSED),
                underrun_frames: AtomicU64::new(0),
                device_presentation_timestamps: AtomicU64::new(0),
                monotonic_fallback_timestamps: AtomicU64::new(0),
                unspecified_timestamps: AtomicU64::new(0),
                timestamp_monotonic_violations: AtomicU64::new(0),
                #[cfg(test)]
                timestamp_snapshot_retries: AtomicU64::new(0),
            }),
        }
    }

    /// Mint and install a fresh renderer scope, resetting all per-scope state.
    ///
    /// If a prior scope exists, this blocks until its final ack is published
    /// and every already-waiting terminal loser has read that same ack. Callers
    /// must not invoke it on the thread responsible for finalizing that scope.
    pub fn mint_scope(&self) -> Result<PlayerScope, ScopeMintError> {
        let mut owner = self.shared.state.lock();
        while owner.current.as_ref().is_some_and(|scope| {
            !matches!(scope.terminal_state, TerminalState::Finalized { .. })
                || scope.terminal_waiters != 0
        }) {
            self.shared.finalized.wait(&mut owner);
        }
        let next = owner
            .next_scope
            .checked_add(1)
            .ok_or(ScopeMintError::Exhausted)?;
        owner.next_scope = next;
        let scope = PlayerScope { id: next };
        debug_assert!(owner.terminal_resource.is_none());
        self.shared.active_scope.store(next, Ordering::Release);
        self.shared.underrun_frames.store(0, Ordering::Release);
        self.shared
            .device_presentation_timestamps
            .store(0, Ordering::Release);
        self.shared
            .monotonic_fallback_timestamps
            .store(0, Ordering::Release);
        self.shared
            .unspecified_timestamps
            .store(0, Ordering::Release);
        self.shared
            .timestamp_monotonic_violations
            .store(0, Ordering::Release);
        #[cfg(test)]
        self.shared
            .timestamp_snapshot_retries
            .store(0, Ordering::Release);
        self.shared
            .timestamp_telemetry_state
            .store(0, Ordering::Release);
        self.shared.callback_state.store(0, Ordering::Release);
        owner.current = Some(ScopeState::new(scope));
        Ok(scope)
    }

    /// Transfer ownership of a real opened CPAL stream into the sole terminal finalizer.
    ///
    /// Only the current open scope can attach one stream. Callers cannot synthesize a
    /// terminal acknowledgement: it is published only after this concrete stream is dropped.
    pub fn attach_output_stream(
        &self,
        scope: PlayerScope,
        stream: Stream,
    ) -> RendererOperationOutcome {
        self.attach_owned_resource(scope, OwnedTerminalResource::Stream(stream))
    }

    /// Start the concrete stream already owned by this scope's terminal finalizer.
    pub fn start_output_stream(
        &self,
        scope: PlayerScope,
    ) -> Result<RendererOperationOutcome, OutputBackendError> {
        let owner = self.shared.state.lock();
        let Some(state) = owner.current.as_ref() else {
            return Ok(RendererOperationOutcome::StaleScope);
        };
        if state.scope != scope {
            return Ok(RendererOperationOutcome::StaleScope);
        }
        if !state.accepting || state.terminal_state != TerminalState::Open {
            return Ok(RendererOperationOutcome::Closed);
        }
        let Some((resource_scope, OwnedTerminalResource::Stream(stream))) =
            owner.terminal_resource.as_ref()
        else {
            return Ok(RendererOperationOutcome::Closed);
        };
        if *resource_scope != scope {
            return Ok(RendererOperationOutcome::StaleScope);
        }
        stream
            .play()
            .map_err(|error| OutputBackendError::new(error.to_string()))?;
        Ok(RendererOperationOutcome::Applied)
    }

    fn attach_owned_resource(
        &self,
        scope: PlayerScope,
        resource: OwnedTerminalResource,
    ) -> RendererOperationOutcome {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_ref() else {
            return RendererOperationOutcome::StaleScope;
        };
        if state.scope != scope {
            return RendererOperationOutcome::StaleScope;
        }
        if !state.accepting
            || state.terminal_state != TerminalState::Open
            || owner.terminal_resource.is_some()
        {
            return RendererOperationOutcome::Closed;
        }
        owner.terminal_resource = Some((scope, resource));
        RendererOperationOutcome::Applied
    }

    #[cfg(test)]
    pub(crate) fn attach_test_terminal_resource(
        &self,
        scope: PlayerScope,
        resource: Box<dyn Send + 'static>,
    ) -> RendererOperationOutcome {
        self.attach_owned_resource(scope, OwnedTerminalResource::Test(resource))
    }

    /// Reserve one queued chunk under all hard limits.
    pub fn enqueue(&self, scope: PlayerScope, frames: usize) -> EnqueueOutcome {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return EnqueueOutcome::StaleScope;
        };
        if state.scope != scope {
            return EnqueueOutcome::StaleScope;
        }
        if !state.accepting || state.terminal_state != TerminalState::Open {
            return EnqueueOutcome::Closed;
        }
        let before = EnqueueOutcome::Full {
            queued_frames: state.queued_frames,
            queued_buffers: state.queued_buffers,
        };
        if frames == 0 || frames > self.limits.max_chunk_frames {
            return before;
        }
        let Some(next_frames) = state.queued_frames.checked_add(frames) else {
            return before;
        };
        let Some(next_buffers) = state.queued_buffers.checked_add(1) else {
            return before;
        };
        if next_frames > self.limits.hard_frames || next_buffers > self.limits.hard_buffers {
            return before;
        }
        state.queued_frames = next_frames;
        state.queued_buffers = next_buffers;
        state.high_water_frames = state.high_water_frames.max(next_frames);
        state.high_water_buffers = state.high_water_buffers.max(next_buffers);
        EnqueueOutcome::Accepted {
            queued_frames: next_frames,
            queued_buffers: next_buffers,
        }
    }

    pub(crate) fn enqueue_with_actual<F>(
        &self,
        scope: PlayerScope,
        frames: usize,
        apply: F,
    ) -> EnqueueOutcome
    where
        F: FnOnce() -> (usize, usize),
    {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return EnqueueOutcome::StaleScope;
        };
        if state.scope != scope {
            return EnqueueOutcome::StaleScope;
        }
        if !state.accepting || state.terminal_state != TerminalState::Open {
            return EnqueueOutcome::Closed;
        }
        let full = EnqueueOutcome::Full {
            queued_frames: state.queued_frames,
            queued_buffers: state.queued_buffers,
        };
        if frames == 0 || frames > self.limits.max_chunk_frames {
            return full;
        }
        let Some(candidate_frames) = state.queued_frames.checked_add(frames) else {
            return full;
        };
        let Some(candidate_buffers) = state.queued_buffers.checked_add(1) else {
            return full;
        };
        if candidate_frames > self.limits.hard_frames
            || candidate_buffers > self.limits.hard_buffers
        {
            return full;
        }

        let (actual_frames, actual_buffers) = apply();
        debug_assert!(actual_frames <= candidate_frames);
        debug_assert!(actual_buffers <= candidate_buffers);
        state.queued_frames = actual_frames;
        state.queued_buffers = actual_buffers;
        state.high_water_frames = state.high_water_frames.max(actual_frames);
        state.high_water_buffers = state.high_water_buffers.max(actual_buffers);
        EnqueueOutcome::Accepted {
            queued_frames: actual_frames,
            queued_buffers: actual_buffers,
        }
    }

    pub(crate) fn clear_with_actual<F>(
        &self,
        scope: PlayerScope,
        clear_queue: F,
    ) -> RendererOperationOutcome
    where
        F: FnOnce(),
    {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return RendererOperationOutcome::StaleScope;
        };
        if state.scope != scope {
            return RendererOperationOutcome::StaleScope;
        }
        if !state.accepting || state.terminal_state != TerminalState::Open {
            return RendererOperationOutcome::Closed;
        }
        clear_queue();
        state.clear_queued_state();
        RendererOperationOutcome::Applied
    }

    pub(crate) fn try_callback_heartbeat(&self, scope: PlayerScope) -> bool {
        if self.shared.active_scope.load(Ordering::Acquire) != scope.id {
            return false;
        }
        self.shared
            .callback_state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                if state & CALLBACK_CLOSED != 0 {
                    None
                } else {
                    let count = (state & CALLBACK_COUNT_MASK).saturating_add(1);
                    Some(count.min(CALLBACK_COUNT_MASK))
                }
            })
            .is_ok()
    }

    pub(crate) fn try_callback_telemetry(
        &self,
        scope: PlayerScope,
        source: OutputTimestampSource,
        monotonic_violation: bool,
    ) -> bool {
        self.try_callback_telemetry_with(scope, source, monotonic_violation, || {})
    }

    fn try_callback_telemetry_with<F>(
        &self,
        scope: PlayerScope,
        source: OutputTimestampSource,
        monotonic_violation: bool,
        before_commit: F,
    ) -> bool
    where
        F: FnOnce(),
    {
        let Some(_writer) = self.begin_timestamp_telemetry(scope) else {
            return false;
        };
        before_commit();
        let heartbeat_recorded = self.try_callback_heartbeat(scope);
        if !heartbeat_recorded {
            return false;
        }

        let source_counter = match source {
            OutputTimestampSource::DevicePresentation => {
                &self.shared.device_presentation_timestamps
            }
            OutputTimestampSource::MonotonicFallback => &self.shared.monotonic_fallback_timestamps,
            OutputTimestampSource::Unspecified => &self.shared.unspecified_timestamps,
        };
        saturating_increment(source_counter);
        if monotonic_violation {
            saturating_increment(&self.shared.timestamp_monotonic_violations);
        }
        true
    }

    fn begin_timestamp_telemetry(
        &self,
        scope: PlayerScope,
    ) -> Option<TimestampTelemetryWriter<'_>> {
        if self.shared.active_scope.load(Ordering::Acquire) != scope.id {
            return None;
        }
        self.shared
            .timestamp_telemetry_state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                let version = state & TIMESTAMP_TELEMETRY_VERSION_MASK;
                if state & TIMESTAMP_TELEMETRY_FLAGS != 0 || version & 1 != 0 {
                    None
                } else {
                    Some((version.wrapping_add(1)) & TIMESTAMP_TELEMETRY_VERSION_MASK)
                }
            })
            .ok()?;
        let writer = TimestampTelemetryWriter {
            state: &self.shared.timestamp_telemetry_state,
        };
        if self.shared.active_scope.load(Ordering::Acquire) != scope.id {
            return None;
        }
        Some(writer)
    }

    pub(crate) fn record_callback_underrun(&self, frames: u64) {
        let _ = self.shared.underrun_frames.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| Some(current.saturating_add(frames)),
        );
    }

    fn close_callback_telemetry(&self) {
        // This runs only on owner/control threads. CLOSING fences new writers;
        // an already-entered real-time writer never waits and publishes the
        // stable even sequence from its Drop guard.
        loop {
            let state = self
                .shared
                .timestamp_telemetry_state
                .load(Ordering::Acquire);
            if state & TIMESTAMP_TELEMETRY_CLOSED != 0 {
                break;
            }
            if state & TIMESTAMP_TELEMETRY_CLOSING == 0 {
                if self
                    .shared
                    .timestamp_telemetry_state
                    .compare_exchange(
                        state,
                        state | TIMESTAMP_TELEMETRY_CLOSING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    continue;
                }
            }
            let closing = self
                .shared
                .timestamp_telemetry_state
                .load(Ordering::Acquire);
            if closing & TIMESTAMP_TELEMETRY_VERSION_MASK & 1 != 0 {
                std::thread::yield_now();
                continue;
            }
            let closed = (closing & TIMESTAMP_TELEMETRY_VERSION_MASK) | TIMESTAMP_TELEMETRY_CLOSED;
            if self
                .shared
                .timestamp_telemetry_state
                .compare_exchange(closing, closed, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
        self.shared
            .callback_state
            .fetch_or(CALLBACK_CLOSED, Ordering::AcqRel);
    }

    pub(crate) fn try_callback_permit(
        &self,
        scope: PlayerScope,
    ) -> Option<RendererCallbackPermit<'_>> {
        let owner = self.shared.state.try_lock()?;
        let state = owner.current.as_ref()?;
        if state.scope != scope || !state.accepting || state.terminal_state != TerminalState::Open {
            return None;
        }
        Some(RendererCallbackPermit { owner, scope })
    }

    /// Consume up to `frames` from the current open queue.
    pub fn consume(&self, scope: PlayerScope, frames: usize) -> RendererOperationOutcome {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return RendererOperationOutcome::StaleScope;
        };
        if state.scope != scope {
            return RendererOperationOutcome::StaleScope;
        }
        if !state.accepting || state.terminal_state != TerminalState::Open {
            return RendererOperationOutcome::Closed;
        }
        state.consume_frames(frames);
        RendererOperationOutcome::Applied
    }

    /// Record one callback heartbeat, consumption, underrun, and optional trusted boundary.
    pub fn record_callback(
        &self,
        scope: PlayerScope,
        consumed_frames: usize,
        underrun_frames: u64,
        boundary_zone_us: Option<i64>,
    ) -> RendererOperationOutcome {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return RendererOperationOutcome::StaleScope;
        };
        if state.scope != scope {
            return RendererOperationOutcome::StaleScope;
        }
        if !state.accepting || state.terminal_state != TerminalState::Open {
            return RendererOperationOutcome::Closed;
        }
        if !self.try_callback_heartbeat(scope) {
            return RendererOperationOutcome::Closed;
        }
        self.record_callback_underrun(underrun_frames);
        if let Some(boundary) = boundary_zone_us {
            state.record_trusted_boundary(boundary);
        }
        state.consume_frames(consumed_frames);
        RendererOperationOutcome::Applied
    }

    /// Clear queued/current accounting without changing the scope or high-water marks.
    pub fn clear(&self, scope: PlayerScope) -> RendererOperationOutcome {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return RendererOperationOutcome::StaleScope;
        };
        if state.scope != scope {
            return RendererOperationOutcome::StaleScope;
        }
        if !state.accepting || state.terminal_state != TerminalState::Open {
            return RendererOperationOutcome::Closed;
        }
        state.clear_queued_state();
        RendererOperationOutcome::Applied
    }

    /// Arm one explicit Zone presentation boundary on an idle open scope.
    pub fn arm_scheduled_start(
        &self,
        scope: PlayerScope,
        start_at_zone_us: i64,
    ) -> ScheduledArmOutcome {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return ScheduledArmOutcome::StaleScope;
        };
        if state.scope != scope {
            return ScheduledArmOutcome::StaleScope;
        }
        if !state.accepting || state.terminal_state != TerminalState::Open {
            return ScheduledArmOutcome::Closed;
        }
        match state.start_state {
            StartState::Idle => {
                state.start_state = StartState::Armed { start_at_zone_us };
                ScheduledArmOutcome::Armed
            }
            StartState::Armed { start_at_zone_us } => {
                ScheduledArmOutcome::AlreadyArmed { start_at_zone_us }
            }
            StartState::BoundaryWon { start_at_zone_us } => {
                ScheduledArmOutcome::BoundaryAlreadyWon { start_at_zone_us }
            }
        }
    }

    /// Read the current scope's scheduled presentation state.
    pub fn start_state(&self, scope: PlayerScope) -> Result<StartState, RendererOperationOutcome> {
        let owner = self.shared.state.lock();
        let Some(state) = owner.current.as_ref() else {
            return Err(RendererOperationOutcome::StaleScope);
        };
        if state.scope != scope {
            return Err(RendererOperationOutcome::StaleScope);
        }
        Ok(state.start_state)
    }

    /// Fence future enqueue and callback consumption for this scope.
    pub fn close(&self, scope: PlayerScope) -> RendererOperationOutcome {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return RendererOperationOutcome::StaleScope;
        };
        if state.scope != scope {
            return RendererOperationOutcome::StaleScope;
        }
        if !state.accepting {
            return RendererOperationOutcome::Closed;
        }
        state.accepting = false;
        self.close_callback_telemetry();
        state.terminal = Some(RendererTerminal::Closed);
        RendererOperationOutcome::Applied
    }

    /// Record a typed fault and close acceptance pending the shared finalizer.
    pub fn fault(&self, scope: PlayerScope, fault: RendererFault) -> RendererOperationOutcome {
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return RendererOperationOutcome::StaleScope;
        };
        if state.scope != scope {
            return RendererOperationOutcome::StaleScope;
        }
        if !state.accepting || state.fault.is_some() || state.terminal_state != TerminalState::Open
        {
            return RendererOperationOutcome::Closed;
        }
        state.fault = Some(fault);
        state.accepting = false;
        self.close_callback_telemetry();
        RendererOperationOutcome::Applied
    }

    /// Read the current scope's owner-generated capacity snapshot.
    pub fn capacity(
        &self,
        scope: PlayerScope,
    ) -> Result<RendererCapacitySnapshot, RendererOperationOutcome> {
        let owner = self.shared.state.lock();
        let Some(state) = owner.current.as_ref() else {
            return Err(RendererOperationOutcome::StaleScope);
        };
        if state.scope != scope {
            return Err(RendererOperationOutcome::StaleScope);
        }
        Ok(RendererCapacitySnapshot {
            hard_frames: self.limits.hard_frames,
            hard_buffers: self.limits.hard_buffers,
            current_frames: state.queued_frames,
            current_buffers: state.queued_buffers,
            high_water_frames: state.high_water_frames,
            high_water_buffers: state.high_water_buffers,
        })
    }

    /// Hint for host-side terminal polling; confirm a positive result with health().
    /// Reads only existing scope/close atomics. This is not a terminal acknowledgement.
    pub fn needs_terminal_check(&self, scope: PlayerScope) -> bool {
        self.shared.active_scope.load(Ordering::Acquire) != scope.id
            || self.shared.callback_state.load(Ordering::Acquire) & CALLBACK_CLOSED != 0
    }

    /// Best-effort diagnostic snapshot. Contention may skip this observation.
    /// Does not wait for the renderer or retry an in-flight telemetry write.
    /// Use health() for authoritative consumption and finalization reads.
    pub fn try_health(
        &self,
        scope: PlayerScope,
    ) -> Result<Option<RendererHealthSnapshot>, RendererOperationOutcome> {
        let Some(owner) = self.shared.state.try_lock() else {
            return Ok(None);
        };
        let state = owner
            .current
            .as_ref()
            .filter(|state| state.scope == scope)
            .ok_or(RendererOperationOutcome::StaleScope)?;
        Ok(self
            .try_callback_telemetry_snapshot()
            .map(|telemetry| self.health_snapshot(state, telemetry)))
    }

    /// Read the current scope's complete typed health snapshot.
    pub fn health(
        &self,
        scope: PlayerScope,
    ) -> Result<RendererHealthSnapshot, RendererOperationOutcome> {
        let owner = self.shared.state.lock();
        let Some(state) = owner.current.as_ref() else {
            return Err(RendererOperationOutcome::StaleScope);
        };
        if state.scope != scope {
            return Err(RendererOperationOutcome::StaleScope);
        }
        Ok(self.health_snapshot(state, self.callback_telemetry_snapshot()))
    }

    fn health_snapshot(
        &self,
        state: &ScopeState,
        (callback_count, output_timestamps): (u64, OutputTimestampEvidenceSnapshot),
    ) -> RendererHealthSnapshot {
        RendererHealthSnapshot {
            scope: state.scope,
            queued_frames: state.queued_frames,
            queued_buffers: state.queued_buffers,
            limits: self.limits,
            consumed_frames: state.consumed_frames,
            callback_count,
            underrun_frames: self.shared.underrun_frames.load(Ordering::Acquire),
            output_timestamps,
            last_presentation_boundary_zone_us: state.last_boundary,
            fault: state.fault,
            terminal: state.terminal,
        }
    }

    fn callback_telemetry_snapshot(&self) -> (u64, OutputTimestampEvidenceSnapshot) {
        // Authoritative readers preserve the existing coherent telemetry contract.
        // Background diagnostics use the single-attempt variant instead.
        loop {
            if let Some(snapshot) = self.try_callback_telemetry_snapshot() {
                return snapshot;
            }
            #[cfg(test)]
            self.shared
                .timestamp_snapshot_retries
                .fetch_add(1, Ordering::Release);
            std::thread::yield_now();
        }
    }

    fn try_callback_telemetry_snapshot(&self) -> Option<(u64, OutputTimestampEvidenceSnapshot)> {
        let before = self
            .shared
            .timestamp_telemetry_state
            .load(Ordering::Acquire);
        if before & TIMESTAMP_TELEMETRY_VERSION_MASK & 1 != 0 {
            return None;
        }
        let callback_count =
            self.shared.callback_state.load(Ordering::Acquire) & CALLBACK_COUNT_MASK;
        let output_timestamps = OutputTimestampEvidenceSnapshot {
            device_presentation: self
                .shared
                .device_presentation_timestamps
                .load(Ordering::Acquire),
            monotonic_fallback: self
                .shared
                .monotonic_fallback_timestamps
                .load(Ordering::Acquire),
            unspecified: self.shared.unspecified_timestamps.load(Ordering::Acquire),
            monotonic_violations: self
                .shared
                .timestamp_monotonic_violations
                .load(Ordering::Acquire),
        };
        let after = self
            .shared
            .timestamp_telemetry_state
            .load(Ordering::Acquire);
        (before == after).then_some((callback_count, output_timestamps))
    }

    /// Read the shared terminal state for the current scope.
    pub fn terminal_state(
        &self,
        scope: PlayerScope,
    ) -> Result<TerminalState, RendererOperationOutcome> {
        let owner = self.shared.state.lock();
        let Some(state) = owner.current.as_ref() else {
            return Err(RendererOperationOutcome::StaleScope);
        };
        if state.scope != scope {
            return Err(RendererOperationOutcome::StaleScope);
        }
        Ok(state.terminal_state)
    }

    fn finalize_request<F>(
        &self,
        scope: PlayerScope,
        requested: TerminalWinner,
        pre_start_only: bool,
        clear_queue_on_open: bool,
        clear_armed_queue: F,
    ) -> FinalizeRequestOutcome
    where
        F: FnOnce(),
    {
        let mut clear_armed_queue = Some(clear_armed_queue);
        let mut owner = self.shared.state.lock();
        let Some(state) = owner.current.as_mut() else {
            return FinalizeRequestOutcome::Terminal(TerminalOutcome::StaleScope);
        };
        if state.scope != scope {
            return FinalizeRequestOutcome::Terminal(TerminalOutcome::StaleScope);
        }
        match state.terminal_state {
            TerminalState::Open => {
                if pre_start_only {
                    if let StartState::BoundaryWon { start_at_zone_us } = state.start_state {
                        return FinalizeRequestOutcome::BoundaryAlreadyWon { start_at_zone_us };
                    }
                }
                if pre_start_only || clear_queue_on_open {
                    clear_armed_queue
                        .take()
                        .expect("pre-start clear runs at most once")();
                    state.queued_frames = 0;
                    state.queued_buffers = 0;
                }
                if pre_start_only {
                    state.last_boundary = None;
                    state.start_state = StartState::Idle;
                }
                state.accepting = false;
                self.close_callback_telemetry();
                state.terminal_state = TerminalState::Finalizing { winner: requested };
            }
            TerminalState::Finalizing { .. } => {
                if clear_queue_on_open && requested == TerminalWinner::ExplicitTeardown {
                    clear_armed_queue
                        .take()
                        .expect("terminal queue reconciliation runs at most once")(
                    );
                    state.queued_frames = 0;
                    state.queued_buffers = 0;
                    state.terminal_waiters = state.terminal_waiters.saturating_add(1);
                    return FinalizeRequestOutcome::TerminalPending;
                }
                state.terminal_waiters = state.terminal_waiters.saturating_add(1);
                while matches!(
                    owner.current.as_ref().map(|s| s.terminal_state),
                    Some(TerminalState::Finalizing { .. })
                ) {
                    self.shared.finalized.wait(&mut owner);
                }
                let outcome = finalized_outcome(&owner, scope, false);
                if let Some(state) = owner.current.as_mut().filter(|state| state.scope == scope) {
                    state.terminal_waiters = state.terminal_waiters.saturating_sub(1);
                }
                self.shared.finalized.notify_all();
                return FinalizeRequestOutcome::Terminal(outcome);
            }
            TerminalState::Finalized { .. } => {
                let outcome = finalized_outcome(&owner, scope, true);
                if clear_queue_on_open {
                    clear_armed_queue
                        .take()
                        .expect("terminal queue reconciliation runs at most once")(
                    );
                    let state = owner
                        .current
                        .as_mut()
                        .expect("finalized scope remains installed");
                    state.queued_frames = 0;
                    state.queued_buffers = 0;
                }
                return FinalizeRequestOutcome::Terminal(outcome);
            }
        }
        let terminal_resource = match owner.terminal_resource.take() {
            Some((resource_scope, resource)) if resource_scope == scope => Some(resource),
            Some(other) => {
                owner.terminal_resource = Some(other);
                None
            }
            None => None,
        };
        drop(owner);

        let resource_was_owned = terminal_resource.is_some();
        match terminal_resource {
            Some(OwnedTerminalResource::Stream(stream)) => drop(stream),
            #[cfg(test)]
            Some(OwnedTerminalResource::Test(resource)) => drop(resource),
            None => {}
        }

        let ack = TerminalAck {
            stream_released: resource_was_owned,
            callback_stopped: resource_was_owned,
        };
        let mut owner = self.shared.state.lock();
        let state = owner
            .current
            .as_mut()
            .expect("winning scope remains installed");
        let finalization = TerminalFinalization {
            winner: requested,
            ack,
        };
        state.terminal_state = TerminalState::Finalized {
            winner: requested,
            ack,
        };
        state.terminal = Some(match state.fault {
            Some(fault) => RendererTerminal::Faulted(fault),
            None if requested == TerminalWinner::ExplicitTeardown => RendererTerminal::TornDown,
            None => RendererTerminal::Closed,
        });
        self.shared.finalized.notify_all();
        FinalizeRequestOutcome::Terminal(TerminalOutcome::Won(finalization))
    }

    /// Explicit teardown using the shared terminal path.
    pub fn teardown(&self, scope: PlayerScope) -> TerminalOutcome {
        match self.finalize_request(scope, TerminalWinner::ExplicitTeardown, false, false, || {}) {
            FinalizeRequestOutcome::Terminal(outcome) => outcome,
            FinalizeRequestOutcome::TerminalPending => {
                unreachable!("ordinary teardown waits for existing finalization")
            }
            FinalizeRequestOutcome::BoundaryAlreadyWon { .. } => {
                unreachable!("explicit teardown has no start precondition")
            }
        }
    }

    pub(crate) fn teardown_with_actual<F>(
        &self,
        scope: PlayerScope,
        clear_queue: F,
    ) -> ActualTeardownOutcome
    where
        F: FnOnce(),
    {
        match self.finalize_request(
            scope,
            TerminalWinner::ExplicitTeardown,
            false,
            true,
            clear_queue,
        ) {
            FinalizeRequestOutcome::Terminal(outcome) => ActualTeardownOutcome::Complete(outcome),
            FinalizeRequestOutcome::TerminalPending => ActualTeardownOutcome::FinalizationPending,
            FinalizeRequestOutcome::BoundaryAlreadyWon { .. } => {
                unreachable!("explicit teardown has no start precondition")
            }
        }
    }

    pub(crate) fn wait_for_claimed_terminal(&self, scope: PlayerScope) -> TerminalOutcome {
        let mut owner = self.shared.state.lock();
        while matches!(
            owner
                .current
                .as_ref()
                .filter(|state| state.scope == scope)
                .map(|state| state.terminal_state),
            Some(TerminalState::Finalizing { .. })
        ) {
            self.shared.finalized.wait(&mut owner);
        }
        let outcome = finalized_outcome(&owner, scope, false);
        let state = owner
            .current
            .as_mut()
            .filter(|state| state.scope == scope)
            .expect("claimed terminal waiter keeps its scope installed");
        debug_assert!(state.terminal_waiters > 0);
        state.terminal_waiters = state.terminal_waiters.saturating_sub(1);
        self.shared.finalized.notify_all();
        outcome
    }

    /// Abort before the scheduled boundary using the shared terminal path.
    pub fn abort_before_start(&self, scope: PlayerScope) -> PreStartAbortOutcome {
        self.abort_before_start_with_actual(scope, || {})
    }

    pub(crate) fn abort_before_start_with_actual<F>(
        &self,
        scope: PlayerScope,
        clear_armed_queue: F,
    ) -> PreStartAbortOutcome
    where
        F: FnOnce(),
    {
        match self.finalize_request(
            scope,
            TerminalWinner::PreStartAbort,
            true,
            true,
            clear_armed_queue,
        ) {
            FinalizeRequestOutcome::Terminal(TerminalOutcome::Won(value)) => {
                PreStartAbortOutcome::Won(value)
            }
            FinalizeRequestOutcome::Terminal(TerminalOutcome::Lost(value)) => {
                PreStartAbortOutcome::Lost(value)
            }
            FinalizeRequestOutcome::Terminal(TerminalOutcome::AlreadyFinalized(value)) => {
                PreStartAbortOutcome::AlreadyFinalized(value)
            }
            FinalizeRequestOutcome::Terminal(TerminalOutcome::StaleScope) => {
                PreStartAbortOutcome::StaleScope
            }
            FinalizeRequestOutcome::TerminalPending => {
                unreachable!("pre-start abort waits for existing finalization")
            }
            FinalizeRequestOutcome::BoundaryAlreadyWon { start_at_zone_us } => {
                PreStartAbortOutcome::BoundaryAlreadyWon { start_at_zone_us }
            }
        }
    }
}

fn finalized_outcome(owner: &OwnerState, scope: PlayerScope, already: bool) -> TerminalOutcome {
    let value = owner
        .current
        .as_ref()
        .filter(|state| state.scope == scope)
        .and_then(|state| match state.terminal_state {
            TerminalState::Finalized { winner, ack } => Some(TerminalFinalization { winner, ack }),
            _ => None,
        });
    let Some(value) = value else {
        return TerminalOutcome::StaleScope;
    };
    if already {
        TerminalOutcome::AlreadyFinalized(value)
    } else {
        TerminalOutcome::Lost(value)
    }
}

fn saturating_increment(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_add(1))
    });
}

struct TimestampTelemetryWriter<'a> {
    state: &'a AtomicU64,
}

impl Drop for TimestampTelemetryWriter<'_> {
    fn drop(&mut self) {
        let _ = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                let flags = state & TIMESTAMP_TELEMETRY_FLAGS;
                let version = state & TIMESTAMP_TELEMETRY_VERSION_MASK;
                Some(flags | (version.wrapping_add(1) & TIMESTAMP_TELEMETRY_VERSION_MASK))
            });
    }
}

pub(crate) struct RendererCallbackPermit<'a> {
    owner: MutexGuard<'a, OwnerState>,
    scope: PlayerScope,
}

impl RendererCallbackPermit<'_> {
    pub(crate) fn start_state(&self) -> StartState {
        self.owner
            .current
            .as_ref()
            .expect("callback permit keeps current scope installed")
            .start_state
    }

    pub(crate) fn scheduled_start(
        &mut self,
        presentation_zone_us: Option<i64>,
    ) -> ScheduledStartOutcome {
        let state = self
            .owner
            .current
            .as_mut()
            .expect("callback permit keeps current scope installed");
        debug_assert_eq!(state.scope, self.scope);
        state.scheduled_start(presentation_zone_us)
    }

    pub(crate) fn record_actual_progress(
        &mut self,
        consumed_frames: usize,
        boundary_zone_us: Option<i64>,
        queued_frames: usize,
        queued_buffers: usize,
    ) {
        let state = self
            .owner
            .current
            .as_mut()
            .expect("callback permit keeps current scope installed");
        debug_assert_eq!(state.scope, self.scope);
        state.consumed_frames = state.consumed_frames.saturating_add(consumed_frames as u64);
        if let Some(boundary) = boundary_zone_us {
            state.record_trusted_boundary(boundary);
        }
        state.queued_frames = queued_frames;
        state.queued_buffers = queued_buffers;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OwnedTerminalResource, PreStartAbortOutcome, RendererOperationOutcome, RendererOwner,
        RendererQueueLimits, ScheduledArmOutcome, ScheduledStartOutcome, ScopeMintError,
        StartState, TerminalOutcome, TIMESTAMP_TELEMETRY_CLOSING, TIMESTAMP_TELEMETRY_VERSION_MASK,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
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

    struct CallbackResource {
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl Drop for CallbackResource {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Some(handle) = self.handle.take() {
                handle.join().unwrap();
            }
        }
    }

    fn owner_and_scope() -> (RendererOwner, super::PlayerScope) {
        let owner = RendererOwner::new(RendererQueueLimits::new(8, 2, 4).unwrap());
        let scope = owner.mint_scope().unwrap();
        (owner, scope)
    }

    fn attach_probe(owner: &RendererOwner, scope: super::PlayerScope, drops: Arc<AtomicUsize>) {
        assert_eq!(
            owner.attach_owned_resource(
                scope,
                OwnedTerminalResource::Test(Box::new(DropProbe(drops))),
            ),
            RendererOperationOutcome::Applied
        );
    }

    fn wait_until(mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for deterministic concurrency checkpoint"
            );
            thread::yield_now();
        }
    }

    #[test]
    fn scope_exhaustion_is_typed_and_does_not_replace_current_scope() {
        let owner = RendererOwner::new(RendererQueueLimits::new(8, 2, 4).unwrap());
        let current = owner.mint_scope().unwrap();
        let _ = owner.teardown(current);
        owner.shared.state.lock().next_scope = u64::MAX;

        assert_eq!(owner.mint_scope(), Err(ScopeMintError::Exhausted));
        assert_eq!(owner.health(current).unwrap().scope(), current);
    }

    #[test]
    fn terminal_race_teardown_and_pre_start_abort_share_ack() {
        let (owner, scope) = owner_and_scope();
        let owner = Arc::new(owner);
        let drops = Arc::new(AtomicUsize::new(0));
        attach_probe(&owner, scope, Arc::clone(&drops));
        let barrier = Arc::new(Barrier::new(3));

        let teardown = {
            let owner = Arc::clone(&owner);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                owner.teardown(scope)
            })
        };
        let abort = {
            let owner = Arc::clone(&owner);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                owner.abort_before_start(scope)
            })
        };
        barrier.wait();
        let a = teardown.join().unwrap();
        let b = abort.join().unwrap();

        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let (won, observed) = match (a, b) {
            (TerminalOutcome::Won(won), PreStartAbortOutcome::Lost(observed))
            | (TerminalOutcome::Won(won), PreStartAbortOutcome::AlreadyFinalized(observed)) => {
                (won, observed)
            }
            (TerminalOutcome::Lost(observed), PreStartAbortOutcome::Won(won))
            | (TerminalOutcome::AlreadyFinalized(observed), PreStartAbortOutcome::Won(won)) => {
                (won, observed)
            }
            outcomes => panic!("one winner and one observer required: {outcomes:?}"),
        };
        assert_eq!(won, observed);
        assert!(won.ack.stream_released());
        assert!(won.ack.callback_stopped());
    }

    #[test]
    fn repeated_teardown_reuses_ack_without_second_drop() {
        let (owner, scope) = owner_and_scope();
        let drops = Arc::new(AtomicUsize::new(0));
        attach_probe(&owner, scope, Arc::clone(&drops));
        let first = owner.teardown(scope);
        let second = owner.teardown(scope);
        let TerminalOutcome::Won(first) = first else {
            panic!("first must win")
        };
        let TerminalOutcome::AlreadyFinalized(second) = second else {
            panic!("repeat must observe finalized ack")
        };
        assert_eq!(first, second);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn concurrent_reopen_waits_for_old_scope_final_ack() {
        let (owner, old_scope) = owner_and_scope();
        let owner = Arc::new(owner);
        let drops = Arc::new(AtomicUsize::new(0));
        attach_probe(&owner, old_scope, Arc::clone(&drops));
        let barrier = Arc::new(Barrier::new(3));

        let mint = {
            let owner = Arc::clone(&owner);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                owner.mint_scope().unwrap()
            })
        };
        let teardown = {
            let owner = Arc::clone(&owner);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                owner.teardown(old_scope)
            })
        };
        barrier.wait();
        let finalization = teardown.join().unwrap();
        let new_scope = mint.join().unwrap();

        assert!(matches!(finalization, TerminalOutcome::Won(_)));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_ne!(old_scope, new_scope);
        assert_eq!(owner.health(new_scope).unwrap().scope(), new_scope);
        assert_eq!(
            owner.health(old_scope),
            Err(RendererOperationOutcome::StaleScope)
        );
    }

    #[test]
    fn waiting_loser_reads_original_ack_before_multi_generation_reopen() {
        let (owner, old_scope) = owner_and_scope();
        let owner = Arc::new(owner);
        let (drop_started_tx, drop_started_rx) = mpsc::channel();
        let (release_drop_tx, release_drop_rx) = mpsc::channel();
        assert_eq!(
            owner.attach_owned_resource(
                old_scope,
                OwnedTerminalResource::Test(Box::new(BlockingDrop {
                    started: drop_started_tx,
                    release: release_drop_rx,
                })),
            ),
            RendererOperationOutcome::Applied
        );

        let winner = {
            let owner = Arc::clone(&owner);
            thread::spawn(move || owner.teardown(old_scope))
        };
        drop_started_rx.recv().unwrap();
        let loser = {
            let owner = Arc::clone(&owner);
            thread::spawn(move || owner.abort_before_start(old_scope))
        };
        while owner
            .shared
            .state
            .lock()
            .current
            .as_ref()
            .is_none_or(|scope| scope.terminal_waiters == 0)
        {
            thread::yield_now();
        }
        let reopen = {
            let owner = Arc::clone(&owner);
            thread::spawn(move || owner.mint_scope().unwrap())
        };

        release_drop_tx.send(()).unwrap();
        let TerminalOutcome::Won(won) = winner.join().unwrap() else {
            panic!("teardown must own the blocked resource")
        };
        let PreStartAbortOutcome::Lost(observed) = loser.join().unwrap() else {
            panic!("registered waiter must observe lost with shared ack")
        };
        assert_eq!(won, observed);
        let second_scope = reopen.join().unwrap();
        assert_ne!(old_scope, second_scope);

        let _ = owner.teardown(second_scope);
        let third_scope = owner.mint_scope().unwrap();
        assert_ne!(second_scope, third_scope);
    }

    #[test]
    fn terminal_ack_is_published_after_callback_resource_joins() {
        let (owner, scope) = owner_and_scope();
        let stop = Arc::new(AtomicBool::new(false));
        let callbacks = Arc::new(AtomicUsize::new(0));
        let handle = {
            let stop = Arc::clone(&stop);
            let callbacks = Arc::clone(&callbacks);
            thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    callbacks.fetch_add(1, Ordering::Relaxed);
                    thread::yield_now();
                }
            })
        };
        while callbacks.load(Ordering::Relaxed) == 0 {
            thread::yield_now();
        }
        assert_eq!(
            owner.attach_owned_resource(
                scope,
                OwnedTerminalResource::Test(Box::new(CallbackResource {
                    stop,
                    handle: Some(handle),
                })),
            ),
            RendererOperationOutcome::Applied
        );

        let TerminalOutcome::Won(finalization) = owner.teardown(scope) else {
            panic!("first teardown must win")
        };
        assert!(finalization.ack.callback_stopped());
        let after_ack = callbacks.load(Ordering::Relaxed);
        for _ in 0..100 {
            thread::yield_now();
        }
        assert_eq!(callbacks.load(Ordering::Relaxed), after_ack);
    }

    #[test]
    fn callback_heartbeat_remains_non_blocking_during_owner_contention() {
        let (owner, scope) = owner_and_scope();
        let owner_guard = owner.shared.state.lock();

        assert!(owner.try_callback_heartbeat(scope));
        owner.record_callback_underrun(7);
        drop(owner_guard);

        let health = owner.health(scope).unwrap();
        assert_eq!(health.callback_count(), 1);
        assert_eq!(health.underrun_frames(), 7);
    }

    #[test]
    fn scheduled_start_success_health_and_explicit_teardown() {
        let (owner, scope) = owner_and_scope();
        let stop = Arc::new(AtomicBool::new(false));
        let callbacks = Arc::new(AtomicUsize::new(0));
        let handle = {
            let stop = Arc::clone(&stop);
            let callbacks = Arc::clone(&callbacks);
            thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    callbacks.fetch_add(1, Ordering::Relaxed);
                    thread::yield_now();
                }
            })
        };
        while callbacks.load(Ordering::Relaxed) == 0 {
            thread::yield_now();
        }
        assert_eq!(
            owner.attach_owned_resource(
                scope,
                OwnedTerminalResource::Test(Box::new(CallbackResource {
                    stop,
                    handle: Some(handle),
                })),
            ),
            RendererOperationOutcome::Applied
        );
        assert_eq!(
            owner.arm_scheduled_start(scope, 55_000),
            ScheduledArmOutcome::Armed
        );
        assert!(matches!(
            owner.enqueue(scope, 2),
            super::EnqueueOutcome::Accepted { .. }
        ));
        assert!(owner.try_callback_heartbeat(scope));
        let mut permit = owner.try_callback_permit(scope).unwrap();
        assert_eq!(
            permit.scheduled_start(Some(55_000)),
            ScheduledStartOutcome::BoundaryWon {
                start_at_zone_us: 55_000,
            }
        );
        permit.record_actual_progress(1, None, 1, 1);
        drop(permit);

        let health = owner.health(scope).unwrap();
        assert_eq!(health.callback_count(), 1);
        assert_eq!(health.consumed_frames(), 1);
        assert_eq!(health.last_presentation_boundary_zone_us(), Some(55_000));
        assert_eq!(
            owner.start_state(scope).unwrap(),
            StartState::BoundaryWon {
                start_at_zone_us: 55_000,
            }
        );

        let TerminalOutcome::Won(finalization) = owner.teardown(scope) else {
            panic!("explicit teardown must win after scheduled success")
        };
        assert!(finalization.ack.stream_released());
        assert!(finalization.ack.callback_stopped());
        let after_ack = callbacks.load(Ordering::Relaxed);
        for _ in 0..100 {
            thread::yield_now();
        }
        assert_eq!(callbacks.load(Ordering::Relaxed), after_ack);
    }

    #[test]
    fn boundary_evidence_cannot_be_overwritten_after_callback_wins() {
        let (owner, scope) = owner_and_scope();
        assert_eq!(
            owner.arm_scheduled_start(scope, 55_000),
            ScheduledArmOutcome::Armed
        );
        let mut permit = owner.try_callback_permit(scope).unwrap();
        assert_eq!(
            permit.scheduled_start(Some(55_000)),
            ScheduledStartOutcome::BoundaryWon {
                start_at_zone_us: 55_000,
            }
        );
        permit.record_actual_progress(0, Some(99_000), 0, 0);
        drop(permit);
        assert_eq!(
            owner.record_callback(scope, 0, 0, Some(101_000)),
            RendererOperationOutcome::Applied
        );
        assert_eq!(
            owner
                .health(scope)
                .unwrap()
                .last_presentation_boundary_zone_us(),
            Some(55_000)
        );
    }

    #[test]
    fn scheduled_success_then_fresh_scope_abort() {
        let (owner, success_scope) = owner_and_scope();
        let drops = Arc::new(AtomicUsize::new(0));
        attach_probe(&owner, success_scope, Arc::clone(&drops));
        assert!(matches!(
            owner.enqueue(success_scope, 2),
            super::EnqueueOutcome::Accepted { .. }
        ));
        assert_eq!(
            owner.arm_scheduled_start(success_scope, 55_000),
            ScheduledArmOutcome::Armed
        );
        assert!(owner.try_callback_heartbeat(success_scope));
        let mut permit = owner.try_callback_permit(success_scope).unwrap();
        assert_eq!(
            permit.scheduled_start(Some(55_000)),
            ScheduledStartOutcome::BoundaryWon {
                start_at_zone_us: 55_000,
            }
        );
        permit.record_actual_progress(1, None, 1, 1);
        drop(permit);
        let health = owner.health(success_scope).unwrap();
        assert_eq!(health.callback_count(), 1);
        assert_eq!(health.consumed_frames(), 1);
        assert_eq!(health.last_presentation_boundary_zone_us(), Some(55_000));
        let TerminalOutcome::Won(success) = owner.teardown(success_scope) else {
            panic!("scheduled-success teardown must win")
        };
        assert!(success.ack.stream_released());
        assert!(success.ack.callback_stopped());
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let abort_scope = owner.mint_scope().expect("success resources released");
        assert_ne!(success_scope, abort_scope);
        attach_probe(&owner, abort_scope, Arc::clone(&drops));
        assert!(matches!(
            owner.enqueue(abort_scope, 4),
            super::EnqueueOutcome::Accepted { .. }
        ));
        assert_eq!(
            owner.arm_scheduled_start(abort_scope, 90_000),
            ScheduledArmOutcome::Armed
        );
        let PreStartAbortOutcome::Won(abort) = owner.abort_before_start(abort_scope) else {
            panic!("fresh pre-start abort must win")
        };
        assert!(abort.ack.stream_released());
        assert!(abort.ack.callback_stopped());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        let capacity = owner.capacity(abort_scope).unwrap();
        assert_eq!(capacity.current_frames(), 0);
        assert_eq!(capacity.current_buffers(), 0);
    }

    #[test]
    fn pre_start_abort_drops_resource_clears_armed_and_reuses_ack() {
        let (owner, scope) = owner_and_scope();
        let drops = Arc::new(AtomicUsize::new(0));
        attach_probe(&owner, scope, Arc::clone(&drops));
        assert_eq!(
            owner.arm_scheduled_start(scope, 90_000),
            ScheduledArmOutcome::Armed
        );
        assert!(matches!(
            owner.enqueue(scope, 4),
            super::EnqueueOutcome::Accepted { .. }
        ));

        let PreStartAbortOutcome::Won(first) = owner.abort_before_start(scope) else {
            panic!("pre-start abort must win")
        };
        let PreStartAbortOutcome::AlreadyFinalized(second) = owner.abort_before_start(scope) else {
            panic!("repeat abort must reuse final ack")
        };
        assert_eq!(first, second);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let capacity = owner.capacity(scope).unwrap();
        assert_eq!(capacity.current_frames(), 0);
        assert_eq!(capacity.current_buffers(), 0);
        assert_eq!(owner.start_state(scope).unwrap(), StartState::Idle);
    }

    #[test]
    fn timestamp_provenance_counts_each_source_and_detects_non_monotonic() {
        let (owner, scope) = owner_and_scope();
        assert!(owner.try_callback_telemetry(
            scope,
            cpal::OutputTimestampSource::DevicePresentation,
            false,
        ));
        assert!(owner.try_callback_telemetry(
            scope,
            cpal::OutputTimestampSource::MonotonicFallback,
            true,
        ));
        assert!(owner.try_callback_telemetry(
            scope,
            cpal::OutputTimestampSource::Unspecified,
            true,
        ));

        let health = owner.health(scope).unwrap();
        let timestamps = health.output_timestamps();
        assert_eq!(timestamps.device_presentation(), 1);
        assert_eq!(timestamps.monotonic_fallback(), 1);
        assert_eq!(timestamps.unspecified(), 1);
        assert_eq!(timestamps.monotonic_violations(), 2);
    }

    #[test]
    fn timestamp_provenance_sum_matches_provenance_count() {
        let (owner, scope) = owner_and_scope();
        for source in [
            cpal::OutputTimestampSource::DevicePresentation,
            cpal::OutputTimestampSource::DevicePresentation,
            cpal::OutputTimestampSource::MonotonicFallback,
            cpal::OutputTimestampSource::Unspecified,
        ] {
            assert!(owner.try_callback_telemetry(scope, source, false));
            let health = owner.health(scope).unwrap();
            let timestamps = health.output_timestamps();
            assert_eq!(
                timestamps.device_presentation()
                    + timestamps.monotonic_fallback()
                    + timestamps.unspecified(),
                timestamps.provenance_callback_count()
            );
        }

        assert_eq!(owner.close(scope), RendererOperationOutcome::Applied);
        let before = owner.health(scope).unwrap();
        assert!(!owner.try_callback_telemetry(
            scope,
            cpal::OutputTimestampSource::DevicePresentation,
            false,
        ));
        assert_eq!(owner.health(scope).unwrap(), before);
    }

    #[test]
    fn timestamp_provenance_is_scope_fenced_and_resets_on_reopen() {
        let (owner, first) = owner_and_scope();
        assert!(owner.try_callback_telemetry(
            first,
            cpal::OutputTimestampSource::MonotonicFallback,
            true,
        ));
        let TerminalOutcome::Won(_) = owner.teardown(first) else {
            panic!("first scope teardown must win")
        };
        let second = owner.mint_scope().unwrap();
        assert!(!owner.try_callback_telemetry(
            first,
            cpal::OutputTimestampSource::DevicePresentation,
            false,
        ));
        let health = owner.health(second).unwrap();
        assert_eq!(health.callback_count(), 0);
        assert_eq!(health.output_timestamps().device_presentation(), 0);
        assert_eq!(health.output_timestamps().monotonic_fallback(), 0);
        assert_eq!(health.output_timestamps().unspecified(), 0);
        assert_eq!(health.output_timestamps().monotonic_violations(), 0);
    }

    #[test]
    fn timestamp_provenance_stops_at_terminal_ack() {
        let (owner, scope) = owner_and_scope();
        let drops = Arc::new(AtomicUsize::new(0));
        attach_probe(&owner, scope, Arc::clone(&drops));
        assert!(owner.try_callback_telemetry(
            scope,
            cpal::OutputTimestampSource::DevicePresentation,
            false,
        ));
        let TerminalOutcome::Won(finalization) = owner.teardown(scope) else {
            panic!("teardown must win")
        };
        assert!(finalization.ack.callback_stopped());
        assert!(finalization.ack.stream_released());
        let final_health = owner.health(scope).unwrap();
        assert!(!owner.try_callback_telemetry(
            scope,
            cpal::OutputTimestampSource::MonotonicFallback,
            true,
        ));
        assert_eq!(owner.health(scope).unwrap(), final_health);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn timestamp_provenance_close_waits_for_inflight_commit() {
        let (owner, scope) = owner_and_scope();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let writer = {
            let owner = owner.clone();
            thread::spawn(move || {
                owner.try_callback_telemetry_with(
                    scope,
                    cpal::OutputTimestampSource::DevicePresentation,
                    false,
                    || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                    },
                )
            })
        };
        entered_rx.recv().unwrap();

        let (closed_tx, closed_rx) = mpsc::channel();
        let closer = {
            let owner = owner.clone();
            thread::spawn(move || {
                let outcome = owner.close(scope);
                closed_tx.send(outcome).unwrap();
            })
        };
        wait_until(|| {
            let state = owner
                .shared
                .timestamp_telemetry_state
                .load(Ordering::Acquire);
            state & TIMESTAMP_TELEMETRY_CLOSING != 0
                && state & TIMESTAMP_TELEMETRY_VERSION_MASK & 1 != 0
        });
        assert_eq!(closed_rx.try_recv(), Err(mpsc::TryRecvError::Empty));

        release_tx.send(()).unwrap();
        assert!(writer.join().unwrap());
        assert_eq!(closed_rx.recv().unwrap(), RendererOperationOutcome::Applied);
        closer.join().unwrap();

        let final_health = owner.health(scope).unwrap();
        assert_eq!(final_health.callback_count(), 1);
        assert_eq!(
            final_health.output_timestamps().provenance_callback_count(),
            1
        );
        for _ in 0..100 {
            thread::yield_now();
        }
        assert_eq!(owner.health(scope).unwrap(), final_health);
    }

    #[test]
    fn diagnostic_health_skips_inflight_telemetry_without_holding_renderer() {
        let (owner, scope) = owner_and_scope();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let writer = {
            let owner = owner.clone();
            thread::spawn(move || {
                owner.try_callback_telemetry_with(
                    scope,
                    cpal::OutputTimestampSource::DevicePresentation,
                    false,
                    || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                    },
                )
            })
        };
        entered_rx.recv().unwrap();
        let (tx, rx) = mpsc::channel();
        let reader = {
            let owner = owner.clone();
            thread::spawn(move || tx.send(owner.try_health(scope)).unwrap())
        };
        let observed = rx.recv_timeout(std::time::Duration::from_secs(1));
        let renderer_available = owner.try_callback_permit(scope).is_some();
        release_tx.send(()).unwrap();
        assert!(writer.join().unwrap());
        reader.join().unwrap();
        assert_eq!(
            observed
                .expect("diagnostic read must not wait for telemetry")
                .unwrap(),
            None
        );
        assert!(renderer_available);
        let health = owner.try_health(scope).unwrap().unwrap();
        assert_eq!(health.callback_count(), 1);
        assert_eq!(health.output_timestamps().device_presentation(), 1);
    }

    #[test]
    fn timestamp_provenance_health_waits_for_inflight_commit() {
        let (owner, scope) = owner_and_scope();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let writer = {
            let owner = owner.clone();
            thread::spawn(move || {
                owner.try_callback_telemetry_with(
                    scope,
                    cpal::OutputTimestampSource::DevicePresentation,
                    true,
                    || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                    },
                )
            })
        };
        entered_rx.recv().unwrap();

        let (health_tx, health_rx) = mpsc::channel();
        let reader = {
            let owner = owner.clone();
            thread::spawn(move || {
                health_tx.send(owner.health(scope).unwrap()).unwrap();
            })
        };
        wait_until(|| {
            owner
                .shared
                .timestamp_snapshot_retries
                .load(Ordering::Acquire)
                > 0
        });
        assert_eq!(health_rx.try_recv(), Err(mpsc::TryRecvError::Empty));

        release_tx.send(()).unwrap();
        assert!(writer.join().unwrap());
        let health = health_rx.recv().unwrap();
        reader.join().unwrap();
        assert_eq!(health.callback_count(), 1);
        assert_eq!(health.output_timestamps().device_presentation(), 1);
        assert_eq!(health.output_timestamps().monotonic_violations(), 1);
        assert_eq!(
            health.output_timestamps().provenance_callback_count(),
            health.callback_count()
        );
    }

    #[test]
    fn timestamp_provenance_terminal_ack_waits_for_inflight_commit() {
        let (owner, scope) = owner_and_scope();
        let drops = Arc::new(AtomicUsize::new(0));
        attach_probe(&owner, scope, Arc::clone(&drops));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let writer = {
            let owner = owner.clone();
            thread::spawn(move || {
                owner.try_callback_telemetry_with(
                    scope,
                    cpal::OutputTimestampSource::MonotonicFallback,
                    true,
                    || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                    },
                )
            })
        };
        entered_rx.recv().unwrap();

        let (terminal_tx, terminal_rx) = mpsc::channel();
        let finalizer = {
            let owner = owner.clone();
            thread::spawn(move || {
                terminal_tx.send(owner.teardown(scope)).unwrap();
            })
        };
        wait_until(|| {
            let state = owner
                .shared
                .timestamp_telemetry_state
                .load(Ordering::Acquire);
            state & TIMESTAMP_TELEMETRY_CLOSING != 0
                && state & TIMESTAMP_TELEMETRY_VERSION_MASK & 1 != 0
        });
        assert_eq!(terminal_rx.try_recv(), Err(mpsc::TryRecvError::Empty));

        release_tx.send(()).unwrap();
        assert!(writer.join().unwrap());
        let TerminalOutcome::Won(finalization) = terminal_rx.recv().unwrap() else {
            panic!("teardown must win")
        };
        assert!(finalization.ack.callback_stopped());
        assert!(finalization.ack.stream_released());
        finalizer.join().unwrap();

        let final_health = owner.health(scope).unwrap();
        let timestamps = final_health.output_timestamps();
        assert_eq!(timestamps.monotonic_fallback(), 1);
        assert_eq!(timestamps.monotonic_violations(), 1);
        for _ in 0..100 {
            thread::yield_now();
        }
        assert_eq!(owner.health(scope).unwrap(), final_health);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

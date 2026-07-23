// ABOUTME: Audio types and processing for sendspin-rs
// ABOUTME: Contains Sample type, AudioFormat, Buffer, and codec definitions

/// Audio decoder implementations (PCM, Opus, FLAC)
pub mod decode;
/// Lock-free volume/mute control
pub mod gain;
/// Scope-fenced bounded renderer and terminal lifecycle contracts
pub mod player_contract;
/// Buffer pool for reusing audio sample buffers
pub mod pool;
/// Sync correction planner for drop/insert cadence
pub mod sync_correction;
/// Synced playback helper using output timestamps
pub mod synced_player;
/// Core audio type definitions (Sample, Codec, AudioFormat, AudioBuffer)
pub mod types;

pub use gain::GainControl;
pub use player_contract::{
    EnqueueOutcome, OpenError, OutputBackendError, PlayerScope, PreStartAbortOutcome,
    RendererCapacitySnapshot, RendererFault, RendererHealthSnapshot, RendererOperationOutcome,
    RendererOwner, RendererQueueLimits, RendererQueueLimitsError, RendererTerminal,
    ScheduledArmOutcome, ScheduledStartOutcome, ScopeMintError, StartState, TerminalAck,
    TerminalFinalization, TerminalOutcome, TerminalState, TerminalWinner,
};
pub use pool::BufferPool;
pub use sync_correction::{CorrectionPlanner, CorrectionSchedule};
pub use synced_player::{
    AudioBufferLifetime, DeviceDelayError, DeviceDelayMs, ProcessCallback, ReanchorRequired,
    SyncedPlayer, SyncedPlayerConfig,
};
pub use types::{AudioBuffer, AudioFormat, Codec};

//! Source provenance and per-presented-frame age policy. No PCM is retained here.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Oldest contributing source time, in the same microsecond clock domain as
/// [`super::AudioBuffer::timestamp`]. Unknown contributors must use `Unknown`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SourceFrameTime {
    /// At least one contributing source has no trustworthy source timestamp.
    #[default]
    Unknown,
    /// `linear_start_us` belongs to the buffer's first media frame and advances
    /// by the media offset. `fixed_oldest_us` contributes to every frame. When
    /// both are present the older time wins; neither present means unknown.
    Known {
        /// Source time of the first PCM frame, advancing with its media offset.
        linear_start_us: Option<i64>,
        /// Earliest source time contributing throughout this PCM buffer.
        fixed_oldest_us: Option<i64>,
    },
}

impl SourceFrameTime {
    pub(super) fn at_offset(self, media_offset_us: i64) -> Option<i64> {
        match self {
            Self::Unknown => None,
            Self::Known {
                linear_start_us,
                fixed_oldest_us,
            } => {
                let linear = match linear_start_us {
                    Some(start) => Some(start.checked_add(media_offset_us)?),
                    None => None,
                };
                match (linear, fixed_oldest_us) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                }
            }
        }
    }
}

/// Inline in the existing publication Arc. Revision fencing makes a setter
/// immediately observable without allowing an older callback to clear its mute.
pub(super) struct Control {
    config: AtomicU64,
    status: AtomicU64,
    live_until_us: AtomicU64,
    origin: Instant,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            config: AtomicU64::new(0),
            status: AtomicU64::new(1),
            live_until_us: AtomicU64::new(0),
            origin: Instant::now(),
        }
    }
}

impl Control {
    pub(super) fn set_limit(&self, limit: Option<NonZeroU32>) {
        let limit = u64::from(limit.map_or(0, NonZeroU32::get));
        let _ = self
            .config
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                if old as u32 == limit as u32 {
                    None
                } else {
                    Some((old.wrapping_add(1 << 32) & !u64::from(u32::MAX)) | limit)
                }
            });
    }

    pub(super) fn config(&self) -> u64 {
        self.config.load(Ordering::Acquire)
    }

    pub(super) fn muted(&self) -> bool {
        let config = self.config();
        if config as u32 == 0 {
            return false;
        }
        let status = self.status.load(Ordering::Acquire);
        status >> 32 != config >> 32
            || status & 1 != 0
            || self.origin.elapsed().as_micros()
                >= u128::from(self.live_until_us.load(Ordering::Relaxed))
    }

    pub(super) fn publish(&self, config: u64, muted: bool, end: Instant) {
        // A stopped device must not leave an indefinitely healthy observation.
        let until = end.saturating_duration_since(self.origin).as_micros();
        self.live_until_us
            .store(until.min(u128::from(u64::MAX)) as u64, Ordering::Relaxed);
        self.status.store(
            (config & !u64::from(u32::MAX)) | u64::from(muted),
            Ordering::Release,
        );
    }
}

/// Device-owned hysteresis, counted in actual presented frames across callback
/// and scratch-block boundaries. Two complete 20 ms healthy blocks restore PCM.
#[derive(Default)]
pub(super) struct Gate {
    config: u64,
    muted: bool,
    healthy_frames: u64,
}

impl Gate {
    pub(super) fn configure(&mut self, config: u64) {
        if self.config != config {
            self.config = config;
            self.unavailable();
        }
    }

    pub(super) fn unavailable(&mut self) {
        self.muted = self.config as u32 != 0;
        self.healthy_frames = 0;
    }

    pub(super) fn muted(&self) -> bool {
        self.muted
    }

    pub(super) fn enabled(&self) -> bool {
        self.config as u32 != 0
    }

    pub(super) fn allow(
        &mut self,
        source: Option<i64>,
        presentation: Option<i64>,
        rate: u32,
    ) -> bool {
        let limit_us = i64::from(self.config as u32) * 1_000;
        if limit_us == 0 {
            return true;
        }
        let age = source
            .zip(presentation)
            .and_then(|(source, now)| now.checked_sub(source));
        if age.is_none_or(|age| age < 0 || age > limit_us) {
            self.unavailable();
            return false;
        }
        if self.muted {
            let margin = (limit_us / 8).min(20_000);
            if age.is_some_and(|age| age <= limit_us - margin) {
                self.healthy_frames = self.healthy_frames.saturating_add(1);
                if self.healthy_frames >= (u64::from(rate) * 40).div_ceil(1_000) {
                    self.muted = false;
                }
            } else {
                self.healthy_frames = 0;
            }
        }
        !self.muted
    }
}

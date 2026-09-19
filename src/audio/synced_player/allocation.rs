//! Checked construction budget for player-owned allocation layouts.
//! This is neither an RSS estimate nor a new public admission limit. Runtime
//! supplies complete inline roots once, then their independent heap allocations.
//! Codec-header capacities and caller-owned opaque resources remain separate.

use super::preparation::{Frame, Window};
use super::transport::Allocation;
use super::QueuedAudioBuffer;
use crate::audio::RendererQueueLimits;
use rtrb::RingBuffer;
use std::alloc::Layout;
use std::sync::atomic::AtomicUsize;

// Mirrors the current toolchain alloc::sync::ArcInner header and its
// arcinner_layout_for_value_layout (repr(C, align(2)), two atomic counts,
// Layout::extend(...).pad_to_align()). Budget only; never used to access an Arc.
#[repr(C, align(2))]
struct ArcHeader {
    strong: AtomicUsize,
    weak: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Budget {
    bytes: usize,
}

impl Budget {
    pub(super) fn bytes(self) -> usize {
        self.bytes
    }

    pub(super) fn add_bytes(&mut self, bytes: usize) -> Option<()> {
        self.bytes = self.bytes.checked_add(bytes)?;
        Some(())
    }

    pub(super) fn add_inline<T>(&mut self) -> Option<()> {
        self.add_bytes(Layout::new::<T>().size())
    }

    pub(super) fn add_array<T>(&mut self, capacity: usize) -> Option<()> {
        self.add_bytes(Layout::array::<T>(capacity).ok()?.size())
    }

    pub(super) fn add_arc<T>(&mut self) -> Option<()> {
        self.add_arc_layout(Layout::new::<T>())
    }

    pub(super) fn add_arc_layout(&mut self, value: Layout) -> Option<()> {
        let layout = Layout::new::<ArcHeader>()
            .extend(value)
            .ok()?
            .0
            .pad_to_align();
        self.add_bytes(layout.size())
    }

    /// Canonical/private source containers plus at most one accepted input.
    /// Include the full current chunk prefix, not only the remaining frame quota.
    /// Call first with requested O capacities, then with the actual reservations.
    pub(super) fn sources(
        limits: RendererQueueLimits,
        channels: usize,
        canonical_capacity: usize,
        private_capacity: usize,
    ) -> Option<Self> {
        if channels == 0
            || canonical_capacity < limits.hard_buffers()
            || private_capacity < limits.hard_buffers()
        {
            return None;
        }
        let one_view = limits
            .hard_frames()
            .checked_add(limits.max_chunk_frames() - 1)?;
        let frames = one_view
            .checked_mul(2)?
            .checked_add(limits.max_chunk_frames())?;
        let samples = frames.checked_mul(channels)?;
        let input_samples = limits.max_chunk_frames().checked_mul(channels)?;
        // The aggregate represents separate source allocations. Validate one
        // maximum legal allocation without pretending their sum is one array.
        Layout::new::<ArcHeader>()
            .extend(Layout::array::<i32>(input_samples).ok()?)
            .ok()?;
        let mut budget = Self::default();
        budget.add_bytes(samples.checked_mul(std::mem::size_of::<i32>())?)?;
        let objects = limits.hard_buffers().checked_mul(2)?.checked_add(1)?;
        // Variable source lengths can incur different trailing Arc alignment.
        let per_arc_overhead = Layout::new::<ArcHeader>()
            .size()
            .checked_add(Layout::new::<ArcHeader>().align() - 1)?;
        budget.add_bytes(objects.checked_mul(per_arc_overhead)?)?;
        budget.add_array::<QueuedAudioBuffer>(canonical_capacity)?;
        budget.add_array::<QueuedAudioBuffer>(private_capacity)?;
        Some(budget)
    }

    /// PCM pipe endpoints are inline in Worker/Device roots, so count only their
    /// independent free/ring arrays, payloads and shared ring allocations here.
    pub(super) fn add_transport(
        &mut self,
        windows: usize,
        frames: usize,
        channels: usize,
        actual: Option<Allocation>,
    ) -> Option<()> {
        if windows == 0 || frames == 0 || channels == 0 {
            return None;
        }
        windows.checked_mul(2)?;
        let (payload, free, ready, retired) = if let Some(actual) = actual {
            if actual.windows != windows.checked_add(1)?
                || actual.frames_per_window != frames
                || actual.samples_per_window != frames.checked_mul(channels)?
            {
                return None;
            }
            (
                actual.window_payload_bytes,
                actual.free_capacity,
                actual.ready_capacity,
                actual.retired_capacity,
            )
        } else {
            let one = Layout::array::<Frame>(frames).ok()?.size().checked_add(
                Layout::array::<i32>(frames.checked_mul(channels)?)
                    .ok()?
                    .size(),
            )?;
            (
                one.checked_mul(windows.checked_add(1)?)?,
                windows,
                windows,
                windows,
            )
        };
        self.add_bytes(payload)?;
        self.add_array::<Window>(free)?;
        self.add_array::<Window>(ready)?;
        self.add_array::<Window>(retired)?;
        self.add_arc::<RingBuffer<Window>>()?;
        self.add_arc::<RingBuffer<Window>>()?;
        Some(())
    }

    pub(super) fn add_feedback<T>(&mut self, capacity: usize) -> Option<()> {
        if capacity == 0 {
            return None;
        }
        capacity.checked_mul(2)?;
        self.add_array::<T>(capacity)?;
        self.add_arc::<RingBuffer<T>>()
    }
}

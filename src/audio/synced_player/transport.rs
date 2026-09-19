//! Fixed-window SPSC ownership transfer. No source PCM Arc or lifetime attachment
//! travels here: a Window contains only its preallocated output and metadata.
//!
//! N free-window credits bound ready/current/retired ownership in aggregate;
//! the worker has one additional builder. Device retirement only moves a Window.
//! Both endpoints must be destroyed on a non-RT thread after the reader stops.

use super::preparation::{Frame, Window};
use rtrb::{Consumer, Producer, PushError, RingBuffer};
use std::alloc::Layout;
use std::mem::{replace, size_of};

/// Actual allocation dimensions for the final player resource accounting.
/// Array sizes describe requested layouts, not allocator/RSS overhead. Ring Arc
/// control allocations and these endpoints belong in the player's root budget.
#[derive(Clone, Copy, Debug)]
pub(super) struct Allocation {
    pub windows: usize,
    pub frames_per_window: usize,
    pub samples_per_window: usize,
    pub frame_capacity_total: usize,
    pub sample_capacity_total: usize,
    pub ready_capacity: usize,
    pub retired_capacity: usize,
    pub free_capacity: usize,
    pub window_payload_bytes: usize,
    pub ready_slot_bytes: usize,
    pub retired_slot_bytes: usize,
    pub free_slot_bytes: usize,
    pub endpoint_bytes: usize,
    pub ring_control_bytes: usize,
    pub requested_bytes_without_arc_headers: usize,
}

pub(super) struct PrepareSide {
    builder: Window,
    free: Vec<Window>,
    ready: Producer<Window>,
    retired: Consumer<Window>,
    limit: usize,
    allocation: Allocation,
}

pub(super) struct DeviceSide {
    ready: Consumer<Window>,
    retired: Producer<Window>,
    offset: usize,
    pending_return: Option<Window>,
}

fn layout_bytes<T>(count: usize) -> Option<usize> {
    Some(Layout::array::<T>(count).ok()?.size())
}

/// Construction only. Invalid/overflowing dimensions are rejected before any
/// payload allocation; the caller maps failure to its existing OpenError.
/// Rust/rtrb allocation exhaustion follows their ordinary allocation behavior.
pub(super) fn pipe(
    limit: usize,
    frames: usize,
    channels: usize,
) -> Option<(PrepareSide, DeviceSide)> {
    if limit == 0 || frames == 0 || channels == 0 {
        return None;
    }
    // rtrb indices span twice the capacity.
    limit.checked_mul(2)?;
    let windows = limit.checked_add(1)?;
    let samples = frames.checked_mul(channels)?;
    let one_payload = layout_bytes::<Frame>(frames)?.checked_add(layout_bytes::<i32>(samples)?)?;
    let window_payload_bytes = one_payload.checked_mul(windows)?;
    let slots = layout_bytes::<Window>(limit)?;
    let endpoint_bytes = size_of::<PrepareSide>().checked_add(size_of::<DeviceSide>())?;
    let ring_control_bytes = size_of::<RingBuffer<Window>>().checked_mul(2)?;
    // Check the combined request as well as each independently legal array.
    window_payload_bytes
        .checked_add(slots.checked_mul(3)?)?
        .checked_add(endpoint_bytes)?
        .checked_add(ring_control_bytes)?;

    let free = Vec::with_capacity(limit);
    let free_capacity = free.capacity();
    let free_slot_bytes = layout_bytes::<Window>(free_capacity)?;
    let (ready_tx, ready_rx) = RingBuffer::new(limit);
    let (retired_tx, retired_rx) = RingBuffer::new(limit);
    let ready_capacity = ready_tx.buffer().capacity();
    let retired_capacity = retired_tx.buffer().capacity();
    let ready_slot_bytes = layout_bytes::<Window>(ready_capacity)?;
    let retired_slot_bytes = layout_bytes::<Window>(retired_capacity)?;
    let requested_bytes_without_arc_headers = window_payload_bytes
        .checked_add(free_slot_bytes)?
        .checked_add(ready_slot_bytes)?
        .checked_add(retired_slot_bytes)?
        .checked_add(endpoint_bytes)?
        .checked_add(ring_control_bytes)?;
    let allocation = Allocation {
        windows,
        frames_per_window: frames,
        samples_per_window: samples,
        frame_capacity_total: 0,
        sample_capacity_total: 0,
        ready_capacity,
        retired_capacity,
        free_capacity,
        window_payload_bytes,
        ready_slot_bytes,
        retired_slot_bytes,
        free_slot_bytes,
        endpoint_bytes,
        ring_control_bytes,
        requested_bytes_without_arc_headers,
    };
    let mut prepare = PrepareSide {
        builder: Window::new(frames, channels)?,
        free,
        ready: ready_tx,
        retired: retired_rx,
        limit,
        allocation,
    };
    for _ in 0..limit {
        prepare.free.push(Window::new(frames, channels)?);
    }
    // Keep Vec allocations in their original form, rather than reboxing them.
    // All payload lengths remain fixed after construction; account for actual
    // capacity even if an allocator/collection reserves more than requested.
    let mut actual_payload = 0usize;
    for window in std::iter::once(&prepare.builder).chain(&prepare.free) {
        prepare.allocation.frame_capacity_total = prepare
            .allocation
            .frame_capacity_total
            .checked_add(window.frames.capacity())?;
        prepare.allocation.sample_capacity_total = prepare
            .allocation
            .sample_capacity_total
            .checked_add(window.pcm.capacity())?;
        actual_payload = actual_payload
            .checked_add(layout_bytes::<Frame>(window.frames.capacity())?)?
            .checked_add(layout_bytes::<i32>(window.pcm.capacity())?)?;
    }
    prepare.allocation.window_payload_bytes = actual_payload;
    prepare.allocation.requested_bytes_without_arc_headers = actual_payload
        .checked_add(free_slot_bytes)?
        .checked_add(ready_slot_bytes)?
        .checked_add(retired_slot_bytes)?
        .checked_add(endpoint_bytes)?
        .checked_add(ring_control_bytes)?;
    Some((
        prepare,
        DeviceSide {
            ready: ready_rx,
            retired: retired_tx,
            offset: 0,
            pending_return: None,
        },
    ))
}

impl PrepareSide {
    /// Credits owned by this producer; device progress can only add returned
    /// windows until the next reclaim, never remove these available credits.
    pub(super) fn available_credits(&self) -> usize {
        self.free.len()
    }

    pub(super) fn allocation(&self) -> Allocation {
        self.allocation
    }

    /// The sole unpublished window. Refill in place; do not replace its buffers.
    /// Failed publication retains this exact builder for the next worker turn.
    pub(super) fn builder_mut(&mut self) -> &mut Window {
        &mut self.builder
    }

    /// Internal preparation backpressure; never a public enqueue Full result.
    pub(super) fn publish(&mut self) -> bool {
        let Some(spare) = self.free.pop() else {
            return false;
        };
        let prepared = replace(&mut self.builder, spare);
        match self.ready.push(prepared) {
            Ok(()) => {
                // Only scalar metadata is reset; all N+1 allocations persist.
                self.builder.valid = 0;
                self.builder.skipped_tail = None;
                true
            }
            Err(PushError::Full(prepared)) => {
                let spare = replace(&mut self.builder, prepared);
                self.free.push(spare);
                false
            }
        }
    }

    /// Non-RT, at most N returns per call. No new windows are constructed.
    pub(super) fn reclaim(&mut self) -> usize {
        let mut reclaimed = 0;
        for _ in 0..self.limit {
            let Ok(window) = self.retired.pop() else {
                break;
            };
            // Each returned window restores exactly one consumed free credit.
            debug_assert!(self.free.len() < self.limit);
            self.free.push(window);
            reclaimed += 1;
        }
        reclaimed
    }
}

impl DeviceSide {
    /// Capture once on callback entry, after return_pending(). The caller must
    /// retire at most this many windows even if the worker publishes more.
    /// The partially consumed current window remains the ready ring's head.
    pub(super) fn visible(&self) -> usize {
        if self.pending_return.is_some() {
            0
        } else {
            self.ready.slots()
        }
    }

    pub(super) fn peek(&self) -> Option<&Window> {
        if self.pending_return.is_some() {
            None
        } else {
            self.ready.peek().ok()
        }
    }

    pub(super) fn offset(&self) -> usize {
        self.offset
    }

    pub(super) fn advance(&mut self, frames: usize) {
        let next = self
            .offset
            .checked_add(frames)
            .expect("device offset remains within the fixed window");
        debug_assert!(self.peek().is_some_and(|window| next <= window.valid));
        self.offset = next;
    }

    /// Remove the current window, including a revoked or zero-frame tail. A
    /// failed return retains its ownership locally and prevents further peeks.
    /// Call only after peek succeeded within this callback's visible budget.
    pub(super) fn retire(&mut self) -> bool {
        if self.pending_return.is_some() {
            return false;
        }
        let Ok(window) = self.ready.pop() else {
            return false;
        };
        self.offset = 0;
        self.pending_return = Some(window);
        self.return_pending()
    }

    /// Exactly one push attempt. Under N credits the retired ring has room for
    /// every outstanding window; preserve ownership even if that invariant fails.
    pub(super) fn return_pending(&mut self) -> bool {
        let Some(window) = self.pending_return.take() else {
            return true;
        };
        match self.retired.push(window) {
            Ok(()) => true,
            Err(PushError::Full(window)) => {
                self.pending_return = Some(window);
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepare(side: &mut PrepareSide, sample: i32) {
        let window = side.builder_mut();
        window.pcm[0] = sample;
        window.valid = 1;
    }

    #[test]
    fn renderer_realtime_transport_paused_reclaimer_preserves_builder_and_resumes() {
        let (mut producer, mut device) = pipe(2, 2, 1).unwrap();
        prepare(&mut producer, 11);
        assert!(producer.publish());
        prepare(&mut producer, 22);
        assert!(producer.publish());
        prepare(&mut producer, 33);
        assert!(!producer.publish());
        let mut output = [0; 2];
        assert!(device.return_pending());
        let visible = device.visible();
        assert_eq!(visible, 2);
        for sample in output.iter_mut().take(visible) {
            *sample = device.peek().unwrap().pcm[device.offset()];
            device.advance(1);
            assert!(device.retire());
        }
        assert_eq!(output, [11, 22]);
        assert!(
            !producer.publish(),
            "retired ownership still consumes credits"
        );
        assert_eq!(producer.builder_mut().pcm[0], 33);
        assert_eq!(producer.reclaim(), 2);
        assert!(producer.publish());
        assert_eq!(device.peek().unwrap().pcm[0], 33);
        assert!(device.retire());
        assert_eq!(producer.reclaim(), 1);
    }

    #[test]
    fn renderer_realtime_transport_partial_window_keeps_offset_until_retirement() {
        let (mut producer, mut device) = pipe(1, 2, 1).unwrap();
        let window = producer.builder_mut();
        window.pcm.copy_from_slice(&[11, 22]);
        window.valid = 2;
        assert!(producer.publish());
        assert_eq!(device.peek().unwrap().pcm[device.offset()], 11);
        device.advance(1);
        // A second device invocation resumes the same window, without taking
        // another credit or changing its offset when a return is unnecessary.
        assert!(device.return_pending());
        assert_eq!(device.visible(), 1);
        assert_eq!(device.peek().unwrap().pcm[device.offset()], 22);
        device.advance(1);
        assert!(device.retire());
        assert_eq!(device.offset(), 0);
        assert_eq!(producer.reclaim(), 1);
        prepare(&mut producer, 33);
        assert!(producer.publish());
        assert_eq!(device.peek().unwrap().pcm[device.offset()], 33);
    }

    #[test]
    fn renderer_realtime_transport_rejects_unrepresentable_configuration() {
        assert!(pipe(0, 1, 1).is_none());
        assert!(pipe(1, 0, 1).is_none());
        assert!(pipe(1, 1, 0).is_none());
        assert!(pipe(usize::MAX, 1, 1).is_none());
        assert!(pipe(1, usize::MAX, 2).is_none());
    }
}

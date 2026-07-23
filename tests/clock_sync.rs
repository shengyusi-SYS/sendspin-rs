use sendspin::sync::{ClockQuality, ClockSync};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

#[derive(Debug)]
struct Gate0Clock {
    now_us: AtomicI64,
}

impl Gate0Clock {
    fn new(now_us: i64) -> Self {
        Self {
            now_us: AtomicI64::new(now_us),
        }
    }

    fn set(&self, now_us: i64) {
        self.now_us.store(now_us, Ordering::SeqCst);
    }
}

impl sendspin::sync::Clock for Gate0Clock {
    fn now_micros(&self) -> i64 {
        self.now_us.load(Ordering::SeqCst)
    }
}

/// Assert that two values are within tolerance (microseconds precision)
fn assert_within(actual: Option<i64>, expected: i64, tolerance: i64) {
    let actual = actual.expect("expected Some value");
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "expected {} ± {}, got {} (diff: {})",
        expected,
        tolerance,
        actual,
        diff
    );
}

fn conversion_snapshot(sync: &ClockSync) -> ([Option<i64>; 4], [Option<i64>; 4]) {
    let points = [-10_000, 0, 3_000, 10_000];
    (
        points.map(|point| sync.client_to_server_micros(point)),
        points.map(|point| sync.server_to_client_micros(point)),
    )
}

#[test]
fn test_fresh_clock_sync_initial_state() {
    let sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    assert_eq!(sync.rtt_micros(), None);
    assert!(!sync.is_synchronized());
    assert_eq!(sync.quality(), ClockQuality::Lost);
    assert!(sync.is_stale());
    assert_eq!(sync.server_to_client_micros(1000), None);
    assert_eq!(sync.client_to_server_micros(1000), None);
}

#[test]
fn test_single_update_not_synchronized() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // One sample isn't enough for the filter to converge (needs count >= 2)
    sync.update(1_000_000, 500_000, 500_010, 1_000_040);

    assert_eq!(sync.rtt_micros(), Some(30));
    assert!(
        !sync.is_synchronized(),
        "single update should not synchronize"
    );
    assert_eq!(
        sync.server_to_client_micros(500_000),
        None,
        "should return None when not synchronized"
    );
}

#[test]
fn test_clock_sync_rtt_calculation() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // Simulate sync: client sends at 1000µs, server receives at 500µs (server loop time)
    let t1 = 1_000_000; // Client transmitted (Unix µs)
    let t2 = 500_000; // Server received (server loop µs)
    let t3 = 500_010; // Server transmitted (server loop µs)
    let t4 = 1_000_050; // Client received (Unix µs)

    sync.update(t1, t2, t3, t4);

    // RTT = (t4 - t1) - (t3 - t2) = 50 - 10 = 40µs
    assert_eq!(sync.rtt_micros(), Some(40));
}

#[test]
fn test_server_to_client_conversion() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    let t1 = 1_000_000;
    let t2 = 1_005_100;
    let t3 = 1_005_100;
    let t4 = 1_000_200;

    sync.update(t1, t2, t3, t4);
    sync.update(2_000_000, 2_005_100, 2_005_100, 2_000_200);

    let client_micros = sync.server_to_client_micros(2_005_000);
    // Kalman filter may introduce small rounding errors
    assert_within(client_micros, 2_000_000, 10);
}

#[test]
fn test_sync_quality() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // Good RTT (30µs)
    sync.update(1_000_000, 500_000, 500_010, 1_000_040);
    assert_eq!(sync.quality(), ClockQuality::Good);

    // Degraded RTT (75ms = 75,000µs)
    sync.update(2_000_000, 600_000, 600_010, 2_075_010);
    assert_eq!(sync.quality(), ClockQuality::Degraded);

    // The accepted upper RTT boundary is inclusive and remains degraded.
    sync.update(3_000_000, 700_000, 700_000, 3_100_000);
    assert_eq!(sync.quality(), ClockQuality::Degraded);
}

#[test]
fn test_sync_quality_recovery() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // Start with degraded RTT (75ms)
    sync.update(1_000_000, 600_000, 600_010, 1_075_010);
    assert_eq!(sync.quality(), ClockQuality::Degraded);

    // Recover with good RTT (20µs)
    sync.update(2_000_000, 700_000, 700_010, 2_000_030);
    assert_eq!(sync.quality(), ClockQuality::Good);
}

#[test]
fn test_sync_quality_unchanged_after_high_rtt() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // First, establish a good RTT (30µs)
    sync.update(1_000_000, 500_000, 500_010, 1_000_040);
    assert_eq!(sync.quality(), ClockQuality::Good);
    assert_eq!(sync.rtt_micros(), Some(30));

    // Now provide a very high RTT (> 100_000µs), which should be discarded
    // RTT = (t4 - t1) - (t3 - t2) = (3_100_020 - 3_000_000) - (700_010 - 700_000)
    //     = 100_020 - 10 = 100_010µs
    sync.update(3_000_000, 700_000, 700_010, 3_100_020);

    // The high RTT should not overwrite the previously stored good RTT,
    // and the quality should remain unchanged.
    assert_eq!(sync.quality(), ClockQuality::Good);
    assert_eq!(sync.rtt_micros(), Some(30));
}

#[test]
fn test_timestamp_boundary_zero_values() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // All-zero timestamps: RTT = (0 - 0) - (0 - 0) = 0, which is valid
    sync.update(0, 0, 0, 0);
    assert_eq!(sync.rtt_micros(), Some(0));
}

#[test]
fn test_clock_drift_correction() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // Low-RTT samples keep drift covariance low enough for the SNR gate to
    // permit drift correction in both conversion directions.
    sync.update(1_000_000, 1_005_100, 1_005_100, 1_000_000);
    sync.update(2_000_000, 2_005_200, 2_005_200, 2_000_000);

    let server_time = sync.client_to_server_micros(3_000_000);
    // Kalman filter may introduce small rounding errors
    assert_within(server_time, 3_005_300, 10);
}

#[test]
fn test_conversions_use_offset_only_when_drift_snr_is_low() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // First sample: offset = +100µs with tight uncertainty.
    sync.update(1_000, 1_100, 1_100, 1_000);
    // Second sample 200ms later (a legitimate drift baseline): offset =
    // +120µs, but a 1ms RTT gives the drift estimate high covariance. Its
    // SNR is below 2σ, so both conversion directions should ignore drift
    // and use only the current offset.
    sync.update(200_000, 200_620, 200_620, 201_000);

    assert!(sync.is_synchronized());
    assert_eq!(sync.server_to_client_micros(3_120), Some(3_000));
    assert_eq!(sync.client_to_server_micros(3_000), Some(3_120));
}

#[test]
fn test_conversions_apply_drift_when_snr_is_high() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // Offset moves +100µs → +4100µs over a 200ms baseline (drift = 0.02),
    // with low-RTT samples. The covariance is small enough for the 2σ SNR
    // gate to pass, so conversions apply the drift estimate.
    sync.update(1_000, 1_100, 1_100, 1_000);
    sync.update(201_000, 205_100, 205_100, 201_000);

    assert!(sync.is_synchronized());
    assert_eq!(sync.server_to_client_micros(3_120), Some(2_980));
    assert_eq!(sync.client_to_server_micros(3_000), Some(3_140));
}

#[test]
fn test_diverged_drift_returns_none() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // Two samples 200ms apart (a legitimate drift baseline) where the NTP
    // offset collapses from +100µs to -99_900µs, giving drift = -0.5 with
    // high SNR — far beyond any real hardware clock skew. The filter would
    // apply that drift, so both conversions must refuse and return None.
    sync.update(1_000, 1_100, 1_100, 1_000);
    sync.update(201_000, 101_100, 101_100, 201_000);

    assert!(
        sync.server_to_client_micros(2000).is_none(),
        "server_to_client should return None when drift has diverged"
    );
    assert!(
        sync.client_to_server_micros(2000).is_none(),
        "client_to_server should return None when drift has diverged"
    );
}

#[test]
fn test_negative_rtt_discarded() {
    use sendspin::sync::ClockUpdateOutcome;

    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // Craft timestamps where t4 < t1 (response "before" request),
    // producing negative RTT. Should be silently discarded.
    assert_eq!(
        sync.update(1000, 500, 500, 900),
        ClockUpdateOutcome::RejectedInvalidRtt
    );
    assert_eq!(sync.rtt_micros(), None, "negative RTT should not be stored");
    assert_eq!(sync.health().rejected_invalid_rtt, 1);
    assert!(
        !sync.is_synchronized(),
        "filter should not advance on invalid RTT"
    );
}

#[test]
fn test_zero_rtt_accepted() {
    let mut sync = ClockSync::new(Arc::new(Gate0Clock::new(5_000_000)));

    // RTT = 0 is legitimate on localhost. The filter clamps max_error
    // to 1µs so zero-variance samples don't corrupt covariance.
    sync.update(1000, 1100, 1100, 1000);
    sync.update(2000, 2100, 2100, 2000);
    assert_eq!(sync.rtt_micros(), Some(0));
    assert!(sync.is_synchronized(), "zero RTT should still allow sync");
}

#[test]
fn clock_update_outcomes_reject_invalid_and_non_monotonic_samples_without_state_mutation() {
    use sendspin::sync::ClockUpdateOutcome;

    let clock = Arc::new(Gate0Clock::new(1_000));
    let mut sync = ClockSync::new(clock.clone());
    assert_eq!(
        sync.update(900, 1_900, 1_900, 1_000),
        ClockUpdateOutcome::Applied
    );
    clock.set(2_000);
    assert_eq!(
        sync.update(1_900, 2_900, 2_900, 2_000),
        ClockUpdateOutcome::Applied
    );
    let applied = sync.health();
    let applied_conversions = conversion_snapshot(&sync);

    assert_eq!(
        sync.update(1_900, 2_900, 2_900, 2_000),
        ClockUpdateOutcome::RejectedNonMonotonicT4
    );
    let duplicate = sync.health();
    let mut expected_duplicate = applied;
    expected_duplicate.rejected_non_monotonic_t4 = 1;
    assert_eq!(duplicate, expected_duplicate);
    assert_eq!(conversion_snapshot(&sync), applied_conversions);

    assert_eq!(
        sync.update(1_900, 2_900, 2_900, 1_999),
        ClockUpdateOutcome::RejectedNonMonotonicT4
    );
    let backwards = sync.health();
    let mut expected_backwards = expected_duplicate;
    expected_backwards.rejected_non_monotonic_t4 = 2;
    assert_eq!(backwards, expected_backwards);
    assert_eq!(conversion_snapshot(&sync), applied_conversions);

    assert_eq!(
        sync.update(i64::MIN, 0, 0, i64::MAX),
        ClockUpdateOutcome::RejectedInvalidRtt
    );
    let extreme = sync.health();
    let mut expected_extreme = expected_backwards;
    expected_extreme.rejected_invalid_rtt = 1;
    assert_eq!(extreme, expected_extreme);
    assert_eq!(conversion_snapshot(&sync), applied_conversions);

    assert_eq!(
        sync.update(4_000, 0, 0, 3_000),
        ClockUpdateOutcome::RejectedInvalidRtt
    );
    let negative = sync.health();
    let mut expected_negative = expected_extreme;
    expected_negative.rejected_invalid_rtt = 2;
    assert_eq!(negative, expected_negative);
    assert_eq!(conversion_snapshot(&sync), applied_conversions);

    clock.set(3_000);
    assert_eq!(
        sync.update(2_900, 3_900, 3_900, 3_000),
        ClockUpdateOutcome::Applied,
        "the next valid sample after all rejection classes must recover"
    );
    let recovered = sync.health();
    assert_eq!(recovered.accepted_samples, applied.accepted_samples + 1);
    assert_eq!(recovered.rejected_non_monotonic_t4, 2);
    assert_eq!(recovered.rejected_invalid_rtt, 2);
    assert_eq!(recovered.last_valid_t4_us, Some(3_000));
    assert!(recovered.synchronized);
    assert!(conversion_snapshot(&sync)
        .0
        .into_iter()
        .all(|value| value.is_some()));
    assert!(conversion_snapshot(&sync)
        .1
        .into_iter()
        .all(|value| value.is_some()));
}

#[test]
fn clock_health_stale_reset_and_bounded_burst_recovery_are_deterministic() {
    use sendspin::sync::{ClockStaleReason, ClockUpdateOutcome};

    let clock = Arc::new(Gate0Clock::new(1_000_000));
    let mut sync = ClockSync::new(clock.clone());
    assert_eq!(
        sync.health().stale_reason,
        Some(ClockStaleReason::NoSamples)
    );

    for index in 0..10_i64 {
        let t4 = 1_000_000 + index * 100_000;
        clock.set(t4);
        assert_eq!(
            sync.update(t4 - 100, t4 + 4_900, t4 + 4_900, t4),
            ClockUpdateOutcome::Applied
        );
    }
    assert!(sync.health().settled);

    clock.set(1_900_000 + 5_000_001);
    let stopped = sync.health();
    assert!(stopped.stale);
    assert_eq!(stopped.stale_reason, Some(ClockStaleReason::SampleExpired));
    assert_eq!(sync.server_to_client_micros(7_000_000), None);

    clock.set(1_899_999);
    let backwards = sync.health();
    assert!(backwards.stale);
    assert_eq!(
        backwards.stale_reason,
        Some(ClockStaleReason::EndpointClockBackwards)
    );

    sync.reset();
    let reset = sync.health();
    assert_eq!(reset.accepted_samples, 0);
    assert_eq!(reset.last_valid_t4_us, None);
    assert!(!reset.synchronized);
    assert!(!reset.settled);
    assert_eq!(reset.stale_reason, Some(ClockStaleReason::NoSamples));

    for index in 0..10_i64 {
        let t4 = 10_000_000 + index * 100_000;
        clock.set(t4);
        assert_eq!(
            sync.update(t4 - 100, t4 + 4_900, t4 + 4_900, t4),
            ClockUpdateOutcome::Applied
        );
    }
    let recovered = sync.health();
    assert!(recovered.synchronized);
    assert!(recovered.settled);
    assert!(!recovered.stale);
}

#[test]
fn clock_extreme_valid_monotonic_span_does_not_overflow() {
    use sendspin::sync::{ClockStaleReason, ClockUpdateOutcome};

    let clock = Arc::new(Gate0Clock::new(i64::MIN));
    let mut sync = ClockSync::new(clock.clone());
    assert_eq!(
        sync.update(i64::MIN, i64::MIN, i64::MIN, i64::MIN),
        ClockUpdateOutcome::Applied
    );

    clock.set(i64::MAX);
    let aged = sync.health();
    assert_eq!(aged.sample_age_us, Some(u64::MAX));
    assert_eq!(aged.stale_reason, Some(ClockStaleReason::SampleExpired));

    assert_eq!(
        sync.update(i64::MAX, i64::MAX, i64::MAX, i64::MAX),
        ClockUpdateOutcome::Applied
    );
    let recovered = sync.health();
    assert!(recovered.synchronized, "{recovered:?}");
    assert_eq!(sync.client_to_server_micros(i64::MIN), Some(i64::MIN));
    assert_eq!(sync.client_to_server_micros(i64::MAX), Some(i64::MAX));
    assert_eq!(sync.server_to_client_micros(i64::MIN), Some(i64::MIN));
    assert_eq!(sync.server_to_client_micros(i64::MAX), Some(i64::MAX));
}

#[test]
fn clock_conversion_returns_none_when_offset_pushes_result_out_of_i64_range() {
    let positive_clock = Arc::new(Gate0Clock::new(1_000));
    let mut positive = ClockSync::new(positive_clock.clone());
    positive.update(0, 100, 100, 0);
    positive.update(1_000, 1_100, 1_100, 1_000);
    assert_eq!(positive.client_to_server_micros(i64::MAX), None);
    assert_eq!(positive.server_to_client_micros(i64::MIN), None);

    let negative_clock = Arc::new(Gate0Clock::new(1_000));
    let mut negative = ClockSync::new(negative_clock);
    negative.update(0, -100, -100, 0);
    negative.update(1_000, 900, 900, 1_000);
    assert_eq!(negative.client_to_server_micros(i64::MIN), None);
    assert_eq!(negative.server_to_client_micros(i64::MAX), None);
}

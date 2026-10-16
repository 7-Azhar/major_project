"""Unit tests for metrics calculation and system statistics tracker."""

import time
import pytest
from app.metrics import MetricsTracker


def test_metrics_empty():
    """Verify metrics behaviour when no frames recorded."""
    tracker = MetricsTracker(resolution_w=1280, resolution_h=720)
    stats = tracker.get_current_stats()

    assert stats["resolution"] == "1280x720"
    assert stats["frames_processed"] == 0
    assert stats["avg_frame_time_ms"] == 0.0
    assert stats["observed_output_fps"] == 0.0


def test_metrics_timing_calculations():
    """Verify average, median, p95, min, max calculations."""
    tracker = MetricsTracker(resolution_w=1920, resolution_h=1080)
    tracker.start()

    # Record 10 frame times in seconds (e.g. 10ms, 20ms, ..., 100ms)
    frame_times_sec = [0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100]

    for t in frame_times_sec:
        tracker.record_frame(processing_time_sec=t, displayed=True)

    tracker.update_capture_counts(frames_received=12, dropped_frames=2)
    time.sleep(0.1)  # Ensure elapsed runtime > 0
    tracker.stop()

    stats = tracker.get_current_stats()

    assert stats["frames_processed"] == 10
    assert stats["frames_received"] == 12
    assert stats["frames_dropped"] == 2
    assert stats["min_frame_time_ms"] == pytest.approx(10.0, rel=1e-2)
    assert stats["max_frame_time_ms"] == pytest.approx(100.0, rel=1e-2)
    assert stats["avg_frame_time_ms"] == pytest.approx(55.0, rel=1e-2)
    assert stats["median_frame_time_ms"] == pytest.approx(55.0, rel=1e-2)
    assert stats["p95_frame_time_ms"] == pytest.approx(95.5, rel=1e-1)


def test_hardware_metrics_fail_safe():
    """Verify hardware metric extraction never throws unhandled exceptions."""
    tracker = MetricsTracker()
    hw = tracker.get_hardware_metrics()

    assert isinstance(hw, dict)
    assert "cpu_utilization_pct" in hw
    assert "ram_utilization_pct" in hw
    assert "gpu_utilization_pct" in hw
    assert "gpu_memory_used_mb" in hw


def test_report_text_generation():
    """Verify textual report formatting."""
    tracker = MetricsTracker(resolution_w=1280, resolution_h=720)
    tracker.start()
    tracker.record_frame(0.016, displayed=True)
    tracker.stop()

    report = tracker.generate_report_text("TEST REPORT")
    assert "TEST REPORT" in report
    assert "1280x720" in report
    assert "Observed Output FPS:" in report

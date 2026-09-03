from __future__ import annotations

import time
import unittest

from Comparacion.resource_monitor import (
    ResourceMonitor,
    ResourceObservation,
    protocol_resource_fields,
)


class ResourceMonitorTests(unittest.TestCase):
    def test_monitor_returns_nonnegative_time_and_explicit_statuses(self) -> None:
        with ResourceMonitor(sample_interval_s=0.001) as monitor:
            payload = bytearray(1024 * 32)
            time.sleep(0.003)

        self.assertTrue(payload)
        self.assertIsNotNone(monitor.result)
        result = monitor.result
        assert result is not None
        self.assertGreaterEqual(result.wall_clock_s, 0.0)
        self.assertTrue(result.process_memory_status)
        self.assertEqual(result.cuda_memory_status, "not_requested")
        if result.process_memory_status == "measured":
            self.assertGreater(result.peak_process_rss_bytes or 0, 0)
            self.assertGreaterEqual(
                result.peak_process_tree_rss_bytes or 0,
                result.peak_process_rss_bytes or 0,
            )

    def test_nonpositive_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ResourceMonitor(sample_interval_s=0)

    def test_protocol_fields_keep_unmeasured_selected_fit_explicit(self) -> None:
        observation = ResourceObservation(
            wall_clock_s=1.0,
            peak_process_rss_bytes=10,
            peak_process_tree_rss_bytes=12,
            peak_cuda_allocated_bytes=None,
            peak_cuda_reserved_bytes=None,
            process_memory_status="measured",
            cuda_memory_status="not_requested",
            sample_interval_s=0.05,
        )
        fields = protocol_resource_fields(
            observation,
            observation,
            measurement_condition="isolated",
        )

        self.assertEqual(fields["selection_peak_process_tree_rss_bytes"], 12)
        self.assertIsNone(fields["selection_peak_cuda_allocated_bytes"])
        self.assertEqual(fields["selection_cuda_resource_status"], "not_requested")
        self.assertIsNone(fields["selected_fit_peak_process_tree_rss_bytes"])
        self.assertEqual(
            fields["selected_fit_resource_status"], "not_measured_separately"
        )
        self.assertTrue(fields["resource_cost_usable"])

    def test_contended_measurement_is_not_cost_usable(self) -> None:
        observation = ResourceObservation(
            wall_clock_s=1.0,
            peak_process_rss_bytes=None,
            peak_process_tree_rss_bytes=None,
            peak_cuda_allocated_bytes=None,
            peak_cuda_reserved_bytes=None,
            process_memory_status="psutil_not_installed",
            cuda_memory_status="not_requested",
            sample_interval_s=0.05,
        )

        fields = protocol_resource_fields(
            observation,
            observation,
            measurement_condition="contended",
        )
        self.assertFalse(fields["resource_cost_usable"])


if __name__ == "__main__":
    unittest.main()

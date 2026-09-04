from __future__ import annotations

import subprocess
import sys
import time
import unittest

from Comparacion.resource_monitor import ResourceMonitor


class ResourceMonitorTests(unittest.TestCase):
    def test_reports_positive_process_peak(self) -> None:
        with ResourceMonitor(sample_interval_s=0.01) as monitor:
            payload = bytearray(2_000_000)

        self.assertTrue(payload)
        self.assertGreater(monitor.measurement()["peak_process_memory_bytes"], 0)

    def test_includes_child_process_memory(self) -> None:
        with ResourceMonitor(sample_interval_s=0.01) as monitor:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time; payload=bytearray(20_000_000); time.sleep(0.25)",
                ]
            )
            time.sleep(0.1)
            child.wait(timeout=5)

        result = monitor.measurement()
        self.assertGreater(result["peak_child_process_memory_bytes"], 0)
        self.assertGreaterEqual(
            result["peak_process_memory_bytes"],
            result["peak_child_process_memory_bytes"],
        )

    def test_cpu_marks_cuda_memory_not_applicable(self) -> None:
        with ResourceMonitor(use_cuda=False) as monitor:
            pass

        result = monitor.measurement()
        self.assertIsNone(result["peak_gpu_memory_bytes"])
        self.assertEqual(result["peak_gpu_memory_status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import subprocess
import sys
import time
import unittest

from Comparacion.classical_models import FiniteGlobalHMM
from Comparacion.resource_monitor import ResourceMonitor
from next_token_experiment.schemas import PreparedPiece


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

    def test_monitor_does_not_change_selection_or_predictions(self) -> None:
        pieces = [
            PreparedPiece(
                piece_id=f"piece-{index}",
                source_path=f"piece-{index}.musicxml",
                title=f"Piece {index}",
                composer="Composer",
                canonical_work_id=f"work-{index}",
                representation="pitch_class",
                vocabulary=[str(value) for value in range(12)],
                tokens=[(index + step) % 12 for step in range(8)],
                n_events=8,
                metadata={},
            )
            for index in range(5)
        ]

        def run(monitored: bool) -> tuple[int, float]:
            model = FiniteGlobalHMM(
                candidate_num_states=(2, 3),
                max_iterations=2,
                tolerance=1e-4,
                seed=7,
            )
            if monitored:
                with ResourceMonitor(sample_interval_s=0.01):
                    fit = model.fit(pieces[:3], pieces[3:4], bos_token_id=12)
                    result = model.evaluate(pieces[4:], bos_token_id=12)
            else:
                fit = model.fit(pieces[:3], pieces[3:4], bos_token_id=12)
                result = model.evaluate(pieces[4:], bos_token_id=12)
            return fit.selected_states, result["summary"]["test_nll_per_token"]

        self.assertEqual(run(monitored=False), run(monitored=True))


if __name__ == "__main__":
    unittest.main()

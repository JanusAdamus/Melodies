from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from Comparacion.cost_scenarios import build_cost_scenarios


class CostScenarioTests(unittest.TestCase):
    def _write_costs(self, path: Path) -> None:
        rows = [
            {
                "model": "transformer",
                "device": "cuda:0",
                "frac": 1.0,
                "selection_wall_clock_s": 100,
                "evaluation_wall_clock_s": 2,
                "resource_measurement_condition": "isolated",
                "resource_cost_usable": True,
                "selection_peak_process_tree_rss_bytes": 1000,
                "selection_peak_cuda_allocated_bytes": 500,
            },
            {
                "model": "finite_hmm",
                "device": "cpu",
                "frac": 1.0,
                "selection_wall_clock_s": 400,
                "evaluation_wall_clock_s": 1,
                "resource_measurement_condition": "isolated",
                "resource_cost_usable": True,
                "selection_peak_process_tree_rss_bytes": 2000,
                "selection_peak_cuda_allocated_bytes": "",
            },
            {
                "model": "vomm",
                "device": "cpu",
                "frac": 1.0,
                "selection_wall_clock_s": 1,
                "evaluation_wall_clock_s": 1,
                "resource_measurement_condition": "contended",
                "resource_cost_usable": False,
                "selection_peak_process_tree_rss_bytes": 100,
                "selection_peak_cuda_allocated_bytes": "",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_tariffs(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "currency": "MXN",
                    "tariffs": [
                        {
                            "device_class": "cpu",
                            "hourly_rate": 1,
                            "source": "documented CPU scenario",
                            "observed_on": "2026-09-01",
                            "region": "test-region",
                            "scope": "complete CPU configuration",
                            "billing_unit": "device_hour",
                            "ownership_model": "remote_service",
                        },
                        {
                            "device_class": "gpu",
                            "hourly_rate": 2,
                            "source": "documented GPU scenario",
                            "observed_on": "2026-09-01",
                            "region": "test-region",
                            "scope": "complete GPU configuration",
                            "billing_unit": "device_hour",
                            "ownership_model": "remote_service",
                        },
                    ],
                    "scenarios": [
                        {"name": "fit", "evaluation_repetitions": 0},
                        {"name": "reuse", "evaluation_repetitions": 10},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_costs_and_break_even_are_computed_from_eligible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            costs = root / "engineering_costs.csv"
            tariffs = root / "tariffs.json"
            self._write_costs(costs)
            self._write_tariffs(tariffs)

            report = build_cost_scenarios(costs, tariffs)

            self.assertEqual(report["status"], "computed_from_isolated_measurements")
            self.assertEqual(
                {row["model"] for row in report["observations"]},
                {"transformer", "finite_hmm"},
            )
            fit_threshold = next(
                row
                for row in report["break_even_rate_ratios"]
                if row["scenario"] == "fit"
            )
            self.assertAlmostEqual(
                fit_threshold["max_gpu_to_cpu_hourly_rate_ratio_for_gpu_to_cost_less"],
                4.0,
            )

    def test_missing_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            costs = root / "engineering_costs.csv"
            tariffs = root / "tariffs.json"
            self._write_costs(costs)
            tariffs.write_text(
                json.dumps(
                    {
                        "currency": "MXN",
                        "tariffs": [
                            {"device_class": "cpu", "hourly_rate": 1},
                            {"device_class": "gpu", "hourly_rate": 2},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "source"):
                build_cost_scenarios(costs, tariffs)

    def test_no_isolated_rows_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            costs = root / "engineering_costs.csv"
            tariffs = root / "tariffs.json"
            self._write_costs(costs)
            text = costs.read_text(encoding="utf-8").replace("isolated", "contended")
            costs.write_text(text, encoding="utf-8")
            self._write_tariffs(tariffs)

            with self.assertRaisesRegex(ValueError, "no isolated"):
                build_cost_scenarios(costs, tariffs)

    def test_duplicate_device_tariff_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            costs = root / "engineering_costs.csv"
            tariffs = root / "tariffs.json"
            self._write_costs(costs)
            self._write_tariffs(tariffs)
            payload = json.loads(tariffs.read_text(encoding="utf-8"))
            payload["tariffs"].append(dict(payload["tariffs"][0]))
            tariffs.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate tariff"):
                build_cost_scenarios(costs, tariffs)

    def test_tariff_requires_scope_region_and_billing_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            costs = root / "engineering_costs.csv"
            tariffs = root / "tariffs.json"
            self._write_costs(costs)
            self._write_tariffs(tariffs)
            payload = json.loads(tariffs.read_text(encoding="utf-8"))
            payload["tariffs"][0].pop("region")
            tariffs.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "region"):
                build_cost_scenarios(costs, tariffs)


if __name__ == "__main__":
    unittest.main()

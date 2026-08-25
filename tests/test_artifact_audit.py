from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from Comparacion.artifact_audit import audit_run, build_manifest, write_audit_reports


_RAW_FIELDS = (
    "model",
    "data_seed",
    "model_seed",
    "frac",
    "n_train_pieces",
    "n_train_tokens",
    "test_nll",
)

_SUMMARY_FIELDS = ("model", "frac", "mean_test_nll", "n_runs")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _raw_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in ("vomm", "finite_hmm"):
        for data_seed in (1, 2):
            for frac in (0.5, 1.0):
                rows.append(
                    {
                        "model": model,
                        "data_seed": data_seed,
                        "model_seed": 1,
                        "frac": frac,
                        "n_train_pieces": 10,
                        "n_train_tokens": 100,
                        "test_nll": 2.0,
                    }
                )
    return rows


def _summary_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in raw_rows:
        grouped.setdefault((str(row["model"]), float(row["frac"])), []).append(float(row["test_nll"]))
    return [
        {
            "model": model,
            "frac": frac,
            "mean_test_nll": sum(values) / len(values),
            "n_runs": len(values),
        }
        for (model, frac), values in sorted(grouped.items())
    ]


def _config() -> dict[str, object]:
    return {
        "data_seeds": [1, 2],
        "model_seeds": [1],
        "train_fractions": [0.5, 1.0],
        "split_seed": 7,
        "include_vomm_control": True,
    }


def _build_run(root: Path, *, complete: bool = True) -> Path:
    run_dir = root / "run_under_audit"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _raw_rows()
    _write_csv(run_dir / "results_raw.csv", raw_rows, _RAW_FIELDS)
    _write_csv(run_dir / "results_summary.csv", _summary_rows(raw_rows), _SUMMARY_FIELDS)
    _write_csv(
        run_dir / "piece_metrics_raw.csv",
        [{"model": "vomm", "data_seed": 1, "model_seed": 1, "frac": 1.0, "n_train_pieces": 10, "n_train_tokens": 100, "test_nll": 2.0}],
        _RAW_FIELDS,
    )
    _write_csv(
        run_dir / "engineering_costs.csv",
        [{"model": "vomm", "data_seed": 1, "model_seed": 1, "frac": 1.0, "n_train_pieces": 10, "n_train_tokens": 100, "test_nll": 2.0}],
        _RAW_FIELDS,
    )
    _write_csv(run_dir / "exclusions.csv", [], _RAW_FIELDS)
    (run_dir / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (run_dir / "pairwise_comparisons.json").write_text(json.dumps({"comparisons": []}), encoding="utf-8")
    (run_dir / "protocol_audit.json").write_text(
        json.dumps({"status": "passed", "unexpected_piece_ids": []}), encoding="utf-8"
    )
    (run_dir / "structural_evaluation.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
    (run_dir / "pareto_summary.json").write_text(json.dumps({"frontier": []}), encoding="utf-8")
    (run_dir / "preprocessing_report.json").write_text(json.dumps({"n_pieces": 10}), encoding="utf-8")

    cells = [f"data_seed={seed},frac={frac}" for seed in (1, 2) for frac in (0.5, 1.0)]
    if not complete:
        cells = cells[:1]
    with (run_dir / "checkpoint.jsonl").open("w", encoding="utf-8") as handle:
        for cell in cells:
            handle.write(json.dumps({"cell": cell, "raw_rows": []}) + "\n")
    return run_dir


class BuildManifestTests(unittest.TestCase):
    def test_manifest_hashes_are_stable_and_paths_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp))
            first = build_manifest(run_dir)
            second = build_manifest(run_dir)

            self.assertEqual(first, second)
            entries = {entry["relative_path"]: entry for entry in first["files"]}
            self.assertIn("results_raw.csv", entries)
            for entry in first["files"]:
                self.assertFalse(Path(entry["relative_path"]).is_absolute())
                self.assertEqual(len(entry["sha256"]), 64)
                self.assertGreaterEqual(entry["size_bytes"], 0)

    def test_manifest_changes_when_a_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp))
            before = build_manifest(run_dir)
            (run_dir / "results_raw.csv").write_text("model\nvomm\n", encoding="utf-8")
            after = build_manifest(run_dir)

            self.assertNotEqual(before, after)


class AuditRunTests(unittest.TestCase):
    def test_complete_run_passes_and_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp))
            audit = audit_run(run_dir)

            self.assertEqual(audit["status"], "passed")
            counts = audit["counts"]
            self.assertEqual(counts["raw_rows"], 8)
            self.assertEqual(counts["models"], 2)
            self.assertEqual(counts["fractions"], 2)
            self.assertEqual(counts["data_seeds"], 2)
            self.assertEqual(counts["completed_cells"], 4)
            self.assertEqual(counts["expected_cells"], 4)

    def test_missing_required_artifact_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp))
            (run_dir / "pareto_summary.json").unlink()
            audit = audit_run(run_dir)

            self.assertEqual(audit["status"], "incomplete")
            missing = _check(audit, "artifacts_present")
            self.assertEqual(missing["status"], "incomplete")
            self.assertIn("pareto_summary.json", missing["missing"])

    def test_invalid_json_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp))
            (run_dir / "pareto_summary.json").write_text("{not json", encoding="utf-8")
            audit = audit_run(run_dir)

            self.assertEqual(audit["status"], "failed")
            check = _check(audit, "json_parsable")
            self.assertEqual(check["status"], "failed")
            self.assertIn("pareto_summary.json", check["unreadable"])

    def test_unfinished_run_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp), complete=False)
            audit = audit_run(run_dir)

            self.assertEqual(audit["status"], "incomplete")
            check = _check(audit, "cells_completed")
            self.assertEqual(check["status"], "incomplete")
            self.assertEqual(check["completed_cells"], 1)
            self.assertEqual(check["expected_cells"], 4)

    def test_summary_disagreeing_with_raw_rows_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp))
            rows = _summary_rows(_raw_rows())
            rows[0]["mean_test_nll"] = 99.0
            _write_csv(run_dir / "results_summary.csv", rows, _SUMMARY_FIELDS)
            audit = audit_run(run_dir)

            self.assertEqual(audit["status"], "failed")
            check = _check(audit, "summary_matches_raw")
            self.assertEqual(check["status"], "failed")
            self.assertTrue(check["mismatches"])

    def test_config_grid_disagreeing_with_rows_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp))
            config = _config()
            config["data_seeds"] = [1, 2, 3]
            (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            audit = audit_run(run_dir)

            self.assertEqual(audit["status"], "incomplete")
            check = _check(audit, "config_matches_rows")
            self.assertIn(check["status"], {"failed", "incomplete"})

    def test_missing_run_summary_is_recorded_without_fabrication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp))
            audit = audit_run(run_dir)
            check = _check(audit, "run_summary_present")

            self.assertEqual(check["status"], "incomplete")
            self.assertFalse((run_dir / "run_summary.json").exists())

    def test_missing_run_directory_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = audit_run(Path(tmp) / "does_not_exist")

            self.assertEqual(audit["status"], "incomplete")
            self.assertEqual(audit["reason"], "original_artifacts_not_found")


class WriteAuditReportsTests(unittest.TestCase):
    def test_reports_are_written_outside_the_original_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            before = build_manifest(run_dir)
            output_dir = root / "audits"

            result = write_audit_reports(run_dir, output_dir)

            manifest_path = Path(result["manifest_path"])
            audit_path = Path(result["audit_path"])
            self.assertTrue(manifest_path.exists())
            self.assertTrue(audit_path.exists())
            self.assertFalse(manifest_path.is_relative_to(run_dir))
            self.assertFalse(audit_path.is_relative_to(run_dir))
            self.assertEqual(build_manifest(run_dir), before)
            self.assertEqual(
                json.loads(audit_path.read_text(encoding="utf-8"))["status"], "passed"
            )


def _check(audit: dict, name: str) -> dict:
    for check in audit["checks"]:
        if check["name"] == name:
            return check
    raise AssertionError(f"missing check {name}: {[item['name'] for item in audit['checks']]}")


if __name__ == "__main__":
    unittest.main()

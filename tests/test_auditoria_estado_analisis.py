"""El estado de análisis de la auditoría sale de lo que la corrida registró (Aclaración 2).

Una corrida anterior al preregistro y al runner corregido no evalúa la regla de B
aunque tenga VOMM, y los huecos frente al preregistro que el runner declaró
(la diferencia de partición sin reportar) aparecen en la auditoría.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Comparacion.artifact_audit import audit_run


def _write(root: Path, name: str, payload: dict) -> None:
    (root / name).write_text(json.dumps(payload), encoding="utf-8")


class AuditAnalysisStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "run"
        self.root.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_run_from_before_the_corrected_runner_does_not_evaluate_rule_b(self) -> None:
        # Así son las corridas de 2026-08: VOMM activado y ningún analysis_status.
        _write(self.root, "config.json", {"include_vomm_control": True})
        _write(self.root, "run_summary.json", {"status": "completed", "artifacts": {}})

        status = audit_run(self.root)["analysis_status"]

        self.assertEqual(status["status"], "predates_corrected_runner")
        self.assertFalse(status["evaluates_preregistered_rule_B"])

    def test_the_status_and_gaps_recorded_by_the_runner_are_reported(self) -> None:
        recorded = {
            "status": "preregistered",
            "evaluates_preregistered_rule_B": True,
            "reason": None,
            "holm_family_models": ["finite_hmm", "hdp_hmm", "transformer", "vomm"],
            "preregistration_gaps": ["test_split_difference_vs_original_run_not_reported"],
        }
        _write(self.root, "config.json", {"include_vomm_control": True})
        _write(self.root, "run_summary.json", {"status": "completed", "analysis_status": recorded, "artifacts": {}})

        status = audit_run(self.root)["analysis_status"]

        self.assertEqual(status["status"], "preregistered")
        self.assertEqual(
            status["preregistration_gaps"], ["test_split_difference_vs_original_run_not_reported"]
        )
        self.assertEqual(status["source"], "run_summary.json")

    def test_a_run_without_vomm_is_exploratory_even_without_a_recorded_status(self) -> None:
        _write(self.root, "config.json", {"include_vomm_control": False})

        status = audit_run(self.root)["analysis_status"]

        self.assertEqual(status["status"], "exploratory")
        self.assertFalse(status["evaluates_preregistered_rule_B"])

    def test_a_recorded_status_that_contradicts_the_config_is_not_trusted(self) -> None:
        _write(self.root, "config.json", {"include_vomm_control": False})
        _write(
            self.root,
            "run_summary.json",
            {"analysis_status": {"status": "preregistered", "evaluates_preregistered_rule_B": True}},
        )

        status = audit_run(self.root)["analysis_status"]

        self.assertEqual(status["status"], "exploratory")
        self.assertFalse(status["evaluates_preregistered_rule_B"])


if __name__ == "__main__":
    unittest.main()

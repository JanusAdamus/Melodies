"""Aclaración 2: una corrida sin VOMM no evalúa la regla de B y se declara exploratoria.

La familia de Holm sigue siendo la preregistrada (cuatro modelos, seis pares): sin
el VOMM sus tres pares quedan no evaluables en vez de encoger la familia a tres.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Comparacion.runner import _analysis_status, _comparison_family_models, run_learning_curve_experiment
from tests.test_comparacion import _runner_config, _standard_model_mocks, _synthetic_preparation


class FamilyWithoutVommTests(unittest.TestCase):
    def test_the_family_keeps_the_four_preregistered_models(self) -> None:
        self.assertEqual(
            _comparison_family_models(SimpleNamespace(include_vomm_control=False)),
            ["finite_hmm", "hdp_hmm", "transformer", "vomm"],
        )

    def test_without_vomm_the_run_is_exploratory(self) -> None:
        status = _analysis_status(SimpleNamespace(include_vomm_control=False))
        self.assertEqual(status["status"], "exploratory")
        self.assertFalse(status["evaluates_preregistered_rule_B"])
        self.assertIn("vomm", status["reason"])

    def test_with_vomm_the_run_is_the_preregistered_one(self) -> None:
        status = _analysis_status(SimpleNamespace(include_vomm_control=True))
        self.assertEqual(status["status"], "preregistered")
        self.assertTrue(status["evaluates_preregistered_rule_B"])


class RunnerOutputsWithoutVommTests(unittest.TestCase):
    def test_every_decision_output_is_marked_exploratory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _runner_config(temporary_directory, include_vomm_control=False)
            mocks = _standard_model_mocks()
            with (
                patch("Comparacion.runner.prepare_corpus", return_value=_synthetic_preparation()),
                patch("Comparacion.runner.FiniteGlobalHMM", return_value=mocks["finite"]),
                patch("Comparacion.runner.GlobalHDPHMM", return_value=mocks["hdp"]),
                patch("Comparacion.runner.SmallTransformerNextTokenModel", return_value=mocks["transformer"]),
                patch(
                    "Comparacion.runner._build_transformer_dataloaders",
                    side_effect=lambda config, train_pieces, validation_pieces, test_pieces, **_: {
                        "train": train_pieces,
                        "validation": validation_pieces,
                        "test": test_pieces,
                    },
                ),
            ):
                result = run_learning_curve_experiment(config, run_name="sin-vomm", max_files=12)

            root = Path(result["output_root"])
            pairwise = json.loads((root / "pairwise_comparisons.json").read_text(encoding="utf-8"))
            self.assertEqual(len(pairwise["comparisons"]), 6)
            self.assertEqual(pairwise["holm_family"]["work_mean_ci"]["family_size"], 6)
            for name in (
                "pairwise_comparisons.json",
                "pareto_summary.json",
                "headline_estimates.json",
                "execution_plan.json",
                "run_summary.json",
            ):
                with self.subTest(output=name):
                    payload = json.loads((root / name).read_text(encoding="utf-8"))
                    self.assertEqual(payload["analysis_status"]["status"], "exploratory")
            self.assertEqual(result["analysis_status"]["status"], "exploratory")
            # El reporte de auditoría de la corrida también lo dice.
            from Comparacion.artifact_audit import audit_run

            report = audit_run(root)
            self.assertEqual(report["analysis_status"]["status"], "exploratory")
            self.assertFalse(report["analysis_status"]["evaluates_preregistered_rule_B"])


if __name__ == "__main__":
    unittest.main()

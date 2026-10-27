"""Las figuras leen las decisiones de los artefactos; no las recalculan (C5, C7, Aclaración 2)."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("figuras_tesis_bajo_prueba", REPO / "scripts" / "figuras_tesis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY_FIELDS = [
    "model",
    "frac",
    "mean_n_train_tokens",
    "mean_test_ppl",
    "std_test_ppl",
    "mean_test_nll",
    "std_test_nll",
    "runs",
    "token_weighted_ppl",
    "test_ppl_ci95_low",
    "test_ppl_ci95_high",
]


def _summary_rows() -> list[dict]:
    rows = []
    for model, base in (("transformer", 6.0), ("finite_hmm", 7.0), ("hdp_hmm", 8.0), ("vomm", 8.1)):
        for frac, tokens in ((0.1, 1000.0), (1.0, 10000.0)):
            ppl = base + (0.5 if frac == 0.1 else 0.0)
            rows.append(
                {
                    "model": model,
                    "frac": frac,
                    "mean_n_train_tokens": tokens,
                    # Media entre corridas: distinta del estimando de C7 a propósito.
                    "mean_test_ppl": ppl + 0.3,
                    "std_test_ppl": 0.01,
                    "mean_test_nll": 2.0,
                    "std_test_nll": 0.01,
                    "runs": 6,
                    "token_weighted_ppl": ppl,
                    "test_ppl_ci95_low": ppl - 0.2,
                    "test_ppl_ci95_high": ppl + 0.25,
                }
            )
    return rows


def _write_summary(run_dir: Path, rows: list[dict], fields=SUMMARY_FIELDS) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _candidate(model: str, nll: float, cost: float) -> dict:
    return {
        "model": model,
        "work_mean_test_nll": nll,
        "work_mean_test_nll_interval": [nll - 0.4, nll + 0.4],
        "mean_test_nll": nll,
        "mean_selected_fit_wall_clock_s": cost,
        "mean_selected_fit_wall_clock_s_interval": [cost * 0.9, cost * 1.1],
        "devices": ["cpu"],
    }


def _pareto_payload() -> dict:
    # Por punto, finite_hmm (2.001) queda dominado por transformer (2.000) al mismo
    # costo; con la regla preregistrada empatan y los dos están en la frontera.
    candidates = [
        _candidate("transformer", 2.000, 10.0),
        _candidate("finite_hmm", 2.001, 10.0),
        _candidate("hdp_hmm", 2.5, 50.0),
        _candidate("vomm", 2.6, 60.0),
    ]
    return {
        "analysis_status": {"status": "preregistered"},
        "predictive_cost_partial_frontier": {
            "status": "ok",
            "axes": ["work_mean_test_nll", "mean_selected_fit_wall_clock_s"],
            "nll_estimand": "mean_over_works_of_work_nll_difference",
            "nll_paired_pairs_skipped": [{"pair": ["hdp_hmm", "vomm"], "reason": "too_few_paired_works", "n_pairs": 3}],
            "devices_differ": False,
            "candidates": candidates,
            "frontier": [candidates[0], candidates[1]],
        },
    }


def _comparison(model_a: str, model_b: str, *, ok: bool = True) -> dict:
    if not ok:
        return {
            "model_a": model_a,
            "model_b": model_b,
            "status": "unavailable",
            "n_pairs": 0,
            "mean_difference": None,
            "bootstrap_95_ci": None,
            "p_value_holm": None,
            "work_mean_holm_decision_ci": None,
            "work_mean_holm_rejects_zero": None,
        }
    return {
        "model_a": model_a,
        "model_b": model_b,
        "status": "ok",
        "n_pairs": 40,
        "mean_difference": -0.1,
        "bootstrap_95_ci": [-0.12, -0.08],
        "p_value_holm": 0.001,
        "work_mean_holm_ci": [-0.13, -0.07],
        "work_mean_holm_decision_ci": [-0.13, -0.07],
        "work_mean_holm_rejects_zero": True,
    }


class FigureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.module = _load_module()
        self.module.COMPARACION = root / "Comparacion"
        self.module.OUTPUT_DIR = root / "figuras"
        self.module.aplicar_estilo()
        self.original = self.module.COMPARACION / self.module.ORIGINAL

    def tearDown(self) -> None:
        import matplotlib.pyplot as plt

        plt.close("all")
        self._tmp.cleanup()


class ParetoFigureTests(FigureTestCase):
    def test_the_frontier_comes_from_pareto_summary_not_from_point_estimates(self) -> None:
        self.original.mkdir(parents=True)
        (self.original / "pareto_summary.json").write_text(json.dumps(_pareto_payload()), encoding="utf-8")

        drawn = self.module.figura_pareto()

        self.assertEqual(drawn["frontera"], ["transformer", "finite_hmm"])
        self.assertEqual(drawn["pares_sin_decidir"], [["hdp_hmm", "vomm"]])
        self.assertTrue((self.module.OUTPUT_DIR / "fig5_pareto_prediccion_costo.png").is_file())

    def test_an_old_pareto_summary_is_rejected(self) -> None:
        self.original.mkdir(parents=True)
        old = {"predictive_cost_partial_frontier": {"status": "ok", "candidates": [], "frontier": []}}
        (self.original / "pareto_summary.json").write_text(json.dumps(old), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "pareto_summary"):
            self.module.figura_pareto()


class PairedFigureTests(FigureTestCase):
    def test_a_pair_without_data_is_skipped_not_a_type_error(self) -> None:
        pairs = [("finite_hmm", "hdp_hmm"), ("finite_hmm", "transformer"), ("hdp_hmm", "transformer")]
        for run in [self.module.ORIGINAL, "sens_stride128", "sens_hmm_grid", "sens_split17", "sens_split29"]:
            run_dir = self.module.COMPARACION / run
            run_dir.mkdir(parents=True)
            comparisons = [_comparison(a, b) for a, b in pairs]
            comparisons.append(_comparison("transformer", "vomm", ok=False))
            (run_dir / "pairwise_comparisons.json").write_text(json.dumps({"comparisons": comparisons}), encoding="utf-8")

        drawn = self.module.figura_comparaciones_pareadas()

        self.assertEqual(drawn["pares"], pairs)
        self.assertIn(("transformer", "vomm"), drawn["sin_datos"])
        # El intervalo dibujado es el de Holm por obra, no el 95 % sin ajustar.
        self.assertEqual(drawn["intervalos_original"][("finite_hmm", "hdp_hmm")], [-0.13, -0.07])


class LearningCurveFigureTests(FigureTestCase):
    def test_the_curve_draws_the_c7_estimand_with_its_interval(self) -> None:
        _write_summary(self.original, _summary_rows())

        drawn = self.module.figura_curva_aprendizaje()

        x, y, low, high = drawn["transformer"][-1]
        self.assertEqual((y, low, high), (6.0, 5.8, 6.25))

    def test_the_perplexity_by_model_is_the_c7_estimand(self) -> None:
        _write_summary(self.original, _summary_rows())

        values = self.module.ppl_en_frac_1(self.module.ORIGINAL)

        self.assertEqual(values["transformer"], 6.0)

    def test_a_summary_without_the_c7_columns_is_rejected(self) -> None:
        _write_summary(self.original, _summary_rows(), fields=SUMMARY_FIELDS[:8])

        with self.assertRaisesRegex(ValueError, "token_weighted_ppl"):
            self.module.figura_curva_aprendizaje()


SENSITIVITY_RUNS = ("sens_stride128", "sens_hmm_grid", "sens_split17", "sens_split29")


class RunWithoutVommTests(FigureTestCase):
    """Una corrida sin VOMM es exploratoria pero sus figuras no deben romperse."""

    def _without_vomm(self) -> list[dict]:
        return [row for row in _summary_rows() if row["model"] != "vomm"]

    def test_the_learning_curve_draws_only_the_models_present(self) -> None:
        _write_summary(self.original, self._without_vomm())

        drawn = self.module.figura_curva_aprendizaje()

        self.assertEqual(set(drawn), {"transformer", "finite_hmm", "hdp_hmm"})
        self.assertTrue((self.module.OUTPUT_DIR / "fig1_curva_aprendizaje.png").is_file())

    def test_the_sensitivity_figure_draws_only_the_models_present(self) -> None:
        for run in (self.module.ORIGINAL, *SENSITIVITY_RUNS):
            _write_summary(self.module.COMPARACION / run, self._without_vomm())

        drawn = self.module.figura_sensibilidades()

        self.assertEqual(drawn, ["transformer", "finite_hmm", "hdp_hmm"])
        self.assertTrue((self.module.OUTPUT_DIR / "fig3_sensibilidades.png").is_file())

    def test_the_unit_of_analysis_figure_is_skipped_without_vomm(self) -> None:
        rows = [row for row in _unit_rows(n_works=12, vomm_effect=0.05) if row["model"] != "vomm"]
        _write_unit_run(self.original, rows)

        self.assertIsNone(self.module.figura_unidad_de_analisis())
        self.assertFalse((self.module.OUTPUT_DIR / "fig6_unidad_de_analisis.png").exists())


PIECE_FIELDS = ["piece_id", "frac", "canonical_work_id", "model", "data_seed", "model_seed",
                "n_tokens", "log_likelihood", "nll_per_token"]


def _unit_rows(*, n_works: int, vomm_effect: float, long_penalty: float = 0.0) -> list[dict]:
    """Obras de longitud creciente; la primera tiene dos archivos de distinta longitud.

    vomm_effect > 0 hace al VOMM mejor (HDP − VOMM > 0); long_penalty le resta esa
    ventaja en proporción a la longitud, para que lo ponderado por token difiera.
    """

    import numpy as np

    rng = np.random.default_rng(4)
    rows = []
    for index in range(n_works):
        files = [(f"w{index}a.xml", 20 + 15 * index)]
        if index == 0:
            files.append(("w0b.xml", 400))
        for model_seed in (1, 2):
            for piece_id, tokens in files:
                base = 2.0 + 0.2 * rng.standard_normal()
                for model, nll in (
                    ("hdp_hmm", base),
                    ("vomm", base - vomm_effect + long_penalty * tokens / 100.0 + 0.01 * rng.standard_normal()),
                ):
                    rows.append({
                        "piece_id": piece_id, "frac": "1.0", "canonical_work_id": f"w{index}",
                        "model": model, "data_seed": 1, "model_seed": model_seed,
                        "n_tokens": tokens, "log_likelihood": -nll * tokens, "nll_per_token": nll,
                    })
    return rows


def _write_unit_run(run_dir: Path, rows: list[dict]) -> dict:
    """Escribe piece_metrics_raw.csv y el pairwise_comparisons.json que el runner sacaría de él."""

    from Comparacion.statistics import pairwise_model_comparisons

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "piece_metrics_raw.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIECE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = pairwise_model_comparisons(
        rows, bootstrap_samples=400, seed=0,
        expected_models=["finite_hmm", "hdp_hmm", "transformer", "vomm"],
    )
    (run_dir / "pairwise_comparisons.json").write_text(json.dumps(payload), encoding="utf-8")
    return next(c for c in payload["comparisons"] if (c["model_a"], c["model_b"]) == ("hdp_hmm", "vomm"))


class UnitOfAnalysisFigureTests(FigureTestCase):
    """fig6 lee los dos estimandos y sus IC de Holm de pairwise_comparisons.json (ronda 2)."""

    def test_both_estimands_and_their_holm_intervals_come_from_the_artifact(self) -> None:
        comparison = _write_unit_run(self.original, _unit_rows(n_works=16, vomm_effect=0.05))

        drawn = self.module.figura_unidad_de_analisis()

        self.assertEqual(drawn["por_obra"], comparison["mean_difference"])
        self.assertEqual(drawn["por_evento"], comparison["token_weighted_difference"])
        self.assertEqual(drawn["ic_por_obra"], comparison["work_mean_holm_decision_ci"])
        self.assertEqual(drawn["ic_por_evento"], comparison["token_weighted_holm_decision_ci"])
        self.assertEqual(drawn["n_obras"], comparison["n_pairs"])
        # El panel A usa la misma NLL por obra que decide: su media es la del artefacto.
        self.assertAlmostEqual(drawn["media_panel_a"], comparison["mean_difference"], places=12)
        self.assertTrue((self.module.OUTPUT_DIR / "fig6_unidad_de_analisis.png").is_file())

    def test_the_title_and_labels_follow_the_holm_verdicts(self) -> None:
        # r2sw_runs/A: el VOMM gana en los dos estimandos; el título no puede decir que pierde.
        comparison = _write_unit_run(self.original, _unit_rows(n_works=16, vomm_effect=0.08))
        self.assertTrue(comparison["work_mean_holm_rejects_zero"])
        self.assertTrue(comparison["token_weighted_holm_rejects_zero"])

        drawn = self.module.figura_unidad_de_analisis()

        self.assertEqual(drawn["ganadores"], ["VOMM", "VOMM"])
        self.assertNotIn("pierde", drawn["titulo"])
        self.assertIn("16 obras", drawn["metodo"])
        self.assertNotIn("414", drawn["metodo"])

    def test_a_difference_holm_does_not_reject_is_a_tie(self) -> None:
        comparison = _write_unit_run(self.original, _unit_rows(n_works=16, vomm_effect=0.0))
        self.assertFalse(comparison["work_mean_holm_rejects_zero"])

        drawn = self.module.figura_unidad_de_analisis()

        self.assertEqual(drawn["ganadores"][0], "empate")
        self.assertNotIn("gana", drawn["titulo"].split(";")[0])

    def test_the_axis_covers_the_intervals_it_draws(self) -> None:
        _write_unit_run(self.original, _unit_rows(n_works=16, vomm_effect=0.6))

        drawn = self.module.figura_unidad_de_analisis()

        low, high = drawn["xlim"]
        for interval in (drawn["ic_por_obra"], drawn["ic_por_evento"]):
            self.assertLessEqual(low, min(interval[0], 0.0))
            self.assertGreaterEqual(high, max(interval[1], 0.0))

    def test_a_pair_with_too_few_works_is_skipped_with_its_reason(self) -> None:
        _write_unit_run(self.original, _unit_rows(n_works=3, vomm_effect=0.05))

        self.assertIsNone(self.module.figura_unidad_de_analisis())
        self.assertFalse((self.module.OUTPUT_DIR / "fig6_unidad_de_analisis.png").exists())

    def test_too_few_works_are_skipped_even_if_the_file_carries_an_interval(self) -> None:
        # r2sw_runs/C: un pairwise de antes del mínimo de Holm trae IC con 3 obras.
        _write_unit_run(self.original, _unit_rows(n_works=3, vomm_effect=0.05))
        payload = json.loads((self.original / "pairwise_comparisons.json").read_text(encoding="utf-8"))
        for item in payload["comparisons"]:
            if (item["model_a"], item["model_b"]) == ("hdp_hmm", "vomm"):
                for prefix in ("work_mean", "token_weighted"):
                    item[f"{prefix}_holm_decision_ci"] = [0.01, 0.09]
                    item[f"{prefix}_holm_rejects_zero"] = True
        (self.original / "pairwise_comparisons.json").write_text(json.dumps(payload), encoding="utf-8")

        self.assertIsNone(self.module.figura_unidad_de_analisis())
        self.assertFalse((self.module.OUTPUT_DIR / "fig6_unidad_de_analisis.png").exists())

    def test_an_old_pairwise_file_is_rejected(self) -> None:
        _write_unit_run(self.original, _unit_rows(n_works=16, vomm_effect=0.05))
        payload = json.loads((self.original / "pairwise_comparisons.json").read_text(encoding="utf-8"))
        for item in payload["comparisons"]:
            item.pop("token_weighted_holm_decision_ci", None)
        (self.original / "pairwise_comparisons.json").write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "pairwise_comparisons"):
            self.module.figura_unidad_de_analisis()


if __name__ == "__main__":
    unittest.main()

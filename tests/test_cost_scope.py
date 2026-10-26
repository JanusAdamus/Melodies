"""C1 (auditoría 2026-10-26): costo comparable entre familias.

El HMM cobraba la selección completa en CPU y el transformador un solo ajuste en
GPU sin su búsqueda. Ahora cada fila separa búsqueda completa, ajuste
seleccionado, evaluación y memoria con su dispositivo; la búsqueda de
configuración del transformador, hecha fuera de la corrida, se registra cuando
se declara y, si no, queda marcada como no medida. Las figuras de costo comparan
ajuste seleccionado contra ajuste seleccionado.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Comparacion.cli import build_parser
from Comparacion.config import build_default_learning_curve_config
from Comparacion.runner import _engineering_cost_rows, _search_cost_fields

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_figures_module():
    spec = importlib.util.spec_from_file_location("figuras_tesis", REPO_ROOT / "scripts" / "figuras_tesis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SearchCostFieldTests(unittest.TestCase):
    def test_classical_search_is_the_measured_in_run_selection(self) -> None:
        config = build_default_learning_curve_config()

        fields = _search_cost_fields("finite_hmm", selection_wall_clock_s=12.0, device="cpu", config=config)

        self.assertEqual(fields["full_search_wall_clock_s"], 12.0)
        self.assertEqual(fields["full_search_status"], "measured_in_run")
        self.assertEqual(fields["full_search_device"], "cpu")
        self.assertIsNone(fields["external_search_wall_clock_s"])
        self.assertIn("capacity", fields["search_scope"])

    def test_transformer_search_outside_the_run_is_marked_not_measured(self) -> None:
        config = build_default_learning_curve_config()

        fields = _search_cost_fields("transformer", selection_wall_clock_s=5.0, device="cuda", config=config)

        self.assertIsNone(fields["full_search_wall_clock_s"])
        self.assertEqual(fields["full_search_status"], "not_measured")
        self.assertEqual(fields["external_search_status"], "not_measured")
        self.assertIn("epochs", fields["search_scope"])

    def test_declared_transformer_search_is_added_to_the_in_run_selection(self) -> None:
        config = replace(
            build_default_learning_curve_config(),
            transformer_search_wall_clock_s=3600.0,
            transformer_search_device="cuda",
        )

        fields = _search_cost_fields("transformer", selection_wall_clock_s=5.0, device="cuda", config=config)

        self.assertEqual(fields["external_search_wall_clock_s"], 3600.0)
        self.assertEqual(fields["external_search_device"], "cuda")
        self.assertEqual(fields["full_search_wall_clock_s"], 3605.0)
        self.assertEqual(fields["full_search_status"], "in_run_plus_declared_external")

    def test_declared_search_time_requires_its_device(self) -> None:
        with self.assertRaises(ValueError):
            replace(build_default_learning_curve_config(), transformer_search_wall_clock_s=10.0)

    def test_cli_parses_the_declared_search(self) -> None:
        args = build_parser().parse_args(
            ["--transformer-search-wall-clock-s", "3600", "--transformer-search-device", "cuda"]
        )

        self.assertEqual(args.transformer_search_wall_clock_s, 3600.0)
        self.assertEqual(args.transformer_search_device, "cuda")

    def test_engineering_costs_carry_search_and_scope_columns(self) -> None:
        row = _engineering_cost_rows([{"model": "transformer", "full_search_status": "not_measured"}])[0]

        for column in (
            "full_search_wall_clock_s",
            "full_search_status",
            "full_search_device",
            "external_search_wall_clock_s",
            "search_scope",
            "selected_fit_wall_clock_s",
            "device",
        ):
            self.assertIn(column, row)
        self.assertEqual(row["full_search_status"], "not_measured")


class FigureCostTests(unittest.TestCase):
    def test_cost_figure_reads_the_selected_fit_not_the_mixed_fit_column(self) -> None:
        figures = _load_figures_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "run"
            run.mkdir()
            (run / "engineering_costs.csv").write_text(
                "model,frac,fit_wall_clock_s,selected_fit_wall_clock_s\n"
                "finite_hmm,1.0,600.0,30.0\n"
                "transformer,1.0,170.0,150.0\n"
                "transformer,0.5,99.0,99.0\n",
                encoding="utf-8",
            )
            with patch.object(figures, "COMPARACION", Path(tmpdir)):
                costs = figures.costos_ajuste("run")

        self.assertEqual(costs, {"finite_hmm": 30.0, "transformer": 150.0})

    def test_cost_figure_refuses_runs_without_a_selected_fit_column(self) -> None:
        figures = _load_figures_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "run"
            run.mkdir()
            (run / "engineering_costs.csv").write_text(
                "model,frac,fit_wall_clock_s\nfinite_hmm,1.0,600.0\n", encoding="utf-8"
            )
            with patch.object(figures, "COMPARACION", Path(tmpdir)):
                with self.assertRaisesRegex(ValueError, "selected_fit_wall_clock_s"):
                    figures.costos_ajuste("run")


if __name__ == "__main__":
    unittest.main()

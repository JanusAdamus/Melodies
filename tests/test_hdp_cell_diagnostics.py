"""H7 (auditoría 2026-10-26): diagnóstico de cadenas por celda y sin burn-in.

Reproduce ``scripts/auditoria_2026_10_26/comparacion/diag_hdp.py``: (a) una cadena
estacionaria tras el burn-in no debe leerse como deriva por incluir el burn-in
en la ventana temprana; (b) celdas distintas con la misma ``model_seed`` no son
una cadena.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from Comparacion.hdp_diagnostics import (
    build_cell_chain_diagnostics,
    build_chain_diagnostics,
    write_cell_traces,
)

BURN_IN = 60
N_ITERS = 120


def _chain(seed: int, **cell: object) -> list[dict[str, object]]:
    """Sube durante el burn-in y después es ruido iid alrededor de un nivel fijo."""

    rng = np.random.default_rng(seed)
    return [
        {
            **cell,
            "model_seed": seed,
            "iteration": index + 1,
            "log_likelihood": -70000 + 4000 * min(index, BURN_IN) / BURN_IN + rng.normal(0, 20),
            "active_states": 11,
        }
        for index in range(N_ITERS)
    ]


def _cell(data_seed: int, frac: float = 1.0) -> dict[str, object]:
    return {"split_seed": 7, "data_seed": data_seed, "frac": frac}


class CellChainDiagnosticsTests(unittest.TestCase):
    def test_burn_in_is_excluded_before_reading_drift(self) -> None:
        rows = _chain(1, **_cell(1))

        report = build_cell_chain_diagnostics(rows, burn_in=BURN_IN, min_iterations=N_ITERS - BURN_IN)

        (cell,) = report["cells"]
        chain = cell["diagnostics"]["chains"]["1"]
        self.assertEqual(chain["n_iterations"], N_ITERS - BURN_IN)
        self.assertTrue(chain["window_stability"])
        self.assertNotEqual(report["verdict"], "drift_detected")
        self.assertEqual(report["burn_in_excluded"], BURN_IN)

    def test_cells_sharing_a_model_seed_are_never_concatenated(self) -> None:
        rows = [row for data_seed in (1, 2, 3) for row in _chain(1, **_cell(data_seed))]

        report = build_cell_chain_diagnostics(rows, burn_in=BURN_IN, min_iterations=N_ITERS - BURN_IN)

        self.assertEqual(report["n_cells"], 3)
        for cell in report["cells"]:
            with self.subTest(cell=cell["cell"]):
                self.assertEqual(cell["diagnostics"]["n_chains"], 1)
                self.assertEqual(
                    cell["diagnostics"]["chains"]["1"]["n_iterations"], N_ITERS - BURN_IN
                )

    def test_model_seeds_within_one_cell_are_compared_as_chains(self) -> None:
        rows = _chain(1, **_cell(1)) + _chain(2, **_cell(1))

        report = build_cell_chain_diagnostics(rows, burn_in=BURN_IN, min_iterations=N_ITERS - BURN_IN)

        (cell,) = report["cells"]
        self.assertEqual(cell["diagnostics"]["n_chains"], 2)
        self.assertEqual(cell["diagnostics"]["verdict"], "stable_within_the_observed_window")
        self.assertFalse(report["convergence_claimed"])

    def test_overall_verdict_reports_the_worst_cell(self) -> None:
        drifting = [
            {**row, "log_likelihood": -70000 + 100 * row["iteration"]}
            for row in _chain(1, **_cell(2))
        ]
        rows = _chain(1, **_cell(1)) + _chain(2, **_cell(1)) + drifting

        report = build_cell_chain_diagnostics(rows, burn_in=BURN_IN, min_iterations=N_ITERS - BURN_IN)

        self.assertEqual(report["verdict"], "drift_detected")
        self.assertEqual(report["verdict_counts"]["drift_detected"], 1)

    def test_a_chain_with_repeated_iterations_is_rejected(self) -> None:
        concatenated = _chain(1, **_cell(1)) + _chain(1, **_cell(2))

        with self.assertRaises(ValueError):
            build_chain_diagnostics({1: concatenated}, min_iterations=1)

    def test_traces_are_written_in_long_format_with_the_burn_in_flag(self) -> None:
        rows = _chain(1, **_cell(1))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_cell_traces(rows, tmpdir, burn_in=BURN_IN)
            lines = Path(path).read_text(encoding="utf-8").splitlines()

        self.assertEqual(Path(path).name, "hdp_traces.csv")
        self.assertIn("in_burn_in", lines[0].split(","))
        self.assertEqual(len(lines), N_ITERS + 1)


if __name__ == "__main__":
    unittest.main()

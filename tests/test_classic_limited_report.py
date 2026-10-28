"""El CLI del experimento clásico limitado escribe sus salidas y su texto dice lo que dicen los datos.

- ``main()`` termina y escribe ``resumen_global.tex``, ``analysis_report.md`` y
  ``nota_tecnica.md`` con la columna 'Estados efectivos HMM finito'.
- ``hdp_effective_states`` es una media posterior (float): el reporte no la trunca.
- Las conclusiones (quién ajusta mejor, quién usa más estados) salen de los números
  y no de frases fijas.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import src.cli.classic_limited_analysis as cla

FIGURES = [
    Path("compare_log_likelihood_scatter.png"),
    Path("compare_log_likelihood_boxplot.png"),
    Path("compare_log_likelihood_gain_by_piece.png"),
    Path("compare_state_counts_scatter.png"),
    Path("compare_state_counts_by_piece.png"),
    Path("composer_state_complexity.png"),
]

# HDP peor en las dos obras y con más estados efectivos que el finito.
HDP_WORSE = pd.DataFrame(
    {
        "title": ["A", "B"],
        "composer": ["X", "Y"],
        "finite_log_likelihood": [-10.0, -12.0],
        "hdp_log_likelihood": [-11.0, -13.0],
        "finite_effective_states": [2.0, 2.0],
        "hdp_effective_states": [1.9667, 4.5],
    }
)

# HDP mejor en las dos obras y con menos estados efectivos que el finito.
HDP_BETTER = pd.DataFrame(
    {
        "title": ["A", "B"],
        "composer": ["X", "Y"],
        "finite_log_likelihood": [-11.0, -13.0],
        "hdp_log_likelihood": [-10.0, -12.0],
        "finite_effective_states": [4.0, 5.0],
        "hdp_effective_states": [2.25, 3.0],
    }
)


def _fake_run(path, obs, treble, params):
    return {
        "path": path,
        "finite_log_likelihood": -10.0,
        "finite_active_states": 5,
        "finite_effective_states": 3.0,
        "hdp_log_likelihood": -9.0,
        "hdp_effective_states": 1.6333,
        "hdp_mean_segment_length": 2.0,
        "finite_mean_segment_length": 2.0,
    }


class ClassicLimitedMainTests(unittest.TestCase):
    def test_main_writes_every_output(self) -> None:
        catalog = pd.DataFrame(
            {"path": ["a.mxl", "b.mxl"], "title": ["A", "B"], "composer": ["X", "Y"], "note_count": [10, 20]}
        )
        with tempfile.TemporaryDirectory() as output, patch.multiple(
            cla,
            build_library_catalog=lambda directory: catalog,
            _selected_catalog=lambda frame, limit: frame,
            summarize_catalog=lambda frame: {"catalog": frame},
            export_tables=lambda *args, **kwargs: None,
            generate_analysis_figures=lambda *args, **kwargs: [],
            _run_piece=_fake_run,
        ), patch.object(sys, "argv", ["classic_limited_analysis", "--output-dir", output]):
            cla.main()
            root = Path(output)
            for relative in ("analysis/resumen_global.tex", "analysis/analysis_report.md", "nota_tecnica.md"):
                self.assertTrue((root / relative).is_file(), relative)
            summary = (root / "analysis" / "resumen_global.tex").read_text(encoding="utf-8")
            self.assertIn("Estados efectivos medios HMM finito", summary)
            self.assertNotIn("Estados activos", summary)


class ClassicLimitedReportTests(unittest.TestCase):
    def test_fractional_effective_states_are_not_truncated(self) -> None:
        text = cla._build_detailed_report(HDP_WORSE, FIGURES)
        self.assertIn("'A' con 1.967 estados", text)
        self.assertNotIn("con 1 estados", text)

    def test_conclusions_follow_the_numbers_when_the_hdp_loses(self) -> None:
        text = cla._build_detailed_report(HDP_WORSE, FIGURES)
        for fixed in (
            "mejora el ajuste sobre todas las obras",
            "todos los puntos quedan por encima de la diagonal",
            "ninguna obra presenta ganancia negativa",
            "el HMM finito necesita mas estados activos",
            "estados activos del HMM finito",
            "Para Elisa",
            "mayor compacidad del HDP-HMM truncado",
        ):
            self.assertNotIn(fixed, text)
        self.assertIn("ninguna obra queda por encima de la diagonal", text)
        self.assertIn("el HDP-HMM truncado usa en promedio mas estados efectivos", text)

    def test_conclusions_follow_the_numbers_when_the_hdp_wins(self) -> None:
        text = cla._build_detailed_report(HDP_BETTER, FIGURES)
        self.assertIn("las 2 obras quedan por encima de la diagonal", text)
        self.assertIn("el HDP-HMM truncado usa en promedio menos estados efectivos", text)
        self.assertNotIn("estados activos del HMM finito", text)


if __name__ == "__main__":
    unittest.main()

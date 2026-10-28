"""Verificador de cifras de la tesis (tarea 3.4 del handoff 2026-10).

Extrae cada número de los ``.tex`` y lo empareja con una fuente (JSON de procedencia del
generador o un reporte); un número sin fuente es un error.  Los números estructurales
(capítulos, años de citas, ecuaciones, etiquetas...) se ignoran con reglas explícitas,
cada una probada aquí.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "verificar_cifras_tesis.py"


@pytest.fixture(scope="module")
def ver():
    spec = importlib.util.spec_from_file_location("verificar_cifras_tesis_bajo_prueba", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses busca su módulo en sys.modules
    spec.loader.exec_module(module)
    return module


def _texts(ver, tex: str) -> list[str]:
    return [n.texto for n in ver.extraer_numeros(tex).numeros]


def _ignored(ver, tex: str) -> list[tuple[str, str]]:
    return [(n.texto, n.regla) for n in ver.extraer_numeros(tex).ignorados]


def test_plain_numbers_decimals_and_thousands(ver):
    assert _texts(ver, "La NLL fue 2.21 nats en 3000 partituras y 10\\,000 remuestreos.") == ["2.21", "3000", "10000"]
    assert _texts(ver, "con coma decimal 2{,}21 y negativo $-0.07$") == ["2.21", "0.07"]


def test_comments_are_ignored(ver):
    assert _texts(ver, "Texto 1.5 % comentario 2.5\notra 3.5 y 4\\% literal") == ["1.5", "3.5", "4"]


def test_reference_like_commands_are_ignored(ver):
    tex = r"Ver \ref{tab:1} y \eqref{eq:2}, \label{sec:3} \cite{fox2011} \citep[p.~4]{x} \input{tables/t5} \url{http://a.b/6}"
    assert _texts(ver, tex) == []
    assert {rule for _, rule in _ignored(ver, tex)} == {"argumento_estructural"}


def test_display_math_is_ignored_and_inline_math_is_counted(ver):
    tex = "Antes 1.25\n\\begin{equation}\nx = 3.5 + 2\n\\end{equation}\n\\[ y = 7 \\]\ny en línea $K = 12$ con $x_{t-1}$ y $\\beta_2$."
    assert _texts(ver, tex) == ["1.25", "12"]
    rules = dict(_ignored(ver, tex))
    assert rules["3.5"] == "ecuacion" and rules["7"] == "ecuacion"
    assert ("1", "subindice") in _ignored(ver, tex) and ("2", "subindice") in _ignored(ver, tex)


def test_structural_references_are_ignored(ver):
    tex = "En el Capítulo 3, la Sección~4.2, las tablas 2 y 3, la Figura 5, la Fase 2 y la Enmienda 4."
    assert _texts(ver, tex) == []
    assert {rule for _, rule in _ignored(ver, tex)} == {"referencia_estructural"}


def test_citation_years_and_dates_are_ignored(ver):
    tex = "Según Fox et al., 2011 y Fox et al. (2011), Rabiner (1989); el 26 de octubre de 2026 y el 2026-10-26."
    assert _texts(ver, tex) == []
    rules = {rule for _, rule in _ignored(ver, tex)}
    assert rules == {"anio_de_cita", "fecha"}


def test_a_year_like_count_is_not_a_year(ver):
    assert _texts(ver, "Se usaron 1837 tokens de entrenamiento.") == ["1837"]


def test_identifiers_lengths_and_catalogue_numbers_are_ignored(ver):
    tex = (r"El hallazgo H10, la corrida pdmx50, el commit 984dfde, \texttt{obra_12.mxl}, "
           r"\includegraphics[width=0.8\textwidth]{fig1}, \hspace{2mm}, \vspace{1em}, el Minueto BWV Anh. 114 y la Op. 28")
    assert _texts(ver, tex) == []


def test_table_structure_is_ignored(ver):
    tex = "\\begin{tabular}{lrr}\n\\toprule\nA & \\multicolumn{2}{c}{B} \\\\\n\\cmidrule(lr){2-3}\nx & 1.5 & 2.5 \\\\\n\\end{tabular}"
    assert _texts(ver, tex) == ["1.5", "2.5"]


def test_pending_markers_are_listed_not_counted(ver):
    result = ver.extraer_numeros("La NLL es \\pendiente{cifra de la Fase 2, tarea 2.1: 3 modelos}.")
    assert result.numeros == []
    assert result.pendientes == ["cifra de la Fase 2, tarea 2.1: 3 modelos"]


def test_only_the_document_body_is_scanned(ver):
    tex = "\\documentclass[12pt]{book}\n\\setcounter{tocdepth}{2}\n\\begin{document}\nResultado 2.5\n\\end{document}"
    assert _texts(ver, tex) == ["2.5"]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _provenance(path: Path, table: str, entries) -> Path:
    return _write(path, json.dumps({"tabla": table, "cifras": [
        {"texto": t, "valor": v, "archivo": "corrida/x.json", "campo": "c", "fila": {}} for t, v in entries]}))


def test_numbers_match_rounded_sources_and_unmatched_are_errors(ver, tmp_path):
    thesis = tmp_path / "tesis"
    _write(thesis / "chapters" / "cap4.tex", "La NLL del transformador es 2.21 y la del HDP 2.30.\n")
    prov = _provenance(tmp_path / "p.procedencia.json", "t.tex", [("2.209", 2.2086)])
    report = ver.verificar(thesis, procedencias=[prov])
    assert report["n_numeros_con_fuente"] == 1
    assert [(e["texto"], e["linea"]) for e in report["errores"]] == [("2.30", 1)]
    assert report["errores"][0]["archivo"] == "chapters/cap4.tex"


def test_report_numbers_are_sources(ver, tmp_path):
    thesis = tmp_path / "tesis"
    _write(thesis / "cap5.tex", "Se analizaron 312 obras con NLL 2.2 y 90\\,\\% del entrenamiento.\n")
    report_md = _write(tmp_path / "resultados.md", "| obras | 312 |\n| NLL | 2.2086 |\n| fracción | 0.9 |\n")
    report = ver.verificar(thesis, reportes=[report_md])
    assert report["errores"] == []
    assert report["n_numeros_con_fuente"] == 3


def test_integer_does_not_match_a_rounded_decimal(ver, tmp_path):
    thesis = tmp_path / "tesis"
    _write(thesis / "a.tex", "Hubo 2 casos.\n")
    report_md = _write(tmp_path / "r.md", "valor 2.2086\n")
    assert [e["texto"] for e in ver.verificar(thesis, reportes=[report_md])["errores"]] == ["2"]


def test_hand_edited_generated_table_is_an_error(ver, tmp_path):
    thesis = tmp_path / "tesis"
    _write(thesis / "chapters" / "tables" / "pdmx_resultados.tex", "x & 2.21 & 2.30 \\\\\n")
    prov = _provenance(tmp_path / "pdmx_resultados.procedencia.json", "pdmx_resultados.tex", [("2.21", 2.21), ("2.31", 2.31)])
    report = ver.verificar(thesis, procedencias=[prov])
    kinds = {e["tipo"] for e in report["errores"]}
    assert "tabla_editada" in kinds


def test_cli_exit_code_and_report_file(tmp_path):
    thesis = tmp_path / "tesis"
    _write(thesis / "a.tex", "Valor 7.5 sin fuente y \\pendiente{falta la cifra}.\n")
    out = tmp_path / "reporte.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--tesis", str(thesis), "--salida", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [e["texto"] for e in data["errores"]] == ["7.5"]
    assert data["pendientes"] == [{"archivo": "a.tex", "linea": 1, "texto": "falta la cifra"}]

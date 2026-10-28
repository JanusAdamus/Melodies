"""Verificador de cifras de la tesis (tarea 3.4 del handoff 2026-10).

Extrae cada número de los ``.tex`` y lo empareja con una fuente (JSON de procedencia del
generador o un reporte); un número sin fuente es un error.  Un número suelto sólo tiene
fuente si su texto, con signo, es el de una cifra; una cifra redondeada se cita con
``\\cifra{<id>}{<número>}``.  La procedencia se recalcula desde los artefactos.  Los
números estructurales (capítulos, años de citas, ecuaciones, etiquetas...) se ignoran
con reglas explícitas, cada una probada aquí.
"""

from __future__ import annotations

import hashlib
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
    return [("-" if n.negativo else "") + n.texto for n in ver.extraer_numeros(tex).numeros]


def _ignored(ver, tex: str) -> list[tuple[str, str]]:
    return [(n.texto, n.regla) for n in ver.extraer_numeros(tex).ignorados]


# --- extracción --------------------------------------------------------------------------


def test_plain_numbers_decimals_and_thousands(ver):
    assert _texts(ver, "La NLL fue 2.21 nats en 3000 partituras y 10\\,000 remuestreos.") == ["2.21", "3000", "10000"]
    assert _texts(ver, "con coma decimal 2{,}21 y negativo $-0.07$") == ["2.21", "-0.07"]


def test_sign_is_read(ver):
    assert _texts(ver, r"$-$0.100 y $-0.061$ y -3 y −2.5 y $+0.4$") == ["-0.100", "-0.061", "-3", "-2.5", "0.4"]
    # un guion entre números es un rango, no un signo
    assert _texts(ver, "entre 0.1--0.3 y 2-3 y [$-$0.2, $-$0.05]") == ["0.1", "0.3", "2", "3", "-0.2", "-0.05"]


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
    tex = "En el Capítulo 3, la Sección~4.2, las tablas 2 y 3, las secciones 4.1, 4.2 y 5.3, la Figura 5, la Fase 2 y la Enmienda 4."
    assert _texts(ver, tex) == []
    assert {rule for _, rule in _ignored(ver, tex)} == {"referencia_estructural"}


@pytest.mark.parametrize(
    ("tex", "expected"),
    [
        ("Como muestra la tabla 3, 0.42 nats separan a los modelos.", ["0.42"]),
        ("En la figura 2 y 37 obras el efecto es 0.9.", ["37", "0.9"]),
        ("Las tablas 2 y 3, 0.42 nats.", ["0.42"]),
        ("Por ejemplo 0.42 nats separan a los modelos.", ["0.42"]),
        ("En el ejemplo 3 hay 12 obras.", ["3", "12"]),
    ],
)
def test_a_result_after_a_reference_is_not_swallowed(ver, tex, expected):
    assert _texts(ver, tex) == expected


def test_citation_years_and_dates_are_ignored(ver):
    tex = ("Según Fox et al., 2011 y Fox et al. (2011), Rabiner (1989), (Fox, 2011), (Fox y Sudderth, 2011; Rabiner, 1989a); "
           "el 26 de octubre de 2026 y el 2026-10-26.")
    assert _texts(ver, tex) == []
    rules = {rule for _, rule in _ignored(ver, tex)}
    assert rules == {"anio_de_cita", "fecha"}


@pytest.mark.parametrize("tex", ["Se usaron 1837 tokens de entrenamiento.", "En MuseTrainer, 1999 obras pasaron el filtro."])
def test_a_year_like_count_is_not_a_year(ver, tex):
    assert _texts(ver, tex) == [t for t in ("1837", "1999") if t in tex]


def test_identifiers_lengths_and_catalogue_numbers_are_ignored(ver):
    tex = (r"El hallazgo H10, la corrida pdmx50, el commit 984dfde, \texttt{obra_12.mxl}, "
           r"\includegraphics[width=0.8\textwidth]{fig1}, \hspace{2mm}, \vspace{1em}, el Minueto BWV Anh. 114 y la Op. 28")
    assert _texts(ver, tex) == []


def test_a_bare_number_in_texttt_is_counted(ver):
    assert _texts(ver, r"La NLL es \texttt{9.876} nats y \texttt{$-$0.5}.") == ["9.876", "-0.5"]


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


def test_cifra_attaches_its_citation(ver):
    numbers = ver.extraer_numeros(r"Gana por \cifra{t:a.c_1}{$-0.06$} nats y 3 obras.").numeros
    assert [(n.texto, n.negativo, n.cita) for n in numbers] == [("0.06", True, "t:a.c_1"), ("3", False, None)]


# --- fuentes de prueba -------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


DATOS = {"a": {"b": 2.2086, "c": 0.061}, "l": [0.487, 0.9, 12]}


def _run(tmp_path: Path, datos=DATOS) -> Path:
    run = tmp_path / "runs" / "corrida_x"
    _write(run / "datos.json", json.dumps(datos))
    return run


def _provenance(tmp_path: Path, table: str, entries, *, run: Path | None = None) -> Path:
    """``entries``: (texto, valor, ruta) o (texto, valor, ruta, operacion)."""

    run = _run(tmp_path) if run is None else run
    sha = hashlib.sha256((run / "datos.json").read_bytes()).hexdigest()
    cifras = []
    for entry in entries:
        texto, valor, ruta, *op = entry
        item = {"id": f"{table.removesuffix('.tex')}:{'.'.join(map(str, ruta))}{':' + op[0] if op else ''}",
                "texto": texto, "valor": valor, "archivo": "corrida_x/datos.json", "campo": ".".join(map(str, ruta)),
                "fila": {}, "ruta": list(ruta)}
        if op:
            item["operacion"] = op[0]
        cifras.append(item)
    return _write(tmp_path / "prov" / table.replace(".tex", ".procedencia.json"), json.dumps(
        {"tabla": table, "corrida": run.resolve().as_posix(), "fuentes_sha256": {"corrida_x/datos.json": sha}, "cifras": cifras}))


STANDARD = [("2.209", 2.2086, ("a", "b")), ("0.061", 0.061, ("a", "c")), ("-0.061", -0.061, ("a", "c"), "negado"),
            ("0.487", 0.487, ("l", 0)), ("0.90", 0.9, ("l", 1)), ("12", 12, ("l", 2))]


def _check(ver, tmp_path, text: str, entries=STANDARD, **kwargs) -> dict:
    thesis = tmp_path / "tesis"
    _write(thesis / "cap.tex", text)
    return ver.verificar(thesis, procedencias=[_provenance(tmp_path, "t.tex", entries)], **kwargs)


def _errors(report) -> list[tuple[str, str | None]]:
    return [(e["tipo"], e.get("texto")) for e in report["errores"]]


# --- emparejamiento ----------------------------------------------------------------------


def test_bare_numbers_must_match_a_source_text_exactly(ver, tmp_path):
    report = _check(ver, tmp_path, "La NLL es 2.209 con 12 obras; el HDP 2.30.\n")
    assert report["n_numeros_con_fuente"] == 2
    assert _errors(report) == [("sin_fuente", "2.30")]
    assert report["errores"][0]["archivo"] == "cap.tex" and report["errores"][0]["linea"] == 1


def test_a_made_up_number_does_not_borrow_an_unrelated_source(ver, tmp_path):
    # 0.5 redondea a 0.487 (un tiempo de ajuste) y 2.21 a 2.2086: sin cita, no cuentan
    report = _check(ver, tmp_path, "El transformador gana por 0.5 y la NLL es 2.21.\n")
    assert _errors(report) == [("sin_fuente", "0.5"), ("sin_fuente", "2.21")]


def test_the_sign_must_match(ver, tmp_path):
    report = _check(ver, tmp_path, "transformador menos finito es $-0.487$: el transformador mejora; luego $-0.061$ y 0.061.\n")
    assert _errors(report) == [("sin_fuente", "-0.487")]


def test_a_cited_number_may_be_rounded_from_its_source(ver, tmp_path):
    report = _check(ver, tmp_path, "La NLL es \\cifra{t:a.b}{2.21} y \\cifra{t:l.1}{90}\\,\\% del total.\n")
    assert report["errores"] == []
    assert report["n_numeros_con_fuente"] == 2


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("La NLL es \\cifra{t:a.b}{2.30}.", "cita_no_coincide"),
        ("La NLL es \\cifra{t:a.b}{$-2.21$}.", "cita_no_coincide"),
        ("La NLL es \\cifra{t:no.existe}{2.21}.", "cita_desconocida"),
    ],
)
def test_a_wrong_citation_is_an_error(ver, tmp_path, text, kind):
    assert [e["tipo"] for e in _check(ver, tmp_path, text + "\n")["errores"]] == [kind]


def test_percent_matches_a_fraction_exactly_and_plain_text(ver, tmp_path):
    report = _check(ver, tmp_path, "El 90\\,\\% y el 0.90 del entrenamiento; 91\\,\\% no.\n")
    assert _errors(report) == [("sin_fuente", "91")]


def test_report_numbers_are_sources_as_written(ver, tmp_path):
    thesis = tmp_path / "tesis"
    _write(thesis / "cap5.tex", "Se analizaron 312 obras con NLL \\cifra{reporte:resultados.md:2}{2.21}, 2.2 y 90\\,\\% del entrenamiento.\n")
    report_md = _write(tmp_path / "resultados.md", "| obras | 312 |\n| NLL | 2.2086 |\n| fracción | 0.9 |\n")
    report = ver.verificar(thesis, reportes=[report_md])
    assert _errors(report) == [("sin_fuente", "2.2")]
    assert report["n_numeros_con_fuente"] == 3


def test_random_numbers_rarely_find_a_source(ver, tmp_path):
    import random

    entries = [(f"{v:.3f}", v, ("l", i)) for i, v in enumerate(round(0.013 * k + 0.001, 3) for k in range(150))]
    run = _run(tmp_path, {"l": [e[1] for e in entries]})
    fuentes = ver.cargar_fuentes([_provenance(tmp_path, "t.tex", entries, run=run)], [])
    random.seed(0)
    hits = 0
    for _ in range(2000):
        decimals = random.choice((1, 2))
        text = f"{random.uniform(0, 2):.{decimals}f}"
        hits += fuentes.buscar(ver.Numero(text, float(text), decimals, 1, False, False, None, 0)) is not None
    assert hits == 0  # ningún texto de la procedencia tiene 1 o 2 decimales


# --- tablas generadas --------------------------------------------------------------------

TABLE = ("% Generado por scripts/generar_tablas_tesis.py desde artefactos; no editar a mano.\n"
         "% Procedencia de cada número: t.procedencia.json\n"
         "\\begin{tabular}{lrr}\n\\toprule\nA & B & C \\\\\n\\midrule\n"
         "x & 2.209 & [$-$0.061, 0.487] \\\\\ny & 0.90 & 12 \\\\\n\\bottomrule\n\\end{tabular}\n")


def _table_report(ver, tmp_path, name: str, text: str) -> dict:
    thesis = tmp_path / "tesis"
    _write(thesis / "tablas" / name, text)
    return ver.verificar(thesis, procedencias=[_provenance(tmp_path, "t.tex", STANDARD[:1] + STANDARD[2:])])


def test_an_untouched_generated_table_passes(ver, tmp_path):
    assert _table_report(ver, tmp_path, "t.tex", TABLE)["errores"] == []


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("t.tex", TABLE.replace("2.209", "2.210")),  # editada
        ("t.tex", TABLE.replace("& 0.90 & 12", "& 12 & 0.90")),  # celdas permutadas
        ("otra.tex", TABLE.replace("0.487", "0.90").replace("& 0.90 & 12", "& 0.487 & 12")),  # renombrada y permutada
        ("otra.tex", TABLE.split("\n", 2)[2].replace("& 0.90 & 12", "& 12 & 0.90")),  # sin cabecera: se reconoce por contenido
    ],
)
def test_an_edited_generated_table_is_an_error(ver, tmp_path, name, text):
    assert "tabla_editada" in {e["tipo"] for e in _table_report(ver, tmp_path, name, text)["errores"]}


# --- la procedencia se recalcula ---------------------------------------------------------


def test_provenance_is_recomputed_from_the_artifacts(ver, tmp_path):
    report = _check(ver, tmp_path, "nada\n")
    assert report["errores"] == [] and report["n_cifras_recalculadas"] == len(STANDARD)


@pytest.mark.parametrize(
    "entries",
    [
        [("2.300", 2.3, ("a", "b"))],  # valor y texto inventados
        [("2.209", 2.2086, ("a", "c"))],  # campo equivocado
        [("0.061", 0.061, ("a", "c"), "negado")],  # signo
        [("2.208", 2.2086, ("a", "b"))],  # texto que no es el redondeo del valor
    ],
)
def test_tampered_provenance_is_an_error(ver, tmp_path, entries):
    assert "procedencia_no_coincide" in {e["tipo"] for e in _check(ver, tmp_path, "nada\n", entries=entries)["errores"]}


def test_a_changed_artifact_is_an_error(ver, tmp_path):
    thesis = tmp_path / "tesis"
    _write(thesis / "cap.tex", "nada\n")
    prov = _provenance(tmp_path, "t.tex", STANDARD)
    _write(tmp_path / "runs" / "corrida_x" / "datos.json", json.dumps({**DATOS, "extra": 1}))
    assert "fuente_modificada" in {e["tipo"] for e in ver.verificar(thesis, procedencias=[prov])["errores"]}


def test_a_missing_run_is_an_error_unless_relocated(ver, tmp_path):
    thesis = tmp_path / "tesis"
    _write(thesis / "cap.tex", "nada\n")
    prov = _provenance(tmp_path, "t.tex", STANDARD)
    (tmp_path / "runs").rename(tmp_path / "movidas")
    assert {e["tipo"] for e in ver.verificar(thesis, procedencias=[prov])["errores"]} == {"corrida_no_encontrada"}
    assert ver.verificar(thesis, procedencias=[prov], corridas=[tmp_path / "movidas"])["errores"] == []


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


def test_cli_fails_on_a_wrong_sign_claim(tmp_path):
    thesis = tmp_path / "tesis"
    _write(thesis / "a.tex", "transformador menos finito es $-0.487$: el transformador mejora.\n")
    prov = _provenance(tmp_path, "t.tex", STANDARD)
    result = subprocess.run([sys.executable, str(SCRIPT), "--tesis", str(thesis), "--procedencia", str(prov)],
                            capture_output=True, text=True)
    assert result.returncode == 1

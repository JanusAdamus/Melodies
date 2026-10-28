"""Generador de tablas de la tesis (tarea 3.2 del handoff 2026-10).

Lee una corrida por obra y una corrida PDMX, escribe tablas ``.tex`` y, al lado de cada
una, un JSON con la procedencia de cada número (archivo, campo, fila).  Los veredictos
siguen la regla del preregistro con el IC de Holm del estimando por obra.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.tesis_artefactos_sinteticos import porobra_rows, porobra_summary, write_pdmx_run, write_porobra_run

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_bajo_prueba", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses busca su módulo en sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load("generar_tablas_tesis")


@pytest.fixture(scope="module")
def ver():
    return _load("verificar_cifras_tesis")


def _provenance(out: Path, table: str) -> dict:
    return json.loads((out / table.replace(".tex", ".procedencia.json")).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("status", "ci", "expected"),
    [
        ("ok", [-0.3, -0.1], "mejora"),
        ("ok", [-0.1, 0.2], "no hay evidencia"),
        ("ok", [-0.1, 0.0], "no hay evidencia"),  # la regla pide extremo superior < 0, estricto
        ("ok", [0.1, 0.3], "es mejor el otro"),
        ("too_few_paired_works", None, "no evaluable"),
        ("not_evaluable", None, "no evaluable"),
    ],
)
def test_verdict_follows_preregistered_rule(gen, status, ci, expected):
    assert gen.veredicto(status, ci) == expected


def test_porobra_comparisons_table_has_verdicts_and_full_provenance(gen, ver, tmp_path):
    run = write_porobra_run(tmp_path / "runs")
    out = tmp_path / "out"
    gen.main(["--porobra", str(run), "--salida", str(out)])

    tex = (out / "porobra_comparaciones.tex").read_text(encoding="utf-8")
    assert "mejora" in tex and "no hay evidencia" in tex
    provenance = _provenance(out, "porobra_comparaciones.tex")
    assert provenance["tabla"] == "porobra_comparaciones.tex"
    assert provenance["cifras"], "la tabla debe tener cifras con procedencia"
    for entry in provenance["cifras"]:
        assert {"texto", "valor", "archivo", "campo", "fila"} <= set(entry)
        assert entry["archivo"].startswith(run.name + "/")
    # cada número de la tabla tiene exactamente una entrada de procedencia
    written = Counter(number.texto for number in ver.extraer_numeros(tex).numeros)
    assert written == Counter(entry["texto"].lstrip("-") for entry in provenance["cifras"])
    # la diferencia media HDP − finito de los datos sintéticos es −0.100
    finite = [e for e in provenance["cifras"] if e["campo"] == "comparisons.comparisons[0].mean_difference"]
    assert finite and finite[0]["texto"] == "-0.100"


def test_porobra_verdict_disagreeing_with_summary_is_an_error(gen, tmp_path):
    rows = porobra_rows()
    summary = porobra_summary(rows)
    summary["comparisons"]["comparisons"][0]["verdict"] = "no_hay_evidencia"  # el IC dice «mejora»
    run = write_porobra_run(tmp_path / "runs", rows=rows, summary=summary)
    with pytest.raises(gen.ArtefactoInconsistente, match="veredicto"):
        gen.main(["--porobra", str(run), "--salida", str(tmp_path / "out")])


def test_porobra_mean_difference_is_recomputed_from_per_work_csv(gen, tmp_path):
    rows = porobra_rows()
    summary = porobra_summary(rows)
    summary["comparisons"]["comparisons"][1]["mean_difference"] += 0.01
    run = write_porobra_run(tmp_path / "runs", rows=rows, summary=summary)
    with pytest.raises(gen.ArtefactoInconsistente, match="per_work.csv"):
        gen.main(["--porobra", str(run), "--salida", str(tmp_path / "out")])


def test_porobra_nll_table_uses_only_converged_works(gen, tmp_path):
    run = write_porobra_run(tmp_path / "runs")
    out = tmp_path / "out"
    gen.main(["--porobra", str(run), "--salida", str(out)])
    entries = _provenance(out, "porobra_nll.tex")["cifras"]
    hdp_all = [e for e in entries if e["campo"] == "hdp_test_nll" and e["fila"]["fuente"] == "todas" and e["operacion"] == "media"]
    assert len(hdp_all) == 1
    expected = sum(r["hdp_test_nll"] for r in porobra_rows() if r["hdp_converged"]) / 11
    assert hdp_all[0]["valor"] == pytest.approx(expected)
    assert len(hdp_all[0]["fila"]["work_ids"]) == 11


def test_pdmx_comparisons_are_oriented_transformer_minus_classical(gen, tmp_path):
    run = write_pdmx_run(tmp_path / "runs")
    out = tmp_path / "out"
    gen.main(["--pdmx", str(run), "--salida", str(out)])
    tex = (out / "pdmx_comparaciones.tex").read_text(encoding="utf-8")
    entries = _provenance(out, "pdmx_comparaciones.tex")["cifras"]
    flipped = [e for e in entries if e["campo"] == "comparisons[1].mean_difference"]
    assert flipped[0]["texto"] == "-0.100" and flipped[0]["operacion"] == "negado"
    ci = [e for e in entries if e["campo"].startswith("comparisons[1].work_mean_holm_decision_ci")]
    assert [e["texto"] for e in ci] == ["-0.200", "-0.050"]
    rows = [line for line in tex.splitlines() if line.startswith("Transformador & HMM finito")]
    assert len(rows) == 1 and "mejora" in rows[0]
    # el par con menos de 10 obras queda «no evaluable», nunca con veredicto
    assert any(line.startswith("Transformador & VOMM") and "no evaluable" in line for line in tex.splitlines())
    # HDP − transformador = 0.17 con IC [0.1, 0.25] → transformador − HDP mejora
    assert any(line.startswith("Transformador & HDP-HMM") and "mejora" in line for line in tex.splitlines())
    # entre clásicos se conserva la orientación del artefacto: finito − VOMM < 0
    assert any(line.startswith("HMM finito & VOMM") and "mejora" in line for line in tex.splitlines())


def test_pdmx_rejects_zero_must_agree_with_decision_interval(gen, tmp_path):
    from tests.tesis_artefactos_sinteticos import pdmx_comparisons

    comparisons = pdmx_comparisons()
    comparisons[2]["work_mean_holm_rejects_zero"] = False  # su IC [-0.08, -0.01] excluye 0
    run = write_pdmx_run(tmp_path / "runs", comparisons=comparisons)
    with pytest.raises(gen.ArtefactoInconsistente, match="rejects_zero"):
        gen.main(["--pdmx", str(run), "--salida", str(tmp_path / "out")])


def test_pdmx_exploratory_run_is_flagged(gen, tmp_path):
    run = write_pdmx_run(tmp_path / "runs", status="exploratory")
    out = tmp_path / "out"
    gen.main(["--pdmx", str(run), "--salida", str(out)])
    summary = json.loads((out / "generador_resumen.json").read_text(encoding="utf-8"))
    assert any("exploratoria" in aviso for aviso in summary["avisos"])
    assert "exploratoria" in (out / "pdmx_comparaciones.tex").read_text(encoding="utf-8").lower()


def test_pdmx_cost_table_declares_the_device_of_every_row(gen, tmp_path):
    run = write_pdmx_run(tmp_path / "runs")
    out = tmp_path / "out"
    gen.main(["--pdmx", str(run), "--salida", str(out)])
    tex = (out / "pdmx_costos.tex").read_text(encoding="utf-8")
    body = [line for line in tex.splitlines() if line.endswith(r"\\") and "&" in line][1:]
    assert len(body) == 4
    assert all(("cpu" in line) or ("cuda" in line) for line in body)
    entries = _provenance(out, "pdmx_costos.tex")["cifras"]
    fit = [e for e in entries if e["campo"] == "selected_fit_wall_clock_s" and e["fila"]["model"] == "transformer"]
    # media de las dos corridas en frac = 1.0 (2.0 y 4.0), nunca las de frac = 0.5
    assert fit[0]["valor"] == pytest.approx(3.0) and fit[0]["operacion"] == "media"
    assert len(fit[0]["fila"]["filas_csv"]) == 2


def test_missing_run_leaves_visible_pending_markers(gen, tmp_path):
    run = write_pdmx_run(tmp_path / "runs")
    out = tmp_path / "out"
    gen.main(["--pdmx", str(run), "--salida", str(out)])
    tex = (out / "porobra_comparaciones.tex").read_text(encoding="utf-8")
    assert r"\pendiente{" in tex
    summary = json.loads((out / "generador_resumen.json").read_text(encoding="utf-8"))
    assert "porobra_comparaciones.tex" in {item["tabla"] for item in summary["pendientes"]}
    assert _provenance(out, "porobra_comparaciones.tex")["cifras"] == []


def test_figures_are_listed_with_their_hash(gen, tmp_path):
    run = write_pdmx_run(tmp_path / "runs")
    out = tmp_path / "out"
    gen.main(["--pdmx", str(run), "--salida", str(out)])
    figures = json.loads((out / "figuras.json").read_text(encoding="utf-8"))["figuras"]
    assert [f["origen"] for f in figures] == [f"{run.name}/learning_curve.png"]
    assert (out / figures[0]["destino"]).read_bytes() == (run / "learning_curve.png").read_bytes()
    assert len(figures[0]["sha256"]) == 64


def test_never_writes_into_a_non_empty_output(gen, tmp_path):
    run = write_pdmx_run(tmp_path / "runs")
    out = tmp_path / "out"
    out.mkdir()
    (out / "algo.tex").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        gen.main(["--pdmx", str(run), "--salida", str(out)])


def test_generated_tables_pass_the_verifier(gen, ver, tmp_path):
    out = tmp_path / "out"
    gen.main(["--porobra", str(write_porobra_run(tmp_path / "runs")),
              "--pdmx", str(write_pdmx_run(tmp_path / "runs")), "--salida", str(out)])
    report = ver.verificar(out, procedencias=sorted(out.glob("*.procedencia.json")))
    assert report["errores"] == []
    assert report["n_numeros_con_fuente"] > 50

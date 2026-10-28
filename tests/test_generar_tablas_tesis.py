"""Generador de tablas de la tesis (tarea 3.2 del handoff 2026-10).

Lee una corrida por obra y una corrida PDMX, escribe tablas ``.tex`` y, al lado de cada
una, un JSON con la procedencia de cada número (archivo, campo, fila).  Los veredictos
siguen la regla del preregistro con el IC de Holm del estimando por obra; la métrica
principal por obra incluye todas las obras (Enmienda 6).
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest

from tests.tesis_artefactos_sinteticos import (
    PDMX_NLL,
    pdmx_comparisons,
    porobra_rows,
    porobra_summary,
    write_pdmx_run,
    write_porobra_run,
)

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


def _rows(out: Path, table: str) -> list[list[str]]:
    """Celdas de las filas de datos (sin encabezado ni notas) de una tabla generada."""

    lines = (out / table).read_text(encoding="utf-8").splitlines()
    start = lines.index(r"\midrule") + 1
    body = []
    for line in lines[start:]:
        if line in (r"\endhead",):
            continue
        if line == r"\bottomrule":
            break
        body.append([cell.strip() for cell in line.removesuffix(r"\\").split(" & ")])
    return body


def _row(out: Path, table: str, *first: str) -> list[str]:
    found = [cells for cells in _rows(out, table) if tuple(cells[: len(first)]) == first]
    assert len(found) == 1, (first, _rows(out, table))
    return found[0]


def _porobra(gen, tmp_path, **kwargs) -> Path:
    rows = porobra_rows()
    summary = porobra_summary(rows, **kwargs)
    out = tmp_path / "out"
    gen.main(["--porobra", str(write_porobra_run(tmp_path / "runs", rows=rows, summary=summary)), "--salida", str(out)])
    return out


def _pdmx(gen, tmp_path, **kwargs) -> Path:
    out = tmp_path / "out"
    gen.main(["--pdmx", str(write_pdmx_run(tmp_path / "runs", **kwargs)), "--salida", str(out)])
    return out


@pytest.mark.parametrize(
    ("status", "ci", "expected"),
    [
        ("ok", [-0.3, -0.1], "mejora"),
        ("ok", [-0.1, 0.2], "no hay evidencia"),
        ("ok", [-0.1, 0.0], "no hay evidencia"),  # la regla pide extremo superior < 0, estricto
        ("ok", [0.1, 0.3], "es mejor el otro"),
        ("too_few_paired_works", None, "no evaluable"),
        ("not_evaluable", None, "no evaluable"),
        ("too_few_paired_works", [-0.3, -0.1], "no evaluable"),  # sin Holm, un IC no decide
    ],
)
def test_verdict_follows_preregistered_rule(gen, status, ci, expected):
    assert gen.veredicto(status, ci) == expected


# --- por obra ----------------------------------------------------------------------------


def test_porobra_comparisons_table_has_verdicts_and_full_provenance(gen, ver, tmp_path):
    out = _porobra(gen, tmp_path)
    tex = (out / "porobra_comparaciones.tex").read_text(encoding="utf-8")
    provenance = _provenance(out, "porobra_comparaciones.tex")
    assert provenance["tabla"] == "porobra_comparaciones.tex"
    assert provenance["cifras"], "la tabla debe tener cifras con procedencia"
    for entry in provenance["cifras"]:
        assert {"texto", "valor", "archivo", "campo", "fila", "id"} <= set(entry)
        assert entry["archivo"].startswith("porobra_heldout_20261101_abcdef1/")
    # cada número de la tabla tiene exactamente una entrada de procedencia, en orden y con signo
    written = [("-" if n.negativo else "") + n.texto for n in ver.extraer_numeros(tex).numeros]
    assert written == [entry["texto"] for entry in provenance["cifras"]]
    # métrica principal: las 12 obras; la media HDP − finito es −0.100
    assert _row(out, "porobra_comparaciones.tex", "HDP-HMM", "HMM finito")[2:6] == ["12", "$-$0.100", "[$-$0.150, $-$0.050]", "mejora"]
    assert _row(out, "porobra_comparaciones.tex", "HDP-HMM", "Bigrama")[5] == "no hay evidencia"


def test_porobra_main_metric_includes_every_work_enmienda_6(gen, tmp_path):
    out = _porobra(gen, tmp_path)
    tex = (out / "porobra_comparaciones.tex").read_text(encoding="utf-8")
    assert "Enmienda 6" in tex and "Sólo obras con" not in tex
    # NLL: media sobre las 12 obras, también la que no converge
    nll = _row(out, "porobra_nll.tex", "Todas", "HDP-HMM")
    values = [r["hdp_test_nll"] for r in porobra_rows()]
    assert nll[2:] == ["12", f"{statistics.fmean(values):.3f}", f"{statistics.stdev(values):.3f}"]
    finite = _row(out, "porobra_nll.tex", "Todas", "HMM finito")
    finite_values = [r["finite_test_nll"] for r in porobra_rows()]
    assert finite[3:] == [f"{statistics.fmean(finite_values):.3f}", f"{statistics.stdev(finite_values):.3f}"]
    symbtr = _row(out, "porobra_nll.tex", "SymbTr", "HDP-HMM")
    assert symbtr[2] == "4"
    # la obra que no converge sigue en la tabla por obra, marcada, y no «fuera de la métrica»
    por_obra = (out / "porobra_por_obra.tex").read_text(encoding="utf-8")
    assert "fuera de la métrica principal" not in por_obra
    assert r"1.40\textsuperscript{*}" in por_obra


def test_porobra_sensitivity_table_is_the_converged_only_comparison(gen, tmp_path):
    out = _porobra(gen, tmp_path)
    cells = _row(out, "porobra_comparaciones_sensibilidad.tex", "HDP-HMM", "HMM finito")
    assert cells[2] == "11"
    tex = (out / "porobra_comparaciones_sensibilidad.tex").read_text(encoding="utf-8")
    assert "sensibilidad" in tex.lower() and "no decide" in tex


def test_old_rule_comparisons_become_the_sensitivity_and_main_is_pending(gen, tmp_path):
    # compare_hdp de 700acba todavía excluye obras que no convergen: eso es la sensibilidad,
    # no la métrica principal de la Enmienda 6
    out = _porobra(gen, tmp_path, poblacion="convergidas")
    main = (out / "porobra_comparaciones.tex").read_text(encoding="utf-8")
    assert r"\pendiente{" in main and "Enmienda 6" in main
    assert _provenance(out, "porobra_comparaciones.tex")["cifras"] == []
    assert _row(out, "porobra_comparaciones_sensibilidad.tex", "HDP-HMM", "HMM finito")[2] == "11"
    summary = json.loads((out / "generador_resumen.json").read_text(encoding="utf-8"))
    assert any("Enmienda 6" in aviso for aviso in summary["avisos"])
    assert "porobra_comparaciones.tex" in {item["tabla"] for item in summary["pendientes"]}


def test_when_every_work_converges_both_tables_are_the_same_comparison(gen, tmp_path):
    rows = porobra_rows()
    rows[-1]["hdp_converged"], rows[-1]["hdp_rhat"] = True, 1.02
    summary = porobra_summary(rows, poblacion="convergidas")  # el formato de 700acba, sin obras fuera
    out = tmp_path / "out"
    gen.main(["--porobra", str(write_porobra_run(tmp_path / "runs", rows=rows, summary=summary)), "--salida", str(out)])
    assert _row(out, "porobra_comparaciones.tex", "HDP-HMM", "HMM finito")[2] == "12"
    assert _row(out, "porobra_comparaciones_sensibilidad.tex", "HDP-HMM", "HMM finito")[2] == "12"
    assert "coincide" in (out / "porobra_comparaciones_sensibilidad.tex").read_text(encoding="utf-8")


def test_main_metric_without_sensitivity_leaves_it_pending(gen, tmp_path):
    out = _porobra(gen, tmp_path, sensibilidad=False)
    assert r"\pendiente{" in (out / "porobra_comparaciones_sensibilidad.tex").read_text(encoding="utf-8")


def test_porobra_hdp_states_wait_for_per_chain_report(gen, tmp_path):
    out = _porobra(gen, tmp_path)
    states = (out / "porobra_estados.tex").read_text(encoding="utf-8")
    assert r"\pendiente{" in states and "por cadena" in states
    # el finito sí: media y d.e. sobre las 12 obras, de su propia columna
    finite = [r["finite_states_mean"] for r in porobra_rows()]
    assert _row(out, "porobra_estados.tex", "Todas")[1:] == ["12", f"{statistics.fmean(finite):.1f}", f"{statistics.stdev(finite):.1f}"]
    assert "hdp_states_mean" not in {e["campo"] for e in _provenance(out, "porobra_estados.tex")["cifras"]}


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


def test_porobra_n_pairs_is_checked_against_per_work_csv(gen, tmp_path):
    rows = porobra_rows()
    summary = porobra_summary(rows)
    summary["comparisons"]["comparisons"][0]["n_pairs"] = 13
    run = write_porobra_run(tmp_path / "runs", rows=rows, summary=summary)
    with pytest.raises(gen.ArtefactoInconsistente, match="per_work.csv"):
        gen.main(["--porobra", str(run), "--salida", str(tmp_path / "out")])


def test_porobra_convergence_reports_the_fraction_below_the_threshold(gen, tmp_path):
    out = _porobra(gen, tmp_path)
    rows = {cells[0]: cells[1] for cells in _rows(out, "porobra_convergencia.tex")}
    assert rows["Obras analizadas"] == "12"
    assert rows[r"Obras con $\hat R \geq$ umbral"] == "1"
    assert rows[r"Porcentaje con $\hat R <$ umbral"] == r"91.7\,\%"  # 11 de 12
    entry = [e for e in _provenance(out, "porobra_convergencia.tex")["cifras"] if e["texto"] == "91.7"][0]
    assert entry["operacion"] == "complemento_por_cien" and entry["valor_fuente"] == pytest.approx(1 / 12)


def test_porobra_hyperparameters_report_value_and_count_including_finite_alpha(gen, tmp_path):
    out = _porobra(gen, tmp_path)
    assert _row(out, "porobra_hiperparametros.tex", r"$\eta$ del HDP-HMM")[1:] == ["3.0", "12"]
    assert _row(out, "porobra_hiperparametros.tex", r"$\alpha$ del finito", "0.1")[2] == "9"
    assert _row(out, "porobra_hiperparametros.tex", r"$\alpha$ del finito", "1.0")[2] == "3"


def test_porobra_corpus_counts_analysis_failures(gen, tmp_path):
    out = _porobra(gen, tmp_path)
    assert _row(out, "porobra_corpus.tex", "Fallas en el análisis")[1] == "1"
    summary = json.loads((out / "generador_resumen.json").read_text(encoding="utf-8"))
    assert any("obra_falla.mxl" in aviso for aviso in summary["avisos"])


def test_old_format_without_finite_alpha_leaves_it_pending(gen, tmp_path):
    out = _porobra(gen, tmp_path, formato="ff9f998")
    hyper = (out / "porobra_hiperparametros.tex").read_text(encoding="utf-8")
    assert r"\pendiente{" in hyper and "Enmienda 5" in hyper


# --- PDMX --------------------------------------------------------------------------------


def test_pdmx_results_table_separates_work_mean_and_token_weighted(gen, tmp_path):
    out = _pdmx(gen, tmp_path)
    cells = _row(out, "pdmx_resultados.tex", "Transformador", "1.00")
    work = PDMX_NLL["transformer"]
    token = PDMX_NLL["transformer"] + 0.3
    assert cells[2:] == ["30", "6", "3361", f"{work:.3f}", f"[{work - 0.1:.3f}, {work + 0.1:.3f}]",
                         f"{token:.3f}", f"[{token - 0.1:.3f}, {token + 0.1:.3f}]"]


def test_pdmx_results_must_agree_between_artifacts(gen, tmp_path):
    import csv

    run = write_pdmx_run(tmp_path / "runs")
    path = run / "results_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["mean_test_nll"] = float(rows[0]["mean_test_nll"]) + 0.01  # la NLL por nota de la primera fila
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(gen.ArtefactoInconsistente, match="difieren"):
        gen.main(["--pdmx", str(run), "--salida", str(tmp_path / "out")])


def test_pdmx_comparisons_are_oriented_transformer_minus_classical(gen, tmp_path):
    out = _pdmx(gen, tmp_path)
    entries = _provenance(out, "pdmx_comparaciones.tex")["cifras"]
    flipped = [e for e in entries if e["campo"] == "comparisons[1].mean_difference"]
    assert flipped[0]["texto"] == "-0.100" and flipped[0]["operacion"] == "negado"
    ci = [e for e in entries if e["campo"].startswith("comparisons[1].work_mean_holm_decision_ci")]
    assert [e["texto"] for e in ci] == ["-0.200", "-0.050"]
    # finito − transformador = 0.10, por nota 0.05: ambos se niegan
    assert _row(out, "pdmx_comparaciones.tex", "Transformador", "HMM finito")[2:] == [
        "30", "$-$0.100", "[$-$0.200, $-$0.050]", "mejora", "$-$0.050", "[$-$0.070, $-$0.030]", "0.010"]
    # el par con menos de 10 obras queda «no evaluable», nunca con veredicto
    assert _row(out, "pdmx_comparaciones.tex", "Transformador", "VOMM")[5].startswith("no evaluable")
    # HDP − transformador = 0.17 con IC [0.1, 0.25] → transformador − HDP mejora
    assert _row(out, "pdmx_comparaciones.tex", "Transformador", "HDP-HMM")[5] == "mejora"
    # entre clásicos se conserva la orientación del artefacto: finito − VOMM < 0
    assert _row(out, "pdmx_comparaciones.tex", "HMM finito", "VOMM")[3:6] == ["$-$0.050", "[$-$0.080, $-$0.010]", "mejora"]


def test_pdmx_rejects_zero_must_agree_with_decision_interval(gen, tmp_path):
    comparisons = pdmx_comparisons()
    comparisons[2]["work_mean_holm_rejects_zero"] = False  # su IC [-0.08, -0.01] excluye 0
    run = write_pdmx_run(tmp_path / "runs", comparisons=comparisons)
    with pytest.raises(gen.ArtefactoInconsistente, match="rejects_zero"):
        gen.main(["--pdmx", str(run), "--salida", str(tmp_path / "out")])


def test_pdmx_exploratory_run_is_flagged(gen, tmp_path):
    out = _pdmx(gen, tmp_path, status="exploratory")
    summary = json.loads((out / "generador_resumen.json").read_text(encoding="utf-8"))
    assert any("exploratoria" in aviso for aviso in summary["avisos"])
    assert "exploratoria" in (out / "pdmx_comparaciones.tex").read_text(encoding="utf-8").lower()


def test_pdmx_preregistration_gaps_are_warned(gen, tmp_path):
    out = _pdmx(gen, tmp_path, gaps=["falta el VOMM en frac = 0.5"])
    summary = json.loads((out / "generador_resumen.json").read_text(encoding="utf-8"))
    assert any("falta el VOMM en frac = 0.5" in aviso for aviso in summary["avisos"])


def test_pdmx_cost_table_declares_the_device_of_every_row(gen, tmp_path):
    out = _pdmx(gen, tmp_path)
    assert [cells[1] for cells in _rows(out, "pdmx_costos.tex")] == ["cpu", "cpu", "cuda", "cpu"]
    assert _row(out, "pdmx_costos.tex", "Transformador")[1:] == ["cuda", "2", "3.00", "15.00", "0.50", "200", "50"]
    entries = _provenance(out, "pdmx_costos.tex")["cifras"]
    fit = [e for e in entries if e["campo"] == "selected_fit_wall_clock_s" and e["fila"]["model"] == "transformer"]
    # media de las dos corridas en frac = 1.0 (2.0 y 4.0), nunca las de frac = 0.5
    assert fit[0]["valor"] == pytest.approx(3.0) and fit[0]["operacion"] == "media"
    assert len(fit[0]["filas_csv"]) == 2


def test_pdmx_pareto_marks_only_the_frontier(gen, tmp_path):
    out = _pdmx(gen, tmp_path)
    assert {cells[0]: cells[-1] for cells in _rows(out, "pdmx_pareto.tex")} == {
        "HMM finito": "no", "HDP-HMM": "no", "Transformador": "sí", "VOMM": "sí"}


def test_pdmx_hdp_diagnostics_average_over_chains(gen, tmp_path):
    out = _pdmx(gen, tmp_path)
    assert _rows(out, "pdmx_hdp_diagnosticos.tex")[0][:3] == ["1.00", "1", "2"]
    assert _rows(out, "pdmx_hdp_diagnosticos.tex")[0][4:6] == ["9.0", "0.85"]


# --- general -----------------------------------------------------------------------------


def test_missing_run_leaves_visible_pending_markers(gen, tmp_path):
    out = _pdmx(gen, tmp_path)
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


def test_provenance_ids_are_unique(gen, tmp_path):
    out = tmp_path / "out"
    gen.main(["--porobra", str(write_porobra_run(tmp_path / "runs")),
              "--pdmx", str(write_pdmx_run(tmp_path / "runs")), "--salida", str(out)])
    ids = [e["id"] for p in out.glob("*.procedencia.json") for e in json.loads(p.read_text(encoding="utf-8"))["cifras"]]
    assert len(ids) == len(set(ids)) > 100


def test_generated_tables_pass_the_verifier_which_recomputes_them(gen, ver, tmp_path):
    out = tmp_path / "out"
    gen.main(["--porobra", str(write_porobra_run(tmp_path / "runs")),
              "--pdmx", str(write_pdmx_run(tmp_path / "runs")), "--salida", str(out)])
    report = ver.verificar(out, procedencias=sorted(out.glob("*.procedencia.json")))
    assert report["errores"] == []
    assert report["n_cifras_recalculadas"] == report["n_cifras_procedencia"] > 100
    assert report["n_numeros_con_fuente"] > 100

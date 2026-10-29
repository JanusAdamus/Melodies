"""Auditoría de una corrida por obra y obras con la prueba expuesta antes de la corrida confirmatoria.

La auditoría separa dos preguntas: si la corrida es íntegra (``status``: passed, incomplete
o failed) y si su configuración es la de la sección A (``preregistration``: preregistered o
not_preregistered). Un smoke íntegro sale ``passed`` (tarea 1.5) sin pasar por la corrida
confirmatoria; ``confirmatory`` sólo es verdadero si las dos se cumplen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.analysis import porobra, porobra_audit
from src.cli import porobra_heldout
from tests.test_porobra_cli import _run, corpus  # noqa: F401 - fixture reutilizada

INTEGRITY_CHECKS = {"artifacts_present", "manifest_hashes", "denominators", "corpus_accounting",
                    "canonicalization", "headline_recompute", "analysis_failures"}


@pytest.fixture(scope="module")
def run_dir(corpus, tmp_path_factory) -> Path:  # noqa: F811
    return _run(corpus, tmp_path_factory.mktemp("auditada"), "--n-workers", "1")


def _copy(run_dir: Path, tmp_path: Path) -> Path:
    target = tmp_path / run_dir.name
    shutil.copytree(run_dir, target)
    return target


def _check(audit: dict, name: str) -> dict:
    (check,) = [item for item in audit["checks"] if item["name"] == name]
    return check


def _as_preregistered(run_dir: Path, tmp_path: Path, monkeypatch, *, principal_truncation=None, **overrides) -> Path:
    """Copia la corrida y toma su configuración real como la «preregistrada».

    Así el manifiesto, su ``command`` y el resumen siguen siendo los de una corrida real y
    coherentes entre sí (la auditoría los compara); ``overrides`` edita sólo
    ``manifest["config"]``. El truncamiento de la corrida cuenta como el principal salvo
    que ``principal_truncation`` diga otro; el tamaño del corpus, el del corpus de prueba.
    """

    copy = _copy(run_dir, tmp_path)
    summary = json.loads((copy / "summary.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_CORPUS", dict(summary["corpus"]["n_by_source_candidates"]))
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_CONFIG", {
        key: value for key, value in config.items() if key not in (*porobra_audit.UNCHECKED_CONFIG_KEYS, "truncation")})
    truncation = config["truncation"] if principal_truncation is None else principal_truncation
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_TRUNCATION",
                        {truncation: "principal", 40: "sensibilidad_truncamiento_40"})
    config.update(overrides)
    manifest["dirty_worktree"] = False
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return copy


def _rewrite_summary(copy: Path, summary: dict) -> None:
    """Reescribe summary.json y actualiza su hash: sólo el recálculo puede detectarlo."""

    (copy / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["outputs"]["summary.json"] = hashlib.sha256((copy / "summary.json").read_bytes()).hexdigest()
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_per_work(copy: Path, works: pd.DataFrame) -> None:
    """Reescribe per_work.csv y actualiza su hash en el manifiesto (cambio «legítimo»)."""

    works.to_csv(copy / "per_work.csv", index=False)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["outputs"]["per_work.csv"] = hashlib.sha256((copy / "per_work.csv").read_bytes()).hexdigest()
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_un_smoke_integro_pasa_la_auditoria_y_se_marca_no_preregistrado(run_dir):
    written = json.loads((run_dir / porobra_audit.AUDIT_FILENAME).read_text(encoding="utf-8"))
    # Integridad en orden: la tarea 1.5 pide un smoke con la auditoría en «passed».
    assert written["status"] == "passed", written
    assert {check["name"] for check in written["checks"]} == INTEGRITY_CHECKS
    assert all(check["status"] == "passed" for check in written["checks"])
    # La configuración rápida de la prueba no es la de la sección A: se dice aparte.
    preregistration = written["preregistration"]
    assert preregistration["status"] == "not_preregistered"
    assert written["confirmatory"] is False
    for key in ("n_chains", "n_iters", "burn_in", "k_grid", "etas", "bootstrap_samples", "truncation",
                "symbtr_n", "corpus_size"):
        assert key in preregistration["deviations"], preregistration
    # Releer la carpeta da lo mismo y no escribe nada.
    before = sorted(p.name for p in run_dir.iterdir())
    assert porobra_audit.audit_run_directory(run_dir)["status"] == "passed"
    assert sorted(p.name for p in run_dir.iterdir()) == before
    assert porobra_audit.main([str(run_dir)]) == 0
    assert porobra_audit.main([str(run_dir), "--require-preregistered"]) == 1


def test_con_la_configuracion_de_la_seccion_a_la_corrida_es_confirmatoria(run_dir, tmp_path, monkeypatch):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "passed", audit
    assert audit["preregistration"]["status"] == "preregistered", audit["preregistration"]
    assert audit["preregistration"]["deviations"] == {} and audit["preregistration"]["run_kind"] == "principal"
    assert audit["confirmatory"] is True
    assert porobra_audit.main([str(copy), "--require-preregistered"]) == 0


def test_las_constantes_de_truncamiento_son_las_del_preregistro():
    assert porobra_audit.PREREGISTERED_TRUNCATION == {20: "principal", 40: "sensibilidad_truncamiento_40"}


def test_truncamiento_40_es_la_sensibilidad_y_no_la_confirmatoria(corpus, tmp_path, monkeypatch):  # noqa: F811
    # Una corrida real con --truncation 40: su command, su config y su resumen coinciden.
    sensitivity = _run(corpus, tmp_path / "t40", "--n-workers", "1", "--truncation", "40")
    copy = _as_preregistered(sensitivity, tmp_path, monkeypatch, principal_truncation=20)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "passed", audit
    preregistration = audit["preregistration"]
    assert preregistration["status"] == "preregistered", preregistration
    assert preregistration["run_kind"] == "sensibilidad_truncamiento_40"
    # Es la sensibilidad preregistrada, no la corrida confirmatoria.
    assert audit["confirmatory"] is False
    assert porobra_audit.main([str(copy), "--require-preregistered"]) == 1
    assert porobra_audit.main([str(copy)]) == 0


def test_config_que_no_coincide_con_el_comando_real_no_pasa(run_dir, tmp_path, monkeypatch):
    # El manifiesto dice 400 iteraciones, pero el argv real de la corrida pidió 8.
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    monkeypatch.setitem(porobra_audit.PREREGISTERED_CONFIG, "n_iters", 400)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["n_iters"] = 400
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = porobra_audit.audit_run_directory(copy)
    deviations = audit["preregistration"]["deviations"]
    assert set(deviations) == {"command"}, deviations
    assert deviations["command"]["observed"] == {"n_iters": 8}
    assert deviations["command"]["expected"] == {"n_iters": 400}
    assert audit["confirmatory"] is False


def test_un_comando_ilegible_no_pasa(run_dir, tmp_path, monkeypatch):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["command"] = manifest["command"][:3] + ["--opcion-que-no-existe"]
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert "command" in porobra_audit.audit_run_directory(copy)["preregistration"]["deviations"]


def test_config_que_no_coincide_con_el_resumen_no_pasa(run_dir, tmp_path, monkeypatch):
    # Config y comando dicen 10 000 remuestreos; el resumen se calculó con 200.
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    monkeypatch.setitem(porobra_audit.PREREGISTERED_CONFIG, "bootstrap_samples", 10_000)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["bootstrap_samples"] = 10_000
    command = manifest["command"]
    command[command.index("--bootstrap-samples") + 1] = "10000"
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = porobra_audit.audit_run_directory(copy)
    deviations = audit["preregistration"]["deviations"]
    assert set(deviations) == {"summary"}, deviations
    assert "comparisons.bootstrap_samples" in deviations["summary"]["observed"]
    assert audit["confirmatory"] is False


@pytest.mark.parametrize("overrides", [
    {"n_iters": 40}, {"bootstrap_samples": 500}, {"k_grid": [2, 4, 8]}, {"musetrainer_limit": 3},
    {"symbtr_n": 0}, {"symbtr_seed": 8}, {"n_chains": 3, "chain_seeds": [1, 2, 3]}, {"etas": [1.0]},
    {"min_events": 25}, {"finite_alphas": [0.1]}, {"alphas": [1.0]}, {"burn_in": 100}, {"truncation": 30},
    # Flags del CLI y semillas que también cambian la corrida (ronda 3 de revisión).
    {"max_states": 32}, {"finite_restarts": 1}, {"n_occupancy_samples": 1}, {"stats_seed": 99},
    {"finite_seed": 123}, {"hdp_hyperparameters": {"alpha": 1.0, "alpha0": 1.0, "gamma": 1.0, "kappa": 50.0}},
    {"occupancy_threshold": 0.05}, {"rhat_threshold": 1.2},
])
def test_cualquier_desviacion_de_la_seccion_a_se_nombra(run_dir, tmp_path, monkeypatch, overrides):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch, **overrides)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "passed"
    assert audit["preregistration"]["status"] == "not_preregistered" and audit["confirmatory"] is False
    deviations = audit["preregistration"]["deviations"]
    assert set(overrides) <= set(deviations), deviations
    assert deviations[next(iter(overrides))]["observed"] == next(iter(overrides.values()))


def test_una_clave_de_configuracion_desconocida_no_pasa_por_preregistrada(run_dir, tmp_path, monkeypatch):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch, opcion_nueva=1)
    deviations = porobra_audit.audit_run_directory(copy)["preregistration"]["deviations"]
    assert deviations["opcion_nueva"] == {"observed": 1, "expected": "clave no prevista por la sección A"}


def test_todas_las_claves_del_manifiesto_estan_previstas(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = set(porobra_audit.PREREGISTERED_CONFIG) | set(porobra_audit.UNCHECKED_CONFIG_KEYS) | {"truncation"}
    assert set(manifest["config"]) == expected


def test_la_configuracion_por_defecto_del_cli_es_la_preregistrada():
    args = porobra_heldout.build_parser().parse_args([])
    config = porobra.PorObraConfig().to_dict()
    for key, value in porobra_audit.PREREGISTERED_CONFIG.items():
        if key in config:
            assert porobra_audit._same(list(config[key]) if isinstance(config[key], tuple) else config[key], value), key
    assert args.max_states == porobra_audit.PREREGISTERED_CONFIG["max_states"]
    assert args.finite_restarts == porobra_audit.PREREGISTERED_CONFIG["finite_restarts"]
    assert args.occupancy_samples == porobra_audit.PREREGISTERED_CONFIG["n_occupancy_samples"]
    assert args.stats_seed == porobra_audit.PREREGISTERED_CONFIG["stats_seed"]


def test_rejilla_del_finito_topada_no_pasa_por_preregistrada(run_dir, tmp_path, monkeypatch):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    works = porobra_audit.read_per_work(copy / "per_work.csv")
    works["finite_k_hit_limit"] = False
    works.loc[0, "finite_k_hit_limit"] = True
    _rewrite_per_work(copy, works)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "passed", audit
    deviations = audit["preregistration"]["deviations"]
    assert deviations["finite_k_hit_limit"] == {"observed": [works.loc[0, "work_id"]], "expected": []}
    assert audit["confirmatory"] is False


def test_arbol_sucio_o_corpus_incompleto_no_pasan(run_dir, tmp_path, monkeypatch):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["dirty_worktree"] = True
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_CORPUS", {"MuseTrainer": 69, "SymbTr": 100})
    deviations = porobra_audit.audit_run_directory(copy)["preregistration"]["deviations"]
    assert set(deviations) == {"dirty_worktree", "corpus_size"}


def test_falta_un_archivo_queda_incompleta(run_dir, tmp_path):
    copy = _copy(run_dir, tmp_path)
    (copy / "summary.json").unlink()
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "incomplete"
    assert _check(audit, "artifacts_present")["missing"] == ["summary.json"]
    assert audit["preregistration"]["status"] == "unknown" and audit["confirmatory"] is False
    assert porobra_audit.main([str(copy)]) == 1


def test_un_csv_alterado_no_pasa_hashes_ni_recalculo(run_dir, tmp_path):
    copy = _copy(run_dir, tmp_path)
    works = porobra_audit.read_per_work(copy / "per_work.csv")
    works.loc[0, "hdp_test_nll"] += 0.5
    works.to_csv(copy / "per_work.csv", index=False)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "failed"
    assert _check(audit, "manifest_hashes")["status"] == "failed"
    assert _check(audit, "headline_recompute")["status"] == "failed"


def _flip_interval(interval):
    return [interval[0] - 1.0, interval[1] + 1.0]


#: Ediciones del resumen en lo que decide; cada una deja el hash al día.
SUMMARY_EDITS = {
    "decision_ci": lambda s: s["comparisons"]["comparisons"][0].update(
        work_mean_holm_decision_ci=_flip_interval(s["comparisons"]["comparisons"][0]["work_mean_holm_decision_ci"])),
    "rejects_zero": lambda s: s["comparisons"]["comparisons"][1].update(
        work_mean_holm_rejects_zero=not s["comparisons"]["comparisons"][1]["work_mean_holm_rejects_zero"]),
    "verdict": lambda s: s["comparisons"]["comparisons"][2].update(verdict="mejora"
        if s["comparisons"]["comparisons"][2]["verdict"] != "mejora" else "no_hay_evidencia"),
    "bootstrap_95_ci": lambda s: s["comparisons"]["comparisons"][0].update(
        bootstrap_95_ci=_flip_interval(s["comparisons"]["comparisons"][0]["bootstrap_95_ci"])),
    "token_weighted": lambda s: s["comparisons"]["comparisons"][0].update(
        token_weighted_difference=s["comparisons"]["comparisons"][0]["token_weighted_difference"] + 0.1),
    "wilcoxon_holm": lambda s: s["comparisons"]["comparisons"][0].update(
        p_value_holm=(s["comparisons"]["comparisons"][0]["p_value_holm"] or 0.5) / 2),
    "sensitivity_verdict": lambda s: s["sensitivity_without_exposed"]["comparisons"][0].update(verdict="mejora"
        if s["sensitivity_without_exposed"]["comparisons"][0]["verdict"] != "mejora" else "no_hay_evidencia"),
    "fraction_converged": lambda s: s["convergence"].update(fraction_converged=0.123),
    "not_converged_ids": lambda s: s["convergence"].update(not_converged_work_ids=["otra"]),
    "bootstrap_samples": lambda s: s["comparisons"].update(bootstrap_samples=201),
    "exposed_removed": lambda s: s["test_exposure"].update(exposed_work_ids=[]),
}


def test_el_recalculo_reproduce_exacto_lo_que_decide(run_dir):
    # Con la misma semilla, recalcular desde per_work.csv da el resumen bit a bit.
    check = _check(porobra_audit.audit_run_directory(run_dir), "headline_recompute")
    assert check["status"] == "passed" and check["mismatches"] == []
    assert set(check["recomputed_blocks"]) == {"comparisons", "sensitivity_converged_only",
                                               "sensitivity_without_exposed", "convergence"}


@pytest.mark.parametrize("edit", sorted(SUMMARY_EDITS))
def test_un_resumen_editado_falla_el_recalculo(run_dir, tmp_path, edit):
    copy = _copy(run_dir, tmp_path)
    summary = json.loads((copy / "summary.json").read_text(encoding="utf-8"))
    SUMMARY_EDITS[edit](summary)
    _rewrite_summary(copy, summary)
    audit = porobra_audit.audit_run_directory(copy)
    assert _check(audit, "manifest_hashes")["status"] == "passed"
    check = _check(audit, "headline_recompute")
    assert check["status"] == "failed" and check["mismatches"], check
    assert audit["status"] == "failed"


def test_denominador_y_huella_duplicada_se_detectan(run_dir, tmp_path):
    copy = _copy(run_dir, tmp_path)
    works = porobra_audit.read_per_work(copy / "per_work.csv")
    works.loc[0, "n_test"] += 1
    works.loc[1, "content_fingerprint"] = works.loc[0, "content_fingerprint"]
    works.to_csv(copy / "per_work.csv", index=False)
    audit = porobra_audit.audit_run_directory(copy)
    denominators = _check(audit, "denominators")
    assert denominators["status"] == "failed" and denominators["bad_work_ids"] == [works.loc[0, "work_id"]]
    assert _check(audit, "canonicalization")["status"] == "failed"


def test_conteo_del_corpus_que_no_cuadra_falla(run_dir, tmp_path):
    copy = _copy(run_dir, tmp_path)
    summary = json.loads((copy / "summary.json").read_text(encoding="utf-8"))
    summary["corpus"]["n_candidates"] += 1
    (copy / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _check(porobra_audit.audit_run_directory(copy), "corpus_accounting")["status"] == "failed"


def test_una_obra_que_falla_deja_la_auditoria_en_failed(corpus, tmp_path, monkeypatch):  # noqa: F811
    original = porobra_heldout.porobra.analyze_work

    def fragile(work_id, tokens, config):
        if work_id.endswith("a.musicxml"):
            raise ValueError("fallo simulado")
        return original(work_id, tokens, config)

    monkeypatch.setattr(porobra_heldout.porobra, "analyze_work", fragile)
    run_dir = _run(corpus, tmp_path, "--n-workers", "1")
    audit = json.loads((run_dir / porobra_audit.AUDIT_FILENAME).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert _check(audit, "analysis_failures")["work_ids"] == ["MuseTrainer:a.musicxml"]
    assert _check(audit, "corpus_accounting")["status"] == "passed"


# --- obras con la prueba expuesta ----------------------------------------------------


def test_las_obras_expuestas_son_las_seis_de_la_aclaracion_8():
    timing_runs = (
        "MuseTrainer:12_Variations_of_Twinkle_Twinkle_Little_Star.mxl",
        "MuseTrainer:Arabesque_L._66_No._1_in_E_Major.mxl",
        "MuseTrainer:Ave_Maria_D839_-_Schubert_-_Solo_Piano_Arrg..mxl",
        "MuseTrainer:Bach_Minuet_in_G_Major_BWV_Anh._114.mxl",
        "MuseTrainer:Bach_Toccata_and_Fugue_in_D_Minor_Piano_solo.mxl",
    )
    # La Aclaración 8 (9117caf) registra las seis: las cinco de las corridas de tiempos de
    # la Aclaración 7 y Passacaglia, cuya NLL de prueba calculó e imprimió la revisión 1.4.
    assert porobra.TEST_EXPOSED_WORK_IDS == timing_runs + ("MuseTrainer:Passacaglia.mxl",)
    assert set(porobra.TEST_EXPOSED_WORKS) == set(porobra.TEST_EXPOSED_WORK_IDS)
    assert all(reason.startswith("Aclaración 8") for reason in porobra.TEST_EXPOSED_WORKS.values())
    assert all("Aclaración 7" in porobra.TEST_EXPOSED_WORKS[work] for work in timing_runs)
    assert "Aclaración 7" not in porobra.TEST_EXPOSED_WORKS["MuseTrainer:Passacaglia.mxl"]
    assert "sin leerse" in porobra.TEST_EXPOSED_WORKS["MuseTrainer:Bach_Minuet_in_G_Major_BWV_Anh._114.mxl"]
    assert "ff9f998" in porobra.TEST_EXPOSED_WORKS["MuseTrainer:Arabesque_L._66_No._1_in_E_Major.mxl"]
    for text in (porobra.TEST_EXPOSURE_NOTE, *porobra.TEST_EXPOSED_WORKS.values()):
        assert "propuesta" not in text and "pendiente" not in text, text


def _rows(n: int) -> list[dict]:
    return [{"work_id": f"w{i}", "n_test": 10, "hdp_test_nll": 1.0 + 0.01 * i, "finite_test_nll": 1.2,
             "bigram_test_nll": 1.3, "unigram_test_nll": 1.4, "hdp_converged": True} for i in range(n)]


def test_compare_hdp_excluye_las_obras_indicadas():
    rows = _rows(12)
    result = porobra.compare_hdp(rows, bootstrap_samples=100, exclude_work_ids=("w0", "w1"))
    assert result["n_works"] == 10 and result["excluded_work_ids"] == ["w0", "w1"]
    assert all("w0" not in c["canonical_work_ids"] for c in result["comparisons"])
    assert porobra.compare_hdp(rows, bootstrap_samples=100)["excluded_work_ids"] == []


def test_el_resumen_divulga_la_exposicion_y_repite_sin_esas_obras(corpus, tmp_path, monkeypatch):  # noqa: F811
    exposed = {"MuseTrainer:a.musicxml": "motivo a", "MuseTrainer:no_esta.mxl": "motivo b"}
    monkeypatch.setattr(porobra, "TEST_EXPOSED_WORKS", exposed)
    monkeypatch.setattr(porobra, "TEST_EXPOSED_WORK_IDS", tuple(exposed))
    run_dir = _run(corpus, tmp_path, "--n-workers", "1")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    exposure = summary["test_exposure"]
    assert exposure["exposed_work_ids"] == ["MuseTrainer:a.musicxml", "MuseTrainer:no_esta.mxl"]
    assert exposure["exposed_works"] == exposed
    assert exposure["exposed_in_run"] == ["MuseTrainer:a.musicxml"]
    # La Aclaración 8 registra la exposición y el análisis adicional sin esas obras, que no decide.
    assert "Aclaración 8" in exposure["note"] and "no decide" in exposure["note"]
    assert "propuesta" not in exposure["note"] and "pendiente" not in exposure["note"]
    without = summary["sensitivity_without_exposed"]
    assert without["n_works"] == summary["comparisons"]["n_works"] - 1
    assert without["excluded_work_ids"] == ["MuseTrainer:a.musicxml"]
    assert summary["preregistro"].endswith("Enmiendas 3, 5 y 6, Aclaraciones 7 y 8")
    assert "Aclaración 8" in without["rule"] and "no decide" in without["rule"]
    # El recálculo de la auditoría usa la misma lista de obras expuestas.
    audit = json.loads((run_dir / porobra_audit.AUDIT_FILENAME).read_text(encoding="utf-8"))
    assert _check(audit, "headline_recompute")["status"] == "passed", audit

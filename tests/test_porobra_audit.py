"""Auditoría de una corrida por obra y obras con la prueba expuesta antes de las Enmiendas 5 y 6."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.analysis import porobra, porobra_audit
from src.cli import porobra_heldout
from tests.test_porobra_cli import _run, corpus  # noqa: F401 - fixture reutilizada


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


def _as_preregistered(run_dir: Path, tmp_path: Path, monkeypatch, **overrides) -> Path:
    """Copia la corrida con la configuración de la sección A en el manifiesto.

    El manifiesto no está entre los ``outputs`` con hash, así que editarlo no rompe los
    demás controles; el tamaño del corpus esperado se ajusta al del corpus de prueba.
    """

    copy = _copy(run_dir, tmp_path)
    summary = json.loads((copy / "summary.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_CORPUS", dict(summary["corpus"]["n_by_source_candidates"]))
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"].update({**porobra_audit.PREREGISTERED_CONFIG, "truncation": 20, **overrides})
    manifest["dirty_worktree"] = False
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return copy


def test_la_corrida_escribe_su_auditoria_y_un_smoke_no_pasa_por_preregistrada(run_dir):
    written = json.loads((run_dir / porobra_audit.AUDIT_FILENAME).read_text(encoding="utf-8"))
    # Integridad en orden, pero la configuración rápida de la prueba no es la de la sección A.
    assert written["status"] == "not_preregistered", written
    names = {check["name"] for check in written["checks"]}
    assert names == {"artifacts_present", "manifest_hashes", "denominators", "corpus_accounting",
                     "canonicalization", "headline_recompute", "analysis_failures", "preregistered_config"}
    assert all(check["status"] == "passed" for check in written["checks"] if check["name"] != "preregistered_config")
    deviations = _check(written, "preregistered_config")["deviations"]
    for key in ("n_chains", "n_iters", "burn_in", "k_grid", "etas", "bootstrap_samples", "truncation",
                "symbtr_n", "corpus_size"):
        assert key in deviations, deviations
    # Releer la carpeta da lo mismo y no escribe nada.
    before = sorted(p.name for p in run_dir.iterdir())
    assert porobra_audit.audit_run_directory(run_dir)["status"] == "not_preregistered"
    assert sorted(p.name for p in run_dir.iterdir()) == before
    assert porobra_audit.main([str(run_dir)]) == 1


def test_con_la_configuracion_de_la_seccion_a_la_auditoria_pasa(run_dir, tmp_path, monkeypatch):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "passed", audit
    check = _check(audit, "preregistered_config")
    assert check["deviations"] == {} and check["run_kind"] == "principal"


def test_truncamiento_40_es_la_sensibilidad_preregistrada(run_dir, tmp_path, monkeypatch):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch, truncation=40)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "passed"
    assert _check(audit, "preregistered_config")["run_kind"] == "sensibilidad_truncamiento_40"


@pytest.mark.parametrize("overrides", [
    {"n_iters": 40}, {"bootstrap_samples": 500}, {"k_grid": [2, 4]}, {"musetrainer_limit": 3},
    {"symbtr_n": 0}, {"symbtr_seed": 8}, {"n_chains": 2, "chain_seeds": [1, 2]}, {"etas": [1.0]},
    {"min_events": 25}, {"finite_alphas": [0.1]}, {"alphas": [1.0]}, {"burn_in": 100}, {"truncation": 30},
])
def test_cualquier_desviacion_de_la_seccion_a_se_nombra(run_dir, tmp_path, monkeypatch, overrides):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch, **overrides)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "not_preregistered"
    deviations = _check(audit, "preregistered_config")["deviations"]
    assert set(overrides) <= set(deviations), deviations
    assert deviations[next(iter(overrides))]["observed"] == next(iter(overrides.values()))


def test_arbol_sucio_o_corpus_incompleto_no_pasan(run_dir, tmp_path, monkeypatch):
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["dirty_worktree"] = True
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_CORPUS", {"MuseTrainer": 69, "SymbTr": 100})
    deviations = _check(porobra_audit.audit_run_directory(copy), "preregistered_config")["deviations"]
    assert set(deviations) == {"dirty_worktree", "corpus_size"}


def test_una_falla_de_integridad_pesa_mas_que_la_configuracion(run_dir, tmp_path):
    copy = _copy(run_dir, tmp_path)
    (copy / "summary.json").unlink()
    assert porobra_audit.audit_run_directory(copy)["status"] == "incomplete"


def test_falta_un_archivo_queda_incompleta(run_dir, tmp_path):
    copy = _copy(run_dir, tmp_path)
    (copy / "summary.json").unlink()
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "incomplete"
    assert _check(audit, "artifacts_present")["missing"] == ["summary.json"]


def test_un_csv_alterado_no_pasa_hashes_ni_recalculo(run_dir, tmp_path):
    copy = _copy(run_dir, tmp_path)
    works = pd.read_csv(copy / "per_work.csv")
    works.loc[0, "hdp_test_nll"] += 0.5
    works.to_csv(copy / "per_work.csv", index=False)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "failed"
    assert _check(audit, "manifest_hashes")["status"] == "failed"
    assert _check(audit, "headline_recompute")["status"] == "failed"


def test_denominador_y_huella_duplicada_se_detectan(run_dir, tmp_path):
    copy = _copy(run_dir, tmp_path)
    works = pd.read_csv(copy / "per_work.csv")
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


def test_las_obras_expuestas_son_las_cinco_primeras_de_musetrainer():
    assert porobra.TEST_EXPOSED_WORK_IDS == (
        "MuseTrainer:12_Variations_of_Twinkle_Twinkle_Little_Star.mxl",
        "MuseTrainer:Arabesque_L._66_No._1_in_E_Major.mxl",
        "MuseTrainer:Ave_Maria_D839_-_Schubert_-_Solo_Piano_Arrg..mxl",
        "MuseTrainer:Bach_Minuet_in_G_Major_BWV_Anh._114.mxl",
        "MuseTrainer:Bach_Toccata_and_Fugue_in_D_Minor_Piano_solo.mxl",
    )


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
    monkeypatch.setattr(porobra, "TEST_EXPOSED_WORK_IDS", ("MuseTrainer:a.musicxml", "MuseTrainer:no_esta.mxl"))
    run_dir = _run(corpus, tmp_path, "--n-workers", "1")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    exposure = summary["test_exposure"]
    assert exposure["exposed_work_ids"] == ["MuseTrainer:a.musicxml", "MuseTrainer:no_esta.mxl"]
    assert exposure["exposed_in_run"] == ["MuseTrainer:a.musicxml"]
    assert "Aclaraciones 7 y 8" in exposure["note"]
    without = summary["sensitivity_without_exposed"]
    assert without["n_works"] == summary["comparisons"]["n_works"] - 1
    assert without["excluded_work_ids"] == ["MuseTrainer:a.musicxml"]
    assert summary["preregistro"].endswith("Enmiendas 3, 5 y 6, Aclaraciones 7 y 8")

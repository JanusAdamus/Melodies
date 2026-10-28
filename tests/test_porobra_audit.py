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


def test_la_corrida_escribe_su_auditoria_en_passed(run_dir):
    written = json.loads((run_dir / porobra_audit.AUDIT_FILENAME).read_text(encoding="utf-8"))
    assert written["status"] == "passed", written
    names = {check["name"] for check in written["checks"]}
    assert names == {"artifacts_present", "manifest_hashes", "denominators", "corpus_accounting",
                     "canonicalization", "headline_recompute", "analysis_failures"}
    # Releer la carpeta da lo mismo y no escribe nada.
    before = sorted(p.name for p in run_dir.iterdir())
    assert porobra_audit.audit_run_directory(run_dir)["status"] == "passed"
    assert sorted(p.name for p in run_dir.iterdir()) == before


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
    assert "Aclaración 7" in exposure["note"]
    without = summary["sensitivity_without_exposed"]
    assert without["n_works"] == summary["comparisons"]["n_works"] - 1
    assert without["excluded_work_ids"] == ["MuseTrainer:a.musicxml"]
    assert summary["preregistro"].endswith("Enmiendas 3, 5 y 6, Aclaración 7")

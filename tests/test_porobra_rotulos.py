"""Rótulos y procedencia de una corrida por obra: qué es, con qué commit y sobre qué corpus.

Ajuste sustituido por cifras sintéticas: la configuración de la sección A corre en segundos y
ninguna cifra de prueba de una obra real se calcula.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import unicodedata

import numpy as np
import pandas as pd
import pytest

from next_token_experiment.tests.test_pipeline import write_scale_score
from src.analysis import library_batch, porobra, porobra_audit
from src.cli import porobra_heldout
from tests.test_porobra_audit import _as_preregistered, _check, _copy, _rewrite_summary, pin_outcomes
from tests.test_porobra_audit import run_dir  # noqa: F401 - fixture reutilizada
from tests.test_porobra_cli import FAST, _melody, corpus  # noqa: F401 - fixture reutilizada

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
#: El destino de cada candidato del corpus de prueba de ``tests.test_porobra_cli.corpus``.
EXPECTED_OUTCOMES = {
    "analizada": ["MuseTrainer:a.musicxml", "MuseTrainer:b.musicxml",
                  "SymbTr:hicaz--sarki--aksak--obra4--anonim.xml", "SymbTr:hicaz--sarki--aksak--obra5--anonim.xml",
                  "SymbTr:rast--sarki--aksak--obra6--anonim.xml"],
    "duplicado": ["MuseTrainer:b_copia.musicxml"],
    "corta": ["MuseTrainer:corta.musicxml"],
    "falla_extraccion": ["MuseTrainer:rota.musicxml"],
    "falla_analisis": [],
}


def _fake_row(work_id, tokens, rng) -> dict:
    split = porobra.split_tokens(tokens)
    chains = (2.0 + rng.normal(0, 0.05, 4)).tolist()
    rhat = float(rng.uniform(1.0, 1.5))
    return dict(work_id=work_id, n_events=len(tokens), n_fit=split.fit.size, n_test=split.test.size,
                n_inner_train=split.inner_train.size, n_validation=split.validation.size,
                unigram_alpha=0.5, bigram_alpha=0.5, unigram_test_nll=float(rng.uniform(2, 3)),
                bigram_test_nll=float(rng.uniform(2, 3)), finite_k=8, finite_alpha=0.1,
                finite_k_grid=[2, 4, 8, 16, 32], finite_k_hit_limit=False, finite_em_hit_max_iterations=False, finite_test_nll=float(rng.uniform(2, 3)),
                hdp_eta=3.0, hdp_rhat=rhat, hdp_converged=rhat < 1.1, hdp_test_nll=float(np.mean(chains)),
                hdp_test_nll_by_chain=chains, hdp_states_pooled_chains_mean=3.0)


def _stub_jobs(on_run=lambda: None):
    def run_jobs(jobs, n_workers, partial_path):
        on_run()
        rng = np.random.default_rng(123)
        return sorted((_fake_row(job[0], job[1], rng) for job in jobs), key=lambda row: row["work_id"])

    return run_jobs


def _stub_git(monkeypatch, state: dict) -> None:
    def git(*args):
        if args[0] == "status":
            return state["status"]
        return "rama" if "--abbrev-ref" in args else state["commit"]

    monkeypatch.setattr(porobra_heldout, "_git", git)


def _corpus_argv(corpus) -> list[str]:  # noqa: F811
    return ["--musetrainer-root", str(corpus["library"]), "--symbtr-root", str(corpus["symbtr"])]


@pytest.fixture
def as_section_a(corpus, monkeypatch):  # noqa: F811
    """Toma el corpus de prueba como el fijado; el commit y el árbol, limpios y estables."""

    catalog = porobra_heldout._catalog(porobra_heldout.build_parser().parse_args(_corpus_argv(corpus)))
    files = porobra_heldout.input_files(catalog)
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_CORPUS", {s: sum(f["source"] == s for f in files) for s in {f["source"] for f in files}})
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_CORPUS_SHA256", porobra_audit.source_aggregates({"inputs": {"files": files}}))
    pin_outcomes(monkeypatch, EXPECTED_OUTCOMES)
    pieces = library_batch.prepare_corpus(catalog, extraction=porobra_heldout.EXTRACTION).pieces
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_CORPUS_CONTENT_SHA256",
                        porobra_audit.corpus_content_sha256(pieces[pieces["n_events"] >= 20]))
    _stub_git(monkeypatch, {"commit": COMMIT_A, "status": ""})
    monkeypatch.setattr(porobra_heldout, "_run_jobs", _stub_jobs())


def _read(run_dir: Path, name: str):  # noqa: F811
    return json.loads((run_dir / name).read_text(encoding="utf-8"))


# --- qué es la corrida ------------------------------------------------------------------


def test_la_corrida_principal_se_rotula_como_tal_sin_aviso(corpus, tmp_path, as_section_a, capsys):  # noqa: F811
    run = porobra_heldout.main([*_corpus_argv(corpus), "--output-root", str(tmp_path)])
    audit, summary = _read(run, porobra_audit.AUDIT_FILENAME), _read(run, "summary.json")
    assert audit["status"] == "passed" and audit["confirmatory"] is True, audit
    assert summary["run"]["run_kind"] == "principal" and summary["run"]["truncation"] == 20
    assert summary["run"]["deviations"] == []
    out = capsys.readouterr().out
    assert "no es la corrida confirmatoria" not in out and "se aparta" not in out
    # Las fallas de extracción se avisan siempre, también en la corrida confirmatoria.
    assert "AVISO: 1 falla(s) de extracción" in out and "MuseTrainer:rota.musicxml" in out


def test_la_sensibilidad_40_no_se_rotula_como_la_que_decide(corpus, tmp_path, as_section_a, capsys):  # noqa: F811
    run = porobra_heldout.main([*_corpus_argv(corpus), "--output-root", str(tmp_path), "--truncation", "40"])
    audit, summary = _read(run, porobra_audit.AUDIT_FILENAME), _read(run, "summary.json")
    assert audit["confirmatory"] is False and audit["preregistration"]["status"] == "preregistered"
    assert summary["run"]["run_kind"] == "sensibilidad_truncamiento_40" and summary["run"]["truncation"] == 40
    assert "no decide" in summary["run"]["note"]
    # El bloque de comparaciones no declara por su cuenta que decide.
    assert "confirmatory" in summary["comparisons"]["rule"]
    assert not summary["comparisons"]["rule"].endswith("; decide.")
    readme = (run / "README.txt").read_text(encoding="utf-8")
    assert "sensibilidad_truncamiento_40" in readme and "truncamiento 40" in readme
    out = capsys.readouterr().out
    assert "AVISO" in out and "sensibilidad_truncamiento_40" in out and "no deciden" in out


def test_un_smoke_no_se_rotula_principal_en_ningun_archivo(corpus, tmp_path, monkeypatch, capsys):  # noqa: F811
    _stub_git(monkeypatch, {"commit": COMMIT_A, "status": ""})
    # Con truncamiento 20: el que corresponde a «principal» si se mira sólo ese número.
    run = porobra_heldout.main([*_corpus_argv(corpus), "--symbtr-n", "2", "--output-root", str(tmp_path), *FAST, "--truncation", "20"])
    audit, summary = _read(run, porobra_audit.AUDIT_FILENAME), _read(run, "summary.json")
    assert audit["preregistration"]["status"] == "not_preregistered"
    assert audit["preregistration"]["run_kind"] == "no_preregistrada"
    assert "confirmatoria de la sección A (" not in audit["preregistration"]["run_kind_note"]
    assert summary["run"]["run_kind"] == "no_preregistrada"
    assert {"n_iters", "corpus_size", "corpus_identity"} <= set(summary["run"]["deviations"])
    assert "no_preregistrada" in (run / "README.txt").read_text(encoding="utf-8")
    assert "AVISO" in capsys.readouterr().out


def test_un_resumen_con_otro_truncamiento_que_el_manifiesto_no_pasa(run_dir, tmp_path, monkeypatch):  # noqa: F811
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    summary = _read(copy, "summary.json")
    summary["run"]["truncation"] = 40
    _rewrite_summary(copy, summary)
    audit = porobra_audit.audit_run_directory(copy)
    assert "run.truncation" in audit["preregistration"]["deviations"]["summary"]["observed"]
    assert audit["confirmatory"] is False


def test_antes_de_auditar_la_principal_es_candidata_y_el_aviso_no_se_contradice(corpus, tmp_path, as_section_a, monkeypatch, capsys):  # noqa: F811
    # Una corrida sin desviaciones cuya integridad falla: run_kind sigue siendo «principal».
    audit_run = porobra_audit.audit_run_directory
    monkeypatch.setattr(porobra_audit, "audit_run_directory",
                        lambda directory: {**audit_run(directory), "status": "failed", "confirmatory": False})
    run = porobra_heldout.main([*_corpus_argv(corpus), "--output-root", str(tmp_path)])
    summary, readme = _read(run, "summary.json"), (run / "README.txt").read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert summary["run"]["run_kind"] == "principal"
    assert "candidata a confirmatoria" in summary["run"]["note"] and "candidata a confirmatoria" in readme
    for text in (summary["run"]["note"], readme, out):
        assert "corrida confirmatoria de la sección A" not in text
    assert "no es la corrida confirmatoria" in out and "candidata a confirmatoria" in out and "integridad: failed" in out


def test_la_nota_de_la_auditoria_no_afirma_lo_que_la_integridad_niega(run_dir, tmp_path, monkeypatch):  # noqa: F811
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    summary = _read(copy, "summary.json")
    summary["comparisons"]["comparisons"][0]["mean_difference"] *= 1 + 1e-9
    _rewrite_summary(copy, summary)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "failed" and audit["confirmatory"] is False
    assert audit["preregistration"]["run_kind"] == "principal"
    assert "candidata a confirmatoria" in audit["preregistration"]["run_kind_note"]


# --- con qué commit corrió --------------------------------------------------------------


def test_el_manifiesto_guarda_el_commit_de_cuando_empezo(corpus, tmp_path, monkeypatch):  # noqa: F811
    state = {"commit": COMMIT_A, "status": ""}
    _stub_git(monkeypatch, state)
    monkeypatch.setattr(porobra_heldout, "_run_jobs", _stub_jobs(lambda: state.update(commit=COMMIT_B)))
    run = porobra_heldout.main([*_corpus_argv(corpus), "--output-root", str(tmp_path)])
    manifest = _read(run, "manifest.json")
    assert run.name.endswith("_" + COMMIT_A[:7])
    assert manifest["commit"] == COMMIT_A and manifest["git_at_end"]["commit"] == COMMIT_B
    audit = _read(run, porobra_audit.AUDIT_FILENAME)
    assert audit["preregistration"]["deviations"]["commit_changed_during_run"] == {
        "observed": COMMIT_B, "expected": COMMIT_A}
    assert audit["confirmatory"] is False


def test_un_arbol_sucio_al_empezar_y_limpio_al_terminar_queda_sucio(corpus, tmp_path, monkeypatch):  # noqa: F811
    state = {"commit": COMMIT_A, "status": " M src/analysis/porobra.py"}
    _stub_git(monkeypatch, state)
    monkeypatch.setattr(porobra_heldout, "_run_jobs", _stub_jobs(lambda: state.update(status="")))
    run = porobra_heldout.main([*_corpus_argv(corpus), "--output-root", str(tmp_path)])
    manifest = _read(run, "manifest.json")
    assert manifest["dirty_worktree"] is True and manifest["git_at_end"]["dirty_worktree"] is False
    audit = _read(run, porobra_audit.AUDIT_FILENAME)
    assert "dirty_worktree" in audit["preregistration"]["deviations"] and audit["confirmatory"] is False


def test_un_arbol_que_se_ensucia_durante_la_corrida_no_es_confirmatorio(corpus, tmp_path, as_section_a, monkeypatch):  # noqa: F811
    state = {"commit": COMMIT_A, "status": ""}
    _stub_git(monkeypatch, state)
    monkeypatch.setattr(porobra_heldout, "_run_jobs", _stub_jobs(lambda: state.update(status=" M x.py")))
    run = porobra_heldout.main([*_corpus_argv(corpus), "--output-root", str(tmp_path)])
    audit = _read(run, porobra_audit.AUDIT_FILENAME)
    assert "dirty_worktree" in audit["preregistration"]["deviations"] and audit["confirmatory"] is False


def test_el_sufijo_de_la_carpeta_debe_ser_el_commit_del_manifiesto(run_dir, tmp_path, monkeypatch):  # noqa: F811
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    manifest = _read(copy, "manifest.json")
    manifest["commit"] = manifest["git_at_end"]["commit"] = COMMIT_B
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = porobra_audit.audit_run_directory(copy)
    assert set(audit["preregistration"]["deviations"]) == {"commit_directory"}, audit["preregistration"]
    assert audit["confirmatory"] is False


def test_un_manifiesto_sin_el_estado_final_de_git_no_es_confirmatorio(run_dir, tmp_path, monkeypatch):  # noqa: F811
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    manifest = _read(copy, "manifest.json")
    del manifest["git_at_end"]
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    deviations = porobra_audit.audit_run_directory(copy)["preregistration"]["deviations"]
    assert {"commit_changed_during_run", "dirty_worktree"} <= set(deviations)


# --- sobre qué corpus -------------------------------------------------------------------


def test_otro_corpus_con_los_mismos_conteos_no_es_confirmatorio(run_dir, tmp_path, monkeypatch):  # noqa: F811
    # Una raíz equivocada con el conteo correcto: el conteo pasa, la identidad no.
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    pinned = dict(porobra_audit.PREREGISTERED_CORPUS_SHA256)
    pinned["SymbTr"] = "0" * 64
    monkeypatch.setattr(porobra_audit, "PREREGISTERED_CORPUS_SHA256", pinned)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "passed"
    deviation = audit["preregistration"]["deviations"]["corpus_identity"]
    assert set(deviation["observed"]) == {"SymbTr"} and deviation["expected"] == {"SymbTr": "0" * 64}
    assert audit["confirmatory"] is False
    assert porobra_audit.main([str(copy), "--require-preregistered"]) == 1


def test_el_corpus_fijado_tiene_hash_por_fuente_de_la_seccion_a():
    pinned = porobra_audit.PREREGISTERED_CORPUS_SHA256
    assert set(pinned) == set(porobra_audit.PREREGISTERED_CORPUS) == {"MuseTrainer", "SymbTr"}
    assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest in pinned.values())


def test_los_registros_ilegibles_no_lanzan(run_dir, tmp_path, monkeypatch):  # noqa: F811
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    manifest = _read(copy, "manifest.json")
    del manifest["inputs"]["files"][0]["sha256"]
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "failed" and "corpus_identity" in audit["preregistration"]["deviations"]


def test_el_hash_de_entrada_no_depende_del_fin_de_linea(tmp_path):
    # git con autocrlf deja CRLF en Windows y LF en Debian: es el mismo corpus (decisión 3).
    lf, crlf = tmp_path / "lf.xml", tmp_path / "crlf.xml"
    lf.write_bytes(b"<a>\n<b/>\n</a>\n")
    crlf.write_bytes(b"<a>\r\n<b/>\r\n</a>\r\n")
    assert porobra_heldout.input_sha256(lf) == porobra_heldout.input_sha256(crlf)
    assert porobra_heldout.input_sha256(lf) == hashlib.sha256(lf.read_bytes()).hexdigest()
    other = tmp_path / "otro.xml"
    other.write_bytes(b"<a>\n<c/>\n</a>\n")
    assert porobra_heldout.input_sha256(other) != porobra_heldout.input_sha256(lf)


def test_una_muestra_ya_recortada_no_coincide_con_el_corpus_fijado(tmp_path):
    # La raíz por descuido del hallazgo: mismos conteos, otros archivos, otro agregado.
    folder = tmp_path / "SymbTr"
    folder.mkdir()
    write_scale_score(folder / "unico--sarki--aksak--uno--anonim.xml", "S", "Y", _melody(1, 24))
    args = porobra_heldout.build_parser().parse_args(["--symbtr-root", str(folder), "--symbtr-n", "100"])
    files = porobra_heldout.input_files(porobra_heldout._catalog(args))
    observed = porobra_audit.source_aggregates({"inputs": {"files": files}})
    assert observed["SymbTr"] != porobra_audit.PREREGISTERED_CORPUS_SHA256["SymbTr"]


# --- qué fue de cada candidato ----------------------------------------------------------


def _break_extraction(monkeypatch, marker: str) -> None:
    """Un error de entorno que tumba la extracción de todo lo que tenga ``marker`` en la ruta."""

    original = library_batch._extract_piece_events

    def extract(path, *args):
        if marker in str(path).replace("\\", "/"):
            raise OSError("entorno roto")
        return original(path, *args)

    monkeypatch.setattr(library_batch, "_extract_piece_events", extract)


def test_si_el_entorno_tumba_la_extraccion_de_un_corpus_no_es_confirmatoria(corpus, tmp_path, as_section_a, monkeypatch, capsys):  # noqa: F811
    _break_extraction(monkeypatch, "/SymbTr/")
    run = porobra_heldout.main([*_corpus_argv(corpus), "--output-root", str(tmp_path)])
    audit, summary = _read(run, porobra_audit.AUDIT_FILENAME), _read(run, "summary.json")
    # Los archivos y los conteos son los fijados; lo que cambió es qué fue de cada uno.
    assert audit["status"] == "passed"
    deviations = audit["preregistration"]["deviations"]
    # Las obras que faltan cambian también la huella de contenido.
    assert set(deviations) == {"corpus_outcomes", "corpus_content"}, deviations
    assert deviations["corpus_outcomes"]["observed"]["falla_extraccion"] == [
        "MuseTrainer:rota.musicxml", *(item for item in EXPECTED_OUTCOMES["analizada"] if item.startswith("SymbTr:"))]
    assert deviations["corpus_outcomes"]["expected"]["falla_extraccion"] == ["MuseTrainer:rota.musicxml"]
    assert audit["confirmatory"] is False
    assert summary["run"]["run_kind"] == "no_preregistrada" and summary["run"]["deviations"] == ["corpus_content", "corpus_outcomes"]
    out = capsys.readouterr().out
    assert "AVISO: 4 falla(s) de extracción" in out and "corpus_outcomes" in out
    for item in deviations["corpus_outcomes"]["observed"]["falla_extraccion"]:
        assert item in out


def test_el_aviso_de_extraccion_sale_aunque_ninguna_obra_se_pueda_analizar(corpus, tmp_path, as_section_a, monkeypatch, capsys):  # noqa: F811
    _break_extraction(monkeypatch, "/")
    with pytest.raises(SystemExit):
        porobra_heldout.main([*_corpus_argv(corpus), "--output-root", str(tmp_path)])
    assert "AVISO: 8 falla(s) de extracción" in capsys.readouterr().out


def test_sin_fallas_de_extraccion_no_hay_aviso_de_extraccion(corpus, tmp_path, monkeypatch, capsys):  # noqa: F811
    library = tmp_path / "limpia"
    shutil.copytree(corpus["library"], library)
    (library / "rota.musicxml").unlink()
    _stub_git(monkeypatch, {"commit": COMMIT_A, "status": ""})
    monkeypatch.setattr(porobra_heldout, "_run_jobs", _stub_jobs())
    porobra_heldout.main(["--musetrainer-root", str(library), "--output-root", str(tmp_path / "salida")])
    assert "extracción" not in capsys.readouterr().out


def test_otro_destino_para_un_solo_candidato_es_desviacion(run_dir, tmp_path, monkeypatch):  # noqa: F811
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    summary = _read(copy, "summary.json")
    outcomes = porobra_audit.corpus_outcomes(porobra_audit.read_per_work(copy / "per_work.csv"), summary)
    expected = {**outcomes, "corta": [], "analizada": sorted([*outcomes["analizada"], *outcomes["corta"]])}
    pin_outcomes(monkeypatch, expected)
    audit = porobra_audit.audit_run_directory(copy)
    assert audit["status"] == "passed"
    deviation = audit["preregistration"]["deviations"]["corpus_outcomes"]
    assert set(audit["preregistration"]["deviations"]) == {"corpus_outcomes"}
    assert deviation["observed"]["corta"] == outcomes["corta"] and deviation["expected"]["corta"] == []
    assert deviation["observed"]["sha256"] != deviation["expected"]["sha256"]
    assert audit["confirmatory"] is False


def test_una_falla_en_el_analisis_tambien_cambia_el_destino(corpus, tmp_path, as_section_a, monkeypatch):  # noqa: F811
    def run_jobs(jobs, n_workers, partial_path):
        rows = _stub_jobs()(jobs, n_workers, partial_path)
        return [{"work_id": rows[0]["work_id"], "error": "ValueError: simulado"}, *rows[1:]]

    monkeypatch.setattr(porobra_heldout, "_run_jobs", run_jobs)
    run = porobra_heldout.main([*_corpus_argv(corpus), "--output-root", str(tmp_path)])
    audit = _read(run, porobra_audit.AUDIT_FILENAME)
    assert audit["status"] == "failed"
    assert audit["preregistration"]["deviations"]["corpus_outcomes"]["observed"]["falla_analisis"] == ["MuseTrainer:a.musicxml"]


def test_la_huella_no_depende_de_la_barra_ni_de_la_forma_unicode():
    # Windows puede dar «\\» y otro sistema de archivos otra forma Unicode: es la misma ruta.
    works = pd.DataFrame({"source": ["SymbTr"], "relative_path": ["sub/caf\u00e9.xml"]})
    other = pd.DataFrame({"source": ["SymbTr"], "relative_path": ["sub\\cafe\u0301.xml"]})
    left = porobra_audit.corpus_outcomes(works, {"corpus": {"duplicates": [{"source": "SymbTr", "relative_path": "x/y.xml"}]}})
    right = porobra_audit.corpus_outcomes(other, {"corpus": {"duplicates": [{"source": "SymbTr", "relative_path": "x\\y.xml"}]}})
    assert left == right and left["analizada"] == ["SymbTr:sub/caf\u00e9.xml"] and left["duplicado"] == ["SymbTr:x/y.xml"]
    assert porobra_audit.corpus_outcomes_sha256(left) == porobra_audit.corpus_outcomes_sha256(right)


def test_la_huella_cambia_con_el_destino_y_no_con_el_orden():
    base = {"analizada": ["a", "b"], "corta": ["c"]}
    assert (porobra_audit.corpus_outcomes_sha256(base)
            == porobra_audit.corpus_outcomes_sha256({"corta": ["c"], "analizada": ["b", "a"]}))
    assert (porobra_audit.corpus_outcomes_sha256(base)
            != porobra_audit.corpus_outcomes_sha256({"analizada": ["a", "c"], "corta": ["b"]}))


def test_los_destinos_fijados_son_de_la_seccion_a_y_estan_normalizados():
    pinned = porobra_audit.PREREGISTERED_CORPUS_OUTCOMES
    digest = porobra_audit.PREREGISTERED_CORPUS_OUTCOMES_SHA256
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert set(pinned) <= {"falla_extraccion", "duplicado", "corta"}
    names = [name for group in pinned.values() for name in group]
    assert len(names) == len(set(names)) < sum(porobra_audit.PREREGISTERED_CORPUS.values())
    assert all(name.split(":", 1)[0] in porobra_audit.PREREGISTERED_CORPUS and "\\" not in name
               and name == unicodedata.normalize("NFC", name) for name in names)
    assert all(list(group) == sorted(group) for group in pinned.values())


# --- recálculo entre sistemas operativos ------------------------------------------------


def _set_p_value(summary: dict, value_of) -> None:
    summary["comparisons"]["comparisons"][0]["p_value"] = value_of(summary["comparisons"]["comparisons"][0]["p_value"])


def test_una_diferencia_de_ultimo_bit_no_falla_el_recalculo(run_dir, tmp_path):  # noqa: F811
    # Otra libm: 1.6188491294922218e-19 contra 1.6188491294922213e-19 en un p secundario.
    copy = _copy(run_dir, tmp_path)
    summary = _read(copy, "summary.json")
    _set_p_value(summary, lambda p: np.nextafter(p, 1.0).item())
    _rewrite_summary(copy, summary)
    check = _check(porobra_audit.audit_run_directory(copy), "headline_recompute")
    assert check["status"] == "passed" and check["mismatches"] == []


def test_una_diferencia_real_si_falla_el_recalculo(run_dir, tmp_path):  # noqa: F811
    copy = _copy(run_dir, tmp_path)
    summary = _read(copy, "summary.json")
    summary["comparisons"]["comparisons"][0]["mean_difference"] *= 1 + 1e-9
    _rewrite_summary(copy, summary)
    check = _check(porobra_audit.audit_run_directory(copy), "headline_recompute")
    assert check["status"] == "failed" and "mean_difference" in check["mismatches"][0]


def test_el_recalculo_sigue_siendo_exacto_en_lo_que_no_es_float():
    assert porobra_audit._differences({"verdict": "mejora"}, {"verdict": "no_hay_evidencia"})
    assert porobra_audit._differences({"n_pairs": 5}, {"n_pairs": 6})
    assert porobra_audit._differences({"x": 1}, {"x": 1.0})
    assert porobra_audit._differences({"x": None}, {"x": 1.0})
    assert porobra_audit._differences({"x": True}, {"x": 1.0})
    assert not porobra_audit._differences({"x": 1.0 + 1e-15}, {"x": 1.0})
    # Cerca de cero rige la tolerancia absoluta: una resta que da 0 en un sistema y 1e-17 en otro.
    assert not porobra_audit._differences({"x": 1e-17}, {"x": 0.0})
    assert porobra_audit._differences({"x": 1e-6}, {"x": 0.0})

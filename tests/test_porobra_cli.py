"""CLI del análisis por obra: directorio nuevo, CSV por obra, resumen, manifiesto y conteos."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import pytest

from next_token_experiment.tests.test_pipeline import write_scale_score
from src.cli import porobra_heldout

FAST = ["--truncation", "4", "--chains", "2", "--iters", "8", "--burn-in", "4", "--k-grid", "2,4",
        "--etas", "1,3", "--bootstrap-samples", "200", "--occupancy-samples", "10", "--finite-restarts", "2"]


def _melody(seed: int, n: int) -> list[int]:
    return [60 + ((seed * 7 + index * (seed + 2)) % 12) for index in range(n)]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("corpus")
    library = root / "library"
    symbtr = root / "SymbTr" / "MusicXML"
    library.mkdir()
    symbtr.mkdir(parents=True)
    write_scale_score(library / "a.musicxml", "A", "X", _melody(1, 30))
    write_scale_score(library / "b.musicxml", "B", "X", _melody(2, 26))
    write_scale_score(library / "b_copia.musicxml", "B2", "X", _melody(2, 26))  # duplicado por contenido
    write_scale_score(library / "corta.musicxml", "C", "X", _melody(3, 12))  # < 20 eventos
    (library / "rota.musicxml").write_text("<score-partwise><no-es-musica", encoding="utf-8")
    for makam, seed in (("hicaz", 4), ("hicaz", 5), ("rast", 6)):
        write_scale_score(symbtr / f"{makam}--sarki--aksak--obra{seed}--anonim.xml", "S", "Y", _melody(seed, 24))
    return {"library": library, "symbtr": root / "SymbTr"}


def _run(corpus, output_root: Path, *extra: str) -> Path:
    argv = ["--musetrainer-root", str(corpus["library"]), "--symbtr-root", str(corpus["symbtr"]),
            "--symbtr-n", "2", "--output-root", str(output_root), "--run-name", "prueba", *FAST, *extra]
    return porobra_heldout.main(argv)


def test_corrida_escribe_directorio_nuevo_con_csv_resumen_y_manifiesto(corpus, tmp_path):
    run_dir = _run(corpus, tmp_path, "--n-workers", "1")
    assert re.fullmatch(r"prueba_\d{8}_[0-9a-f]{7}", run_dir.name)
    works = pd.read_csv(run_dir / "per_work.csv")
    # 2 de MuseTrainer (b_copia es duplicado, corta tiene < 20) + 2 de SymbTr (muestra).
    assert len(works) == 4
    assert set(works["source"]) == {"MuseTrainer", "SymbTr"}
    for column in ("hdp_test_nll", "finite_test_nll", "bigram_test_nll", "unigram_test_nll", "hdp_eta",
                   "finite_k", "unigram_alpha", "bigram_alpha", "hdp_rhat", "hdp_converged",
                   "hdp_states_mean", "finite_states_mean"):
        assert column in works.columns
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["corpus"]["n_excluded_short"] == 1
    assert summary["corpus"]["excluded_short_works"][0]["n_events"] == 12
    assert summary["corpus"]["n_duplicates"] == 1
    assert summary["corpus"]["n_failures"] == 1
    assert summary["corpus"]["failures"][0]["path"].endswith("rota.musicxml")
    assert summary["corpus"]["n_analyzed"] == 4
    assert summary["convergence"]["n_works"] == 4
    assert [c["model_b"] for c in summary["comparisons"]["comparisons"]] == ["finite", "bigram", "unigram"]
    assert summary["comparisons"]["holm_family"]["family_size"] == 3
    # Enmienda 6: la métrica principal usa todas las obras; la regla original es sensibilidad.
    assert summary["comparisons"]["n_works"] == 4 and summary["comparisons"]["converged_only"] is False
    sensitivity = summary["sensitivity_converged_only"]
    assert sensitivity["converged_only"] is True
    assert sensitivity["n_works"] == int(works["hdp_converged"].sum())
    assert summary["preregistro"].endswith("Enmiendas 3, 5 y 6")
    assert "Enmienda 6" in summary["convergence"]["rule"]
    assert "media posterior" in summary["hdp_states_note"]
    for column in ("hdp_test_nll_by_chain", "hdp_test_nll_chain_range", "hdp_predictive_rhat",
                   "hdp_states_mean_by_chain", "hdp_states_sd_by_chain"):
        assert column in works.columns
    assert all(len(json.loads(value)) == 2 for value in works["hdp_test_nll_by_chain"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["commit"]) == 40
    assert manifest["config"]["truncation"] == 4 and manifest["config"]["n_chains"] == 2
    assert manifest["config"]["chain_seeds"] == [1, 2]
    assert manifest["config"]["finite_restarts"] == 2
    assert set(works["finite_restarts"]) == {2}
    assert len(manifest["inputs"]["files"]) == 7  # 5 de MuseTrainer + 2 de la muestra de SymbTr
    assert all(len(item["sha256"]) == 64 for item in manifest["inputs"]["files"])
    assert "numpy" in manifest["versions"] and "music21" in manifest["versions"]
    assert set(manifest["outputs"]) == {"per_work.csv", "summary.json"}


def test_no_sobrescribe_una_corrida_existente(corpus, tmp_path):
    run_dir = porobra_heldout.run_directory(tmp_path, "prueba")
    run_dir.mkdir()
    with pytest.raises(FileExistsError):
        _run(corpus, tmp_path)


def test_en_paralelo_da_lo_mismo_que_en_serie(corpus, tmp_path):
    serial = pd.read_csv(_run(corpus, tmp_path / "serie", "--n-workers", "1") / "per_work.csv")
    parallel = pd.read_csv(_run(corpus, tmp_path / "paralelo", "--n-workers", "2") / "per_work.csv")
    timing = [c for c in serial.columns if c.endswith("wall_clock_s")]
    pd.testing.assert_frame_equal(serial.drop(columns=timing), parallel.drop(columns=timing))


def test_muestra_de_symbtr_estratificada_por_makam(corpus):
    paths = porobra_heldout.symbtr_paths(corpus["symbtr"], n=2, seed=7)
    assert sorted(p.name.split("--")[0] for p in paths) == ["hicaz", "rast"]


def test_rechaza_un_minimo_de_eventos_menor_que_el_del_preregistro(corpus, tmp_path):
    # La partición exige 20 eventos: aceptar --min-events 10 sólo aplazaba el fallo al análisis.
    with pytest.raises(SystemExit):
        _run(corpus, tmp_path, "--min-events", "10")
    assert not list(tmp_path.iterdir())


def test_una_obra_que_falla_no_tumba_la_corrida(corpus, tmp_path, monkeypatch):
    original = porobra_heldout.porobra.analyze_work

    def fragile(work_id, tokens, config):
        if work_id.endswith("a.musicxml"):
            raise ValueError("fallo simulado")
        return original(work_id, tokens, config)

    monkeypatch.setattr(porobra_heldout.porobra, "analyze_work", fragile)
    run_dir = _run(corpus, tmp_path, "--n-workers", "1")
    works = pd.read_csv(run_dir / "per_work.csv")
    assert len(works) == 3 and not works["work_id"].str.endswith("a.musicxml").any()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    failed = summary["analysis_failures"]
    assert [item["work_id"] for item in failed] == ["MuseTrainer:a.musicxml"]
    assert "ValueError" in failed[0]["error"] and "fallo simulado" in failed[0]["error"]
    assert summary["corpus"]["n_analyzed"] == 3
    assert summary["convergence"]["n_works"] == 3


def test_resultados_parciales_quedan_en_disco_si_la_corrida_se_interrumpe(corpus, tmp_path, monkeypatch):
    original = porobra_heldout.porobra.analyze_work
    calls = []

    def interrupted(work_id, tokens, config):
        calls.append(work_id)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(work_id, tokens, config)

    monkeypatch.setattr(porobra_heldout.porobra, "analyze_work", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(corpus, tmp_path, "--n-workers", "1")
    (run_dir,) = tmp_path.iterdir()
    lines = (run_dir / "per_work.partial.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["work_id"] for line in lines] == [calls[0]]
    assert not (run_dir / "summary.json").exists()

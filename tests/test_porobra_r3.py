"""Revisión r3 de la tarea 1.4: contenido fijado, tope de iteraciones de EM y fallas en el README."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analysis import porobra, porobra_audit
from src.cli import porobra_heldout
from tests.test_porobra_audit import _as_preregistered, _rewrite_per_work, run_dir  # noqa: F401
from tests.test_porobra_cli import corpus  # noqa: F401


def test_la_huella_de_contenido_no_depende_del_orden_ni_del_separador():
    works = pd.DataFrame({"source": ["B", "A"], "relative_path": ["x\\y.xml", "z.xml"], "content_fingerprint": ["f2", "f1"]})
    flipped = pd.DataFrame({"source": ["A", "B"], "relative_path": ["z.xml", "x/y.xml"], "content_fingerprint": ["f1", "f2"]})
    assert porobra_audit.corpus_content_sha256(works) == porobra_audit.corpus_content_sha256(flipped)
    changed = flipped.assign(content_fingerprint=["f1", "otra"])
    assert porobra_audit.corpus_content_sha256(changed) != porobra_audit.corpus_content_sha256(flipped)


def test_un_cambio_de_eventos_sin_cambio_de_destino_es_desviacion(run_dir, tmp_path, monkeypatch):  # noqa: F811
    copy = _as_preregistered(run_dir, tmp_path, monkeypatch)
    assert porobra_audit.audit_run_directory(copy)["confirmatory"] is True
    works = porobra_audit.read_per_work(copy / "per_work.csv")
    works.loc[0, "content_fingerprint"] = "0" * 40
    _rewrite_per_work(copy, works)
    audit = porobra_audit.audit_run_directory(copy)
    assert "corpus_content" in audit["preregistration"]["deviations"]
    assert audit["confirmatory"] is False


def test_el_ajuste_registra_si_em_llego_al_tope_de_iteraciones():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 12, size=200)
    capped = porobra.baum_welch(tokens, n_states=8, seed=0, max_iterations=2, pseudocount=0.1)
    assert capped.hit_max_iterations is True
    free = porobra.baum_welch(tokens, n_states=2, seed=0, max_iterations=300, pseudocount=1.0)
    assert free.hit_max_iterations is False


def test_fit_finite_reporta_el_tope_de_em_del_ajuste_seleccionado():
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 12, size=120)
    result = porobra.fit_finite(tokens, porobra.split_tokens(tokens), grid=(2, 4), n_restarts=2, n_occupancy_samples=5)
    assert isinstance(result["finite_em_hit_max_iterations"], bool)


def test_el_readme_nombra_las_fallas_de_extraccion():
    run = {"truncation": 20, "run_kind": "principal", "note": "candidata", "decides": "x",
           "extraction_failures": ["MuseTrainer:rota.musicxml"]}
    assert "MuseTrainer:rota.musicxml" in porobra_heldout._readme(run)
    assert "Enmienda 11" in porobra_heldout._readme(run)

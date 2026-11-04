"""Enmienda 14: en la sección B, el tope de K es final; tocarlo marca la fila y no detiene la corrida."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from Comparacion import runner
from Comparacion.classical_models import FiniteGlobalHMM
from Comparacion.cli import build_parser
from Comparacion.config import build_default_learning_curve_config
from next_token_experiment.schemas import PreparedPiece


def _piece(i: int) -> PreparedPiece:
    tokens = [i % 12, (i + 1) % 12, (i + 2) % 12, (i + 3) % 12]
    return PreparedPiece(piece_id=f"p{i}", source_path="x", title="t", composer="c", canonical_work_id=f"w{i}",
                         representation="pitch_class", vocabulary=[str(v) for v in range(12)], tokens=tokens,
                         n_events=len(tokens))


def _fake_candidate(nll_of_k):
    def fake(self, *, n_states, **_):
        return (np.ones(n_states) / n_states, np.ones((n_states, n_states)) / n_states,
                np.ones((n_states, self.vocab_size)) / self.vocab_size, nll_of_k(n_states), [{"iteration": 1}],
                {"fit_wall_clock_s": 0.0, "validation_wall_clock_s": 0.0})

    return fake


def _fit(**kwargs):
    hmm = FiniteGlobalHMM(candidate_num_states=(2, 4), max_iterations=1, tolerance=1e-4, seed=0, vocab_size=13,
                          max_states=8, **kwargs)
    return hmm.fit([_piece(1)], [_piece(2)], bos_token_id=12)


def test_sin_tope_final_tocar_el_limite_sigue_fallando(monkeypatch):
    monkeypatch.setattr(FiniteGlobalHMM, "_fit_candidate", _fake_candidate(lambda k: 10.0 / k))
    with pytest.raises(ValueError):
        _fit()


def test_con_tope_final_la_seleccion_se_queda_en_el_tope_y_se_marca(monkeypatch):
    monkeypatch.setattr(FiniteGlobalHMM, "_fit_candidate", _fake_candidate(lambda k: 10.0 / k))
    result = _fit(final_ceiling=True)
    assert result.selected_states == 8
    assert result.candidate_grid == (2, 4, 6, 8)
    assert result.hit_final_ceiling is True


def test_sin_tocar_el_tope_no_hay_marca(monkeypatch):
    monkeypatch.setattr(FiniteGlobalHMM, "_fit_candidate", _fake_candidate(float))
    result = _fit(final_ceiling=True)
    assert result.selected_states == 2
    assert result.hit_final_ceiling is False


def test_cli_y_configuracion_llevan_el_tope_final():
    config = build_default_learning_curve_config()
    assert config.finite_hmm_final_ceiling is False
    assert build_parser().parse_args(["--finite-hmm-final-ceiling"]).finite_hmm_final_ceiling is True
    assert build_parser().parse_args([]).finite_hmm_final_ceiling is False


def test_la_auditoria_de_rejilla_declara_el_tope_final():
    config = build_default_learning_curve_config()
    assert runner._finite_grid_limit_reason(config) is None
    final = replace(config, finite_hmm_max_states=768, finite_hmm_final_ceiling=True)
    reason = runner._finite_grid_limit_reason(final)
    assert "768" in reason and "Enmienda 14" in reason
    rows = [{"model": "finite_hmm", "selected_states": 768, "candidate_grid": "[12, 768]"}]
    audit = runner.build_finite_hmm_grid_audit(rows, grid=(12, 768), resource_limit_reason=reason)
    assert audit["verdict"] == "grid_limited_by_resources"
    assert audit["plateau_claimed"] is False

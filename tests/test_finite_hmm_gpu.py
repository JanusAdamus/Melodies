"""Paso E del HMM finito en torch: mismo resultado que el de numpy, en CPU y en CUDA."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from Comparacion.classical_models import FiniteGlobalHMM, _expectation_batch, _expectation_batch_torch, _pad_sequences  # noqa: E402
from Comparacion.cli import build_parser  # noqa: E402
from Comparacion.config import build_default_learning_curve_config  # noqa: E402
from next_token_experiment.schemas import PreparedPiece  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _random_problem(seed: int, n_states: int, vocab: int = 13):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 40, size=25)
    sequences = [rng.integers(0, vocab, size=int(n)) for n in lengths]
    initial = rng.dirichlet(np.ones(n_states))
    transition = rng.dirichlet(np.ones(n_states), size=n_states)
    emission = rng.dirichlet(np.ones(vocab), size=n_states)
    return initial, transition, emission, _pad_sequences(sequences)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_states", [3, 17])
def test_el_paso_e_en_torch_coincide_con_numpy(device, n_states):
    initial, transition, emission, batches = _random_problem(n_states, n_states)
    for padded, lengths in batches:
        expected = _expectation_batch(initial, transition, emission, padded, lengths)
        observed = _expectation_batch_torch(initial, transition, emission, padded, lengths, device=device)
        for exp, obs in zip(expected[:3], observed[:3]):
            assert isinstance(obs, np.ndarray) and obs.dtype == np.float64
            np.testing.assert_allclose(obs, exp, rtol=1e-10, atol=1e-12)
        assert observed[3] == pytest.approx(expected[3], rel=1e-12, abs=1e-9)


def _pieces(seed: int, n: int) -> list[PreparedPiece]:
    rng = np.random.default_rng(seed)
    pieces = []
    for i in range(n):
        tokens = [int(t) for t in rng.integers(0, 12, size=int(rng.integers(8, 30)))]
        pieces.append(PreparedPiece(piece_id=f"p{seed}_{i}", source_path="x", title="t", composer="c",
                                    canonical_work_id=f"w{seed}_{i}", representation="pitch_class",
                                    vocabulary=[str(v) for v in range(12)], tokens=tokens, n_events=len(tokens)))
    return pieces


@pytest.mark.parametrize("device", DEVICES)
def test_el_ajuste_completo_en_torch_da_lo_mismo_que_en_numpy(device, monkeypatch):
    from Comparacion import classical_models

    monkeypatch.setattr(classical_models, "FINITE_HMM_GPU_MIN_STATES", 1)
    calls = []
    real = classical_models._expectation_batch_torch
    monkeypatch.setattr(classical_models, "_expectation_batch_torch", lambda *a, **k: calls.append(1) or real(*a, **k))
    train, validation = _pieces(1, 30), _pieces(2, 10)
    results = []
    for chosen in ("numpy", device):
        hmm = FiniteGlobalHMM(candidate_num_states=(2, 4, 8), max_iterations=15, tolerance=1e-4, seed=3,
                              vocab_size=13, device=None if chosen == "numpy" else chosen)
        fit = hmm.fit(train, validation, bos_token_id=12)
        results.append((fit, hmm))
    (base_fit, base), (fit, hmm) = results
    assert calls, "el ajuste con dispositivo no pasó por torch"
    assert fit.selected_states == base_fit.selected_states
    assert fit.validation_nll == pytest.approx(base_fit.validation_nll, rel=1e-9)
    assert [row["n_iterations"] for row in fit.candidate_log] == [row["n_iterations"] for row in base_fit.candidate_log]
    np.testing.assert_allclose(hmm.transition_matrix, base.transition_matrix, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(hmm.emission_matrix, base.emission_matrix, rtol=1e-7, atol=1e-10)


def test_sin_dispositivo_se_conserva_el_camino_de_numpy():
    assert FiniteGlobalHMM(candidate_num_states=(2,), max_iterations=1, tolerance=1e-4, seed=0).device is None


def test_la_configuracion_y_la_cli_llevan_el_dispositivo_del_hmm_finito():
    config = build_default_learning_curve_config()
    assert config.finite_hmm_device is None
    assert replace(config, finite_hmm_device="cuda").finite_hmm_device == "cuda"
    assert build_parser().parse_args(["--finite-hmm-device", "cuda"]).finite_hmm_device == "cuda"
    assert build_parser().parse_args([]).finite_hmm_device is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sin CUDA")
def test_el_paso_e_en_cuda_es_reproducible_bit_a_bit():
    initial, transition, emission, batches = _random_problem(5, 24)
    padded, lengths = batches[-1]
    runs = [_expectation_batch_torch(initial, transition, emission, padded, lengths, device="cuda") for _ in range(4)]
    for other in runs[1:]:
        for a, b in zip(runs[0][:3], other[:3]):
            assert np.array_equal(a, b)
        assert runs[0][3] == other[3]


def test_un_dispositivo_invalido_falla_al_construir_el_modelo():
    with pytest.raises((ValueError, RuntimeError)):
        FiniteGlobalHMM(candidate_num_states=(2,), max_iterations=1, tolerance=1e-4, seed=0, device="gpu")


@pytest.mark.parametrize("bad", ["", "gpu", "CPU", "cuda:x", "tpu"])
def test_la_configuracion_rechaza_dispositivos_mal_escritos(bad):
    with pytest.raises(ValueError):
        replace(build_default_learning_curve_config(), finite_hmm_device=bad)


def test_los_k_chicos_usan_numpy_aunque_haya_dispositivo(monkeypatch):
    from Comparacion import classical_models

    used = []
    real_numpy, real_torch = classical_models._expectation_batch, classical_models._expectation_batch_torch
    monkeypatch.setattr(classical_models, "_expectation_batch",
                        lambda *a, **k: used.append(("numpy", a[1].shape[0])) or real_numpy(*a, **k))
    monkeypatch.setattr(classical_models, "_expectation_batch_torch",
                        lambda *a, **k: used.append(("torch", a[1].shape[0])) or real_torch(*a, **k))
    hmm = FiniteGlobalHMM(candidate_num_states=(4, classical_models.FINITE_HMM_GPU_MIN_STATES), max_iterations=1,
                          tolerance=1e-4, seed=0, vocab_size=13, device="cpu")
    hmm.fit(_pieces(1, 6), _pieces(2, 3), bos_token_id=12)
    assert {kind for kind, k in used if k == 4} == {"numpy"}
    assert {kind for kind, k in used if k == classical_models.FINITE_HMM_GPU_MIN_STATES} == {"torch"}
    used.clear()
    base = FiniteGlobalHMM(candidate_num_states=(classical_models.FINITE_HMM_GPU_MIN_STATES,), max_iterations=1,
                           tolerance=1e-4, seed=0, vocab_size=13)
    base.fit(_pieces(1, 6), _pieces(2, 3), bos_token_id=12)
    assert {kind for kind, _ in used} == {"numpy"}


def test_el_runner_rotula_el_dispositivo_y_mide_la_gpu_del_hmm_finito():
    from Comparacion import runner

    config = build_default_learning_curve_config()
    assert runner._finite_hmm_device_label(config) == "cpu"
    assert runner._finite_hmm_measures_cuda(config) is False
    gpu = replace(config, finite_hmm_device="cuda")
    assert runner._finite_hmm_device_label(gpu) == "cuda"
    assert runner._finite_hmm_measures_cuda(gpu) is True
    source = open(runner.__file__, encoding="utf-8").read()
    assert source.count("_finite_hmm_device_label(config)") >= 2
    assert source.count("measure_cuda=_finite_hmm_measures_cuda(config)") == 2

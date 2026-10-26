"""Chequeos de Comparacion/classical_models.py (FiniteGlobalHMM): paso E, verosimilitud condicional, BOS."""
import itertools
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")
from Comparacion.classical_models import FiniteGlobalHMM, _expectation_batch, conditional_log_likelihood
from src.models.utils import length_buckets

rng = np.random.default_rng(0)
K, V = 3, 4
pi = rng.dirichlet(np.ones(K)); A = rng.dirichlet(np.ones(K), K); B = rng.dirichlet(np.ones(V), K)
seqs = [rng.integers(0, V, n) for n in (1, 2, 5, 6)]


def brute(x):
    """Recuentos esperados exactos por enumeracion de caminos."""
    T = len(x); Z = 0.0
    g0 = np.zeros(K); tc = np.zeros((K, K)); ec = np.zeros((K, V))
    for path in itertools.product(range(K), repeat=T):
        p = pi[path[0]] * B[path[0], x[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * B[path[t], x[t]]
        Z += p; g0[path[0]] += p
        for t in range(T):
            ec[path[t], x[t]] += p
        for t in range(1, T):
            tc[path[t - 1], path[t]] += p
    return g0 / Z, tc / Z, ec / Z, np.log(Z)


ref = [brute(x) for x in seqs]
refs = [sum(r[i] for r in ref) for i in range(4)]
got = [np.zeros(K), np.zeros((K, K)), np.zeros((K, V)), 0.0]
for _, padded, lengths in length_buckets(seqs):
    out = _expectation_batch(pi, A, B, padded, lengths)
    got = [g + o for g, o in zip(got, out)]
for name, r, g in zip(["inicial", "transicion", "emision", "logL"], refs, got):
    print(f"E-step {name:10s} max|diff| = {np.max(np.abs(np.asarray(r) - np.asarray(g))):.2e}")

# Verosimilitud condicional log p(y_1:n | y_0 = BOS) por fuerza bruta
bos = V - 1
tgt = [0, 2, 1, 1]
num = brute(np.array([bos] + tgt))[3]; den = brute(np.array([bos]))[3]
print("cond. LL brute =", num - den, " codigo =", conditional_log_likelihood(
    initial_probs=pi, transition_matrix=A, emission_matrix=B, context_token=bos, target_tokens=tgt))

# Masa que un HMM entrenado reserva a BOS como siguiente simbolo tras el primer paso.
# Entrenar el HMM real en datos sinteticos de 12 simbolos + BOS
V2 = 13; BOS = 12
def piece(i, n):
    return SimpleNamespace(piece_id=str(i), title="", composer="", tokens=list(rng.choice([0, 2, 4, 5, 7, 9, 11], n)))
train = [piece(i, 60) for i in range(30)]; val = [piece(100 + i, 60) for i in range(10)]
h = FiniteGlobalHMM(candidate_num_states=(4,), max_iterations=30, tolerance=1e-4, seed=1, vocab_size=V2)
fr = h.fit(train, val, bos_token_id=BOS, max_context_length=128)
alpha = h.initial_probs * h.emission_matrix[:, BOS]; alpha /= alpha.sum()
for t in range(1, 5):
    pred = (alpha @ h.transition_matrix) @ h.emission_matrix
    print(f"  paso {t}: P(siguiente = BOS) = {pred[BOS]:.2e}")
    alpha = alpha @ h.transition_matrix * h.emission_matrix[:, 0]; alpha /= alpha.sum()
print("iteraciones por candidato:", [c["n_iterations"] for c in fr.candidate_log],
      "| validacion por iter:", [round(r["validation_nll_per_token"], 5) for r in fr.train_log][-4:])

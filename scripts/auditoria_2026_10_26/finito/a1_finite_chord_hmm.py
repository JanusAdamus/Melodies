"""Chequeos de FiniteChordHMM (src/models/finite_hmm.py) y de la comparacion por obra."""
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
from src.models.finite_hmm import FiniteChordHMM
from src.models.inference import forward_log_likelihood, scaled_forward_log_likelihood, viterbi_decode
from src.data.parsing import extract_events, parse_score
from src.data.observations import build_observation_sequence
from src.models.hdp_hmm import TruncatedHDPHMM

m = FiniteChordHMM()
pi, A, B = m.initial_probs, m.transition_matrix, m.emission_matrix
print("K =", m.n_states, "| filas A suman 1:", np.allclose(A.sum(1), 1), "| filas B:", np.allclose(B.sum(1), 1))
print("min A =", A.min(), " min B =", B.min())

# 1. forward vs referencia exacta en dominio lineal (producto de matrices) y vs fuerza bruta en T=2
rng = np.random.default_rng(0)
for T in (1, 2, 5, 50, 400):
    x = rng.integers(0, 12, T)
    ll, _ = forward_log_likelihood(pi, A, B, x)
    a = pi * B[:, x[0]]
    ref = np.log(a.sum()); a = a / a.sum()
    for t in range(1, T):
        a = (a @ A) * B[:, x[t]]; ref += np.log(a.sum()); a = a / a.sum()
    print(f"T={T:4d} forward={ll:.10f} ref_lineal={ref:.10f} diff={ll-ref:.2e} scaled={scaled_forward_log_likelihood(pi,A,B,x)-ref:.2e}")
x = np.array([3, 7])
brute = np.log(sum(pi[i] * B[i, x[0]] * A[i, j] * B[j, x[1]] for i in range(252) for j in range(252)))
print("fuerza bruta T=2:", brute, "forward:", forward_log_likelihood(pi, A, B, x)[0])

# 2. marginal vs Viterbi
x = rng.integers(0, 12, 200)
ll = forward_log_likelihood(pi, A, B, x)[0]
_, vs = viterbi_decode(pi, A, B, x)
print(f"log p(x) marginal={ll:.3f}  Viterbi max_z log p(x,z)={vs:.3f}  (marginal>=viterbi: {ll>=vs})")

# 3. estados con filas de emision identicas (no identificables por emision)
rows = {}
for k in range(252):
    rows.setdefault(tuple(np.round(B[k], 12)), []).append(k)
dups = [v for v in rows.values() if len(v) > 1]
print("filas de emision distintas:", len(rows), "de 252; grupos duplicados:", len(dups),
      "ej:", [[m.state_labels[i] for i in g] for g in dups[:3]])

# 4. parametros libres: el finito no se ajusta a nada
print("FiniteChordHMM: parametros estimados de los datos = 0 (pi uniforme, A y B fijos)")

# 5. comparacion por obra sobre la partitura de ejemplo, tal como library_batch
events = extract_events(parse_score("examples/example_score.musicxml"))
obs = build_observation_sequence(events, "pitch_class")
x = obs.tokens
fin = m.fit_predict(obs)
T = x.size
counts = np.bincount(x, minlength=12)
p = counts / T
uni = float(np.sum(counts[counts > 0] * np.log(p[counts > 0])))
big = np.zeros((12, 12)) + 0
for a_, b_ in zip(x[:-1], x[1:]):
    big[a_, b_] += 1
bg = float(np.log(p[x[0]]) + sum(np.log(big[a_, b_] / big[a_].sum()) for a_, b_ in zip(x[:-1], x[1:])))
print(f"\nObra de ejemplo T={T}: log L_fin={fin.log_likelihood:.2f}  K_fin(activos Viterbi)={len(fin.active_states)}")
print(f"  uniforme 12 simbolos = {T*np.log(1/12):.2f}")
print(f"  unigrama MLE en la misma obra (11 params) = {uni:.2f}")
print(f"  bigrama MLE en la misma obra = {bg:.2f}")
hdp_params = {"n_states": 20, "n_iters": 80, "burn_in": 40, "seed": 7}
h = TruncatedHDPHMM(**hdp_params).fit(obs)
hist = np.array(h.diagnostics.log_likelihood_history)
pm = forward_log_likelihood(h.posterior_initial_mean, h.posterior_transition_mean, h.posterior_emission_mean, x)[0]
print(f"  HDP (library_batch): log_likelihood reportado={h.log_likelihood:.2f} = max historia? {np.isclose(h.log_likelihood, hist.max())}")
print(f"     media historia post-burn-in={hist[40:].mean():.2f}  log p(x|media posterior)={pm:.2f}  effective_states={h.effective_states} (tope 20)")
print(f"  Parametros HDP por obra (K=20,V=12): {19 + 20*19 + 20*11}")

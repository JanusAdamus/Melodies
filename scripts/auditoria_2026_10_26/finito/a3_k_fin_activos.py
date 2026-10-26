"""K_fin = estados distintos del camino de Viterbi (252 posibles) vs effective_states del HDP (tope 20)."""
import sys

import numpy as np

sys.path.insert(0, ".")
from src.models.finite_hmm import FiniteChordHMM
from src.models.inference import viterbi_decode

m = FiniteChordHMM()
pi, A, B = m.initial_probs, m.transition_matrix, m.emission_matrix
rows = {}
for k in range(252):
    rows.setdefault(tuple(np.round(B[k], 12)), []).append(k)
twin = {k: tuple(g) for g in rows.values() for k in g if len(g) > 1}
rng = np.random.default_rng(3)
for T in (24, 100, 300, 1000):
    x = rng.integers(0, 12, T)
    z, _ = viterbi_decode(pi, A, B, x)
    act = set(z.tolist())
    groups = {twin[k] for k in act if k in twin}
    both = sum(1 for g in groups if len(act.intersection(g)) > 1)
    # mismos datos, contando estados por fila de emision distinta
    distinct_emissions = len({tuple(np.round(B[k], 12)) for k in act})
    print(f"T={T:5d}: K_fin={len(act):3d} | filas de emision distintas entre activos={distinct_emissions:3d} | "
          f"grupos gemelos con >1 miembro activo={both}")
# melodia tonal larga: escala C mayor aleatoria
x = rng.choice([0, 2, 4, 5, 7, 9, 11], 300)
z, _ = viterbi_decode(pi, A, B, x)
print("escala C mayor aleatoria T=300: K_fin =", len(set(z.tolist())))

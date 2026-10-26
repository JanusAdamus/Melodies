"""Chequeos de Comparacion/vomm.py: normalizacion, backoff, no vistos, masa en BOS, fuga."""
import math
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")
from Comparacion.vomm import VariableOrderMarkovModel, select_vomm_by_validation

V, BOS = 13, 12
rng = np.random.default_rng(0)
train = [list(rng.choice([0, 2, 4, 5, 7, 9, 11], 50)) for _ in range(20)]
m = VariableOrderMarkovModel(max_order=3, vocabulary_size=V, bos_token_id=BOS).fit(train)

# 1. normalizacion en contextos vistos, no vistos y con BOS
for ctx in ([BOS], [BOS, 0], [BOS, 0, 2, 4], [1, 3, 6], [BOS, 1, 3, 6, 8, 10]):
    d = m.predict_distribution(ctx)
    print(f"ctx={ctx!s:22s} suma={d.sum():.15f} min={d.min():.2e} P(BOS)={d[BOS]:.4f} P(token no visto 1)={d[1]:.4f}")

# 2. referencia independiente de la interpolacion (recursion escrita a mano)
def ref(ctx):
    c0 = np.zeros(V)
    for s in train:
        for t in s: c0[t] += 1
    dist = (c0 + .5) / (c0.sum() + .5 * V)
    for k in range(1, min(3, len(ctx)) + 1):
        c = np.zeros(V)
        for s in train:
            tok = [BOS] + s
            for i in range(k, len(tok)):
                if tok[i - k:i] == ctx[-k:]: c[tok[i]] += 1
        if c.sum() == 0: continue
        w = c.sum() / (c.sum() + 1.0)
        dist = w * (c + .5) / (c.sum() + .5 * V) + (1 - w) * dist
    return dist
for ctx in ([BOS, 0, 2, 4], [5, 7, 9]):
    print("ref vs codigo max|diff|:", np.abs(ref(ctx) - m.predict_distribution(ctx)).max())

# 3. masa media asignada a BOS (simbolo que nunca es objetivo) en evaluacion
test = [SimpleNamespace(piece_id=str(i), title="", composer="", tokens=list(rng.choice([0, 2, 4, 5, 7, 9, 11], 50))) for i in range(10)]
mass = []
for p in test:
    ctx = [BOS]
    for t in p.tokens:
        mass.append(m.predict_distribution(ctx)[BOS]); ctx.append(int(t))
print(f"P(BOS) media en prueba={np.mean(mass):.4f}, max={np.max(mass):.4f}; costo en nats/token = {-math.log(1-np.mean(mass)):.4f}")

# 4. fuga: el ajuste solo ve train; mutar prueba/validacion no cambia los conteos
val = [SimpleNamespace(piece_id="v", title="", composer="", tokens=list(rng.choice([0, 2, 4], 40)))]
sel = select_vomm_by_validation(train, val, candidate_orders=(1, 2, 3), vocabulary_size=V, bos_token_id=BOS)
m2 = VariableOrderMarkovModel(max_order=sel.max_order, vocabulary_size=V, bos_token_id=BOS).fit(train)
same = all(np.array_equal(sel._next_token_counts[o][c], m2._next_token_counts[o][c]) for o in m2._next_token_counts for c in m2._next_token_counts[o])
print("conteos del seleccionado == conteos solo-train:", same, "| orden elegido:", sel.selected_order)

# 5. borde: pieza vacia y slices
e = m.evaluate([SimpleNamespace(piece_id="e", title="", composer="", tokens=[])], 128)
print("pieza vacia n_tokens:", e["summary"]["n_tokens"], "nll:", e["summary"]["test_nll_per_token"])

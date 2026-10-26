"""Reproducciones sobre Comparacion.statistics / decision con datos sinteticos."""
import sys
sys.path.insert(0, sys.argv[1])
from Comparacion.statistics import pairwise_model_comparisons
from Comparacion.decision import pareto_front

# 1) Dos semillas de particion: la misma obra W aparece en ambas; el bloque principal la promedia.
rows = []
for split_seed, delta in ((7, -1.0), (17, +1.0)):
    for w in ("W", f"solo_{split_seed}_a", f"solo_{split_seed}_b"):
        rows.append(dict(model="A", frac=1.0, canonical_work_id=w, split_seed=split_seed, test_nll=2.0 + (delta if w == "W" else 0.1)))
        rows.append(dict(model="B", frac=1.0, canonical_work_id=w, split_seed=split_seed, test_nll=2.0))
out = pairwise_model_comparisons(rows, bootstrap_samples=200, seed=0)
c = out["comparisons"][0]
print("pooling declarado:", out["pooling"])
print("bloque 'comparisons' n_pairs =", c["n_pairs"], "obras:", c["canonical_work_ids"])
print("  W entra una vez con diferencia promediada entre particiones (−1 y +1 -> 0); n_pairs suma obras de ambas particiones")

# 2) Pareto con 3 ejes y 4 modelos: cada mejor-en-un-eje queda en la frontera aunque pierda por mucho en NLL
cands = [dict(model="transformer", nll=1.76, fit=169, ev=5.0), dict(model="finite_hmm", nll=2.00, fit=628, ev=1.0),
         dict(model="hdp_hmm", nll=2.08, fit=578, ev=0.9), dict(model="vomm", nll=2.08, fit=41, ev=3.0)]
print("frontera:", [r["model"] for r in pareto_front(cands, minimize=("nll", "fit", "ev"), maximize=())])

"""Uso: python s9_potencia_permutacion.py SALIDA_S8.txt [...]

Bajo independencia (permutación de K medio), ¿cuántas veces pasa la regla vieja por bloques y la nueva agrupada?"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HDP = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s8", HDP / "s8_criterio_b_bloques.py")
s8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s8)
v3 = s8.v3


def old_block_status(r, n):
    low, high = v3.fisher_interval(r, n)
    if abs(r) < 0.3 and v3.equivalence_pvalue(r, n) < 0.05:
        return "cumple"
    if low > 0.3 or high < -0.3:
        return "incumple"
    return "indeciso"


def old_rule(blocks):
    statuses = [old_block_status(r, n) for r, n in blocks]
    if all(s == "cumple" for s in statuses):
        return "met_robustly"
    if "incumple" in statuses:
        return "not_met"
    return "inconclusive"


for name in sys.argv[1:]:
    sections = v3.parse_rows((HDP / name).read_text(encoding="utf8"))
    x = np.concatenate([np.asarray(s[1], float) for s in sections])
    y = np.concatenate([np.asarray(s[2], float) for s in sections])
    rng = np.random.default_rng(0)
    old, new = {}, {}
    for _ in range(5000):
        yp = rng.permutation(y)
        blocks = [(float(stats.pearsonr(x[i:i + 50], yp[i:i + 50]).statistic), 50) for i in range(0, 200, 50)]
        old[old_rule(blocks)] = old.get(old_rule(blocks), 0) + 1
        d = s8.criterion_b_decision(float(stats.pearsonr(x, yp).statistic), 200)
        new[d] = new.get(d, 0) + 1
    print(f"{name}: 5000 permutaciones (semilla 0); regla vieja por bloques {old}; regla agrupada {new}")

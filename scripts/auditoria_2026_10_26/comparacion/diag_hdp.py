"""build_chain_diagnostics: (a) incluye el burn-in en la ventana temprana; (b) concatena celdas de la misma model_seed."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from Comparacion.hdp_diagnostics import build_chain_diagnostics
rng = np.random.default_rng(0)
def chain(burn=60, n=120):  # sube durante el burn-in y luego es estacionaria (ruido iid)
    ll = [-70000 + 4000 * min(i, burn) / burn + rng.normal(0, 20) for i in range(n)]
    return [dict(iteration=i + 1, log_likelihood=v, active_states=11) for i, v in enumerate(ll)]
c = chain()
d = build_chain_diagnostics({1: c}, min_iterations=60)
print("(a) cadena estacionaria tras burn-in:", d["verdict"], "| drift", round(d["chains"]["1"]["log_likelihood_drift"]))
post = build_chain_diagnostics({1: c[60:]}, min_iterations=60)
print("    misma cadena sin burn-in:", post["verdict"], "| lag1", round(post["chains"]["1"]["lag_one_autocorrelation"], 3))
three = [dict(r, data_seed=s) for s in (1, 2, 3) for r in c]   # lo que hace runner.py con 3 data_seeds en frac=1.0
d3 = build_chain_diagnostics({1: three}, min_iterations=60)
print("(b) 3 celdas concatenadas como 'cadena 1': n_iterations =", d3["chains"]["1"]["n_iterations"], "| lag1 =", round(d3["chains"]["1"]["lag_one_autocorrelation"], 3))

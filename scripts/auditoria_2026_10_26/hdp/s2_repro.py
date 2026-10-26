"""Reproduce valores por obra con el codigo actual y parametros del CLI (K=20, 80 it, burn 40, seed 7)."""
from common import *
from src.models.hdp_hmm import TruncatedHDPHMM
from src.models.finite_hmm import FiniteChordHMM
R = rows()
idx = [0, 1, 2, 50, 100, 166]
for i in idx:
    r = R[i]; obs = load(r["path"])
    fin = FiniteChordHMM().fit_predict(obs)
    m = TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=7)
    res = m.fit(obs); d = res.diagnostics
    print(f"[{i}] T={obs.size} fin stored={float(r['finite_log_likelihood']):.3f} now={fin.log_likelihood:.3f} | hdp stored={float(r['hdp_log_likelihood']):.3f} now={res.log_likelihood:.3f} K stored={r['hdp_effective_states']} now={res.effective_states} best_it={d.best_iteration} mode={d.beta_update_mode}")

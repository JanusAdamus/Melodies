"""hdp_diagnostics con trazas acumuladas como en runner.py:1436 (varias celdas frac/data_seed bajo la misma model_seed)."""
import pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from common import REPO
sys.path.insert(0, REPO)
from Comparacion.hdp_diagnostics import build_chain_diagnostics, _chain_diagnostics
from src.data.observations import ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM
data = pickle.load(open("obs_cache.pkl", "rb"))
pieces = [data[i] for i in range(0, 167, 4)]
def trace(n_pieces, seed):
    seqs = [ObservationSequence("pitch_class", p["tokens"], p["vocab"], [""]*len(p["tokens"]), [], {}) for p in pieces[:n_pieces]]
    return TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=seed).fit_sequences(seqs).diagnostics.trace_rows()
for seed in (7, 8):
    small, big = trace(10, seed), trace(20, seed)
    for name, rows in (("solo frac=0.5", small), ("solo frac=1.0", big)):
        c = _chain_diagnostics(rows); print(seed, name, "stable:", c["window_stability"], "drift", round(c["log_likelihood_drift"], 1))
    c = _chain_diagnostics(small + big)
    print(seed, "concatenado como runner", "stable:", c["window_stability"], "early", round(c["early_window_mean_log_likelihood"], 1), "late", round(c["late_window_mean_log_likelihood"], 1))
t7, t8 = trace(20, 7), trace(20, 8)
d = build_chain_diagnostics({7: t7, 8: t8}, min_iterations=40)
print("2 cadenas (seed 7 vs 8, mismos datos):", d["verdict"], d["cross_chain"])

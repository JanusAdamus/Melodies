"""Label switching en state_samples y variabilidad entre semillas."""
import pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import REPO
from src.data.observations import ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM
from src.models.inference import scaled_forward_log_likelihood as LL
data = pickle.load(open("obs_cache.pkl", "rb")); R = pickle.load(open("s3_rows.pkl", "rb"))
worst = sorted(R, key=lambda r: r["ll_pm"] - r["best"])[:5]
def fit(t, V, seed=7):
    return TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=seed).fit(ObservationSequence("pitch_class", t, V, [""]*len(t), [], {}))
for r in worst:
    o = data[r["i"]]; res = fit(o["tokens"], o["vocab"]); S = res.diagnostics.state_samples; ref = S[-1]
    nonid = 0; agree_raw = []; agree_al = []
    for s in S:
        C = np.zeros((20, 20)); np.add.at(C, (s, ref), 1)
        ri, ci = linear_sum_assignment(-C)
        perm = dict(zip(ri, ci)); used = np.unique(s)
        nonid += any(perm[k] != k for k in used)
        agree_raw.append((s == ref).mean()); agree_al.append((np.vectorize(perm.get)(s) == ref).mean())
    print(f"obra {r['i']} T={r['T']} ll(medias)-ll(best)={r['ll_pm']-r['best']:.1f}  muestras con permutacion no-identidad: {nonid}/{len(S)}  acuerdo crudo medio={np.mean(agree_raw):.3f} alineado={np.mean(agree_al):.3f}")
print("--- semillas 7..11, 15 obras espaciadas ---")
sdK, sdL, rngK = [], [], []
for i in range(0, 167, 12):
    o = data[i]; Ks, Ls = [], []
    for sd in range(7, 12):
        res = fit(o["tokens"], o["vocab"], sd); Ks.append(res.effective_states); Ls.append(res.log_likelihood)
    sdK.append(np.std(Ks, ddof=1)); sdL.append(np.std(Ls, ddof=1)); rngK.append(max(Ks)-min(Ks))
    print(i, "T", len(o["tokens"]), "K", Ks, "logL", np.round(Ls, 1))
print("sd K media", np.mean(sdK), " rango K medio", np.mean(rngK), " sd logL media", np.mean(sdL))

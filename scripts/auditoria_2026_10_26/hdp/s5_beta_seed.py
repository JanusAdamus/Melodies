"""K_hdp vs beta inicial de la semilla (beta nunca se actualiza)."""
import pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from common import REPO
from src.models.utils import sample_truncated_stick_breaking
from src.data.observations import ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM
data = pickle.load(open("obs_cache.pkl", "rb"))
idx = list(range(0, 167, 12))
seeds = list(range(7, 27))
nb, Km = [], []
for sd in seeds:
    _, b = sample_truncated_stick_breaking(2.0, 20, np.random.default_rng(sd))
    n_big = int((20 * 8.0 * b > 1.0).sum())  # componentes con alpha*beta_k > 1 (no colapsan en la Dirichlet)
    Ks = [TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=sd).fit(ObservationSequence("pitch_class", data[i]["tokens"], data[i]["vocab"], [""]*len(data[i]["tokens"]), [], {})).effective_states for i in idx]
    nb.append((b > 0.01).sum()); Km.append(np.mean(Ks))
    print(f"seed {sd}: #beta_k>0.01={(b>0.01).sum():2d}  #alpha*beta_k>0.05={(8*b>0.05).sum():2d}  K_hdp medio (15 obras)={np.mean(Ks):.2f}  sd entre obras={np.std(Ks):.2f}")
print("corr(#beta_k>0.01, K medio) =", np.corrcoef(nb, Km)[0, 1])
print("varianza de K medio entre semillas", np.var(Km, ddof=1), " rango", min(Km), max(Km))

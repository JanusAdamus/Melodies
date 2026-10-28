"""s5 con la API corregida: ¿K_hdp sigue al primer β de la semilla?

Uso: python s5_beta_seed_v2.py [ruta/obs_cache.pkl] [semilla_inicial=7]   (el caché lo genera s0_cache.py)
Corre contra el código de este checkout, no contra common.REPO. Mismas obras
(cada 12.ª de 167), semillas 7..26, K=20, 80 iteraciones, burn-in 40.
El primer β es diagnostics.initial_beta (en el código viejo era el stick-breaking
del primer sorteo); K es effective_states (media posterior, ocupación >= 2 %).
"""
import pickle, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.observations import ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM

data = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "obs_cache.pkl", "rb"))
idx = list(range(0, 167, 12))
first_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
seeds = list(range(first_seed, first_seed + 20))
nb, Km, Kraw = [], [], []
for sd in seeds:
    fits = [TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=sd).fit(ObservationSequence("pitch_class", data[i]["tokens"], data[i]["vocab"], [""]*len(data[i]["tokens"]), [], {})) for i in idx]
    b = fits[0].diagnostics.initial_beta
    Ks = [f.effective_states for f in fits]
    nb.append(int((b > 0.01).sum())); Km.append(np.mean(Ks)); Kraw.append(np.mean([f.diagnostics.active_state_history[-1] for f in fits]))
    print(f"seed {sd}: #beta_k>0.01={nb[-1]:2d}  K_hdp medio (15 obras)={Km[-1]:.2f}  sd entre obras={np.std(Ks):.2f}  modo={fits[0].diagnostics.beta_update_mode}", flush=True)
print("corr(#beta_k>0.01, K medio) =", np.corrcoef(nb, Km)[0, 1], " spearman =", spearmanr(nb, Km).statistic)
slope = np.polyfit(nb, Km, 1)[0]
print("pendiente de K medio por componente de beta > 0.01:", slope)
print("varianza de K medio entre semillas", np.var(Km, ddof=1), " rango", min(Km), max(Km))
print("estados con alguna visita (última iteración), medio:", np.mean(Kraw))

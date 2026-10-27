"""Criterio (b) con >= 50 semillas: ¿el K del HDP sigue al primer β de la semilla?

Uso:
    python s5_beta_seed_v3.py CACHE.jsonl [--n-seeds 50] [--first-seed 7] [--every 200]
                              [--n-works 15] [--single-work INDICE]

CACHE.jsonl es un caché de corpus del runner (una obra por línea con `tokens` en
clases de altura 0..11), p. ej. artifacts/corpus_cache_3000.jsonl. Se toman las obras
0, every, 2*every, ... (n_works de ellas) o, con --single-work, una sola obra.
Mismo modelo que s5_beta_seed_v2.py: K=20, 80 iteraciones, burn-in 40.
Reporta Pearson, Spearman, pendiente por mínimos cuadrados con IC 95 % (t) e IC 95 %
bootstrap por semilla (10 000 remuestreos, semilla 0).
Corre contra el código de este checkout.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.observations import ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM

VOCABULARY = [str(index) for index in range(12)]


def load_works(path, every, n_works, single_work):
    rows = [json.loads(line) for line in open(path, encoding="utf8")]
    rows = [row for row in rows if "tokens" in row]
    indices = [single_work] if single_work is not None else list(range(0, len(rows), every))[:n_works]
    return indices, [np.asarray(rows[index]["tokens"], dtype=int) for index in indices]


def observation(tokens):
    return ObservationSequence("pitch_class", tokens, VOCABULARY, [""] * len(tokens), [], {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=7)
    parser.add_argument("--every", type=int, default=200)
    parser.add_argument("--n-works", type=int, default=15)
    parser.add_argument("--single-work", type=int, default=None)
    args = parser.parse_args()

    indices, works = load_works(args.cache, args.every, args.n_works, args.single_work)
    print(f"obras (índice en el caché): {indices}; eventos: {[int(w.size) for w in works]}", flush=True)
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    n_large, mean_k = [], []
    for seed in seeds:
        fits = [TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=seed).fit(observation(w)) for w in works]
        n_large.append(int((fits[0].diagnostics.initial_beta > 0.01).sum()))
        mean_k.append(float(np.mean([fit.effective_states for fit in fits])))
        print(f"seed {seed}: #beta_k>0.01={n_large[-1]:2d}  K medio={mean_k[-1]:.3f}", flush=True)

    x, y = np.asarray(n_large, float), np.asarray(mean_k, float)
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    fit = stats.linregress(x, y)
    t_crit = stats.t.ppf(0.975, x.size - 2)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(10_000):
        sample = rng.integers(0, x.size, x.size)
        if np.ptp(x[sample]) > 0:
            boot.append(np.polyfit(x[sample], y[sample], 1)[0])
    print(f"n semillas = {x.size}")
    print(f"Pearson r = {pearson.statistic:.3f} (p = {pearson.pvalue:.3g})")
    print(f"Spearman rho = {spearman.statistic:.3f} (p = {spearman.pvalue:.3g})")
    print(f"pendiente = {fit.slope:.4f} estados por componente; IC 95 % (t) = "
          f"[{fit.slope - t_crit * fit.stderr:.4f}, {fit.slope + t_crit * fit.stderr:.4f}]")
    print(f"IC 95 % bootstrap de la pendiente = [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]")
    print(f"rango de #beta_k>0.01 = {int(x.min())}..{int(x.max())}; efecto de la pendiente sobre ese rango = "
          f"{fit.slope * np.ptp(x):.3f} estados")
    print(f"K medio: media {y.mean():.3f}, d.e. entre semillas {y.std(ddof=1):.3f}, rango {y.min():.3f}..{y.max():.3f}")


if __name__ == "__main__":
    main()

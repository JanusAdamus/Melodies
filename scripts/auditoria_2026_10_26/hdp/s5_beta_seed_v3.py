"""Criterio (b) con >= 50 semillas: ¿el K del HDP sigue al primer β de la semilla?

Uso:
    python s5_beta_seed_v3.py CACHE.jsonl [--n-seeds 50] [--first-seed 7] [--every 200]
                              [--n-works 15] [--single-work INDICE]
    python s5_beta_seed_v3.py --resumen SALIDA.txt

CACHE.jsonl es un caché de corpus del runner (una obra por línea con `tokens` en
clases de altura 0..11), p. ej. artifacts/corpus_cache_3000.jsonl. Se toman las obras
0, every, 2*every, ... (n_works de ellas) o, con --single-work, una sola obra.
Mismo modelo que s5_beta_seed_v2.py: K=20, 80 iteraciones, burn-in 40.
Reporta Pearson, Spearman, IC 95 % de Fisher para r, prueba de equivalencia TOST para
|ρ| < 0.3 (z de Fisher), pendiente por mínimos cuadrados con IC 95 % (t) e IC 95 % bootstrap
por semilla (10 000 remuestreos, semilla 0).  Con --resumen recalcula todo desde las filas
«seed N: ...» de una salida guardada, sin volver a ajustar.
Corre contra el código de este checkout.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.observations import ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM

VOCABULARY = [str(index) for index in range(12)]
EQUIVALENCE_BOUND = 0.3
ROW = re.compile(r"^seed (\d+): #beta_k>0\.01=\s*(\d+)\s+K medio=([0-9.]+)\s*$")


def fisher_interval(r, n, level=0.95):
    """IC de r por la transformación z de Fisher (error estándar 1 / sqrt(n - 3))."""

    if n <= 3:
        return float("nan"), float("nan")
    half = stats.norm.ppf(0.5 + level / 2) / np.sqrt(n - 3)
    return float(np.tanh(np.arctanh(r) - half)), float(np.tanh(np.arctanh(r) + half))


def equivalence_pvalue(r, n, bound=EQUIVALENCE_BOUND):
    """TOST de H0: |ρ| >= bound contra H1: |ρ| < bound, con la z de Fisher."""

    if n <= 3:
        return float("nan")
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    p_lower = stats.norm.sf((z - np.arctanh(-bound)) / se)
    p_upper = stats.norm.cdf((z - np.arctanh(bound)) / se)
    return float(max(p_lower, p_upper))


def verdict(p_value, equivalence_p, alpha=0.05):
    """Un r puntual pequeño no basta: |r| < 0.3 sólo se afirma si la equivalencia es significativa."""

    if p_value < alpha:
        return "dependencia"
    if equivalence_p < alpha:
        return f"|r| < {EQUIVALENCE_BOUND} demostrado (equivalencia)"
    return f"sin evidencia de dependencia ni de |r| < {EQUIVALENCE_BOUND}"


def parse_rows(text):
    """Secciones (semillas, #beta_k>0.01, K medio) de una salida guardada; cada «obras (» abre una."""

    sections = []
    for line in text.splitlines():
        if line.startswith("obras ("):
            sections.append(([], [], []))
        match = ROW.match(line.strip())
        if match and sections:
            for values, value in zip(sections[-1], (int(match[1]), int(match[2]), float(match[3]))):
                values.append(value)
    return [section for section in sections if section[0]]


def summary(x, y):
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    fit = stats.linregress(x, y)
    t_crit = stats.t.ppf(0.975, x.size - 2)
    low, high = fisher_interval(pearson.statistic, x.size)
    equivalence_p = equivalence_pvalue(pearson.statistic, x.size)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(10_000):
        sample = rng.integers(0, x.size, x.size)
        if np.ptp(x[sample]) > 0:
            boot.append(np.polyfit(x[sample], y[sample], 1)[0])
    slope_low, slope_high = fit.slope - t_crit * fit.stderr, fit.slope + t_crit * fit.stderr
    return [
        f"n semillas = {x.size}",
        f"Pearson r = {pearson.statistic:.3f} (p = {pearson.pvalue:.3g})",
        f"IC 95 % de Fisher para r = [{low:.3f}, {high:.3f}]",
        f"TOST |r| < {EQUIVALENCE_BOUND}: p = {equivalence_p:.3f}",
        f"Spearman rho = {spearman.statistic:.3f} (p = {spearman.pvalue:.3g})",
        f"pendiente = {fit.slope:.4f} estados por componente; IC 95 % (t) = [{slope_low:.4f}, {slope_high:.4f}]",
        f"IC 95 % bootstrap de la pendiente = [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]",
        f"rango de #beta_k>0.01 = {int(x.min())}..{int(x.max())}; efecto de la pendiente sobre ese rango = "
        f"{fit.slope * np.ptp(x):.3f} estados, IC 95 % (t) = [{slope_low * np.ptp(x):.2f}, {slope_high * np.ptp(x):.2f}]",
        f"K medio: media {y.mean():.3f}, d.e. entre semillas {y.std(ddof=1):.3f}, rango {y.min():.3f}..{y.max():.3f}",
        f"veredicto: {verdict(pearson.pvalue, equivalence_p)}",
    ]


def load_works(path, every, n_works, single_work):
    with open(path, encoding="utf8") as handle:
        rows = [json.loads(line) for line in handle]
    rows = [row for row in rows if "tokens" in row]
    indices = [single_work] if single_work is not None else list(range(0, len(rows), every))[:n_works]
    return indices, [np.asarray(rows[index]["tokens"], dtype=int) for index in indices]


def observation(tokens):
    return ObservationSequence("pitch_class", tokens, VOCABULARY, [""] * len(tokens), [], {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", nargs="?")
    parser.add_argument("--resumen", default=None, help="salida guardada de la que recalcular el resumen")
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=7)
    parser.add_argument("--every", type=int, default=200)
    parser.add_argument("--n-works", type=int, default=15)
    parser.add_argument("--single-work", type=int, default=None)
    args = parser.parse_args()

    if args.resumen is not None:
        for seeds, n_large, mean_k in parse_rows(Path(args.resumen).read_text(encoding="utf8")):
            print(f"== semillas {seeds[0]}..{seeds[-1]} ==")
            print("\n".join(summary(np.asarray(n_large, float), np.asarray(mean_k, float))))
        return
    if args.cache is None:
        parser.error("falta CACHE.jsonl (o --resumen SALIDA.txt)")

    indices, works = load_works(args.cache, args.every, args.n_works, args.single_work)
    print(f"obras (índice en el caché): {indices}; eventos: {[int(w.size) for w in works]}", flush=True)
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    n_large, mean_k = [], []
    for seed in seeds:
        fits = [TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=seed).fit(observation(w)) for w in works]
        n_large.append(int((fits[0].diagnostics.initial_beta > 0.01).sum()))
        mean_k.append(float(np.mean([fit.effective_states for fit in fits])))
        print(f"seed {seed}: #beta_k>0.01={n_large[-1]:2d}  K medio={mean_k[-1]:.3f}", flush=True)
    print("\n".join(summary(np.asarray(n_large, float), np.asarray(mean_k, float))))


if __name__ == "__main__":
    main()

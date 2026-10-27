"""Criterio (b) en bloques DISJUNTOS de semillas sobre una misma obra, con la cadena del preregistro.

Uso:
    python s8_criterio_b_bloques.py simulada [--n-blocks 4] [--block-size 50] [--first-seed 0]
    python s8_criterio_b_bloques.py CACHE.jsonl --single-work 200 [...]
    opciones comunes: [--n-iters 400] [--burn-in 200] [--n-workers N]

«simulada» es la obra de las pruebas: three_state_tokens(300, seed=5) de tests/test_hdp_sampler.py
(HMM de 3 estados, 6 símbolos).  Con un caché del runner se toma la obra --single-work (clases de
altura 0..11).  Truncamiento K=20; por omisión 400 iteraciones y burn-in 200
(docs/preregistro-2026-10-26.md).  Para cada semilla: #β_k > 0.01 del primer sorteo y K ocupados
(effective_states, ocupación ≥ 2 %).  Por bloque se reporta lo mismo que s5_beta_seed_v3.summary
(r, IC de Fisher, TOST, Spearman, pendiente con IC t y bootstrap) y al final la decisión.

Regla de decisión (fijada antes de correr; el umbral 0.3 es el de PROMPT_CLAUDE.md, tarea 1.1 (b)):
- un bloque CUMPLE si |r| < 0.3 y la equivalencia TOST de |ρ| < 0.3 es significativa (p < 0.05);
- un bloque INCUMPLE CLARAMENTE si su IC 95 % de Fisher queda entero fuera de (-0.3, 0.3);
- met_robustly si todos los bloques cumplen; not_met si alguno incumple claramente;
  inconclusive en otro caso (los IC no permiten decidir).
Se exigen al menos 3 bloques de al menos 50 semillas.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.observations import ObservationSequence  # noqa: E402
from src.models.hdp_hmm import TruncatedHDPHMM  # noqa: E402

_spec = importlib.util.spec_from_file_location("s5_beta_seed_v3", Path(__file__).with_name("s5_beta_seed_v3.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)

BOUND = v3.EQUIVALENCE_BOUND
MIN_BLOCKS, MIN_BLOCK_SIZE = 3, 50


def block_status(r, n, alpha=0.05):
    """«cumple», «incumple» (IC 95 % entero fuera de ±0.3) o «indeciso»."""

    low, high = v3.fisher_interval(r, n)
    if abs(r) < BOUND and v3.equivalence_pvalue(r, n) < alpha:
        return "cumple"
    if low > BOUND or high < -BOUND:
        return "incumple"
    return "indeciso"


def criterion_b_decision(blocks):
    """Decisión sobre [(r, n), ...] de bloques disjuntos (ver la regla en el docstring del módulo)."""

    if len(blocks) < MIN_BLOCKS or any(n < MIN_BLOCK_SIZE for _, n in blocks):
        raise ValueError(f"se necesitan >= {MIN_BLOCKS} bloques de >= {MIN_BLOCK_SIZE} semillas")
    statuses = [block_status(r, n) for r, n in blocks]
    if all(status == "cumple" for status in statuses):
        return "met_robustly"
    if "incumple" in statuses:
        return "not_met"
    return "inconclusive"


def load_work(source, single_work):
    """(tokens, vocabulario, descripción) de la obra simulada o de una obra del caché."""

    if source == "simulada":
        from tests.test_hdp_sampler import VOCABULARY, three_state_tokens

        return three_state_tokens(300, seed=5), list(VOCABULARY), "simulada: three_state_tokens(300, seed=5)"
    with open(source, encoding="utf8") as handle:
        rows = [row for row in map(json.loads, handle) if "tokens" in row]
    return np.asarray(rows[single_work]["tokens"], dtype=int), [str(i) for i in range(12)], f"{source} obra {single_work}"


def fit_point(job):
    source, single_work, n_iters, burn_in, seed = job
    tokens, vocabulary, _ = load_work(source, single_work)
    observations = ObservationSequence("symbolic", tokens, vocabulary, [""] * len(tokens), [], {})
    result = TruncatedHDPHMM(n_states=20, n_iters=n_iters, burn_in=burn_in, seed=seed).fit(observations)
    return seed, int((result.diagnostics.initial_beta > 0.01).sum()), float(result.effective_states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="«simulada» o un caché .jsonl del runner")
    parser.add_argument("--single-work", type=int, default=200)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--n-iters", type=int, default=v3.PREREGISTERED_CHAIN[0])
    parser.add_argument("--burn-in", type=int, default=v3.PREREGISTERED_CHAIN[1])
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    tokens, _, description = load_work(args.source, args.single_work)
    print("\n".join(v3.chain_lines(20, args.n_iters, args.burn_in)))
    print(f"obra: {description}; eventos: {tokens.size}")
    seeds = list(range(args.first_seed, args.first_seed + args.n_blocks * args.block_size))
    jobs = [(args.source, args.single_work, args.n_iters, args.burn_in, seed) for seed in seeds]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        points = sorted(pool.map(fit_point, jobs))
    elapsed = time.perf_counter() - start
    print(f"{len(jobs)} ajustes en {elapsed:.1f} s con {args.n_workers} procesos")

    blocks = []
    for index in range(args.n_blocks):
        block = points[index * args.block_size:(index + 1) * args.block_size]
        x = np.asarray([row[1] for row in block], float)
        y = np.asarray([row[2] for row in block], float)
        print(f"obras (bloque {index}): semillas {block[0][0]}..{block[-1][0]}")
        for seed, n_large, occupied in block:
            print(f"seed {seed}: #beta_k>0.01={n_large:2d}  K medio={occupied:.3f}")
        print("\n".join(v3.summary(x, y)))
        r = float(stats.pearsonr(x, y).statistic)
        print(f"estado del bloque: {block_status(r, x.size)}")
        blocks.append((r, x.size))
    print("== resumen por bloque ==")
    for index, (r, n) in enumerate(blocks):
        low, high = v3.fisher_interval(r, n)
        print(f"bloque {index}: r = {r:.3f}, IC 95 % [{low:.3f}, {high:.3f}], "
              f"TOST p = {v3.equivalence_pvalue(r, n):.3f}, {block_status(r, n)}")
    print(f"decisión del criterio (b): {criterion_b_decision(blocks)}")


if __name__ == "__main__":
    main()

"""Criterio (b) con todas las semillas juntas sobre una misma obra, con la cadena del preregistro.

Uso:
    python s8_criterio_b_bloques.py simulada [--n-blocks 4] [--block-size 50] [--first-seed 0]
    python s8_criterio_b_bloques.py CACHE.jsonl --single-work 200 [...]
    python s8_criterio_b_bloques.py --resumen SALIDA.txt
    opciones comunes: [--n-iters 400] [--burn-in 200] [--n-workers N] [--congelar-beta]

--congelar-beta es el control negativo: β queda fijo en el primer sorteo (el defecto H1), así que la
regla debe dejar de dar «met».

«simulada» es la obra de las pruebas: three_state_tokens(300, seed=5) de tests/test_hdp_sampler.py
(HMM de 3 estados, 6 símbolos).  Con un caché del runner se toma la obra --single-work (clases de
altura 0..11).  Truncamiento K=20; por omisión 400 iteraciones y burn-in 200
(docs/preregistro-2026-10-26.md, que sólo fija la cadena).  Para cada semilla: #β_k > 0.01 del
primer sorteo y K ocupados (effective_states, ocupación ≥ 2 %).  Con --resumen recalcula todo
desde las filas «seed N: ...» de una salida guardada, sin volver a ajustar.

Decisión (criterio escrito en PROMPT_CLAUDE.md, tarea 1.1 (b): |r| < 0.3), con TODAS las semillas
juntas:
- met si la estimación puntual cumple |r| < 0.3 y la equivalencia TOST de |ρ| < 0.3 (z de Fisher)
  es significativa (p < 0.05), que es la evidencia con potencia;
- not_met si |r| ≥ 0.3;
- inconclusive si |r| < 0.3 pero la TOST no es significativa.
Se exigen al menos 10 semillas (las que pide PROMPT_CLAUDE.md).

Historia, para que no se lea mal: la versión anterior de este script (commits a3044e2 y a183bd2)
decidía con una regla por bloques («cada uno de 4 bloques de 50 semillas debe pasar TOST») que se
escribió DESPUÉS de conocer las cifras, no antes de correr como decía su docstring.  Una revisión
adversarial mostró que esa regla no tiene potencia: con independencia real (K medio permutado) casi
nunca la pasa.  s9_potencia_permutacion.py lo reproduce sobre las filas guardadas (5000
permutaciones, semilla 0; s9_potencia_permutacion_salida_20261026.txt): la regla por bloques da
met_robustly en 104/5000 (simulada) y 109/5000 (PDMX obra 200), ~2 %; la regla agrupada de abajo
da met en 4980/5000 y 4969/5000.  La regla por bloques se descartó y manda el criterio escrito.
Los bloques se siguen imprimiendo, pero sólo como descriptivo de estabilidad entre semillas.
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
MIN_SEEDS = 10


def criterion_b_decision(r, n, alpha=0.05):
    """met, not_met o inconclusive para el r de todas las semillas juntas (regla en el docstring)."""

    if n < MIN_SEEDS:
        raise ValueError(f"se necesitan >= {MIN_SEEDS} semillas")
    if abs(r) >= BOUND:
        return "not_met"
    if v3.equivalence_pvalue(r, n) < alpha:
        return "met"
    return "inconclusive"


def pooled_report(sections):
    """Líneas del resumen agrupado y la decisión, a partir de [(semillas, #β>0.01, K medio), ...]."""

    seeds = [seed for section in sections for seed in section[0]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("hay semillas repetidas entre bloques")
    lines = ["== descriptivo por bloque (estabilidad; no decide) =="]
    for block_seeds, n_large, occupied in sections:
        if len(block_seeds) > 3 and np.ptp(n_large) > 0:
            r = float(stats.pearsonr(n_large, occupied).statistic)
            low, high = v3.fisher_interval(r, len(block_seeds))
            lines.append(f"semillas {block_seeds[0]}..{block_seeds[-1]}: r = {r:.3f}, IC 95 % [{low:.3f}, {high:.3f}], "
                         f"TOST p = {v3.equivalence_pvalue(r, len(block_seeds)):.3f}")
    x = np.asarray([value for section in sections for value in section[1]], float)
    y = np.asarray([value for section in sections for value in section[2]], float)
    lines.append(f"== todas las semillas juntas: semillas {min(seeds)}..{max(seeds)} ==")
    lines.extend(v3.summary(x, y))
    r = float(stats.pearsonr(x, y).statistic)
    lines.append(f"TOST |r| < {BOUND} con todas las semillas (más cifras): p = {v3.equivalence_pvalue(r, x.size):.2g}")
    lines.append(f"decisión del criterio (b): {criterion_b_decision(r, x.size)}")
    return lines


def load_work(source, single_work):
    """(tokens, vocabulario, descripción) de la obra simulada o de una obra del caché."""

    if source == "simulada":
        from tests.test_hdp_sampler import VOCABULARY, three_state_tokens

        return three_state_tokens(300, seed=5), list(VOCABULARY), "simulada: three_state_tokens(300, seed=5)"
    with open(source, encoding="utf8") as handle:
        rows = [row for row in map(json.loads, handle) if "tokens" in row]
    return np.asarray(rows[single_work]["tokens"], dtype=int), [str(i) for i in range(12)], f"{source} obra {single_work}"


def fit_point(job):
    source, single_work, n_iters, burn_in, seed, freeze_beta = job
    tokens, vocabulary, _ = load_work(source, single_work)
    observations = ObservationSequence("symbolic", tokens, vocabulary, [""] * len(tokens), [], {})
    model_class = TruncatedHDPHMM
    if freeze_beta:
        from tests.test_hdp_sampler import _FrozenBetaHDPHMM as model_class
    result = model_class(n_states=20, n_iters=n_iters, burn_in=burn_in, seed=seed).fit(observations)
    return seed, int((result.diagnostics.initial_beta > 0.01).sum()), float(result.effective_states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", help="«simulada» o un caché .jsonl del runner")
    parser.add_argument("--resumen", default=None, help="salida guardada de la que recalcular el resumen")
    parser.add_argument("--single-work", type=int, default=200)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--n-iters", type=int, default=v3.PREREGISTERED_CHAIN[0])
    parser.add_argument("--burn-in", type=int, default=v3.PREREGISTERED_CHAIN[1])
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--congelar-beta", action="store_true",
                        help="control negativo: β fijo en el primer sorteo (defecto H1)")
    args = parser.parse_args()

    if args.resumen is not None:
        text = Path(args.resumen).read_text(encoding="utf8")
        chain = v3.CHAIN.search(text)
        if chain is None:
            print("AVISO: la salida no declara la cadena; no evalúa el criterio (b)")
        else:
            print("\n".join(v3.chain_lines(*(int(value) for value in chain.groups()))))
        print(f"recalculado desde las filas de {args.resumen}")
        print("\n".join(pooled_report(v3.parse_rows(text))))
        return
    if args.source is None:
        parser.error("falta «simulada», CACHE.jsonl o --resumen SALIDA.txt")

    tokens, _, description = load_work(args.source, args.single_work)
    print("\n".join(v3.chain_lines(20, args.n_iters, args.burn_in)))
    print(f"obra: {description}; eventos: {tokens.size}")
    if args.congelar_beta:
        print("CONTROL NEGATIVO: β congelado en el primer sorteo (no es el muestreador de la tesis)")
    seeds = list(range(args.first_seed, args.first_seed + args.n_blocks * args.block_size))
    jobs = [(args.source, args.single_work, args.n_iters, args.burn_in, seed, args.congelar_beta) for seed in seeds]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        points = sorted(pool.map(fit_point, jobs))
    print(f"{len(jobs)} ajustes en {time.perf_counter() - start:.1f} s con {args.n_workers} procesos")

    sections = []
    for index in range(args.n_blocks):
        block = points[index * args.block_size:(index + 1) * args.block_size]
        print(f"obras (bloque {index}): semillas {block[0][0]}..{block[-1][0]}")
        for seed, n_large, occupied in block:
            print(f"seed {seed}: #beta_k>0.01={n_large:2d}  K medio={occupied:.3f}")
        sections.append(tuple(list(column) for column in zip(*block)))
    print("\n".join(pooled_report(sections)))


if __name__ == "__main__":
    main()

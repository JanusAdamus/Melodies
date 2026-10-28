"""Mezcla entre cadenas del HDP-HMM corregido con la cadena del preregistro (K=20, 400/200).

Uso:
    python s10_mezcla_pdmx200.py CACHE.jsonl [--single-work 200] [--seeds 1-8] [--n-workers N]

Para cada semilla: K ocupados (ocupación >= 2 %) y log p(x | θ_t) tras el burn-in.  Imprime la
media y la d.e. dentro de cada cadena, y el R-hat dividido (src.models.hdp_hmm.split_rhat) de K y
de log L entre las semillas 1-4 (las 4 cadenas del preregistro) y entre todas.  El preregistro
(«Convergencia») exige R-hat < 1.1 de la log-verosimilitud entre las 4 cadenas.

Hallazgo H9 de docs/auditoria-2026-10-26.md: en PDMX obra 200 cada cadena se queda cerca de su
propio K y el R-hat de log L de las semillas 1-4 es ≈ 2.6.  El criterio (b) se cumple igual
(K ya no sigue al primer sorteo de β), pero K sigue dependiendo mucho de la semilla.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.models.hdp_hmm import split_rhat  # noqa: E402

N_STATES, N_ITERS, BURN_IN = 20, 400, 200


def chain(job):
    source, single_work, seed = job
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from s8_criterio_b_bloques import load_work

    from src.data.observations import ObservationSequence
    from src.models.hdp_hmm import TruncatedHDPHMM

    tokens, vocabulary, _ = load_work(source, single_work)
    observations = ObservationSequence("symbolic", tokens, vocabulary, [""] * len(tokens), [], {})
    result = TruncatedHDPHMM(n_states=N_STATES, n_iters=N_ITERS, burn_in=BURN_IN, seed=seed).fit(observations)
    log_likelihood = [sample.log_likelihood for sample in result.posterior_samples]
    return seed, np.asarray(result.diagnostics.occupied_state_samples, float), np.asarray(log_likelihood, float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="caché .jsonl del runner")
    parser.add_argument("--single-work", type=int, default=200)
    parser.add_argument("--seeds", default="1-8", help="rango a-b de semillas")
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    first, last = (int(value) for value in args.seeds.split("-"))
    seeds = list(range(first, last + 1))

    print(f"cadena: K={N_STATES}, {N_ITERS} iteraciones, burn-in {BURN_IN}")
    print(f"obra: {args.source} obra {args.single_work}; semillas {first}..{last}")
    with ProcessPoolExecutor(max_workers=min(args.n_workers, len(seeds))) as pool:
        out = sorted(pool.map(chain, [(args.source, args.single_work, seed) for seed in seeds]))
    for seed, occupied, log_likelihood in out:
        print(f"seed {seed}: K medio={occupied.mean():.2f} (d.e. dentro={occupied.std():.2f}, "
              f"rango {occupied.min():.0f}-{occupied.max():.0f}); log L medio={log_likelihood.mean():.1f} "
              f"(d.e. dentro={log_likelihood.std():.1f})")
    preregistered = [row for row in out if 1 <= row[0] <= 4]
    if len(preregistered) == 4:
        print(f"R-hat dividido, semillas 1-4 (preregistro): K = {split_rhat([r[1] for r in preregistered]):.3f}, "
              f"log L = {split_rhat([r[2] for r in preregistered]):.3f}")
    print(f"R-hat dividido, {len(out)} cadenas: K = {split_rhat([r[1] for r in out]):.3f}, "
          f"log L = {split_rhat([r[2] for r in out]):.3f}")
    print("umbral del preregistro: R-hat de log L < 1.1")


if __name__ == "__main__":
    main()

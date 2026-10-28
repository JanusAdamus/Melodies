"""Criterio (c) con la configuración del preregistro (K=20, 400/200, semillas 1-4) y con ruido.

Uso:
    python s11_estados_con_ruido.py [--n-workers N]

Datos: HMM de 3 estados pegajosos de tests/test_hdp_sampler.py, 500 eventos, semilla 11.
- «limpia»: three_state_tokens (0.95 en el símbolo propio, 0.01 en los otros).
- «ruido 0.80»: noisy_three_state_tokens (0.80 en el símbolo propio, 0.04 en cada otro).
Para «ruido 0.80» se arranca por omisión, desde los estados verdaderos y desde un estado por
símbolo, con η = 1 (por omisión) y η = 6 (prior de emisión uniforme, η/V = 1).  Si los estados
extra fueran falta de mezcla, la cadena que arranca en la verdad se quedaría en ≈ 3.

Hallazgo H10 de docs/auditoria-2026-10-26.md.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def run(job):
    from tests.test_hdp_sampler import (
        CRITERION_B_CHAIN,
        PREREGISTERED_TRUNCATION,
        TruncatedHDPHMM,
        _observations,
        _StartFromStatesHDPHMM,
        noisy_three_state_tokens,
        three_state_tokens,
    )

    data, start, eta, seed = job
    if data == "limpia":
        tokens, truth = three_state_tokens(500, seed=11), None
    else:
        tokens, truth = noisy_three_state_tokens(500, seed=11, dominant=0.80)
    n_iters, burn_in = CRITERION_B_CHAIN
    kwargs = dict(n_states=PREREGISTERED_TRUNCATION, n_iters=n_iters, burn_in=burn_in, seed=seed, eta=eta)
    if start == "omision":
        model = TruncatedHDPHMM(**kwargs)
    else:
        model = _StartFromStatesHDPHMM(truth if start == "verdad" else tokens, **kwargs)
    return job, model.fit(_observations(tokens)).effective_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    seeds = (1, 2, 3, 4)
    jobs = [("limpia", "omision", 1.0, seed) for seed in seeds]
    jobs += [("ruido 0.80", start, eta, seed) for start in ("omision", "verdad", "simbolos")
             for eta in (1.0, 6.0) for seed in seeds]
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        out = list(pool.map(run, jobs))
    print("cadena: K=20, 400 iteraciones, burn-in 200; semillas 1-4; 500 eventos (semilla 11)")
    groups = {}
    for (data, start, eta, _), occupied in out:
        groups.setdefault((data, start, eta), []).append(occupied)
    for (data, start, eta), values in groups.items():
        print(f"{data:10s} arranque {start:8s} eta {eta}: K ocupados por cadena "
              f"{[round(v, 3) for v in values]} media {np.mean(values):.2f}")


if __name__ == "__main__":
    main()

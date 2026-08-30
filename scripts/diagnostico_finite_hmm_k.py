"""Diagnostico: ¿la rejilla de estados del HMM finito estaba topada?

En la corrida `tesis_3000_gpu_20260823_1941` el HMM finito eligio K=48 en las 30
corridas, y 48 es el techo de `finite_hmm_states=(12, 24, 48)`. Un modelo que siempre
elige el maximo de la rejilla puede estar limitado por la rejilla, no por la familia de
modelos, asi que su meseta en la curva de aprendizaje no es interpretable.

Este script responde la pregunta con el minimo de computo: reconstruye exactamente la
particion de la corrida, ajusta el HMM finito para cada K por separado sobre la fraccion
completa de entrenamiento, y compara la NLL de validacion, que es el criterio con el que
`FiniteGlobalHMM.fit` selecciona.

Si la NLL de validacion sigue bajando en K=96, el techo era vinculante y hay que rehacer
la curva del HMM finito con una rejilla que no se agote. Si no baja, K=48 era suficiente y
la meseta reportada es del modelo.

Cada K se ajusta en su propia instancia, de modo que todos reciben la misma semilla
derivada. Es un emparejamiento deliberado: aisla el efecto de K. Dentro de una rejilla real
el generador avanza entre candidatos, asi que los numeros no seran identicos bit a bit a
los de una corrida completa posterior.

Uso:

    .\\.venv\\Scripts\\python.exe scripts\\diagnostico_finite_hmm_k.py

    .\\.venv\\Scripts\\python.exe scripts\\diagnostico_finite_hmm_k.py --states 48,96,192
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Comparacion.classical_models import FiniteGlobalHMM
from Comparacion.config import build_default_learning_curve_config
from Comparacion.runner import _build_grouped_fixed_splits
from Comparacion.splits import build_nested_training_subsets
from next_token_experiment.data.preprocess import prepare_corpus
from next_token_experiment.data.tokenizer import build_tokenizer


# NLL de validacion del K seleccionado en la corrida de referencia, frac=1.0, model_seed=1.
# Fuente: artifacts/Comparacion/tesis_3000_gpu_20260823_1941/results_raw.csv
REFERENCE_RUN = "tesis_3000_gpu_20260823_1941"
REFERENCE_VALIDATION_PPL = 7.2514142350543835
REFERENCE_TEST_PPL = 7.324683800941208
REFERENCE_STATES = 48


def _parse_states(value: str) -> tuple[int, ...]:
    states = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not states:
        raise argparse.ArgumentTypeError("--states requires at least one integer")
    if any(state < 2 for state in states):
        raise argparse.ArgumentTypeError("--states values must be at least 2")
    if len(set(states)) != len(states):
        raise argparse.ArgumentTypeError("--states must not repeat values")
    return states


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", type=_parse_states, default=(24, 48, 96), help="Comma-separated state counts to fit, for example 24,48,96.")
    parser.add_argument("--corpus-root", default=str(REPO_ROOT / "external" / "PDMX" / "mxl"), help="Corpus root directory.")
    parser.add_argument("--max-files", type=int, default=3000, help="Cap on number of scores; must match the reference run.")
    parser.add_argument("--corpus-sample-seed", type=int, default=7, help="Corpus sampling seed; must match the reference run.")
    # El cache lo escribe `prepare_corpus` con exactamente las obras que pidio esta
    # invocacion, asi que compartirlo entre corridas con distinto --max-files borra las
    # entradas de la corrida grande y obliga a reparsear el corpus entero. El nombre por
    # defecto lleva --max-files para que cada tamano tenga el suyo.
    parser.add_argument("--corpus-cache", default=None, help="Parsed-score cache, so the diagnostic does not reparse the corpus. Defaults to artifacts/corpus_cache_<max_files>.jsonl; do not share one cache between different --max-files.")
    parser.add_argument("--n-workers", type=int, default=6, help="Parsing processes; only used if the cache is cold.")
    parser.add_argument("--data-seed", type=int, default=1, help="Data seed for the training subset.")
    parser.add_argument("--model-seed", type=int, default=1, help="Model seed passed to FiniteGlobalHMM.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Iteration budget per fit. Fitting stops early when the validation NLL stops improving, so a fit that reaches this cap was limited by the budget and not by the model. Defaults to the configured finite_hmm_max_iterations.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Training fraction to diagnose.")
    parser.add_argument("--output", default=None, help="Where to write the JSON report. Defaults to artifacts/diagnostico_finite_hmm_k.json.")
    return parser.parse_args()


def _select_subset(nested_subsets, fraction: float):
    for subset_fraction, pieces in nested_subsets:
        if math.isclose(subset_fraction, fraction, rel_tol=0.0, abs_tol=1e-9):
            return pieces
    available = ", ".join(str(item[0]) for item in nested_subsets)
    raise SystemExit(f"fraction {fraction} not in the configured training fractions: {available}")


def main() -> int:
    args = _parse_args()
    config = build_default_learning_curve_config(corpus_root=args.corpus_root)
    max_iterations = args.max_iterations if args.max_iterations is not None else config.finite_hmm_max_iterations
    if max_iterations <= 0:
        raise SystemExit("--max-iterations must be a positive integer.")

    corpus_cache = args.corpus_cache or str(REPO_ROOT / "artifacts" / f"corpus_cache_{args.max_files}.jsonl")
    print(f"[diagnostico] preparando corpus (cache: {corpus_cache})", flush=True)
    preparation = prepare_corpus(
        config.experiment,
        max_files=args.max_files,
        n_workers=args.n_workers,
        cache_path=corpus_cache,
        sample_seed=args.corpus_sample_seed,
    )
    fixed_splits = _build_grouped_fixed_splits(
        preparation.pieces,
        test_fraction=config.test_fraction,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )
    nested_subsets = build_nested_training_subsets(
        fixed_splits.train_pool_pieces,
        fractions=config.train_fractions,
        data_seed=args.data_seed,
    )
    train_pieces = _select_subset(nested_subsets, args.fraction)

    tokenizer = build_tokenizer(config.experiment.representation)
    bos_token_id = tokenizer.bos_token_id
    comparison_vocabulary_size = bos_token_id + 1
    max_context_length = config.experiment.windows.max_context_length

    n_train_tokens = sum(len(piece.tokens) for piece in train_pieces)
    print(
        f"[diagnostico] frac={args.fraction} data_seed={args.data_seed} "
        f"model_seed={args.model_seed} | train {len(train_pieces)} piezas, {n_train_tokens} tokens | "
        f"val {len(fixed_splits.validation_pieces)} | test {len(fixed_splits.test_pieces)}",
        flush=True,
    )
    print(
        f"[diagnostico] referencia {REFERENCE_RUN}: K={REFERENCE_STATES} "
        f"val_ppl={REFERENCE_VALIDATION_PPL:.4f} test_ppl={REFERENCE_TEST_PPL:.4f}",
        flush=True,
    )

    results: list[dict[str, object]] = []
    for n_states in args.states:
        print(f"[diagnostico] ajustando K={n_states} ...", flush=True)
        started = time.perf_counter()
        model = FiniteGlobalHMM(
            candidate_num_states=(n_states,),
            max_iterations=max_iterations,
            tolerance=config.finite_hmm_tolerance,
            seed=args.model_seed,
            vocab_size=comparison_vocabulary_size,
        )
        fit_result = model.fit(
            train_pieces,
            fixed_splits.validation_pieces,
            bos_token_id=bos_token_id,
            max_context_length=max_context_length,
        )
        evaluation = model.evaluate(
            fixed_splits.test_pieces,
            bos_token_id=bos_token_id,
            max_context_length=max_context_length,
        )
        summary = evaluation["summary"]
        elapsed = time.perf_counter() - started
        row = {
            "n_states": n_states,
            "validation_nll_per_token": fit_result.validation_nll,
            "validation_ppl": math.exp(fit_result.validation_nll),
            "test_nll_per_token": summary["test_nll_per_token"],
            "test_ppl": math.exp(float(summary["test_nll_per_token"])),
            "n_params": summary["n_params"],
            "em_iterations": len(fit_result.train_log),
            "fit_wall_clock_s": fit_result.train_wall_clock_s,
            "total_wall_clock_s": elapsed,
        }
        results.append(row)
        print(
            f"[diagnostico] K={n_states:4d}  val_ppl={row['validation_ppl']:.4f}  "
            f"test_ppl={row['test_ppl']:.4f}  iteraciones={row['em_iterations']}  "
            f"{elapsed:.0f}s",
            flush=True,
        )

    best = min(results, key=lambda row: row["validation_nll_per_token"])
    largest = max(results, key=lambda row: row["n_states"])
    ceiling_is_binding = best["n_states"] == largest["n_states"] and len(results) > 1
    verdict = (
        "techo_vinculante: la validacion sigue prefiriendo el K mas grande de la rejilla probada; "
        "rehacer la curva del HMM finito con una rejilla mayor"
        if ceiling_is_binding
        else f"techo_no_vinculante: la validacion selecciona K={best['n_states']}, interior a la rejilla probada"
    )

    report = {
        "diagnostic": "finite_hmm_state_grid_ceiling",
        "reference_run": REFERENCE_RUN,
        "reference": {
            "n_states": REFERENCE_STATES,
            "validation_ppl": REFERENCE_VALIDATION_PPL,
            "test_ppl": REFERENCE_TEST_PPL,
        },
        "protocol": {
            "fraction": args.fraction,
            "data_seed": args.data_seed,
            "model_seed": args.model_seed,
            "max_files": args.max_files,
            "corpus_sample_seed": args.corpus_sample_seed,
            "split_seed": config.split_seed,
            "n_train_pieces": len(train_pieces),
            "n_train_tokens": n_train_tokens,
            "n_validation_pieces": len(fixed_splits.validation_pieces),
            "n_test_pieces": len(fixed_splits.test_pieces),
            "max_iterations": max_iterations,
            "tolerance": config.finite_hmm_tolerance,
            "seeding": "each K fitted in its own instance, so all K receive the same derived seed",
        },
        "candidates": results,
        "selected_by_validation": best["n_states"],
        "ceiling_is_binding": ceiling_is_binding,
        "verdict": verdict,
    }

    output_path = Path(args.output) if args.output else REPO_ROOT / "artifacts" / "diagnostico_finite_hmm_k.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"{'K':>6}  {'val_ppl':>9}  {'test_ppl':>9}  {'params':>9}  {'iter':>5}  {'seg':>7}")
    for row in results:
        print(
            f"{row['n_states']:>6}  {row['validation_ppl']:>9.4f}  {row['test_ppl']:>9.4f}  "
            f"{row['n_params']:>9}  {row['em_iterations']:>5}  {row['total_wall_clock_s']:>7.0f}"
        )
    print()
    print(f"[diagnostico] {verdict}")
    print(f"[diagnostico] reporte: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from next_token_experiment.config import HardwareConfig, TransformerConfig

from .config import build_default_learning_curve_config
from .runner import run_learning_curve_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the learning-curve comparison in Comparacion/.")
    parser.add_argument("--run-name", default="learning_curve", help="Output directory name under artifacts/Comparacion.")
    parser.add_argument("--corpus-root", default=None, help="Override corpus root directory.")
    parser.add_argument("--results-root", default=None, help="Override results root directory.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of scores.")
    parser.add_argument("--corpus-sample-seed", type=int, default=None, help="Draw the --max-files scores as a seeded random sample instead of the first N in path order.")
    parser.add_argument("--transformer-max-epochs", type=int, default=None, help="Override Transformer max epochs.")
    parser.add_argument("--transformer-device", default=None, choices=("cpu", "auto", "cuda", "mps"), help="Override Transformer device.")
    parser.add_argument("--data-seeds", default=None, help="Comma-separated data seeds, for example 1,2,3.")
    parser.add_argument("--model-seeds", default=None, help="Comma-separated model seeds, for example 1,2.")
    parser.add_argument("--fractions", default=None, help="Comma-separated training fractions, for example 0.1,0.5,1.0.")
    parser.add_argument("--n-workers", type=int, default=None, help="Parsing processes for corpus preparation; 1 forces the serial path. Defaults to CPU count minus one.")
    parser.add_argument("--corpus-cache", default=None, help="JSONL file caching parsed scores so an interrupted run resumes instead of reparsing.")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run in the same --run-name directory, skipping the (data_seed, fraction) cells already recorded in checkpoint.jsonl.")
    parser.add_argument("--plan-only", action="store_true", help="Write an execution plan without constructing or fitting models.")
    parser.add_argument("--without-vomm", action="store_true", help="Disable the optional PPM-inspired VOMM diagnostic control.")
    parser.add_argument("--structural-annotations", default=None, help="Optional CSV with piece_id,event_index,segment_label,boundary columns.")
    return parser


def _parse_int_tuple(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    return values or None


def _parse_float_tuple(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    return values or None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = build_default_learning_curve_config(
        corpus_root=args.corpus_root,
        results_root=args.results_root,
    )
    experiment = config.experiment

    if args.transformer_max_epochs is not None:
        experiment = replace(
            experiment,
            transformer=replace(experiment.transformer, max_epochs=args.transformer_max_epochs),
        )
    if args.transformer_device is not None:
        experiment = replace(
            experiment,
            hardware=replace(experiment.hardware, target_device=args.transformer_device, gpu_required=args.transformer_device == "cuda"),
        )

    updates = {
        "experiment": experiment,
        "include_vomm_control": not args.without_vomm,
        "structural_annotations_path": args.structural_annotations,
    }
    parsed_data_seeds = _parse_int_tuple(args.data_seeds)
    if parsed_data_seeds is not None:
        updates["data_seeds"] = parsed_data_seeds
    parsed_model_seeds = _parse_int_tuple(args.model_seeds)
    if parsed_model_seeds is not None:
        updates["model_seeds"] = parsed_model_seeds
    parsed_fractions = _parse_float_tuple(args.fractions)
    if parsed_fractions is not None:
        updates["train_fractions"] = parsed_fractions

    config = replace(config, **updates)
    result = run_learning_curve_experiment(
        config,
        run_name=args.run_name,
        max_files=args.max_files,
        plan_only=args.plan_only,
        n_workers=args.n_workers,
        corpus_cache_path=args.corpus_cache,
        corpus_sample_seed=args.corpus_sample_seed,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import replace
import json

from next_token_experiment.experiment.runner import run_small_transformer_experiment
from next_token_experiment.profiles import build_profile_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a large PDMX GPU next-token experiment with Windows-safe multiprocessing.")
    parser.add_argument("--run-name", default="pdmx_gpu_max_65536_e80_verbose")
    parser.add_argument("--corpus-root", default="external/PDMX/mxl")
    parser.add_argument("--max-files", type=int, default=16384)
    parser.add_argument("--train-windows", type=int, default=65536)
    parser.add_argument("--validation-windows", type=int, default=8192)
    parser.add_argument("--test-windows", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=2,
        help="Use 0 if Windows multiprocessing gives trouble.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = build_profile_config("gpu_extended", corpus_root=args.corpus_root)
    config = replace(
        config,
        hardware=replace(config.hardware, dataloader_workers=args.dataloader_workers),
        transformer=replace(
            config.transformer,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ),
    )

    result = run_small_transformer_experiment(
        config=config,
        run_name=args.run_name,
        max_files=args.max_files,
        max_windows_per_split={
            "train": args.train_windows,
            "validation": args.validation_windows,
            "test": args.test_windows,
        },
    )

    payload = {
        "profile": "gpu_extended_custom",
        "result_dir": result["result_dir"],
        "test_summary": result["test_summary"]["summary"],
        "fit_summary": result["fit_result"]["summary"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

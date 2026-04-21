from __future__ import annotations

import argparse
import json

from .experiment.runner import run_small_transformer_experiment
from .profiles import PROFILE_DESCRIPTIONS, build_profile_config, list_profiles, profile_requires_scope_validation
from .protocol import validate_experiment_scope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline or research next-token transformer experiments.")
    parser.add_argument("--profile", choices=list_profiles(), required=True, help="Named experiment profile.")
    parser.add_argument("--run-name", default=None, help="Optional run directory name.")
    parser.add_argument("--corpus-root", default=None, help="Override corpus root directory.")
    parser.add_argument("--results-root", default=None, help="Override artifacts results root.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of scores.")
    parser.add_argument("--max-windows-train", type=int, default=None, help="Optional cap for train windows.")
    parser.add_argument(
        "--max-windows-validation",
        type=int,
        default=None,
        help="Optional cap for validation windows.",
    )
    parser.add_argument("--max-windows-test", type=int, default=None, help="Optional cap for test windows.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = build_profile_config(
        args.profile,
        corpus_root=args.corpus_root,
        results_root=args.results_root,
    )

    if profile_requires_scope_validation(args.profile):
        issues = validate_experiment_scope(config)
        if issues:
            raise SystemExit(
                "Profile violates the bounded baseline scope:\n- " + "\n- ".join(issues)
            )

    max_windows = {
        "train": args.max_windows_train,
        "validation": args.max_windows_validation,
        "test": args.max_windows_test,
    }
    max_windows = {key: value for key, value in max_windows.items() if value is not None}

    result = run_small_transformer_experiment(
        config=config,
        run_name=args.run_name,
        max_files=args.max_files,
        max_windows_per_split=max_windows or None,
    )
    payload = {
        "profile": args.profile,
        "profile_description": PROFILE_DESCRIPTIONS[args.profile],
        "result_dir": result["result_dir"],
        "test_summary": result["test_summary"]["summary"],
        "fit_summary": result["fit_result"]["summary"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

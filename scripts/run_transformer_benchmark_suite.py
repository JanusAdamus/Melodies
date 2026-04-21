from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from next_token_experiment.benchmarks.suite import (
    DEFAULT_MANIFEST_PATH,
    collect_benchmark_suite,
    execute_benchmark_suite,
    load_benchmark_run_specs,
    resolve_benchmark_suite,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute or collect canonical transformer benchmark runs.")
    parser.add_argument("--mode", choices=("execute", "collect"), default="collect", help="Whether to execute runs or only collect existing artifacts.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Path to the canonical benchmark manifest.")
    parser.add_argument("--run-id", action="append", default=None, help="Select a specific run id. May be passed multiple times.")
    parser.add_argument("--group", default=None, help="Filter by comparison_group.")
    parser.add_argument("--only-smoke", action="store_true", help="Keep only smoke runs.")
    parser.add_argument("--only-full", action="store_true", help="Keep only full runs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve configs without executing experiments.")
    parser.add_argument("--corpus-root", default=None, help="Optional override for the corpus root.")
    parser.add_argument(
        "--results-root",
        default="artifacts/next_token_experiment/results",
        help="Root where experiment runs and benchmark summaries are stored.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_specs = load_benchmark_run_specs(args.manifest)
    resolved = resolve_benchmark_suite(
        run_specs,
        run_ids=set(args.run_id) if args.run_id else None,
        comparison_group=args.group,
        only_smoke=args.only_smoke,
        only_full=args.only_full,
    )
    if not resolved:
        raise SystemExit("No benchmark runs matched the requested filters.")

    if args.mode == "execute":
        payload = execute_benchmark_suite(
            resolved,
            corpus_root=args.corpus_root,
            results_root=args.results_root,
            dry_run=args.dry_run,
        )
    else:
        payload = collect_benchmark_suite(
            resolved,
            results_root=args.results_root,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

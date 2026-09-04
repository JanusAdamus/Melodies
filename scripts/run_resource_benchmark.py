from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparacion.resource_benchmark import run_resource_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repite los ajustes seleccionados y audita tiempo y memoria."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-corpus-fingerprint", required=True)
    parser.add_argument("--corpus-cache")
    parser.add_argument(
        "--measurement-condition",
        choices=("isolated", "contended", "unknown"),
        default="isolated",
    )
    args = parser.parse_args()
    result = run_resource_benchmark(
        source_run=args.source_run,
        fraction=args.fraction,
        split_seed=args.split_seed,
        repetitions=args.repetitions,
        output_dir=args.output_dir,
        expected_corpus_fingerprint=args.expected_corpus_fingerprint,
        corpus_cache=args.corpus_cache,
        measurement_condition=args.measurement_condition,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

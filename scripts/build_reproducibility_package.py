from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparacion.engineering_requirements import build_reproducibility_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a corpus-free evidence package.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_reproducibility_package(
        source_run=args.source_run,
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

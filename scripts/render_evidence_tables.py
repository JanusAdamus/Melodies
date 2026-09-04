from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparacion.engineering_requirements import render_evidence_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Render deterministic R4/R5 Markdown tables.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in render_evidence_tables(
        source_run=args.source_run,
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()

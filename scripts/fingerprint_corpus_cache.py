from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparacion.corpus_fingerprint import fingerprint_corpus_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a path-independent fingerprint for a corpus JSONL cache."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = fingerprint_corpus_cache(args.cache)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

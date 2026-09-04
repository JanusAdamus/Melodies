from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparacion.engineering_requirements import write_validation_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate thesis engineering evidence.")
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument("--requirements-file")
    args = parser.parse_args()
    print(
        json.dumps(
            write_validation_reports(
                args.repo_root,
                requirements_path=args.requirements_file,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

"""Ejecuta ambas suites y guarda un informe JSON consumible por la matriz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_ids: list[str] = []

    def startTest(self, test: unittest.TestCase) -> None:  # noqa: N802 - API unittest
        self.test_ids.append(test.id())
        super().startTest(test)


class RecordingRunner(unittest.TextTestRunner):
    resultclass = RecordingResult


def _commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        [
            loader.discover("tests", pattern="test*.py", top_level_dir="."),
            loader.discover(
                "next_token_experiment/tests", pattern="test*.py", top_level_dir="."
            ),
        ]
    )
    runner = RecordingRunner(verbosity=2)
    result = runner.run(suite)
    assert isinstance(result, RecordingResult)
    payload = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "skipped": [test.id() for test, _ in result.skipped],
        "test_ids": result.test_ids,
        "python_version": platform.python_version(),
        "commit": _commit(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

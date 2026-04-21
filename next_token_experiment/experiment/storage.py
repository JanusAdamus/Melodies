from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_result_file_plan(results_root: str, model_name: str) -> dict[str, str]:
    """Return the expected file layout for a model run."""

    root = Path(results_root) / model_name
    return {
        "config": str(root / "config.json"),
        "train_log": str(root / "train_log.csv"),
        "validation_metrics": str(root / "validation_metrics.csv"),
        "validation_slice_metrics": str(root / "validation_slice_metrics.json"),
        "test_piece_metrics": str(root / "test_piece_metrics.csv"),
        "test_slice_metrics": str(root / "test_slice_metrics.json"),
        "test_summary": str(root / "test_summary.json"),
        "generated_continuations": str(root / "generated_continuations.json"),
    }


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    pd.DataFrame(rows).to_csv(target, index=False)

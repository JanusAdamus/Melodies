"""Reconciliación entre archivos evaluados y obras canónicas comparadas.

El reporte de la tesis cita dos denominadores distintos: el número de archivos
de prueba y el número de obras usadas en las comparaciones pareadas. Este módulo
explica la diferencia con evidencia: agrupación por obra canónica, descartes con
motivo, o ambos.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
import csv
import json
import math
from pathlib import Path

AUDIT_FILENAME = "denominator_audit.json"

_FULL_FRACTION = 1.0


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _row_nll(row: Mapping[str, object]) -> float | None:
    if "nll_per_token" in row:
        return _finite_float(row["nll_per_token"])
    return _finite_float(row.get("test_nll"))


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def build_denominator_audit(
    piece_metric_rows: Iterable[Mapping[str, object]],
    *,
    test_piece_ids: Sequence[str] | None = None,
    validation_piece_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Explica la distancia entre archivos evaluados y obras comparadas.

    Sólo cuentan las filas de ``frac == 1.0``: son las que alimentan las
    comparaciones pareadas. Cada fila descartada queda registrada con su motivo
    en lugar de desaparecer del conteo.
    """

    scored_files: set[str] = set()
    works: set[str] = set()
    work_by_file: dict[str, str] = {}
    files_by_work: dict[str, set[str]] = {}
    works_by_model: dict[str, set[str]] = {}
    files_by_model: dict[str, set[str]] = {}
    discards: list[dict[str, object]] = []

    for row in piece_metric_rows:
        if not isinstance(row, Mapping):
            continue
        fraction = _finite_float(row.get("frac"))
        if fraction != _FULL_FRACTION:
            continue
        model = _text(row.get("model"))
        piece_id = _text(row.get("piece_id"))
        canonical_work_id = _text(row.get("canonical_work_id"))
        nll = _row_nll(row)

        if piece_id is None:
            discards.append({"piece_id": None, "model": model, "reason": "missing_piece_id"})
            continue
        if canonical_work_id is None:
            discards.append(
                {"piece_id": piece_id, "model": model, "reason": "missing_canonical_work_id"}
            )
            continue
        if nll is None:
            discards.append(
                {
                    "piece_id": piece_id,
                    "canonical_work_id": canonical_work_id,
                    "model": model,
                    "reason": "non_finite_nll",
                }
            )
            continue

        scored_files.add(piece_id)
        works.add(canonical_work_id)
        work_by_file[piece_id] = canonical_work_id
        files_by_work.setdefault(canonical_work_id, set()).add(piece_id)
        if model is not None:
            works_by_model.setdefault(model, set()).add(canonical_work_id)
            files_by_model.setdefault(model, set()).add(piece_id)

    grouped_works = {work: files for work, files in files_by_work.items() if len(files) > 1}
    absorbed = sum(len(files) - 1 for files in grouped_works.values())

    unscored_test_files: list[str] = []
    if test_piece_ids is not None:
        unscored_test_files = sorted(set(test_piece_ids) - scored_files)

    pairs: list[dict[str, object]] = []
    for model_a, model_b in combinations(sorted(works_by_model), 2):
        works_a = works_by_model[model_a]
        works_b = works_by_model[model_b]
        pairs.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_common_works": len(works_a & works_b),
                "only_model_a": sorted(works_a - works_b),
                "only_model_b": sorted(works_b - works_a),
            }
        )

    has_grouping = absorbed > 0
    has_discards = bool(discards) or bool(unscored_test_files)
    if has_grouping and has_discards:
        explanation = "canonicalization_and_discards"
    elif has_grouping:
        explanation = "canonicalization_only"
    elif has_discards:
        explanation = "discards_only"
    else:
        explanation = "files_equal_works"

    return {
        "n_scored_files": len(scored_files),
        "n_canonical_works": len(works),
        "n_works_with_multiple_files": len(grouped_works),
        "n_files_absorbed_by_grouping": absorbed,
        "grouped_works": {work: sorted(files) for work, files in sorted(grouped_works.items())},
        "n_test_split_files": len(test_piece_ids) if test_piece_ids is not None else None,
        "n_validation_split_files": (
            len(validation_piece_ids) if validation_piece_ids is not None else None
        ),
        "unscored_test_files": unscored_test_files,
        "discards": discards,
        "per_model": {
            model: {
                "n_scored_files": len(files_by_model.get(model, set())),
                "n_canonical_works": len(works_by_model[model]),
            }
            for model in sorted(works_by_model)
        },
        "pairs": pairs,
        "explanation": explanation,
    }


def read_piece_metric_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_piece_ids(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    piece_ids = payload.get("piece_ids") if isinstance(payload, Mapping) else None
    return [str(item) for item in piece_ids] if isinstance(piece_ids, Sequence) else None


def audit_run_directory(run_dir: str | Path) -> dict[str, object]:
    """Construye la auditoría de denominadores desde los artefactos de una corrida."""

    root = Path(run_dir)
    metrics_path = root / "piece_metrics_raw.csv"
    if not metrics_path.exists():
        return {"status": "incomplete", "reason": "piece_metrics_raw_not_found", "run_dir": str(root)}

    rows = read_piece_metric_rows(metrics_path)
    audit = build_denominator_audit(
        rows,
        test_piece_ids=_split_piece_ids(root / "splits" / "test_pieces.json"),
        validation_piece_ids=_split_piece_ids(root / "splits" / "val_pieces.json"),
    )
    return {"status": "ok", "run_dir": str(root), **audit}

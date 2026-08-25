"""Informe de riesgo del agrupamiento por obra canónica.

El agrupamiento decide qué archivos cuentan como la misma obra y, por lo tanto,
qué observaciones son independientes en las comparaciones. Este módulo no cambia
ninguna asignación: sólo expone los grupos que necesitan revisión humana.

La huella melódica es un diagnóstico. Nunca prueba por sí sola que dos archivos
sean la misma obra ni justifica fusionar o separar grupos de forma automática.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path

from next_token_experiment.experiment.splits import canonicalization_details

AUDIT_FILENAME = "canonicalization_audit.json"

FINGERPRINT_POLICY = "diagnostic_only_never_automatic_identity_proof"

_FINGERPRINT_LENGTH = 32


def melodic_fingerprint(tokens: Sequence[int] | None, *, length: int = _FINGERPRINT_LENGTH) -> str | None:
    """Huella de los primeros intervalos. Invariante a transposición constante."""

    if not tokens:
        return None
    values = [int(token) for token in tokens[: length + 1]]
    if len(values) < 2:
        return None
    intervals = [second - first for first, second in zip(values, values[1:])]
    digest = hashlib.sha256(",".join(str(interval) for interval in intervals).encode("utf-8"))
    return digest.hexdigest()[:16]


def _field(piece: Mapping[str, object] | object, name: str) -> object:
    if isinstance(piece, Mapping):
        return piece.get(name)
    return getattr(piece, name, None)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_canonicalization_audit(
    pieces: Iterable[Mapping[str, object] | object],
) -> dict[str, object]:
    """Agrupa por obra canónica y señala lo que un humano debe revisar."""

    records: list[dict[str, object]] = []
    for piece in pieces:
        title = _text(_field(piece, "title"))
        composer = _text(_field(piece, "composer"))
        canonical_work_id = _text(_field(piece, "canonical_work_id"))
        details = canonicalization_details(f"{composer} {title}")
        tokens = _field(piece, "tokens")
        records.append(
            {
                "piece_id": _text(_field(piece, "piece_id")),
                "title": title,
                "composer": composer,
                "canonical_work_id": canonical_work_id or str(details["canonical_work_id"]),
                "aggressive_key": details["aggressive_key"],
                "is_empty": details["is_empty"],
                "is_generic": details["is_generic"],
                "is_short": details["is_short"],
                "dropped_suffix": details["dropped_suffix"],
                "dropped_suffix_text": details["dropped_suffix_text"],
                "fingerprint": melodic_fingerprint(tokens if isinstance(tokens, Sequence) else None),
            }
        )

    by_work: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_work.setdefault(str(record["canonical_work_id"]), []).append(record)

    multi_file_groups: list[dict[str, object]] = []
    suspicious: list[dict[str, object]] = []
    for canonical_work_id, group in sorted(by_work.items()):
        if len(group) < 2:
            continue
        fingerprints = {record["fingerprint"] for record in group if record["fingerprint"]}
        composers = sorted({str(record["composer"]).lower() for record in group if record["composer"]})
        fingerprints_agree = len(fingerprints) == 1 if fingerprints else None
        entry = {
            "canonical_work_id": canonical_work_id,
            "piece_ids": sorted(str(record["piece_id"]) for record in group),
            "titles": sorted({str(record["title"]) for record in group}),
            "composers": composers,
            "fingerprints_agree": fingerprints_agree,
        }
        multi_file_groups.append(entry)
        if any(record["dropped_suffix"] for record in group):
            suspicious.append(
                {
                    "canonical_work_id": canonical_work_id,
                    "reason": "dropped_suffix",
                    "piece_ids": entry["piece_ids"],
                    "dropped": sorted(
                        str(record["dropped_suffix_text"])
                        for record in group
                        if record["dropped_suffix"]
                    ),
                }
            )
        if fingerprints_agree is False:
            suspicious.append({"canonical_work_id": canonical_work_id, "reason": "fingerprints_disagree", "piece_ids": entry["piece_ids"]})
        elif len(composers) > 1:
            suspicious.append({"canonical_work_id": canonical_work_id, "reason": "multiple_composers", "piece_ids": entry["piece_ids"]})

    empty_metadata = [
        {"piece_id": record["piece_id"], "title": record["title"], "composer": record["composer"]}
        for record in records
        if record["is_empty"] or not record["title"] or not record["composer"]
    ]
    generic_labels = [
        {"piece_id": record["piece_id"], "canonical_work_id": record["canonical_work_id"]}
        for record in records
        if record["is_generic"] or record["is_short"]
    ]

    dropped_suffixes = [
        {
            "piece_id": record["piece_id"],
            "canonical_work_id": record["canonical_work_id"],
            "dropped_suffix_text": record["dropped_suffix_text"],
        }
        for record in records
        if record["dropped_suffix"]
    ]

    by_aggressive_key: dict[str, set[str]] = {}
    for record in records:
        key = str(record["aggressive_key"])
        if not key:
            continue
        by_aggressive_key.setdefault(key, set()).add(str(record["canonical_work_id"]))
    near_miss_groups = [
        {"aggressive_key": key, "canonical_work_ids": sorted(work_ids)}
        for key, work_ids in sorted(by_aggressive_key.items())
        if len(work_ids) > 1
    ]

    review_required = sorted(
        {str(item["canonical_work_id"]) for item in suspicious}
        | {str(item["canonical_work_id"]) for item in generic_labels}
        | {str(item["canonical_work_id"]) for item in dropped_suffixes}
        | {work_id for group in near_miss_groups for work_id in group["canonical_work_ids"]}
    )

    return {
        "n_files": len(records),
        "n_canonical_works": len(by_work),
        "multi_file_groups": multi_file_groups,
        "empty_metadata": empty_metadata,
        "generic_labels": generic_labels,
        "dropped_suffixes": dropped_suffixes,
        "near_miss_groups": near_miss_groups,
        "suspicious_collisions": suspicious,
        "review_required": review_required,
        "fingerprint_policy": FINGERPRINT_POLICY,
        "manual_review": [],
    }


def write_canonicalization_audit(
    pieces: Iterable[Mapping[str, object] | object], output_path: str | Path
) -> dict[str, object]:
    audit = build_canonicalization_audit(pieces)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit

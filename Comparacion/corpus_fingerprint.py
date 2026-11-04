"""Huella portátil del caché preprocesado del corpus.

El SHA-256 del archivo JSONL no es comparable entre máquinas porque cada
entrada conserva la ruta local de la partitura. Esta huella elimina sólo esos
campos de ubicación, ordena las entradas por ``piece_id`` y mantiene el
contenido que determina el experimento: tokens, metadatos musicales y motivos
de exclusión.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


FINGERPRINT_ALGORITHM = "sha256-canonical-corpus-v1"
_PIECE_FIELDS = (
    "kind",
    "piece_id",
    "title",
    "composer",
    "canonical_work_id",
    "representation",
    "tokens",
    "n_events",
    "metadata",
)
_EXCLUSION_FIELDS = ("kind", "piece_id", "reason")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_entry(payload: Mapping[str, object], *, line_number: int) -> dict[str, object]:
    kind = payload.get("kind")
    if kind not in {"piece", "exclusion"}:
        raise ValueError(f"cache line {line_number} has invalid kind: {kind!r}")
    piece_id = payload.get("piece_id")
    if not isinstance(piece_id, str) or not piece_id.strip():
        raise ValueError(f"cache line {line_number} has no piece_id")

    fields = _PIECE_FIELDS if kind == "piece" else _EXCLUSION_FIELDS
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"cache line {line_number} is missing fields: {missing}")
    entry = {field: payload[field] for field in fields}
    if kind == "piece":
        tokens = entry["tokens"]
        n_events = entry["n_events"]
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise ValueError(f"cache line {line_number} has invalid tokens")
        if not isinstance(n_events, int) or n_events < 0:
            raise ValueError(f"cache line {line_number} has invalid n_events")
    return entry


def fingerprint_corpus_cache(cache_path: str | Path) -> dict[str, object]:
    """Return a path-independent fingerprint and corpus counts for a JSONL cache."""

    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    entries: list[dict[str, object]] = []
    seen_piece_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(payload, Mapping):
                raise ValueError(f"cache line {line_number} is not an object")
            entry = _canonical_entry(payload, line_number=line_number)
            piece_id = str(entry["piece_id"])
            if piece_id in seen_piece_ids:
                raise ValueError(f"duplicate piece_id in cache: {piece_id}")
            seen_piece_ids.add(piece_id)
            entries.append(entry)

    if not entries:
        raise ValueError(f"empty corpus cache: {path}")

    entries.sort(key=lambda item: (str(item["piece_id"]), str(item["kind"])))
    digest = hashlib.sha256()
    for entry in entries:
        encoded = json.dumps(
            entry,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")

    pieces = [entry for entry in entries if entry["kind"] == "piece"]
    exclusions = [entry for entry in entries if entry["kind"] == "exclusion"]
    return {
        "algorithm": FINGERPRINT_ALGORITHM,
        "sha256": digest.hexdigest().upper(),
        "raw_file_sha256": _file_sha256(path),
        "counts": {
            "entries": len(entries),
            "prepared_pieces": len(pieces),
            "exclusions": len(exclusions),
            "events": sum(int(entry["n_events"]) for entry in pieces),
            "tokens": sum(len(entry["tokens"]) for entry in pieces),
        },
        "excluded_fields": ["source_path", "path", "detail"],
    }

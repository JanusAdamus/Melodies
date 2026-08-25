from __future__ import annotations

import math
from pathlib import Path
import random
import re
import unicodedata

from ..config import SplitConfig
from ..schemas import PreparedPiece


NOISE_TOKENS = {
    "easy",
    "beginner",
    "fingered",
    "arrangement",
    "arrg",
    "solo",
    "piano",
    "pedagogical",
    "annotated",
    "version",
}


def canonicalize_work_label(label: str) -> str:
    """Normalize filenames or titles to reduce duplicate leakage across splits."""

    text = Path(label).stem.lower()
    text = re.sub(r"[_.\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [token for token in text.split() if token not in NOISE_TOKENS]
    return " ".join(tokens).strip()


GENERIC_LABELS = {
    "untitled",
    "unknown",
    "score",
    "sheet music",
    "no title",
    "sin titulo",
}


def canonicalization_details(label: str) -> dict[str, object]:
    """Diagnóstico del agrupamiento por obra canónica.

    Devuelve el mismo identificador que ``canonicalize_work_label`` más señales
    de riesgo. ``aggressive_key`` ignora puntuación, espacios y diacríticos: sólo
    sirve para señalar títulos cercanos que quedaron separados, nunca para
    fusionarlos automáticamente.
    """

    canonical_work_id = canonicalize_work_label(label)
    # canonicalize_work_label usa Path().stem, así que un sufijo final con punto
    # ("Sonata No.2") desaparece del identificador. Se registra como riesgo en
    # lugar de cambiar el agrupamiento de corridas ya publicadas.
    dropped_suffix_text = Path(label).suffix
    folded = unicodedata.normalize("NFKD", canonical_work_id)
    folded = "".join(character for character in folded if not unicodedata.combining(character))
    aggressive_key = re.sub(r"[^a-z0-9]+", "", folded)
    return {
        "canonical_work_id": canonical_work_id,
        "aggressive_key": aggressive_key,
        "is_empty": not canonical_work_id,
        "is_generic": canonical_work_id in GENERIC_LABELS,
        "is_short": bool(canonical_work_id) and len(canonical_work_id.split()) < 2,
        "dropped_suffix": bool(dropped_suffix_text),
        "dropped_suffix_text": dropped_suffix_text,
    }


def _allocate_group_counts(
    n_total: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    ratios = [train_ratio, validation_ratio, test_ratio]
    raw = [n_total * ratio for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    positive_splits = [index for index, ratio in enumerate(ratios) if ratio > 0]

    if n_total >= len(positive_splits):
        for index in positive_splits:
            counts[index] = max(counts[index], 1)

    while sum(counts) > n_total:
        for index in reversed(range(3)):
            minimum = 1 if index in positive_splits and n_total >= len(positive_splits) else 0
            if counts[index] > minimum and sum(counts) > n_total:
                counts[index] -= 1

    remainder = n_total - sum(counts)
    fractional = [raw[index] - math.floor(raw[index]) for index in range(3)]
    order = sorted(range(3), key=lambda index: fractional[index], reverse=True)
    for index in order:
        if remainder <= 0:
            break
        counts[index] += 1
        remainder -= 1

    return counts[0], counts[1], counts[2]


def deterministic_group_split(
    group_ids: list[str],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign each group id to a split using a reproducible shuffle."""

    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")

    unique_group_ids = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_group_ids)

    n_train, n_validation, _ = _allocate_group_counts(
        n_total=len(unique_group_ids),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    assignments: dict[str, str] = {}
    for index, group_id in enumerate(unique_group_ids):
        if index < n_train:
            assignments[group_id] = "train"
        elif index < n_train + n_validation:
            assignments[group_id] = "validation"
        else:
            assignments[group_id] = "test"
    return assignments


def assign_piece_splits(pieces: list[PreparedPiece], config: SplitConfig) -> dict[str, str]:
    """Assign splits by canonical work id to avoid leakage across variants."""

    group_ids = [piece.canonical_work_id for piece in pieces]
    return deterministic_group_split(
        group_ids=group_ids,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

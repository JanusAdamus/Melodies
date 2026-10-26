"""Deduplicacion por contenido (D4): dos archivos con la misma melodia son una sola obra."""

from __future__ import annotations

import hashlib
from pathlib import PurePath
from typing import Iterable

import pandas as pd


def content_fingerprint(pitch_classes: Iterable[int]) -> str:
    """Huella de la secuencia de clases de altura (independiente del archivo y los metadatos)."""

    return hashlib.sha1(" ".join(str(int(value)) for value in pitch_classes).encode("ascii")).hexdigest()


def deduplicate_by_fingerprint(
    frame: pd.DataFrame, fingerprint_column: str = "content_fingerprint"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conserva una obra por huella y devuelve ``(conservadas, quitadas)``.

    Gana la primera ruta en orden POSIX (independiente del sistema operativo); las quitadas
    llevan ``duplicate_of`` con la ruta conservada.
    """

    ordered = frame.assign(_key=frame["path"].map(lambda value: PurePath(str(value)).as_posix())).sort_values("_key", kind="stable")
    is_duplicate = ordered.duplicated(subset=[fingerprint_column], keep="first")
    kept = ordered[~is_duplicate]
    owner = dict(zip(kept[fingerprint_column], kept["path"]))
    removed = ordered[is_duplicate].assign(duplicate_of=lambda df: df[fingerprint_column].map(owner))
    return kept.drop(columns="_key").reset_index(drop=True), removed.drop(columns="_key").reset_index(drop=True)

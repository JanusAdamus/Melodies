from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoreRecord:
    piece_id: str
    path: str
    title: str
    composer: str
    canonical_work_id: str
    split: str | None = None
    n_events: int = 0
    n_tokens: int = 0


@dataclass(frozen=True)
class ExclusionRecord:
    piece_id: str
    path: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class PreparedPiece:
    piece_id: str
    source_path: str
    title: str
    composer: str
    canonical_work_id: str
    representation: str
    vocabulary: list[str]
    tokens: list[int]
    n_events: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenSequence:
    piece_id: str
    representation: str
    vocabulary: list[str]
    tokens: list[int]
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowExample:
    piece_id: str
    start_index: int
    stop_index: int
    input_tokens: list[int]
    target_tokens: list[int]
    split: str


@dataclass(frozen=True)
class CorpusPreparationResult:
    pieces: list[PreparedPiece]
    exclusions: list[ExclusionRecord]


@dataclass(frozen=True)
class DatasetBundle:
    train_pieces: list[PreparedPiece]
    validation_pieces: list[PreparedPiece]
    test_pieces: list[PreparedPiece]
    manifest: list[ScoreRecord]
    exclusions: list[ExclusionRecord]
    train_dataset_size: int
    validation_dataset_size: int
    test_dataset_size: int
    stats: dict[str, Any]


@dataclass(frozen=True)
class EvaluationSummary:
    model_name: str
    primary_metric: float
    secondary_metrics: dict[str, float]
    notes: list[str]

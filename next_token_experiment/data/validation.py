from __future__ import annotations

from collections import defaultdict

from ..config import RepresentationConfig, SplitConfig, WindowConfig
from ..schemas import DatasetBundle, PreparedPiece
from .tokenizer import supported_representations


def validate_representation_config(config: RepresentationConfig) -> list[str]:
    issues: list[str] = []
    allowed = set(supported_representations())
    if config.primary not in allowed:
        issues.append("Primary representation is not supported.")
    if config.alternative is not None and config.alternative not in allowed:
        issues.append("Alternative representation is not supported.")
    if tuple(sorted(config.duration_bins)) != config.duration_bins:
        issues.append("Duration bins must be monotonic increasing.")
    return issues


def validate_window_config(config: WindowConfig) -> list[str]:
    issues: list[str] = []
    if config.max_context_length <= 1:
        issues.append("Context length must be greater than 1.")
    if config.train_stride <= 0 or config.eval_stride <= 0:
        issues.append("Strides must be positive.")
    if config.min_window_length > config.max_context_length:
        issues.append("Minimum window length cannot exceed context length.")
    return issues


def validate_split_config(config: SplitConfig) -> list[str]:
    total = config.train_ratio + config.validation_ratio + config.test_ratio
    if abs(total - 1.0) > 1e-9:
        return ["Split ratios must sum to 1.0."]
    return []


def validate_prepared_pieces(pieces: list[PreparedPiece]) -> list[str]:
    issues: list[str] = []
    if not pieces:
        issues.append("No prepared pieces were produced.")
        return issues
    vocabulary = tuple(pieces[0].vocabulary)
    for piece in pieces:
        if not piece.tokens:
            issues.append(f"Piece {piece.piece_id} has no tokens.")
        if tuple(piece.vocabulary) != vocabulary:
            issues.append(f"Piece {piece.piece_id} uses a different vocabulary.")
        if min(piece.tokens) < 0:
            issues.append(f"Piece {piece.piece_id} contains negative token ids.")
        if max(piece.tokens) >= len(vocabulary):
            issues.append(f"Piece {piece.piece_id} contains token ids outside the vocabulary.")
    return issues


def validate_dataset_bundle(bundle: DatasetBundle) -> list[str]:
    issues: list[str] = []
    if bundle.train_dataset_size == 0:
        issues.append("Train split has no windows.")
    if bundle.validation_dataset_size == 0:
        issues.append("Validation split has no windows.")
    if bundle.test_dataset_size == 0:
        issues.append("Test split has no windows.")

    group_to_splits: dict[str, set[str]] = defaultdict(set)
    for record in bundle.manifest:
        if record.split is None:
            issues.append(f"Manifest record {record.piece_id} has no split.")
            continue
        group_to_splits[record.canonical_work_id].add(record.split)
    for canonical_work_id, splits in group_to_splits.items():
        if len(splits) > 1:
            issues.append(f"Canonical work leakage detected for {canonical_work_id}: {sorted(splits)}")
    return issues

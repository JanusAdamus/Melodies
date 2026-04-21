from __future__ import annotations

from dataclasses import asdict
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import ExperimentConfig
from ..schemas import DatasetBundle, PreparedPiece, ScoreRecord, WindowExample
from .tokenizer import SequenceTokenizer


def build_window_slices(
    sequence_length: int,
    max_context_length: int,
    stride: int,
    min_window_length: int,
) -> list[tuple[int, int]]:
    """Return deterministic window boundaries for a token sequence."""

    if sequence_length <= 0:
        return []
    if max_context_length <= 1:
        raise ValueError("max_context_length must be greater than 1.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if min_window_length <= 1:
        raise ValueError("min_window_length must be greater than 1.")

    windows: list[tuple[int, int]] = []
    start = 0
    while start < sequence_length:
        stop = min(start + max_context_length, sequence_length)
        if stop - start >= min_window_length:
            windows.append((start, stop))
        if stop == sequence_length:
            break
        start += stride
    return windows


class WindowedSequenceDataset(Dataset):
    """Dataset of autoregressive windows built from prepared symbolic pieces."""

    def __init__(
        self,
        pieces: list[PreparedPiece],
        tokenizer: SequenceTokenizer,
        split: str,
        max_context_length: int,
        stride: int,
        min_window_length: int,
        max_windows: int | None = None,
    ) -> None:
        self.pieces = list(pieces)
        self.tokenizer = tokenizer
        self.split = split
        self.examples: list[WindowExample] = []

        for piece in self.pieces:
            slices = build_window_slices(
                sequence_length=len(piece.tokens),
                max_context_length=max_context_length,
                stride=stride,
                min_window_length=min_window_length,
            )
            for start_index, stop_index in slices:
                window_tokens = piece.tokens[start_index:stop_index]
                input_tokens, target_tokens = tokenizer.encode_window(window_tokens)
                self.examples.append(
                    WindowExample(
                        piece_id=piece.piece_id,
                        start_index=start_index,
                        stop_index=stop_index,
                        input_tokens=input_tokens,
                        target_tokens=target_tokens,
                        split=split,
                    )
                )

        self.examples.sort(key=lambda item: (item.piece_id, item.start_index))
        if max_windows is not None:
            self.examples = self.examples[:max_windows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "piece_id": example.piece_id,
            "start_index": example.start_index,
            "stop_index": example.stop_index,
            "input_ids": example.input_tokens,
            "target_ids": example.target_tokens,
            "sequence_length": len(example.input_tokens),
            "split": example.split,
        }

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(example) for example in self.examples]


def collate_window_batch(
    batch: list[dict[str, Any]],
    pad_token_id: int,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Pad a batch of variable-length windows for Transformer training."""

    if not batch:
        raise ValueError("Batch cannot be empty.")

    batch_size = len(batch)
    max_length = max(int(item["sequence_length"]) for item in batch)

    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    target_ids = torch.full((batch_size, max_length), ignore_index, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    piece_ids: list[str] = []
    start_indices: list[int] = []
    stop_indices: list[int] = []
    sequence_lengths: list[int] = []
    for row_index, item in enumerate(batch):
        sequence_length = int(item["sequence_length"])
        input_ids[row_index, :sequence_length] = torch.tensor(item["input_ids"], dtype=torch.long)
        target_ids[row_index, :sequence_length] = torch.tensor(item["target_ids"], dtype=torch.long)
        attention_mask[row_index, :sequence_length] = True
        piece_ids.append(str(item["piece_id"]))
        start_indices.append(int(item["start_index"]))
        stop_indices.append(int(item["stop_index"]))
        sequence_lengths.append(sequence_length)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
        "piece_ids": piece_ids,
        "start_indices": start_indices,
        "stop_indices": stop_indices,
        "sequence_lengths": sequence_lengths,
    }


def build_dataset_bundle(
    prepared_pieces: list[PreparedPiece],
    exclusions,
    split_assignments: dict[str, str],
    tokenizer: SequenceTokenizer,
    config: ExperimentConfig,
    max_windows_per_split: dict[str, int] | None = None,
) -> DatasetBundle:
    """Build train/validation/test datasets from prepared pieces."""

    split_to_pieces = {"train": [], "validation": [], "test": []}
    manifest: list[ScoreRecord] = []

    for piece in sorted(prepared_pieces, key=lambda item: item.piece_id):
        split = split_assignments[piece.canonical_work_id]
        split_to_pieces[split].append(piece)
        manifest.append(
            ScoreRecord(
                piece_id=piece.piece_id,
                path=piece.source_path,
                title=piece.title,
                composer=piece.composer,
                canonical_work_id=piece.canonical_work_id,
                split=split,
                n_events=piece.n_events,
                n_tokens=len(piece.tokens),
            )
        )

    max_windows_per_split = max_windows_per_split or {}
    train_dataset = WindowedSequenceDataset(
        pieces=split_to_pieces["train"],
        tokenizer=tokenizer,
        split="train",
        max_context_length=config.windows.max_context_length,
        stride=config.windows.train_stride,
        min_window_length=config.windows.min_window_length,
        max_windows=max_windows_per_split.get("train"),
    )
    validation_dataset = WindowedSequenceDataset(
        pieces=split_to_pieces["validation"],
        tokenizer=tokenizer,
        split="validation",
        max_context_length=config.windows.max_context_length,
        stride=config.windows.eval_stride,
        min_window_length=config.windows.min_window_length,
        max_windows=max_windows_per_split.get("validation"),
    )
    test_dataset = WindowedSequenceDataset(
        pieces=split_to_pieces["test"],
        tokenizer=tokenizer,
        split="test",
        max_context_length=config.windows.max_context_length,
        stride=config.windows.eval_stride,
        min_window_length=config.windows.min_window_length,
        max_windows=max_windows_per_split.get("test"),
    )

    stats = {
        "n_train_pieces": len(split_to_pieces["train"]),
        "n_validation_pieces": len(split_to_pieces["validation"]),
        "n_test_pieces": len(split_to_pieces["test"]),
        "n_train_windows": len(train_dataset),
        "n_validation_windows": len(validation_dataset),
        "n_test_windows": len(test_dataset),
        "n_prepared_pieces": len(prepared_pieces),
        "n_exclusions": len(exclusions),
        "n_total_tokens": sum(len(piece.tokens) for piece in prepared_pieces),
    }

    return DatasetBundle(
        train_pieces=split_to_pieces["train"],
        validation_pieces=split_to_pieces["validation"],
        test_pieces=split_to_pieces["test"],
        manifest=manifest,
        exclusions=list(exclusions),
        train_dataset_size=len(train_dataset),
        validation_dataset_size=len(validation_dataset),
        test_dataset_size=len(test_dataset),
        stats=stats,
    )


def build_dataloaders(
    tokenizer: SequenceTokenizer,
    config: ExperimentConfig,
    train_dataset: WindowedSequenceDataset,
    validation_dataset: WindowedSequenceDataset,
    test_dataset: WindowedSequenceDataset,
) -> dict[str, DataLoader]:
    """Build dataloaders for the Transformer experiment."""

    collate = partial(collate_window_batch, pad_token_id=tokenizer.pad_token_id)
    generator = torch.Generator()
    generator.manual_seed(config.split.seed)
    num_workers = max(0, int(config.hardware.dataloader_workers))
    pin_memory = bool(config.hardware.pin_memory and config.hardware.target_device != "cpu")
    persistent_workers = num_workers > 0

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config.transformer.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate,
            generator=generator,
        ),
        "validation": DataLoader(
            validation_dataset,
            batch_size=config.transformer.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.transformer.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate,
        ),
    }

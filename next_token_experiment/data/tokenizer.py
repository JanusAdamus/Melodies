from __future__ import annotations

from dataclasses import dataclass

from src.data.observations import PITCH_CLASS_NAMES

from ..config import RepresentationConfig
from .preprocess import build_representation_vocabulary

BOS_TOKEN = "<BOS>"
PAD_TOKEN = "<PAD>"


def supported_representations() -> tuple[str, ...]:
    return ("pitch_class", "pitch_class_duration", "event_pitch_duration_metrical")


def estimate_vocabulary_size(config: RepresentationConfig) -> int:
    """Return the expected vocabulary size for the primary representation."""

    if config.primary == "pitch_class":
        return len(PITCH_CLASS_NAMES)
    if config.primary == "pitch_class_duration":
        return len(PITCH_CLASS_NAMES) * len(config.duration_bins)
    if config.primary == "event_pitch_duration_metrical":
        return (len(PITCH_CLASS_NAMES) + 1) * len(config.duration_bins) * len(config.metrical_levels)
    raise ValueError(f"Unsupported representation: {config.primary}")


@dataclass(frozen=True)
class SequenceTokenizer:
    """Tokenizer that adds only the internal special symbols needed by the Transformer."""

    representation: str
    musical_vocabulary: tuple[str, ...]
    bos_token: str = BOS_TOKEN
    pad_token: str = PAD_TOKEN

    @property
    def musical_vocab_size(self) -> int:
        return len(self.musical_vocabulary)

    @property
    def bos_token_id(self) -> int:
        return self.musical_vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.musical_vocab_size + 1

    @property
    def vocab_size(self) -> int:
        return self.musical_vocab_size + 2

    def encode_window(self, musical_tokens: list[int]) -> tuple[list[int], list[int]]:
        """Convert a musical window into autoregressive input/target sequences."""

        if not musical_tokens:
            raise ValueError("Window cannot be empty.")
        return [self.bos_token_id] + musical_tokens[:-1], list(musical_tokens)

    def decode_musical_token(self, token_id: int) -> str:
        if token_id < 0 or token_id >= self.musical_vocab_size:
            raise ValueError(f"Token id out of musical vocabulary range: {token_id}")
        return self.musical_vocabulary[token_id]


def build_tokenizer(config: RepresentationConfig, representation: str | None = None) -> SequenceTokenizer:
    """Build the internal tokenizer used by the Transformer experiment."""

    target = representation or config.primary
    vocabulary = tuple(build_representation_vocabulary(config, target))
    return SequenceTokenizer(
        representation=target,
        musical_vocabulary=vocabulary,
    )


def describe_representation(config: RepresentationConfig, representation: str | None = None) -> dict[str, object]:
    """Return a compact description of the active symbolic representation."""

    tokenizer = build_tokenizer(config, representation=representation)
    return {
        "representation": tokenizer.representation,
        "musical_vocab_size": tokenizer.musical_vocab_size,
        "vocab_size_with_special_tokens": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "duration_bins": list(config.duration_bins),
        "metrical_levels": list(config.metrical_levels),
    }

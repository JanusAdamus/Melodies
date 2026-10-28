"""T1 (auditoría 2026-10-26): con label smoothing, la «NLL de validación» debe ser NLL.

El early stopping y ``best_validation_nll`` usan cross-entropy sin suavizado; el
suavizado sólo entra en la pérdida que se optimiza.
"""

from __future__ import annotations

from dataclasses import replace
import random
import unittest

import torch
from torch import nn

from next_token_experiment.data.dataset import WindowedSequenceDataset, build_dataloaders
from next_token_experiment.data.tokenizer import build_tokenizer
from next_token_experiment.models.small_transformer import SmallTransformerNextTokenModel
from next_token_experiment.tests.test_transformer import build_config, build_spec, make_piece

LABEL_SMOOTHING = 0.2


class PeakedLogits(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*input_ids.shape, self.vocab_size), device=input_ids.device)
        logits[..., 0] = 3.0
        return logits

    def parameter_count(self) -> int:
        return 0


def _smoothed_model(config) -> SmallTransformerNextTokenModel:
    tokenizer = build_tokenizer(config.representation)
    spec = replace(build_spec(config), label_smoothing=LABEL_SMOOTHING)
    return SmallTransformerNextTokenModel(
        spec=spec,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_context_length=config.windows.max_context_length,
        cpu_threads=1,
        seed=7,
    )


def _batch(tokenizer, tokens: list[int]) -> dict[str, object]:
    inputs = torch.tensor([[tokenizer.bos_token_id, *tokens[:-1]]])
    return {
        "input_ids": inputs,
        "target_ids": torch.tensor([tokens]),
        "attention_mask": torch.ones_like(inputs, dtype=torch.bool),
        "piece_ids": ["p"],
        "start_indices": [0],
        "stop_indices": [len(tokens)],
        "sequence_lengths": [len(tokens)],
    }


class ValidationNllWithoutSmoothingTests(unittest.TestCase):
    def test_validation_epoch_reports_true_nll_not_smoothed_loss(self) -> None:
        config = build_config()
        tokenizer = build_tokenizer(config.representation)
        model = _smoothed_model(config)
        model.model = PeakedLogits(tokenizer.vocab_size).to(model.device)
        batch = _batch(tokenizer, [0, 0, 3, 0, 7])

        validation = model._run_epoch([batch], train=False)
        evaluation = model.evaluate([batch])["summary"]

        self.assertAlmostEqual(validation["nll_per_token"], evaluation["nll_per_token"], places=6)

    def test_best_validation_nll_from_fit_equals_evaluated_validation_nll(self) -> None:
        config = build_config()
        config = replace(config, transformer=replace(config.transformer, max_epochs=3))
        tokenizer = build_tokenizer(config.representation)
        rng = random.Random(0)
        pieces = [
            make_piece(f"p{index}", [(start + step) % 12 for step in range(40)])
            for index, start in enumerate(rng.randrange(12) for _ in range(12))
        ]
        datasets = {
            split: WindowedSequenceDataset(
                pieces=subset,
                tokenizer=tokenizer,
                split=split,
                max_context_length=config.windows.max_context_length,
                stride=config.windows.eval_stride,
                min_window_length=config.windows.min_window_length,
            )
            for split, subset in (("train", pieces[:8]), ("validation", pieces[8:10]), ("test", pieces[10:]))
        }
        loaders = build_dataloaders(
            tokenizer=tokenizer,
            config=config,
            train_dataset=datasets["train"],
            validation_dataset=datasets["validation"],
            test_dataset=datasets["test"],
        )
        model = _smoothed_model(config)

        fit = model.fit(loaders["train"], loaders["validation"])
        evaluated = model.evaluate(loaders["validation"])["summary"]["nll_per_token"]

        self.assertAlmostEqual(fit["summary"]["best_validation_nll"], evaluated, places=5)


if __name__ == "__main__":
    unittest.main()

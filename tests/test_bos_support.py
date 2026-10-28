"""F3 (auditoría 2026-10-26): BOS es contexto, nunca un símbolo predecible.

Un modelo que reparte la masa uniformemente sobre las 13 clases (12 alturas más
BOS) debe puntuar ln 12 por evento, no ln 13: al puntuar, la predictiva se
renormaliza sobre los símbolos musicales.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from torch import nn

from Comparacion.classical_models import FiniteGlobalHMM, FiniteHMMFitResult, conditional_log_likelihood
from Comparacion.vomm import VariableOrderMarkovModel
from next_token_experiment.data.tokenizer import build_tokenizer
from next_token_experiment.models.small_transformer import SmallTransformerNextTokenModel
from next_token_experiment.schemas import PreparedPiece
from next_token_experiment.tests.test_transformer import build_config, build_spec

N_MUSICAL = 12
BOS = N_MUSICAL


def _piece(tokens: list[int]) -> PreparedPiece:
    return PreparedPiece(
        piece_id="p",
        source_path="p",
        title="p",
        composer="c",
        canonical_work_id="w",
        representation="pitch_class",
        vocabulary=[str(index) for index in range(N_MUSICAL)],
        tokens=tokens,
        n_events=len(tokens),
        metadata={},
    )


TOKENS = [0, 3, 7, 11, 2, 5]


class ClassicalBosSupportTests(unittest.TestCase):
    def test_uniform_hmm_scores_log_12_not_log_13(self) -> None:
        uniform = np.full((1, N_MUSICAL + 1), 1.0 / (N_MUSICAL + 1))
        score = conditional_log_likelihood(
            initial_probs=np.ones(1),
            transition_matrix=np.ones((1, 1)),
            emission_matrix=uniform,
            context_token=BOS,
            target_tokens=TOKENS,
        )

        self.assertAlmostEqual(-score / len(TOKENS), math.log(N_MUSICAL), places=12)

    def test_uniform_finite_hmm_evaluation_scores_log_12(self) -> None:
        model = FiniteGlobalHMM(candidate_num_states=(1,), max_iterations=1, tolerance=1e-4, seed=1)
        model.initial_probs = np.ones(1)
        model.transition_matrix = np.ones((1, 1))
        model.emission_matrix = np.full((1, N_MUSICAL + 1), 1.0 / (N_MUSICAL + 1))
        model.selected_states = 1
        model.fit_result = FiniteHMMFitResult(
            selected_states=1, validation_nll=0.0, train_log=[], train_wall_clock_s=0.0
        )

        summary = model.evaluate([_piece(TOKENS)], bos_token_id=BOS, max_context_length=4)["summary"]

        self.assertAlmostEqual(summary["test_nll_per_token"], math.log(N_MUSICAL), places=12)

    def test_uniform_vomm_scores_log_12(self) -> None:
        # Sin datos de entrenamiento la VOMM cae al uniforme sobre 13 clases.
        model = VariableOrderMarkovModel(max_order=2, vocabulary_size=N_MUSICAL + 1, bos_token_id=BOS).fit([])

        summary = model.evaluate([_piece(TOKENS)], max_context_length=4)["summary"]

        self.assertAlmostEqual(summary["test_nll_per_token"], math.log(N_MUSICAL), places=12)


class ConstantLogits(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*input_ids.shape, self.vocab_size), device=input_ids.device)

    def parameter_count(self) -> int:
        return 0


class TransformerBosSupportTests(unittest.TestCase):
    def test_uniform_transformer_scores_log_12(self) -> None:
        config = build_config()
        tokenizer = build_tokenizer(config.representation)
        model = SmallTransformerNextTokenModel(
            spec=build_spec(config),
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_context_length=config.windows.max_context_length,
            cpu_threads=1,
            seed=7,
        )
        model.model = ConstantLogits(tokenizer.vocab_size).to(model.device)
        targets = torch.tensor([TOKENS])
        inputs = torch.tensor([[tokenizer.bos_token_id, *TOKENS[:-1]]])
        batch = {
            "input_ids": inputs,
            "target_ids": targets,
            "attention_mask": torch.ones_like(inputs, dtype=torch.bool),
            "piece_ids": ["p"],
            "start_indices": [0],
            "stop_indices": [len(TOKENS)],
            "sequence_lengths": [len(TOKENS)],
        }

        summary = model.evaluate([batch])["summary"]

        self.assertAlmostEqual(summary["nll_per_token"], math.log(N_MUSICAL), places=5)


if __name__ == "__main__":
    unittest.main()

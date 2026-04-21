from __future__ import annotations

import unittest

from next_token_experiment.config import (
    CorpusConfig,
    ExperimentConfig,
    FiniteHMMConfig,
    HDPHMMConfig,
    HardwareConfig,
    MetricsConfig,
    PreprocessingConfig,
    RepresentationConfig,
    SplitConfig,
    StorageConfig,
    TaskConfig,
    TransformerConfig,
    WindowConfig,
)
from next_token_experiment.data.dataset import WindowedSequenceDataset, build_dataloaders
from next_token_experiment.data.tokenizer import build_tokenizer
from next_token_experiment.models.small_transformer import SmallTransformerNextTokenModel, SmallTransformerStudySpec
from next_token_experiment.schemas import PreparedPiece


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskConfig(),
        corpus=CorpusConfig(name="synthetic", root_dir="."),
        preprocessing=PreprocessingConfig(),
        representation=RepresentationConfig(primary="pitch_class", alternative=None),
        windows=WindowConfig(max_context_length=16, min_window_length=8, train_stride=4, eval_stride=16),
        split=SplitConfig(seed=7),
        metrics=MetricsConfig(),
        hardware=HardwareConfig(cpu_threads=1),
        finite_hmm=FiniteHMMConfig(),
        hdp_hmm=HDPHMMConfig(),
        transformer=TransformerConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            ff_dim=128,
            dropout=0.0,
            learning_rate=5e-3,
            batch_size=8,
            max_epochs=8,
            early_stopping_patience=3,
        ),
        storage=StorageConfig(results_root="artifacts/next_token_experiment/results"),
    )


def make_piece(piece_id: str, values: list[int]) -> PreparedPiece:
    return PreparedPiece(
        piece_id=piece_id,
        source_path=piece_id,
        title=piece_id,
        composer="Synthetic",
        canonical_work_id=piece_id,
        representation="pitch_class",
        vocabulary=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        tokens=values,
        n_events=len(values),
        metadata={},
    )


class TransformerTests(unittest.TestCase):
    def test_small_transformer_learns_non_trivial_pattern(self) -> None:
        config = build_config()
        tokenizer = build_tokenizer(config.representation)
        pattern = [0, 2, 4, 5] * 20
        train_dataset = WindowedSequenceDataset(
            pieces=[make_piece("train_1", pattern), make_piece("train_2", pattern)],
            tokenizer=tokenizer,
            split="train",
            max_context_length=config.windows.max_context_length,
            stride=config.windows.train_stride,
            min_window_length=config.windows.min_window_length,
        )
        validation_dataset = WindowedSequenceDataset(
            pieces=[make_piece("validation_1", pattern)],
            tokenizer=tokenizer,
            split="validation",
            max_context_length=config.windows.max_context_length,
            stride=config.windows.eval_stride,
            min_window_length=config.windows.min_window_length,
        )
        test_dataset = WindowedSequenceDataset(
            pieces=[make_piece("test_1", pattern)],
            tokenizer=tokenizer,
            split="test",
            max_context_length=config.windows.max_context_length,
            stride=config.windows.eval_stride,
            min_window_length=config.windows.min_window_length,
        )

        dataloaders = build_dataloaders(
            tokenizer=tokenizer,
            config=config,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )
        spec = SmallTransformerStudySpec(
            architecture=config.transformer.architecture,
            n_layers=config.transformer.n_layers,
            d_model=config.transformer.d_model,
            n_heads=config.transformer.n_heads,
            ff_dim=config.transformer.ff_dim,
            dropout=config.transformer.dropout,
            learning_rate=config.transformer.learning_rate,
            weight_decay=config.transformer.weight_decay,
            batch_size=config.transformer.batch_size,
            max_epochs=config.transformer.max_epochs,
            early_stopping_patience=config.transformer.early_stopping_patience,
            gradient_accumulation_steps=config.transformer.gradient_accumulation_steps,
            grad_clip_norm=config.transformer.grad_clip_norm,
            label_smoothing=config.transformer.label_smoothing,
            lr_scheduler_factor=config.transformer.lr_scheduler_factor,
            lr_scheduler_patience=config.transformer.lr_scheduler_patience,
            min_learning_rate=config.transformer.min_learning_rate,
            tie_input_output_embeddings=config.transformer.tie_input_output_embeddings,
            use_relative_position_bias=config.transformer.use_relative_position_bias,
            relative_attention_num_buckets=config.transformer.relative_attention_num_buckets,
            relative_attention_max_distance=config.transformer.relative_attention_max_distance,
            generation_num_prompts=config.transformer.generation_num_prompts,
            generation_prompt_length=config.transformer.generation_prompt_length,
            generation_max_new_tokens=config.transformer.generation_max_new_tokens,
            generation_temperature=config.transformer.generation_temperature,
            generation_top_k=config.transformer.generation_top_k,
        )
        model = SmallTransformerNextTokenModel(
            spec=spec,
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_context_length=config.windows.max_context_length,
            cpu_threads=1,
            seed=7,
            piece_metadata_by_id={
                "test_1": {"title": "test_1", "composer": "Synthetic", "length_bucket": "medium"}
            },
            token_rarity_by_id={0: "common", 2: "common", 4: "common", 5: "common"},
        )
        fit_result = model.fit(dataloaders["train"], dataloaders["validation"])
        test_result = model.evaluate(dataloaders["test"])
        continuation_a = model.generate(pattern[:8], max_new_tokens=8, temperature=0.8, top_k=4, seed=11, do_sample=True)
        continuation_b = model.generate(pattern[:8], max_new_tokens=8, temperature=0.8, top_k=4, seed=11, do_sample=True)

        self.assertGreaterEqual(fit_result["summary"]["epochs_completed"], 2)
        self.assertLess(fit_result["summary"]["best_validation_nll"], 1.0)
        self.assertGreater(test_result["summary"]["accuracy"], 0.80)
        self.assertGreater(test_result["summary"]["top_3_accuracy"], 0.90)
        self.assertIn("token_rarity", test_result["slice_metrics"])
        self.assertGreater(test_result["summary"]["parameter_count"], 0)
        self.assertEqual(continuation_a, continuation_b)
        self.assertEqual(len(continuation_a), 8)

    def test_relative_position_bias_mode_runs(self) -> None:
        config = build_config()
        config = ExperimentConfig(
            task=config.task,
            corpus=config.corpus,
            preprocessing=config.preprocessing,
            representation=config.representation,
            windows=config.windows,
            split=config.split,
            metrics=config.metrics,
            hardware=config.hardware,
            finite_hmm=config.finite_hmm,
            hdp_hmm=config.hdp_hmm,
            transformer=TransformerConfig(
                d_model=64,
                n_layers=2,
                n_heads=4,
                ff_dim=128,
                dropout=0.0,
                learning_rate=5e-3,
                batch_size=8,
                max_epochs=2,
                early_stopping_patience=2,
                use_relative_position_bias=True,
                relative_attention_num_buckets=16,
                relative_attention_max_distance=64,
            ),
            storage=config.storage,
        )
        tokenizer = build_tokenizer(config.representation)
        pattern = [0, 2, 4, 5] * 12
        train_dataset = WindowedSequenceDataset(
            pieces=[make_piece("train_1", pattern)],
            tokenizer=tokenizer,
            split="train",
            max_context_length=config.windows.max_context_length,
            stride=config.windows.train_stride,
            min_window_length=config.windows.min_window_length,
        )
        validation_dataset = WindowedSequenceDataset(
            pieces=[make_piece("validation_1", pattern)],
            tokenizer=tokenizer,
            split="validation",
            max_context_length=config.windows.max_context_length,
            stride=config.windows.eval_stride,
            min_window_length=config.windows.min_window_length,
        )
        dataloaders = build_dataloaders(
            tokenizer=tokenizer,
            config=config,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=validation_dataset,
        )
        spec = SmallTransformerStudySpec(
            architecture=config.transformer.architecture,
            n_layers=config.transformer.n_layers,
            d_model=config.transformer.d_model,
            n_heads=config.transformer.n_heads,
            ff_dim=config.transformer.ff_dim,
            dropout=config.transformer.dropout,
            learning_rate=config.transformer.learning_rate,
            weight_decay=config.transformer.weight_decay,
            batch_size=config.transformer.batch_size,
            max_epochs=config.transformer.max_epochs,
            early_stopping_patience=config.transformer.early_stopping_patience,
            gradient_accumulation_steps=config.transformer.gradient_accumulation_steps,
            grad_clip_norm=config.transformer.grad_clip_norm,
            label_smoothing=config.transformer.label_smoothing,
            lr_scheduler_factor=config.transformer.lr_scheduler_factor,
            lr_scheduler_patience=config.transformer.lr_scheduler_patience,
            min_learning_rate=config.transformer.min_learning_rate,
            tie_input_output_embeddings=config.transformer.tie_input_output_embeddings,
            use_relative_position_bias=config.transformer.use_relative_position_bias,
            relative_attention_num_buckets=config.transformer.relative_attention_num_buckets,
            relative_attention_max_distance=config.transformer.relative_attention_max_distance,
            generation_num_prompts=config.transformer.generation_num_prompts,
            generation_prompt_length=config.transformer.generation_prompt_length,
            generation_max_new_tokens=config.transformer.generation_max_new_tokens,
            generation_temperature=config.transformer.generation_temperature,
            generation_top_k=config.transformer.generation_top_k,
        )
        model = SmallTransformerNextTokenModel(
            spec=spec,
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_context_length=config.windows.max_context_length,
            cpu_threads=1,
            seed=7,
        )
        fit_result = model.fit(dataloaders["train"], dataloaders["validation"])
        self.assertGreaterEqual(fit_result["summary"]["epochs_completed"], 1)
        self.assertTrue(model.describe_runtime()["use_relative_position_bias"])


if __name__ == "__main__":
    unittest.main()

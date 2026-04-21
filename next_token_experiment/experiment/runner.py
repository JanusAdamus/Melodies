from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import time

from ..config import ExperimentConfig, TransformerConfig
from ..data.dataset import WindowedSequenceDataset, build_dataloaders, build_dataset_bundle
from ..data.preprocess import build_preprocessing_report, prepare_corpus
from ..data.tokenizer import build_tokenizer
from ..data.validation import validate_dataset_bundle, validate_prepared_pieces
from ..experiment.splits import assign_piece_splits
from ..models.small_transformer import SmallTransformerNextTokenModel, SmallTransformerStudySpec
from ..schemas import CorpusPreparationResult, DatasetBundle
from .storage import ensure_directory, write_csv, write_json


def describe_execution_order(config: ExperimentConfig) -> list[str]:
    """Return the implementation and execution order enforced by the protocol."""

    return [
        "freeze_protocol",
        "build_manifest",
        "preprocess_scores",
        "tokenize_sequences",
        "build_windows",
        "validate_data_pipeline",
        "fit_finite_hmm",
        "fit_hdp_hmm",
        "fit_small_transformer",
        "run_smoke_experiment",
        "run_full_comparison",
        "decide_whether_to_scale",
    ]


def _split_datasets(
    preparation: CorpusPreparationResult,
    config: ExperimentConfig,
    max_windows_per_split: dict[str, int] | None = None,
) -> tuple[DatasetBundle, dict[str, WindowedSequenceDataset]]:
    tokenizer = build_tokenizer(config.representation)
    split_assignments = assign_piece_splits(preparation.pieces, config.split)
    bundle = build_dataset_bundle(
        prepared_pieces=preparation.pieces,
        exclusions=preparation.exclusions,
        split_assignments=split_assignments,
        tokenizer=tokenizer,
        config=config,
        max_windows_per_split=max_windows_per_split,
    )

    split_to_pieces = {
        "train": bundle.train_pieces,
        "validation": bundle.validation_pieces,
        "test": bundle.test_pieces,
    }
    datasets = {
        "train": WindowedSequenceDataset(
            pieces=split_to_pieces["train"],
            tokenizer=tokenizer,
            split="train",
            max_context_length=config.windows.max_context_length,
            stride=config.windows.train_stride,
            min_window_length=config.windows.min_window_length,
            max_windows=(max_windows_per_split or {}).get("train"),
        ),
        "validation": WindowedSequenceDataset(
            pieces=split_to_pieces["validation"],
            tokenizer=tokenizer,
            split="validation",
            max_context_length=config.windows.max_context_length,
            stride=config.windows.eval_stride,
            min_window_length=config.windows.min_window_length,
            max_windows=(max_windows_per_split or {}).get("validation"),
        ),
        "test": WindowedSequenceDataset(
            pieces=split_to_pieces["test"],
            tokenizer=tokenizer,
            split="test",
            max_context_length=config.windows.max_context_length,
            stride=config.windows.eval_stride,
            min_window_length=config.windows.min_window_length,
            max_windows=(max_windows_per_split or {}).get("test"),
        ),
    }
    return bundle, datasets


def run_small_transformer_experiment(
    config: ExperimentConfig,
    run_name: str | None = None,
    max_files: int | None = None,
    max_windows_per_split: dict[str, int] | None = None,
    transformer_override: TransformerConfig | None = None,
) -> dict:
    """Run the constrained Transformer experiment and persist its artifacts."""

    effective_config = config if transformer_override is None else replace(config, transformer=transformer_override)
    total_start = time.perf_counter()
    preprocessing_start = time.perf_counter()
    preparation = prepare_corpus(effective_config, max_files=max_files)
    preprocessing_wall_clock_s = time.perf_counter() - preprocessing_start
    preparation_issues = validate_prepared_pieces(preparation.pieces)
    if preparation_issues:
        raise ValueError(f"Prepared corpus validation failed: {preparation_issues}")

    dataset_start = time.perf_counter()
    bundle, datasets = _split_datasets(
        preparation=preparation,
        config=effective_config,
        max_windows_per_split=max_windows_per_split,
    )
    dataset_wall_clock_s = time.perf_counter() - dataset_start
    dataset_issues = validate_dataset_bundle(bundle)
    if dataset_issues:
        raise ValueError(f"Dataset bundle validation failed: {dataset_issues}")

    tokenizer = build_tokenizer(effective_config.representation)
    dataloaders = build_dataloaders(
        tokenizer=tokenizer,
        config=effective_config,
        train_dataset=datasets["train"],
        validation_dataset=datasets["validation"],
        test_dataset=datasets["test"],
    )
    spec = SmallTransformerStudySpec(
        architecture=effective_config.transformer.architecture,
        n_layers=effective_config.transformer.n_layers,
        d_model=effective_config.transformer.d_model,
        n_heads=effective_config.transformer.n_heads,
        ff_dim=effective_config.transformer.ff_dim,
        dropout=effective_config.transformer.dropout,
        learning_rate=effective_config.transformer.learning_rate,
        weight_decay=effective_config.transformer.weight_decay,
        batch_size=effective_config.transformer.batch_size,
        max_epochs=effective_config.transformer.max_epochs,
        early_stopping_patience=effective_config.transformer.early_stopping_patience,
        gradient_accumulation_steps=effective_config.transformer.gradient_accumulation_steps,
        grad_clip_norm=effective_config.transformer.grad_clip_norm,
        label_smoothing=effective_config.transformer.label_smoothing,
        lr_scheduler_factor=effective_config.transformer.lr_scheduler_factor,
        lr_scheduler_patience=effective_config.transformer.lr_scheduler_patience,
        min_learning_rate=effective_config.transformer.min_learning_rate,
        tie_input_output_embeddings=effective_config.transformer.tie_input_output_embeddings,
    )
    model = SmallTransformerNextTokenModel(
        spec=spec,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        max_context_length=effective_config.windows.max_context_length,
        hardware=effective_config.hardware,
        seed=effective_config.split.seed,
    )

    fit_start = time.perf_counter()
    fit_result = model.fit(dataloaders["train"], dataloaders["validation"])
    fit_wall_clock_s = time.perf_counter() - fit_start
    eval_start = time.perf_counter()
    validation_summary = model.evaluate(dataloaders["validation"])
    test_summary = model.evaluate(dataloaders["test"])
    eval_wall_clock_s = time.perf_counter() - eval_start
    total_wall_clock_s = time.perf_counter() - total_start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_name = run_name or f"transformer_run_{timestamp}"
    result_root = ensure_directory(effective_config.storage.results_root) / resolved_run_name
    model_root = ensure_directory(result_root / "transformer")

    write_json(result_root / "config.json", effective_config.to_dict())
    write_json(
        result_root / "data_summary.json",
        {
            "preprocessing": build_preprocessing_report(preparation),
            "dataset": bundle.stats,
            "runtime": model.describe_runtime(),
            "timings": {
                "preprocessing_wall_clock_s": preprocessing_wall_clock_s,
                "dataset_wall_clock_s": dataset_wall_clock_s,
                "fit_wall_clock_s": fit_wall_clock_s,
                "evaluation_wall_clock_s": eval_wall_clock_s,
                "total_wall_clock_s": total_wall_clock_s,
            },
        },
    )
    write_csv(result_root / "split_manifest.csv", [record.__dict__ for record in bundle.manifest])
    write_csv(result_root / "exclusions.csv", [record.__dict__ for record in bundle.exclusions])
    write_csv(model_root / "train_log.csv", fit_result["train_log"])
    write_csv(model_root / "validation_piece_metrics.csv", validation_summary["piece_metrics"])
    write_csv(model_root / "test_piece_metrics.csv", test_summary["piece_metrics"])
    write_json(model_root / "fit_summary.json", fit_result["summary"])
    write_json(model_root / "validation_summary.json", validation_summary["summary"])
    write_json(model_root / "test_summary.json", test_summary["summary"])
    model.save(model_root / "best_model.pt")

    return {
        "run_name": resolved_run_name,
        "result_dir": str(result_root),
        "fit_result": fit_result,
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "dataset_stats": bundle.stats,
        "timings": {
            "preprocessing_wall_clock_s": preprocessing_wall_clock_s,
            "dataset_wall_clock_s": dataset_wall_clock_s,
            "fit_wall_clock_s": fit_wall_clock_s,
            "evaluation_wall_clock_s": eval_wall_clock_s,
            "total_wall_clock_s": total_wall_clock_s,
        },
    }

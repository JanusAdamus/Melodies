from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import os
import time

from ..config import ExperimentConfig, TransformerConfig
from ..data.dataset import WindowedSequenceDataset, build_dataloaders, build_dataset_bundle
from ..data.preprocess import build_preprocessing_report, prepare_corpus
from ..data.tokenizer import build_tokenizer, describe_representation
from ..data.validation import validate_dataset_bundle, validate_prepared_pieces
from ..experiment.splits import assign_piece_splits
from ..models.small_transformer import SmallTransformerNextTokenModel, SmallTransformerStudySpec
from ..schemas import CorpusPreparationResult, DatasetBundle
from .storage import ensure_directory, write_csv, write_json


def _verbose(message: str) -> None:
    if os.environ.get("MELODIES_VERBOSE"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)


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


def _length_bucket(n_tokens: int) -> str:
    if n_tokens < 64:
        return "short"
    if n_tokens < 192:
        return "medium"
    if n_tokens < 512:
        return "long"
    return "very_long"


def _build_piece_metadata(bundle: DatasetBundle) -> dict[str, dict[str, object]]:
    metadata_by_id: dict[str, dict[str, object]] = {}
    for record in bundle.manifest:
        metadata_by_id[record.piece_id] = {
            "piece_id": record.piece_id,
            "title": record.title,
            "composer": record.composer,
            "split": record.split,
            "canonical_work_id": record.canonical_work_id,
            "n_tokens": record.n_tokens,
            "n_events": record.n_events,
            "length_bucket": _length_bucket(int(record.n_tokens)),
        }
    return metadata_by_id


def _build_token_rarity_lookup(pieces) -> dict[int, str]:
    frequency_by_id: dict[int, int] = {}
    for piece in pieces:
        for token in piece.tokens:
            frequency_by_id[int(token)] = frequency_by_id.get(int(token), 0) + 1

    if not frequency_by_id:
        return {}

    counts = sorted(frequency_by_id.values())
    low_cut = counts[max(0, len(counts) // 3 - 1)]
    high_cut = counts[max(0, (2 * len(counts)) // 3 - 1)]

    rarity_by_id: dict[int, str] = {}
    for token_id, count in frequency_by_id.items():
        if count <= low_cut:
            rarity_by_id[token_id] = "rare"
        elif count <= high_cut:
            rarity_by_id[token_id] = "medium"
        else:
            rarity_by_id[token_id] = "common"
    return rarity_by_id


def _decode_tokens(tokenizer, tokens: list[int]) -> list[str]:
    return [tokenizer.decode_musical_token(int(token)) for token in tokens]


def _build_generation_prompts(bundle: DatasetBundle, config: ExperimentConfig) -> list[dict[str, object]]:
    prompts: list[dict[str, object]] = []
    desired_prompt_length = max(4, int(config.transformer.generation_prompt_length))
    desired_num_prompts = max(1, int(config.transformer.generation_num_prompts))
    desired_continuation_length = max(1, int(config.transformer.generation_max_new_tokens))

    for piece in bundle.test_pieces:
        if len(prompts) >= desired_num_prompts:
            break
        if len(piece.tokens) <= desired_prompt_length:
            continue
        prompt_length = min(desired_prompt_length, max(4, len(piece.tokens) // 2))
        target_continuation = piece.tokens[prompt_length : prompt_length + desired_continuation_length]
        prompts.append(
            {
                "piece_id": piece.piece_id,
                "title": piece.title,
                "composer": piece.composer,
                "prompt_tokens": piece.tokens[:prompt_length],
                "target_continuation_tokens": target_continuation,
            }
        )
    return prompts


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
    _verbose(f"Starting corpus preprocessing: root={effective_config.corpus.root_dir}, max_files={max_files}")
    preparation = prepare_corpus(effective_config, max_files=max_files)
    preprocessing_wall_clock_s = time.perf_counter() - preprocessing_start
    _verbose(
        "Finished corpus preprocessing: "
        f"pieces={len(preparation.pieces)}, exclusions={len(preparation.exclusions)}, "
        f"seconds={preprocessing_wall_clock_s:.2f}"
    )
    preparation_issues = validate_prepared_pieces(preparation.pieces)
    if preparation_issues:
        raise ValueError(f"Prepared corpus validation failed: {preparation_issues}")

    dataset_start = time.perf_counter()
    _verbose(f"Building datasets and windows: limits={max_windows_per_split}")
    bundle, datasets = _split_datasets(
        preparation=preparation,
        config=effective_config,
        max_windows_per_split=max_windows_per_split,
    )
    dataset_wall_clock_s = time.perf_counter() - dataset_start
    _verbose(
        "Finished datasets: "
        f"train_windows={bundle.train_dataset_size}, "
        f"validation_windows={bundle.validation_dataset_size}, "
        f"test_windows={bundle.test_dataset_size}, seconds={dataset_wall_clock_s:.2f}"
    )
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
        attention_implementation=effective_config.transformer.attention_implementation,
        use_relative_position_bias=effective_config.transformer.use_relative_position_bias,
        relative_attention_num_buckets=effective_config.transformer.relative_attention_num_buckets,
        relative_attention_max_distance=effective_config.transformer.relative_attention_max_distance,
        generation_num_prompts=effective_config.transformer.generation_num_prompts,
        generation_prompt_length=effective_config.transformer.generation_prompt_length,
        generation_max_new_tokens=effective_config.transformer.generation_max_new_tokens,
        generation_temperature=effective_config.transformer.generation_temperature,
        generation_top_k=effective_config.transformer.generation_top_k,
    )
    piece_metadata_by_id = _build_piece_metadata(bundle)
    token_rarity_by_id = _build_token_rarity_lookup(bundle.train_pieces)
    model = SmallTransformerNextTokenModel(
        spec=spec,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_context_length=effective_config.windows.max_context_length,
        hardware=effective_config.hardware,
        seed=effective_config.split.seed,
        piece_metadata_by_id=piece_metadata_by_id,
        token_rarity_by_id=token_rarity_by_id,
    )

    fit_start = time.perf_counter()
    _verbose("Starting transformer fit")
    fit_result = model.fit(dataloaders["train"], dataloaders["validation"])
    fit_wall_clock_s = time.perf_counter() - fit_start
    _verbose(f"Finished transformer fit: seconds={fit_wall_clock_s:.2f}")
    eval_start = time.perf_counter()
    _verbose("Starting validation/test evaluation")
    validation_summary = model.evaluate(dataloaders["validation"])
    test_summary = model.evaluate(dataloaders["test"])
    eval_wall_clock_s = time.perf_counter() - eval_start
    _verbose(f"Finished evaluation: seconds={eval_wall_clock_s:.2f}")
    prompts = _build_generation_prompts(bundle, effective_config)
    generated_continuations = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        prompt_tokens = list(prompt["prompt_tokens"])
        greedy_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=effective_config.transformer.generation_max_new_tokens,
            temperature=0.0,
            top_k=None,
            seed=effective_config.split.seed + prompt_index,
            do_sample=False,
        )
        sampled_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=effective_config.transformer.generation_max_new_tokens,
            temperature=effective_config.transformer.generation_temperature,
            top_k=effective_config.transformer.generation_top_k,
            seed=effective_config.split.seed + prompt_index,
            do_sample=True,
        )
        generated_continuations.append(
            {
                "piece_id": prompt["piece_id"],
                "title": prompt["title"],
                "composer": prompt["composer"],
                "prompt_tokens": prompt_tokens,
                "prompt_decoded": _decode_tokens(tokenizer, prompt_tokens),
                "target_continuation_tokens": list(prompt["target_continuation_tokens"]),
                "target_continuation_decoded": _decode_tokens(tokenizer, list(prompt["target_continuation_tokens"])),
                "greedy_continuation_tokens": greedy_tokens,
                "greedy_continuation_decoded": _decode_tokens(tokenizer, greedy_tokens),
                "sampled_continuation_tokens": sampled_tokens,
                "sampled_continuation_decoded": _decode_tokens(tokenizer, sampled_tokens),
                "sampling_seed": effective_config.split.seed + prompt_index,
                "sampling_temperature": effective_config.transformer.generation_temperature,
                "sampling_top_k": effective_config.transformer.generation_top_k,
            }
        )
    total_wall_clock_s = time.perf_counter() - total_start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_name = run_name or f"transformer_run_{timestamp}"
    result_root = ensure_directory(effective_config.storage.results_root) / resolved_run_name
    model_root = ensure_directory(result_root / "transformer")

    write_json(result_root / "config.json", effective_config.to_dict())
    write_json(
        result_root / "data_summary.json",
        {
            "execution_order": describe_execution_order(effective_config),
            "preprocessing": build_preprocessing_report(preparation),
            "dataset": bundle.stats,
            "representation": describe_representation(effective_config.representation),
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
    write_json(model_root / "validation_slice_metrics.json", validation_summary["slice_metrics"])
    write_json(model_root / "test_slice_metrics.json", test_summary["slice_metrics"])
    write_json(model_root / "generated_continuations.json", {"samples": generated_continuations})
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
        "generated_continuations": generated_continuations,
        "dataset_stats": bundle.stats,
        "timings": {
            "preprocessing_wall_clock_s": preprocessing_wall_clock_s,
            "dataset_wall_clock_s": dataset_wall_clock_s,
            "fit_wall_clock_s": fit_wall_clock_s,
            "evaluation_wall_clock_s": eval_wall_clock_s,
            "total_wall_clock_s": total_wall_clock_s,
        },
    }

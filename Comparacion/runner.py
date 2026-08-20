from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

from next_token_experiment.data.dataset import WindowedSequenceDataset, build_dataloaders
from next_token_experiment.data.preprocess import build_preprocessing_report, prepare_corpus
from next_token_experiment.data.tokenizer import build_tokenizer
from next_token_experiment.experiment.storage import ensure_directory, write_csv, write_json
from next_token_experiment.models.small_transformer import SmallTransformerNextTokenModel, SmallTransformerStudySpec

from .classical_models import BOS_TOKEN, FiniteGlobalHMM, GlobalHDPHMM
from .config import LearningCurveConfig
from .splits import build_fixed_splits, build_nested_training_subsets


def _bos_token_id() -> int:
    return 12


def _build_transformer_model(config: LearningCurveConfig, model_seed: int) -> SmallTransformerNextTokenModel:
    tokenizer = build_tokenizer(config.experiment.representation)
    transformer = config.experiment.transformer
    spec = SmallTransformerStudySpec(
        architecture=transformer.architecture,
        n_layers=transformer.n_layers,
        d_model=transformer.d_model,
        n_heads=transformer.n_heads,
        ff_dim=transformer.ff_dim,
        dropout=transformer.dropout,
        learning_rate=transformer.learning_rate,
        weight_decay=transformer.weight_decay,
        batch_size=transformer.batch_size,
        max_epochs=transformer.max_epochs,
        early_stopping_patience=transformer.early_stopping_patience,
        gradient_accumulation_steps=transformer.gradient_accumulation_steps,
        grad_clip_norm=transformer.grad_clip_norm,
        label_smoothing=transformer.label_smoothing,
        lr_scheduler_factor=transformer.lr_scheduler_factor,
        lr_scheduler_patience=transformer.lr_scheduler_patience,
        min_learning_rate=transformer.min_learning_rate,
        tie_input_output_embeddings=transformer.tie_input_output_embeddings,
        attention_implementation=transformer.attention_implementation,
        use_relative_position_bias=transformer.use_relative_position_bias,
        relative_attention_num_buckets=transformer.relative_attention_num_buckets,
        relative_attention_max_distance=transformer.relative_attention_max_distance,
        generation_num_prompts=1,
        generation_prompt_length=8,
        generation_max_new_tokens=8,
        generation_temperature=1.0,
        generation_top_k=4,
    )
    return SmallTransformerNextTokenModel(
        spec=spec,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_context_length=config.experiment.windows.max_context_length,
        hardware=config.experiment.hardware,
        seed=model_seed,
    )


def _build_transformer_dataloaders(config: LearningCurveConfig, train_pieces, validation_pieces, test_pieces):
    tokenizer = build_tokenizer(config.experiment.representation)
    train_dataset = WindowedSequenceDataset(
        pieces=train_pieces,
        tokenizer=tokenizer,
        split="train",
        max_context_length=config.experiment.windows.max_context_length,
        stride=config.experiment.windows.train_stride,
        min_window_length=config.experiment.windows.min_window_length,
    )
    validation_dataset = WindowedSequenceDataset(
        pieces=validation_pieces,
        tokenizer=tokenizer,
        split="validation",
        max_context_length=config.experiment.windows.max_context_length,
        stride=config.experiment.windows.eval_stride,
        min_window_length=config.experiment.windows.min_window_length,
    )
    test_dataset = WindowedSequenceDataset(
        pieces=test_pieces,
        tokenizer=tokenizer,
        split="test",
        max_context_length=config.experiment.windows.max_context_length,
        stride=config.experiment.windows.eval_stride,
        min_window_length=config.experiment.windows.min_window_length,
    )
    return build_dataloaders(
        tokenizer=tokenizer,
        config=config.experiment,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
    )


def _transformer_summary_row(fit_result: dict, eval_result: dict) -> dict[str, object]:
    summary = eval_result["summary"]
    fit_summary = fit_result["summary"]
    return {
        "validation_nll_per_token": fit_summary["best_validation_nll"],
        "test_nll_per_token": summary["nll_per_token"],
        "test_perplexity": summary["perplexity"],
        "n_params": summary["parameter_count"],
        "train_time_sec": sum(row["train_wall_clock_s"] + row["validation_wall_clock_s"] for row in fit_result["train_log"]),
    }


def _build_piece_metric_rows(model_name: str, fraction: float, data_seed: int, model_seed: int, piece_metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for item in piece_metrics:
        row = dict(item)
        row["model"] = model_name
        row["frac"] = fraction
        row["data_seed"] = data_seed
        row["model_seed"] = model_seed
        rows.append(row)
    return rows


def _aggregate_summary(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault((str(row["model"]), float(row["frac"])), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (model, fraction), rows in sorted(grouped.items()):
        perplexities = [float(row["test_ppl"]) for row in rows]
        nlls = [float(row["test_nll"]) for row in rows]
        train_tokens = [int(row["n_train_tokens"]) for row in rows]
        summary_rows.append(
            {
                "model": model,
                "frac": fraction,
                "mean_n_train_tokens": mean(train_tokens),
                "mean_test_ppl": mean(perplexities),
                "std_test_ppl": pstdev(perplexities) if len(perplexities) > 1 else 0.0,
                "mean_test_nll": mean(nlls),
                "std_test_nll": pstdev(nlls) if len(nlls) > 1 else 0.0,
                "runs": len(rows),
            }
        )
    return summary_rows


def _plot_learning_curve(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    if not summary_rows:
        return

    plt.figure(figsize=(8, 5))
    frame = pd.DataFrame(summary_rows)
    for model_name, group in frame.groupby("model"):
        group = group.sort_values("mean_n_train_tokens")
        plt.plot(group["mean_n_train_tokens"], group["mean_test_ppl"], marker="o", label=model_name)
        plt.fill_between(
            group["mean_n_train_tokens"],
            group["mean_test_ppl"] - group["std_test_ppl"],
            group["mean_test_ppl"] + group["std_test_ppl"],
            alpha=0.2,
        )
    plt.xscale("log")
    plt.xlabel("Tokens de entrenamiento")
    plt.ylabel("Perplejidad de test")
    plt.title("Curva de aprendizaje HMM vs HDP-HMM vs Transformer")
    plt.legend()
    plt.tight_layout()
    ensure_directory(output_path.parent)
    plt.savefig(output_path)
    plt.close()


def _compute_wilcoxon(piece_metric_rows: list[dict[str, object]], output_path: Path) -> dict[str, object]:
    candidates = [
        row for row in piece_metric_rows
        if row["model"] in {"hdp_hmm", "transformer"} and math.isclose(float(row["frac"]), 1.0, rel_tol=0.0, abs_tol=1e-9)
    ]
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in candidates:
        grouped.setdefault((str(row["model"]), str(row["piece_id"])), []).append(float(row["nll_per_token"]))

    common_piece_ids = sorted(
        set(piece_id for model, piece_id in grouped if model == "hdp_hmm")
        & set(piece_id for model, piece_id in grouped if model == "transformer")
    )
    hdp_values = [mean(grouped[("hdp_hmm", piece_id)]) for piece_id in common_piece_ids]
    transformer_values = [mean(grouped[("transformer", piece_id)]) for piece_id in common_piece_ids]

    if len(common_piece_ids) < 2:
        payload = {"status": "insufficient_pairs", "n_pairs": len(common_piece_ids)}
        write_json(output_path, payload)
        return payload

    statistic, p_value = wilcoxon(hdp_values, transformer_values)
    payload = {
        "status": "ok",
        "n_pairs": len(common_piece_ids),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "direction": "transformer_better" if mean(transformer_values) < mean(hdp_values) else "hdp_hmm_better",
        "piece_ids": common_piece_ids,
    }
    write_json(output_path, payload)
    return payload


def run_learning_curve_experiment(
    config: LearningCurveConfig,
    *,
    run_name: str = "learning_curve",
    max_files: int | None = None,
) -> dict[str, object]:
    output_root = ensure_directory(Path(config.results_root) / run_name)
    preparation = prepare_corpus(config.experiment, max_files=max_files)
    fixed_splits = build_fixed_splits(
        preparation.pieces,
        test_fraction=config.test_fraction,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )

    write_json(output_root / "config.json", json.loads(json.dumps(asdict(config))))
    write_json(output_root / "preprocessing_report.json", build_preprocessing_report(preparation))
    write_csv(
        output_root / "exclusions.csv",
        [item.__dict__ for item in preparation.exclusions],
    )
    split_root = ensure_directory(output_root / "splits")
    write_json(split_root / "test_pieces.json", {"piece_ids": [piece.piece_id for piece in fixed_splits.test_pieces]})
    write_json(split_root / "val_pieces.json", {"piece_ids": [piece.piece_id for piece in fixed_splits.validation_pieces]})

    raw_rows: list[dict[str, object]] = []
    piece_metric_rows: list[dict[str, object]] = []

    for data_seed in config.data_seeds:
        nested_subsets = build_nested_training_subsets(
            fixed_splits.train_pool_pieces,
            fractions=config.train_fractions,
            data_seed=data_seed,
        )
        write_json(
            split_root / f"train_fractions_seed{data_seed}.json",
            {
                "fractions": [
                    {"frac": fraction, "piece_ids": [piece.piece_id for piece in subset]}
                    for fraction, subset in nested_subsets
                ]
            },
        )

        for fraction, train_pieces in nested_subsets:
            n_train_tokens = sum(len(piece.tokens) for piece in train_pieces)
            n_train_pieces = len(train_pieces)
            for model_seed in config.model_seeds:
                finite_hmm = FiniteGlobalHMM(
                    candidate_num_states=config.finite_hmm_states,
                    max_iterations=config.finite_hmm_max_iterations,
                    tolerance=config.finite_hmm_tolerance,
                    seed=model_seed,
                )
                finite_hmm.fit(train_pieces, fixed_splits.validation_pieces, bos_token_id=_bos_token_id())
                finite_eval = finite_hmm.evaluate(fixed_splits.test_pieces, bos_token_id=_bos_token_id())
                finite_summary = finite_eval["summary"]
                raw_rows.append(
                    {
                        "model": "finite_hmm",
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "frac": fraction,
                        "n_train_pieces": n_train_pieces,
                        "n_train_tokens": n_train_tokens,
                        "n_params": finite_summary["n_params"],
                        "val_ppl": math.exp(float(finite_summary["validation_nll_per_token"])),
                        "test_ppl": finite_summary["test_perplexity"],
                        "test_nll": finite_summary["test_nll_per_token"],
                        "train_time_sec": finite_summary["train_time_sec"],
                        "hyperparams_json": json.dumps({"selected_states": finite_summary["selected_states"]}),
                    }
                )
                piece_metric_rows.extend(
                    _build_piece_metric_rows("finite_hmm", fraction, data_seed, model_seed, finite_eval["piece_metrics"])
                )

                hdp_hmm = GlobalHDPHMM(
                    truncation_level=config.hdp_truncation_level,
                    n_iters=config.hdp_n_iters,
                    burn_in=config.hdp_burn_in,
                    hyperparameter_grid=config.hdp_hyperparameter_grid,
                    seed=model_seed,
                )
                hdp_fit = hdp_hmm.fit(train_pieces, fixed_splits.validation_pieces, bos_token_id=_bos_token_id())
                hdp_eval = hdp_hmm.evaluate(fixed_splits.test_pieces, bos_token_id=_bos_token_id())
                hdp_summary = hdp_eval["summary"]
                raw_rows.append(
                    {
                        "model": "hdp_hmm",
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "frac": fraction,
                        "n_train_pieces": n_train_pieces,
                        "n_train_tokens": n_train_tokens,
                        "n_params": hdp_summary["n_params"],
                        "val_ppl": math.exp(float(hdp_summary["validation_nll_per_token"])),
                        "test_ppl": hdp_summary["test_perplexity"],
                        "test_nll": hdp_summary["test_nll_per_token"],
                        "train_time_sec": hdp_summary["train_time_sec"],
                        "hyperparams_json": json.dumps(hdp_fit["selected_hyperparameters"]),
                    }
                )
                piece_metric_rows.extend(
                    _build_piece_metric_rows("hdp_hmm", fraction, data_seed, model_seed, hdp_eval["piece_metrics"])
                )

                transformer_model = _build_transformer_model(config, model_seed)
                dataloaders = _build_transformer_dataloaders(
                    config,
                    train_pieces=train_pieces,
                    validation_pieces=fixed_splits.validation_pieces,
                    test_pieces=fixed_splits.test_pieces,
                )
                transformer_fit = transformer_model.fit(dataloaders["train"], dataloaders["validation"])
                transformer_eval = transformer_model.evaluate(dataloaders["test"])
                transformer_summary = _transformer_summary_row(transformer_fit, transformer_eval)
                raw_rows.append(
                    {
                        "model": "transformer",
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "frac": fraction,
                        "n_train_pieces": n_train_pieces,
                        "n_train_tokens": n_train_tokens,
                        "n_params": transformer_summary["n_params"],
                        "val_ppl": math.exp(float(transformer_summary["validation_nll_per_token"])),
                        "test_ppl": transformer_summary["test_perplexity"],
                        "test_nll": transformer_summary["test_nll_per_token"],
                        "train_time_sec": transformer_summary["train_time_sec"],
                        "hyperparams_json": json.dumps({"architecture": config.experiment.transformer.architecture}),
                    }
                )
                piece_metric_rows.extend(
                    _build_piece_metric_rows("transformer", fraction, data_seed, model_seed, transformer_eval["piece_metrics"])
                )

    summary_rows = _aggregate_summary(raw_rows)
    write_csv(output_root / "results_raw.csv", raw_rows)
    write_csv(output_root / "results_summary.csv", summary_rows)
    write_csv(output_root / "piece_metrics_raw.csv", piece_metric_rows)
    _plot_learning_curve(summary_rows, output_root / "learning_curve.png")
    wilcoxon_payload = _compute_wilcoxon(piece_metric_rows, output_root / "wilcoxon_test.json")

    return {
        "output_root": str(output_root),
        "n_prepared_pieces": len(preparation.pieces),
        "n_exclusions": len(preparation.exclusions),
        "n_runs": len(raw_rows),
        "wilcoxon": wilcoxon_payload,
    }

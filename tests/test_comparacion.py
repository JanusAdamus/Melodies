from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from Comparacion.classical_models import FiniteGlobalHMM, FiniteHMMFitResult, GlobalHDPHMM
from Comparacion.cli import build_parser
from Comparacion.config import LearningCurveConfig, build_default_learning_curve_config
from Comparacion.runner import _build_pareto_summary, run_learning_curve_experiment
from Comparacion.splits import build_fixed_splits, build_nested_training_subsets
from next_token_experiment.schemas import CorpusPreparationResult, PreparedPiece
from src.models.inference import forward_log_likelihood


def _piece(piece_id: str, token: int, length: int, source: str) -> PreparedPiece:
    return PreparedPiece(
        piece_id=piece_id,
        source_path=f"/tmp/{source}/{piece_id}.musicxml",
        title=piece_id,
        composer="composer",
        canonical_work_id=piece_id,
        representation="pitch_class",
        vocabulary=[str(index) for index in range(12)],
        tokens=[token for _ in range(length)],
        n_events=length,
        metadata={},
    )


def _synthetic_preparation(n_pieces: int = 12, length: int = 5) -> CorpusPreparationResult:
    pieces = [
        replace(
            _piece(f"piece-{index}", index % 12, length + (index % 3), "Synthetic"),
            canonical_work_id=f"canonical-work-{index}",
        )
        for index in range(n_pieces)
    ]
    return CorpusPreparationResult(pieces=pieces, exclusions=[])


def _runner_config(
    results_root: str,
    *,
    include_vomm_control: bool = True,
    structural_annotations_path: str | None = None,
) -> LearningCurveConfig:
    config = build_default_learning_curve_config(
        corpus_root="synthetic-corpus",
        results_root=results_root,
    )
    experiment = replace(
        config.experiment,
        windows=replace(
            config.experiment.windows,
            max_context_length=3,
            min_window_length=2,
            train_stride=2,
            eval_stride=3,
        ),
        hardware=replace(config.experiment.hardware, cpu_threads=1),
        transformer=replace(
            config.experiment.transformer,
            d_model=8,
            n_layers=1,
            n_heads=1,
            ff_dim=16,
            batch_size=2,
            max_epochs=1,
            early_stopping_patience=1,
        ),
    )
    return replace(
        config,
        experiment=experiment,
        train_fractions=(1.0,),
        data_seeds=(11,),
        model_seeds=(13,),
        finite_hmm_states=(2,),
        finite_hmm_max_iterations=1,
        hdp_truncation_level=2,
        hdp_n_iters=1,
        hdp_burn_in=0,
        hdp_hyperparameter_grid=((1.0, 1.0, 1.0),),
        include_vomm_control=include_vomm_control,
        vomm_candidate_orders=(1, 2),
        bootstrap_samples=8,
        bootstrap_seed=19,
        boundary_tolerance=1,
        structural_annotations_path=structural_annotations_path,
    )


def _piece_metrics(
    pieces: list[PreparedPiece],
    *,
    nll: float,
    scored_indices_by_piece: dict[str, list[int]] | None = None,
) -> list[dict[str, object]]:
    scored_indices_by_piece = scored_indices_by_piece or {}
    return [
        {
            "piece_id": piece.piece_id,
            "title": piece.title,
            "composer": piece.composer,
            "n_tokens": len(piece.tokens),
            "nll_per_token": nll,
            "perplexity": math.exp(nll),
            "accuracy": 1.0,
            "brier_score": 0.0,
            "scored_event_indices": scored_indices_by_piece.get(
                piece.piece_id,
                list(range(len(piece.tokens))),
            ),
        }
        for piece in pieces
    ]


class ComparisonSplitTests(unittest.TestCase):
    def test_fixed_splits_and_nested_subsets_are_deterministic(self) -> None:
        pieces = []
        for index in range(6):
            pieces.append(_piece(f"mt_short_{index}", index % 12, 40, "MuseTrainer"))
            pieces.append(_piece(f"symbtr_long_{index}", (index + 1) % 12, 140, "SymbTr"))

        splits = build_fixed_splits(pieces, test_fraction=0.15, validation_fraction=0.10, seed=7)
        self.assertTrue(splits.test_pieces)
        self.assertTrue(splits.validation_pieces)
        self.assertTrue(splits.train_pool_pieces)

        nested = build_nested_training_subsets(splits.train_pool_pieces, fractions=(0.25, 0.50, 1.0), data_seed=3)
        subset_ids = [set(piece.piece_id for piece in subset) for _, subset in nested]
        self.assertTrue(subset_ids[0].issubset(subset_ids[1]))
        self.assertTrue(subset_ids[1].issubset(subset_ids[2]))


class ComparisonConfigAndCliTests(unittest.TestCase):
    def test_multidimensional_defaults_validate_and_serialize(self) -> None:
        self.assertIn(
            "include_vomm_control",
            LearningCurveConfig.__dataclass_fields__,
            "Task 4 multidimensional configuration fields must exist",
        )
        config = build_default_learning_curve_config()

        self.assertTrue(config.include_vomm_control)
        self.assertEqual(config.vomm_candidate_orders, (1, 2, 4, 8))
        self.assertEqual(config.bootstrap_samples, 10000)
        self.assertEqual(config.bootstrap_seed, 17)
        self.assertEqual(config.boundary_tolerance, 1)
        self.assertIsNone(config.structural_annotations_path)
        payload = json.loads(json.dumps(asdict(config), allow_nan=False))
        self.assertEqual(payload["vomm_candidate_orders"], [1, 2, 4, 8])

        invalid_updates = (
            {"include_vomm_control": 1},
            {"vomm_candidate_orders": ()},
            {"vomm_candidate_orders": (1, 1)},
            {"vomm_candidate_orders": (1, -1)},
            {"bootstrap_samples": 0},
            {"bootstrap_samples": 1.5},
            {"bootstrap_seed": -1},
            {"boundary_tolerance": -1},
            {"structural_annotations_path": "   "},
        )
        for updates in invalid_updates:
            with self.subTest(updates=updates):
                with self.assertRaises((TypeError, ValueError)):
                    replace(config, **updates)

    def test_cli_parses_plan_vomm_and_structural_annotation_flags(self) -> None:
        args, unknown = build_parser().parse_known_args(
            [
                "--plan-only",
                "--without-vomm",
                "--structural-annotations",
                "annotations.csv",
            ]
        )

        self.assertEqual(unknown, [])
        self.assertTrue(args.plan_only)
        self.assertTrue(args.without_vomm)
        self.assertEqual(args.structural_annotations, "annotations.csv")


class ClassicalModelTests(unittest.TestCase):
    def test_finite_hmm_retains_matrices_for_selected_state_count(self) -> None:
        class DeterministicFiniteHMM(FiniteGlobalHMM):
            def _fit_candidate(self, *args, n_states: int, **kwargs):
                initial_probs = np.full(n_states, 1.0 / n_states)
                transition_matrix = np.eye(n_states)
                emission_matrix = np.full((n_states, self.vocab_size), 1.0 / self.vocab_size)
                self.initial_probs = initial_probs
                self.transition_matrix = transition_matrix
                self.emission_matrix = emission_matrix
                validation_nll = 0.1 if n_states == 2 else 0.9
                return (
                    initial_probs,
                    transition_matrix,
                    emission_matrix,
                    validation_nll,
                    [{"n_states": n_states, "validation_nll_per_token": validation_nll}],
                )

        model = DeterministicFiniteHMM(
            candidate_num_states=(2, 3),
            max_iterations=1,
            tolerance=1e-4,
            seed=1,
        )
        fit_result = model.fit(
            [_piece("train", 0, 2, "MuseTrainer")],
            [_piece("validation", 0, 2, "SymbTr")],
            bos_token_id=12,
        )

        self.assertEqual(fit_result.selected_states, 2)
        self.assertEqual(model.initial_probs.shape, (2,))
        self.assertEqual(model.transition_matrix.shape, (2, 2))
        self.assertEqual(model.emission_matrix.shape, (2, 13))

    def test_classical_score_conditions_on_bos_without_counting_it(self) -> None:
        train_pieces = [_piece("train", 0, 4, "MuseTrainer")]
        validation_pieces = [_piece("validation", 0, 2, "MuseTrainer")]
        test_pieces = [_piece("test", 0, 2, "SymbTr")]
        model = FiniteGlobalHMM(candidate_num_states=(1,), max_iterations=2, tolerance=1e-4, seed=1)

        try:
            model.fit(train_pieces, validation_pieces, bos_token_id=12, max_context_length=128)
            evaluation = model.evaluate(test_pieces, bos_token_id=12, max_context_length=128)
        except TypeError as error:
            self.fail(str(error))

        expected_score = 2 * math.log(float(model.emission_matrix[0, 0]) + 1e-12)
        self.assertEqual(evaluation["summary"]["n_tokens"], 2)
        self.assertAlmostEqual(evaluation["piece_metrics"][0]["log_likelihood"], expected_score)

    def test_classical_score_resets_bos_at_evaluation_slice_boundaries(self) -> None:
        model = FiniteGlobalHMM(candidate_num_states=(2,), max_iterations=1, tolerance=1e-4, seed=1)
        model.initial_probs = np.array([0.6, 0.4])
        model.transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        model.emission_matrix = np.array(
            [
                [0.4, 0.01, *([0.019] * 10), 0.4],
                [0.01, 0.6, *([0.029] * 10), 0.1],
            ]
        )
        model.selected_states = 2
        model.fit_result = FiniteHMMFitResult(
            selected_states=2,
            validation_nll=0.0,
            train_log=[],
            train_wall_clock_s=0.0,
        )
        piece = _piece("test", 0, 3, "SymbTr")
        piece.tokens[1:] = [1, 1]

        def joint_log_likelihood(tokens: list[int]) -> float:
            value, _ = forward_log_likelihood(
                initial_probs=model.initial_probs,
                transition_matrix=model.transition_matrix,
                emission_matrix=model.emission_matrix,
                observations=np.array(tokens),
            )
            return value

        context_score = joint_log_likelihood([12])
        expected_score = (
            joint_log_likelihood([12, 0, 1])
            - context_score
            + joint_log_likelihood([12, 1])
            - context_score
        )
        evaluation = model.evaluate([piece], bos_token_id=12, max_context_length=2)

        self.assertEqual(evaluation["summary"]["n_tokens"], 3)
        self.assertAlmostEqual(evaluation["piece_metrics"][0]["log_likelihood"], expected_score)
        self.assertIn("scored_event_indices", evaluation["piece_metrics"][0])
        self.assertEqual(evaluation["piece_metrics"][0]["scored_event_indices"], [0, 1, 2])

    def test_finite_hmm_fits_and_scores_synthetic_sequences(self) -> None:
        train_pieces = [_piece("a", 0, 12, "MuseTrainer"), _piece("b", 0, 12, "SymbTr")]
        validation_pieces = [_piece("c", 0, 10, "MuseTrainer")]
        test_pieces = [_piece("d", 0, 10, "SymbTr")]

        model = FiniteGlobalHMM(candidate_num_states=(2, 3), max_iterations=10, tolerance=1e-4, seed=1)
        fit_result = model.fit(train_pieces, validation_pieces, bos_token_id=12)
        evaluation = model.evaluate(test_pieces, bos_token_id=12)

        self.assertIn(fit_result.selected_states, {2, 3})
        self.assertLess(evaluation["summary"]["test_perplexity"], 2.0)
        self.assertEqual(len(evaluation["piece_metrics"]), 1)

    def test_hdp_hmm_fits_smoke_sequence(self) -> None:
        train_pieces = [_piece("a", 4, 14, "MuseTrainer"), _piece("b", 4, 14, "SymbTr")]
        validation_pieces = [_piece("c", 4, 10, "MuseTrainer")]
        test_pieces = [_piece("d", 4, 10, "SymbTr")]

        model = GlobalHDPHMM(
            truncation_level=4,
            n_iters=8,
            burn_in=4,
            hyperparameter_grid=((8.0, 4.0, 2.0),),
            seed=2,
        )
        fit_result = model.fit(train_pieces, validation_pieces, bos_token_id=12)
        evaluation = model.evaluate(test_pieces, bos_token_id=12)

        self.assertIn("validation_nll_per_token", fit_result)
        self.assertLess(evaluation["summary"]["test_perplexity"], 3.0)

    def test_hdp_hmm_retains_result_for_selected_hyperparameters(self) -> None:
        class DeterministicHDPHMM:
            def __init__(self, *, alpha: float, **kwargs) -> None:
                self.alpha = alpha

            def fit_sequences(self, observations):
                token_probability = 0.8 if self.alpha == 1.0 else 0.2
                emission = np.full((1, 13), (1.0 - token_probability) / 12.0)
                emission[0, 0] = token_probability
                return SimpleNamespace(
                    posterior_initial_mean=np.array([1.0]),
                    posterior_transition_mean=np.array([[1.0]]),
                    posterior_emission_mean=emission,
                    log_likelihood=-self.alpha,
                    effective_states=1,
                )

        model = GlobalHDPHMM(
            truncation_level=1,
            n_iters=1,
            burn_in=0,
            hyperparameter_grid=((1.0, 4.0, 2.0), (2.0, 4.0, 2.0)),
            seed=2,
        )
        with patch("Comparacion.classical_models.TruncatedHDPHMM", DeterministicHDPHMM):
            fit_result = model.fit(
                [_piece("train", 0, 2, "MuseTrainer")],
                [_piece("validation", 0, 2, "SymbTr")],
                bos_token_id=12,
                max_context_length=128,
            )

        self.assertEqual(fit_result["selected_hyperparameters"]["alpha"], 1.0)
        self.assertAlmostEqual(model.best_result.posterior_emission_mean[0, 0], 0.8)


class ComparisonRunnerTests(unittest.TestCase):
    def test_full_pareto_is_incomparable_when_a_model_lacks_structure(self) -> None:
        raw_rows = [
            {
                "model": "finite_hmm",
                "frac": 1.0,
                "test_nll": 1.0,
                "fit_wall_clock_s": 2.0,
                "evaluation_wall_clock_s": 0.2,
            },
            {
                "model": "transformer",
                "frac": 1.0,
                "test_nll": 0.9,
                "fit_wall_clock_s": 3.0,
                "evaluation_wall_clock_s": 0.3,
            },
        ]
        structural = {
            "status": "ok",
            "model_metrics": [{"model": "finite_hmm", "mean_boundary_f1": 0.8}],
        }

        result = _build_pareto_summary(raw_rows, structural)

        full = result["full_three_axis_frontier"]
        self.assertEqual(full["status"], "incomparable")
        self.assertEqual(full["reason"], "missing_structural_measurements_for_models")
        self.assertEqual(full["missing_models"], ["transformer"])
        self.assertEqual(full["frontier"], [])

    def test_plan_only_writes_auditable_plan_without_constructing_models(self) -> None:
        self.assertIn("include_vomm_control", LearningCurveConfig.__dataclass_fields__)
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = replace(
                _runner_config(temporary_directory),
                train_fractions=(0.5, 1.0),
            )
            preparation = _synthetic_preparation()
            with (
                patch("Comparacion.runner.prepare_corpus", return_value=preparation),
                patch("Comparacion.runner.FiniteGlobalHMM") as finite_constructor,
                patch("Comparacion.runner.GlobalHDPHMM") as hdp_constructor,
                patch("Comparacion.runner.SmallTransformerNextTokenModel") as transformer_constructor,
                patch("Comparacion.runner.select_vomm_by_validation") as vomm_selector,
                patch("Comparacion.runner._build_transformer_dataloaders") as dataloader_builder,
            ):
                result = run_learning_curve_experiment(
                    config,
                    run_name="synthetic-plan",
                    max_files=12,
                    plan_only=True,
                )

            finite_constructor.assert_not_called()
            hdp_constructor.assert_not_called()
            transformer_constructor.assert_not_called()
            vomm_selector.assert_not_called()
            dataloader_builder.assert_not_called()

            output_root = Path(result["output_root"])
            self.assertEqual(result["status"], "planned_no_evidence")
            self.assertEqual(result["n_runs"], 0)
            self.assertTrue((output_root / "config.json").is_file())
            plan = json.loads((output_root / "execution_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["status"], "planned_no_evidence")
            self.assertFalse(plan["claims_evidence"])
            self.assertEqual(plan["seeds"], {"split_seed": 7, "data_seeds": [11], "model_seeds": [13]})
            self.assertEqual(plan["fits_by_family"]["finite_hmm"]["run_rows"], 2)
            self.assertEqual(plan["fits_by_family"]["hdp_hmm"]["candidate_fits"], 2)
            self.assertEqual(plan["fits_by_family"]["vomm"]["candidate_fits"], 4)
            self.assertEqual(plan["fits_by_family"]["transformer"]["workload_class"], "neural")
            self.assertEqual(plan["common_predictive_support"]["bos_token_id"], 12)
            self.assertEqual(plan["common_predictive_support"]["pad_token_id"], 13)
            self.assertFalse(plan["common_predictive_support"]["pad_in_scoring_support"])
            for split in plan["splits"].values():
                self.assertEqual(split["n_pieces"], len(split["piece_ids"]))
                self.assertEqual(split["n_canonical_groups"], len(split["canonical_work_ids"]))
            for coverage in plan["exact_expected_coverage"]["test"]:
                self.assertEqual(coverage["expected_event_indices"], list(range(coverage["n_events"])))
            self.assertFalse((output_root / "results_raw.csv").exists())
            self.assertFalse((output_root / "piece_metrics_raw.csv").exists())

    def test_runner_propagates_protocol_and_writes_multidimensional_artifacts(self) -> None:
        self.assertIn("include_vomm_control", LearningCurveConfig.__dataclass_fields__)
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotations_path = Path(temporary_directory) / "annotations.csv"
            annotations_path.write_text(
                "piece_id,event_index,segment_label,boundary\n"
                "piece-0,0,A,1\n",
                encoding="utf-8",
            )
            config = _runner_config(
                temporary_directory,
                structural_annotations_path=str(annotations_path),
            )
            preparation = _synthetic_preparation()

            finite_model = MagicMock()
            finite_model.fit.return_value = SimpleNamespace(selected_states=2)
            finite_model.evaluate.side_effect = lambda pieces, **kwargs: {
                "summary": {
                    "model": "finite_hmm",
                    "selected_states": 2,
                    "validation_nll_per_token": 1.4,
                    "test_nll_per_token": 1.5,
                    "test_perplexity": math.exp(1.5),
                    "n_tokens": sum(len(piece.tokens) for piece in pieces),
                    "n_params": 25,
                    "train_time_sec": 0.2,
                },
                "piece_metrics": _piece_metrics(pieces, nll=1.5),
            }
            hdp_model = MagicMock()
            hdp_model.fit.return_value = {
                "selected_hyperparameters": {"alpha": 1.0, "alpha0": 1.0, "gamma": 1.0}
            }
            hdp_model.evaluate.side_effect = lambda pieces, **kwargs: {
                "summary": {
                    "model": "hdp_hmm",
                    "truncation_level": 2,
                    "validation_nll_per_token": 1.3,
                    "test_nll_per_token": 1.4,
                    "test_perplexity": math.exp(1.4),
                    "n_tokens": sum(len(piece.tokens) for piece in pieces),
                    "n_params": 25,
                    "effective_states": 2,
                    "train_time_sec": 0.3,
                },
                "piece_metrics": _piece_metrics(pieces, nll=1.4),
            }
            transformer_model = MagicMock()
            transformer_model.fit.return_value = {
                "summary": {"best_validation_nll": 1.1},
                "train_log": [{"train_wall_clock_s": 0.4, "validation_wall_clock_s": 0.1}],
            }
            transformer_model.evaluate.side_effect = lambda pieces: {
                "summary": {
                    "nll_per_token": 1.2,
                    "perplexity": math.exp(1.2),
                    "accuracy": 0.5,
                    "brier_score": 0.7,
                    "n_tokens": sum(len(piece.tokens) for piece in pieces),
                    "eval_wall_clock_s": 0.15,
                    "parameter_count": 100,
                    "runtime": {"device": "cpu"},
                },
                "piece_metrics": _piece_metrics(pieces, nll=1.2),
            }
            vomm_model = MagicMock()
            vomm_model.selected_order = 2
            vomm_model.evaluate.side_effect = lambda pieces, max_context_length: {
                "summary": {
                    "model": "vomm",
                    "selected_order": 2,
                    "validation_nll_per_token": 1.2,
                    "test_nll_per_token": 1.3,
                    "test_perplexity": math.exp(1.3),
                    "accuracy": 0.5,
                    "brier_score": 0.8,
                    "n_tokens": sum(len(piece.tokens) for piece in pieces),
                    "n_params": 20,
                    "count_table_size": 20,
                    "train_time_sec": 0.05,
                    "evaluation_wall_clock_s": 0.02,
                },
                "piece_metrics": _piece_metrics(pieces, nll=1.3),
            }

            with (
                patch("Comparacion.runner.prepare_corpus", return_value=preparation),
                patch("Comparacion.runner.FiniteGlobalHMM", return_value=finite_model) as finite_constructor,
                patch("Comparacion.runner.GlobalHDPHMM", return_value=hdp_model) as hdp_constructor,
                patch(
                    "Comparacion.runner.SmallTransformerNextTokenModel",
                    return_value=transformer_model,
                ) as transformer_constructor,
                patch(
                    "Comparacion.runner._build_transformer_dataloaders",
                    side_effect=lambda config, train_pieces, validation_pieces, test_pieces: {
                        "train": train_pieces,
                        "validation": validation_pieces,
                        "test": test_pieces,
                    },
                ),
                patch(
                    "Comparacion.runner.select_vomm_by_validation",
                    return_value=vomm_model,
                ) as vomm_selector,
            ):
                result = run_learning_curve_experiment(config, run_name="synthetic-run", max_files=12)

            finite_model.fit.assert_called_once()
            finite_model.evaluate.assert_called_once()
            self.assertEqual(finite_model.fit.call_args.kwargs["max_context_length"], 3)
            self.assertEqual(finite_model.evaluate.call_args.kwargs["max_context_length"], 3)
            self.assertEqual(finite_model.fit.call_args.kwargs["bos_token_id"], 12)
            hdp_model.fit.assert_called_once()
            hdp_model.evaluate.assert_called_once()
            self.assertEqual(hdp_model.fit.call_args.kwargs["max_context_length"], 3)
            self.assertEqual(hdp_model.evaluate.call_args.kwargs["max_context_length"], 3)
            self.assertEqual(hdp_model.fit.call_args.kwargs["bos_token_id"], 12)
            self.assertEqual(vomm_selector.call_args.kwargs["max_context_length"], 3)
            self.assertEqual(vomm_selector.call_args.kwargs["bos_token_id"], 12)
            self.assertEqual(vomm_selector.call_args.kwargs["vocabulary_size"], 13)
            vomm_model.evaluate.assert_called_once()
            self.assertEqual(vomm_model.evaluate.call_args.args[1], 3)
            self.assertEqual(transformer_constructor.call_args.kwargs["max_context_length"], 3)
            self.assertEqual(transformer_constructor.call_args.kwargs["bos_token_id"], 12)
            self.assertEqual(transformer_constructor.call_args.kwargs["pad_token_id"], 13)
            self.assertEqual(transformer_constructor.call_args.kwargs["vocab_size"], 14)
            self.assertEqual(finite_constructor.call_count, 1)
            self.assertEqual(hdp_constructor.call_count, 1)

            output_root = Path(result["output_root"])
            required_artifacts = {
                "config.json",
                "results_raw.csv",
                "results_summary.csv",
                "piece_metrics_raw.csv",
                "pairwise_comparisons.json",
                "engineering_costs.csv",
                "protocol_audit.json",
                "structural_evaluation.json",
                "pareto_summary.json",
                "learning_curve.png",
            }
            self.assertTrue(required_artifacts.issubset({path.name for path in output_root.iterdir()}))
            self.assertFalse((output_root / "wilcoxon_test.json").exists())

            with (output_root / "results_raw.csv").open(encoding="utf-8", newline="") as stream:
                raw_rows = list(csv.DictReader(stream))
            self.assertEqual({row["model"] for row in raw_rows}, {"finite_hmm", "hdp_hmm", "vomm", "transformer"})
            self.assertEqual(len(raw_rows), 4)
            with (output_root / "piece_metrics_raw.csv").open(encoding="utf-8", newline="") as stream:
                piece_rows = list(csv.DictReader(stream))
            self.assertTrue(piece_rows)
            self.assertTrue(all(row["canonical_work_id"].strip() for row in piece_rows))

            protocol_audit = json.loads((output_root / "protocol_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol_audit["status"], "passed")
            self.assertTrue(protocol_audit["evidence"])
            self.assertTrue(all(item["expected_event_indices"] == item["scored_event_indices"] for item in protocol_audit["evidence"]))
            pairwise = json.loads((output_root / "pairwise_comparisons.json").read_text(encoding="utf-8"))
            self.assertEqual(pairwise["models"], ["finite_hmm", "hdp_hmm", "transformer", "vomm"])
            self.assertEqual(pairwise["n_comparisons"], 6)
            structural = json.loads((output_root / "structural_evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(structural["status"], "not_evaluated")
            self.assertEqual(structural["reason"], "missing_inferred_structure_artifact")
            pareto = json.loads((output_root / "pareto_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(pareto["predictive_cost_partial_frontier"]["status"], "ok")
            self.assertEqual(pareto["full_three_axis_frontier"]["status"], "not_evaluated")

            json_paths = list(output_root.glob("*.json")) + list((output_root / "splits").glob("*.json"))
            for path in json_paths:
                with self.subTest(path=path.name):
                    json.loads(
                        path.read_text(encoding="utf-8"),
                        parse_constant=lambda value: self.fail(f"non-standard JSON constant {value} in {path}"),
                    )

    def test_protocol_audit_fails_fast_on_asymmetric_scored_indices(self) -> None:
        self.assertIn("include_vomm_control", LearningCurveConfig.__dataclass_fields__)
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _runner_config(temporary_directory, include_vomm_control=False)
            preparation = _synthetic_preparation()
            finite_model = MagicMock()
            finite_model.fit.return_value = SimpleNamespace(selected_states=2)

            def incomplete_evaluation(pieces, **kwargs):
                first_piece = pieces[0]
                scored = {
                    first_piece.piece_id: [0, 2, *range(3, len(first_piece.tokens))]
                }
                return {
                    "summary": {
                        "selected_states": 2,
                        "validation_nll_per_token": 1.0,
                        "test_nll_per_token": 1.0,
                        "test_perplexity": math.e,
                        "n_params": 25,
                        "train_time_sec": 0.1,
                    },
                    "piece_metrics": _piece_metrics(
                        pieces,
                        nll=1.0,
                        scored_indices_by_piece=scored,
                    ),
                }

            finite_model.evaluate.side_effect = incomplete_evaluation
            with (
                patch("Comparacion.runner.prepare_corpus", return_value=preparation),
                patch("Comparacion.runner.FiniteGlobalHMM", return_value=finite_model),
                patch("Comparacion.runner.GlobalHDPHMM") as hdp_constructor,
                patch("Comparacion.runner.SmallTransformerNextTokenModel") as transformer_constructor,
            ):
                with self.assertRaisesRegex(ValueError, "protocol coverage"):
                    run_learning_curve_experiment(config, run_name="bad-coverage", max_files=12)

            hdp_constructor.assert_not_called()
            transformer_constructor.assert_not_called()
            audit_path = Path(temporary_directory) / "bad-coverage" / "protocol_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "failed")
            failed = [item for item in audit["evidence"] if item["status"] == "failed"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["omitted_event_indices"], [1])
            self.assertEqual(failed[0]["scored_event_indices"][:2], [0, 2])

    def test_malformed_structural_annotation_columns_fail_before_model_construction(self) -> None:
        self.assertIn("include_vomm_control", LearningCurveConfig.__dataclass_fields__)
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotations_path = Path(temporary_directory) / "bad-annotations.csv"
            annotations_path.write_text("piece_id,event_index\npiece-0,0\n", encoding="utf-8")
            config = _runner_config(
                temporary_directory,
                structural_annotations_path=str(annotations_path),
            )
            with (
                patch("Comparacion.runner.prepare_corpus", return_value=_synthetic_preparation()),
                patch("Comparacion.runner.FiniteGlobalHMM") as finite_constructor,
            ):
                with self.assertRaisesRegex(ValueError, "segment_label.*boundary"):
                    run_learning_curve_experiment(config, run_name="bad-annotations", max_files=12)

            finite_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Integración de las particiones por contenido en el runner (PROMPT_CLAUDE.md, tarea 1.2)."""

from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import Comparacion.runner as runner
from Comparacion.evidence_package import export_evidence_package
from Comparacion.runner import _build_transformer_dataloaders, run_learning_curve_experiment
from Comparacion.statistics import headline_estimates, pairwise_model_comparisons
from next_token_experiment.data.dataset import dataloader_seed
from next_token_experiment.schemas import CorpusPreparationResult, PreparedPiece
from tests.test_artifact_audit import _build_run
from tests.test_comparacion import _runner_config, _standard_model_mocks, _synthetic_preparation


def _varied_piece(piece_id: str, title: str, composer: str, tokens: list[int]) -> PreparedPiece:
    return PreparedPiece(
        piece_id=piece_id,
        source_path=f"/tmp/Synthetic/{piece_id}",
        title=title,
        composer=composer,
        canonical_work_id=f"{composer} {title}".strip() or piece_id,
        representation="pitch_class",
        vocabulary=[str(index) for index in range(12)],
        tokens=tokens,
        n_events=len(tokens),
        metadata={},
    )


def _random_tokens(seed: int, length: int = 40) -> list[int]:
    generator = random.Random(seed)
    return [generator.randrange(12) for _ in range(length)]


def _run(
    directory: str,
    name: str,
    *,
    config=None,
    preparation: CorpusPreparationResult | None = None,
    resume: bool = False,
    plan_only: bool = False,
) -> dict[str, MagicMock]:
    config = config or _runner_config(directory)
    mocks = _standard_model_mocks()
    dataloaders = MagicMock(
        side_effect=lambda config, train_pieces, validation_pieces, test_pieces, **_: {
            "train": train_pieces,
            "validation": validation_pieces,
            "test": test_pieces,
        }
    )
    with (
        patch("Comparacion.runner.prepare_corpus", return_value=preparation or _synthetic_preparation()),
        patch("Comparacion.runner.FiniteGlobalHMM", return_value=mocks["finite"]) as finite,
        patch("Comparacion.runner.GlobalHDPHMM", return_value=mocks["hdp"]),
        patch("Comparacion.runner.SmallTransformerNextTokenModel", return_value=mocks["transformer"]),
        patch("Comparacion.runner._build_transformer_dataloaders", dataloaders),
        patch("Comparacion.runner.select_vomm_by_validation", return_value=mocks["vomm"]),
    ):
        run_learning_curve_experiment(
            config, run_name=name, max_files=12, resume=resume, plan_only=plan_only
        )
    return {**mocks, "finite_constructor": finite, "dataloaders": dataloaders}


class GroupedSplitsInRunnerTests(unittest.TestCase):
    def test_runner_uses_the_shared_split_builder(self) -> None:
        self.assertFalse(hasattr(runner, "_build_grouped_fixed_splits"))

    def test_pieces_with_shared_content_never_cross_splits(self) -> None:
        # Diez pares con el mismo contenido y metadatos distintos: sólo el
        # contenido los une. Por azar, diez pares no caen juntos.
        pieces = [
            _varied_piece(f"p{index:02d}.mxl", f"Obra numero {index}", f"Autor {index}", _random_tokens(index // 2))
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "plan", preparation=CorpusPreparationResult(pieces=pieces, exclusions=[]), plan_only=True)
            plan = json.loads((Path(directory) / "plan" / "execution_plan.json").read_text(encoding="utf-8"))
        located = {
            piece_id: name
            for name, payload in plan["splits"].items()
            for piece_id in payload["piece_ids"]
        }
        for index in range(0, 20, 2):
            self.assertEqual(located[f"p{index:02d}.mxl"], located[f"p{index + 1:02d}.mxl"], index)


class SplitManifestTests(unittest.TestCase):
    def test_manifest_hash_is_recorded_in_the_execution_plan(self) -> None:
        for plan_only in (True, False):
            with self.subTest(plan_only=plan_only), tempfile.TemporaryDirectory() as directory:
                _run(directory, "run", plan_only=plan_only)
                root = Path(directory) / "run"
                manifest = json.loads((root / "splits" / "manifest.json").read_text(encoding="utf-8"))
                plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
                self.assertEqual(plan["split_sha256"], manifest["sha256"])
                self.assertEqual(len(manifest["sha256"]), 64)

    def test_manifest_compares_against_the_reference_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "original", plan_only=True)
            reference_path = Path(directory) / "original" / "execution_plan.json"
            config = replace(_runner_config(directory), reference_splits_path=str(reference_path))
            _run(directory, "nueva", config=config, plan_only=True)
            manifest = json.loads(
                (Path(directory) / "nueva" / "splits" / "manifest.json").read_text(encoding="utf-8")
            )
        self.assertIn("comparison", manifest)
        for name in ("test", "validation", "train_pool"):
            self.assertEqual(manifest["comparison"][name]["n_only_new"], 0)
            self.assertEqual(manifest["comparison"][name]["n_only_reference"], 0)
            self.assertGreater(manifest["comparison"][name]["n_shared"], 0)


class OriginalRunReferenceTests(unittest.TestCase):
    """La corrida original (tesis_3000_gpu_20260823_1941) no tiene execution_plan.json:
    sus particiones son splits/test_pieces.json y splits/val_pieces.json, cada una
    {n_pieces, piece_ids}.  Preregistro B: el conjunto de prueba nuevo se reporta
    junto con su diferencia respecto del anterior."""

    def _original_layout(self, directory: str) -> Path:
        _run(directory, "original", plan_only=True)
        original = Path(directory) / "original"
        (original / "execution_plan.json").unlink()
        (original / "splits" / "manifest.json").unlink()
        return original

    def test_the_original_run_directory_is_a_valid_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = self._original_layout(directory)
            reference = runner._load_reference_splits(str(original))
            test_ids = json.loads((original / "splits" / "test_pieces.json").read_text(encoding="utf-8"))["piece_ids"]
        self.assertEqual(sorted(reference), ["test", "validation"])
        self.assertEqual(reference["test"], test_ids)

    def test_a_single_split_file_of_the_original_run_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = self._original_layout(directory)
            reference = runner._load_reference_splits(str(original / "splits" / "test_pieces.json"))
            validation = runner._load_reference_splits(str(original / "splits" / "val_pieces.json"))
        self.assertEqual(list(reference), ["test"])
        self.assertEqual(list(validation), ["validation"])

    def test_the_difference_is_reported_in_the_plan_and_the_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = self._original_layout(directory)
            config = replace(_runner_config(directory), reference_splits_path=str(original))
            _run(directory, "nueva", config=config)
            root = Path(directory) / "nueva"
            plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
        for payload in (plan, summary):
            comparison = payload["reference_split_comparison"]
            self.assertEqual(comparison["status"], "reported")
            self.assertEqual(comparison["splits"]["test"]["n_only_new"], 0)
            self.assertGreater(comparison["splits"]["test"]["n_shared"], 0)

    def test_without_a_reference_the_gap_is_declared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "sin-ref")
            root = Path(directory) / "sin-ref"
            plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "splits" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["reference_split_comparison"]["status"], "not_reported_no_reference_given")
        self.assertIn("preregistro", plan["reference_split_comparison"]["required_by"])
        self.assertEqual(manifest["comparison_status"], "not_reported_no_reference_given")
        self.assertIn(
            "test_split_difference_vs_original_run_not_reported",
            plan["analysis_status"]["preregistration_gaps"],
        )

    def test_the_reference_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = self._original_layout(directory)
            before = {path.name: path.stat().st_mtime_ns for path in original.rglob("*") if path.is_file()}
            config = replace(_runner_config(directory), reference_splits_path=str(original))
            _run(directory, "nueva", config=config, plan_only=True)
            after = {path.name: path.stat().st_mtime_ns for path in original.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


class ReferenceSplitsCliTests(unittest.TestCase):
    def test_cli_accepts_the_reference_partition(self) -> None:
        from Comparacion.cli import build_parser

        args = build_parser().parse_args(["--reference-splits", "plan/execution_plan.json"])
        self.assertEqual(args.reference_splits, "plan/execution_plan.json")


class DataloaderSeedTests(unittest.TestCase):
    def test_shuffle_seed_comes_from_model_and_data_seed(self) -> None:
        config = _runner_config("unused")
        pieces = _synthetic_preparation().pieces
        with patch("Comparacion.runner.build_dataloaders", return_value={}) as builder:
            _build_transformer_dataloaders(
                config,
                train_pieces=pieces[:6],
                validation_pieces=pieces[6:9],
                test_pieces=pieces[9:],
                model_seed=13,
                data_seed=11,
            )
        self.assertEqual(
            builder.call_args.kwargs["shuffle_seed"],
            dataloader_seed(model_seed=13, data_seed=11),
        )

    def test_runner_passes_both_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mocks = _run(directory, "seeds")
        kwargs = mocks["dataloaders"].call_args.kwargs
        self.assertEqual((kwargs["model_seed"], kwargs["data_seed"]), (13, 11))


class ResumeGuardTests(unittest.TestCase):
    def _config(self, directory: str):
        return replace(_runner_config(directory), train_fractions=(0.5, 1.0))

    def test_resume_rejects_a_changed_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "run", config=self._config(directory))
            changed = replace(self._config(directory), finite_hmm_max_iterations=7)
            with self.assertRaisesRegex(ValueError, "configuraci"):
                _run(directory, "run", config=changed, resume=True)

    def test_resume_rejects_a_changed_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "run", config=self._config(directory))
            with self.assertRaisesRegex(ValueError, "partici"):
                _run(
                    directory,
                    "run",
                    config=self._config(directory),
                    preparation=_synthetic_preparation(n_pieces=15),
                    resume=True,
                )

    def test_rejected_resume_fits_nothing_and_keeps_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "run", config=self._config(directory))
            config_path = Path(directory) / "run" / "config.json"
            before = config_path.read_text(encoding="utf-8")
            changed = replace(self._config(directory), finite_hmm_max_iterations=7)
            mocks = _standard_model_mocks()
            with (
                patch("Comparacion.runner.prepare_corpus", return_value=_synthetic_preparation()),
                patch("Comparacion.runner.FiniteGlobalHMM", return_value=mocks["finite"]) as finite,
                self.assertRaises(ValueError),
            ):
                run_learning_curve_experiment(changed, run_name="run", max_files=12, resume=True)
            self.assertEqual(finite.call_count, 0)
            self.assertEqual(config_path.read_text(encoding="utf-8"), before)

    def test_resume_rejects_changed_melodies_under_the_same_piece_ids(self) -> None:
        # Mismos ids y misma partición, pero otra extracción de la melodía (p. ej. tras
        # cambiar el preprocesamiento): mezclar celdas viejas y nuevas no es reanudar.
        original = _synthetic_preparation()
        shifted = CorpusPreparationResult(
            pieces=[replace(piece, tokens=[(token + 5) % 12 for token in piece.tokens]) for piece in original.pieces],
            exclusions=list(original.exclusions),
        )
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "run", config=self._config(directory), preparation=original)
            with self.assertRaisesRegex(ValueError, "corpus"):
                _run(directory, "run", config=self._config(directory), preparation=shifted, resume=True)

    def test_resume_rejects_a_checkpoint_without_a_corpus_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "run", config=self._config(directory))
            inputs_path = Path(directory) / "run" / runner.RESUME_INPUTS_FILENAME
            recorded = json.loads(inputs_path.read_text(encoding="utf-8"))
            recorded.pop("corpus_sha256", None)
            inputs_path.write_text(json.dumps(recorded), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corpus"):
                _run(directory, "run", config=self._config(directory), resume=True)

    def test_resume_with_the_same_inputs_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "run", config=self._config(directory))
            mocks = _run(directory, "run", config=self._config(directory), resume=True)
        self.assertEqual(mocks["finite_constructor"].call_count, 0)


class ResumeReferenceSplitTests(unittest.TestCase):
    """--resume no puede borrar ni cambiar la diferencia con la partición de
    referencia que la corrida ya reportó (preregistro B)."""

    def _config(self, directory: str, reference: Path | None):
        return replace(
            _runner_config(directory),
            train_fractions=(0.5, 1.0),
            reference_splits_path=None if reference is None else str(reference),
        )

    def _reference(self, directory: str, name: str = "ref.json", ids=None) -> Path:
        path = Path(directory) / name
        if ids is None:
            _run(directory, "original", plan_only=True)
            plan = json.loads((Path(directory) / "original" / "execution_plan.json").read_text(encoding="utf-8"))
            path.write_text(json.dumps({"splits": plan["splits"]}), encoding="utf-8")
        else:
            path.write_text(json.dumps({"test": ids}), encoding="utf-8")
        return path

    def test_resume_without_the_reference_is_rejected_and_keeps_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = self._reference(directory)
            _run(directory, "run", config=self._config(directory, reference))
            summary_path = Path(directory) / "run" / "run_summary.json"
            before = summary_path.read_text(encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "referencia"):
                _run(directory, "run", config=self._config(directory, None), resume=True)
            self.assertEqual(summary_path.read_text(encoding="utf-8"), before)
            self.assertEqual(json.loads(before)["reference_split_comparison"]["status"], "reported")

    def test_resume_with_a_different_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = self._reference(directory)
            other = self._reference(directory, "otra.json", ids=["x", "y"])
            _run(directory, "run", config=self._config(directory, reference))
            with self.assertRaisesRegex(ValueError, "referencia"):
                _run(directory, "run", config=self._config(directory, other), resume=True)

    def test_resume_adding_a_reference_the_run_did_not_have_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = self._reference(directory)
            _run(directory, "run", config=self._config(directory, None))
            with self.assertRaisesRegex(ValueError, "referencia"):
                _run(directory, "run", config=self._config(directory, reference), resume=True)

    def test_the_same_reference_moved_elsewhere_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = self._reference(directory)
            _run(directory, "run", config=self._config(directory, reference))
            moved = Path(directory) / "otra_unidad" / "ref.json"
            moved.parent.mkdir()
            moved.write_bytes(reference.read_bytes())
            mocks = _run(directory, "run", config=self._config(directory, moved), resume=True)
            summary = json.loads((Path(directory) / "run" / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(mocks["finite_constructor"].call_count, 0)
        self.assertEqual(summary["reference_split_comparison"]["status"], "reported")


class ArtifactPlanTests(unittest.TestCase):
    def test_every_file_a_full_run_writes_is_in_the_artifact_plan(self) -> None:
        import fnmatch

        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "run")
            root = Path(directory) / "run"
            written = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
            plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
        planned = [item["path"] for item in plan["artifact_plan"]]
        for name in ("vomm_order_grid_audit.json", "hdp_traces.csv"):
            self.assertIn(name, planned)
        unlisted = [name for name in written if not any(fnmatch.fnmatch(name, pattern) for pattern in planned)]
        self.assertEqual(unlisted, [])

    def test_without_vomm_its_grid_audit_is_not_planned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(_runner_config(directory, include_vomm_control=False))
            _run(directory, "plan", config=config, plan_only=True)
            root = Path(directory) / "plan"
            plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
            written = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        self.assertNotIn("vomm_order_grid_audit.json", [item["path"] for item in plan["artifact_plan"]])
        # plan_only_status dice la verdad sobre lo que el modo plan escribe.
        for item in plan["artifact_plan"]:
            if "*" in item["path"]:
                continue
            with self.subTest(path=item["path"]):
                expected = "written" if item["path"] in written else "not_written_until_execution"
                self.assertEqual(item["plan_only_status"], expected)


class ConfigHashTests(unittest.TestCase):
    """--resume compara contenido: mover el corpus de unidad no cambia la configuración."""

    def test_moving_the_corpus_to_another_drive_keeps_the_hash(self) -> None:
        base = _runner_config("C:/runs")
        moved = replace(
            base,
            results_root="D:/runs",
            reference_splits_path="D:/original",
            experiment=replace(
                base.experiment,
                corpus=replace(base.experiment.corpus, root_dir="D:/otra/ruta/PDMX"),
                storage=replace(base.experiment.storage, results_root="D:/otros"),
            ),
        )
        self.assertEqual(runner._config_sha256(base), runner._config_sha256(moved))

    def test_a_real_configuration_change_still_changes_the_hash(self) -> None:
        base = _runner_config("C:/runs")
        self.assertNotEqual(
            runner._config_sha256(base), runner._config_sha256(replace(base, finite_hmm_max_iterations=7))
        )

    def test_structural_annotations_count_by_content_not_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.csv"
            second = Path(directory) / "otra" / "b.csv"
            second.parent.mkdir()
            text = "piece_id,event_index,segment_label,boundary\np,0,A,1\n"
            first.write_text(text, encoding="utf-8")
            second.write_text(text, encoding="utf-8")
            base = _runner_config(directory)
            same = runner._config_sha256(replace(base, structural_annotations_path=str(first)))
            moved = runner._config_sha256(replace(base, structural_annotations_path=str(second)))
            second.write_text(text + "p,1,B,0\n", encoding="utf-8")
            changed = runner._config_sha256(replace(base, structural_annotations_path=str(second)))
        self.assertEqual(same, moved)
        self.assertNotEqual(same, changed)

    def test_the_partition_message_names_the_hash_it_compares(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint.jsonl").write_text('{"cell": "x"}\n', encoding="utf-8")
            (root / runner.RESUME_INPUTS_FILENAME).write_text(
                json.dumps({"split_sha256": "a", "corpus_sha256": "c", "config_sha256": "k"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError) as caught:
                runner._guard_resume_inputs(
                    root,
                    root / "checkpoint.jsonl",
                    {"split_sha256": "b", "corpus_sha256": "c", "config_sha256": "k"},
                )
        message = str(caught.exception)
        self.assertIn("ids normalizados", message)
        self.assertNotIn("sha256 de splits/manifest.json", message)


class DistinctRunCountTests(unittest.TestCase):
    def _rows(self, subsets: dict[int, str]) -> list[dict[str, object]]:
        return [
            {
                "model": model,
                "canonical_work_id": work,
                "frac": 1.0,
                "split_seed": 7,
                "data_seed": data_seed,
                "model_seed": 1,
                "train_subset_sha256": subset,
                "n_tokens": 10,
                "nll_per_token": 1.0 + (0.1 if model == "a" else 0.0),
            }
            for model in ("a", "b")
            for data_seed, subset in subsets.items()
            for work in ("w1", "w2", "w3")
        ]

    def test_identical_training_sets_count_as_one_run(self) -> None:
        payload = headline_estimates(self._rows({1: "same", 2: "same"}), bootstrap_samples=5, seed=0)
        estimate = payload["estimates"][0]
        self.assertEqual(estimate["n_runs"], 1)
        self.assertEqual(estimate["n_nominal_runs"], 2)

    def test_distinct_training_sets_count_separately(self) -> None:
        payload = headline_estimates(self._rows({1: "x", 2: "y"}), bootstrap_samples=5, seed=0)
        self.assertEqual(payload["estimates"][0]["n_runs"], 2)

    def test_pairwise_declares_distinct_full_fraction_runs(self) -> None:
        payload = pairwise_model_comparisons(self._rows({1: "same", 2: "same"}), bootstrap_samples=5, seed=0)
        self.assertEqual(payload["full_fraction_runs"]["a"], {"n_distinct_runs": 1, "n_nominal_runs": 2})

    def test_runner_rows_carry_the_training_subset_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(_runner_config(directory), data_seeds=(11, 12))
            _run(directory, "run", config=config)
            root = Path(directory) / "run"
            for name in ("results_raw.csv", "piece_metrics_raw.csv"):
                with (root / name).open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(all(len(row["train_subset_sha256"]) == 64 for row in rows), name)
            with (root / "results_summary.csv").open(encoding="utf-8", newline="") as handle:
                summary = list(csv.DictReader(handle))
            self.assertTrue(all(row["n_distinct_runs"] for row in summary))


class PieceIdWorkTests(unittest.TestCase):
    def test_piece_path_work_ids_flow_through_every_grouped_analysis(self) -> None:
        # Títulos genéricos: la agrupación deja cada pieza como «pieza:<ruta>».
        pieces = [
            _varied_piece(
                f"external/PDMX/mxl/a,b/{index}.v{index}.mxl",
                "Untitled",
                "",
                _random_tokens(index, length=6 + index % 3),
            )
            for index in range(12)
        ]
        with tempfile.TemporaryDirectory() as directory:
            _run(directory, "run", preparation=CorpusPreparationResult(pieces=pieces, exclusions=[]))
            root = Path(directory) / "run"
            with (root / "piece_metrics_raw.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            work_ids = {row["canonical_work_id"] for row in rows}
            self.assertTrue(work_ids)
            self.assertTrue(all(work.startswith("pieza:external/PDMX/") for work in work_ids))
            headline = json.loads((root / "headline_estimates.json").read_text(encoding="utf-8"))
            pairwise = json.loads((root / "pairwise_comparisons.json").read_text(encoding="utf-8"))
            denominator = json.loads((root / "denominator_audit.json").read_text(encoding="utf-8"))
            canonical = json.loads((root / "canonicalization_audit.json").read_text(encoding="utf-8"))
        self.assertEqual({item["n_works"] for item in headline["estimates"]}, {len(work_ids)})
        self.assertTrue(all(item["n_pairs"] == len(work_ids) for item in pairwise["comparisons"]))
        self.assertEqual(denominator["n_canonical_works"], len(work_ids))
        self.assertEqual(denominator["explanation"], "files_equal_works")
        # La auditoría describe la agrupación que se usó, no las etiquetas crudas,
        # y una pieza aislada por «pieza:<ruta>» ya no es una etiqueta ambigua.
        self.assertEqual(canonical["n_canonical_works"], len(pieces))
        self.assertEqual(canonical["generic_labels"], [])
        self.assertFalse(any(work.startswith("pieza:") for work in canonical["review_required"]))


class EvidencePackageContentsTests(unittest.TestCase):
    def test_headline_and_vomm_grid_audit_are_packaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            (run_dir / "headline_estimates.json").write_text(json.dumps({"estimates": []}), encoding="utf-8")
            (run_dir / "vomm_order_grid_audit.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            registry = root / "registry.json"
            registry.write_text(
                json.dumps({"runs": [{"name": "main", "path": str(run_dir), "role": "primary", "cost_usable": True}]}),
                encoding="utf-8",
            )
            export_evidence_package(registry, root / "public")
            packaged = root / "public" / "runs" / "main"
            self.assertTrue((packaged / "headline_estimates.json").is_file())
            self.assertTrue((packaged / "vomm_order_grid_audit.json").is_file())

    def test_hdp_traces_execution_plan_and_resume_inputs_are_packaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            # El nombre que escribe write_cell_traces, no el viejo hdp_chain_trace_seed*.csv.
            (run_dir / "hdp_traces.csv").write_text(
                "split_seed,data_seed,model_seed,frac,iteration,log_likelihood,active_states,in_burn_in\n"
                "7,1,1,1.0,1,-1.0,3,True\n",
                encoding="utf-8",
            )
            (run_dir / "execution_plan.json").write_text(json.dumps({"split_sha256": "ab" * 32}), encoding="utf-8")
            (run_dir / runner.RESUME_INPUTS_FILENAME).write_text(
                json.dumps({"split_sha256": "ab" * 32, "corpus_sha256": "cd" * 32, "config_sha256": "ef" * 32}),
                encoding="utf-8",
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps({"runs": [{"name": "main", "path": str(run_dir), "role": "primary", "cost_usable": True}]}),
                encoding="utf-8",
            )
            export_evidence_package(registry, root / "public")
            packaged = root / "public" / "runs" / "main"
            self.assertTrue((packaged / "hdp_traces.csv").is_file())
            self.assertTrue((packaged / runner.RESUME_INPUTS_FILENAME).is_file())
            plan = json.loads((packaged / "execution_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["split_sha256"], "ab" * 32)


if __name__ == "__main__":
    unittest.main()

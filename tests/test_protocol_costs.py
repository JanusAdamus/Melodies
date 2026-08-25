from __future__ import annotations

import unittest

from Comparacion.cli import build_parser
from Comparacion.classical_models import FiniteGlobalHMM, GlobalHDPHMM
from Comparacion.runner import (
    PROTOCOL_COST_FIELDS,
    _build_hardware_manifest,
    _protocol_cost_fields,
)
from next_token_experiment.schemas import PreparedPiece


class ProtocolCostFieldTests(unittest.TestCase):
    def test_every_family_reports_the_same_five_fields(self) -> None:
        self.assertEqual(
            PROTOCOL_COST_FIELDS,
            (
                "selection_wall_clock_s",
                "selected_fit_wall_clock_s",
                "selected_validation_wall_clock_s",
                "evaluation_wall_clock_s",
                "total_protocol_wall_clock_s",
            ),
        )

    def test_costs_are_separated_and_totalled(self) -> None:
        fields = _protocol_cost_fields(
            {
                "selection_wall_clock_s": 10.0,
                "selected_fit_wall_clock_s": 4.0,
                "selected_validation_wall_clock_s": 1.0,
            },
            evaluation_wall_clock_s=2.0,
        )

        self.assertEqual(fields["selection_wall_clock_s"], 10.0)
        self.assertEqual(fields["selected_fit_wall_clock_s"], 4.0)
        self.assertEqual(fields["selected_validation_wall_clock_s"], 1.0)
        self.assertEqual(fields["evaluation_wall_clock_s"], 2.0)
        # La selección ya contiene el ajuste y la validación seleccionados.
        self.assertEqual(fields["total_protocol_wall_clock_s"], 12.0)
        self.assertEqual(fields["cost_status"], "separated")

    def test_missing_selection_falls_back_to_the_selected_parts(self) -> None:
        fields = _protocol_cost_fields(
            {"selected_fit_wall_clock_s": 4.0, "selected_validation_wall_clock_s": 1.0},
            evaluation_wall_clock_s=2.0,
        )

        self.assertIsNone(fields["selection_wall_clock_s"])
        self.assertEqual(fields["total_protocol_wall_clock_s"], 7.0)
        self.assertEqual(fields["cost_status"], "partially_separated")

    def test_a_single_ambiguous_time_is_never_presented_as_training_cost(self) -> None:
        fields = _protocol_cost_fields({"train_time_sec": 9.0}, evaluation_wall_clock_s=2.0)

        self.assertIsNone(fields["selection_wall_clock_s"])
        self.assertIsNone(fields["selected_fit_wall_clock_s"])
        self.assertIsNone(fields["selected_validation_wall_clock_s"])
        self.assertEqual(fields["total_protocol_wall_clock_s"], 11.0)
        self.assertEqual(fields["cost_status"], "unseparated_legacy_total")

    def test_non_finite_values_are_dropped_with_a_reason(self) -> None:
        fields = _protocol_cost_fields(
            {"selection_wall_clock_s": float("nan")}, evaluation_wall_clock_s=2.0
        )

        self.assertIsNone(fields["selection_wall_clock_s"])
        self.assertEqual(fields["cost_status"], "unmeasured")
        self.assertEqual(fields["total_protocol_wall_clock_s"], 2.0)


class HardwareManifestTests(unittest.TestCase):
    def test_manifest_reports_versions_and_precision(self) -> None:
        manifest = _build_hardware_manifest(target_device="cpu", precision="fp32")

        self.assertEqual(manifest["target_device"], "cpu")
        self.assertEqual(manifest["precision"], "fp32")
        self.assertIn("python_version", manifest)
        self.assertIn("numpy_version", manifest)
        self.assertIn("torch_version", manifest)
        self.assertIn("cpu_count", manifest)
        self.assertIn("peak_gpu_memory_bytes", manifest)

    def test_unavailable_measurements_are_null_with_a_reason(self) -> None:
        manifest = _build_hardware_manifest(target_device="cpu", precision="fp32")

        for field, reason_field in (
            ("peak_gpu_memory_bytes", "peak_gpu_memory_status"),
            ("total_ram_bytes", "total_ram_status"),
        ):
            with self.subTest(field=field):
                if manifest[field] is None:
                    self.assertTrue(manifest[reason_field])



class ClassicalFamilyCostContractTests(unittest.TestCase):
    """Las familias clásicas deben separar selección, ajuste y validación."""

    def _pieces(self, count: int, *, prefix: str) -> list[PreparedPiece]:
        return [
            PreparedPiece(
                piece_id=f"{prefix}-{index}",
                source_path=f"/tmp/{prefix}/{index}.musicxml",
                title=f"{prefix} {index}",
                composer="composer",
                canonical_work_id=f"{prefix}-work-{index}",
                representation="pitch_class",
                vocabulary=[str(value) for value in range(12)],
                tokens=[(index + step) % 12 for step in range(8)],
                n_events=8,
                metadata={},
            )
            for index in range(count)
        ]

    def test_finite_hmm_separates_selection_from_the_selected_fit(self) -> None:
        model = FiniteGlobalHMM(
            candidate_num_states=(2, 3),
            max_iterations=2,
            tolerance=1e-3,
            seed=1,
        )
        fit_result = model.fit(
            self._pieces(3, prefix="train"),
            self._pieces(2, prefix="validation"),
            bos_token_id=12,
            max_context_length=4,
        )

        self.assertEqual(len(fit_result.candidate_log), 2)
        self.assertGreater(fit_result.selection_wall_clock_s, 0.0)
        self.assertGreater(fit_result.selected_fit_wall_clock_s, 0.0)
        self.assertGreaterEqual(
            fit_result.selection_wall_clock_s,
            fit_result.selected_fit_wall_clock_s + fit_result.selected_validation_wall_clock_s,
        )

        summary = model.evaluate(
            self._pieces(2, prefix="test"), bos_token_id=12, max_context_length=4
        )["summary"]
        for field in ("selection_wall_clock_s", "selected_fit_wall_clock_s", "selected_validation_wall_clock_s"):
            with self.subTest(field=field):
                self.assertIsNotNone(summary[field])

    def test_hdp_hmm_separates_selection_from_the_selected_fit(self) -> None:
        model = GlobalHDPHMM(
            truncation_level=3,
            n_iters=2,
            burn_in=1,
            hyperparameter_grid=((1.0, 1.0, 1.0), (0.5, 1.0, 1.0)),
            seed=1,
        )
        fit_summary = model.fit(
            self._pieces(3, prefix="train"),
            self._pieces(2, prefix="validation"),
            bos_token_id=12,
            max_context_length=4,
        )

        self.assertGreater(fit_summary["selection_wall_clock_s"], 0.0)
        self.assertGreater(fit_summary["selected_fit_wall_clock_s"], 0.0)
        self.assertGreaterEqual(
            fit_summary["selection_wall_clock_s"],
            fit_summary["selected_fit_wall_clock_s"]
            + fit_summary["selected_validation_wall_clock_s"],
        )
        self.assertTrue(
            all("fit_wall_clock_s" in row for row in fit_summary["train_log"]),
            "cada candidato del HDP-HMM registra su propio costo",
        )


class TrainStrideCliTests(unittest.TestCase):
    def test_train_stride_flag_is_parsed(self) -> None:
        args = build_parser().parse_args(["--train-stride", "128"])

        self.assertEqual(args.train_stride, 128)

    def test_train_stride_defaults_to_the_configured_value(self) -> None:
        args = build_parser().parse_args([])

        self.assertIsNone(args.train_stride)

if __name__ == "__main__":
    unittest.main()

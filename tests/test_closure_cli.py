from __future__ import annotations

import unittest

from Comparacion.cli import build_parser


class ClosureCliTests(unittest.TestCase):
    def test_evidence_export_arguments_are_parsed(self) -> None:
        args = build_parser().parse_args(
            ["--export-evidence", "runs.json", "--evidence-output", "release"]
        )

        self.assertEqual(args.export_evidence, "runs.json")
        self.assertEqual(args.evidence_output, "release")

    def test_cost_arguments_are_parsed(self) -> None:
        args = build_parser().parse_args(
            [
                "--cost-input",
                "costs.csv",
                "--tariffs",
                "tariffs.json",
                "--cost-output",
                "report.json",
            ]
        )

        self.assertEqual(args.cost_input, "costs.csv")
        self.assertEqual(args.tariffs, "tariffs.json")

    def test_requirement_arguments_are_parsed(self) -> None:
        args = build_parser().parse_args(
            [
                "--validate-requirements",
                "requirements.json",
                "--validation-context",
                "context.json",
                "--validation-output",
                "validation",
            ]
        )

        self.assertEqual(args.validate_requirements, "requirements.json")
        self.assertEqual(args.validation_context, "context.json")

    def test_parser_keeps_operation_inputs_distinct(self) -> None:
        args = build_parser().parse_args(
            ["--verify-evidence", "release", "--run-name", "unused"]
        )

        self.assertEqual(args.verify_evidence, "release")


if __name__ == "__main__":
    unittest.main()

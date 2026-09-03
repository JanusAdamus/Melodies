from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .artifact_audit import write_audit_reports
from .cost_scenarios import write_cost_scenarios
from .evidence_package import export_evidence_package, verify_evidence_package
from .requirements_validation import write_requirement_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the learning-curve comparison in Comparacion/.")
    parser.add_argument("--run-name", default="learning_curve", help="Output directory name under artifacts/Comparacion.")
    parser.add_argument("--corpus-root", default=None, help="Override corpus root directory.")
    parser.add_argument("--results-root", default=None, help="Override results root directory.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of scores.")
    parser.add_argument("--corpus-sample-seed", type=int, default=None, help="Draw the --max-files scores as a seeded random sample instead of the first N in path order.")
    parser.add_argument("--transformer-max-epochs", type=int, default=None, help="Override Transformer max epochs.")
    parser.add_argument("--transformer-device", default=None, choices=("cpu", "auto", "cuda", "mps"), help="Override Transformer device.")
    parser.add_argument("--data-seeds", default=None, help="Comma-separated data seeds, for example 1,2,3.")
    parser.add_argument("--model-seeds", default=None, help="Comma-separated model seeds, for example 1,2.")
    parser.add_argument("--fractions", default=None, help="Comma-separated training fractions, for example 0.1,0.5,1.0.")
    parser.add_argument("--n-workers", type=int, default=None, help="Parsing processes for corpus preparation; 1 forces the serial path. Defaults to CPU count minus one.")
    parser.add_argument("--corpus-cache", default=None, help="JSONL file caching parsed scores so an interrupted run resumes instead of reparsing.")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run in the same --run-name directory, skipping the (data_seed, fraction) cells already recorded in checkpoint.jsonl.")
    parser.add_argument("--plan-only", action="store_true", help="Write an execution plan without constructing or fitting models.")
    parser.add_argument("--without-vomm", action="store_true", help="Disable the optional PPM-inspired VOMM diagnostic control.")
    parser.add_argument("--structural-annotations", default=None, help="Optional CSV with piece_id,event_index,segment_label,boundary columns.")
    parser.add_argument("--split-seed", type=int, default=None, help="Seed of the fixed train/validation/test partition. Repeat the run with different values to measure sensitivity to the partition.")
    parser.add_argument("--finite-hmm-states", default=None, help="Comma-separated candidate state counts for the finite HMM, for example 48,72,96. Must be increasing, unique and at least 2.")
    parser.add_argument("--finite-hmm-max-iterations", type=int, default=None, help="Iteration budget of the finite HMM. Fitting stops early when the validation NLL stops improving, so reaching this cap means the budget was binding and not that the model converged.")
    parser.add_argument("--train-stride", type=int, default=None, help="Training window stride. Equal to --max-context-length means non-overlapping training windows; the default 64 exposes each event more than once per epoch.")
    parser.add_argument("--audit-run", default=None, help="Audit an existing run directory read-only and exit; writes the manifest and the audit outside it.")
    parser.add_argument("--audit-output", default=None, help="Directory for --audit-run reports. Defaults to <run>/../audits/<run name>.")
    parser.add_argument(
        "--resource-measurement-condition",
        choices=("isolated", "contended", "unknown"),
        default="unknown",
        help="Declare whether timing and memory were measured without competing workloads. Only isolated rows are eligible for monetary scenarios.",
    )
    parser.add_argument("--export-evidence", default=None, metavar="REGISTRY.json", help="Build a sanitized, self-verifying evidence package from an explicit run registry.")
    parser.add_argument("--evidence-output", default=None, help="Output directory for --export-evidence.")
    parser.add_argument("--evidence-archive", default=None, help="Optional ZIP path for --export-evidence.")
    parser.add_argument("--verify-evidence", default=None, metavar="DIR", help="Verify hashes, audits and path sanitization in an evidence package, then exit.")
    parser.add_argument("--cost-input", default=None, metavar="engineering_costs.csv", help="Compute monetary scenarios from isolated resource measurements.")
    parser.add_argument("--tariffs", default=None, metavar="tariffs.json", help="Documented CPU/GPU tariffs used with --cost-input.")
    parser.add_argument("--cost-output", default=None, metavar="cost_scenarios.json", help="Output report used with --cost-input.")
    parser.add_argument("--validate-requirements", default=None, metavar="requirements.json", help="Generate a formal validation matrix from requirements and evidence context.")
    parser.add_argument("--validation-context", default=None, metavar="context.json", help="Evidence paths used by --validate-requirements.")
    parser.add_argument("--validation-output", default=None, metavar="DIR", help="Output directory used by --validate-requirements.")
    return parser


def _parse_int_tuple(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    return values or None


def _parse_float_tuple(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    return values or None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    operations = [
        args.audit_run,
        args.verify_evidence,
        args.export_evidence,
        args.cost_input,
        args.validate_requirements,
    ]
    if sum(value is not None for value in operations) > 1:
        parser.error(
            "choose only one operation: audit, export, verify, cost or requirement validation"
        )

    if args.verify_evidence is not None:
        report = verify_evidence_package(args.verify_evidence)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["status"] != "passed":
            raise SystemExit(1)
        return

    if args.export_evidence is not None:
        if not args.evidence_output:
            parser.error("--export-evidence requires --evidence-output")
        report = export_evidence_package(
            args.export_evidence,
            args.evidence_output,
            archive_path=args.evidence_archive,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.cost_input is not None:
        if not args.tariffs or not args.cost_output:
            parser.error("--cost-input requires --tariffs and --cost-output")
        report = write_cost_scenarios(args.cost_input, args.tariffs, args.cost_output)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.validate_requirements is not None:
        if not args.validation_context or not args.validation_output:
            parser.error(
                "--validate-requirements requires --validation-context and --validation-output"
            )
        report = write_requirement_validation(
            args.validate_requirements,
            args.validation_context,
            args.validation_output,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["status"] == "failed":
            raise SystemExit(1)
        return

    if args.audit_run is not None:
        from .canonicalization_audit import (
            AUDIT_FILENAME as CANONICALIZATION_AUDIT_FILENAME,
        )
        from .canonicalization_audit import write_canonicalization_audit
        from .denominator_audit import AUDIT_FILENAME as DENOMINATOR_AUDIT_FILENAME
        from .denominator_audit import audit_run_directory, read_piece_metric_rows

        run_dir = Path(args.audit_run)
        output_dir = Path(args.audit_output) if args.audit_output else run_dir.parent / "audits" / run_dir.name
        report = write_audit_reports(run_dir, output_dir)
        denominators = audit_run_directory(run_dir)
        denominator_path = Path(report["audit_path"]).parent / DENOMINATOR_AUDIT_FILENAME
        denominator_path.write_text(json.dumps(denominators, indent=2, ensure_ascii=False), encoding="utf-8")
        report["denominator_audit_path"] = str(denominator_path)
        metrics_path = run_dir / "piece_metrics_raw.csv"
        if metrics_path.exists():
            # Una fila por (pieza, modelo, celda): basta una por pieza para el
            # informe de agrupamiento, y los CSV guardados no traen tokens, así
            # que la huella melódica queda vacía en corridas ya terminadas.
            unique_pieces = {}
            for row in read_piece_metric_rows(metrics_path):
                unique_pieces.setdefault(row.get("piece_id"), row)
            canonicalization = write_canonicalization_audit(
                unique_pieces.values(),
                denominator_path.parent / CANONICALIZATION_AUDIT_FILENAME,
            )
            report["canonicalization_audit_path"] = str(
                denominator_path.parent / CANONICALIZATION_AUDIT_FILENAME
            )
            report["canonicalization"] = {
                "n_files": canonicalization["n_files"],
                "n_canonical_works": canonicalization["n_canonical_works"],
                "n_review_required": len(canonicalization["review_required"]),
            }
        report["denominators"] = {
            key: denominators.get(key)
            for key in ("status", "n_scored_files", "n_canonical_works", "n_files_absorbed_by_grouping", "explanation")
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    # El runner y el stack numérico se cargan sólo para una corrida. Las
    # utilidades de auditoría y empaquetado funcionan en una instalación mínima.
    from .config import build_default_learning_curve_config
    from .runner import run_learning_curve_experiment

    config = build_default_learning_curve_config(
        corpus_root=args.corpus_root,
        results_root=args.results_root,
    )
    experiment = config.experiment

    if args.transformer_max_epochs is not None:
        experiment = replace(
            experiment,
            transformer=replace(experiment.transformer, max_epochs=args.transformer_max_epochs),
        )
    if args.train_stride is not None:
        if args.train_stride <= 0:
            parser.error("--train-stride must be positive")
        experiment = replace(
            experiment,
            windows=replace(experiment.windows, train_stride=args.train_stride),
        )
    if args.transformer_device is not None:
        experiment = replace(
            experiment,
            hardware=replace(experiment.hardware, target_device=args.transformer_device, gpu_required=args.transformer_device == "cuda"),
        )

    updates = {
        "experiment": experiment,
        "include_vomm_control": not args.without_vomm,
        "structural_annotations_path": args.structural_annotations,
    }
    parsed_data_seeds = _parse_int_tuple(args.data_seeds)
    if parsed_data_seeds is not None:
        updates["data_seeds"] = parsed_data_seeds
    parsed_model_seeds = _parse_int_tuple(args.model_seeds)
    if parsed_model_seeds is not None:
        updates["model_seeds"] = parsed_model_seeds
    parsed_fractions = _parse_float_tuple(args.fractions)
    if parsed_fractions is not None:
        updates["train_fractions"] = parsed_fractions
    if args.split_seed is not None:
        updates["split_seed"] = args.split_seed
        experiment = replace(
            experiment,
            split=replace(experiment.split, seed=args.split_seed),
        )
        updates["experiment"] = experiment
    parsed_states = _parse_int_tuple(args.finite_hmm_states)
    if parsed_states is not None:
        updates["finite_hmm_states"] = parsed_states
    if args.finite_hmm_max_iterations is not None:
        updates["finite_hmm_max_iterations"] = args.finite_hmm_max_iterations

    config = replace(config, **updates)
    result = run_learning_curve_experiment(
        config,
        run_name=args.run_name,
        max_files=args.max_files,
        plan_only=args.plan_only,
        n_workers=args.n_workers,
        corpus_cache_path=args.corpus_cache,
        corpus_sample_seed=args.corpus_sample_seed,
        resume=args.resume,
        resource_measurement_condition=args.resource_measurement_condition,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

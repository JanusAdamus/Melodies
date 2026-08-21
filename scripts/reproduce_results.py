from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.generate_example_score import build_demo_score
from next_token_experiment.cli import main as next_token_main
from src.cli.main import main as classical_main
from src.project_paths import external_root, project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal reproducible Melodies workflow from the repository root.")
    parser.add_argument(
        "--output-root",
        default=str(project_root() / "artifacts" / "outputs" / "reproducibility_demo"),
        help="Directory where reproducible demo outputs will be written.",
    )
    parser.add_argument(
        "--skip-next-token",
        action="store_true",
        help="Skip the next-token demo even if the required corpus is available.",
    )
    return parser


def _run_classical_demo(output_root: Path) -> None:
    example_path = project_root() / "examples" / "example_score.musicxml"
    if not example_path.exists():
        example_path.parent.mkdir(parents=True, exist_ok=True)
        build_demo_score().write("musicxml", fp=str(example_path))

    sys.argv = [
        "melodies-analyze",
        "--input",
        str(example_path),
        "--obs",
        "pitch_class",
        "--model",
        "both",
        "--output-dir",
        str(output_root / "single_piece_analysis"),
    ]
    classical_main()


def _run_next_token_demo() -> bool:
    corpus_root = external_root() / "library" / "scores"
    if not corpus_root.exists():
        print(
            "Skipping next-token demo: missing corpus at "
            f"{corpus_root}. Restore external/library/scores to enable it."
        )
        return False

    sys.argv = [
        "melodies-next-token",
        "--profile",
        "cpu_baseline",
        "--run-name",
        "reproducibility_demo_cpu_baseline",
        "--results-root",
        str(project_root() / "artifacts" / "next_token_experiment" / "results"),
        "--max-files",
        "8",
        "--max-windows-train",
        "64",
        "--max-windows-validation",
        "16",
        "--max-windows-test",
        "16",
    ]
    next_token_main()
    return True


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root()}")
    print(f"Writing demo outputs under: {output_root}")
    _run_classical_demo(output_root)
    print("Classical single-piece demo completed.")

    if not args.skip_next_token:
        ran_next_token = _run_next_token_demo()
        if ran_next_token:
            print("Next-token demo completed.")


if __name__ == "__main__":
    main()

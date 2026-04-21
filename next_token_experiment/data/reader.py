from __future__ import annotations

from pathlib import Path

from ..config import CorpusConfig


def discover_score_paths(config: CorpusConfig) -> list[Path]:
    """Collect score files for the configured corpus."""

    root = Path(config.root_dir)
    extensions = {extension.lower() for extension in config.extensions}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def build_piece_id(path: Path, root_dir: str) -> str:
    """Build a stable, relative piece id for manifests and outputs."""

    root = Path(root_dir)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name

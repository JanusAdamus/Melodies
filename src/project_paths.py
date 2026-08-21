from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root from a stable location inside `src/`."""

    return Path(__file__).resolve().parents[1]


def artifacts_root() -> Path:
    return project_root() / "artifacts"


def external_root() -> Path:
    return project_root() / "external"


def default_thesis_export_root() -> Path:
    """Resolve a thesis export directory without relying on machine-specific paths."""

    env_value = os.environ.get("MELODIES_THESIS_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return artifacts_root() / "thesis_export"

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .library_catalog import MUSICXML_EXTENSIONS, classify_score, iter_musicxml_files


@dataclass(frozen=True)
class CorpusSource:
    """Describe una fuente externa de partituras simbolicas."""

    name: str
    source_type: str
    root_dir: str | Path


def _slug_to_title(text: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", text).strip()
    return cleaned.title() if cleaned else ""


def _limit_paths(paths: list[Path], file_limit: int | None) -> list[Path]:
    if file_limit is None:
        return paths
    return paths[: max(file_limit, 0)]


def _safe_read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_symbtr_filename(path: str | Path) -> dict[str, str]:
    """Extrae el bloque semantico principal del nombre de archivo de SymbTr."""

    stem = Path(path).stem
    parts = stem.split("--")
    padded = (parts + [""] * 5)[:5]
    makam, form, usul, title_slug, composer_slug = padded
    return {
        "makam": makam,
        "form": form or "piece",
        "usul": usul,
        "title_hint": _slug_to_title(title_slug) or stem,
        "composer_hint": _slug_to_title(composer_slug) or "Unknown",
    }


def build_symbtr_catalog(root_dir: str | Path, file_limit: int | None = None) -> pd.DataFrame:
    """Construye un catalogo enriquecido para SymbTr."""

    root = Path(root_dir)
    musicxml_dir = root / "MusicXML" if (root / "MusicXML").exists() else root
    rows: list[dict[str, object]] = []
    for path in _limit_paths(iter_musicxml_files(musicxml_dir), file_limit):
        parsed = parse_symbtr_filename(path)
        row = classify_score(path)
        row.update(
            {
                "source_name": "SymbTr",
                "source_type": "symbtr",
                "genre_family": "turkish_art_music",
                "style_system": "makam",
                "modal_system": parsed["makam"],
                "makam": parsed["makam"],
                "usul": parsed["usul"],
                "symbtr_form": parsed["form"],
                "composer": parsed["composer_hint"] if row.get("composer") in {"Unknown", ""} else row.get("composer"),
                "form": parsed["form"] or row.get("form", "piece"),
                "title": row.get("title") or parsed["title_hint"],
                "ingest_note": "Metadatos principales inferidos desde el nombre canónico SymbTr.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _looks_like_pdmx_manifest(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".csv" and ("pdmx" in name or "metadata" in name or "dataset" in name)


def find_pdmx_manifest(root_dir: str | Path) -> Path | None:
    root = Path(root_dir)
    manifests = sorted(path for path in root.rglob("*.csv") if _looks_like_pdmx_manifest(path))
    return manifests[0] if manifests else None


def _resolve_pdmx_musicxml_path(root_dir: Path, row: pd.Series) -> Path | None:
    candidates: list[Path] = []
    for key in ("mxl_path", "musicxml_path", "mxml_path", "xml_path", "path"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root_dir / candidate
        candidates.append(candidate)
        if candidate.suffix.lower() in MUSICXML_EXTENSIONS and candidate.exists():
            return candidate
        for suffix in (".mxl", ".musicxml", ".xml"):
            alt = candidate.with_suffix(suffix)
            if alt.exists():
                return alt
    return next((path for path in candidates if path.exists()), None)


def build_pdmx_catalog(root_dir: str | Path, subset: str = "no_license_conflict", file_limit: int | None = None) -> pd.DataFrame:
    """Integra PDMX cuando el usuario ya descargo localmente el dataset desde Zenodo."""

    root = Path(root_dir)
    manifest = find_pdmx_manifest(root)
    if manifest is None:
        rows = []
        for path in _limit_paths(iter_musicxml_files(root), file_limit):
            row = classify_score(path)
            row.update(
                {
                    "source_name": "PDMX",
                    "source_type": "pdmx",
                    "genre_family": "mixed",
                    "style_system": "mixed_public_domain",
                    "modal_system": "",
                    "ingest_note": "No se encontro un manifiesto PDMX; se uso descubrimiento recursivo de MusicXML.",
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    manifest_df = pd.read_csv(manifest)
    subset_column = f"subset:{subset}"
    if subset_column in manifest_df.columns:
        filtered = manifest_df[manifest_df[subset_column].fillna(False)].copy()
    elif subset in manifest_df.columns:
        filtered = manifest_df[manifest_df[subset].fillna(False)].copy()
    else:
        filtered = manifest_df.copy()

    rows: list[dict[str, object]] = []
    if file_limit is not None:
        filtered = filtered.head(file_limit).copy()
    for _, meta in filtered.iterrows():
        score_path = _resolve_pdmx_musicxml_path(root, meta)
        if score_path is None:
            continue
        row = classify_score(score_path)
        row.update(
            {
                "source_name": "PDMX",
                "source_type": "pdmx",
                "genre_family": str(meta.get("genre", "")) or "mixed",
                "style_system": "mixed_public_domain",
                "modal_system": "",
                "license_conflict": meta.get("license_conflict", ""),
                "pdmx_subset": subset,
                "ingest_note": f"Catalogado desde manifiesto PDMX ({manifest.name}).",
            }
        )
        for src_key, dst_key in (
            ("title", "title"),
            ("song_name", "title"),
            ("composer", "composer"),
            ("artist_name", "composer"),
            ("rating", "rating"),
            ("genres", "genre_family"),
        ):
            value = meta.get(src_key)
            if pd.notna(value) and str(value).strip():
                row[dst_key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _extract_jazzmus_musicxml_payload(payload: dict) -> str:
    encodings = payload.get("encodings", {})
    if isinstance(encodings, dict):
        return str(encodings.get("musicxml") or encodings.get("MusicXML") or "")
    if isinstance(encodings, list):
        for item in encodings:
            if not isinstance(item, dict):
                continue
            if "musicxml" in item:
                return str(item["musicxml"])
            if "MusicXML" in item:
                return str(item["MusicXML"])
    return ""


def prepare_jazzmus_musicxml(json_dir: str | Path, output_dir: str | Path, overwrite: bool = False) -> pd.DataFrame:
    """Exporta MusicXML desde los JSON de JAZZMUS para que el pipeline actual pueda analizarlos."""

    json_root = Path(json_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for path in sorted(json_root.glob("*.json")):
        payload = _safe_read_json(path)
        musicxml = _extract_jazzmus_musicxml_payload(payload)
        target = output_root / f"{path.stem}.musicxml"
        status = "skipped"
        if musicxml:
            if overwrite or not target.exists():
                target.write_text(musicxml, encoding="utf-8")
                status = "written"
            else:
                status = "existing"
        rows.append(
            {
                "json_path": str(path),
                "musicxml_path": str(target),
                "status": status if musicxml else "missing_musicxml_encoding",
            }
        )
    return pd.DataFrame(rows)


def build_jazzmus_catalog(root_dir: str | Path, file_limit: int | None = None) -> pd.DataFrame:
    """Construye un catalogo para JAZZMUS a partir de MusicXML ya exportado o disponible."""

    root = Path(root_dir)
    rows: list[dict[str, object]] = []
    for path in _limit_paths(iter_musicxml_files(root), file_limit):
        row = classify_score(path)
        row.update(
            {
                "source_name": "JAZZMUS",
                "source_type": "jazzmus",
                "genre_family": "jazz",
                "style_system": "lead_sheet_jazz",
                "modal_system": "",
                "ingest_note": "Lead sheet / partitura jazz integrada al pipeline armonico.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_generic_catalog(root_dir: str | Path, source_name: str = "generic", file_limit: int | None = None) -> pd.DataFrame:
    rows = []
    for path in _limit_paths(iter_musicxml_files(root_dir), file_limit):
        row = classify_score(path)
        row.update(
            {
                "source_name": source_name,
                "source_type": "generic",
                "genre_family": "unknown",
                "style_system": "unknown",
                "modal_system": "",
                "ingest_note": "Catalogo generico de MusicXML/MXL.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_source_catalog(
    source: CorpusSource,
    staging_dir: str | Path | None = None,
    file_limit: int | None = None,
) -> pd.DataFrame:
    source_type = source.source_type.lower()
    if source_type == "symbtr":
        return build_symbtr_catalog(source.root_dir, file_limit=file_limit)
    if source_type == "pdmx":
        return build_pdmx_catalog(source.root_dir, file_limit=file_limit)
    if source_type == "jazzmus":
        root = Path(source.root_dir)
        if not any(root.glob("*.musicxml")) and any(root.glob("*.json")) and staging_dir is not None:
            prepared_dir = Path(staging_dir) / "prepared_jazzmus_musicxml"
            prepare_jazzmus_musicxml(root, prepared_dir)
            return build_jazzmus_catalog(prepared_dir, file_limit=file_limit)
        return build_jazzmus_catalog(root, file_limit=file_limit)
    return build_generic_catalog(source.root_dir, source_name=source.name, file_limit=file_limit)


def build_multicorpus_catalog(
    sources: Iterable[CorpusSource],
    staging_dir: str | Path | None = None,
    file_limit_per_source: int | None = None,
) -> pd.DataFrame:
    """Concatena catalogos de varias fuentes con columnas comunes para el analisis."""

    frames = [build_source_catalog(source, staging_dir=staging_dir, file_limit=file_limit_per_source) for source in sources]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    catalog = pd.concat(frames, ignore_index=True, sort=False)
    if "source_name" not in catalog.columns:
        catalog["source_name"] = "unknown"
    return catalog.sort_values(["source_name", "composer", "title"]).reset_index(drop=True)

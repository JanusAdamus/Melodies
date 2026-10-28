from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis.diagnostics import compare_models, export_tables
from src.analysis.interpretation import interpret_hdp_states
from src.analysis.library_visualization import build_analysis_explanation, generate_analysis_figures, generate_catalog_figures
from src.data.dedup import content_fingerprint, deduplicate_by_fingerprint
from src.data.library_catalog import build_library_catalog, summarize_catalog
from src.data.multicorpus import CorpusSource, build_multicorpus_catalog
from src.data.observations import build_observation_sequence
from src.data.parsing import MusicalEvent, extract_events_for_mode, parse_score
from src.models.finite_hmm import FiniteChordHMM
from src.models.hdp_hmm import TruncatedHDPHMM


def _top_labels(labels: list[str], top_n: int = 5) -> str:
    counts = Counter(labels)
    return ", ".join(f"{label} ({count})" for label, count in counts.most_common(top_n))


@dataclass(frozen=True)
class PreparedCorpus:
    """Obras listas para analizar, mas todo lo que se quedo fuera y por que."""

    pieces: pd.DataFrame
    events: dict[str, list[MusicalEvent]]
    failures: pd.DataFrame
    duplicates: pd.DataFrame


def _failure(row: pd.Series, stage: str, error: str) -> dict[str, object]:
    return {**row.to_dict(), "error": error, "error_stage": stage}


def _extract_piece_events(path: str, extraction: str, prefer_treble: bool) -> list[MusicalEvent]:
    events = extract_events_for_mode(parse_score(path), extraction, prefer_treble=prefer_treble)
    if not events:
        raise ValueError("La extraccion no produjo eventos.")
    missing = sum(event.representative_pitch_class is None for event in events)
    if missing:
        raise ValueError(f"{missing} eventos sin altura (p. ej. NoChord).")
    return events


def prepare_corpus(catalog: pd.DataFrame, extraction: str = "melody", prefer_treble: bool = False) -> PreparedCorpus:
    """Extrae cada obra una vez, cuenta los errores de catalogo y de extraccion juntos y
    quita duplicados por huella de contenido antes de cualquier ajuste (D3, D4)."""

    failures: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    events_by_path: dict[str, list[MusicalEvent]] = {}
    errors = catalog["error"].fillna("") if "error" in catalog.columns else pd.Series("", index=catalog.index)
    for index, row in catalog.iterrows():
        if errors[index]:
            failures.append(_failure(row, "catalogo", str(errors[index])))
            continue
        path = str(row["path"])
        try:
            events = _extract_piece_events(path, extraction, prefer_treble)
        except Exception as exc:
            failures.append(_failure(row, "extraccion", f"{exc.__class__.__name__}: {exc}"))
            continue
        events_by_path[path] = events
        fingerprint = content_fingerprint(event.representative_pitch_class for event in events)
        rows.append({**row.to_dict(), "extraction": extraction, "n_events": len(events), "content_fingerprint": fingerprint})

    columns = list(catalog.columns) + ["extraction", "n_events", "content_fingerprint"]
    kept, duplicates = deduplicate_by_fingerprint(pd.DataFrame(rows, columns=columns))
    kept_events = {path: events_by_path[path] for path in kept["path"]}
    return PreparedCorpus(
        pieces=kept,
        events=kept_events,
        failures=pd.DataFrame(failures, columns=list(catalog.columns) + ["error_stage"]),
        duplicates=duplicates,
    )


def analyze_catalog_piece(
    row: pd.Series,
    events: list[MusicalEvent],
    obs_type: str,
    model: str,
    hdp_params: dict,
) -> dict[str, object]:
    observations = build_observation_sequence(events, obs_type)
    record = row.to_dict()
    record.update({"observation_type": obs_type, "n_events": len(events)})

    finite_result = None
    hdp_result = None

    if model in {"finite_hmm", "both"}:
        finite_observations = observations if observations.observation_type == "pitch_class" else build_observation_sequence(events, "pitch_class")
        finite_result = FiniteChordHMM().fit_predict(finite_observations)
        record.update(
            {
                "finite_log_likelihood": finite_result.log_likelihood,
                "finite_active_states": len(finite_result.active_states),
                "finite_effective_states": finite_result.effective_states,
                "finite_top_labels": _top_labels(finite_result.latent_labels),
                "finite_top_modes": _top_labels(finite_result.modal_labels),
            }
        )

    if model in {"hdp_hmm", "both"}:
        hdp_result = TruncatedHDPHMM(**hdp_params).fit(observations)
        interpretation = interpret_hdp_states(hdp_result)
        record.update(
            {
                "hdp_log_likelihood": hdp_result.log_likelihood,
                "hdp_effective_states": hdp_result.effective_states,
                "hdp_top_interpretations": ", ".join(interpretation["tentative_label"].head(3).tolist()),
            }
        )

    comparison = compare_models(finite_result, hdp_result)
    if not comparison.empty:
        for _, comp_row in comparison.iterrows():
            prefix = comp_row["model"]
            record[f"{prefix}_mean_segment_length"] = comp_row["mean_segment_length"]
            record[f"{prefix}_trajectory_stability"] = comp_row["trajectory_stability"]

    return record


def analyze_library(
    library_dir: str | Path,
    output_dir: str | Path,
    obs_type: str = "pitch_class",
    model: str = "finite_hmm",
    limit: int | None = None,
    prefer_treble: bool = False,
    hdp_params: dict | None = None,
    extraction: str = "melody",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = build_library_catalog(library_dir)
    return analyze_catalog(
        catalog=catalog,
        output_dir=output_dir,
        obs_type=obs_type,
        model=model,
        limit=limit,
        prefer_treble=prefer_treble,
        hdp_params=hdp_params,
        extraction=extraction,
    )


def analyze_catalog(
    catalog: pd.DataFrame,
    output_dir: str | Path,
    obs_type: str = "pitch_class",
    model: str = "finite_hmm",
    limit: int | None = None,
    prefer_treble: bool = False,
    hdp_params: dict | None = None,
    extraction: str = "melody",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_tables = summarize_catalog(catalog)
    export_tables(catalog_tables, output_dir / "catalog", excel_name="catalog.xlsx")
    generate_catalog_figures(catalog, output_dir / "catalog" / "figures")

    prepared = prepare_corpus(catalog, extraction=extraction, prefer_treble=prefer_treble)
    export_tables({"duplicates_removed": prepared.duplicates}, output_dir / "catalog", excel_name="duplicates_removed.xlsx")
    analyzable_catalog = prepared.pieces
    if not analyzable_catalog.empty and "note_count" in analyzable_catalog.columns:
        analyzable_catalog = analyzable_catalog.sort_values(["note_count", "title"]).reset_index(drop=True)
    if limit is not None:
        analyzable_catalog = analyzable_catalog.head(limit).reset_index(drop=True)

    hdp_params = hdp_params or {
        "n_states": 20,
        "n_iters": 80,
        "burn_in": 40,
        "seed": 7,
    }

    records = prepared.failures.to_dict("records")
    for _, row in analyzable_catalog.iterrows():
        try:
            records.append(analyze_catalog_piece(row, prepared.events[row["path"]], obs_type=obs_type, model=model, hdp_params=hdp_params))
        except Exception as exc:
            records.append(_failure(row, "analisis", f"{exc.__class__.__name__}: {exc}"))

    analysis_df = pd.DataFrame(records)
    tables = {
        "analysis": analysis_df,
    }
    if not analysis_df.empty and "composer" in analysis_df.columns:
        numeric = analysis_df.select_dtypes(include=["number"])
        if not numeric.empty:
            summary = analysis_df.groupby("composer")[numeric.columns].mean(numeric_only=True).reset_index()
            tables["analysis_by_composer"] = summary
    export_tables(tables, output_dir / "analysis", excel_name="analysis.xlsx")
    generate_analysis_figures(analysis_df, output_dir / "analysis" / "figures")
    (output_dir / "analysis" / "analysis_report.md").write_text(build_analysis_explanation(analysis_df), encoding="utf-8")

    return {
        "catalog_dir": output_dir / "catalog",
        "analysis_dir": output_dir / "analysis",
    }


def analyze_multicorpus(
    sources: list[CorpusSource],
    output_dir: str | Path,
    obs_type: str = "pitch_class",
    model: str = "finite_hmm",
    limit: int | None = None,
    file_limit_per_source: int | None = None,
    prefer_treble: bool = False,
    hdp_params: dict | None = None,
    extraction: str = "melody",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = build_multicorpus_catalog(sources, staging_dir=output_dir, file_limit_per_source=file_limit_per_source)
    return analyze_catalog(
        catalog=catalog,
        output_dir=output_dir,
        obs_type=obs_type,
        model=model,
        limit=limit,
        prefer_treble=prefer_treble,
        hdp_params=hdp_params,
        extraction=extraction,
    )

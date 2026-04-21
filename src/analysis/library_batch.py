from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from src.analysis.diagnostics import compare_models, export_tables
from src.analysis.interpretation import interpret_hdp_states
from src.analysis.library_visualization import build_analysis_explanation, generate_analysis_figures, generate_catalog_figures
from src.data.library_catalog import build_library_catalog, summarize_catalog
from src.data.multicorpus import CorpusSource, build_multicorpus_catalog
from src.data.observations import build_observation_sequence
from src.data.parsing import extract_events, parse_score
from src.models.finite_hmm import FiniteChordHMM
from src.models.hdp_hmm import TruncatedHDPHMM


def _top_labels(labels: list[str], top_n: int = 5) -> str:
    counts = Counter(labels)
    return ", ".join(f"{label} ({count})" for label, count in counts.most_common(top_n))


def analyze_catalog_piece(
    row: pd.Series,
    obs_type: str,
    model: str,
    prefer_treble: bool,
    hdp_params: dict,
) -> dict[str, object]:
    score = parse_score(row["path"])
    events = extract_events(score, prefer_treble=prefer_treble)
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
    )


def analyze_catalog(
    catalog: pd.DataFrame,
    output_dir: str | Path,
    obs_type: str = "pitch_class",
    model: str = "finite_hmm",
    limit: int | None = None,
    prefer_treble: bool = False,
    hdp_params: dict | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_tables = summarize_catalog(catalog)
    export_tables(catalog_tables, output_dir / "catalog", excel_name="catalog.xlsx")
    generate_catalog_figures(catalog, output_dir / "catalog" / "figures")

    analyzable_catalog = catalog[catalog["error"].fillna("") == ""].copy()
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

    records = []
    for _, row in analyzable_catalog.iterrows():
        try:
            records.append(analyze_catalog_piece(row, obs_type=obs_type, model=model, prefer_treble=prefer_treble, hdp_params=hdp_params))
        except Exception as exc:
            failed = row.to_dict()
            failed["error"] = f"{exc.__class__.__name__}: {exc}"
            records.append(failed)

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
    )

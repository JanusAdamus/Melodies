from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:  # pragma: no cover
    sns = None
    HAS_SEABORN = False


def _ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _apply_theme() -> None:
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="notebook")
        sns.set_context("notebook", rc={"axes.titlesize": 18, "axes.labelsize": 12})
    else:
        plt.style.use("default")


def _save_current(path: Path) -> Path:
    if HAS_SEABORN:
        sns.despine()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _annotate_bars(ax: plt.Axes, horizontal: bool = False) -> None:
    for patch in ax.patches:
        if horizontal:
            value = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            ax.text(value, y, f" {value:.0f}", va="center", ha="left", fontsize=9)
        else:
            value = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2
            ax.text(x, value, f"{value:.0f}", va="bottom", ha="center", fontsize=9)


def _clean_label(text: object) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("_", " ").replace(".musicxml", "").replace(".mxl", "").replace(".xml", "")
    cleaned = " ".join(cleaned.split())
    return cleaned


def _short_label(text: object, max_len: int = 32) -> str:
    cleaned = _clean_label(text)
    if len(cleaned) <= max_len:
        return cleaned
    return textwrap.shorten(cleaned, width=max_len, placeholder="...")


def _finalize_axes(ax: plt.Axes, x_grid: bool = True, y_grid: bool = False) -> None:
    ax.grid(axis="x" if x_grid else "both", color="#d9d9d9", linewidth=0.8)
    if y_grid:
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    else:
        ax.grid(axis="y", visible=False)


def _legend_or_none(ax: plt.Axes, title: str | None = None) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    if title is not None:
        legend.set_title(title)
    legend.get_frame().set_alpha(0.95)


def generate_catalog_figures(catalog: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = _ensure_dir(output_dir)
    paths: list[Path] = []
    if catalog.empty:
        return paths
    _apply_theme()

    composer_counts = (
        catalog.groupby("composer")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="count")
    )
    composer_counts["composer"] = composer_counts["composer"].map(lambda value: _short_label(value, max_len=26))
    plt.figure(figsize=(9, 5.5))
    if HAS_SEABORN:
        ax = sns.barplot(data=composer_counts, x="count", y="composer", color="#4c78a8")
    else:
        ax = plt.gca()
        ax.barh(composer_counts["composer"], composer_counts["count"], color="#4c78a8")
    _annotate_bars(ax, horizontal=True)
    _finalize_axes(ax)
    plt.title("Compositores mas representados")
    plt.xlabel("Numero de obras")
    plt.ylabel("Compositor")
    paths.append(_save_current(output_dir / "catalog_by_composer.png"))

    difficulty_counts = (
        catalog.groupby("difficulty_bucket")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )
    plt.figure(figsize=(8, 4))
    if HAS_SEABORN:
        ax = sns.barplot(data=difficulty_counts, x="difficulty_bucket", y="count", color="#f58518")
    else:
        ax = plt.gca()
        ax.bar(difficulty_counts["difficulty_bucket"], difficulty_counts["count"], color="#f58518")
    _annotate_bars(ax)
    _finalize_axes(ax, x_grid=False, y_grid=True)
    plt.title("Distribucion de dificultad")
    plt.xlabel("Dificultad")
    plt.ylabel("Numero de obras")
    paths.append(_save_current(output_dir / "catalog_by_difficulty.png"))

    form_counts = (
        catalog.groupby("form")
        .size()
        .sort_values(ascending=False)
        .head(8)
        .reset_index(name="count")
    )
    form_counts["form"] = form_counts["form"].map(lambda value: _short_label(value, max_len=18))
    plt.figure(figsize=(8.5, 4.5))
    if HAS_SEABORN:
        ax = sns.barplot(data=form_counts, x="count", y="form", color="#54a24b")
    else:
        ax = plt.gca()
        ax.barh(form_counts["form"], form_counts["count"], color="#54a24b")
    _annotate_bars(ax, horizontal=True)
    _finalize_axes(ax)
    plt.title("Formas mas frecuentes")
    plt.xlabel("Numero de obras")
    plt.ylabel("Forma")
    paths.append(_save_current(output_dir / "catalog_by_form.png"))

    if "source_name" in catalog.columns:
        source_counts = (
            catalog.groupby("source_name")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="count")
        )
        plt.figure(figsize=(8, 4))
        if HAS_SEABORN:
            ax = sns.barplot(data=source_counts, x="source_name", y="count", color="#e45756")
        else:
            ax = plt.gca()
            ax.bar(source_counts["source_name"], source_counts["count"], color="#e45756")
        _annotate_bars(ax)
        _finalize_axes(ax, x_grid=False, y_grid=True)
        plt.title("Obras por fuente")
        plt.xlabel("Fuente")
        plt.ylabel("Numero de obras")
        paths.append(_save_current(output_dir / "catalog_by_source.png"))

    if {"source_name", "form"}.issubset(catalog.columns):
        source_form = (
            catalog.groupby(["source_name", "form"])
            .size()
            .reset_index(name="count")
            .pivot(index="source_name", columns="form", values="count")
            .fillna(0.0)
        )
        if not source_form.empty:
            top_forms = source_form.sum(axis=0).sort_values(ascending=False).head(6).index
            mix = source_form.loc[:, top_forms].copy()
            mix.columns = [_short_label(col, max_len=14) for col in mix.columns]
            mix = mix.div(mix.sum(axis=1).replace(0, 1), axis=0)
            ax = mix.plot(
                kind="barh",
                stacked=True,
                figsize=(9, 4.5),
                colormap="Set2",
                width=0.65,
            )
            ax.legend(title="Forma", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            _finalize_axes(ax)
            plt.title("Mezcla de formas por fuente")
            plt.xlabel("Proporcion dentro de la fuente")
            plt.ylabel("Fuente")
            paths.append(_save_current(output_dir / "catalog_source_form_heatmap.png"))

    if {"note_count", "measure_count"}.issubset(catalog.columns):
        scatter_df = catalog.dropna(subset=["note_count", "measure_count"]).copy()
        if not scatter_df.empty:
            plt.figure(figsize=(8.5, 5.5))
            ax = plt.gca()
            if HAS_SEABORN and "source_name" in scatter_df.columns:
                ax = sns.scatterplot(
                    data=scatter_df,
                    x="measure_count",
                    y="note_count",
                    hue="source_name",
                    palette="Set2",
                    s=55,
                    alpha=0.8,
                )
            else:
                ax.scatter(scatter_df["measure_count"], scatter_df["note_count"], alpha=0.7, s=50, color="#4c78a8")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _legend_or_none(ax, title="Fuente")
            _finalize_axes(ax, x_grid=True, y_grid=True)
            plt.title("Compases vs notas")
            plt.xlabel("Numero de compases (escala log)")
            plt.ylabel("Numero de notas (escala log)")
            paths.append(_save_current(output_dir / "catalog_piece_size_scatter.png"))
    return paths


def generate_analysis_figures(analysis: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = _ensure_dir(output_dir)
    paths: list[Path] = []
    if analysis.empty:
        return paths
    _apply_theme()

    if {"finite_log_likelihood", "hdp_log_likelihood"}.issubset(analysis.columns):
        comparable = analysis.dropna(subset=["finite_log_likelihood", "hdp_log_likelihood"]).copy()
        if not comparable.empty:
            comparable["ll_gain_hdp_minus_finite"] = comparable["hdp_log_likelihood"] - comparable["finite_log_likelihood"]
            comparable = comparable.sort_values("ll_gain_hdp_minus_finite", ascending=False).reset_index(drop=True)

            plt.figure(figsize=(7.2, 5.8))
            ax = plt.gca()
            if HAS_SEABORN:
                ax = sns.scatterplot(
                    data=comparable,
                    x="finite_log_likelihood",
                    y="hdp_log_likelihood",
                    hue="source_name" if "source_name" in comparable.columns else None,
                    palette="Set2",
                    s=65,
                    alpha=0.85,
                )
            else:
                ax.scatter(comparable["finite_log_likelihood"], comparable["hdp_log_likelihood"], s=60, color="#4c78a8")
            low = min(comparable["finite_log_likelihood"].min(), comparable["hdp_log_likelihood"].min())
            high = max(comparable["finite_log_likelihood"].max(), comparable["hdp_log_likelihood"].max())
            plt.plot([low, high], [low, high], linestyle="--", color="gray")
            _legend_or_none(ax, title="Fuente")
            _finalize_axes(ax, x_grid=True, y_grid=True)
            plt.title("Log-verosimilitud por obra")
            plt.xlabel("HMM finito")
            plt.ylabel("HDP-HMM truncado")
            paths.append(_save_current(output_dir / "compare_log_likelihood_scatter.png"))

            melted = comparable.melt(
                value_vars=["finite_log_likelihood", "hdp_log_likelihood"],
                var_name="model",
                value_name="log_likelihood",
            )
            melted["model"] = melted["model"].map(
                {
                    "finite_log_likelihood": "HMM finito",
                    "hdp_log_likelihood": "HDP-HMM truncado",
                }
            )
            plt.figure(figsize=(6.6, 4.8))
            if HAS_SEABORN:
                ax = sns.boxplot(
                    data=melted,
                    x="model",
                    y="log_likelihood",
                    hue="model",
                    palette=["#9ecae9", "#3182bd"],
                    legend=False,
                )
                sns.stripplot(data=melted, x="model", y="log_likelihood", color="#444444", alpha=0.35, size=4)
            else:
                groups = [melted[melted["model"] == model]["log_likelihood"] for model in ["HMM finito", "HDP-HMM truncado"]]
                plt.boxplot(groups, labels=["HMM finito", "HDP-HMM truncado"])
                ax = plt.gca()
            _finalize_axes(ax, x_grid=False, y_grid=True)
            plt.title("Distribucion de log-verosimilitud")
            plt.xlabel("Modelo")
            plt.ylabel("Log-verosimilitud")
            paths.append(_save_current(output_dir / "compare_log_likelihood_boxplot.png"))

            top_n = min(15, len(comparable))
            gain_df = comparable.head(top_n).copy()
            gain_df["title_short"] = gain_df["title"].map(lambda value: _short_label(value, max_len=28))
            plt.figure(figsize=(9, max(4.5, top_n * 0.32)))
            colors = ["#1f77b4" if value >= 0 else "#d62728" for value in gain_df["ll_gain_hdp_minus_finite"]]
            ax = plt.gca()
            ax.barh(gain_df["title_short"], gain_df["ll_gain_hdp_minus_finite"], color=colors)
            plt.axvline(0.0, color="black", linewidth=1)
            plt.gca().invert_yaxis()
            _finalize_axes(ax)
            plt.title("Obras con mayor ganancia del HDP-HMM")
            plt.xlabel("HDP-HMM truncado - HMM finito")
            plt.ylabel("Obra")
            paths.append(_save_current(output_dir / "compare_log_likelihood_gain_by_piece.png"))

            if "source_name" in comparable.columns:
                source_gain = (
                    comparable.groupby("source_name")["ll_gain_hdp_minus_finite"]
                    .agg(["mean", "median", "count"])
                    .reset_index()
                    .sort_values("mean", ascending=False)
                )
                plt.figure(figsize=(7.2, 4.6))
                if HAS_SEABORN:
                    ax = sns.barplot(data=source_gain, x="source_name", y="mean", color="#72b7b2")
                else:
                    ax = plt.gca()
                    ax.bar(source_gain["source_name"], source_gain["mean"], color="#72b7b2")
                plt.axhline(0.0, color="black", linewidth=1)
                for idx, row in source_gain.iterrows():
                    ax.text(idx, row["mean"], f"n={int(row['count'])}", ha="center", va="bottom" if row["mean"] >= 0 else "top", fontsize=9)
                _finalize_axes(ax, x_grid=False, y_grid=True)
                plt.title("Ganancia media por fuente")
                plt.xlabel("Fuente")
                plt.ylabel("Ganancia media")
                paths.append(_save_current(output_dir / "source_log_likelihood_gain.png"))

    if {"finite_active_states", "hdp_effective_states"}.issubset(analysis.columns):
        comparable = analysis.dropna(subset=["finite_active_states", "hdp_effective_states"]).copy()
        if not comparable.empty:
            plt.figure(figsize=(7.2, 5.8))
            ax = plt.gca()
            if HAS_SEABORN:
                ax = sns.scatterplot(
                    data=comparable,
                    x="finite_active_states",
                    y="hdp_effective_states",
                    hue="source_name" if "source_name" in comparable.columns else None,
                    palette="Set2",
                    s=65,
                    alpha=0.85,
                )
            else:
                ax.scatter(comparable["finite_active_states"], comparable["hdp_effective_states"], s=60, color="#4c78a8")
            low = min(comparable["finite_active_states"].min(), comparable["hdp_effective_states"].min())
            high = max(comparable["finite_active_states"].max(), comparable["hdp_effective_states"].max())
            plt.plot([low, high], [low, high], linestyle="--", color="gray")
            _legend_or_none(ax, title="Fuente")
            _finalize_axes(ax, x_grid=True, y_grid=True)
            plt.title("Estados por obra")
            plt.xlabel("Estados activos del HMM finito")
            plt.ylabel("Estados efectivos del HDP-HMM truncado")
            paths.append(_save_current(output_dir / "compare_state_counts_scatter.png"))

            top_n = min(12, len(comparable))
            compact = comparable.sort_values("finite_active_states", ascending=False).head(top_n).copy()
            compact["title_short"] = compact["title"].map(lambda value: _short_label(value, max_len=30))
            positions = np.arange(len(compact))
            plt.figure(figsize=(8.5, max(4.6, top_n * 0.34)))
            ax = plt.gca()
            ax.hlines(positions, compact["hdp_effective_states"], compact["finite_active_states"], color="#c7c7c7", linewidth=2)
            ax.scatter(compact["finite_active_states"], positions, color="#f58518", s=50, label="HMM finito", zorder=3)
            ax.scatter(compact["hdp_effective_states"], positions, color="#4c78a8", s=50, label="HDP-HMM truncado", zorder=3)
            plt.yticks(positions, compact["title_short"])
            plt.gca().invert_yaxis()
            _finalize_axes(ax)
            plt.xlabel("Numero de estados")
            plt.ylabel("Obra")
            plt.title("Obras mas complejas")
            plt.legend(frameon=False, loc="lower right")
            paths.append(_save_current(output_dir / "compare_state_counts_by_piece.png"))

    if {"note_count", "finite_log_likelihood", "hdp_log_likelihood"}.issubset(analysis.columns):
        complexity_df = analysis.dropna(subset=["note_count", "finite_log_likelihood", "hdp_log_likelihood"]).copy()
        if not complexity_df.empty:
            complexity_df["ll_gain_hdp_minus_finite"] = complexity_df["hdp_log_likelihood"] - complexity_df["finite_log_likelihood"]
            plt.figure(figsize=(7.5, 5.4))
            ax = plt.gca()
            if HAS_SEABORN:
                ax = sns.scatterplot(
                    data=complexity_df,
                    x="note_count",
                    y="ll_gain_hdp_minus_finite",
                    hue="source_name" if "source_name" in complexity_df.columns else None,
                    palette="Set2",
                    s=65,
                    alpha=0.8,
                )
            else:
                ax.scatter(complexity_df["note_count"], complexity_df["ll_gain_hdp_minus_finite"], alpha=0.75, s=55, color="#4c78a8")
            plt.axhline(0.0, color="black", linewidth=1)
            ax.set_xscale("log")
            _legend_or_none(ax, title="Fuente")
            _finalize_axes(ax, x_grid=True, y_grid=True)
            plt.title("Ganancia vs tamano de obra")
            plt.xlabel("Numero de notas (escala log)")
            plt.ylabel("HDP-HMM truncado - HMM finito")
            paths.append(_save_current(output_dir / "note_count_vs_log_likelihood_gain.png"))

    if {"composer", "finite_active_states"}.issubset(analysis.columns):
        composer_summary = (
            analysis.groupby("composer")[["finite_active_states"] + ([col for col in ["hdp_effective_states"] if col in analysis.columns])]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("finite_active_states", ascending=False)
            .head(10)
        )
        if not composer_summary.empty:
            composer_summary["composer"] = composer_summary["composer"].map(lambda value: _short_label(value, max_len=24))
            plt.figure(figsize=(9, 5))
            plot_df = composer_summary.melt(id_vars="composer", var_name="metric", value_name="value")
            plot_df["metric"] = plot_df["metric"].map(
                {
                    "finite_active_states": "HMM finito",
                    "hdp_effective_states": "HDP-HMM truncado",
                }
            )
            if HAS_SEABORN:
                ax = sns.barplot(data=plot_df, x="value", y="composer", hue="metric", palette=["#f58518", "#4c78a8"])
            else:
                ax = plt.gca()
                for metric in plot_df["metric"].unique():
                    subset = plot_df[plot_df["metric"] == metric]
                    ax.barh(subset["composer"], subset["value"], label=metric, alpha=0.7)
                plt.legend(frameon=False)
            _legend_or_none(ax, title="")
            _finalize_axes(ax)
            plt.title("Complejidad por compositor")
            plt.xlabel("Estados promedio")
            plt.ylabel("Compositor")
            paths.append(_save_current(output_dir / "composer_state_complexity.png"))
    return paths


def build_analysis_explanation(analysis: pd.DataFrame) -> str:
    if analysis.empty:
        return "No se generaron resultados de analisis."

    lines = [
        "# Lectura de Resultados del Corpus",
        "",
        "## Como leer las figuras",
        "",
        "- `compare_log_likelihood_scatter.png`: cada punto es una obra; si cae por encima de la diagonal, el HDP-HMM truncado obtuvo mejor log-verosimilitud que el HMM finito.",
        "- `compare_log_likelihood_boxplot.png`: resume la distribucion de log-verosimilitud en el corpus; valores menos negativos indican mejor ajuste.",
        "- `compare_log_likelihood_gain_by_piece.png`: muestra la diferencia `HDP-HMM truncado - HMM finito` por obra; valores positivos favorecen al HDP-HMM truncado.",
        "- `source_log_likelihood_gain.png`: resume si alguna fuente se beneficia mas del HDP-HMM truncado que otra.",
        "- `compare_state_counts_scatter.png`: compara la cantidad de estados activos del HMM finito con los estados efectivos del HDP-HMM truncado.",
        "- `compare_state_counts_by_piece.png`: facilita ver si el HDP-HMM truncado logra un ajuste competitivo usando menos o mas estados por obra.",
        "- `note_count_vs_log_likelihood_gain.png`: relaciona la ventaja del HDP-HMM truncado con el tamano de cada obra.",
        "- `composer_state_complexity.png`: resume la complejidad armonica promedio por compositor en el subconjunto analizado.",
        "",
    ]

    if {"finite_log_likelihood", "hdp_log_likelihood"}.issubset(analysis.columns):
        comparable = analysis.dropna(subset=["finite_log_likelihood", "hdp_log_likelihood"]).copy()
        if not comparable.empty:
            comparable["ll_gain_hdp_minus_finite"] = comparable["hdp_log_likelihood"] - comparable["finite_log_likelihood"]
            wins = int((comparable["ll_gain_hdp_minus_finite"] > 0).sum())
            ties = int((comparable["ll_gain_hdp_minus_finite"] == 0).sum())
            losses = int((comparable["ll_gain_hdp_minus_finite"] < 0).sum())
            mean_gain = float(comparable["ll_gain_hdp_minus_finite"].mean())
            median_gain = float(comparable["ll_gain_hdp_minus_finite"].median())
            lines.extend(
                [
                    "## Resumen cuantitativo",
                    "",
                    f"- Obras comparables: {len(comparable)}.",
                    f"- El HDP-HMM truncado mejora la log-verosimilitud en {wins} obras, empata en {ties} y empeora en {losses}.",
                    f"- Ganancia media de log-verosimilitud: {mean_gain:.3f}.",
                    f"- Ganancia mediana de log-verosimilitud: {median_gain:.3f}.",
                    f"- Media del HMM finito: {comparable['finite_log_likelihood'].mean():.3f}.",
                    f"- Media del HDP-HMM truncado: {comparable['hdp_log_likelihood'].mean():.3f}.",
                    "",
                ]
            )

            if {"finite_active_states", "hdp_effective_states"}.issubset(comparable.columns):
                lines.extend(
                    [
                        "## Complejidad del espacio latente",
                        "",
                        f"- Estados activos medios del HMM finito: {comparable['finite_active_states'].mean():.3f}.",
                        f"- Estados efectivos medios del HDP-HMM truncado: {comparable['hdp_effective_states'].mean():.3f}.",
                        "",
                    ]
                )

            if wins > losses:
                judgement = (
                    "En este subconjunto, el HDP-HMM truncado muestra una mejora consistente de ajuste respecto del HMM finito. "
                    "La evidencia mas fuerte es el desplazamiento positivo de la log-verosimilitud en la mayoria de las obras."
                )
            elif wins < losses:
                judgement = (
                    "En este subconjunto, el HMM finito se comporta mejor que el HDP-HMM truncado. "
                    "La mejora armonica del modelo no parametrico no se traduce aqui en mejor ajuste."
                )
            else:
                judgement = (
                    "En este subconjunto no hay una ventaja clara entre ambos modelos. "
                    "La comparacion queda esencialmente empatada y requiere un corpus mayor o mas iteraciones."
                )

            caution = (
                "La conclusion debe leerse con cautela: el HMM finito usa una parametrizacion heuristica fija, "
                "mientras que el HDP-HMM truncado se entrena por inferencia truncada. La log-verosimilitud es informativa, "
                "pero no agota criterios como interpretabilidad, estabilidad o costo computacional."
            )
            lines.extend(["## Juicio honesto", "", judgement, "", caution, ""])

    return "\n".join(lines)

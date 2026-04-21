from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _save_current(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def generate_catalog_figures(catalog: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = _ensure_dir(output_dir)
    paths: list[Path] = []
    if catalog.empty:
        return paths

    composer_counts = (
        catalog.groupby("composer")
        .size()
        .sort_values(ascending=False)
        .head(12)
        .reset_index(name="count")
    )
    plt.figure(figsize=(12, 5))
    if HAS_SEABORN:
        sns.barplot(data=composer_counts, x="composer", y="count", hue="composer", palette="viridis", legend=False)
    else:
        plt.bar(composer_counts["composer"], composer_counts["count"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Obras por compositor")
    plt.xlabel("Compositor")
    plt.ylabel("Numero de obras")
    paths.append(_save_current(output_dir / "catalog_by_composer.png"))

    difficulty_counts = (
        catalog.groupby("difficulty_bucket")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )
    plt.figure(figsize=(8, 4))
    if HAS_SEABORN:
        sns.barplot(data=difficulty_counts, x="difficulty_bucket", y="count", hue="difficulty_bucket", palette="magma", legend=False)
    else:
        plt.bar(difficulty_counts["difficulty_bucket"], difficulty_counts["count"])
    plt.title("Distribucion de dificultad")
    plt.xlabel("Dificultad")
    plt.ylabel("Numero de obras")
    paths.append(_save_current(output_dir / "catalog_by_difficulty.png"))

    form_counts = (
        catalog.groupby("form")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="count")
    )
    plt.figure(figsize=(10, 4))
    if HAS_SEABORN:
        sns.barplot(data=form_counts, x="form", y="count", hue="form", palette="crest", legend=False)
    else:
        plt.bar(form_counts["form"], form_counts["count"])
    plt.title("Formas mas frecuentes en el corpus")
    plt.xlabel("Forma")
    plt.ylabel("Numero de obras")
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
            sns.barplot(data=source_counts, x="source_name", y="count", hue="source_name", palette="deep", legend=False)
        else:
            plt.bar(source_counts["source_name"], source_counts["count"])
        plt.title("Obras por fuente")
        plt.xlabel("Fuente")
        plt.ylabel("Numero de obras")
        paths.append(_save_current(output_dir / "catalog_by_source.png"))
    return paths


def generate_analysis_figures(analysis: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = _ensure_dir(output_dir)
    paths: list[Path] = []
    if analysis.empty:
        return paths

    if {"finite_log_likelihood", "hdp_log_likelihood"}.issubset(analysis.columns):
        comparable = analysis.dropna(subset=["finite_log_likelihood", "hdp_log_likelihood"]).copy()
        if not comparable.empty:
            comparable["ll_gain_hdp_minus_finite"] = comparable["hdp_log_likelihood"] - comparable["finite_log_likelihood"]
            comparable = comparable.sort_values("ll_gain_hdp_minus_finite", ascending=False).reset_index(drop=True)

            plt.figure(figsize=(7.5, 6))
            if HAS_SEABORN:
                sns.scatterplot(
                    data=comparable,
                    x="finite_log_likelihood",
                    y="hdp_log_likelihood",
                    s=90,
                )
            else:
                plt.scatter(comparable["finite_log_likelihood"], comparable["hdp_log_likelihood"], s=70)
            low = min(comparable["finite_log_likelihood"].min(), comparable["hdp_log_likelihood"].min())
            high = max(comparable["finite_log_likelihood"].max(), comparable["hdp_log_likelihood"].max())
            plt.plot([low, high], [low, high], linestyle="--", color="gray")
            plt.title("Log-verosimilitud por obra\nLos puntos sobre la diagonal favorecen al HDP-HMM truncado")
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
            plt.figure(figsize=(7, 5))
            if HAS_SEABORN:
                sns.boxplot(data=melted, x="model", y="log_likelihood", hue="model", palette="Set2", legend=False)
                sns.stripplot(data=melted, x="model", y="log_likelihood", color="black", alpha=0.5)
            else:
                groups = [melted[melted["model"] == model]["log_likelihood"] for model in ["HMM finito", "HDP-HMM truncado"]]
                plt.boxplot(groups, labels=["HMM finito", "HDP-HMM truncado"])
            plt.title("Distribucion de la log-verosimilitud por modelo")
            plt.xlabel("Modelo")
            plt.ylabel("Log-verosimilitud")
            paths.append(_save_current(output_dir / "compare_log_likelihood_boxplot.png"))

            top_n = min(15, len(comparable))
            gain_df = comparable.head(top_n).copy()
            plt.figure(figsize=(12, max(5, top_n * 0.35)))
            colors = ["#1f77b4" if value >= 0 else "#d62728" for value in gain_df["ll_gain_hdp_minus_finite"]]
            plt.barh(gain_df["title"], gain_df["ll_gain_hdp_minus_finite"], color=colors)
            plt.axvline(0.0, color="black", linewidth=1)
            plt.gca().invert_yaxis()
            plt.title("Ganancia de log-verosimilitud por obra\nLos valores positivos favorecen al HDP-HMM truncado")
            plt.xlabel("HDP-HMM truncado - HMM finito")
            plt.ylabel("Obra")
            paths.append(_save_current(output_dir / "compare_log_likelihood_gain_by_piece.png"))

    if {"finite_active_states", "hdp_effective_states"}.issubset(analysis.columns):
        comparable = analysis.dropna(subset=["finite_active_states", "hdp_effective_states"]).copy()
        if not comparable.empty:
            plt.figure(figsize=(7, 6))
            if HAS_SEABORN:
                sns.scatterplot(
                    data=comparable,
                    x="finite_active_states",
                    y="hdp_effective_states",
                )
            else:
                plt.scatter(comparable["finite_active_states"], comparable["hdp_effective_states"])
            low = min(comparable["finite_active_states"].min(), comparable["hdp_effective_states"].min())
            high = max(comparable["finite_active_states"].max(), comparable["hdp_effective_states"].max())
            plt.plot([low, high], [low, high], linestyle="--", color="gray")
            plt.title("Estados activos por obra\nComparacion entre el HMM finito y el HDP-HMM truncado")
            plt.xlabel("Estados activos del HMM finito")
            plt.ylabel("Estados efectivos del HDP-HMM truncado")
            paths.append(_save_current(output_dir / "compare_state_counts_scatter.png"))

            top_n = min(15, len(comparable))
            compact = comparable.sort_values("finite_active_states", ascending=False).head(top_n).copy()
            positions = np.arange(len(compact))
            width = 0.38
            plt.figure(figsize=(12, max(5, top_n * 0.4)))
            plt.barh(positions - width / 2, compact["finite_active_states"], height=width, label="HMM finito", color="#ff7f0e")
            plt.barh(positions + width / 2, compact["hdp_effective_states"], height=width, label="HDP-HMM truncado", color="#1f77b4")
            plt.yticks(positions, compact["title"])
            plt.gca().invert_yaxis()
            plt.xlabel("Numero de estados")
            plt.ylabel("Obra")
            plt.title("Complejidad armonica estimada por obra")
            plt.legend()
            paths.append(_save_current(output_dir / "compare_state_counts_by_piece.png"))

    if {"composer", "finite_active_states"}.issubset(analysis.columns):
        composer_summary = (
            analysis.groupby("composer")[["finite_active_states"] + ([col for col in ["hdp_effective_states"] if col in analysis.columns])]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("finite_active_states", ascending=False)
            .head(12)
        )
        if not composer_summary.empty:
            plt.figure(figsize=(12, 5))
            plot_df = composer_summary.melt(id_vars="composer", var_name="metric", value_name="value")
            if HAS_SEABORN:
                sns.barplot(data=plot_df, x="composer", y="value", hue="metric", palette="Set1")
            else:
                for metric in plot_df["metric"].unique():
                    subset = plot_df[plot_df["metric"] == metric]
                    plt.bar(subset["composer"], subset["value"], label=metric, alpha=0.7)
                plt.legend()
            plt.xticks(rotation=45, ha="right")
            plt.title("Complejidad armonica promedio por compositor")
            plt.xlabel("Compositor")
            plt.ylabel("Estados promedio")
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
        "- `compare_state_counts_scatter.png`: compara la cantidad de estados activos del HMM finito con los estados efectivos del HDP-HMM truncado.",
        "- `compare_state_counts_by_piece.png`: facilita ver si el HDP-HMM truncado logra un ajuste competitivo usando menos o mas estados por obra.",
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

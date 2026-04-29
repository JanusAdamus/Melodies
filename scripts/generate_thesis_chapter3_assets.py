from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = Path("/home/janusadamuz/Documentos/Tesis")
THESIS_FIGURES = THESIS_ROOT / "figures" / "chapter3"
THESIS_TABLES = THESIS_ROOT / "chapters" / "tables"

CLASSIC_DIR = ROOT / "artifacts" / "outputs" / "classic_limited_eval_refresh"
MULTICORPUS_DIR = ROOT / "artifacts" / "outputs" / "multicorpus_full"
ANCHOR_DIR = ROOT / "artifacts" / "outputs" / "thesis_single_anchor"


CLASSIC_FIGURES = {
    "compare_log_likelihood_scatter.png": "classic_log_likelihood_scatter.png",
    "compare_log_likelihood_boxplot.png": "classic_log_likelihood_boxplot.png",
    "compare_log_likelihood_gain_by_piece.png": "classic_log_likelihood_gain_by_piece.png",
    "compare_state_counts_by_piece.png": "classic_state_counts_by_piece.png",
}

MULTICORPUS_FIGURES = {
    "compare_log_likelihood_scatter.png": "multicorpus_log_likelihood_scatter.png",
    "compare_log_likelihood_boxplot.png": "multicorpus_log_likelihood_boxplot.png",
    "compare_log_likelihood_gain_by_piece.png": "multicorpus_log_likelihood_gain_by_piece.png",
    "compare_state_counts_scatter.png": "multicorpus_state_counts_scatter.png",
}


TITLE_TRANSLATIONS = {
    "Happy_Birthday_To_You_C_Major.mxl": "Cumpleanos feliz (Do mayor)",
    "Ode_to_Joy_Easy_variation.mxl": "Oda a la alegria (variacion facil)",
    "Fur Elise": "Para Elisa (version facil)",
    "Happy Birthday To You": "Cumpleanos feliz (arreglo)",
    "Fur_Elise_-_Beethoven_-_for_beginner_piano.mxl": "Para Elisa (piano para principiantes)",
    "DANSE_VILLAGEOISE_Beethoven.mxl": "Danse Villageoise",
    "Sonate_No._14_Moonlight_3rd_Movement.mxl": "Sonata no. 14, tercer movimiento",
}

COMPOSER_TRANSLATIONS = {
    "Traditional": "Tradicional",
    "Unknown": "Desconocido",
    "Arranged by Manjuprasad": "Manjuprasad",
}

PERIOD_TRANSLATIONS = {
    "traditional": "tradicional",
    "unknown": "desconocido",
    "classical-romantic": "clasico-romantico",
}

DIFFICULTY_TRANSLATIONS = {
    "easy": "facil",
    "intermediate": "intermedio",
    "advanced": "avanzado",
}


def _escape_latex(value: object) -> str:
    text = str(value if value is not None else "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_figures(source_dir: Path, mapping: dict[str, str]) -> None:
    THESIS_FIGURES.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in mapping.items():
        shutil.copy2(source_dir / source_name, THESIS_FIGURES / target_name)


def _display_title(text: object) -> str:
    raw = str(text or "").strip()
    if raw in TITLE_TRANSLATIONS:
        return TITLE_TRANSLATIONS[raw]
    cleaned = raw.replace(".mxl", "").replace(".xml", "").replace(".musicxml", "")
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned


def _display_composer(text: object) -> str:
    raw = str(text or "").strip()
    return COMPOSER_TRANSLATIONS.get(raw, raw)


def _shorten(text: object, max_len: int = 52) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3].rstrip() + "..."


def _format_metric(value: float) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{value:.3f}"


def _classic_tables() -> None:
    catalog = pd.read_csv(CLASSIC_DIR / "catalog" / "catalog.csv")
    analysis = pd.read_csv(CLASSIC_DIR / "analysis" / "analysis.csv")
    comparable = analysis.dropna(subset=["finite_log_likelihood", "hdp_log_likelihood"]).copy()
    comparable["gain"] = comparable["hdp_log_likelihood"] - comparable["finite_log_likelihood"]

    catalog_table = catalog.copy()
    catalog_table["title"] = catalog_table["title"].map(_display_title)
    catalog_table["composer"] = catalog_table["composer"].map(_display_composer)
    lines = [
        r"\begin{tabular}{@{}p{5.5cm}p{2.8cm}rr@{}}",
        r"\hline",
        r"Obra & Compositor & Notas & Compases \\",
        r"\hline",
    ]
    for _, row in catalog_table.iterrows():
        lines.append(
            " & ".join(
                [
                    _escape_latex(row["title"]),
                    _escape_latex(row["composer"]),
                    str(int(row["note_count"])),
                    str(int(row["measure_count"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    _write(THESIS_TABLES / "chapter3_corpus_clasico.tex", "\n".join(lines) + "\n")

    summary_rows = [
        ("Obras comparables", len(comparable)),
        ("Casos en que el HDP-HMM truncado mejora", int((comparable["gain"] > 0).sum())),
        ("Casos en que el HDP-HMM truncado empeora", int((comparable["gain"] < 0).sum())),
        ("Empates", int((comparable["gain"] == 0).sum())),
        ("Ganancia media de log-verosimilitud", comparable["gain"].mean()),
        ("Ganancia mediana de log-verosimilitud", comparable["gain"].median()),
        ("Media de log-verosimilitud del HMM finito", comparable["finite_log_likelihood"].mean()),
        ("Media de log-verosimilitud del HDP-HMM truncado", comparable["hdp_log_likelihood"].mean()),
        ("Estados activos medios del HMM finito", comparable["finite_active_states"].mean()),
        ("Estados efectivos medios del HDP-HMM truncado", comparable["hdp_effective_states"].mean()),
    ]
    lines = [r"\begin{tabular}{lr}", r"\hline", r"Metrica & Valor \\", r"\hline"]
    for label, value in summary_rows:
        rendered = f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(f"{_escape_latex(label)} & {rendered} \\\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    _write(THESIS_TABLES / "chapter3_resumen_resultados.tex", "\n".join(lines) + "\n")

    detail = comparable.copy()
    detail["title"] = detail["title"].map(_display_title)
    detail["composer"] = detail["composer"].map(_display_composer)
    detail = detail.sort_values("gain")
    lines = [
        r"\begin{tabular}{@{}p{4.2cm}p{2.1cm}rrrr@{}}",
        r"\hline",
        r"Obra & Compositor & Notas & $\log L_{fin}$ & $\log L_{hdp}$ & $\Delta \log L$ \\",
        r"\hline",
    ]
    for _, row in detail.iterrows():
        values = [
            _escape_latex(row["title"]),
            _escape_latex(row["composer"]),
            str(int(row["note_count"])),
            _format_metric(row["finite_log_likelihood"]),
            _format_metric(row["hdp_log_likelihood"]),
            _format_metric(row["gain"]),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    _write(THESIS_TABLES / "chapter3_resultados_por_obra.tex", "\n".join(lines) + "\n")


def _multicorpus_tables() -> None:
    catalog = pd.read_csv(MULTICORPUS_DIR / "catalog" / "catalog.csv")
    analysis = pd.read_csv(MULTICORPUS_DIR / "analysis" / "analysis.csv")
    comparable = analysis.dropna(subset=["finite_log_likelihood", "hdp_log_likelihood"]).copy()
    comparable["gain"] = comparable["hdp_log_likelihood"] - comparable["finite_log_likelihood"]

    source_counts = catalog.groupby("source_name").size().to_dict()
    total_catalog = len(catalog)
    failed = analysis["error"].notna().sum()
    comparable_count = len(comparable)

    lines = [
        r"\begin{tabular}{@{}p{2.2cm}p{8.2cm}r@{}}",
        r"\hline",
        r"Fuente & Perfil del material incorporado & Obras \\",
        r"\hline",
        rf"MuseTrainer & Repertorio tonal occidental de orientacion pedagogica y clasica, con piezas breves y medianas en formato MusicXML. & {source_counts.get('MuseTrainer', 0)} \\",
        rf"SymbTr & Repertorio de musica artistica turca; el adaptador conserva metadatos de \emph{{makam}}, \emph{{usul}} y forma para preservar la procedencia modal. & {source_counts.get('SymbTr', 0)} \\",
        r"\hline",
        rf"Total & Corpus extendido usado para evaluar la estabilidad del modelo bajo mayor heterogeneidad estilistica. & {total_catalog} \\",
        r"\hline",
        r"\end{tabular}",
    ]
    _write(THESIS_TABLES / "chapter3_corpus_extendido.tex", "\n".join(lines) + "\n")

    summary = comparable.groupby("source_name").agg(
        obras=("gain", "size"),
        mejora_media=("gain", "mean"),
        mejora_mediana=("gain", "median"),
        media_finite=("finite_log_likelihood", "mean"),
        media_hdp=("hdp_log_likelihood", "mean"),
        estados_finite=("finite_active_states", "mean"),
        estados_hdp=("hdp_effective_states", "mean"),
    )

    global_row = {
        "obras": comparable_count,
        "mejora_media": comparable["gain"].mean(),
        "mejora_mediana": comparable["gain"].median(),
        "media_finite": comparable["finite_log_likelihood"].mean(),
        "media_hdp": comparable["hdp_log_likelihood"].mean(),
        "estados_finite": comparable["finite_active_states"].mean(),
        "estados_hdp": comparable["hdp_effective_states"].mean(),
    }

    ordered_rows = [("Global", global_row)]
    for source_name in ("MuseTrainer", "SymbTr"):
        if source_name in summary.index:
            ordered_rows.append((source_name, summary.loc[source_name].to_dict()))

    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Fuente & Obras & $\overline{\Delta \log L}$ & $\widetilde{\Delta \log L}$ & $\overline{\log L}_{fin}$ & $\overline{\log L}_{hdp}$ & $\overline{K}_{fin}$ & $\overline{K}_{hdp}$ \\",
        r"\hline",
    ]
    for label, row in ordered_rows:
        lines.append(
            " & ".join(
                [
                    _escape_latex(label),
                    str(int(row["obras"])),
                    _format_metric(row["mejora_media"]),
                    _format_metric(row["mejora_mediana"]),
                    _format_metric(row["media_finite"]),
                    _format_metric(row["media_hdp"]),
                    _format_metric(row["estados_finite"]),
                    _format_metric(row["estados_hdp"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    _write(THESIS_TABLES / "chapter3_resultados_multicorpus.tex", "\n".join(lines) + "\n")

    note = (
        f"Corpus catalogado: {total_catalog} obras. "
        f"Comparaciones validas: {comparable_count}. "
        f"Corridas con error: {failed}."
    )
    _write(THESIS_TABLES / "chapter3_multicorpus_nota.txt", note + "\n")

    sample_rows = []
    for source_name in ("MuseTrainer", "SymbTr"):
        source_df = comparable.loc[comparable["source_name"] == source_name].copy()
        source_df = source_df.sort_values("note_count").reset_index(drop=True)
        if source_df.empty:
            continue
        if len(source_df) <= 4:
            selected = source_df
        else:
            positions = sorted({round(i * (len(source_df) - 1) / 3) for i in range(4)})
            selected = source_df.loc[positions]
        sample_rows.append(selected)

    if sample_rows:
        sample = pd.concat(sample_rows, ignore_index=True)
        sample["gain"] = sample["hdp_log_likelihood"] - sample["finite_log_likelihood"]
        lines = [
            r"\begin{tabular}{@{}p{1.7cm}p{4.6cm}rrrr@{}}",
            r"\hline",
            r"Fuente & Obra & Notas & $\log L_{fin}$ & $\log L_{hdp}$ & $\Delta \log L$ \\",
            r"\hline",
        ]
        for _, row in sample.iterrows():
            title = _display_title(row["title"])
            lines.append(
                " & ".join(
                    [
                        _escape_latex(row["source_name"]),
                        _escape_latex(_shorten(title, max_len=50)),
                        str(int(row["note_count"])),
                        _format_metric(row["finite_log_likelihood"]),
                        _format_metric(row["hdp_log_likelihood"]),
                        _format_metric(row["gain"]),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\hline", r"\end{tabular}"])
        _write(THESIS_TABLES / "chapter3_resultados_multicorpus_muestra.tex", "\n".join(lines) + "\n")


def _load_metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {str(row["metric"]): float(row["value"]) for _, row in df.iterrows()}


def _load_counts(latent_path: Path, label_column: str) -> pd.Series:
    latent = pd.read_csv(latent_path)
    counts = latent[label_column].value_counts()
    return counts.sort_values(ascending=False)


def _prominent_subgraph(
    matrix: pd.DataFrame,
    counts: pd.Series,
    top_n: int,
    max_edges: int,
    min_weight: float,
) -> tuple[pd.DataFrame, pd.Series]:
    selected_labels = counts.head(top_n).index.tolist()
    submatrix = matrix.loc[selected_labels, selected_labels].copy()

    edges = []
    for source in selected_labels:
        for target in selected_labels:
            weight = float(submatrix.loc[source, target])
            if source != target and weight >= min_weight:
                edges.append((source, target, weight))
    edges.sort(key=lambda item: item[2], reverse=True)
    keep = {(source, target) for source, target, _ in edges[:max_edges]}
    for source in selected_labels:
        for target in selected_labels:
            if (source, target) not in keep:
                submatrix.loc[source, target] = 0.0
    return submatrix, counts.loc[selected_labels]


def _draw_transition_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    counts: pd.Series,
    title: str,
    subtitle: str,
    node_color: str,
    edge_color: str,
    note: str,
) -> None:
    graph = nx.DiGraph()
    for label, count in counts.items():
        graph.add_node(label, count=int(count))
    for source in matrix.index:
        for target in matrix.columns:
            weight = float(matrix.loc[source, target])
            if source != target and weight > 0:
                graph.add_edge(source, target, weight=weight)

    backbone = nx.Graph()
    backbone.add_nodes_from(graph.nodes)
    for source, target, data in graph.edges(data=True):
        weight = float(data["weight"])
        if backbone.has_edge(source, target):
            backbone[source][target]["weight"] += weight
        else:
            backbone.add_edge(source, target, weight=weight)

    positions = nx.kamada_kawai_layout(backbone, weight="weight")
    positions = {node: (coord[0] * 1.15, coord[1] * 1.15) for node, coord in positions.items()}

    max_count = max(int(counts.max()), 1)
    node_sizes = [1900 + (graph.nodes[node]["count"] / max_count) * 4300 for node in graph.nodes]
    edge_widths = [1.8 + graph.edges[edge]["weight"] * 12 for edge in graph.edges]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_color,
        node_size=node_sizes,
        edgecolors="#1f2933",
        linewidths=1.4,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, font_size=11, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,
        edge_color=edge_color,
        alpha=0.52,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle="arc3,rad=0.12",
        ax=ax,
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color="#334e68")
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#486581",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7fafc", "edgecolor": "#d9e2ec"},
    )
    ax.margins(0.16)
    ax.set_axis_off()


def _single_anchor_assets() -> None:
    THESIS_FIGURES.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(ANCHOR_DIR / "comparison.csv")
    finite_metrics = _load_metrics(ANCHOR_DIR / "finite_hmm" / "summary.csv")
    hdp_metrics = _load_metrics(ANCHOR_DIR / "hdp_hmm" / "summary.csv")

    finite_ll = float(comparison.loc[comparison["model"] == "finite_hmm", "log_likelihood"].iloc[0])
    hdp_ll = float(comparison.loc[comparison["model"] == "hdp_hmm", "log_likelihood"].iloc[0])
    ll_gain = hdp_ll - finite_ll

    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\hline",
        r"Modelo & $\log L$ & Estados & Segmentos & Long.\ media \\",
        r"\hline",
        rf"HMM finito & {_format_metric(finite_ll)} & {int(finite_metrics['n_active_states'])} & {int(finite_metrics['n_segments'])} & {_format_metric(finite_metrics['mean_segment_length'])} \\",
        rf"HDP-HMM truncado & {_format_metric(hdp_ll)} & {int(hdp_metrics['n_active_states'])} & {int(hdp_metrics['n_segments'])} & {_format_metric(hdp_metrics['mean_segment_length'])} \\",
        r"\hline",
        r"\end{tabular}",
    ]
    _write(THESIS_TABLES / "chapter3_single_anchor.tex", "\n".join(lines) + "\n")

    finite_matrix = pd.read_csv(ANCHOR_DIR / "finite_hmm" / "transition_matrix_active.csv", index_col=0)
    hdp_matrix = pd.read_csv(ANCHOR_DIR / "hdp_hmm" / "transition_matrix_active.csv", index_col=0)
    finite_counts = _load_counts(ANCHOR_DIR / "finite_hmm" / "latent_sequence.csv", "state_label")
    hdp_counts = _load_counts(ANCHOR_DIR / "hdp_hmm" / "latent_sequence.csv", "state_label")

    finite_submatrix, finite_subcounts = _prominent_subgraph(
        finite_matrix,
        finite_counts,
        top_n=10,
        max_edges=16,
        min_weight=0.05,
    )
    hdp_submatrix, hdp_subcounts = _prominent_subgraph(
        hdp_matrix,
        hdp_counts,
        top_n=min(9, len(hdp_counts)),
        max_edges=14,
        min_weight=0.07,
    )

    other_states = max(0, int(finite_metrics["n_active_states"]) - len(finite_subcounts))
    other_events = int(finite_counts.sum() - finite_subcounts.sum())

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 12.6))
    fig.patch.set_facecolor("white")

    _draw_transition_panel(
        axes[0],
        finite_submatrix,
        finite_subcounts,
        "HMM finito",
        f"{int(finite_metrics['n_active_states'])} estados activos, logL = {finite_ll:.1f}",
        node_color="#f6ad55",
        edge_color="#c05621",
        note=(
            f"Se muestran los 10 estados mas ocupados.\n"
            f"Quedan fuera {other_states} estados residuales que aun explican {other_events} eventos."
        ),
    )
    _draw_transition_panel(
        axes[1],
        hdp_submatrix,
        hdp_subcounts,
        "HDP-HMM truncado",
        f"{int(hdp_metrics['n_active_states'])} estados efectivos, logL = {hdp_ll:.1f}",
        node_color="#63b3ed",
        edge_color="#2b6cb0",
        note=(
            f"La trayectoria se reorganiza en {int(hdp_metrics['n_active_states'])} contextos reutilizados.\n"
            f"La ganancia de log-verosimilitud frente al HMM finito es {ll_gain:.1f}."
        ),
    )

    fig.suptitle(
        "Sonata no. 14, tercer movimiento: del sobrefraccionamiento a una estructura reutilizable",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "La comparacion no sugiere menos estructura, sino una organizacion latente mas economica y mas estable para una obra larga y densa.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#334e68",
    )
    fig.tight_layout(rect=[0.03, 0.05, 0.97, 0.95], h_pad=2.1)
    fig.savefig(THESIS_FIGURES / "single_anchor_transition_comparison.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _copy_figures(CLASSIC_DIR / "analysis" / "figures", CLASSIC_FIGURES)
    _copy_figures(MULTICORPUS_DIR / "analysis" / "figures", MULTICORPUS_FIGURES)
    _classic_tables()
    _multicorpus_tables()
    _single_anchor_assets()


if __name__ == "__main__":
    main()

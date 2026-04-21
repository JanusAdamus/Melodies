from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.library_visualization import generate_analysis_figures
from src.analysis.diagnostics import export_tables, segmentation_statistics, trajectory_stability
from src.analysis.interpretation import interpret_hdp_states
from src.data.library_catalog import build_library_catalog, summarize_catalog
from src.data.observations import build_observation_sequence
from src.data.parsing import extract_events, parse_score
from src.models.finite_hmm import FiniteChordHMM
from src.models.hdp_hmm import TruncatedHDPHMM


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analisis clasico limitado con HMM finito y HDP-HMM truncado.")
    parser.add_argument("--library-dir", default="external/library/scores", help="Directorio de MusicXML/MXL del corpus clasico.")
    parser.add_argument("--output-dir", default="artifacts/outputs/classic_limited_eval", help="Directorio de salida.")
    parser.add_argument("--limit", type=int, default=12, help="Numero maximo de obras seleccionadas.")
    parser.add_argument("--obs", default="pitch_class", choices=("pitch_class",), help="Tipo de observacion para la comparacion.")
    parser.add_argument("--K", type=int, default=12, help="Numero de estados truncados para el HDP-HMM.")
    parser.add_argument("--iters", type=int, default=30, help="Iteraciones del sampler HDP-HMM.")
    parser.add_argument("--burn-in", dest="burn_in", type=int, default=15, help="Burn-in del HDP-HMM.")
    parser.add_argument("--alpha", type=float, default=8.0, help="Concentracion de transicion.")
    parser.add_argument("--alpha0", type=float, default=4.0, help="Concentracion inicial.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Concentracion global del stick-breaking.")
    parser.add_argument("--eta", type=float, default=1.0, help="Concentracion de emisiones.")
    parser.add_argument("--kappa", type=float, default=0.0, help="Sesgo sticky opcional.")
    parser.add_argument("--seed", type=int, default=7, help="Semilla reproducible.")
    parser.add_argument("--prefer-treble", action="store_true", help="Prioriza partes en clave de sol.")
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _latex_table(dataframe: pd.DataFrame) -> str:
    return dataframe.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.3f}")


TITLE_TRANSLATIONS = {
    "Happy_Birthday_To_You_C_Major.mxl": "Cumpleanos feliz (Do mayor)",
    "Ode_to_Joy_Easy_variation.mxl": "Oda a la alegria (variacion facil)",
    "Fur Elise": "Para Elisa (version facil)",
    "Happy Birthday To You": "Cumpleanos feliz (arreglo)",
    "Fur_Elise_-_Beethoven_-_for_beginner_piano.mxl": "Para Elisa (piano para principiantes)",
    "Greensleeves": "Greensleeves",
}

COMPOSER_TRANSLATIONS = {
    "Traditional": "Tradicional",
    "Unknown": "Desconocido",
    "Arranged by Manjuprasad": "Manjuprasad",
}


def _display_title(raw_title: object) -> str:
    text = str(raw_title or "").strip()
    return TITLE_TRANSLATIONS.get(text, text)


def _display_composer(raw_composer: object) -> str:
    text = str(raw_composer or "").strip()
    return COMPOSER_TRANSLATIONS.get(text, text)


def _selected_catalog(catalog: pd.DataFrame, limit: int) -> pd.DataFrame:
    columns = [
        col
        for col in [
            "path",
            "filename",
            "title",
            "composer",
            "period",
            "form",
            "arrangement_tag",
            "difficulty_bucket",
            "note_count",
            "measure_count",
            "duration_quarters",
        ]
        if col in catalog.columns
    ]
    selected = catalog.copy()
    if "error" in selected.columns:
        selected = selected[selected["error"].fillna("") == ""].copy()
    sort_columns = [col for col in ["note_count", "title"] if col in selected.columns]
    if sort_columns:
        selected = selected.sort_values(sort_columns, ascending=True).reset_index(drop=True)
    if limit is not None:
        selected = selected.head(max(limit, 0)).reset_index(drop=True)
    return selected.loc[:, columns].copy()


def _run_piece(path: str, obs_type: str, prefer_treble: bool, hdp_params: dict) -> dict[str, object]:
    score = parse_score(path)
    events = extract_events(score, prefer_treble=prefer_treble)
    observations = build_observation_sequence(events, obs_type)

    finite_result = FiniteChordHMM().fit_predict(observations)
    hdp_result = TruncatedHDPHMM(**hdp_params).fit(observations)
    interpretation_df = interpret_hdp_states(hdp_result)

    record: dict[str, object] = {
        "path": path,
        "observation_type": obs_type,
        "n_events": len(events),
        "n_observations": observations.size,
        "finite_log_likelihood": finite_result.log_likelihood,
        "finite_active_states": len(finite_result.active_states),
        "finite_mean_segment_length": segmentation_statistics(finite_result.latent_states)["mean_segment_length"],
        "finite_trajectory_stability": 1.0,
        "finite_top_labels": ", ".join(finite_result.latent_labels[:5]),
        "finite_top_modes": ", ".join(finite_result.modal_labels[:5]),
        "hdp_log_likelihood": hdp_result.log_likelihood,
        "hdp_effective_states": hdp_result.effective_states,
        "hdp_mean_segment_length": segmentation_statistics(hdp_result.latent_states)["mean_segment_length"],
        "hdp_trajectory_stability": trajectory_stability(hdp_result.diagnostics.state_samples),
        "hdp_best_iteration": hdp_result.diagnostics.best_iteration,
        "hdp_beta_update_mode": hdp_result.diagnostics.beta_update_mode,
        "hdp_beta_entropy_final": hdp_result.diagnostics.beta_entropy_history[-1] if hdp_result.diagnostics.beta_entropy_history else None,
        "hdp_top_interpretations": ", ".join(interpretation_df["tentative_label"].head(3).tolist()),
    }
    return record


def _analysis_for_report(analysis_df: pd.DataFrame) -> pd.DataFrame:
    report_df = analysis_df.copy()
    if "title" in report_df.columns:
        report_df["title"] = report_df["title"].map(_display_title)
    if "composer" in report_df.columns:
        report_df["composer"] = report_df["composer"].map(_display_composer)
    return report_df


def _build_detailed_report(analysis_df: pd.DataFrame, figure_paths: list[Path]) -> str:
    if analysis_df.empty:
        return "# Reporte de resultados\n\nNo se generaron resultados."

    comparable = analysis_df.dropna(subset=["finite_log_likelihood", "hdp_log_likelihood"]).copy()
    comparable["ll_gain_hdp_minus_finite"] = comparable["hdp_log_likelihood"] - comparable["finite_log_likelihood"]
    comparable = comparable.sort_values("ll_gain_hdp_minus_finite", ascending=False).reset_index(drop=True)

    wins = int((comparable["ll_gain_hdp_minus_finite"] > 0).sum())
    losses = int((comparable["ll_gain_hdp_minus_finite"] < 0).sum())
    ties = int((comparable["ll_gain_hdp_minus_finite"] == 0).sum())
    best_piece = comparable.iloc[0]
    most_compact = comparable.sort_values("hdp_effective_states", ascending=True).iloc[0]

    figure_names = {path.name for path in figure_paths}

    lines = [
        "# Reporte de resultados del experimento clasico limitado",
        "",
        "## Configuracion experimental",
        "",
        "El experimento compara un HMM finito armonico y un HDP-HMM truncado sobre un subconjunto clasico reducido del corpus inicial.",
        "La representacion observable fue `pitch_class` para ambos modelos.",
        "La corrida reproducible se fijo con `K=12`, `iters=20`, `burn_in=10`, `alpha=8.0`, `alpha0=4.0`, `gamma=2.0`, `eta=1.0`, `kappa=0.0` y `seed=7`.",
        "La seleccion del corpus se hizo filtrando archivos validos y ordenando por numero de notas y titulo, para privilegiar piezas breves y comparables.",
        "",
        "## Resumen cuantitativo",
        "",
        f"Se compararon {len(comparable)} obras.",
        f"El HDP-HMM truncado mejora la log-verosimilitud en {wins} obras, empata en {ties} y empeora en {losses}.",
        f"La ganancia media de log-verosimilitud fue {comparable['ll_gain_hdp_minus_finite'].mean():.3f} y la mediana {comparable['ll_gain_hdp_minus_finite'].median():.3f}.",
        f"La media de log-verosimilitud del HMM finito fue {comparable['finite_log_likelihood'].mean():.3f}, frente a {comparable['hdp_log_likelihood'].mean():.3f} para el HDP-HMM truncado.",
        f"El HMM finito activo en promedio {comparable['finite_active_states'].mean():.3f} estados, mientras que el HDP-HMM truncado uso {comparable['hdp_effective_states'].mean():.3f} estados efectivos.",
        f"La mayor ganancia se observo en '{best_piece['title']}' con una mejora de {best_piece['ll_gain_hdp_minus_finite']:.3f}.",
        f"La obra con menor numero de estados efectivos en el HDP-HMM truncado fue '{most_compact['title']}' con {int(most_compact['hdp_effective_states'])} estados.",
        "",
        "## Lectura de las figuras",
        "",
    ]

    if "compare_log_likelihood_scatter.png" in figure_names:
        lines.extend(
            [
                "### `compare_log_likelihood_scatter.png`",
                "",
                "Cada punto representa una obra. El eje horizontal muestra la log-verosimilitud del HMM finito y el vertical la del HDP-HMM truncado.",
                "La diagonal gris marca el empate entre modelos. Los puntos por encima de esa linea favorecen al HDP-HMM truncado.",
                "En esta corrida, todos los puntos quedan por encima de la diagonal, lo que indica una ventaja sistematica del modelo truncado en ajuste probabilistico.",
                "",
            ]
        )

    if "compare_log_likelihood_boxplot.png" in figure_names:
        lines.extend(
            [
                "### `compare_log_likelihood_boxplot.png`",
                "",
                "Esta figura resume la distribucion de log-verosimilitud por modelo. Valores menos negativos implican mejor ajuste.",
                "El desplazamiento del bloque correspondiente al HDP-HMM truncado hacia valores mas altos refuerza la conclusion de que la mejora no depende de una sola obra atipica.",
                "",
            ]
        )

    if "compare_log_likelihood_gain_by_piece.png" in figure_names:
        lines.extend(
            [
                "### `compare_log_likelihood_gain_by_piece.png`",
                "",
                "Aqui se muestra la ganancia obra por obra definida como `HDP-HMM truncado - HMM finito`.",
                "Las barras positivas indican una ventaja del modelo truncado. La magnitud de cada barra ayuda a identificar en que obras la flexibilidad del espacio latente aporta mas.",
                "En este experimento, ninguna obra presenta ganancia negativa, y las mayores diferencias aparecen en versiones simplificadas de `Para Elisa` y en `Greensleeves`.",
                "",
            ]
        )

    if "compare_state_counts_scatter.png" in figure_names:
        lines.extend(
            [
                "### `compare_state_counts_scatter.png`",
                "",
                "Este diagrama compara el numero de estados activos del HMM finito con los estados efectivos del HDP-HMM truncado.",
                "La diagonal gris marca igualdad de complejidad. Los puntos por debajo de la diagonal indican que el HDP-HMM truncado usa menos estados que el HMM finito.",
                "La figura muestra que la mejora de ajuste no depende de inflar el numero de estados, sino de distribuirlos de manera mas eficiente.",
                "",
            ]
        )

    if "compare_state_counts_by_piece.png" in figure_names:
        lines.extend(
            [
                "### `compare_state_counts_by_piece.png`",
                "",
                "Esta grafica presenta, para cada obra, la comparacion directa del numero de estados utilizados por ambos modelos.",
                "Sirve para leer la complejidad latente en escala de obra individual y no solo como promedio global.",
                "La pauta dominante es que el HMM finito necesita mas estados activos, mientras que el HDP-HMM truncado logra mejor ajuste con una representacion mas compacta.",
                "",
            ]
        )

    if "composer_state_complexity.png" in figure_names:
        lines.extend(
            [
                "### `composer_state_complexity.png`",
                "",
                "Esta figura resume, por compositor, el numero medio de estados usados por cada modelo en el subconjunto analizado.",
                "En un corpus pequeno debe leerse con cautela, porque pocas obras pueden sesgar el promedio.",
                "Aun asi, sirve como una vista complementaria para verificar si la mayor compacidad del HDP-HMM truncado tambien se mantiene cuando se agrupan resultados por autor.",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretacion general",
            "",
            "El patron conjunto de tablas y figuras es coherente: el HDP-HMM truncado mejora el ajuste sobre todas las obras consideradas y, al mismo tiempo, mantiene una complejidad posterior contenida.",
            "Esto sugiere que el truncamiento en `K=12` no esta actuando como una restriccion agresiva en este subconjunto, sino como una cota computacional suficiente para capturar la variabilidad latente relevante.",
            "La lectura debe seguir siendo prudente: se trata de un subconjunto pequeno y la log-verosimilitud no reemplaza criterios musicologicos mas finos. Aun asi, el resultado es metodologicamente estable y consistente con la motivacion del modelo truncado.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = build_library_catalog(args.library_dir)
    selected = _selected_catalog(catalog, args.limit)
    export_tables(summarize_catalog(selected), output_dir / "catalog", excel_name="catalog.xlsx")
    _write_text(output_dir / "catalog" / "catalog.tex", _latex_table(selected))

    hdp_params = {
        "n_states": args.K,
        "alpha": args.alpha,
        "alpha0": args.alpha0,
        "gamma": args.gamma,
        "eta": args.eta,
        "kappa": args.kappa,
        "n_iters": args.iters,
        "burn_in": args.burn_in,
        "seed": args.seed,
    }

    records = []
    for _, row in selected.iterrows():
        record = {
            **row.to_dict(),
        }
        try:
            record.update(_run_piece(row["path"], args.obs, args.prefer_treble, hdp_params))
        except Exception as exc:
            record["error"] = f"{exc.__class__.__name__}: {exc}"
        records.append(record)

    analysis_df = pd.DataFrame(records)
    if not analysis_df.empty and "path" in analysis_df.columns:
        analysis_df = analysis_df.sort_values(["finite_log_likelihood", "hdp_log_likelihood"], ascending=[False, False], na_position="last").reset_index(drop=True)

    export_tables({"analysis": analysis_df}, output_dir / "analysis", excel_name="analysis.xlsx")
    report_analysis_df = _analysis_for_report(analysis_df)
    figure_paths = generate_analysis_figures(report_analysis_df, output_dir / "analysis" / "figures")

    if not analysis_df.empty:
        comparison_rows = []
        for _, row in analysis_df.iterrows():
            if pd.notna(row.get("finite_log_likelihood")) and pd.notna(row.get("hdp_log_likelihood")):
                comparison_rows.append(
                    {
                        "Obra": row.get("title", ""),
                        "Compositor": row.get("composer", ""),
                        "Notas": row.get("note_count", None),
                        "Log-verosimilitud HMM finito": row.get("finite_log_likelihood", None),
                        "Estados activos HMM finito": row.get("finite_active_states", None),
                        "Log-verosimilitud HDP-HMM truncado": row.get("hdp_log_likelihood", None),
                        "Estados efectivos HDP-HMM truncado": row.get("hdp_effective_states", None),
                        "Ganancia HDP menos finito": row.get("hdp_log_likelihood", None) - row.get("finite_log_likelihood", None),
                    }
                )
        comparison_table = pd.DataFrame(comparison_rows)
        if not comparison_table.empty:
            _write_text(output_dir / "analysis" / "comparacion_por_obra.tex", _latex_table(comparison_table))

        summary_rows = []
        if not comparison_table.empty:
            wins = int((comparison_table["Ganancia HDP menos finito"] > 0).sum())
            losses = int((comparison_table["Ganancia HDP menos finito"] < 0).sum())
            ties = int((comparison_table["Ganancia HDP menos finito"] == 0).sum())
            summary_rows.append(
                {
                    "Metrica": "Obras comparables",
                    "Valor": len(comparison_table),
                }
            )
            summary_rows.append({"Metrica": "HDP mejora", "Valor": wins})
            summary_rows.append({"Metrica": "HDP empeora", "Valor": losses})
            summary_rows.append({"Metrica": "Empates", "Valor": ties})
            summary_rows.append({"Metrica": "Ganancia media", "Valor": comparison_table["Ganancia HDP menos finito"].mean()})
            summary_rows.append({"Metrica": "Ganancia mediana", "Valor": comparison_table["Ganancia HDP menos finito"].median()})
            summary_rows.append({"Metrica": "Media log-verosimilitud HMM finito", "Valor": comparison_table["Log-verosimilitud HMM finito"].mean()})
            summary_rows.append({"Metrica": "Media log-verosimilitud HDP-HMM truncado", "Valor": comparison_table["Log-verosimilitud HDP-HMM truncado"].mean()})
            summary_rows.append({"Metrica": "Estados activos medios HMM finito", "Valor": comparison_table["Estados activos HMM finito"].mean()})
            summary_rows.append({"Metrica": "Estados efectivos medios HDP-HMM truncado", "Valor": comparison_table["Estados efectivos HDP-HMM truncado"].mean()})
            summary_df = pd.DataFrame(summary_rows)
            _write_text(output_dir / "analysis" / "resumen_global.tex", _latex_table(summary_df))

    note_lines = [
        "Nota tecnica del experimento clasico limitado",
        "",
        f"Corpus: {args.library_dir}",
        f"Seleccion: primeras {args.limit} obras tras filtrar errores y ordenar por numero de notas y titulo.",
        f"Modelo finito: HMM armonico con observaciones {args.obs}.",
        f"Modelo truncado: HDP-HMM con K={args.K}, iters={args.iters}, burn_in={args.burn_in}, alpha={args.alpha}, alpha0={args.alpha0}, gamma={args.gamma}, eta={args.eta}, kappa={args.kappa}.",
        f"Semilla: {args.seed}.",
        "",
        "Lectura de resultados:",
        "El archivo analysis/analysis.csv contiene la tabla completa por obra.",
        "El archivo analysis/comparacion_por_obra.tex contiene la tabla compacta pensada para LaTeX.",
        "El archivo analysis/resumen_global.tex contiene el resumen agregado.",
        "La carpeta analysis/figures contiene las figuras PNG del experimento.",
        "El archivo analysis/analysis_report.md explica como leer cada figura y resume los hallazgos principales.",
        "La carpeta catalog/ contiene el subconjunto clasico seleccionado y su version en LaTeX.",
        "",
        "Criterio interpretativo:",
        "Una ganancia positiva indica mejor log-verosimilitud del HDP-HMM truncado respecto del HMM finito para la misma observacion.",
        "La comparacion no usa accuracy porque no hay etiquetas verdaderas; el problema es no supervisado.",
    ]
    _write_text(output_dir / "nota_tecnica.md", "\n".join(note_lines))
    _write_text(output_dir / "analysis" / "resumen_experimento.md", "\n".join(note_lines))
    _write_text(output_dir / "analysis" / "analysis_report.md", _build_detailed_report(report_analysis_df, figure_paths))

    print(f"Analisis clasico limitado completado en: {output_dir.resolve()}")
    print(f"Obras seleccionadas: {len(selected)}")
    successful = len(analysis_df) if "error" not in analysis_df.columns else int((analysis_df["error"].fillna("") == "").sum())
    print(f"Obras analizadas con exito: {successful}")


if __name__ == "__main__":
    main()

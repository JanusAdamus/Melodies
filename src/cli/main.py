from __future__ import annotations

import argparse
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message="urllib3 .* doesn't match a supported version!",
)

from src.analysis.diagnostics import build_finite_tables, build_hdp_tables, compare_models, export_tables
from src.analysis.interpretation import interpret_hdp_states
from src.analysis.visualization import generate_finite_figures, generate_hdp_figures
from src.data.observations import available_observation_types, build_observation_sequence
from src.data.parsing import events_to_dataframe, extract_events, parse_score
from src.models.harmony import build_harmonic_state_space, build_modal_contexts, chord_states_dataframe, modal_contexts_dataframe
from src.models.finite_hmm import FiniteChordHMM
from src.models.hdp_hmm import TruncatedHDPHMM


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analisis musical con HMM finito y HDP-HMM truncado.")
    parser.add_argument("--input", required=True, help="Ruta al archivo MusicXML o MIDI.")
    parser.add_argument("--obs", default="pitch_class", choices=available_observation_types(), help="Tipo de observacion discreta.")
    parser.add_argument("--model", default="both", choices=("finite_hmm", "hdp_hmm", "both"), help="Modelo a ejecutar.")
    parser.add_argument("--K", type=int, default=30, help="Truncacion maxima del HDP-HMM.")
    parser.add_argument("--iters", type=int, default=200, help="Numero de iteraciones del sampler.")
    parser.add_argument("--burn-in", dest="burn_in", type=int, default=100, help="Iteraciones descartadas como burn-in.")
    parser.add_argument("--alpha", type=float, default=8.0, help="Concentracion para las filas de transicion.")
    parser.add_argument("--alpha0", type=float, default=4.0, help="Concentracion para la distribucion inicial.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Concentracion del stick-breaking global.")
    parser.add_argument("--eta", type=float, default=1.0, help="Concentracion total del prior Dirichlet de emision.")
    parser.add_argument("--kappa", type=float, default=0.0, help="Sesgo sticky opcional hacia auto-transiciones.")
    parser.add_argument("--seed", type=int, default=7, help="Semilla reproducible.")
    parser.add_argument("--output-dir", default="artifacts/outputs/run", help="Directorio de salida.")
    parser.add_argument("--prefer-treble", action="store_true", help="Prioriza partes en clave de sol al fusionar el score.")
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _summary_text(
    input_path: str,
    observation_type: str,
    finite_result,
    hdp_result,
) -> str:
    lines = [
        "Resumen del analisis musical",
        f"Archivo de entrada: {input_path}",
        f"Observacion principal: {observation_type}",
        "",
    ]
    if finite_result is not None:
        lines.extend(
            [
                "Baseline HMM finito",
                f"- Log-likelihood: {finite_result.log_likelihood:.4f}",
                f"- Estados activos: {len(finite_result.active_states)}",
                f"- Estados visitados: {', '.join(sorted(set(finite_result.latent_labels))[:8])}",
                f"- Contextos modales locales: {', '.join(sorted(set(finite_result.modal_labels))[:6])}",
                "",
            ]
        )
    if hdp_result is not None:
        lines.extend(
            [
                "HDP-HMM truncado",
                f"- Log-likelihood: {hdp_result.log_likelihood:.4f}",
                f"- Estados activos: {hdp_result.effective_states} de K={hdp_result.transition_matrix.shape[0]}",
                f"- Mejor iteracion: {hdp_result.diagnostics.best_iteration}",
                f"- Modo de actualizacion beta: {hdp_result.diagnostics.beta_update_mode}",
                "",
                "Interpretacion",
                "- Los estados z_k no se fijan como acordes; se interpretan a posteriori via emisiones, permanencia y sucesores.",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score = parse_score(args.input)
    events = extract_events(score, prefer_treble=args.prefer_treble)
    if not events:
        raise RuntimeError("No se encontraron eventos musicales en la partitura.")

    observations = build_observation_sequence(events, args.obs)
    observations_df = observations.to_dataframe()
    event_df = events_to_dataframe(events)
    harmonic_vocabulary_df = chord_states_dataframe(build_harmonic_state_space())
    modal_contexts_df = modal_contexts_dataframe(build_modal_contexts())

    export_tables(
        {
            "events": event_df,
            "observations": observations_df,
            "harmonic_vocabulary": harmonic_vocabulary_df,
            "modal_contexts": modal_contexts_df,
        },
        output_dir / "common",
        excel_name="common.xlsx",
    )

    finite_result = None
    hdp_result = None

    if args.model in {"finite_hmm", "both"}:
        finite_output = output_dir / "finite_hmm"
        finite_output.mkdir(parents=True, exist_ok=True)

        if observations.observation_type == "pitch_class":
            finite_observations = observations
        else:
            finite_observations = build_observation_sequence(events, "pitch_class")

        finite_result = FiniteChordHMM().fit_predict(finite_observations)
        finite_tables = build_finite_tables(finite_observations.to_dataframe(), finite_result)
        export_tables(finite_tables, finite_output, excel_name="finite_hmm.xlsx")
        generate_finite_figures(finite_result, finite_output)

    if args.model in {"hdp_hmm", "both"}:
        hdp_output = output_dir / "hdp_hmm"
        hdp_output.mkdir(parents=True, exist_ok=True)

        model = TruncatedHDPHMM(
            n_states=args.K,
            alpha=args.alpha,
            alpha0=args.alpha0,
            gamma=args.gamma,
            eta=args.eta,
            kappa=args.kappa,
            n_iters=args.iters,
            burn_in=args.burn_in,
            seed=args.seed,
        )
        hdp_result = model.fit(observations)
        interpretation_df = interpret_hdp_states(hdp_result)
        hdp_tables = build_hdp_tables(observations_df, hdp_result, interpretation_df=interpretation_df)
        export_tables(hdp_tables, hdp_output, excel_name="hdp_hmm.xlsx")
        generate_hdp_figures(hdp_result, hdp_output)

    comparison_df = compare_models(finite_result, hdp_result)
    if not comparison_df.empty:
        export_tables({"comparison": comparison_df}, output_dir, excel_name="comparison.xlsx")

    summary = _summary_text(args.input, observations.observation_type, finite_result, hdp_result)
    _write_text(output_dir / "summary.txt", summary)
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.library_batch import analyze_library


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clasificacion y analisis por lotes de una biblioteca MusicXML/MXL.")
    parser.add_argument("--library-dir", required=True, help="Directorio con archivos .mxl/.musicxml/.xml.")
    parser.add_argument("--output-dir", default="artifacts/outputs/library_batch", help="Directorio de salida.")
    parser.add_argument("--obs", default="pitch_class", help="Tipo de observacion para el analisis.")
    parser.add_argument("--model", default="finite_hmm", choices=("finite_hmm", "hdp_hmm", "both"), help="Modelo a ejecutar sobre la biblioteca.")
    parser.add_argument("--limit", type=int, default=None, help="Limita el numero de obras a analizar.")
    parser.add_argument("--prefer-treble", action="store_true", help="Prioriza partes en clave de sol al fusionar la partitura.")
    parser.add_argument("--K", type=int, default=20, help="Numero maximo de estados truncados para HDP-HMM.")
    parser.add_argument("--iters", type=int, default=80, help="Iteraciones del HDP-HMM.")
    parser.add_argument("--burn-in", dest="burn_in", type=int, default=40, help="Burn-in del HDP-HMM.")
    parser.add_argument("--seed", type=int, default=7, help="Semilla reproducible.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    outputs = analyze_library(
        library_dir=args.library_dir,
        output_dir=args.output_dir,
        obs_type=args.obs,
        model=args.model,
        limit=args.limit,
        prefer_treble=args.prefer_treble,
        hdp_params={
            "n_states": args.K,
            "n_iters": args.iters,
            "burn_in": args.burn_in,
            "seed": args.seed,
        },
    )

    print("Clasificacion y analisis completados.")
    print(f"Catalogo: {Path(outputs['catalog_dir']).resolve()}")
    print(f"Analisis: {Path(outputs['analysis_dir']).resolve()}")


if __name__ == "__main__":
    main()

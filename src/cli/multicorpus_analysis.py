from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.library_batch import analyze_multicorpus
from src.data.multicorpus import CorpusSource


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analisis multicorpus para MusicXML/MXL procedente de varias colecciones.")
    parser.add_argument("--output-dir", default="artifacts/outputs/multicorpus_batch", help="Directorio de salida.")
    parser.add_argument("--obs", default="pitch_class", help="Tipo de observacion.")
    parser.add_argument("--model", default="finite_hmm", choices=("finite_hmm", "hdp_hmm", "both"), help="Modelo a ejecutar.")
    parser.add_argument("--limit", type=int, default=None, help="Limita el numero de obras analizadas tras construir el catalogo.")
    parser.add_argument("--sample-per-source", type=int, default=None, help="Limita el numero de archivos muestreados por fuente antes del catalogo profundo.")
    parser.add_argument("--prefer-treble", action="store_true", help="Prioriza partes en clave de sol.")
    parser.add_argument("--K", type=int, default=20, help="Numero maximo de estados truncados para HDP-HMM.")
    parser.add_argument("--iters", type=int, default=80, help="Iteraciones del HDP-HMM.")
    parser.add_argument("--burn-in", dest="burn_in", type=int, default=40, help="Burn-in del HDP-HMM.")
    parser.add_argument("--seed", type=int, default=7, help="Semilla reproducible.")
    parser.add_argument("--include-library", help="Directorio del corpus MusicXML generico o de musetrainer/library.")
    parser.add_argument("--include-asap", help="Directorio raiz del dataset ASAP con metadata.csv.")
    parser.add_argument("--include-symbtr", help="Directorio raiz de SymbTr o su carpeta MusicXML.")
    parser.add_argument("--include-pdmx", help="Directorio local del dataset PDMX ya descargado desde Zenodo.")
    parser.add_argument("--include-jazzmus", help="Directorio con MusicXML o JSON exportables de JAZZMUS.")
    return parser


def _collect_sources(args: argparse.Namespace) -> list[CorpusSource]:
    sources: list[CorpusSource] = []
    if args.include_library:
        sources.append(CorpusSource(name="MuseTrainer", source_type="generic", root_dir=args.include_library))
    if args.include_asap:
        sources.append(CorpusSource(name="ASAP", source_type="asap", root_dir=args.include_asap))
    if args.include_symbtr:
        sources.append(CorpusSource(name="SymbTr", source_type="symbtr", root_dir=args.include_symbtr))
    if args.include_pdmx:
        sources.append(CorpusSource(name="PDMX", source_type="pdmx", root_dir=args.include_pdmx))
    if args.include_jazzmus:
        sources.append(CorpusSource(name="JAZZMUS", source_type="jazzmus", root_dir=args.include_jazzmus))
    if not sources:
        raise SystemExit("Debes indicar al menos una fuente con --include-library, --include-asap, --include-symbtr, --include-pdmx o --include-jazzmus.")
    return sources


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    sources = _collect_sources(args)

    outputs = analyze_multicorpus(
        sources=sources,
        output_dir=args.output_dir,
        obs_type=args.obs,
        model=args.model,
        limit=args.limit,
        file_limit_per_source=args.sample_per_source,
        prefer_treble=args.prefer_treble,
        hdp_params={
            "n_states": args.K,
            "n_iters": args.iters,
            "burn_in": args.burn_in,
            "seed": args.seed,
        },
    )

    print("Analisis multicorpus completado.")
    print(f"Catalogo: {Path(outputs['catalog_dir']).resolve()}")
    print(f"Analisis: {Path(outputs['analysis_dir']).resolve()}")


if __name__ == "__main__":
    main()

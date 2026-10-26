"""Reporte reproducible del corpus por obra (tarea 1.4): cuantas obras quedan, cuantas se
deduplican, cuantas fallan y por que, para MuseTrainer + la muestra estratificada de SymbTr.

Uso (desde la raiz del repo):
    python scripts/reporte_datos_corpus.py [--library DIR] [--symbtr DIR] [--n-symbtr 100] [--seed 7]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.analysis.library_batch import prepare_corpus  # noqa: E402
from src.data.multicorpus import CorpusSource, build_source_catalog, build_symbtr_catalog  # noqa: E402

EXTERNAL = Path(__file__).resolve().parents[2] / "Melodies" / "external"
MIN_EVENTS = 20  # preregistro A: obras con menos de 20 eventos se excluyen y se cuentan


def _report_source(name: str, catalog: pd.DataFrame, extraction: str) -> None:
    prepared = prepare_corpus(catalog, extraction=extraction)
    short = prepared.pieces[prepared.pieces["n_events"] < MIN_EVENTS]
    print(f"\n== {name}: {len(catalog)} archivos en el catalogo")
    print(f"  fallan: {len(prepared.failures)}")
    for _, row in prepared.failures.iterrows():
        print(f"    [{row['error_stage']}] {Path(str(row['path'])).name}: {str(row['error'])[:140]}")
    print(f"  duplicadas quitadas: {len(prepared.duplicates)}")
    for _, row in prepared.duplicates.iterrows():
        print(f"    {Path(str(row['path'])).name} == {Path(str(row['duplicate_of'])).name}")
    print(f"  quedan: {len(prepared.pieces)} (de ellas {len(short)} con < {MIN_EVENTS} eventos; quedarian {len(prepared.pieces) - len(short)})")
    for _, row in short.iterrows():
        print(f"    corta: {Path(str(row['path'])).name} ({row['n_events']} eventos)")
    if "makam" in prepared.pieces.columns:
        print(f"  makams en la muestra: {catalog['makam'].nunique()}; tras fallas y duplicados: {prepared.pieces['makam'].nunique()}")
    print(f"  eventos: total {int(prepared.pieces['n_events'].sum())}, mediana por obra {prepared.pieces['n_events'].median():.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", default=str(EXTERNAL / "library" / "scores"))
    parser.add_argument("--symbtr", default=str(EXTERNAL / "SymbTr"))
    parser.add_argument("--n-symbtr", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--extraction", default="melody")
    args = parser.parse_args()

    print(f"Extraccion: {args.extraction}; SymbTr: {args.n_symbtr} obras estratificadas por makam, semilla {args.seed}")
    library = build_source_catalog(CorpusSource("MuseTrainer", "generic", args.library))
    _report_source("MuseTrainer", library, args.extraction)
    symbtr_root = Path(args.symbtr)
    symbtr_dir = symbtr_root / "MusicXML" if (symbtr_root / "MusicXML").exists() else symbtr_root
    symbtr = build_symbtr_catalog(symbtr_dir, file_limit=args.n_symbtr, seed=args.seed)
    _report_source("SymbTr", symbtr, args.extraction)


if __name__ == "__main__":
    main()

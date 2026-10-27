"""Construye la partición PDMX corregida (tarea 1.5) y la compara con la de la tesis.

Lee la caché de corpus de la corrida de tesis, agrupa por obra con metadatos y
contenido (``Comparacion.splits.assign_work_groups``), parte con el orden
independiente del sistema operativo y reporta:

- obras y piezas de prueba;
- grupos fusionados por título y por contenido;
- solape máximo de 16-gramas de intervalos entre prueba y entrenamiento, en la
  partición nueva y en la original;
- diferencias con ``plan_pdmx_canonical_20260825``.

Uso:

    python scripts/particion_pdmx.py CACHE.jsonl PLAN_DIR [--manifest SALIDA.json]
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Comparacion.splits import (  # noqa: E402
    CONTENT_CONTAINMENT_THRESHOLD,
    _interval_ngrams,
    assign_work_groups,
    build_grouped_fixed_splits,
    normalized_piece_key,
    split_manifest,
)
from next_token_experiment.schemas import PreparedPiece  # noqa: E402

# Los mismos de la corrida de tesis (Comparacion/config.py).
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.10
SPLIT_SEED = 7


def load_pieces(cache_path: Path) -> list[PreparedPiece]:
    names = {field.name for field in fields(PreparedPiece)}
    pieces = []
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") != "piece":
                continue
            payload = {key: value for key, value in row.items() if key in names}
            payload.setdefault("vocabulary", [])
            pieces.append(PreparedPiece(**payload))
    return pieces


def max_test_overlap(
    pieces: list[PreparedPiece], test_ids: set[str], train_ids: set[str]
) -> tuple[float, str, str, int]:
    """Mayor fracción de 16-gramas de una pieza de prueba presentes en una de entrenamiento."""

    ngrams = {normalized_piece_key(piece.piece_id): _interval_ngrams(list(piece.tokens)) for piece in pieces}
    titles = {normalized_piece_key(piece.piece_id): piece.title for piece in pieces}
    index: dict[tuple[int, ...], set[str]] = {}
    for key in train_ids:
        for gram in ngrams[key]:
            index.setdefault(gram, set()).add(key)
    best = (0.0, "", "")
    n_over_threshold = 0
    for key in sorted(test_ids):
        grams = ngrams[key]
        counts: dict[str, int] = {}
        for gram in grams:
            for other in index.get(gram, ()):
                counts[other] = counts.get(other, 0) + 1
        if not counts:
            continue
        other, shared = max(counts.items(), key=lambda item: (item[1], item[0]))
        overlap = shared / len(grams)
        n_over_threshold += overlap >= CONTENT_CONTAINMENT_THRESHOLD
        if overlap > best[0]:
            best = (overlap, titles[key], titles[other])
    return (*best, n_over_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cache", type=Path)
    parser.add_argument("plan_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    pieces = load_pieces(args.cache)
    grouping = assign_work_groups(pieces)
    splits = build_grouped_fixed_splits(
        pieces,
        test_fraction=TEST_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        seed=SPLIT_SEED,
        grouping=grouping,
    )
    plan = json.loads((args.plan_dir / "execution_plan.json").read_text(encoding="utf-8"))["splits"]
    reference = {name: plan[name]["piece_ids"] for name in ("test", "validation", "train_pool")}
    manifest = split_manifest(splits, reference=reference)
    manifest["content_merges"] = grouping.content_merges
    manifest["title_merges"] = grouping.title_merges

    print(f"piezas: {len(pieces)}  obras: {len(set(grouping.group_of.values()))}")
    print(f"sha256 de la partición: {manifest['sha256']}")
    for name, part in manifest["splits"].items():
        print(f"  {name:10} piezas={part['n_pieces']:5}  obras={part['n_works']:5}")

    print(f"\ngrupos fusionados por título (otro compositor): {len(grouping.title_merges)}")
    for merge in grouping.title_merges:
        print(f"  {merge['labels'][0][:40]!r} <-> {merge['labels'][1][:40]!r}")

    print(f"\ngrupos fusionados por contenido: {len(grouping.content_merges)}")
    for merge in grouping.content_merges:
        left, right = merge["titles"]
        print(f"  {merge['containment']:.2f} ({merge['n_shared_ngrams']:3} n-gramas)  {left[:38]!r} <-> {right[:38]!r}")

    def keys(names: tuple[str, ...], source: dict[str, list[str]]) -> set[str]:
        return {normalized_piece_key(piece_id) for name in names for piece_id in source[name]}

    new_ids = {name: part["piece_ids"] for name, part in manifest["splits"].items()}
    print("\nsolape máximo de 16-gramas prueba -> entrenamiento (pool + validación):")
    for label, source in (("nueva", new_ids), ("original", reference)):
        overlap, test_title, train_title, n_over = max_test_overlap(
            pieces, keys(("test",), source), keys(("train_pool", "validation"), source)
        )
        print(
            f"  {label:8} máx={overlap:.2f} ({test_title[:30]!r} <- {train_title[:30]!r});"
            f" piezas de prueba >= {CONTENT_CONTAINMENT_THRESHOLD:.0%}: {n_over}"
        )

    print("\ndiferencias con la partición original:")
    for name, change in manifest["comparison"].items():
        print(
            f"  {name:10} compartidas={change['n_shared']:5}  sólo nueva={change['n_only_new']:5}"
            f"  sólo original={change['n_only_reference']:5}"
        )

    if args.manifest is not None:
        args.manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nmanifiesto: {args.manifest}")


if __name__ == "__main__":
    main()

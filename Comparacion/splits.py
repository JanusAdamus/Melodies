from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import PurePosixPath
import random

from next_token_experiment.experiment.splits import (
    GENERIC_LABELS,
    canonicalization_details,
    canonicalize_work_label,
)
from next_token_experiment.schemas import PreparedPiece

# Agrupamiento por contenido (auditoría C3/T3, 2026-10-26). Dos piezas son la
# misma obra si comparten al menos el 25 % de los 16-gramas de intervalos (mod 12,
# invariantes a transposición) de la más corta y al menos 8 de esos 16-gramas.
# El mínimo absoluto evita que una sola nota repetida 17 veces fusione obras.
CONTENT_NGRAM = 16
CONTENT_CONTAINMENT_THRESHOLD = 0.25
CONTENT_MIN_SHARED_NGRAMS = 8
# Preregistro B: en frac = 1.0 cada data_seed entrena con un 90 % del pool.
FULL_FRACTION_SUBSAMPLE = 0.9
# Prefijo de la obra de una pieza cuyos metadatos no identifican obra.
PIECE_WORK_PREFIX = "pieza:"


@dataclass(frozen=True)
class FixedSplits:
    test_pieces: list[PreparedPiece]
    validation_pieces: list[PreparedPiece]
    train_pool_pieces: list[PreparedPiece]


def normalized_piece_key(piece_id: str) -> str:
    """Clave de orden independiente del sistema operativo (ruta POSIX relativa)."""

    return piece_id.replace("\\", "/")


def _infer_source_label(piece: PreparedPiece) -> str:
    source_path = PurePosixPath(normalized_piece_key(piece.source_path))
    path_parts = [part.lower() for part in source_path.parts]
    for label in ("musetrainer", "symbtr", "asap", "pdmx"):
        if label in path_parts:
            return label
    if "external" in path_parts:
        index = path_parts.index("external")
        if index + 1 < len(path_parts):
            return path_parts[index + 1]
    return "unknown"


def _length_bucket(piece: PreparedPiece, short_cut: int, medium_cut: int) -> str:
    length = len(piece.tokens)
    if length <= short_cut:
        return "short"
    if length <= medium_cut:
        return "medium"
    return "long"


def _allocate_group_counts(n_total: int, target_fraction: float) -> int:
    if n_total <= 0:
        return 0
    raw = n_total * target_fraction
    return max(0, min(n_total, int(round(raw))))


def _key(piece: PreparedPiece) -> str:
    return normalized_piece_key(piece.piece_id)


def _sorted_shuffle(pieces: list[PreparedPiece], seed: int) -> list[PreparedPiece]:
    shuffled = sorted(pieces, key=_key)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def build_fixed_splits(
    pieces: list[PreparedPiece],
    *,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> FixedSplits:
    if not pieces:
        raise ValueError("At least one prepared piece is required.")
    if test_fraction <= 0.0 or validation_fraction <= 0.0 or test_fraction + validation_fraction >= 1.0:
        raise ValueError("Invalid split fractions.")

    ordered = sorted(pieces, key=_key)
    lengths = sorted(len(piece.tokens) for piece in ordered)
    short_cut = lengths[min(len(lengths) - 1, max(0, math.floor(len(lengths) / 3) - 1))]
    medium_cut = lengths[min(len(lengths) - 1, max(0, math.floor(2 * len(lengths) / 3) - 1))]

    strata: dict[tuple[str, str], list[PreparedPiece]] = {}
    for piece in ordered:
        key = (_infer_source_label(piece), _length_bucket(piece, short_cut, medium_cut))
        strata.setdefault(key, []).append(piece)

    test_pieces: list[PreparedPiece] = []
    validation_pieces: list[PreparedPiece] = []
    train_pool_pieces: list[PreparedPiece] = []

    for index, key in enumerate(sorted(strata)):
        stratum = _sorted_shuffle(strata[key], seed + index)
        n_total = len(stratum)
        n_test = _allocate_group_counts(n_total, test_fraction)
        remaining_after_test = n_total - n_test
        adjusted_validation_fraction = validation_fraction / (1.0 - test_fraction)
        n_validation = min(
            remaining_after_test,
            _allocate_group_counts(remaining_after_test, adjusted_validation_fraction),
        )
        if remaining_after_test >= 3 and n_validation == 0:
            n_validation = 1
        if n_total >= 4 and n_test == 0:
            n_test = 1
            if n_validation >= remaining_after_test:
                n_validation = max(0, remaining_after_test - 1)

        test_pieces.extend(stratum[:n_test])
        validation_pieces.extend(stratum[n_test : n_test + n_validation])
        train_pool_pieces.extend(stratum[n_test + n_validation :])

    if not test_pieces or not validation_pieces or not train_pool_pieces:
        raise ValueError("Stratified split produced an empty partition.")

    return FixedSplits(
        test_pieces=sorted(test_pieces, key=_key),
        validation_pieces=sorted(validation_pieces, key=_key),
        train_pool_pieces=sorted(train_pool_pieces, key=_key),
    )


def build_nested_training_subsets(
    train_pool_pieces: list[PreparedPiece],
    *,
    fractions: tuple[float, ...],
    data_seed: int,
    full_fraction_subsample: float = FULL_FRACTION_SUBSAMPLE,
) -> list[tuple[float, list[PreparedPiece]]]:
    if not train_pool_pieces:
        raise ValueError("Training pool cannot be empty.")

    ordered_pool = _sorted_shuffle(train_pool_pieces, data_seed)
    nested: list[tuple[float, list[PreparedPiece]]] = []
    for fraction in fractions:
        # Sin el submuestreo, frac = 1.0 era el pool entero para toda data_seed
        # y las réplicas de datos no variaban nada (auditoría C4).
        effective = full_fraction_subsample if fraction >= 1.0 else fraction
        n_items = max(1, min(len(ordered_pool), int(round(effective * len(ordered_pool)))))
        subset = sorted(ordered_pool[:n_items], key=_key)
        nested.append((fraction, subset))
    return nested


@dataclass(frozen=True)
class WorkGrouping:
    """Obra asignada a cada pieza: etiqueta de metadatos más fusiones por contenido."""

    group_of: dict[str, str]
    content_merges: list[dict[str, object]]
    title_merges: list[dict[str, object]]


def _label_group(piece: PreparedPiece) -> str:
    label = canonicalize_work_label(f"{piece.composer} {piece.title}")
    # Etiquetas vacías, genéricas o de una palabra («anon», «j», «2») no
    # identifican una obra: la pieza queda sola salvo que el contenido la una.
    if not label or label in GENERIC_LABELS or len(label.split()) < 2:
        return f"{PIECE_WORK_PREFIX}{_key(piece)}"
    return label


def _title_key(piece: PreparedPiece) -> str | None:
    """Título sin compositor, sin diacríticos ni puntuación; None si no identifica."""

    details = canonicalization_details(piece.title)
    if details["is_empty"] or details["is_generic"] or details["is_short"]:
        return None
    return str(details["aggressive_key"]) or None


def _interval_ngrams(tokens: list[int]) -> set[tuple[int, ...]]:
    intervals = [(right - left) % 12 for left, right in zip(tokens, tokens[1:])]
    return {
        tuple(intervals[start : start + CONTENT_NGRAM])
        for start in range(len(intervals) - CONTENT_NGRAM + 1)
    }


def _shared_ngram_counts(ngrams: dict[str, set[tuple[int, ...]]]) -> dict[tuple[str, str], int]:
    index: dict[tuple[int, ...], list[str]] = {}
    for piece_key in sorted(ngrams):
        for gram in ngrams[piece_key]:
            index.setdefault(gram, []).append(piece_key)
    shared: dict[tuple[str, str], int] = {}
    for members in index.values():
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                shared[(left, right)] = shared.get((left, right), 0) + 1
    return shared


def assign_work_groups(pieces: list[PreparedPiece]) -> WorkGrouping:
    """Agrupa piezas por obra (unión transitiva) con tres reglas.

    1. Misma etiqueta compositor + título (``canonicalize_work_label``).
    2. Mismo título normalizado, si tiene al menos dos palabras y no es genérico.
    3. Contenido: ver ``CONTENT_*`` arriba.
    """

    by_key = {_key(piece): piece for piece in pieces}
    if len(by_key) != len(pieces):
        raise ValueError("Piece ids collide after path normalization.")
    labels = {piece_key: _label_group(piece) for piece_key, piece in by_key.items()}
    parent = {label: label for label in labels.values()}

    def find(label: str) -> str:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: str, right: str) -> bool:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return False
        parent[max(root_left, root_right)] = min(root_left, root_right)
        return True

    # Mismo título con el compositor escrito de otra forma («Beethoven» frente a
    # «arr. Verona»): arreglos simplificados que no comparten 16-gramas.
    title_merges: list[dict[str, object]] = []
    first_with_title: dict[str, str] = {}
    for piece_key in sorted(by_key):
        title_key = _title_key(by_key[piece_key])
        if title_key is None:
            continue
        anchor = first_with_title.setdefault(title_key, piece_key)
        if union(labels[anchor], labels[piece_key]):
            title_merges.append({"pieces": [anchor, piece_key], "labels": [labels[anchor], labels[piece_key]]})

    ngrams = {piece_key: _interval_ngrams(list(piece.tokens)) for piece_key, piece in by_key.items()}
    merges: list[dict[str, object]] = []
    for (left, right), n_shared in sorted(_shared_ngram_counts(ngrams).items()):
        smaller = min(len(ngrams[left]), len(ngrams[right]))
        containment = n_shared / smaller
        if n_shared < CONTENT_MIN_SHARED_NGRAMS or containment < CONTENT_CONTAINMENT_THRESHOLD:
            continue
        if not union(labels[left], labels[right]):
            continue
        merges.append(
            {
                "pieces": [left, right],
                "titles": [by_key[left].title, by_key[right].title],
                "labels": [labels[left], labels[right]],
                "containment": containment,
                "n_shared_ngrams": n_shared,
            }
        )
    group_of = {by_key[piece_key].piece_id: find(label) for piece_key, label in labels.items()}
    return WorkGrouping(group_of=group_of, content_merges=merges, title_merges=title_merges)


def build_grouped_fixed_splits(
    pieces: list[PreparedPiece],
    *,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
    grouping: WorkGrouping | None = None,
) -> FixedSplits:
    """Partición por obra (metadatos + contenido), independiente del sistema operativo.

    Las piezas devueltas llevan en ``canonical_work_id`` la obra agrupada, de
    modo que la unidad de inferencia por obra respeta las fusiones por contenido.
    """

    grouping = grouping or assign_work_groups(pieces)
    grouped = [replace(piece, canonical_work_id=grouping.group_of[piece.piece_id]) for piece in pieces]
    members: dict[str, list[PreparedPiece]] = {}
    for piece in sorted(grouped, key=_key):
        members.setdefault(piece.canonical_work_id, []).append(piece)
    representatives = [group[0] for _, group in sorted(members.items())]
    chosen = build_fixed_splits(
        representatives,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    def expand(selected: list[PreparedPiece]) -> list[PreparedPiece]:
        return sorted(
            [piece for representative in selected for piece in members[representative.canonical_work_id]],
            key=_key,
        )

    return FixedSplits(
        test_pieces=expand(chosen.test_pieces),
        validation_pieces=expand(chosen.validation_pieces),
        train_pool_pieces=expand(chosen.train_pool_pieces),
    )


def _split_lists(splits: FixedSplits) -> dict[str, list[PreparedPiece]]:
    return {
        "test": splits.test_pieces,
        "validation": splits.validation_pieces,
        "train_pool": splits.train_pool_pieces,
    }


def split_manifest(
    splits: FixedSplits,
    *,
    reference: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    """Manifiesto con hash de la partición y, si se da, su diferencia con otra.

    El hash usa ids normalizados, así que no depende del sistema operativo.
    ``reference`` mapea nombre de partición a ids de piezas (p. ej. del
    ``execution_plan.json`` de la corrida original).
    """

    normalized = {name: sorted(_key(piece) for piece in part) for name, part in _split_lists(splits).items()}
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    manifest: dict[str, object] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "grouping": {
            "ngram": CONTENT_NGRAM,
            "containment_threshold": CONTENT_CONTAINMENT_THRESHOLD,
            "min_shared_ngrams": CONTENT_MIN_SHARED_NGRAMS,
        },
        "splits": {
            name: {
                "n_pieces": len(part),
                "n_works": len({piece.canonical_work_id for piece in part}),
                "piece_ids": normalized[name],
            }
            for name, part in _split_lists(splits).items()
        },
    }
    if reference is not None:
        comparison = {}
        for name, ids in reference.items():
            new_ids = set(normalized[name])
            old_ids = {normalized_piece_key(piece_id) for piece_id in ids}
            comparison[name] = {
                "n_shared": len(new_ids & old_ids),
                "n_only_new": len(new_ids - old_ids),
                "n_only_reference": len(old_ids - new_ids),
            }
        manifest["comparison"] = comparison
        manifest["comparison_status"] = "reported"
    else:
        # Preregistro B pide la diferencia con el conjunto de prueba anterior: si
        # falta la referencia, el manifiesto lo dice en vez de callarlo.
        manifest["comparison_status"] = "not_reported_no_reference_given"
    return manifest

from __future__ import annotations

import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
import warnings

from src.data.library_catalog import infer_composer
from src.data.observations import PITCH_CLASS_NAMES
from src.data.parsing import extract_events, parse_score

from ..config import ExperimentConfig, PreprocessingConfig, RepresentationConfig
from ..experiment.splits import canonicalize_work_label
from ..schemas import CorpusPreparationResult, ExclusionRecord, PreparedPiece
from .reader import build_piece_id, discover_score_paths

try:
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:  # pragma: no cover - depende del entorno.
    pass

REST_EVENT_LABEL = "REST"


def _duration_to_bin_index(duration: float, duration_bins: tuple[float, ...]) -> int:
    closest = min(duration_bins, key=lambda item: abs(item - duration))
    return duration_bins.index(closest)


def _metrical_position_label(beat: float | None) -> str:
    if beat is None:
        return "unknown"

    beat_value = float(beat)
    if math.isclose(beat_value, 1.0, abs_tol=1e-3):
        return "downbeat"

    nearest_integer = round(beat_value)
    if math.isclose(beat_value, float(nearest_integer), abs_tol=1e-3):
        return "beat"

    half_beat = beat_value * 2.0
    if math.isclose(half_beat, round(half_beat), abs_tol=1e-3):
        return "offbeat"

    return "subbeat"


def build_representation_vocabulary(config: RepresentationConfig, representation: str | None = None) -> list[str]:
    """Build a corpus-wide, deterministic vocabulary for the supported representations."""

    target = representation or config.primary
    if target == "pitch_class":
        return PITCH_CLASS_NAMES.copy()
    if target == "pitch_class_duration":
        values = []
        for pitch_class in PITCH_CLASS_NAMES:
            for duration in config.duration_bins:
                values.append(f"{pitch_class}|dur={duration:g}")
        return values
    if target == "event_pitch_duration_metrical":
        values = []
        pitch_labels = PITCH_CLASS_NAMES + [REST_EVENT_LABEL]
        for pitch_label in pitch_labels:
            for duration in config.duration_bins:
                for metrical_level in config.metrical_levels:
                    values.append(f"{pitch_label}|dur={duration:g}|metric={metrical_level}")
        return values
    raise ValueError(f"Unsupported representation: {target}")


def build_representation_tokens(
    events,
    config: RepresentationConfig,
    representation: str | None = None,
) -> tuple[list[int], list[str]]:
    """Encode events with a corpus-consistent discrete representation."""

    target = representation or config.primary
    vocabulary = build_representation_vocabulary(config, target)
    tokens: list[int] = []

    if target == "pitch_class":
        for event in events:
            if event.representative_pitch_class is None:
                raise ValueError("Found an event without representative pitch class.")
            tokens.append(int(event.representative_pitch_class))
        return tokens, vocabulary

    if target == "pitch_class_duration":
        n_bins = len(config.duration_bins)
        for event in events:
            if event.representative_pitch_class is None:
                raise ValueError("Found an event without representative pitch class.")
            duration_index = _duration_to_bin_index(float(event.duration), config.duration_bins)
            token = int(event.representative_pitch_class) * n_bins + duration_index
            tokens.append(token)
        return tokens, vocabulary

    if target == "event_pitch_duration_metrical":
        n_duration_bins = len(config.duration_bins)
        n_metrical_levels = len(config.metrical_levels)
        metrical_lookup = {label: index for index, label in enumerate(config.metrical_levels)}
        rest_index = len(PITCH_CLASS_NAMES)

        for event in events:
            duration_index = _duration_to_bin_index(float(event.duration), config.duration_bins)
            metrical_label = _metrical_position_label(event.beat)
            metrical_index = metrical_lookup.get(metrical_label, metrical_lookup["unknown"])
            pitch_index = rest_index if event.representative_pitch_class is None else int(event.representative_pitch_class)
            token = (pitch_index * n_duration_bins * n_metrical_levels) + (duration_index * n_metrical_levels) + metrical_index
            tokens.append(token)
        return tokens, vocabulary

    raise ValueError(f"Unsupported representation: {target}")


def _piece_title_and_composer(score, path: Path) -> tuple[str, str]:
    metadata = getattr(score, "metadata", None)
    title = ""
    if metadata is not None:
        title = metadata.title or metadata.movementName or ""
    if not title:
        title = path.stem.replace("_", " ")
    composer = infer_composer(path.name, metadata.composer if metadata is not None else None)
    return title, composer


def preprocess_score_file(
    path: Path,
    root_dir: str,
    preprocessing_config: PreprocessingConfig,
    representation_config: RepresentationConfig,
    min_events_per_piece: int,
    representation: str | None = None,
) -> PreparedPiece | ExclusionRecord:
    """Parse a score file and convert it into a discrete token sequence."""

    piece_id = build_piece_id(path, root_dir)
    try:
        score = parse_score(path)
    except Exception as exc:
        return ExclusionRecord(
            piece_id=piece_id,
            path=str(path),
            reason="parse_error",
            detail=f"{exc.__class__.__name__}: {exc}",
        )

    try:
        events = extract_events(
            score,
            include_rests=preprocessing_config.include_rests,
            prefer_treble=preprocessing_config.prefer_treble,
        )
    except Exception as exc:
        return ExclusionRecord(
            piece_id=piece_id,
            path=str(path),
            reason="event_extraction_error",
            detail=f"{exc.__class__.__name__}: {exc}",
        )

    if len(events) < min_events_per_piece:
        return ExclusionRecord(
            piece_id=piece_id,
            path=str(path),
            reason="too_short",
            detail=f"events={len(events)} < min_events_per_piece={min_events_per_piece}",
        )

    try:
        tokens, vocabulary = build_representation_tokens(events, representation_config, representation)
    except Exception as exc:
        return ExclusionRecord(
            piece_id=piece_id,
            path=str(path),
            reason="tokenization_error",
            detail=f"{exc.__class__.__name__}: {exc}",
        )

    title, composer = _piece_title_and_composer(score, path)
    canonical_work_id = canonicalize_work_label(f"{composer} {title}")
    return PreparedPiece(
        piece_id=piece_id,
        source_path=str(path),
        title=title,
        composer=composer,
        canonical_work_id=canonical_work_id,
        representation=representation or representation_config.primary,
        vocabulary=vocabulary,
        tokens=tokens,
        n_events=len(events),
        metadata={
            "relative_path": piece_id,
            "time_signature_count": len(
                list(score.recurse().getElementsByClass("TimeSignature"))
            ),
            "part_count": len(score.parts) if score.parts else 1,
            "contains_rests": any(event.kind == "rest" for event in events),
        },
    )


def _silence_worker_warnings() -> None:
    """Mute music21 parser chatter inside pool workers.

    A full corpus pass emits hundreds of thousands of MusicXMLWarnings that cost real
    time to format and write, and that say nothing actionable about the run.
    """

    warnings.simplefilter("ignore")


def _prepare_one_without_vocabulary(
    path: Path,
    **kwargs: object,
) -> PreparedPiece | ExclusionRecord:
    """Run ``preprocess_score_file`` in a worker, dropping the corpus-wide vocabulary.

    Every piece carries an identical vocabulary list, so shipping it back through the
    pool once per score would dominate the inter-process traffic. The parent restores
    a single shared instance.
    """

    prepared = preprocess_score_file(path=path, **kwargs)  # type: ignore[arg-type]
    if isinstance(prepared, PreparedPiece):
        return replace(prepared, vocabulary=[])
    return prepared


def _encode_cache_entry(entry: PreparedPiece | ExclusionRecord) -> str:
    payload = asdict(entry)
    if isinstance(entry, ExclusionRecord):
        payload["kind"] = "exclusion"
    else:
        payload["kind"] = "piece"
        payload.pop("vocabulary", None)
    return json.dumps(payload, ensure_ascii=False)


def _decode_cache_entry(
    payload: dict[str, object],
    vocabulary: list[str],
) -> PreparedPiece | ExclusionRecord:
    kind = payload.pop("kind")
    if kind == "exclusion":
        return ExclusionRecord(**payload)  # type: ignore[arg-type]
    return PreparedPiece(vocabulary=vocabulary, **payload)  # type: ignore[arg-type]


def _entry_source_path(entry: PreparedPiece | ExclusionRecord) -> str:
    return entry.path if isinstance(entry, ExclusionRecord) else entry.source_path


def prepare_corpus(
    config: ExperimentConfig,
    representation: str | None = None,
    max_files: int | None = None,
    *,
    n_workers: int | None = None,
    cache_path: str | Path | None = None,
    progress_every: int = 1000,
    chunksize: int = 4,
    sample_seed: int | None = None,
) -> CorpusPreparationResult:
    """Prepare the configured corpus as a list of tokenized pieces plus exclusions.

    Parsing is CPU bound and dominates the pipeline, so scores are parsed in a process
    pool. Pass ``n_workers=1`` to force the serial path. When ``cache_path`` is set,
    each finished score is appended to a JSONL file as soon as it lands, so an
    interrupted run resumes from the cache instead of reparsing the whole corpus.
    """

    score_paths = discover_score_paths(config.corpus)
    if max_files is not None and max_files < len(score_paths):
        if sample_seed is None:
            # Path order follows the corpus directory layout, so a head slice is a
            # convenience cap, not a corpus-representative sample.
            score_paths = score_paths[:max_files]
        else:
            score_paths = sorted(random.Random(sample_seed).sample(score_paths, max_files))

    vocabulary = build_representation_vocabulary(
        config.representation,
        representation or config.representation.primary,
    )

    # El cache es del corpus entero y sobrevive entre corridas, asi que solo aportan las
    # entradas que estan en la seleccion de ESTA corrida. Sin este filtro, un cache con
    # mas partituras que `max_files` las arrastraria todas e ignoraria el muestreo.
    selected = {str(path) for path in score_paths}

    entries: list[PreparedPiece | ExclusionRecord] = []
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None and cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = _decode_cache_entry(json.loads(line), vocabulary)
                    if _entry_source_path(entry) in selected:
                        entries.append(entry)

    done_paths = {_entry_source_path(entry) for entry in entries}
    pending = [path for path in score_paths if str(path) not in done_paths]

    if entries:
        print(
            f"[prepare_corpus] resuming from cache: {len(entries)} done, {len(pending)} pending",
            file=sys.stderr,
            flush=True,
        )

    worker = partial(
        _prepare_one_without_vocabulary,
        root_dir=config.corpus.root_dir,
        preprocessing_config=config.preprocessing,
        representation_config=config.representation,
        min_events_per_piece=config.corpus.min_events_per_piece,
        representation=representation,
    )

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    cache_handle = cache_file.open("a", encoding="utf-8") if cache_file is not None else None
    try:
        if n_workers == 1 or not pending:
            _silence_worker_warnings()
            stream = (worker(path) for path in pending)
            _consume_prepared(stream, entries, cache_handle, vocabulary, len(pending), progress_every)
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_silence_worker_warnings,
            ) as pool:
                stream = pool.map(worker, pending, chunksize=chunksize)
                _consume_prepared(
                    stream, entries, cache_handle, vocabulary, len(pending), progress_every
                )
    finally:
        if cache_handle is not None:
            cache_handle.close()

    pieces = [entry for entry in entries if isinstance(entry, PreparedPiece)]
    exclusions = [entry for entry in entries if isinstance(entry, ExclusionRecord)]
    return CorpusPreparationResult(
        pieces=sorted(pieces, key=lambda item: item.piece_id),
        exclusions=sorted(exclusions, key=lambda item: item.piece_id),
    )


def _consume_prepared(
    stream,
    entries: list[PreparedPiece | ExclusionRecord],
    cache_handle,
    vocabulary: list[str],
    total: int,
    progress_every: int,
) -> None:
    """Drain prepared scores, restoring the shared vocabulary and flushing the cache."""

    for index, prepared in enumerate(stream, start=1):
        if cache_handle is not None:
            cache_handle.write(_encode_cache_entry(prepared) + "\n")
            if index % 200 == 0:
                cache_handle.flush()
        if isinstance(prepared, PreparedPiece):
            prepared = replace(prepared, vocabulary=vocabulary)
        entries.append(prepared)
        if progress_every and index % progress_every == 0:
            print(f"[prepare_corpus] {index}/{total} scores", file=sys.stderr, flush=True)


def build_preprocessing_report(preparation: CorpusPreparationResult) -> dict[str, int]:
    """Summarize the preprocessing output for logging and tests."""

    return {
        "n_prepared_pieces": len(preparation.pieces),
        "n_exclusions": len(preparation.exclusions),
        "n_total_tokens": sum(len(piece.tokens) for piece in preparation.pieces),
        "n_total_events": sum(piece.n_events for piece in preparation.pieces),
    }

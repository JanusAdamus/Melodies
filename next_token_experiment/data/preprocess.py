from __future__ import annotations

import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
import warnings

from src.data.library_catalog import infer_composer
from src.data.observations import PITCH_CLASS_NAMES
from src.data.parsing import EXTRACTION_MODES, extract_events_for_mode, parse_score

from ..config import CorpusConfig, ExperimentConfig, PreprocessingConfig, RepresentationConfig
from ..experiment.splits import canonicalize_work_label
from ..schemas import CorpusPreparationResult, ExclusionRecord, PreparedPiece
from .reader import build_piece_id, discover_score_paths

try:
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:  # pragma: no cover - depende del entorno.
    pass

REST_EVENT_LABEL = "REST"


CACHE_MODE_KEY = "extraction_mode"
# Los caches anteriores al hallazgo D1b no guardan modo: todos salieron de la extraccion
# antigua (``extract_events``), que hoy se llama ``legacy_all_parts``.
UNRECORDED_CACHE_MODE = "legacy_all_parts"
# En ``legacy_all_parts`` los tokens dependen tambien de estos parametros, asi que el cache
# los guarda. ``melody`` no los usa (y rechaza silencios), asi que no forman parte de su firma.
LEGACY_PARAMETER_KEYS = ("prefer_treble", "include_rests")
# Todas las corridas de agosto (config.json en artifacts/) usaron estos valores; un cache sin
# campos de extraccion es de agosto y por tanto de esta configuracion.
AUGUST_LEGACY_PARAMETERS = {"prefer_treble": True, "include_rests": False}
CACHE_SIGNATURE_KEYS = (CACHE_MODE_KEY, *LEGACY_PARAMETER_KEYS)


class CorpusCacheModeError(ValueError):
    """El cache de partituras se extrajo con una configuracion distinta a la de la corrida."""


def _require_extraction_mode(mode: str) -> None:
    if mode not in EXTRACTION_MODES:
        raise ValueError(f"Modo de extraccion no soportado: {mode!r}; usa uno de {EXTRACTION_MODES}.")


def _signature(mode: str, prefer_treble: object, include_rests: object) -> tuple:
    if mode == "legacy_all_parts":
        return (mode, prefer_treble, include_rests)
    return (mode,)


def extraction_signature(preprocessing: PreprocessingConfig) -> tuple:
    """Todo lo que determina los eventos extraidos: el modo y, en legacy, sus parametros."""

    _require_extraction_mode(preprocessing.extraction_mode)
    return _signature(
        preprocessing.extraction_mode,
        bool(preprocessing.prefer_treble),
        bool(preprocessing.include_rests),
    )


def recorded_extraction_signature(payload: dict[str, object]) -> tuple:
    """Firma de extraccion de una fila de cache o de un bloque ``preprocessing`` guardado.

    Sin modo es de agosto: legacy con los parametros de agosto, salvo que el bloque
    guarde los suyos (los config.json de agosto si los guardan). Con modo legacy y sin
    parametros no se sabe con que se extrajo: la firma lleva ``None`` y no coincide con
    ninguna configuracion.
    """

    if CACHE_MODE_KEY not in payload:
        return _signature(
            UNRECORDED_CACHE_MODE,
            *(payload.get(key, AUGUST_LEGACY_PARAMETERS[key]) for key in LEGACY_PARAMETER_KEYS),
        )
    return _signature(
        str(payload[CACHE_MODE_KEY]),
        *(payload.get(key) for key in LEGACY_PARAMETER_KEYS),
    )


def default_corpus_cache_name(
    max_files: int | None,
    mode: str,
    *,
    prefer_treble: bool = AUGUST_LEGACY_PARAMETERS["prefer_treble"],
    include_rests: bool = AUGUST_LEGACY_PARAMETERS["include_rests"],
) -> str:
    """Nombre de cache por configuracion de extraccion.

    El de ``legacy_all_parts`` con los parametros de agosto es el de agosto, para
    reproducirlo; otra configuracion legacy lleva sus parametros en el nombre.
    """

    _require_extraction_mode(mode)
    size = "all" if max_files is None else str(max_files)
    if mode != UNRECORDED_CACHE_MODE:
        return f"corpus_cache_{size}_{mode}.jsonl"
    if {"prefer_treble": prefer_treble, "include_rests": include_rests} == AUGUST_LEGACY_PARAMETERS:
        return f"corpus_cache_{size}.jsonl"
    return (
        f"corpus_cache_{size}_{mode}_treble{int(bool(prefer_treble))}"
        f"_rests{int(bool(include_rests))}.jsonl"
    )


def read_corpus_cache_signatures(cache_path: str | Path) -> set[tuple]:
    """Firmas de extraccion presentes en un cache JSONL (vacio si no existe o no tiene filas)."""

    path = Path(cache_path)
    signatures: set[tuple] = set()
    if not path.exists():
        return signatures
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                signatures.add(recorded_extraction_signature(json.loads(line)))
    return signatures


def check_corpus_cache_mode(
    cache_path: str | Path | None, preprocessing: PreprocessingConfig
) -> None:
    """Se niega a usar un cache cuyas filas no salieron todas de la extraccion pedida."""

    expected = extraction_signature(preprocessing)
    if cache_path is None:
        return
    signatures = read_corpus_cache_signatures(cache_path)
    if signatures and signatures != {expected}:
        suggestion = default_corpus_cache_name(
            3000,
            preprocessing.extraction_mode,
            prefer_treble=preprocessing.prefer_treble,
            include_rests=preprocessing.include_rests,
        )
        raise CorpusCacheModeError(
            f"El cache {cache_path} tiene extraccion {sorted(signatures, key=repr)} y la "
            f"corrida pide {expected}; usa un cache propio de esa extraccion (por ejemplo "
            f"{suggestion} para 3000 partituras)."
        )


# Los tokens tambien dependen de la representacion y, segun cual, de duration_bins y
# metrical_levels; el cache guarda esos campos en cada fila y no se reutiliza con otros.
REPRESENTATION_KEY = "representation"
TOKEN_PARAMETER_KEYS = ("duration_bins", "metrical_levels")
_TOKEN_PARAMETERS_BY_REPRESENTATION = {
    "pitch_class": (),
    "pitch_class_duration": ("duration_bins",),
    "event_pitch_duration_metrical": ("duration_bins", "metrical_levels"),
}


def token_fields(config: RepresentationConfig, representation: str | None = None) -> dict:
    """Campos que, junto con la extraccion, determinan los tokens de una partitura."""

    target = representation or config.primary
    if target not in _TOKEN_PARAMETERS_BY_REPRESENTATION:
        raise ValueError(f"Unsupported representation: {target}")
    fields: dict[str, object] = {REPRESENTATION_KEY: target}
    for key in _TOKEN_PARAMETERS_BY_REPRESENTATION[target]:
        fields[key] = list(getattr(config, key))
    return fields


def _token_signature(fields: dict[str, object]) -> tuple | None:
    """Firma de tokens de una fila; ``None`` si la fila no guarda representacion.

    Las exclusiones de los caches anteriores no guardan representacion; las piezas de esos
    mismos caches si, y son las que fijan la firma del cache.
    """

    if REPRESENTATION_KEY not in fields:
        return None
    target = str(fields[REPRESENTATION_KEY])
    parameters = []
    for key in _TOKEN_PARAMETERS_BY_REPRESENTATION.get(target, TOKEN_PARAMETER_KEYS):
        value = fields.get(key)
        parameters.append(None if value is None else tuple(value))
    return (target, *parameters)


def check_corpus_cache_representation(
    cache_path: str | Path | None,
    config: RepresentationConfig,
    representation: str | None = None,
) -> None:
    """Se niega a usar un cache cuyos tokens salieron de otra representacion."""

    expected = _token_signature(token_fields(config, representation))
    if cache_path is None or not Path(cache_path).exists():
        return
    signatures: set[tuple] = set()
    with Path(cache_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                signature = _token_signature(json.loads(line))
                if signature is not None:
                    signatures.add(signature)
    if signatures and signatures != {expected}:
        raise CorpusCacheModeError(
            f"El cache {cache_path} tiene tokens {sorted(signatures, key=repr)} y la "
            f"corrida pide {expected}; usa un cache propio de esa representacion."
        )


# Hallazgo D1b (auditoria): el contenido de una fila depende tambien de los filtros del
# corpus. ``min_events_per_piece`` decide entre pieza y exclusion ``too_short``; un cache
# hecho con otro umbral devolvia piezas que una corrida limpia excluye. ``extractor_version``
# se sube cuando cambia el codigo de extraccion o de tokenizacion, para que los caches
# viejos se rechacen en vez de mezclarse.
MIN_EVENTS_KEY = "min_events_per_piece"
EXTRACTOR_VERSION_KEY = "extractor_version"
EXTRACTOR_VERSION = 1
# Todas las filas de los caches de artifacts/ dicen "min_events_per_piece=32" en el detalle
# de sus exclusiones too_short, y todos los config.json de agosto usan 32.
AUGUST_MIN_EVENTS_PER_PIECE = 32
# Las filas sin version salieron del extractor vigente al introducir la version.
UNRECORDED_EXTRACTOR_VERSION = 1
CORPUS_FILTER_KEYS = (MIN_EVENTS_KEY, EXTRACTOR_VERSION_KEY)


def corpus_filter_fields(corpus: CorpusConfig) -> dict[str, object]:
    """Campos del corpus y del codigo que, ademas de extraccion y tokens, fijan una fila."""

    return {
        MIN_EVENTS_KEY: int(corpus.min_events_per_piece),
        EXTRACTOR_VERSION_KEY: EXTRACTOR_VERSION,
    }


def _filter_signature(fields: dict[str, object]) -> tuple:
    return (
        fields.get(MIN_EVENTS_KEY, AUGUST_MIN_EVENTS_PER_PIECE),
        fields.get(EXTRACTOR_VERSION_KEY, UNRECORDED_EXTRACTOR_VERSION),
    )


def check_corpus_cache_filters(cache_path: str | Path | None, corpus: CorpusConfig) -> None:
    """Se niega a usar un cache hecho con otro umbral de eventos o con otro extractor."""

    expected = _filter_signature(corpus_filter_fields(corpus))
    if cache_path is None or not Path(cache_path).exists():
        return
    signatures: set[tuple] = set()
    with Path(cache_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                signatures.add(_filter_signature(json.loads(line)))
    if signatures and signatures != {expected}:
        raise CorpusCacheModeError(
            f"El cache {cache_path} tiene (min_events_per_piece, extractor_version) "
            f"{sorted(signatures, key=repr)} y la corrida pide {expected}; usa un cache "
            "propio de esa configuracion."
        )


def check_corpus_cache(
    cache_path: str | Path | None,
    config: ExperimentConfig,
    representation: str | None = None,
) -> None:
    """Toda la firma del cache: extraccion, representacion, filtros y version."""

    check_corpus_cache_mode(cache_path, config.preprocessing)
    check_corpus_cache_representation(cache_path, config.representation, representation)
    check_corpus_cache_filters(cache_path, config.corpus)


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

    # Un modo invalido es un error de configuracion: no debe convertirse en una exclusion
    # por partitura que vacie el corpus en silencio.
    extraction_mode = preprocessing_config.extraction_mode
    _require_extraction_mode(extraction_mode)
    if extraction_mode == "melody" and preprocessing_config.include_rests:
        raise ValueError("El modo 'melody' no extrae silencios; usa include_rests=False.")

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
        events = extract_events_for_mode(
            score,
            mode=extraction_mode,
            prefer_treble=preprocessing_config.prefer_treble,
            include_rests=preprocessing_config.include_rests,
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
            "extraction_mode": extraction_mode,
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


def _encode_cache_entry(
    entry: PreparedPiece | ExclusionRecord, signature: tuple, tokens: dict[str, object]
) -> str:
    payload = asdict(entry)
    payload.update(zip(CACHE_SIGNATURE_KEYS, signature))
    payload.update(tokens)
    if isinstance(entry, ExclusionRecord):
        payload["kind"] = "exclusion"
    else:
        payload["kind"] = "piece"
        payload.pop("vocabulary", None)
    return json.dumps(payload, ensure_ascii=False)


def _decode_cache_entry(
    payload: dict[str, object],
    vocabulary: list[str],
    root_dir: str,
) -> PreparedPiece | ExclusionRecord:
    kind = payload.pop("kind")
    for key in (*CACHE_SIGNATURE_KEYS, *TOKEN_PARAMETER_KEYS, *CORPUS_FILTER_KEYS):  # ya validados
        payload.pop(key, None)
    # El piece_id es relativo a la raiz del corpus: se rehace con la raiz de ESTA corrida
    # para que coincida con el de una corrida limpia.
    if kind == "exclusion":
        payload.pop(REPRESENTATION_KEY, None)
        payload["piece_id"] = build_piece_id(Path(str(payload["path"])), root_dir)
        return ExclusionRecord(**payload)  # type: ignore[arg-type]
    piece_id = build_piece_id(Path(str(payload["source_path"])), root_dir)
    payload["piece_id"] = piece_id
    metadata = dict(payload.get("metadata") or {})  # type: ignore[arg-type]
    if "relative_path" in metadata:
        metadata["relative_path"] = piece_id
    payload["metadata"] = metadata
    # Las cachés anteriores al 2026-10-26 guardan la obra canónica truncada
    # por Path().stem (auditoría C3); se recalcula con la regla vigente.
    canonical_work_id = canonicalize_work_label(f"{payload['composer']} {payload['title']}")
    return PreparedPiece(  # type: ignore[arg-type]
        vocabulary=vocabulary, **{**payload, "canonical_work_id": canonical_work_id}
    )


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

    signature = extraction_signature(config.preprocessing)
    check_corpus_cache(cache_path, config, representation)
    tokens = {**token_fields(config.representation, representation), **corpus_filter_fields(config.corpus)}

    entries: list[PreparedPiece | ExclusionRecord] = []
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None and cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = _decode_cache_entry(
                        json.loads(line), vocabulary, config.corpus.root_dir
                    )
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
            _consume_prepared(
                stream,
                entries,
                cache_handle,
                vocabulary,
                len(pending),
                progress_every,
                signature,
                tokens,
            )
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_silence_worker_warnings,
            ) as pool:
                stream = _map_with_serial_fallback(pool, worker, pending, chunksize)
                _consume_prepared(
                    stream,
                    entries,
                    cache_handle,
                    vocabulary,
                    len(pending),
                    progress_every,
                    signature,
                    tokens,
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


def _map_with_serial_fallback(pool, worker, pending: list, chunksize: int):
    """Resultados de ``pool.map`` en orden; si el pool muere, el resto en secuencia.

    Con la máquina ocupada el pool puede morir (BrokenProcessPool, u OSError como
    «[WinError 6] Controlador no válido»).  La preparación es determinista: lo que falta
    se prepara aquí en secuencia y el resultado es el mismo.  Un error de la propia
    preparación reaparece al repetirla en este proceso.  El respaldo sólo cambia la
    fuente de resultados: los consume la misma llamada a ``_consume_prepared``.
    """

    done = 0
    try:
        # pool.map entrega en orden, así que lo entregado son los primeros de `pending`.
        for prepared in pool.map(worker, pending, chunksize=chunksize):
            yield prepared
            done += 1
    except (BrokenProcessPool, OSError) as error:
        warnings.warn(
            f"pool de procesos caído tras {done}/{len(pending)} partituras ({error!r}); "
            "el resto se prepara en secuencia",
            RuntimeWarning,
            stacklevel=2,
        )
        for path in pending[done:]:
            with warnings.catch_warnings():
                _silence_worker_warnings()
                prepared = worker(path)
            yield prepared


def _consume_prepared(
    stream,
    entries: list[PreparedPiece | ExclusionRecord],
    cache_handle,
    vocabulary: list[str],
    total: int,
    progress_every: int,
    signature: tuple,
    tokens: dict[str, object],
) -> None:
    """Drain prepared scores, restoring the shared vocabulary and flushing the cache."""

    for index, prepared in enumerate(stream, start=1):
        if cache_handle is not None:
            cache_handle.write(_encode_cache_entry(prepared, signature, tokens) + "\n")
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

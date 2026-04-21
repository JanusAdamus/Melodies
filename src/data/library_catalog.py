from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

try:
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:  # pragma: no cover
    pass

import music21

from .parsing import parse_score

MUSICXML_EXTENSIONS = {".xml", ".musicxml", ".mxl"}

COMPOSER_PATTERNS = {
    "Bach": ["bach", "bwv", "j._s._bach", "js bach"],
    "Beethoven": ["beethoven", "moonlight", "pathetique", "fur_elise"],
    "Chopin": ["chopin", "nocturne", "ballade", "waltz", "prelude op", "opus_28"],
    "Debussy": ["debussy", "clair_de_lune", "arabesque"],
    "Mozart": ["mozart", "k._545", "k._331", "turkish march", "marche_turque"],
    "Schubert": ["schubert", "ave_maria", "serenade", "standchen"],
    "Satie": ["satie", "gymnopedie", "gnossienne"],
    "Joplin": ["joplin", "entertainer", "maple_leaf"],
    "Tchaikovsky": ["swan_lake", "sugar_plum", "flowers"],
    "Pachelbel": ["canon_in_d", "pachelbel"],
    "Liszt": ["liszt", "campanella", "liebestraum"],
    "Brahms": ["hungarian_dance", "hungarian_sonata"],
    "Traditional": ["happy_birthday", "greensleeves", "bella_ciao", "carol_of_the_bells", "twinkle"],
}

PERIOD_BY_COMPOSER = {
    "Bach": "baroque",
    "Beethoven": "classical-romantic",
    "Chopin": "romantic",
    "Debussy": "modern-impressionist",
    "Mozart": "classical",
    "Schubert": "classical-romantic",
    "Satie": "modern",
    "Joplin": "ragtime",
    "Tchaikovsky": "romantic",
    "Pachelbel": "baroque",
    "Liszt": "romantic",
    "Brahms": "romantic",
    "Traditional": "traditional",
    "Unknown": "unknown",
}

FORM_KEYWORDS = {
    "sonata": ["sonata", "sonate"],
    "waltz": ["waltz", "walzer"],
    "nocturne": ["nocturne"],
    "prelude": ["prelude", "prelud", "prlude"],
    "dance": ["dance", "danse"],
    "minuet": ["minuet"],
    "canon": ["canon"],
    "rag": ["rag", "entertainer"],
    "song": ["serenade", "maria", "bells", "birthday"],
}


def iter_musicxml_files(library_dir: str | Path) -> list[Path]:
    root = Path(library_dir)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MUSICXML_EXTENSIONS)


def _normalize_text(text: str) -> str:
    return text.lower().replace("-", "_").replace(" ", "_")


def infer_composer(filename: str, metadata_composer: str | None = None) -> str:
    if metadata_composer:
        normalized = metadata_composer.strip()
        if normalized:
            return normalized
    text = _normalize_text(filename)
    for composer, patterns in COMPOSER_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return composer
    return "Unknown"


def infer_form(title: str) -> str:
    text = _normalize_text(title)
    for form, patterns in FORM_KEYWORDS.items():
        if any(pattern in text for pattern in patterns):
            return form
    return "piece"


def infer_arrangement_tag(title: str) -> str:
    text = _normalize_text(title)
    if any(keyword in text for keyword in ["easy", "beginner"]):
        return "pedagogical_easy"
    if "fingered" in text:
        return "annotated_fingering"
    if any(keyword in text for keyword in ["arrangement", "arrg", "solo_piano", "variation"]):
        return "arrangement"
    return "standard"


def infer_difficulty_bucket(title: str, note_count: int, avg_notes_per_measure: float) -> str:
    text = _normalize_text(title)
    if any(keyword in text for keyword in ["easy", "beginner"]):
        return "easy"
    if note_count >= 900 or avg_notes_per_measure >= 7.5:
        return "advanced"
    if note_count <= 120 and avg_notes_per_measure <= 3.0:
        return "easy"
    if note_count >= 350 or avg_notes_per_measure >= 4.5:
        return "intermediate-advanced"
    return "intermediate"


def _declared_key_label(score: music21.stream.Score) -> str:
    key_obj = next(iter(score.recurse().getElementsByClass(music21.key.Key)), None)
    if key_obj is not None:
        return f"{key_obj.tonic.name}:{key_obj.mode}"
    key_signature = next(iter(score.recurse().getElementsByClass(music21.key.KeySignature)), None)
    if key_signature is not None:
        return f"sharps={key_signature.sharps}"
    return ""


def _estimated_key_label(score: music21.stream.Score) -> str:
    try:
        estimated = score.analyze("key")
        return f"{estimated.tonic.name}:{estimated.mode}"
    except Exception:
        return ""


def classify_score(path: str | Path) -> dict[str, object]:
    score_path = Path(path)
    base_title = score_path.stem.replace("_", " ")
    base_composer = infer_composer(score_path.name)
    base_record = {
        "path": str(score_path),
        "filename": score_path.name,
        "title": base_title,
        "composer": base_composer,
        "period": PERIOD_BY_COMPOSER.get(base_composer, "unknown"),
        "form": infer_form(base_title),
        "arrangement_tag": infer_arrangement_tag(base_title),
        "difficulty_bucket": "",
        "part_count": None,
        "measure_count": None,
        "note_count": None,
        "chord_count": None,
        "duration_quarters": None,
        "avg_notes_per_measure": None,
        "unique_pitch_classes": None,
        "declared_key": "",
        "estimated_key": "",
        "time_signatures": "",
        "error": "",
    }
    try:
        score = parse_score(score_path)
    except Exception as exc:
        base_record["difficulty_bucket"] = infer_difficulty_bucket(base_title, 0, 0.0)
        base_record["error"] = f"{exc.__class__.__name__}: {exc}"
        return base_record

    metadata = score.metadata
    title = ""
    if metadata is not None:
        title = metadata.title or metadata.movementName or ""
    if not title:
        title = base_title

    composer = infer_composer(score_path.name, metadata.composer if metadata is not None else None)
    period = PERIOD_BY_COMPOSER.get(composer, "unknown")
    measures = list(score.recurse().getElementsByClass(music21.stream.Measure))
    notes = list(score.recurse().notes)
    note_count = len(notes)
    chord_count = len(list(score.recurse().getElementsByClass(music21.chord.Chord)))
    unique_pitch_classes = len({int(note.pitch.pitchClass) for note in score.recurse().notes if hasattr(note, "pitch")})
    measure_count = len(measures)
    duration_quarters = float(score.duration.quarterLength)
    part_count = len(score.parts) if score.parts else 1
    avg_notes_per_measure = note_count / max(measure_count, 1)
    max_measure_number = max((measure.number for measure in measures if measure.number is not None), default=measure_count)

    base_record.update(
        {
            "title": title,
            "composer": composer,
            "period": period,
            "form": infer_form(title),
            "arrangement_tag": infer_arrangement_tag(title),
            "difficulty_bucket": infer_difficulty_bucket(title, note_count, avg_notes_per_measure),
            "part_count": part_count,
            "measure_count": max_measure_number,
            "note_count": note_count,
            "chord_count": chord_count,
            "duration_quarters": duration_quarters,
            "avg_notes_per_measure": round(avg_notes_per_measure, 3),
            "unique_pitch_classes": unique_pitch_classes,
            "declared_key": _declared_key_label(score),
            "estimated_key": _estimated_key_label(score),
            "time_signatures": ", ".join(sorted({ratio.ratioString for ratio in score.recurse().getElementsByClass(music21.meter.TimeSignature)})),
        }
    )
    return base_record


def build_library_catalog(library_dir: str | Path) -> pd.DataFrame:
    rows = [classify_score(path) for path in iter_musicxml_files(library_dir)]
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    return dataframe.sort_values(["composer", "period", "form", "title"]).reset_index(drop=True)


def summarize_catalog(catalog: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if catalog.empty:
        return {"catalog": catalog}
    summary = {
        "catalog": catalog,
        "by_composer": catalog.groupby(["composer", "period"]).size().reset_index(name="count").sort_values("count", ascending=False),
        "by_form": catalog.groupby("form").size().reset_index(name="count").sort_values("count", ascending=False),
        "by_difficulty": catalog.groupby("difficulty_bucket").size().reset_index(name="count").sort_values("count", ascending=False),
        "by_arrangement": catalog.groupby("arrangement_tag").size().reset_index(name="count").sort_values("count", ascending=False),
    }
    if "source_name" in catalog.columns:
        summary["by_source"] = catalog.groupby(["source_name", "genre_family"]).size().reset_index(name="count").sort_values("count", ascending=False)
    return summary

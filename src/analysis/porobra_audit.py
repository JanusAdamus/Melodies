"""Auditoría de sólo lectura de una corrida de ``src.cli.porobra_heldout`` (tareas 1.5 y 2.5).

Las auditorías de ``Comparacion/`` entienden los archivos del runner PDMX; esta revisa los
de la corrida por obra: archivos presentes, hashes del manifiesto, denominadores de la
partición 80/20, conteo del corpus, unicidad de obras y huellas de contenido, recálculo de
las cifras titulares desde ``per_work.csv`` y obras que fallaron en el análisis. Eso es la
integridad: ``status`` es ``passed``, ``incomplete`` o ``failed``.

El recálculo repite desde ``per_work.csv``, con la semilla y los remuestreos del resumen,
todo lo que decide (IC de Holm por pasos, rechazo, veredicto, IC del 95 %, ponderada por
tokens, Wilcoxon con Holm, sensibilidades y convergencia) y exige igualdad exacta.

Aparte, ``preregistration`` dice si la configuración del manifiesto es la de la sección A
del preregistro (``preregistered``) o no (``not_preregistered``, con ``deviations``: qué
cambió), incluida la regla de ampliar la rejilla del finito; ``manifest["config"]`` se
compara además con la que sale del argv real (``manifest["command"]``) y con lo que dice el
resumen. Un smoke íntegro sale ``passed`` y ``not_preregistered``. ``confirmatory`` sólo es
verdadero si la corrida es íntegra, preregistrada y la principal (truncamiento 20): la de
truncamiento 40 es la sensibilidad (``run_kind``).

    python -m src.analysis.porobra_audit <directorio de la corrida> [--require-preregistered]

imprime el JSON y sale con 0 si la corrida es íntegra (con ``--require-preregistered``, si
además es la corrida confirmatoria). El CLI escribe el mismo resultado en
``porobra_audit.json`` al terminar; esta función nunca escribe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.analysis import porobra

AUDIT_FILENAME = "porobra_audit.json"
REQUIRED_ARTIFACTS = ("per_work.csv", "summary.json", "manifest.json")
TOLERANCE = 1e-9
_STATUS_ORDER = {"passed": 0, "incomplete": 1, "failed": 2}

#: Sección A del preregistro (más Enmiendas 3 y 5), tal como la escribe ``manifest["config"]``.
#: Las claves de implementación que el preregistro no fija (techo de la rejilla, inicios de
#: Baum-Welch, trayectorias FFBS, semillas, hiperparámetros del HDP fuera de η) quedan con
#: los valores registrados en la bitácora: cambiarlas cambia la corrida.
PREREGISTERED_CONFIG = {
    "max_states": 512,
    "finite_seed": 0,
    "finite_restarts": 5,
    "n_occupancy_samples": 200,
    "hdp_hyperparameters": {"alpha": 8.0, "alpha0": 4.0, "gamma": 2.0, "kappa": 0.0},
    "stats_seed": 0,
    "alphas": [0.1, 0.5, 1.0],
    "k_grid": [2, 4, 8, 16, 32],
    "finite_alphas": [0.1, 0.5, 1.0],
    "etas": [1.0, 3.0, 6.0],
    "n_chains": 4,
    "chain_seeds": [1, 2, 3, 4],
    "n_iters": 400,
    "burn_in": 200,
    "occupancy_threshold": 0.02,
    "rhat_threshold": 1.1,
    "min_events": 20,
    "extraction": "melody",
    "musetrainer_limit": None,
    "symbtr_n": 100,
    "symbtr_seed": 7,
    "bootstrap_samples": 10_000,
}
#: Claves del manifiesto que no cambian los resultados. Cualquier otra clave que no esté en
#: PREREGISTERED_CONFIG (ni sea «truncation») es una desviación: un flag nuevo no pasa solo.
UNCHECKED_CONFIG_KEYS = ("n_workers",)
#: Truncamiento del HDP: 20 es la corrida principal; 40, la sensibilidad preregistrada.
PREREGISTERED_TRUNCATION = {20: "principal", 40: "sensibilidad_truncamiento_40"}
RUN_KIND_NOTES = {
    "principal": "corrida confirmatoria de la sección A (truncamiento 20)",
    "sensibilidad_truncamiento_40": (
        "sensibilidad preregistrada del truncamiento (40); no es la corrida confirmatoria y no decide"
    ),
}
#: Candidatos por fuente: MuseTrainer completo (69 partituras) y la muestra de 100 de SymbTr.
PREREGISTERED_CORPUS = {"MuseTrainer": 69, "SymbTr": 100}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(ok: bool) -> str:
    return "passed" if ok else "failed"


def _manifest_hashes(root: Path, manifest: dict) -> dict:
    outputs = manifest.get("outputs") or {}
    mismatched = sorted(name for name, digest in outputs.items() if not (root / name).exists() or _sha256(root / name) != digest)
    missing_fields = [key for key in ("commit", "config", "inputs", "versions") if not manifest.get(key)]
    unhashed_inputs = [item.get("path") for item in (manifest.get("inputs") or {}).get("files", []) if len(str(item.get("sha256", ""))) != 64]
    ok = bool(outputs) and not mismatched and not missing_fields and not unhashed_inputs
    return {"name": "manifest_hashes", "status": _status(ok), "mismatched": mismatched,
            "missing_fields": missing_fields, "unhashed_inputs": unhashed_inputs}


def _denominators(works: pd.DataFrame, summary: dict) -> dict:
    min_events = int(summary.get("corpus", {}).get("min_events", porobra.MIN_EVENTS))
    bad = []
    for row in works.itertuples():
        expected = porobra.contiguous_split(int(row.n_events), min_events=1)
        observed = (row.n_fit, row.n_test, row.n_inner_train, row.n_validation)
        nll = [float(getattr(row, f"{model}_test_nll")) for model in porobra.MODELS]
        if (row.n_events < min_events or observed != (expected.n_fit, expected.n_test, expected.n_inner_train, expected.n_validation)
                or not all(math.isfinite(value) and value > 0 for value in nll)):
            bad.append(str(row.work_id))
    return {"name": "denominators", "status": _status(not bad), "bad_work_ids": bad, "min_events": min_events}


def _corpus_accounting(works: pd.DataFrame, summary: dict) -> dict:
    corpus = summary.get("corpus", {})
    parts = ("n_failures", "n_duplicates", "n_excluded_short", "n_analysis_failures", "n_analyzed")
    accounted = sum(int(corpus.get(key, 0)) for key in parts)
    by_source = sum(int(value) for value in corpus.get("n_by_source_analyzed", {}).values())
    ok = corpus.get("n_candidates") == accounted and corpus.get("n_analyzed") == len(works) == by_source
    return {"name": "corpus_accounting", "status": _status(ok), "n_candidates": corpus.get("n_candidates"),
            "accounted": accounted, "n_rows": len(works)}


def _canonicalization(works: pd.DataFrame) -> dict:
    duplicated_ids = sorted(works.loc[works["work_id"].duplicated(keep=False), "work_id"].astype(str).unique())
    fingerprints = works["content_fingerprint"].fillna("").astype(str)
    duplicated = sorted(works.loc[fingerprints.duplicated(keep=False) & (fingerprints != ""), "work_id"].astype(str))
    empty = sorted(works.loc[fingerprints == "", "work_id"].astype(str))
    ok = not duplicated_ids and not duplicated and not empty
    return {"name": "canonicalization", "status": _status(ok), "duplicated_work_ids": duplicated_ids,
            "duplicated_fingerprint_work_ids": duplicated, "missing_fingerprint_work_ids": empty}


def read_per_work(path: str | Path) -> pd.DataFrame:
    """Lee ``per_work.csv`` con los floats exactos que escribió el CLI."""

    return pd.read_csv(path, float_precision="round_trip")


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "1.0")


def comparison_records(works: pd.DataFrame) -> list[dict]:
    """Filas por obra con lo que usan las comparaciones y la convergencia."""

    records = []
    for row in works.to_dict("records"):
        record = {"work_id": str(row["work_id"]), "n_test": int(row["n_test"]), "hdp_converged": _as_bool(row["hdp_converged"])}
        for model in porobra.MODELS:
            value = row.get(f"{model}_test_nll")
            record[f"{model}_test_nll"] = float(value) if value is not None else math.nan
        records.append(record)
    return records


def _jsonable(value):
    """Lo mismo que el CLI hace antes de escribir el resumen (NaN e infinitos a None)."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


def _differences(observed, expected, path: str = "") -> list[str]:
    """Rutas donde dos estructuras JSON difieren (igualdad exacta, sin tolerancia)."""

    if isinstance(expected, dict) and isinstance(observed, dict):
        found = []
        for key in sorted(set(expected) | set(observed)):
            where = f"{path}.{key}" if path else str(key)
            if key not in observed:
                found.append(f"{where}: falta en el resumen")
            elif key not in expected:
                found.append(f"{where}: sobra en el resumen")
            else:
                found += _differences(observed[key], expected[key], where)
        return found
    if isinstance(expected, list) and isinstance(observed, list) and len(expected) == len(observed):
        return [item for index, (a, b) in enumerate(zip(observed, expected)) for item in _differences(a, b, f"{path}[{index}]")]
    if type(observed) is not type(expected) or observed != expected:
        return [f"{path}: {observed!r} en el resumen, {expected!r} al recalcular"]
    return []


def _recompute(works: pd.DataFrame, summary: dict) -> dict:
    """Recalcula desde ``per_work.csv`` todo lo que decide o se reporta con la regla.

    Comparación principal, las dos sensibilidades y la convergencia, con los remuestreos y la
    semilla del resumen: IC bootstrap por obra con Holm por pasos, rechazo, veredicto,
    bootstrap_95_ci, ponderada por tokens, Wilcoxon con Holm, fraction_converged y
    not_converged_work_ids. Deben coincidir exacto: un resumen editado no pasa. La lista de
    obras expuestas del resumen debe ser la de ``porobra.TEST_EXPOSED_WORK_IDS`` y
    ``exposed_in_run``, su intersección con las obras de la corrida.
    """

    blocks = ("convergence", "comparisons", "sensitivity_converged_only", "sensitivity_without_exposed")
    mismatches = [f"{key}: falta en el resumen" for key in blocks if not isinstance(summary.get(key), dict)]
    exposure = summary.get("test_exposure") or {}
    exposed = list(exposure.get("exposed_work_ids") or [])
    if exposed != list(porobra.TEST_EXPOSED_WORK_IDS):
        mismatches.append(f"test_exposure.exposed_work_ids: {exposed!r} contra {list(porobra.TEST_EXPOSED_WORK_IDS)!r}")
    in_run = sorted(set(exposed) & set(works["work_id"].astype(str)))
    if exposure.get("exposed_in_run") != in_run:
        mismatches.append(f"test_exposure.exposed_in_run: {exposure.get('exposed_in_run')!r} contra {in_run!r}")
    if not mismatches:
        main = summary["comparisons"]
        try:
            expected = _jsonable(porobra.summary_blocks(
                comparison_records(works), bootstrap_samples=int(main["bootstrap_samples"]), seed=int(main["seed"]),
                exposed_work_ids=exposed))
        except (KeyError, TypeError, ValueError) as error:
            mismatches.append(f"no se pudo recalcular: {type(error).__name__}: {error}")
        else:
            for key in blocks:
                mismatches += _differences(summary[key], expected[key], key)
    return {"name": "headline_recompute", "status": _status(not mismatches), "recomputed_blocks": list(blocks),
            "mismatches": mismatches}


def _same(observed, expected) -> bool:
    if isinstance(expected, list):
        return isinstance(observed, (list, tuple)) and len(observed) == len(expected) and all(
            _same(a, b) for a, b in zip(observed, expected))
    if isinstance(expected, float):
        return isinstance(observed, (int, float)) and not isinstance(observed, bool) and abs(observed - expected) <= TOLERANCE
    return observed == expected


def _same_dict(observed, expected: dict) -> bool:
    return isinstance(observed, dict) and set(observed) == set(expected) and all(
        _same(observed[key], value) for key, value in expected.items())


def _hit_limit(works: pd.DataFrame) -> list[str]:
    if "finite_k_hit_limit" not in works:
        return []
    hit = works["finite_k_hit_limit"].astype(str).str.lower().isin(("true", "1"))
    return sorted(works.loc[hit, "work_id"].astype(str))


def _command_config(manifest: dict) -> dict:
    """Vuelve a derivar la configuración del argv real guardado en ``manifest["command"]``."""

    from src.cli import porobra_heldout  # el CLI importa este módulo: import diferido

    command = manifest.get("command")
    if not isinstance(command, list) or command[1:3] != ["-m", "src.cli.porobra_heldout"]:
        raise ValueError(f"no es un comando de src.cli.porobra_heldout: {command!r}")
    parser = porobra_heldout.build_parser()
    parser.exit_on_error = False
    try:
        args = parser.parse_args([str(item) for item in command[3:]])
    except (argparse.ArgumentError, SystemExit) as error:
        raise ValueError(f"argv ilegible: {error}") from None
    return porobra_heldout.manifest_config(args)


def _command_deviation(manifest: dict, config: dict) -> dict | None:
    try:
        derived = _jsonable(_command_config(manifest))
    except ValueError as error:
        return {"observed": str(error), "expected": "argv de src.cli.porobra_heldout que reproduzca manifest['config']"}
    keys = sorted(key for key in set(derived) | set(config) if not _same(_listify(config.get(key)), _listify(derived.get(key))))
    if not keys:
        return None
    return {"observed": {key: derived.get(key) for key in keys}, "expected": {key: config.get(key) for key in keys}}


def _listify(value):
    if isinstance(value, tuple):
        return [_listify(item) for item in value]
    if isinstance(value, list):
        return [_listify(item) for item in value]
    if isinstance(value, dict):
        return {key: _listify(item) for key, item in value.items()}
    return value


def _summary_deviation(summary: dict, config: dict) -> dict | None:
    """Lo que el resumen dice de la configuración debe ser lo del manifiesto."""

    observed, expected = {}, {}
    for key in ("comparisons", "sensitivity_converged_only", "sensitivity_without_exposed"):
        block = summary.get(key) or {}
        for field, config_key in (("bootstrap_samples", "bootstrap_samples"), ("seed", "stats_seed")):
            if block.get(field) != config.get(config_key):
                observed[f"{key}.{field}"] = block.get(field)
                expected[f"{key}.{field}"] = config.get(config_key)
    corpus = summary.get("corpus") or {}
    if corpus.get("min_events") != config.get("min_events"):
        observed["corpus.min_events"], expected["corpus.min_events"] = corpus.get("min_events"), config.get("min_events")
    if corpus.get("extraction") != config.get("extraction"):
        observed["corpus.extraction"], expected["corpus.extraction"] = corpus.get("extraction"), config.get("extraction")
    rhat = (summary.get("convergence") or {}).get("rhat_threshold")
    if not _same(rhat, config.get("rhat_threshold")):
        observed["convergence.rhat_threshold"], expected["convergence.rhat_threshold"] = rhat, config.get("rhat_threshold")
    return {"observed": observed, "expected": expected} if observed else None


def _preregistration(manifest: dict, summary: dict, works: pd.DataFrame) -> dict:
    config = manifest.get("config") or {}
    deviations = {}
    command = _command_deviation(manifest, config)
    if command:
        deviations["command"] = command
    in_summary = _summary_deviation(summary, config)
    if in_summary:
        deviations["summary"] = in_summary
    for key, expected in PREREGISTERED_CONFIG.items():
        observed = config.get(key)
        if not (_same_dict(observed, expected) if isinstance(expected, dict) else _same(observed, expected)):
            deviations[key] = {"observed": observed, "expected": expected}
    known = set(PREREGISTERED_CONFIG) | set(UNCHECKED_CONFIG_KEYS) | {"truncation"}
    for key in sorted(set(config) - known):
        deviations[key] = {"observed": config[key], "expected": "clave no prevista por la sección A"}
    # Sección A: «si se elige el máximo, se amplía la rejilla antes de reportar».
    hit_limit = _hit_limit(works)
    if hit_limit:
        deviations["finite_k_hit_limit"] = {"observed": hit_limit, "expected": []}
    truncation = config.get("truncation")
    if truncation not in PREREGISTERED_TRUNCATION:
        deviations["truncation"] = {"observed": truncation, "expected": sorted(PREREGISTERED_TRUNCATION)}
    corpus = dict(summary.get("corpus", {}).get("n_by_source_candidates", {}))
    if corpus != PREREGISTERED_CORPUS:
        deviations["corpus_size"] = {"observed": corpus, "expected": dict(PREREGISTERED_CORPUS)}
    if manifest.get("dirty_worktree") is not False:
        deviations["dirty_worktree"] = {"observed": manifest.get("dirty_worktree"), "expected": False}
    run_kind = PREREGISTERED_TRUNCATION.get(truncation)
    return {"status": "not_preregistered" if deviations else "preregistered",
            "run_kind": run_kind, "run_kind_note": RUN_KIND_NOTES.get(run_kind), "deviations": deviations}


def audit_run_directory(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).exists()]
    checks = [{"name": "artifacts_present", "status": "incomplete" if missing else "passed", "missing": missing}]
    preregistration = {"status": "unknown", "run_kind": None, "run_kind_note": None, "deviations": {}}
    if not missing:
        works = read_per_work(root / "per_work.csv")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        failures = [str(item.get("work_id")) for item in summary.get("analysis_failures", [])]
        checks += [
            _manifest_hashes(root, manifest),
            _denominators(works, summary),
            _corpus_accounting(works, summary),
            _canonicalization(works),
            _recompute(works, summary),
            {"name": "analysis_failures", "status": _status(not failures), "work_ids": failures},
        ]
        preregistration = _preregistration(manifest, summary, works)
    status = max((check["status"] for check in checks), key=_STATUS_ORDER.__getitem__)
    return {
        "status": status,
        "preregistration": preregistration,
        # Sólo la corrida principal (truncamiento 20) es la confirmatoria; la de 40 es la
        # sensibilidad preregistrada y --require-preregistered no la acepta.
        "confirmatory": (status == "passed" and preregistration["status"] == "preregistered"
                         and preregistration["run_kind"] == "principal"),
        "run_dir": root.as_posix(),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audita una corrida de src.cli.porobra_heldout (sólo lectura).")
    parser.add_argument("run_dir")
    parser.add_argument("--require-preregistered", action="store_true",
                        help="Sale con 0 sólo si además es la corrida confirmatoria de la sección A "
                             "(truncamiento 20; la de 40 es la sensibilidad).")
    args = parser.parse_args(argv)
    audit = audit_run_directory(args.run_dir)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if (audit["confirmatory"] if args.require_preregistered else audit["status"] == "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

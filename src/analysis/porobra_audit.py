"""Auditoría de sólo lectura de una corrida de ``src.cli.porobra_heldout`` (tareas 1.5 y 2.5).

Las auditorías de ``Comparacion/`` entienden los archivos del runner PDMX; esta revisa los
de la corrida por obra: archivos presentes, hashes del manifiesto, denominadores de la
partición 80/20, conteo del corpus, unicidad de obras y huellas de contenido, recálculo de
las cifras titulares desde ``per_work.csv`` y obras que fallaron en el análisis.

    python -m src.analysis.porobra_audit <directorio de la corrida>

imprime el JSON y sale con 0 sólo si el estado es ``passed``. El CLI escribe el mismo
resultado en ``porobra_audit.json`` al terminar; esta función nunca escribe.
"""

from __future__ import annotations

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


def _recompute(works: pd.DataFrame, summary: dict) -> dict:
    exposed = set(summary.get("test_exposure", {}).get("exposed_work_ids", []))
    converged = works["hdp_converged"].astype(str).str.lower().isin(("true", "1"))
    blocks = {
        "comparisons": works,
        "sensitivity_converged_only": works[converged],
        "sensitivity_without_exposed": works[~works["work_id"].isin(exposed)],
    }
    mismatches = []
    for key, subset in blocks.items():
        block = summary.get(key)
        if block is None:
            mismatches.append(f"{key}: falta en el resumen")
            continue
        if block.get("n_works") != len(subset):
            mismatches.append(f"{key}: n_works {block.get('n_works')} contra {len(subset)}")
        for comparison in block.get("comparisons", []):
            other = comparison["model_b"]
            differences = (subset["hdp_test_nll"] - subset[f"{other}_test_nll"]).dropna().to_numpy()
            if comparison.get("status") != "ok":
                if len(differences):
                    mismatches.append(f"{key}/{other}: sin comparar con {len(differences)} pares")
                continue
            if comparison.get("n_pairs") != len(differences) or abs(comparison["mean_difference"] - float(np.mean(differences))) > TOLERANCE:
                mismatches.append(f"{key}/{other}: media {comparison['mean_difference']} contra {float(np.mean(differences))}")
    n_converged = summary.get("convergence", {}).get("n_converged")
    if n_converged != int(converged.sum()):
        mismatches.append(f"convergence: n_converged {n_converged} contra {int(converged.sum())}")
    return {"name": "headline_recompute", "status": _status(not mismatches), "mismatches": mismatches}


def audit_run_directory(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).exists()]
    checks = [{"name": "artifacts_present", "status": "incomplete" if missing else "passed", "missing": missing}]
    if not missing:
        works = pd.read_csv(root / "per_work.csv")
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
    status = max((check["status"] for check in checks), key=_STATUS_ORDER.__getitem__)
    return {"status": status, "run_dir": root.as_posix(), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    (run_dir,) = argv if argv is not None else sys.argv[1:]
    audit = audit_run_directory(run_dir)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

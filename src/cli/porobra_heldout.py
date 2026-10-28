"""Análisis por obra held-out (preregistro 2026-10-26, sección A; tarea 1.4 del handoff).

Ejemplo (corrida principal de la Fase 2):

    python -m src.cli.porobra_heldout --musetrainer-root external/library/scores \
        --symbtr-root external/SymbTr --symbtr-n 100 --symbtr-seed 7 --n-workers 6

Escribe en un directorio NUEVO ``<output-root>/<run-name>_<AAAAMMDD>_<commit7>`` (falla si
existe): ``per_work.csv``, ``summary.json`` y ``manifest.json``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src.analysis import porobra
from src.analysis.library_batch import prepare_corpus
from src.data.library_catalog import iter_musicxml_files
from src.data.multicorpus import parse_symbtr_filename, stratified_symbtr_sample

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTION = "melody"
PACKAGES = ("numpy", "scipy", "pandas", "music21")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def run_directory(output_root: str | Path, run_name: str) -> Path:
    return Path(output_root) / f"{run_name}_{datetime.now():%Y%m%d}_{_git('rev-parse', 'HEAD')[:7]}"


def symbtr_paths(root: str | Path, *, n: int, seed: int) -> list[Path]:
    root = Path(root)
    folder = root / "MusicXML" if (root / "MusicXML").is_dir() else root
    return stratified_symbtr_sample(iter_musicxml_files(folder), n, seed=seed)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Análisis por obra held-out: HDP-HMM contra finito, bigrama y unigrama.")
    parser.add_argument("--musetrainer-root", help="Carpeta de MuseTrainer (se usa completa).")
    parser.add_argument("--musetrainer-limit", type=int, default=None, help="Sólo las primeras N (orden POSIX); para smoke.")
    parser.add_argument("--symbtr-root", help="Raíz de SymbTr o su carpeta MusicXML.")
    parser.add_argument("--symbtr-n", type=int, default=100, help="Tamaño de la muestra estratificada por makam.")
    parser.add_argument("--symbtr-seed", type=int, default=7)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "outputs"))
    parser.add_argument("--run-name", default="porobra_heldout")
    parser.add_argument("--n-workers", type=int, default=1, help="Obras en paralelo (procesos).")
    parser.add_argument("--min-events", type=int, default=porobra.MIN_EVENTS)
    parser.add_argument("--alphas", type=_floats, default=(0.1, 0.5, 1.0))
    parser.add_argument("--k-grid", type=_ints, default=(2, 4, 8, 16, 32))
    parser.add_argument("--finite-restarts", type=int, default=5, help="Inicios de Baum-Welch por K (gana el de mayor log-verosimilitud).")
    parser.add_argument("--max-states", type=int, default=512, help="Techo de la ampliación de la rejilla del finito.")
    parser.add_argument("--etas", type=_floats, default=(1.0, 3.0, 6.0))
    parser.add_argument("--truncation", type=int, default=20, help="K de truncamiento del HDP (40: sensibilidad).")
    parser.add_argument("--chains", type=int, default=4, help="Cadenas con semillas 1..n.")
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--occupancy-threshold", type=float, default=0.02)
    parser.add_argument("--occupancy-samples", type=int, default=200, help="Trayectorias FFBS del finito.")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--stats-seed", type=int, default=0)
    return parser


def _catalog(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    if args.musetrainer_root:
        root = Path(args.musetrainer_root)
        paths = iter_musicxml_files(root)
        if args.musetrainer_limit is not None:
            paths = sorted(paths, key=lambda p: p.relative_to(root).as_posix())[: args.musetrainer_limit]
        rows += [{"path": str(p), "source": "MuseTrainer", "relative_path": p.relative_to(root).as_posix(), "makam": "", "error": ""} for p in paths]
    if args.symbtr_root:
        for path in symbtr_paths(args.symbtr_root, n=args.symbtr_n, seed=args.symbtr_seed):
            rows.append({"path": str(path), "source": "SymbTr", "relative_path": path.name, "makam": parse_symbtr_filename(path)["makam"], "error": ""})
    if not rows:
        raise SystemExit("Indica --musetrainer-root y/o --symbtr-root.")
    return pd.DataFrame(rows)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _analyze(item: tuple[str, list[int], porobra.PorObraConfig]) -> dict:
    return porobra.analyze_work(*item)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _distribution(values: pd.Series) -> dict:
    return {str(key): int(count) for key, count in values.value_counts().sort_index().items()}


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    run_dir = run_directory(args.output_root, args.run_name)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} ya existe: nunca se sobrescribe una corrida.")
    config = porobra.PorObraConfig(
        alphas=tuple(args.alphas),
        k_grid=tuple(args.k_grid),
        max_states=args.max_states,
        finite_restarts=args.finite_restarts,
        etas=tuple(args.etas),
        truncation=args.truncation,
        n_chains=args.chains,
        n_iters=args.iters,
        burn_in=args.burn_in,
        occupancy_threshold=args.occupancy_threshold,
        n_occupancy_samples=args.occupancy_samples,
    )

    catalog = _catalog(args)
    prepared = prepare_corpus(catalog, extraction=EXTRACTION)
    pieces = prepared.pieces
    short = pieces[pieces["n_events"] < args.min_events]
    kept = pieces[pieces["n_events"] >= args.min_events].copy()
    kept["work_id"] = kept["source"] + ":" + kept["relative_path"]
    kept = kept.sort_values("work_id").reset_index(drop=True)
    jobs = [
        (row.work_id, [int(event.representative_pitch_class) for event in prepared.events[row.path]], config)
        for row in kept.itertuples()
    ]
    if args.n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            results = list(pool.map(_analyze, jobs))
    else:
        results = [_analyze(job) for job in jobs]

    per_work = pd.DataFrame(results)
    per_work = kept[["work_id", "source", "relative_path", "makam", "content_fingerprint"]].merge(per_work, on="work_id")
    for column in per_work.columns:
        if per_work[column].map(lambda value: isinstance(value, (dict, list))).any():
            per_work[column] = per_work[column].map(lambda value: json.dumps(_jsonable(value), sort_keys=True))
    records = [dict(zip(("work_id", "n_test", *(f"{m}_test_nll" for m in porobra.MODELS), "hdp_converged"), values))
               for values in zip(per_work["work_id"], per_work["n_test"], *(per_work[f"{m}_test_nll"] for m in porobra.MODELS),
                                 per_work["hdp_converged"])]

    summary = {
        "preregistro": "docs/preregistro-2026-10-26.md, sección A, Aclaración 2, Enmienda 3",
        "corpus": {
            "extraction": EXTRACTION,
            "n_candidates": int(len(catalog)),
            "n_by_source_candidates": _distribution(catalog["source"]),
            "n_failures": int(len(prepared.failures)),
            "failures": prepared.failures[["path", "error_stage", "error"]].to_dict("records") if len(prepared.failures) else [],
            "n_duplicates": int(len(prepared.duplicates)),
            "duplicates": prepared.duplicates[["path", "duplicate_of"]].to_dict("records") if len(prepared.duplicates) else [],
            "min_events": int(args.min_events),
            "n_excluded_short": int(len(short)),
            "excluded_short_works": short[["path", "n_events"]].to_dict("records"),
            "n_analyzed": int(len(per_work)),
            "n_by_source_analyzed": _distribution(per_work["source"]) if len(per_work) else {},
        },
        "convergence": porobra.convergence_summary(records),
        "comparisons": porobra.compare_hdp(records, bootstrap_samples=args.bootstrap_samples, seed=args.stats_seed),
        "selected_hyperparameters": {
            "hdp_eta": _distribution(per_work["hdp_eta"]) if len(per_work) else {},
            "finite_k": _distribution(per_work["finite_k"]) if len(per_work) else {},
            "finite_k_hit_limit": per_work.loc[per_work["finite_k_hit_limit"], "work_id"].tolist() if len(per_work) else [],
            "unigram_alpha": _distribution(per_work["unigram_alpha"]) if len(per_work) else {},
            "bigram_alpha": _distribution(per_work["bigram_alpha"]) if len(per_work) else {},
        },
        "wall_clock_s": None,
    }

    run_dir.mkdir(parents=True)
    per_work.to_csv(run_dir / "per_work.csv", index=False)
    summary["wall_clock_s"] = round(time.perf_counter() - started, 1)
    (run_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    input_files = [{"path": Path(p).as_posix(), "sha256": _sha256(p)} for p in catalog["path"]]
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_worktree": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "command": [sys.executable, "-m", "src.cli.porobra_heldout", *(argv if argv is not None else sys.argv[1:])],
        "config": {
            **config.to_dict(),
            "chain_seeds": list(range(1, config.n_chains + 1)),
            "extraction": EXTRACTION,
            "min_events": args.min_events,
            "symbtr_n": args.symbtr_n if args.symbtr_root else None,
            "symbtr_seed": args.symbtr_seed if args.symbtr_root else None,
            "musetrainer_limit": args.musetrainer_limit,
            "bootstrap_samples": args.bootstrap_samples,
            "stats_seed": args.stats_seed,
            "n_workers": args.n_workers,
        },
        "inputs": {
            "musetrainer_root": args.musetrainer_root,
            "symbtr_root": args.symbtr_root,
            "aggregate_sha256": hashlib.sha256("".join(f"{i['path']}\n{i['sha256']}\n" for i in input_files).encode()).hexdigest(),
            "files": input_files,
        },
        "versions": _versions(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "outputs": {name: _sha256(run_dir / name) for name in ("per_work.csv", "summary.json")},
    }
    (run_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Corrida escrita en {run_dir} ({summary['wall_clock_s']} s, {len(per_work)} obras).")
    if summary["convergence"]["exceeds_10_percent"]:
        print("AVISO: " + summary["convergence"]["action_required"])
    return run_dir


if __name__ == "__main__":
    main()

"""Análisis por obra held-out (preregistro 2026-10-26, sección A; tarea 1.4 del handoff).

Ejemplo (corrida principal de la Fase 2):

    python -m src.cli.porobra_heldout --musetrainer-root external/library/scores \
        --symbtr-root external/SymbTr --symbtr-n 100 --symbtr-seed 7 --n-workers 6

Escribe en un directorio NUEVO ``<output-root>/<run-name>_<AAAAMMDD>_<commit7>`` (falla si
existe): ``per_work.csv``, ``summary.json`` y ``manifest.json``. El directorio se crea al
empezar y cada obra terminada se agrega a ``per_work.partial.jsonl``: si la corrida se
interrumpe, lo ya calculado queda en disco. Una obra que falla se registra en
``analysis_failures`` del resumen y queda fuera de las comparaciones; las demás siguen.
La comparación principal usa todas las obras analizadas (Enmienda 6);
``sensitivity_converged_only`` la repite sólo con las que tienen R-hat de log L < 1.1 y
``sensitivity_without_exposed`` sin las obras con exposición a cifras de prueba (análisis
adicional de la Aclaración 8; no decide); las dos dicen si algún veredicto cambia.
``README.txt`` lleva la advertencia de la Enmienda 6 sobre los estados del HDP. Al final
escribe ``porobra_audit.json`` (``src.analysis.porobra_audit``): integridad y, aparte, si es
la configuración de la sección A.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from src.analysis import porobra, porobra_audit
from src.analysis.library_batch import prepare_corpus
from src.data.library_catalog import iter_musicxml_files
from src.data.multicorpus import parse_symbtr_filename, stratified_symbtr_sample

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTION = "melody"
#: Archivos de la corrida con hash en ``manifest["outputs"]``.
OUTPUTS = ("per_work.csv", "summary.json", "README.txt")
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


def _min_events(text: str) -> int:
    value = int(text)
    if value < porobra.MIN_EVENTS:
        raise argparse.ArgumentTypeError(
            f"la partición del preregistro exige al menos {porobra.MIN_EVENTS} eventos por obra"
        )
    return value


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
    parser.add_argument("--min-events", type=_min_events, default=porobra.MIN_EVENTS, help="Mínimo 20 (preregistro).")
    parser.add_argument("--alphas", type=_floats, default=(0.1, 0.5, 1.0))
    parser.add_argument("--k-grid", type=_ints, default=(2, 4, 8, 16, 32))
    parser.add_argument("--finite-alphas", type=_floats, default=(0.1, 0.5, 1.0), help="Seudoconteo de Baum-Welch (Enmienda 5).")
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
    """Candidatos por fuente, cada fuente en orden de su ruta relativa POSIX como cadena.

    ``sorted(Path)`` no distingue mayúsculas en Windows y sí en Linux; la cadena POSIX da el
    mismo orden en los dos (y con él, el mismo manifiesto y las mismas listas del resumen).
    """

    rows = []
    if args.musetrainer_root:
        root = Path(args.musetrainer_root)
        paths = sorted(iter_musicxml_files(root), key=lambda p: p.relative_to(root).as_posix())
        if args.musetrainer_limit is not None:
            paths = paths[: args.musetrainer_limit]
        rows += [{"path": str(p), "source": "MuseTrainer", "relative_path": p.relative_to(root).as_posix(), "makam": "", "error": ""} for p in paths]
    if args.symbtr_root:
        sample = sorted(symbtr_paths(args.symbtr_root, n=args.symbtr_n, seed=args.symbtr_seed), key=lambda p: p.name)
        for path in sample:
            rows.append({"path": str(path), "source": "SymbTr", "relative_path": path.name, "makam": parse_symbtr_filename(path)["makam"], "error": ""})
    if not rows:
        raise SystemExit("Indica --musetrainer-root y/o --symbtr-root.")
    return pd.DataFrame(rows)


def manifest_config(args: argparse.Namespace) -> dict:
    """La configuración que escribe ``manifest["config"]``, derivada de los argumentos.

    La auditoría la vuelve a derivar de ``manifest["command"]`` para comprobar que coincide.
    """

    config = _config(args)
    return {
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
    }


def _config(args: argparse.Namespace) -> porobra.PorObraConfig:
    return porobra.PorObraConfig(
        alphas=tuple(args.alphas),
        k_grid=tuple(args.k_grid),
        finite_alphas=tuple(args.finite_alphas),
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


def input_files(catalog: pd.DataFrame) -> list[dict]:
    """Entradas con su ruta relativa a la raíz de su corpus (como Comparacion/corpus_fingerprint.py).

    Sin la ruta local, el mismo corpus en otra carpeta u otra máquina da los mismos registros
    y el mismo ``aggregate_sha256``.
    """

    files = [{"source": row.source, "path": row.relative_path, "sha256": _sha256(row.path)} for row in catalog.itertuples()]
    return sorted(files, key=lambda item: (item["source"], item["path"]))


def aggregate_sha256(files: list[dict]) -> str:
    return hashlib.sha256("".join(f"{i['source']}\n{i['path']}\n{i['sha256']}\n" for i in files).encode("utf-8")).hexdigest()


def _readme() -> str:
    return (
        "Corrida de src.cli.porobra_heldout (preregistro 2026-10-26, sección A).\n\n"
        "Archivos: per_work.csv (una fila por obra), summary.json (comparaciones, convergencia, "
        "sensibilidades), manifest.json (commit, comando, configuración, hashes) y porobra_audit.json.\n\n"
        "Advertencia sobre los estados del HDP-HMM en per_work.csv:\n"
        f"{porobra.HDP_STATES_NOTE}\n"
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _analyze(item: tuple[str, list[int], porobra.PorObraConfig]) -> dict:
    """Analiza una obra; si falla, devuelve el error en lugar de propagarlo."""

    try:
        return porobra.analyze_work(*item)
    except Exception as error:  # noqa: BLE001 - una obra no debe tumbar la corrida
        return {"work_id": item[0], "error": f"{type(error).__name__}: {error}"}


def _run_jobs(jobs: list, n_workers: int, partial_path: Path) -> list[dict]:
    """Corre las obras y agrega cada resultado a ``partial_path`` en cuanto termina."""

    results = []

    def keep(result: dict) -> None:
        results.append(result)
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(result), ensure_ascii=False) + "\n")

    if n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_analyze, job): job[0] for job in jobs}
            for future in as_completed(futures):
                try:
                    keep(future.result())
                except Exception as error:  # noqa: BLE001 - p. ej., un proceso que muere
                    keep({"work_id": futures[future], "error": f"{type(error).__name__}: {error}"})
    else:
        for job in jobs:
            keep(_analyze(job))
    return sorted(results, key=lambda result: str(result["work_id"]))


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
    config = _config(args)

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
    run_dir.mkdir(parents=True)
    results = _run_jobs(jobs, args.n_workers, run_dir / "per_work.partial.jsonl")
    analysis_failures = [result for result in results if "error" in result]
    columns = ["work_id", "source", "relative_path", "makam", "content_fingerprint"]
    per_work = pd.DataFrame([result for result in results if "error" not in result])
    if per_work.empty:
        raise SystemExit(f"Ninguna obra se pudo analizar; detalles en {run_dir / 'per_work.partial.jsonl'}.")
    per_work = kept[columns].merge(per_work, on="work_id")
    for column in per_work.columns:
        if per_work[column].map(lambda value: isinstance(value, (dict, list))).any():
            per_work[column] = per_work[column].map(lambda value: json.dumps(_jsonable(value), sort_keys=True))
    per_work.to_csv(run_dir / "per_work.csv", index=False)
    (run_dir / "README.txt").write_text(_readme(), encoding="utf-8")
    # Las comparaciones salen del CSV tal como quedó en disco: la auditoría las recalcula
    # desde el mismo archivo y, con la misma semilla, deben coincidir exacto.
    records = porobra_audit.comparison_records(porobra_audit.read_per_work(run_dir / "per_work.csv"))
    blocks = porobra.summary_blocks(records, bootstrap_samples=args.bootstrap_samples, seed=args.stats_seed,
                                    exposed_work_ids=porobra.TEST_EXPOSED_WORK_IDS)

    summary = {
        "preregistro": "docs/preregistro-2026-10-26.md, sección A, Aclaración 2, Enmiendas 3, 5 y 6, Aclaraciones 7 y 8",
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
            "n_analysis_failures": len(analysis_failures),
            "n_analyzed": int(len(per_work)),
            "n_by_source_analyzed": _distribution(per_work["source"]) if len(per_work) else {},
        },
        "analysis_failures": analysis_failures,
        "convergence": blocks["convergence"],
        "comparisons": blocks["comparisons"],
        "sensitivity_converged_only": blocks["sensitivity_converged_only"],
        "test_exposure": {
            "exposed_work_ids": list(porobra.TEST_EXPOSED_WORK_IDS),
            "exposed_works": dict(porobra.TEST_EXPOSED_WORKS),
            "exposed_in_run": sorted(set(porobra.TEST_EXPOSED_WORK_IDS) & set(per_work["work_id"])),
            "note": porobra.TEST_EXPOSURE_NOTE,
        },
        "sensitivity_without_exposed": blocks["sensitivity_without_exposed"],
        "hdp_states_note": porobra.HDP_STATES_NOTE,
        "selected_hyperparameters": {
            "hdp_eta": _distribution(per_work["hdp_eta"]) if len(per_work) else {},
            "finite_k": _distribution(per_work["finite_k"]) if len(per_work) else {},
            "finite_alpha": _distribution(per_work["finite_alpha"]) if len(per_work) else {},
            "finite_k_hit_limit": per_work.loc[per_work["finite_k_hit_limit"], "work_id"].tolist() if len(per_work) else [],
            "unigram_alpha": _distribution(per_work["unigram_alpha"]) if len(per_work) else {},
            "bigram_alpha": _distribution(per_work["bigram_alpha"]) if len(per_work) else {},
        },
        "wall_clock_s": None,
    }

    summary["wall_clock_s"] = round(time.perf_counter() - started, 1)
    (run_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    files = input_files(catalog)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_worktree": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "command": [sys.executable, "-m", "src.cli.porobra_heldout", *(argv if argv is not None else sys.argv[1:])],
        "config": manifest_config(args),
        "inputs": {
            # Las raíces locales se guardan como referencia; no entran en el hash.
            "musetrainer_root": args.musetrainer_root,
            "symbtr_root": args.symbtr_root,
            "path_base": "ruta relativa a la raíz de cada corpus (source), en POSIX",
            "aggregate_sha256": aggregate_sha256(files),
            "files": files,
        },
        "versions": _versions(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "outputs": {name: _sha256(run_dir / name) for name in OUTPUTS},
    }
    (run_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    audit = porobra_audit.audit_run_directory(run_dir)
    (run_dir / porobra_audit.AUDIT_FILENAME).write_text(json.dumps(_jsonable(audit), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Corrida escrita en {run_dir} ({summary['wall_clock_s']} s, {len(per_work)} obras).")
    print(f"Auditoría de integridad: {audit['status']}; preregistro: {audit['preregistration']['status']} "
          f"({porobra_audit.AUDIT_FILENAME}).")
    deviations = audit["preregistration"]["deviations"]
    if deviations:
        print(f"AVISO: no es la configuración de la sección A del preregistro ({', '.join(sorted(deviations))}); "
              "no se reporta como la corrida confirmatoria.")
    if analysis_failures:
        print(f"AVISO: {len(analysis_failures)} obra(s) fallaron en el análisis; ver analysis_failures en summary.json.")
    convergence = summary["convergence"]
    print(f"R-hat de log L < {porobra.RHAT_THRESHOLD}: {convergence['n_converged']} de {convergence['n_works']} obras "
          "(se reporta; no excluye obras, Enmienda 6).")
    return run_dir


if __name__ == "__main__":
    main()

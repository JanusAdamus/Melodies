# Bitácora del cierre

Una entrada por tarea cerrada (ver PROMPT_CLAUDE.md, «Bitácora»).

## 2026-10-26, laptop Debian

- Auditoría (5 revisores), plan y preregistro: commits 7196f0e y siguientes.
- Mezcladas en `fix/auditoria-tesis`: `fix/auditoria-particiones` (6 commits) y `fix/auditoria-datos` (6 commits). Suite: 240 pruebas en verde con `python -m pytest -q`.
- Subidas sin mezclar y posiblemente incompletas: `fix/auditoria-hdp` (5 commits, último 64f62ec) y `fix/auditoria-comparacion` (8 commits, último 4ae5fe8). Revisar con los criterios de la Fase 1.
- Nuevo hallazgo D1b (PDMX usa la extracción antigua) agregado a la auditoría.
- `visualizaciones/` agregado con los scripts de los tres videos.
- Abierto: todo lo de PROMPT_CLAUDE.md desde la Fase 0.

## 2026-10-26, actualización: `fix/auditoria-comparacion` TERMINADA (último commit 4ae5fe8, ya en origin)

- 8 commits (F3, T1, T2, H7, C7, C5, C2, C1), cada uno con prueba vista en rojo. Suite de la rama: 245 pruebas. Smoke en CPU con 54 piezas (251 s): `protocol_audit`, `artifact_audit`, `denominator_audit`, `finite_hmm_grid_audit` y `vomm_order_grid_audit` en `passed`; diagnóstico HDP `drift_detected` (esperado mientras siga el bug de β y con cadenas de 30 iteraciones).
- Notas para integrarla (Fase 1, tarea 1.2) además de lo que ya dice PROMPT_CLAUDE.md:
  1. `build_cell_chain_diagnostics(..., burn_in=config.hdp_burn_in)` descarta filas con `iteration <= burn_in` y supone que `trace_rows()` numera desde 1 e incluye el burn-in. Si la API nueva del HDP (rama `fix/auditoria-hdp`) ya excluye o renumera el burn-in, pasar `burn_in=0` o se pierden muestras.
  2. Si el HDP promedia la predictiva sobre muestras posteriores, debe renormalizar sobre las 12 alturas paso a paso, igual que `conditional_log_likelihood` (BOS fuera del soporte).
  3. El runner todavía usa `_build_grouped_fixed_splits`: cambiarlo a `Comparacion.splits.build_grouped_fixed_splits` (pedido del equipo de particiones).
  4. `Comparacion/evidence_package.py` debe incluir `headline_estimates.json` y `vomm_order_grid_audit.json`.
  5. `figuras_tesis.costos_ajuste` exige la columna `selected_fit_wall_clock_s`; los títulos fijos de `scripts/figuras_tesis.py` y la documentación del benchmark todavía describen agosto: revisarlos en la Fase 3.
  6. Riesgo: si el HMM llega a K = 384 (`finite_hmm_max_states`) tocando el máximo, la corrida falla tras horas con `ValueError`. Hacer el piloto de la tarea 1.5 antes de la corrida completa.
  7. Artefactos nuevos o cambiados: `headline_estimates.json`, `vomm_order_grid_audit.json`, `hdp_traces.csv` (formato largo, columna `in_burn_in`; reemplaza `hdp_trace_<seed>.csv`), `hdp_chain_diagnostics.json` con `cells`; columnas nuevas en `results_raw.csv`, `engineering_costs.csv`, `results_summary.csv` y `pairwise_comparisons.json`.
- `fix/auditoria-hdp` seguía en curso al cerrar esta entrada (último commit subido: 64f62ec).

## 2026-10-26, cierre en la laptop: `fix/auditoria-hdp` DETENIDA a petición de Adrián

- 6 commits en origin: 988cf41 (Dirichlet sin ceros), 0dc8dc5 (β por CRT y generador por ajuste), 2968a83 (reporte posterior y predictiva held-out), fda3816 (compatibilidad con Comparacion), 64f62ec (alineación de etiquetas, H6), y la evidencia del criterio (b). Sin trabajo sin commit. 219 pruebas en verde en la rama.
- Criterio (b): r = −0.40 (umbral |r| < 0.3 NO cumplido), pendiente −0.05, rango de K medio 5.28–6.08 contra 4.07–11.71 antes. Ver `scripts/auditoria_2026_10_26/hdp/s5_beta_seed_v2_salida_20261026.txt`. Repetir con ≥ 50 semillas y decidir con la pendiente y su IC.
- Sin verificar: criterios (c) (recuperar 3 estados de un HMM simulado) y (h) (dos `fit` con la misma semilla), y revisión adversarial. Lo último que el desarrollador anunció antes de detenerse fue reescribir el muestreador; el commit 0dc8dc5 ya trae una versión CRT: revisar que sea correcta (Fox et al. 2011) antes de mezclar.
- Ningún agente queda trabajando en la laptop.

## 2026-10-26, computadora con GPU: tarea 1.2 (`fix/auditoria-comparacion`) implementada, sin mezclar

- Base traída con `git merge origin/fix/auditoria-tesis` (commit 0c1986e). Suite tras la mezcla: `282 passed, 3 skipped` (`python -m pytest -q`).
- C1, C2, C5, C7, H7, F3, T1, T2: las pruebas de cada commit de la rama (`tests/test_bos_support.py`, `next_token_experiment/tests/test_label_smoothing.py`, `next_token_experiment/tests/test_benchmark_representations.py`, `tests/test_hdp_cell_diagnostics.py`, `tests/test_headline_estimates.py`, `tests/test_pareto_uncertainty.py`, `tests/test_grid_ceiling.py`, `tests/test_cost_scope.py`) fallan o no importan sobre la base previa c8c49a0 y pasan en la rama.
- Puntos (a)-(f) y notas 3 y 4 de esta bitácora: commit 4a64c48, pruebas en `tests/test_integracion_particiones.py` (vistas en rojo). Archivos nuevos por corrida: `splits/manifest.json`, `checkpoint_inputs.json`; `execution_plan.json` lleva `split_sha256` y se escribe también en corridas completas; columna `train_subset_sha256` en `results_raw.csv` y `piece_metrics_raw.csv`.
- Abierto: revisión adversarial (2 subagentes) y merge a `fix/auditoria-tesis`; nota 1 (burn-in del diagnóstico del HDP) depende de cómo quede `fix/auditoria-hdp`; nota 5 (títulos de `scripts/figuras_tesis.py`, p. ej. «media de 6 corridas») queda para la Fase 3.

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

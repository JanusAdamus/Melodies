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

## 2026-10-26, tarea 0 (computadora con GPU, Windows 11)

- Entorno: venv Python 3.12.10, torch 2.11.0+cu128, `cuda_avail True` en RTX 4060 8 GB.
- Suite antes de la Fase 1: 237 pasadas + 3 saltadas en 83.66 s (`python -m pytest -q`).
- PDMX: `C:\Melodies\external\PDMX\mxl`, 254035 archivos. Hash agregado (sha256 de la lista ordenada «ruta relativa POSIX + espacio + tamaño en bytes», unida con `\n`): `5f83b4c45b7f15b4dbae7b31738b6d26203d7c6719e8b247687fc0acbef8eb26`. Salida de `python -I pdmx_hash.py` (os.walk + getsize): `254035 5f83b4c4...eb26`.
- `artifacts/corpus_cache_3000.jsonl` existe (7561926 bytes).
- MuseTrainer y SymbTr NO están en esta computadora: parada pendiente con Adrián (bloquea 1.4/1.5 con MuseTrainer).

## 2026-10-26, tarea 1.1: `fix/auditoria-hdp` NO mezclada (revisión adversarial con hallazgos confirmados)

- Qué se hizo en la rama (último commit 409adfd, en origin): β por CRT (0dc8dc5), Dirichlet sin ceros (988cf41), reporte posterior y predictiva held-out (2968a83, d2ea2b2 renormalizada con BOS fuera), β | z contra la posterior exacta de Fox et al. 2011 (fd80291, d01227e), criterio (b) con IC de Fisher y TOST (3e98620, ff4ad7a, 66c5fda, 8be093c), `s5_beta_seed_v3` con la cadena del preregistro 400/200 (ad9048c, 409adfd).
- Evidencia: en PDMX (obra 200, 50 semillas, 400/200) r = −0.118, IC de Fisher [−0.383, 0.166], TOST p = 0.095. Lo honesto es «sin evidencia de dependencia», no «K no depende del primer β»; la equivalencia sólo se demuestra con datos simulados (r = 0.025, IC 90 % [−0.212, 0.259], por poco).
- Hallazgos confirmados de la revisión (bloquean el merge):
  1. (alta) `tests/test_hdp_sampler.py:124` `test_criterion_b_detects_the_short_chain_dependence` fija un falso positivo: con 80/40 y bloques de 50 semillas, r = 0.318, 0.040, 0.076, −0.364, 0.054 (el signo cambia; Spearman en 0..49 rho = 0.251, p = 0.078). No mide potencia.
  2. (alta) `tests/test_hdp_sampler.py:114` `test_occupied_states_do_not_follow_the_first_beta_draw` depende de las semillas 0..49: con 400/200, semillas 100..149 (r = 0.152, TOST p = 0.142) y 150..199 (r = 0.177, p = 0.184) no dan «equivalencia».
  3. (media) La afirmación «80/40 sí muestra dependencia (r = 0.318, p = 0.025)» está en el docstring de `s5_beta_seed_v3.py`, en `s5_beta_seed_v3_salida_20261026.txt` y en dos comentarios de la prueba; no se reproduce. La razón para descartar 80/40 es que no es la cadena preregistrada.
- Abierto: corregir 1–3 (pruebas robustas a la semilla o con más semillas/obras); las 15 obras de PDMX con 400/200 (unos 750 ajustes largos) no se corrieron; las pruebas de CI de (b) tardan unos 35 s y 10 s.

## 2026-10-26, tarea 1.2: `fix/auditoria-comparacion` NO mezclada (revisión con hallazgo confirmado)

- Qué se hizo en la rama (último commit 8e26294, en origin): además de F3, T1, T2, H7, C7, C5, C2, C1 (ver entrada anterior), partición por contenido verificable (4a64c48), diagnóstico del HDP sólo entre cadenas de la misma posterior (a642ff7), Pareto por diferencia pareada por obra con IC de Holm, mínimo de obras y robustez a ciclos (4c2d086, 3e43f44), `--resume` exige el mismo contenido (88e1777), canonicalización sin casi-choques ocultos (937fdd9), celdas HDP con hiperparámetros mixtos (8e26294).
- Notas: el contraejemplo de Holm de la revisión no era válido (comparaba contra un Bonferroni de primer paso). Con el código corregido, `fast slow p_boot 0.0404 alpha 0.05 ... rejects True`; el cambio de frontera sólo aparece cuando Holm no rechaza, y eso está cubierto por pruebas. `PARETO_MIN_PAIRED_WORKS = 10` es una constante nueva que el preregistro no fija: Adrián debe confirmarla o registrarla como enmienda. Los `pareto_summary.json` viejos sin campos `token_weighted_holm_*` vuelven a los marginales (`no_holm_adjusted_interval`); hay que regenerarlos.
- Hallazgo confirmado (media, bloquea): `Comparacion/statistics.py:283` `_apply_token_weighted_holm` usa `n_tests = len(token_weighted_draws)`. Si falta `n_tokens` en una obra de un modelo, la familia baja de 6 a 3 (α del primer paso 0.00833 → 0.01667) sin dejar registro. Fijar la familia en las 6 comparaciones preregistradas y registrar su tamaño.
- Abierto: el hallazgo anterior; la integración con la API nueva del HDP (notas 1 y 2: `burn_in` en `build_cell_chain_diagnostics` y renormalización de la predictiva) se revisa al mezclar después de 1.1.

## 2026-10-26, tarea 1.3: `fix/auditoria-d1b` NO mezclada (revisión con hallazgo confirmado)

- Qué se hizo en la rama (último commit 66a19b9, en origin): el preprocesamiento de PDMX extrae la melodía y el caché guarda el modo (e5d31d8), D1b documentado en la auditoría (7b4da5f), el caché guarda los parámetros legacy y `--resume` valida la extracción (5ff431f), el caché firma la representación y sus parámetros (66a19b9). El error usa la clase `CorpusCacheModeError` que ya existía.
- Hallazgo confirmado (media, bloquea): `next_token_experiment/data/preprocess.py:538`, la firma del caché no incluye `corpus.min_events_per_piece`. Evidencia (probe_r3.py): `min_events=99 from cache: pieces 1 excl 0` contra `min_events=99 no cache: pieces 0 excl ['too_short']`, sin `CorpusCacheModeError`. Impacto práctico bajo: ningún perfil ni CLI cambia el valor por omisión (32).
- Abierto: agregar el umbral a la firma (con prueba); caso límite de un caché viejo sólo con exclusiones (sin firma de tokens, se acepta).

## 2026-10-26, integración de la Fase 1

- El integrador mezcla sólo las ramas con `clean=true`. Las tres ramas (hdp, comparacion, d1b) tienen `clean=false` por los hallazgos anteriores, así que no se mezcló ninguna. `fix/auditoria-tesis` sigue en de5d5d9 más esta entrada.
- Siguiente paso: corregir con TDD los hallazgos de cada rama, repetir la revisión y mezclar en el orden hdp → comparacion → d1b.
## 2026-10-27, estado para retomar (sesión cortada o sin uso)

- Insumos completos en esta computadora (`external/` no se versiona):
  - MuseTrainer: `external/library`, clon de https://github.com/musetrainer/library en 9128876 (69 partituras en `scores/`). Es el mismo commit que la copia de la USB (`backup-pendrive-temporal/Melodies_preclone_backup_20260814/external/library`).
  - SymbTr: `external/SymbTr`, clon de https://github.com/MTG/SymbTr en 246bbe3 (2025-04-18), 2200 archivos `.xml`. Decisión de Adrián: usar el público; puede no ser la versión de la laptop.
  - Falta anotar el hash agregado de MuseTrainer y SymbTr (mismo método que PDMX, tarea 0.3).
- Preregistro: Enmienda 1 del 2026-10-27 (cd8338b), regla de la frontera de Pareto con mínimo 10 obras en común, aprobada por Adrián antes de correr.
- Ramas (todas en origin, sin cambios sin commit al escribir esta entrada):
  - `fix/auditoria-hdp` a183bd2: criterio (b) en 4 bloques de 50 semillas (400/200) con r = 0.025, −0.034, 0.152, 0.177 (simulada) y −0.109, −0.162, −0.117, 0.087 (PDMX obra 200); salidas `scripts/auditoria_2026_10_26/hdp/s8_criterio_b_bloques_*_salida_20261026.txt`. La revisión adversarial confirmó que la regla «cada bloque pasa TOST» no tiene potencia (≈ 1.8 % bajo independencia) y que con el criterio escrito (|r| < 0.3) y las 200 semillas juntas la equivalencia sí se demuestra (cifras del revisor, sin verificar por el implementador: simulada r = 0.069, TOST p = 0.0004; PDMX r = −0.100, TOST p = 0.0016). Un agente estaba reescribiendo la decisión con el criterio escrito y quitando el xfail; si no hay commit posterior a a183bd2, repetir esa tarea.
  - `fix/auditoria-comparacion` 7282d1f: familia de Holm fija en las comparaciones del diseño (prueba sin datos = p 1, «no evaluable», `holm_family` en el JSON); suite 331 pasadas + 3 saltadas. Pendiente: revisión adversarial (estaba en curso) y `scripts/figuras_tesis.py:513`, que falla con `p_value_holm` None.
  - `fix/auditoria-d1b` a2a8ddc: firma del caché con `min_events_per_piece` y `extractor_version`; suite 271 pasadas + 3 saltadas. Pendiente: revisión adversarial (estaba en curso).
- Siguiente: terminar revisiones → mezclar hdp → comparacion → d1b con suite en verde tras cada una → 1.4 → 1.5 → Parada 1.
- Entorno de Claude Code en esta máquina (no es del repo): se repararon `C:\Program Files\Git\etc\fstab` (estaba en blanco), el PATH de usuario (`Git\bin`) y los hooks de ECC (`~/.claude/ecc-windows-fix.js`, volver a correrlo si ECC se actualiza). Bash todavía reduce `\` a `\` en comandos; escribir scripts con la herramienta de archivos.
## 2026-10-26, computadora con GPU: tarea 1.2 (`fix/auditoria-comparacion`) implementada, sin mezclar

- Base traída con `git merge origin/fix/auditoria-tesis` (commit 0c1986e). Suite tras la mezcla: `282 passed, 3 skipped` (`python -m pytest -q`).
- C1, C2, C5, C7, H7, F3, T1, T2: las pruebas de cada commit de la rama (`tests/test_bos_support.py`, `next_token_experiment/tests/test_label_smoothing.py`, `next_token_experiment/tests/test_benchmark_representations.py`, `tests/test_hdp_cell_diagnostics.py`, `tests/test_headline_estimates.py`, `tests/test_pareto_uncertainty.py`, `tests/test_grid_ceiling.py`, `tests/test_cost_scope.py`) fallan o no importan sobre la base previa c8c49a0 y pasan en la rama.
- Puntos (a)-(f) y notas 3 y 4 de esta bitácora: commit 4a64c48, pruebas en `tests/test_integracion_particiones.py` (vistas en rojo). Archivos nuevos por corrida: `splits/manifest.json`, `checkpoint_inputs.json`; `execution_plan.json` lleva `split_sha256` y se escribe también en corridas completas; columna `train_subset_sha256` en `results_raw.csv` y `piece_metrics_raw.csv`.
- Abierto: revisión adversarial (2 subagentes) y merge a `fix/auditoria-tesis`; nota 1 (burn-in del diagnóstico del HDP) depende de cómo quede `fix/auditoria-hdp`; nota 5 (títulos de `scripts/figuras_tesis.py`, p. ej. «media de 6 corridas») queda para la Fase 3.

## 2026-10-28, tarea 1.4: análisis por obra held-out (`feat/porobra-heldout`, desde `fix/integracion-fase1` dee896d), sin mezclar

- Qué se hizo: `src/analysis/porobra.py` (sección A del preregistro, Aclaración 2, Enmienda 3) y el CLI `src/cli/porobra_heldout.py` (commits d260c34 y 8b44437). Pruebas: `tests/test_porobra_heldout.py` (22) y `tests/test_porobra_cli.py` (4), vistas en rojo antes de implementar (ImportError del módulo; KeyError de la columna `error` con un archivo roto).
- Decisiones de implementación que el preregistro no fija: n-gramas con la predictiva posterior exacta de Dirichlet (los conteos se actualizan con cada nota de prueba); finito por Baum-Welch con una inicialización Dirichlet(1), semilla 0, seudoconteo 1e-3, tolerancia 1e-4 y techo de rejilla 512; hiperparámetros del HDP fuera de η: alpha 8, alpha0 4, gamma 2, kappa 0 (los de siempre); las cadenas de validación de η también son 4; partición ⌊0.8 n⌋. Menos de 10 obras convergidas en común: «no_evaluable» (regla de la Enmienda 1 reutilizada).
- Smoke real (3 obras de MuseTrainer, cadena corta 40/20, `--n-workers 3`): 108 s de reloj, 3 obras, 0 fallas, 0 duplicados, 0 cortas; con 40 iteraciones ninguna obra pasa R-hat < 1.1 (2.63, 4.15, 1.91) y el resumen marca el aviso del 10 %. El finito eligió K = 8, 128 y 32 (la rejilla se amplió en dos obras).
- Abierto: revisión adversarial y merge; extrapolación de tiempo de la corrida completa (tarea 1.5).

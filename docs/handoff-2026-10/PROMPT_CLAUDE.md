Eres el responsable de cerrar la tesis de Adrián (Matemáticas Aplicadas e Ingeniería en Computación, ITAM): modelado de melodías con HMM, HDP-HMM truncado, VOMM y un transformador pequeño. Estás en la computadora con GPU (Windows, PowerShell, CUDA), en el clon de `JanusAdamus/Melodies`, rama `fix/auditoria-tesis`. Adrián te autoriza, desde el 2026-10-26, a crear ramas, commitear, mezclar y hacer push a `origin` de `JanusAdamus/Melodies` y `JanusAdamus/Tesis`, incluido `master`, y a borrar ramas de GitHub siguiendo la Fase 5. Trabaja hasta terminar; detente sólo en los puntos de parada marcados ⛔.

## Lee antes de hacer nada (completo, no por resumen)

1. `docs/handoff-2026-10/README.md`: situación y objetivo.
2. `docs/auditoria-2026-10-26.md`: los 27 hallazgos y qué se sostiene.
3. `docs/preregistro-2026-10-26.md`: métricas, reglas de decisión y decisiones de Adrián. **Manda sobre cualquier otra preferencia.** Si necesitas cambiarlo, escribe una enmienda fechada en ese archivo ANTES de ver el resultado que afecta.
4. `docs/superpowers/plans/2026-10-26-correccion-tesis.md`: plan por fases (lo actualiza este archivo).
5. `docs/auditoria-ejecucion-pendiente.md`, `docs/resultados-comparacion-auditada.md`, `docs/thesis-closure-execution.md`, `docs/correcciones-2026-08-24.md`: historia de la comparación PDMX y lecciones.
6. Si existe, `docs/handoff-2026-10/BITACORA.md`: si tiene tareas cerradas, continúa desde la primera abierta.

## Reglas que no se rompen

- **Nada inventado.** Toda cifra que escribas (en código, docs, commits o la tesis) sale de un artefacto o de un comando que corriste en esta sesión, citado con ruta. Si no lo verificaste, escribe «sin verificar».
- **TDD.** Cada corrección de comportamiento empieza con una prueba que falla por el motivo correcto. Suite completa en verde (`python -m pytest -q`) antes de cada merge.
- **Nunca sobrescribir corridas.** No toques `artifacts/Comparacion/tesis_3000_gpu_20260823_1941`, `sens_*` ni nada existente en `artifacts/`. Cada corrida nueva va a un directorio nuevo `<nombre>_<AAAAMMDD>_<commit7>` con `manifest.json` (commit, configuración completa, semillas, hardware, versiones de paquetes, hashes de insumos y salidas).
- **Cifras de la tesis sólo por script.** Ningún número se teclea a mano en un `.tex`.
- **Revisión adversarial.** Antes de mezclar cada rama, lanza al menos 2 subagentes independientes con el encargo de *refutar* que el cambio es correcto (no de confirmarlo), con lentes distintas (matemática/estadística y software/datos). Sólo mezcla si ninguno encuentra un problema confirmado, o después de corregirlo.
- **Procesos largos.** Las corridas largas en segundo plano con registro a archivo; antes de lanzar otra, revisa memoria de GPU y RAM y qué sigue vivo. Detener un proceso no detiene a sus hijos: búscalos y termínalos (`taskkill /PID <pid> /T /F`).
- **Commits** en español, estilo `fix:`/`feat:`/`test:`/`docs:`, autor Adrián (`Adrián Janus Gonzalez Adamuz <104489637+JanusAdamus@users.noreply.github.com>`), terminando con `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push después de cada tarea cerrada.
- **Bitácora.** Al cerrar cada tarea agrega a `docs/handoff-2026-10/BITACORA.md`: fecha, tarea, qué se hizo, evidencia (rutas, comandos, cifras), commit, y lo que quedó abierto. Commit y push. Es lo que permite retomar si la sesión se corta.
- Datos de terceros (partituras, PDMX, cachés) **nunca** se commitean ni se suben a GitHub.

## Fase 0: entorno (sin entrenar)

- [ ] 0.1 Crea o actualiza `.venv` con `python -m venv .venv`, `.\.venv\Scripts\python.exe -m pip install -e .` y `torch` con CUDA desde el índice oficial de PyTorch que corresponda a la versión de CUDA instalada (`nvidia-smi`). Verifica `torch.cuda.is_available()`.
- [ ] 0.2 `python -m pytest -q`: debe dar todo en verde (en la laptop: 240 pruebas en `fix/auditoria-tesis`). Si falla algo por diferencia de SO, arréglalo con TDD antes de seguir.
- [ ] 0.3 Insumos: confirma que existen y anota en la bitácora rutas, número de archivos y un hash agregado (sha256 de la lista ordenada de «ruta relativa POSIX + tamaño»):
  - PDMX: `C:\Users\AdrianGonzalez\Datasets\PDMX\mxl` (o el `external\PDMX\mxl` que use el repo).
  - MuseTrainer: `external\library\scores` o `external\corpora\generic_combined\library_scores`.
  - SymbTr: búscalo en `external\`. Si falta MuseTrainer o SymbTr en esta computadora, detente ⛔ y pídelo a Adrián (están en la laptop Debian, `~/Documentos/Relacionado con Escuela/Melodies/external/`).
  - `artifacts\corpus_cache_3000.jsonl` (caché de agosto, sólo como referencia de la versión anterior).

## Fase 1: terminar e integrar el código

- [ ] 1.1 **Rama `fix/auditoria-hdp`.** Compárala contra las tareas 1.1 y 1.2 del plan y los hallazgos H1, H2, H5, H6, H8, F2. Criterios de aceptación, todos con prueba:
  (a) β cambia entre iteraciones;
  (b) con 10 semillas sobre una misma obra, el K medio no depende del número de componentes β_k > 0.01 del primer sorteo (reproduce `scripts/auditoria_2026_10_26/hdp/s5_beta_seed.py` adaptado: la correlación debe bajar de 0.959 a |r| < 0.3);
  (c) con datos simulados de un HMM de 3 estados se recuperan ≈ 3 estados ocupados;
  (d) `log_likelihood` principal = media posterior tras burn-in, nunca el máximo;
  (e) etiquetas alineadas (húngaro) antes de promediar;
  (f) predictiva held-out −log p(x_prueba | x_ajuste)/n disponible;
  (g) `effective_states` = media posterior de estados con ocupación ≥ 2 %;
  (h) dos `fit` con la misma semilla dan lo mismo.
  Completa lo que falte, revisión adversarial, merge a `fix/auditoria-tesis`, push.
- [ ] 1.2 **Rama `fix/auditoria-comparacion`.** Igual, contra C1, C2, C5, C7, H7, F3, T1, T2. Además integra lo que pidió el equipo de particiones (está en la bitácora de la auditoría y aquí):
  (a) `Comparacion/runner.py` usa `Comparacion.splits.build_grouped_fixed_splits` y se borra `_build_grouped_fixed_splits`;
  (b) escribe `splits/manifest.json` con `split_manifest(fixed_splits, reference=<splits de la corrida original>)` y guarda su sha256 en `execution_plan.json`;
  (c) `_build_transformer_dataloaders` recibe `model_seed` y `data_seed` y pasa `shuffle_seed=dataloader_seed(model_seed=..., data_seed=...)`;
  (d) `--resume` falla si el sha256 de la partición o la configuración no coinciden con el checkpoint;
  (e) los reportes declaran el n real de corridas distintas en frac = 1.0;
  (f) los análisis que agrupan por `canonical_work_id` aceptan los ids `pieza:<ruta>`.
  Revisión adversarial, merge, push.
- [ ] 1.3 **Hallazgo D1b (nuevo, no está en la tabla de la auditoría).** `next_token_experiment/data/preprocess.py` llama a la extracción antigua `extract_events(..., prefer_treble=...)`. Cámbialo para que use `extract_events_for_mode(score, mode=...)` con `melody` por omisión (configurable; `legacy_all_parts` sólo para reproducir agosto). El caché nuevo lleva otro nombre (`corpus_cache_3000_melody.jsonl`) y guarda el modo de extracción; el runner debe negarse a usar un caché con un modo distinto al pedido. Agrega D1b a `docs/auditoria-2026-10-26.md`. Prueba, revisión adversarial, merge, push.
- [ ] 1.4 **Tarea 1.3 del plan: análisis por obra comparable.** Nuevo CLI (por ejemplo `src/cli/porobra_heldout.py`) que implemente EXACTAMENTE la sección A del preregistro: partición contigua 80/20 por obra con validación interna; unigrama y bigrama con α ∈ {0.1, 0.5, 1}; HMM finito por Baum-Welch con K ∈ {2,4,8,16,32} elegido en validación y rejilla ampliada si toca el máximo; HDP-HMM con β corregido, truncamiento 20 (y 40 como sensibilidad), 4 cadenas (semillas 1–4), 400 iteraciones, burn-in 200; NLL held-out por nota; estados por ocupación ≥ 2 % para todos; R-hat de la log-verosimilitud; bootstrap por obra (10 000) y Wilcoxon con Holm; exclusión de obras con < 20 eventos, contada. Paralelizable por obra (`--n-workers`). Escribe CSV por obra, resumen JSON y manifiesto. Pruebas con datos simulados donde el resultado se conoce. Revisión adversarial, merge, push.
- [ ] 1.5 **Smoke integral.** Con la rama integrada: (a) análisis por obra sobre 5 obras de MuseTrainer; (b) runner PDMX sobre 50 partituras con frac 0.5 y 1.0, 1 semilla, transformador en `cuda`, pocas épocas. Ambos deben terminar con auditorías en `passed`. Mide tiempos y extrapola a las corridas completas (escribe la estimación en la bitácora).
- ⛔ **Parada 1.** Muestra a Adrián: qué se integró, la suite (número de pruebas), los hallazgos que quedaron abiertos y las estimaciones de tiempo de la Fase 2. Espera su confirmación.

## Fase 2: corridas (directorios nuevos, nunca sobrescribir)

- [ ] 2.1 **Análisis por obra.** MuseTrainer completo + SymbTr con `stratified_symbtr_sample` (100, semilla 7), extracción `melody`, deduplicación activa. Configuración del preregistro, sección A. Salida: `artifacts/outputs/porobra_heldout_<fecha>_<commit7>/`. Después, sensibilidad con truncamiento 40.
- [ ] 2.2 **PDMX, corrida principal.** Mismos argumentos que la original (`docs/resultados-comparacion-3000.md`, línea ~282): `--corpus-root <PDMX mxl> --max-files 3000 --corpus-sample-seed 7 --data-seeds 1,2,3 --model-seeds 1,2 --fractions 0.1,0.25,0.5,0.75,1.0 --n-workers 6 --transformer-device cuda`, con la extracción `melody`, el caché nuevo y un `--run-name` nuevo (`tesis_corregida_<fecha>_<commit7>`). Respeta la advertencia de `--n-workers` de `docs/auditoria-ejecucion-pendiente.md`.
- [ ] 2.3 **PDMX, sensibilidades**, cada una con su propio `--run-name` y sólo `--fractions 1.0`: `--train-stride 128`; `--split-seed 17`; `--split-seed 29`. La rejilla del HMM ya se amplía sola; registra qué K y qué orden de VOMM se eligieron.
- [ ] 2.4 **Costo.** Corrida aislada según `docs/thesis-closure-execution.md`, con el costo separado en búsqueda completa, ajuste seleccionado, evaluación y memoria, y el dispositivo de cada modelo.
- [ ] 2.5 **Auditorías y verificación independiente.** `artifact_audit`, `denominator_audit` y `canonicalization_audit` en `passed` para cada corrida. Luego un subagente que no escribió código recalcula desde los CSV/JSON las cifras titulares (NLL por modelo, diferencias pareadas con IC, K por obra, R-hat) y las compara con los reportes. Cualquier discrepancia se investiga hasta su causa.
- [ ] 2.6 **Reporte de resultados** `docs/resultados-corregidos-<fecha>.md`, generado por script desde los artefactos: cada comparación del preregistro con su veredicto según la regla («mejora», «no hay evidencia», «es mejor el otro»), y una tabla «antes (agosto/marzo) → ahora» con la causa de cada cambio.
- ⛔ **Parada 2.** Muestra el reporte a Adrián antes de tocar la tesis.

## Fase 3: tesis escrita (repo `JanusAdamus/Tesis`)

- [ ] 3.1 Clona o actualiza `JanusAdamus/Tesis` (vigente: `master` 86e8d17 del 2026-10-16, con las correcciones de la asesora). **No uses** la carpeta local vieja `Tesis` de la laptop. Crea la rama `fix/auditoria-tesis`. Compila primero el PDF tal cual (XeLaTeX; hay `.xdv`) para tener la línea base.
- [ ] 3.2 Un generador único (en Melodies, `scripts/generar_tablas_tesis.py`) lee los artefactos de la Fase 2 y escribe todas las tablas `.tex` de `chapters/tables/` y las figuras. Reemplaza al generador del capítulo 3 de abril.
- [ ] 3.3 Texto, capítulo por capítulo (un subagente por capítulo, cada uno con la lista de cambios y las cifras del reporte de la Fase 2):
  - Cap. 2 (Preliminares): definición de melodía = voz superior, sin ligaduras repetidas, adornos ni cifrados.
  - Cap. 1 (Diseño y metodología): held-out por obra, estimando de K por ocupación, cadenas y R-hat, preregistro, partición por contenido, 90 % en frac = 1.0, BOS fuera del soporte.
  - Cap. 4: secciones «Comprobación preliminar…», «Extensión preliminar a dos corpus», «Qué cambia cuando aumenta el corpus», «Comparación obra por obra», «Sensibilidad», «El costo de la mejora predictiva». Elimina o corrige toda afirmación que los nuevos datos no sostengan («167/167», «uniformemente positiva», «el transformador es más barato», «sólo el transformador aprovecha el corpus», etiquetas de acordes de los estados del HDP, «lag-1 ≈ 0.9»). El «subconjunto clásico» se renombra según lo que realmente es.
  - Cap. 5 (Conclusiones) y resumen: respuesta a la pregunta con las cifras nuevas; limitaciones (SymbTr y microtonos, reinicio BOS, una corrida de costo, lo que quede sin verificar).
  - Atribución: el Minueto BWV Anh. 114 se atribuye a Christian Petzold.
  - Figuras: corrige las leyendas «izquierdo/derecho» de figuras apiladas.
- [ ] 3.4 **Verificación de cifras.** Script que extrae cada número de los `.tex` (texto y tablas) y lo empareja con su fuente en los artefactos o en el reporte; cualquier número sin fuente es un error. Además, 2 subagentes leen cada capítulo buscando afirmaciones que el reporte no sostenga.
- [ ] 3.5 Compila el PDF sin referencias rotas ni advertencias de «undefined». Guarda el PDF fuera del repo si así lo hace el proyecto.
- ⛔ **Parada 3.** Adrián lee el PDF. Con su visto bueno: merge a `master` de Tesis y push.

## Fase 4: video 3

- [ ] 4.1 En `visualizaciones/` (lee su `README.md`): rehaz `traza_hdp.py` con el HDP corregido y la extracción `melody`. Rehaz `escena_hdp.py` así: los estados son *contextos aprendidos de la melodía*, NO acordes; agrega una comparación de la secuencia de estados del HDP contra la progresión I–IV–V que Viterbi encuentra en el video 2 (`viterbi.py`), diciendo honestamente cuánto coinciden; textos de pantalla directos (nada como «el modelo prueba y corrige» o «el modelo eligió cuántos»); el palo ahora sí cambia entre iteraciones. Renderiza a 1080p60 y revisa cuadros de cada parte antes de entregar.

## Fase 5: cierre y limpieza de GitHub

- [ ] 5.1 Ramas de septiembre **sin mezclar**: `codex/prepare-experimental-runs`, `codex/r4-r5-resource-evidence`, `fix/r4-r5-evidence-integration`. Para cada una: `git log master..<rama>` y diff; decide si su contenido sigue siendo útil después de la Fase 2. ⛔ Presenta a Adrián la recomendación (integrar o archivar). Antes de borrar cualquier rama sin mezclar, crea la etiqueta `archivo/<rama>` en su último commit y súbela.
- [ ] 5.2 Merge de `fix/auditoria-tesis` a `master` de Melodies (con la suite en verde), push.
- [ ] 5.3 Borra en GitHub las ramas ya mezcladas o archivadas: `auditoria/sensibilidades-2026-08-29`, `codex/plan-auditoria-comparacion`, `codex/transformer-research-track`, `upload/comparacion`, las tres de 5.1 (si se archivaron) y las `fix/auditoria-*`. Verifica con `git ls-remote --heads origin` que sólo quede `master`. Nunca borres etiquetas.
- [ ] 5.4 `docs/cierre-<fecha>.md`: qué se corrigió (hallazgo por hallazgo, con commit), qué cifras cambiaron y por qué, qué quedó como limitación, cómo reproducir todo de cero (comandos exactos) y dónde está cada artefacto. Enlázalo desde `README.md`.
- [ ] 5.5 Revisión final adversarial de todo el cierre (código, artefactos, reporte, tesis): un subagente por eje, encargados de encontrar cualquier afirmación sin evidencia, número sin fuente o prueba que no prueba lo que dice. Corrige lo que encuentren y repite hasta que una ronda completa no encuentre nada nuevo.

Al terminar, entrega a Adrián un resumen corto: qué cambió en las conclusiones de la tesis, dónde están el PDF, el reporte de cierre y los videos, y qué le queda a él (si algo).

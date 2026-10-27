# Auditoría de corrección, 2026-10-26

Cinco revisiones independientes sobre `master` cbbd8e1 (modelos HDP, modelos
finitos y VOMM, datos y tablas, evaluación comparativa, transformador). Cada
hallazgo marcado CONFIRMADO se reprodujo con un script; los scripts están en
`scripts/auditoria_2026_10_26/<área>/`. No se modificó código ni artefactos.

La suite existente pasa 200/200 y no detecta ninguno de los hallazgos críticos:
las pruebas verifican forma y aritmética, no comportamiento estadístico.

## Lo que se verificó correcto

- Forward, forward escalado y FFBS coinciden con enumeración exacta (≤ 1e-10).
- La log-verosimilitud reportada es la marginal p(x|θ), no la de Viterbi.
- Baum-Welch escalado, VOMM (backoff, normalización) y su ausencia de fuga.
- Transformador: máscara causal, desplazamiento de etiquetas, padding fuera de
  la pérdida, NLL por token, early stopping sólo con validación.
- Comparación PDMX: mismos tokens y mismo denominador para las cuatro familias,
  unidad de inferencia = obra, Holm sobre las seis comparaciones.
- Todas las cifras de las tablas `.tex` del capítulo 3 coinciden con
  `artifacts/outputs/multicorpus_full` (corrida del 2026-03-26).

## Hallazgos

Área: H = HDP-HMM, F = finito/VOMM, D = datos y tablas, C = comparación PDMX,
T = transformador. Capítulos según `Tesis` en GitHub (86e8d17).

| ID | Sev. | Área | Hallazgo | Evidencia | Afecta |
|---|---|---|---|---|---|
| H1 | CRITICAL | H | β nunca se actualiza: las filas de transición muestreadas tienen ceros exactos, `dirichlet_logpdf` da −inf, el objetivo vale 1e12 constante, L-BFGS-B para con nit=0 y el modo dice `map_stick_breaking` | `hdp/` (traza: Δβ máx = 0.0 en 80 iteraciones) | todo resultado HDP |
| H2 | CRITICAL | H | Consecuencia de H1: K_hdp lo fija el primer sorteo de β. corr(#β_k>0.01, K medio) = 0.959 en 20 semillas; la misma obra da K de 4 a 12 según la semilla | `hdp/s5_beta_seed.py`, `s4_switch_seed.py` | K_hdp = 8.222, K por obra |
| F1 | CRITICAL | F | `FiniteChordHMM` no se ajusta (0 parámetros estimados) y equivale al uniforme: en las 167 obras gana 0.003 nats/nota sobre −ln 12 | `finito/a2_*`; verificado en `analysis.csv` | ΔlogL = 452.751, «167/167» |
| H3 | HIGH | H | Las cifras HDP de la tesis no se regeneran con ningún commit (la corrida es anterior al primer commit). Con el código actual: ΔlogL 488.644, K 7.922 | `hdp/s2_repro*.py`, `s3_*` | capítulo 4, §§ preliminares |
| H4 | HIGH | H,F | Comparación dentro de muestra, modelo ajustado contra fijo: un unigrama de 11 parámetros ya gana 167/167. En held-out (80/20 por obra) el HDP gana 161/167 | `hdp/s3_report.py` | lectura de ΔlogL |
| H5 | HIGH | H | La log L reportada es el máximo de la cadena, incluido el burn-in (17/167 obras eligen una iteración de burn-in); infla ≈ +20 nats por obra | `hdp/s3_report.py` | magnitud de ΔlogL |
| F2 | HIGH | F | K_fin (Viterbi distintos de 252, crece con T, supera el tope 20 del HDP) y K_hdp no son el mismo estimando; 43 grupos de estados con emisiones idénticas | `finito/a3_*` | «32.2 frente a 8.2» |
| D1 | HIGH | D | La «melodía» mezcla ambas manos (`prefer_treble=False`), cuenta notas ligadas como nuevas (1328), notas de adorno (861+390) y cifrados (243); acordes reducidos a su raíz | `datos/probe_events.py`, `scan_corpus.py` | toda tabla del cap. 3/4 y la definición de melodía del cap. 2 |
| D1b | HIGH | D,C | La comparación PDMX (cap. 4) también usa la extracción antigua (`next_token_experiment/data/preprocess.py` → `extract_events`, `prefer_treble=True`): no quita ligaduras, adornos ni cifrados; PDMX tiene muchas *lead sheets* | lectura de código; configs `plan_*` con `"prefer_treble": true`; `tests/test_d1b_extraccion_pdmx.py` (detalle abajo) | toda cifra PDMX |
| D2 | HIGH | D | SymbTr = primeros 100 archivos en orden alfabético: 4 de 155 makams | `datos/` | «repertorio heterogéneo» |
| C1 | HIGH | C | Costo desigual: HMM cobra la selección completa en CPU; transformador un solo ajuste en GPU sin su búsqueda | `comparacion/` | «el transformador es más barato» |
| C2 | HIGH | C | Curva de aprendizaje confundida: HMM eligió K=48 (techo) en 30/30 celdas; VOMM eligió el orden máximo; HDP con β fijo | `comparacion/`, `hmm-grid-sensitivity.md` | «sólo el transformador aprovecha el corpus» |
| H6 | MEDIUM | H | Medias posteriores promediadas sin alinear etiquetas y cruzadas con `best_states` en `interpretation.py` | `hdp/s4_switch_seed.py` | etiquetas «D:7sus4 en contexto…» |
| H7 | MEDIUM | H,C | Diagnóstico de cadenas: concatena celdas distintas y toma el burn-in como ventana temprana | `hdp/s6_diag.py`, `comparacion/diag_hdp.py` | §9 «lag-1 ≈ 0.9» |
| C3 | MEDIUM | C,T | Canonicalización rota (`Path().stem` corta en el punto; grupo `anon` con 44 piezas); fuga por contenido: 4/414 obras de test con ≥25 % en train, una idéntica | `comparacion/fuga_*.py`, `transformer/dup_check.py` | denominador 414 (impacto bajo en PDMX) |
| C4 | MEDIUM | C,T | `data_seed` no hace nada en frac=1.0: las «6 corridas» son 2 réplicas ×3; el DataLoader se siembra con el split y no con `model_seed` | `comparacion/semillas_frac1.py`, `transformer/t6_replicas.py` | barras de error, §2 y §7 |
| C5 | MEDIUM | C | Pareto sin incertidumbre, decidido por milisegundos de evaluación | `comparacion/` | frontera de Pareto |
| C6 | MEDIUM | C | La partición depende del sistema operativo (orden por cadena de ruta): con rutas POSIX coinciden 94/424 obras de test | `comparacion/reproducir_split.py` | reproducibilidad |
| C7 | MEDIUM | C | Estimandos mezclados: perplejidad titular ponderada por token y sin IC; pruebas pareadas por obra sin ponderar | `comparacion/denominador.py` | tablas del cap. 4 |
| D3 | MEDIUM | D | Brahms cae por `NoChord` (indicaciones «rit.») y Mozart se filtra sin contarse: «errores: 1» cuando fueron 2 | `datos/` | denominador del cap. 3 |
| D4 | MEDIUM | D | Dos obras duplicadas (Chopin op. 28 n.º 4, Passacaglia): 165 distintas, no 167 | `datos/dupcheck.py` | 167 obras |
| D5 | MEDIUM | D | SymbTr: microtonos perdidos o asignados por redondeo; `pitchClass` y `midi % 12` discrepan | `datos/probe_microtone.py` | fila SymbTr |
| D6 | MEDIUM | D | Ganancia bruta ≈ longitud de la obra (Spearman 0.903); la figura muestra sólo las 15 mayores | `datos/` | «uniformemente positiva» |
| D7 | MEDIUM | D | Cifras distintas para la misma obra en tablas distintas (corridas con hiperparámetros no declarados) | `datos/` | coherencia del cap. 3/4 |
| T1 | MEDIUM | T | Con label smoothing, la «NLL de validación» no es NLL (+0.19) | `transformer/t3_*` | perfiles research (no la tesis) |
| T2 | MEDIUM | T | Benchmark compara NLL entre representaciones distintas | `transformer/` | docs del benchmark |
| T3 | MEDIUM | T | Variantes de una obra en particiones distintas en el corpus *library* | `transformer/variant_splits.py` | smokes locales |
| F3 | LOW | F,T | BOS forma parte del soporte predictivo (13 clases para 12 alturas) en todos los modelos | `finito/a4_*`, `transformer/t1_*` | perplejidades absolutas |
| H8 | LOW | H | El generador aleatorio vive en la instancia: dos `fit` seguidos dan resultados distintos | `hdp/s7_seed_reuse.py` | reutilización del modelo |
| — | LOW | varios | `max_windows` sesgado, `--resume` no valida configuración, subconjunto «clásico» = 6 obras más cortas, leyendas «izquierdo/derecho» en figuras apiladas, repeticiones no expandidas | informes | menor |

### D1b: la comparación PDMX usaba la extracción antigua

Hallazgo nuevo, posterior a las cinco revisiones; se registró en el handoff
(tarea 1.3) y se corrigió en la rama `fix/auditoria-d1b`.

- **Qué pasaba.** `preprocess_score_file` llamaba a
  `extract_events(score, include_rests, prefer_treble)`, la extracción de D1
  (partes fusionadas, acorde reducido a su raíz, ligaduras, adornos y cifrados
  como eventos).  Toda corrida de `Comparacion` y el caché
  `artifacts/corpus_cache_3000.jsonl` salieron de ahí.
- **Prueba que lo muestra.** Partitura de dos partes: en sol, el acorde Do-Mi-Sol
  y luego Re5; en fa, Do3 y Sol2.  Antes del cambio, el preprocesamiento por
  omisión daba los tokens `[0, 2]` (raíz del acorde); la melodía vigente es
  `[7, 2]` (nota superior; el bajo no entra).
- **Corrección.** `PreprocessingConfig.extraction_mode` (por omisión `melody`)
  y extracción con `extract_events_for_mode`.  `legacy_all_parts` sólo sirve
  para reproducir agosto y conserva `prefer_treble` e `include_rests`.  Pedir
  silencios con `melody` es un error, no se ignora; el perfil
  `research_richer_events`, que los necesita, declara `legacy_all_parts`.  Un
  modo desconocido detiene la corrida en lugar de excluir cada partitura.
  `Comparacion/cli.py` acepta `--extraction {melody,legacy_all_parts}`.
- **Caché.** Cada fila guarda `extraction_mode`; las filas sin ese campo (todos
  los cachés anteriores) cuentan como `legacy_all_parts`.  `prepare_corpus` y el
  runner se niegan a usar un caché con un modo distinto al pedido, y el runner
  lo rechaza antes de crear el directorio de la corrida.  Nombre por modo:
  `corpus_cache_3000_melody.jsonl`; el de agosto conserva
  `corpus_cache_3000.jsonl`.  Comprobado en modo lectura: el caché de agosto de
  esta máquina sólo contiene `legacy_all_parts` y se rechaza para `melody`.
- **Pendiente.** Las cifras PDMX del capítulo 4 siguen siendo las de la
  extracción antigua hasta la corrida de la Fase 2 con `melody` y caché nuevo.

## Lo que se sostiene

- La ventaja predictiva del transformador en PDMX (≈ 0.25 nats por obra) y el
  orden transformador < HMM finito sobreviven a todos los hallazgos.
- En held-out por obra, el HDP-HMM supera al HMM finito heurístico en 161/167.
- El orden VOMM frente a HDP-HMM (+0.03) es frágil.

## Sin verificar

- Las corridas `tesis_3000_gpu_*` y `sens_*` no están en esta máquina; los
  hallazgos de la comparación se reprodujeron con `plan_pdmx_canonical_20260825`
  (mismas particiones) y con los documentos de resultados.
- El efecto de C3 y del reinicio BOS sobre las cifras de la tesis no está medido.

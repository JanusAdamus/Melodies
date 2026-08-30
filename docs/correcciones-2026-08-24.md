# Correcciones y decisiones, 2026-08-24

Registro de lo que se afirmó, lo que resultó ser falso o incompleto, y con qué evidencia se
corrigió. Sesión de análisis de la corrida `tesis_3000_gpu_20260823_1941`.

Existe porque varias de estas afirmaciones llegaron a estar en un documento antes de
verificarse, y conviene que quede escrito cuáles y por qué cambiaron.

---

## 1. Correcciones a afirmaciones propias

### 1.1 «Las pruebas por pares podrían tratar las filas repetidas de vomm como réplicas»

**Se afirmó**: que al haber tres filas idénticas de `vomm` en frac=1.0, cualquier prueba
estadística que las tratara como réplicas independientes estaría inflando el tamaño
muestral, y que había que revisar `pairwise_comparisons.json` por ese motivo.

**Es falso.** `Comparacion/statistics.py` empareja por obra, no por semilla. Las seis
comparaciones se calculan sobre 414 obras de test con métrica válida en ambos modelos, y
las filas repetidas no entran como réplicas en ningún punto.

**Evidencia**: `pairwise_comparisons.json` reporta `n=414` en cada comparación, y el
filtro `_full_fraction` de `Comparacion/statistics.py` selecciona frac=1.0 antes de
emparejar por obra.

**Alcance**: el problema de las filas repetidas es real pero se limita a las barras de
error de la figura. No toca los valores p. Quedó acotado así en
[`resultados-comparacion-3000.md`](resultados-comparacion-3000.md), §6.3.

### 1.2 Patrón de búsqueda incorrecto para `results_raw.csv`

**Se sugirió**: `Select-String "vomm,1.0"` para aislar las filas de `vomm` en frac=1.0. No
devuelve nada.

**Causa**: las columnas son `model,data_seed,model_seed,frac,...`, así que entre `vomm` y
`1.0` van dos campos. El literal no existe en el archivo.

**Correcto**: `Select-String "vomm,\d+,deterministic,1\.0"`.

### 1.3 «Al eje estructural solo le falta el archivo de anotaciones»

**Se escribió** en la primera versión de `resultados-comparacion-3000.md`, §6.1: que el
código de métricas existía y estaba probado y que solo faltaba la entrada.

**Es incompleto, y de forma que subestima el trabajo pendiente.** Faltan dos piezas:

- **A**: no hay anotaciones de referencia. PDMX no las trae y `external/` no tiene otro
  corpus. Esto sí estaba bien identificado.
- **B**: ningún modelo produce segmentaciones. `Comparacion/runner.py:703` recoge
  `evaluation["structural_predictions"]`, pero ningún evaluador escribe esa clave. El lado
  consumidor está construido y probado; el productor no existe.

**Evidencia**: `structural_evaluation.json` reporta además de la anotación faltante
`"missing_artifact": "per-model inferred segment labels and boundaries"`. La búsqueda de
`structural_predictions` en el árbol solo encuentra consumidores, el runner y los tests.

**Consecuencia de diseño que no se había visto**: `adjusted_rand_index` y
`normalized_mutual_information` necesitan etiquetas de segmento, y el transformer no tiene
estados latentes de los que derivarlas. `boundary_f1` es comparable entre los cuatro
modelos; ARI y NMI solo entre los dos HMM.

**Dónde quedó**: §6.1 de [`resultados-comparacion-3000.md`](resultados-comparacion-3000.md),
reescrito con los dos huecos y la asimetría entre modelos.

### 1.4 La prueba de humo del diagnóstico truncó `corpus_cache_3000.jsonl`

**Qué pasó**: para validar `scripts/diagnostico_finite_hmm_k.py` antes de lanzarlo en
serio se corrió con `--max-files 60`, apuntando al cache por defecto, que era
`artifacts/corpus_cache_3000.jsonl`. `prepare_corpus` escribe el cache con exactamente las
obras que pidió esa invocación, así que dejó 60 entradas donde había 3000.

**Efecto**: la corrida real del diagnóstico arrancó con
`resuming from cache: 60 done, 2940 pending` y tuvo que reparsear el corpus. Es
recuperable solo: al terminar, el cache vuelve a tener las 3000. El costo fue tiempo de
parseo, no datos perdidos, y no afecta a ningún resultado ya escrito, porque
`artifacts/Comparacion/tesis_3000_gpu_20260823_1941/` es independiente del cache.

**Arreglo**: el script ya no comparte cache entre tamaños. `--corpus-cache` por defecto es
`artifacts/corpus_cache_<max_files>.jsonl`, de modo que una prueba con 60 obras escribe
`corpus_cache_60.jsonl` y no toca el de 3000.

### 1.5 Ranking de `hdp_hmm` contra `vomm`

No es una corrección de una afirmación errónea sino de una lectura incompleta. Las medias
marginales de `results_summary.csv` ponen `hdp_hmm` (NLL 2.0754) por delante de `vomm`
(2.0822); la comparación pareada por obra invierte el signo (`+0.0336` a favor de `vomm`,
con intervalo que no cruza el cero).

Son dos estimandos distintos: la media marginal pondera por tokens dentro de cada corrida,
la pareada promedia sin ponderar sobre obras. En vez de elegir uno en silencio, quedó
documentado como límite en §6.4, con la instrucción de declarar un criterio y usarlo de
forma consistente.

---

## 2. Hallazgos que cambiaron la lectura de los resultados

No son correcciones de algo dicho antes, sino cosas que la revisión encontró y que alteran
lo que se puede afirmar.

### 2.1 El HMM finito estaba topado por la rejilla

`selected_states` es 48.0 en las 30 corridas de `finite_hmm`, en las cinco fracciones, y 48
es el máximo de `finite_hmm_states=(12, 24, 48)`. Un modelo que siempre elige el techo pudo
haber querido más capacidad.

Esto ataca directamente la conclusión principal: la meseta del HMM finito puede ser un
artefacto de la rejilla, no una propiedad de la familia de modelos.

Quedó en §6.2, con la redacción provisional «el HMM finito con K ≤ 48 satura» hasta que el
diagnóstico lo resuelva.

### 2.2 En frac=1.0 la varianza de datos es cero por construcción

Las tres filas por `model_seed` son idénticas dígito a dígito: mismo `n_train_pieces`
(2173), mismo `n_train_tokens` (500 959), mismo `test_ppl`. Con el test fijo, usar toda la
fracción hace que `data_seed` no tenga efecto. Se detecta porque `vomm` es determinista y
reporta `std_test_ppl = 0.0` exacto con `runs=3`.

La barra de error del último punto de la curva no es del mismo tipo que las demás. Quedó
en §6.3.

### 2.3 El VOMM cambia de capacidad a mitad de curva

`selected_order` salta de 2 a 4 entre frac=0.5 y frac=0.75, y la tabla de conteos pasa de
~2000 a ~88 000 entradas. Parte de su pendiente es ese salto, no escalado con los datos.
Quedó en §6.5.

### 2.4 El transformer domina a los dos HMM también en costo

169 s de ajuste contra 578 y 628, además de mejor NLL. Invierte el argumento habitual del
canje entre calidad y costo. Con la advertencia obligatoria de que corrió en GPU mientras
los clásicos corrieron en CPU. Quedó en §4.

---

## 3. Decisiones tomadas

| Decisión | Fecha | Consecuencia |
|---|---|---|
| Se conservan los tres ejes del marco multidimensional | 2026-08-24 | Hay que conseguir anotaciones e implementar los productores de segmentación para los cuatro modelos. Al reportar, declarar que ARI y NMI solo cubren los dos HMM. |
| Se diagnostica el techo de la rejilla antes de rehacer la curva | 2026-08-24 | ~35 min de cómputo en vez de ~14 h, y la rehecha solo si el diagnóstico confirma que el techo era vinculante. |

---

## 4. Cambios en el repositorio

| Archivo | Cambio |
|---|---|
| `docs/resultados-comparacion-3000.md` | Nuevo. Reporte de la corrida: protocolo, resultados de predicción y costo, comparaciones pareadas, cinco límites, guía de redacción para la tesis. |
| `docs/sources-and-methods.md` | §1.1 nueva, con la segunda ronda de búsqueda y sus 7 consultas textuales. §4 pasa de 20 a 67 referencias: corpus, representación, VOMM, HMM finito, transformer, curvas de aprendizaje, estadística, métricas estructurales y decisión multiobjetivo. |
| `docs/index.md` | Entradas para el reporte y para esta nota. |
| `scripts/diagnostico_finite_hmm_k.py` | Nuevo. Diagnóstico del techo de la rejilla del HMM finito. |
| `docs/correcciones-2026-08-24.md` | Este archivo. |

---

## 5. Advertencia sobre las referencias

Las entradas 21, 22, 28, 34, 35, 36, 58 y 63–66 de `sources-and-methods.md` §4 se
verificaron contra resultados de búsqueda y llevan DOI o URL. Las canónicas de la
formulación del transformer y de la estadística clásica (32, 33, 37–47, 48–57, 59–62, 67)
se citan de memoria por ser estándar: **sus DOI no se han comprobado uno a uno**. Pasarlas
por un gestor bibliográfico antes de entregar.

`parallel-cli`, `PARALLEL_API_KEY`, `OPENROUTER_API_KEY`, `gget` y `pandoc` no están en
esta máquina, así que la búsqueda usó `WebSearch` y no se generó PDF ni figuras
esquemáticas. No es una revisión PRISMA; es búsqueda dirigida, y así está declarado en
`sources-and-methods.md` §1 y §1.1.

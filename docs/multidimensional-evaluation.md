# Evaluacion multidimensional

Melodies estudia compromisos entre tres dimensiones. La descripcion estructural
es la pregunta principal: fronteras, segmentos y particiones latentes. La
prediccion del siguiente evento mediante NLL es una tarea secundaria comun que
permite comparar familias distintas sobre la misma salida observable. El costo
de ingenieria, la complejidad y la opacidad se reportan horizontalmente; no se
mezclan en una puntuacion unica.

Esta infraestructura prepara el protocolo y sus artefactos. Las pruebas
unitarias y los smoke tests usan datos sinteticos o dobles pequenos. No son
evidencia de tesis. Las corridas pesadas o canonicas, sus tiempos y sus
conclusiones deben ser ejecutados y revisados por el autor.

## Familias y alcance

- `finite_hmm` y `hdp_hmm` son modelos de estados latentes con interes
  descriptivo, ademas de su puntuacion predictiva.
- `transformer` es una referencia neuronal causal para la tarea predictiva. No
  convierte atencion en explicacion estructural.
- `vomm` es un control diagnostico opcional, inspirado en PPM, que conecta
  dependencias locales de orden variable con la comparacion. No es IDyOM y no
  aisla causalmente un mecanismo de memoria.

La representacion primaria comparable es `pitch_class`. Las representaciones
mas ricas son ablaciones y no sustituyen el benchmark principal.

## Unidad experimental y cobertura

La particion se hace por `canonical_work_id`, no por archivo: todas las
variantes de una obra permanecen en el mismo split. Las fracciones de
entrenamiento son anidadas para cada semilla. La inferencia pareada vuelve a
agrupar variantes y repeticiones por obra canonica, por lo que un arreglo o una
semilla adicional no se convierte en una observacion independiente.

Validacion y test se dividen en segmentos no superpuestos de hasta
`max_context_length`. La cola positiva se conserva aunque tenga un solo evento.
Cada segmento empieza con BOS como contexto y cada evento musical se puntua una
sola vez. `protocol_audit.json` compara los indices que cada evaluador declara
haber puntuado con `range(len(piece.tokens))`; duplicados, omisiones, indices
fuera de rango, orden incorrecto o conteos distintos detienen la corrida.

El soporte predictivo compartido contiene los simbolos musicales y BOS. BOS no
es objetivo. PAD queda fuera de la normalizacion, `argmax`, top-k y Brier del
Transformer, aunque sigue siendo una entrada valida para rellenar batches. El
VOMM recibe de forma explicita `bos_token_id` y un `vocabulary_size` igual a
`bos_token_id + 1`.

NLL por evento es la medida predictiva primaria. La perplejidad es solamente
`exp(NLL)` y nunca evidencia independiente. Exactitud y Brier son diagnosticos
adicionales cuando existe una distribucion completa comparable.

## Estructura e inferencia

Las anotaciones estructurales opcionales son un CSV con estas columnas:

```text
piece_id,event_index,segment_label,boundary
```

Los indices deben ser enteros no negativos, las coordenadas no pueden
duplicarse y `boundary` debe ser booleano o `0/1`. Cuando falta el CSV,
`structural_evaluation.json` usa `status=not_evaluated` e identifica
`structural_annotations_path` como entrada ausente. Si hay referencia pero la
corrida no produce etiquetas y fronteras inferidas comparables, el artefacto
declara `missing_inferred_structure_artifact`. No inventa estados, fronteras,
significados musicales ni puntuaciones. F1 de fronteras, NMI y ARI se invocan
solo cuando existen ambos arreglos validos.

En la fraccion completa, `pairwise_comparisons.json` contiene todos los pares de
modelos disponibles. Primero promedia variantes y semillas dentro de cada obra
canonica; despues calcula diferencias pareadas de NLL, intervalo bootstrap del
95 %, Wilcoxon cuando es aplicable y ajuste Holm sobre la familia completa de
pares (la del preregistro: no se encoge; un par sin datos cuenta como p = 1 y se
reporta como no evaluable). Los pares los fijan los modelos que el diseno corre
(`expected_models`), no los que dejaron filas: un modelo sin filas aporta pares
no evaluables y un modelo fuera del diseno es un error. El bloque `holm_family`
del JSON registra el tamano de la familia, sus modelos, sus pruebas y cuales no
se evaluaron; `holm_family_by_split_seed` hace lo mismo dentro de cada particion,
que hereda la familia completa.

### Estimando que decide (Aclaración 2 al preregistro)

Decide la media sobre las obras de la diferencia de NLL por nota de cada obra
(`mean_difference`), con el mismo peso para cada obra. Su IC es bootstrap por
obra ajustado por Holm por pasos sobre la familia preregistrada
(`work_mean_holm_*`). La diferencia ponderada por tokens (`token_weighted_*`) se
reporta como secundaria, con su propio IC de Holm, y no decide. El bloque
`decision` de `pairwise_comparisons.json` nombra los campos que deciden, y
`headline_estimates.json` marca `work_mean_nll` con `role: decides`.

Cada prueba trae dos intervalos de Holm. `*_holm_ci` es el IC percentil al nivel
α/(m − r) de su rango: tras el primer no rechazo de Holm puede excluir 0 aunque
la prueba no se rechace, así que no se lee solo. `*_holm_decision_ci` es el que
decide: igual al anterior si Holm rechaza y ensanchado hasta incluir 0 si no
(`holm_family.ci_usage` lo repite en el JSON).

La familia es la del preregistro: cuatro modelos y seis pares. Con
`--without-vomm` la familia no se encoge: los tres pares del VOMM quedan no
evaluables y todas las salidas llevan `analysis_status.status = exploratory`,
porque una corrida sin VOMM no evalúa la regla de B.

Un par con menos de `MIN_PAIRED_WORKS = 10` obras en común tampoco entra a los
pasos de Holm: cuenta en la familia como p = 1 y queda con
`*_holm_status = too_few_paired_works`, sin IC de Holm ni rechazo. Con una sola
obra todas las réplicas bootstrap valen lo mismo y su p es 0; si entrara,
rechazaría al nivel más estricto y aflojaría el α de las demás pruebas. El mínimo
es el mismo de la frontera de Pareto (Enmienda 1) y queda escrito en
`holm_family.work_mean_ci.min_paired_works`.

### Frontera de Pareto (Enmienda 1)

`pareto_summary.json` distingue dos alcances. Si faltan mediciones
estructurales, puede calcular una frontera parcial, etiquetada como
`predictive_cost_partial_frontier`, usando NLL y tiempos finitos. La frontera
completa de estructura, prediccion y costo queda `not_evaluated` o
`incomparable`; nunca se presenta la parcial como si tuviera tres ejes.

En el eje de NLL (`work_mean_test_nll`) decide `work_mean_holm_decision_ci` de la
diferencia pareada: un modelo es peor sólo si Holm rechaza y el IC queda del lado
peor de 0; un par que Holm no rechaza es empate. Un par con menos de
`PARETO_MIN_PAIRED_WORKS = MIN_PAIRED_WORKS = 10` obras en común, o sin obras en común, no decide con
la pareada: se comparan los IC marginales de la media por obra y el par queda en
`nll_paired_pairs_skipped` con su motivo. La NLL ponderada por token
(`mean_test_nll`) se reporta y no decide. Las figuras leen esta frontera; no la
recalculan.

En el eje de costo, el intervalo de cada modelo es
`[min × (1 − t), max × (1 + t)]` del tiempo del ajuste seleccionado entre sus
corridas en frac = 1.0, con `t = PARETO_TIME_RELATIVE_TOLERANCE = 0.10`
(`time_interval` en el JSON). Dos tiempos cuyos intervalos se traslapan son
empate. El 10 % es una decisión de diseño fijada antes de las corridas
corregidas: no está en el preregistro, que sólo pide «la incertidumbre de cada
medición de costo». El tiempo de evaluación se reporta y no decide.

### Partición de referencia

El preregistro (B, Corpus y partición) pide reportar el conjunto de prueba nuevo
con su diferencia respecto del anterior. `--reference-splits` acepta, sólo para
leer, el directorio de la corrida original `tesis_3000_gpu_20260823_1941` (que no
tiene `execution_plan.json`: sus particiones son `splits/test_pieces.json` y
`splits/val_pieces.json`), su `splits/`, uno de esos archivos o un
`execution_plan.json`. La diferencia queda en `splits/manifest.json`,
`execution_plan.json` y `run_summary.json` (`reference_split_comparison`); sin
referencia, esas salidas y `analysis_status.preregistration_gaps` declaran que no
se reportó.

## CLI y planificacion

La entrada instalada es `melodies-comparacion`; tambien puede usarse
`python -m Comparacion.cli`.

```bash
melodies-comparacion --plan-only --max-files 12 --run-name audit
melodies-comparacion --without-vomm --structural-annotations annotations.csv
```

Flags:

- `--plan-only`: prepara el corpus acotado, los splits y el plan, pero retorna
  antes de construir modelos o llamar a `fit`.
- `--without-vomm`: excluye el control VOMM; la corrida queda exploratoria
  (Aclaración 2) y la familia de Holm sigue siendo la de seis pares.
- `--reference-splits PATH`: partición de la corrida original, sólo lectura;
  ver «Partición de referencia».
- `--structural-annotations PATH`: valida el CSV estructural de referencia.
- `--max-files N`: limita los archivos considerados por preparacion.
- `--run-name NAME`, `--corpus-root PATH` y `--results-root PATH`: controlan la
  identidad y ubicacion de la corrida.
- `--transformer-max-epochs N` y `--transformer-device {cpu,auto,cuda,mps}`:
  sobrescriben el trabajo neuronal.
- `--data-seeds`, `--model-seeds` y `--fractions`: aceptan listas separadas por
  comas.

`execution_plan.json` enumera piezas y grupos canonicos por split, tamanos de
fracciones anidadas, filas y ajustes candidatos por familia, semillas,
contexto/reset, soporte comun, cobertura esperada, artefactos y la distincion
entre trabajo clasico/ligero y neuronal. Su estado es
`planned_no_evidence`; no contiene filas de resultados ni afirma evidencia.

## Contrato de artefactos

Una ejecucion conserva `config.json`, `preprocessing_report.json`,
`exclusions.csv` y los manifiestos bajo `splits/`. El modo normal agrega:

- `results_raw.csv`: una fila por familia, fraccion y semillas;
- `results_summary.csv`: curva predictiva agregada;
- `piece_metrics_raw.csv`: metricas por pieza, `canonical_work_id` e indices
  puntuados;
- `pairwise_comparisons.json`: inferencia pareada por obra canonica;
- `engineering_costs.csv`: tiempos, dispositivo y medidas de complejidad en
  columnas separadas, sin indice compuesto entre unidades distintas;
- `protocol_audit.json`: evidencia esperada y observada por modelo/pieza;
- `structural_evaluation.json`: mediciones validas o estado honesto de entrada
  o inferencia ausente;
- `pareto_summary.json`: alcance parcial y alcance completo separados;
- `learning_curve.png`: visualizacion de perplejidad, que sigue siendo
  `exp(NLL)`.

Todos los JSON se escriben sin `NaN` ni `Infinity`. En `--plan-only` se escribe
`execution_plan.json`, pero no se crean `results_raw.csv`,
`piece_metrics_raw.csv` ni otros artefactos que pudieran confundirse con
resultados medidos.

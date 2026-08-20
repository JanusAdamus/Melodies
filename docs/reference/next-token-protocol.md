# Protocolo predictivo comun

## Papel metodologico

La prediccion del siguiente evento es una tarea secundaria compartida para
comparar HMM finito, HDP-HMM, Transformer y el control VOMM opcional. La
descripcion estructural sigue siendo el objetivo principal del proyecto. Una
mejor NLL no demuestra por si misma mejor descripcion musical.

No se asume que exista una corrida canonica. Los smoke tests sinteticos
verifican implementacion, no evidencia. Corpus completos, entrenamientos
pesados, conteos, tiempos y resultados deben ser producidos por el autor.

## Unidad, representacion y splits

La unidad es una secuencia lineal de eventos discretos por pieza. La
representacion comparable primaria es `pitch_class`; configuraciones mas ricas
son ablaciones internas.

Cada `PreparedPiece` tiene un `canonical_work_id` no vacio. La particion fija se
hace por ese identificador y expande despues todas las variantes del grupo, de
modo que una obra canonica no cruza train, validacion y test. Las fracciones de
entrenamiento son anidadas por semilla. La inferencia estadistica vuelve a
agrupar variantes y repeticiones por obra canonica para evitar
pseudorreplicacion.

## Soporte compartido

El soporte de puntuacion es:

```text
simbolos musicales + BOS
```

BOS reinicia el contexto pero nunca es objetivo. PAD no pertenece al soporte
de puntuacion: se excluye de log-softmax, argmax, top-k y Brier del Transformer.
PAD sigue siendo valido en `input_ids` y en la mascara de batches. Para VOMM,
el runner pasa `bos_token_id=tokenizer.bos_token_id` y
`vocabulary_size=bos_token_id+1`. HMM y HDP-HMM usan el mismo soporte
musica+BOS.

## Contexto y cobertura exacta

Entrenamiento puede usar ventanas solapadas. Validacion y test no: cada pieza
se divide en segmentos consecutivos de hasta `max_context_length`, sin
solapamiento. Toda cola positiva se conserva, incluso una cola de un evento.
Cada segmento empieza desde BOS.

Los evaluadores publican `scored_event_indices` por pieza. El runner los
compara con `range(len(piece.tokens))` y escribe ambos arreglos en
`protocol_audit.json`. Duplicados, omisiones, valores fuera de rango, orden
incorrecto o conteos distintos producen `status=failed` y detienen la corrida.
La auditoria no se construye solo con indices esperados.

## Seleccion y medidas

La seleccion de estados, hiperparametros, orden VOMM y checkpoint Transformer
usa solamente validacion. El modelo conservado debe corresponder al candidato
reportado. VOMM contabiliza todo el tiempo de seleccion entre ordenes.

La medida primaria es NLL media por evento musical. Perplejidad es exactamente
`exp(NLL)` y no una observacion independiente. Exactitud, top-k y Brier son
diagnosticos secundarios cuando la familia expone distribuciones completas.

## Inferencia y costo

En `frac=1.0`, `pairwise_comparisons.json` forma todos los pares disponibles.
Promedia primero dentro de `(modelo, obra canonica)`, calcula la diferencia de
NLL `modelo_a - modelo_b`, un intervalo bootstrap pareado del 95 %, Wilcoxon
cuando hay suficientes diferencias no nulas y correccion Holm para los valores
p validos.

`engineering_costs.csv` conserva por corrida tiempo de ajuste, tiempo de
evaluacion, dispositivo, piezas/eventos de entrenamiento y las medidas de
complejidad que cada familia puede justificar. Memoria y energia quedan vacias
con razon explicita cuando no se miden de forma fiable. No se combinan unidades
distintas en un indice de costo.

## Relacion con estructura y Pareto

Las anotaciones opcionales requieren
`piece_id,event_index,segment_label,boundary`. Sin referencia,
`structural_evaluation.json` identifica la entrada ausente. Con referencia pero
sin etiquetas/fronteras inferidas comparables, identifica el artefacto inferido
ausente. Solo con ambos lados validos se calculan F1 de fronteras, NMI y ARI.

Cuando estructura no esta disponible, `pareto_summary.json` puede contener una
frontera parcial predictiva-costo. La frontera completa de estructura,
prediccion y costo se marca `not_evaluated` o `incomparable`.

## Planificacion

`melodies-comparacion --plan-only` prepara solamente el alcance acotado por
`--max-files`, configuracion, manifiestos y `execution_plan.json`. El plan
enumera splits, grupos, fracciones, ajustes, semillas, soporte, reset, cobertura
y artefactos; separa carga clasica/ligera de carga neuronal. Retorna antes de
construir modelos o ejecutar `fit`, no escribe filas de resultados y no afirma
evidencia.

El contrato completo de CLI y artefactos esta en
[evaluacion multidimensional](../multidimensional-evaluation.md).

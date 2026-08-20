# Guia de experimentos

## Pregunta y ejes

El estudio compara que obtiene y que sacrifica cada familia bajo los mismos
datos observables y divisiones por obra canonica.

1. Descripcion estructural: eje primario, evaluado con fronteras y particiones
   solo cuando existen referencia e inferencia comparables.
2. Prediccion: tarea secundaria comun, con NLL por evento como medida primaria.
   Perplejidad significa exclusivamente `exp(NLL)`.
3. Costo de ingenieria: tiempos, dispositivo y medidas de complejidad se
   mantienen en unidades separadas; no se suman en una puntuacion.

HMM finito y HDP-HMM conservan el papel estructural. El Transformer causal es
una referencia predictiva. VOMM/PPM es un puente diagnostico opcional, no IDyOM
y no una intervencion causal que aisle memoria.

## Superficies

`src/` ofrece analisis por pieza, catalogo, corridas multicorpus y modelos
estructurales mediante `melodies-analyze`, `melodies-library` y
`melodies-multicorpus`.

`next_token_experiment/` contiene la preparacion de ventanas y el Transformer.
`Comparacion/` ensambla las cuatro familias, las curvas, la inferencia pareada,
el costo, la auditoria y el estado estructural mediante
`melodies-comparacion`.

## Flujo recomendado

Primero genera un plan acotado:

```bash
melodies-comparacion --plan-only --max-files 12 --run-name audit
```

Revisa `config.json`, `splits/` y `execution_plan.json`: piezas y grupos por
split, fracciones anidadas, ajustes por familia, semillas, soporte musica+BOS,
reinicios de contexto, cobertura esperada y artefactos. El plan distingue
trabajo clasico/ligero del neuronal y declara `planned_no_evidence`.

Una ejecucion normal puede desactivar el VOMM con `--without-vomm` o validar
anotaciones con `--structural-annotations PATH`. Los demas flags de corpus,
salida, semillas, fracciones, epocas y dispositivo se describen en
[evaluacion multidimensional](multidimensional-evaluation.md).

## Equidad y analisis

- Splits agrupados por `canonical_work_id`; variantes no cruzan particiones.
- Segmentos de validacion/test no superpuestos y cola positiva exacta.
- BOS es contexto, no objetivo. PAD queda fuera de la distribucion puntuada,
  pero puede rellenar entradas del Transformer.
- `protocol_audit.json` compara indices realmente expuestos por cada evaluador
  con todos los eventos esperados y falla ante cualquier asimetria.
- La inferencia de fraccion completa promedia variantes y semillas por obra,
  aplica bootstrap pareado, Wilcoxon y Holm.
- Sin mediciones estructurales solo existe una frontera predictiva-costo
  explicitamente parcial. La frontera completa queda sin evaluar.

## Limite de evidencia

Las suites unitarias y los smoke tests sinteticos prueban contratos de
software; no respaldan conclusiones musicales ni comparaciones de rendimiento.
No se documenta aqui ningun conteo, tiempo, cobertura, puntuacion o corrida
canonica ya obtenida. Las ejecuciones pesadas y canonicas corresponden al autor.

# Transformer

## Estado Actual

La linea transformer ya no debe leerse como un experimento unico. Desde ahora el
repo distingue dos tracks:

- `baseline_tesis`: pequeño, reproducible y metodologicamente conservador.
- `research_transformer`: orientado a ampliar representacion, contexto y
  capacidad del modelo.

La hoja de ruta completa vive en
[`transformer-roadmap.md`](transformer-roadmap.md).

## Baseline Historico

El baseline de referencia proviene de:

- `artifacts/next_token_experiment/results/library_smoke_8_timed/transformer`

Metricas reportadas:

- `nll_per_token`: `2.31295`
- `perplexity`: `10.10422`
- `accuracy`: `0.24859`
- `parameter_count`: `415,872`
- `best_epoch`: `10`

La ficha completa esta en
[`baselines/transformer-baseline.md`](baselines/transformer-baseline.md).

## Cambio de Direccion

La siguiente etapa del proyecto deja de tratar al transformer como un simple
"smoke baseline". La prioridad cambia en este orden:

1. mejorar la representacion simbolica;
2. ampliar la evaluacion;
3. ordenar el programa experimental;
4. escalar el modelo con criterio.

La motivacion tecnica se apoya en dos resultados fuertes de la literatura:

- `Music Transformer` muestra la importancia de dependencias largas y sesgos
  relativos para estructura musical:
  https://arxiv.org/abs/1809.04281
- `Pop Music Transformer` muestra que introducir estructura metrica en la
  representacion cambia sustancialmente el modelado:
  https://arxiv.org/abs/2002.00212

## Perfiles

### `cpu_baseline`

Pensado para:

- reproducir la comparacion base en CPU;
- mantener una corrida defendible para tesis;
- preservar comparabilidad historica.

### `gpu_extended`

Pensado para:

- continuar la ruta extendida historica con una maquina mas fuerte;
- ampliar capacidad sin cambiar todavia el tipo de representacion base.

### `research_richer_events`

Pensado para:

- activar el nuevo track de investigacion;
- usar una representacion `event_pitch_duration_metrical`;
- permitir rests y estructura metrica;
- ampliar contexto y capacidad del decoder.

## Nueva Representacion de Investigacion

`event_pitch_duration_metrical` combina:

- pitch class o `REST`;
- duracion cuantizada;
- clase metrica (`downbeat`, `beat`, `offbeat`, `subbeat`, `unknown`).

No pretende ser la representacion final del proyecto. Es el primer paso serio
para salir del flujo plano de `pitch_class`.

## Ejecucion

Baseline CPU:

```bash
python -m next_token_experiment.cli \
  --profile cpu_baseline \
  --run-name cpu_baseline_smoke \
  --max-files 8
```

Track de investigacion:

```bash
python -m next_token_experiment.cli \
  --profile research_richer_events \
  --run-name research_richer_events_smoke \
  --max-files 32
```

## Que Guarda Cada Corrida

- configuracion resuelta;
- orden de ejecucion del pipeline;
- resumen de preprocessing y dataset;
- descripcion de la representacion activa y su vocabulario;
- resumen de entrenamiento;
- metricas de validacion y test;
- metricas `top_3_accuracy` y `top_5_accuracy`;
- metricas por pieza;
- slices por longitud de pieza, compositor y rareza de token;
- continuaciones generadas de forma reproducible desde prompts fijos;
- checkpoint del mejor modelo;
- metadatos de dispositivo y precision usados.

## Siguiente Tramo de Trabajo

Lo siguiente que ya quedo habilitado en el repo es:

- metricas top-k en entrenamiento y evaluacion;
- evaluacion por slices (`piece_length`, `composer`, `token_rarity`);
- continuaciones reproducibles guardadas en `generated_continuations.json`;
- sesgo posicional relativo para el track `research_richer_events`.

Lo siguiente que todavia falta despues de esta etapa es:

- estudiar contextos de 512+ con costo controlado;
- comparar sesgo relativo contra otras variantes para contexto largo;
- comparacion formal entre `pitch_class`, `pitch_class_duration` y
  `event_pitch_duration_metrical`.

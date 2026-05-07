# Plan de ejecucion: comparacion GPU realmente comparable

## Objetivo

Preparar una corrida en la computadora buena donde el `Transformer` pueda
compararse de forma defendible contra `HMM finito` y `HDP-HMM truncado`.

La regla central es simple:

- la GPU solo debe cambiar el hardware;
- no debe cambiar al mismo tiempo corpus, representacion, split, ventanas,
  metrica primaria y presupuesto de evaluacion.

## Problema actual

Las corridas GPU nuevas sobre PDMX sirven como evidencia de escalamiento, pero
no como comparacion justa con los modelos clasicos porque hoy cambian a la vez:

- corpus;
- tamano efectivo del dataset;
- numero de ventanas;
- tamano del modelo;
- precision;
- costo computacional;
- metrica reportada frente a los resultados clasicos ya existentes.

## Principio metodologico

La comparacion justa debe responder esta pregunta:

> dado el mismo corpus, la misma representacion y el mismo protocolo de
> evaluacion, que modelo predice mejor el siguiente token y a que costo
> computacional e interpretativo.

## Condiciones que deben congelarse

Estas condiciones deben ser identicas para `finite_hmm`, `hdp_hmm`,
`transformer_cpu_comparable` y `transformer_gpu_comparable`.

### Datos

- corpus inicial: `external/library/scores`
- split por `canonical_work_id`
- semilla: `7`
- exclusiones registradas desde el inicio

### Representacion

- representacion principal: `pitch_class`
- sin rests
- sin transposicion tonal
- sin metadata extra

### Ventanas

- `max_context_length=128`
- `min_window_length=32`
- `train_stride=64`
- `eval_stride=128`

### Metricas

- metrica primaria: `test_nll_per_token`
- secundarias:
  - `test_perplexity`
  - `test_accuracy`
  - `train_wall_clock_s`
  - `eval_wall_clock_s`
  - `effective_states`
  - `selected_states`

## Corridas que si valen para la comparacion

### 1. Baseline clasico

- `finite_hmm_comparable`
- `hdp_hmm_comparable`

Estas corridas deben producir `next-token` metrics sobre el mismo split que usa
el transformer. No basta con la log-verosimilitud por obra del pipeline
clasico; se necesita evaluacion autoregresiva compatible.

### 2. Baseline transformer CPU

- `transformer_cpu_comparable`

Condiciones:

- perfil base pequeno;
- `fp32`;
- `attention=eager`;
- arquitectura chica y tesis-friendly;
- mismo corpus y mismas ventanas que HMM/HDP-HMM.

### 3. Baseline transformer GPU comparable

- `transformer_gpu_comparable`

Condiciones:

- mismo perfil del baseline CPU;
- misma arquitectura;
- mismo numero de ventanas;
- misma representacion;
- misma evaluacion;
- idealmente mismo `fp32` para evitar meter otra variable;
- cambio permitido: `target_device="cuda"`.

## Corridas que no deben mezclarse con la comparacion base

Estas corridas deben quedar etiquetadas como `research` o `scaling`, no como
evidencia comparativa directa contra HMM/HDP-HMM:

- `pdmx_gpu_serious_4096`
- `pdmx_gpu_serious_8192`
- `pdmx_gpu_final_16384`
- `pdmx_gpu_overnight_131072_e120`

Sirven para discutir escalamiento, no comparabilidad estricta.

## Trabajo de implementacion previo

Antes de correr en la computadora buena, hay que cerrar estas piezas en el repo.

### Fase 1. Perfil comparable de GPU

Crear un perfil nuevo, por ejemplo `gpu_comparable`, con:

- mismo config de `cpu_baseline`;
- `target_device="cuda"`;
- `gpu_required=True`;
- `dataloader_workers` ajustado a la maquina;
- sin crecer arquitectura ni contexto.

### Fase 2. Adaptadores `next-token` para modelos clasicos

Hay que agregar una capa que permita evaluar `FiniteChordHMM` y
`TruncatedHDPHMM` como predictores de siguiente token.

Salida esperada por modelo:

- `validation_metrics.csv`
- `test_piece_metrics.csv`
- `test_summary.json`

### Fase 3. Benchmark unificado

Extender el agregador para leer resultados de:

- `finite_hmm/`
- `hdp_hmm/`
- `transformer/`

y consolidar una tabla unica por corrida.

### Fase 4. Etiquetado de corridas

Separar formalmente:

- `baseline_vs_classical`
- `research_scaling`

para no mezclar evidencia comparable con evidencia exploratoria.

## Orden recomendado de ejecucion en la compu buena

### Etapa A. Validacion corta

1. correr `finite_hmm_comparable_smoke`
2. correr `hdp_hmm_comparable_smoke`
3. correr `transformer_cpu_comparable_smoke`
4. correr `transformer_gpu_comparable_smoke`
5. verificar que todos escriban artefactos con la misma estructura

### Etapa B. Comparacion formal

1. correr `finite_hmm_comparable_full`
2. correr `hdp_hmm_comparable_full`
3. correr `transformer_cpu_comparable_full`
4. correr `transformer_gpu_comparable_full`
5. consolidar tabla comparativa unica

### Etapa C. Solo despues, escalamiento

1. correr `pdmx_gpu_*`
2. reportarlos aparte como investigacion o scaling

## Estructura esperada de resultados

```text
artifacts/next_token_experiment/results/<run_name>/
  split_manifest.csv
  exclusions.csv
  finite_hmm/
    config.json
    validation_metrics.csv
    test_piece_metrics.csv
    test_summary.json
  hdp_hmm/
    config.json
    validation_metrics.csv
    test_piece_metrics.csv
    test_summary.json
  transformer/
    config.json
    train_log.csv
    validation_metrics.csv
    test_piece_metrics.csv
    test_summary.json
```

## Criterio de exito

La comparacion queda lista solo si al final puedes responder, con una sola tabla:

1. quien gana en `test_nll_per_token`;
2. quien gana en `perplexity`;
3. que costo de entrenamiento paga cada uno;
4. que costo de interpretabilidad paga cada uno;
5. si la GPU mejora al transformer sin cambiar el protocolo;
6. si la mejora del transformer compensa frente a HDP-HMM.

## Juicio esperado

La hipotesis razonable hoy es esta:

- `HDP-HMM` seguira fuerte en compacidad e interpretabilidad;
- `Transformer GPU comparable` podria mejorar al `Transformer CPU comparable`
  por velocidad o estabilidad de entrenamiento;
- pero no debe declararse ganador absoluto hasta comparar bajo el mismo
  protocolo.

## Nota de disciplina

Si en la computadora buena aparece tentacion de aumentar contexto, corpus,
precision mixta, vocabulario o arquitectura, eso ya no pertenece a este plan.
Eso debe ir a otra familia de corridas marcada como `research`.

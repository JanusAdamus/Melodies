# Transformer

## Baseline Histórico

El baseline de referencia proviene de:

- `artifacts/next_token_experiment/results/library_smoke_8_timed/transformer`

Métricas reportadas:

- `nll_per_token`: `2.31295`
- `perplexity`: `10.10422`
- `accuracy`: `0.24859`
- `parameter_count`: `415,872`
- `best_epoch`: `10`

La ficha completa está en
[`baselines/transformer-baseline.md`](baselines/transformer-baseline.md).

## Objetivo de la Reorganización

- dejar una ruta CPU reproducible y comparable;
- agregar soporte real de dispositivo para CPU/GPU;
- mejorar trazabilidad de entrenamiento, checkpoints y metadatos;
- separar baseline metodológico de perfil extendido para GPU.

## Perfiles

### `cpu_baseline`

Pensado para:

- reproducir la comparación base en una máquina sin GPU;
- mantener tamaño y complejidad razonables;
- servir como punto de comparación honesto.

### `gpu_extended`

Pensado para:

- aprovechar una máquina más fuerte con GPU;
- entrenar un transformer más capaz en la misma tarea;
- explorar mejora predictiva sin mezclar esa corrida con la comparación base.

## Ejecución

Baseline CPU:

```bash
python -m next_token_experiment.cli \
  --profile cpu_baseline \
  --run-name cpu_baseline_smoke \
  --max-files 8
```

Perfil extendido GPU:

```bash
python -m next_token_experiment.cli \
  --profile gpu_extended \
  --run-name gpu_extended_full
```

## Qué Debe Guardar Cada Corrida

- configuración resuelta;
- manifiesto de split y exclusiones;
- resumen de entrenamiento;
- métricas de validación y test;
- checkpoint del mejor modelo;
- metadatos de dispositivo y precisión usados.

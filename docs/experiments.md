# Guía de Experimentos

El repositorio tiene dos superficies experimentales distintas.

## 1. Pipeline Principal en `src/`

Se usa para:

- análisis por pieza;
- catálogo de biblioteca;
- corridas multicorpus;
- comparación HMM finito vs HDP-HMM.

Entradas típicas:

- `python -m src.cli.main`
- `python -m src.cli.library_analysis`
- `python -m src.cli.multicorpus_analysis`
- `melodies-analyze`
- `melodies-library`
- `melodies-multicorpus`

Sus salidas se escriben en `artifacts/outputs/`.

## 2. Experimento en `next_token_experiment/`

Se usa para una comparación acotada de predicción de siguiente token entre:

- HMM finito;
- HDP-HMM truncado;
- Transformer autoregresivo pequeño.

Ahora se divide en dos pistas:

- `cpu_baseline`: comparación conservadora y reproducible.
- `research_richer_events`: primera pista seria para ampliar representación y capacidad del transformer.

Sus resultados se escriben en:

- `artifacts/next_token_experiment/results/`
- Tambien puede correrse via `melodies-next-token`.

## Criterio Metodológico

- El baseline CPU mantiene la comparación conservadora y defendible para tesis.
- El perfil GPU extiende capacidad sin cambiar la tarea ni la representación
  base.
- Las mejoras del transformer deben reportarse contra el baseline histórico, no
  solo contra intuiciones.

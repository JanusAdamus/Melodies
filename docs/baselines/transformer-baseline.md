# Baseline del Transformer

## Fuente

Corrida histórica usada como baseline:

- `artifacts/next_token_experiment/results/library_smoke_8_timed/transformer`

## Resumen

```json
{
  "nll_per_token": 2.312953411043891,
  "perplexity": 10.104222553111534,
  "accuracy": 0.24858984689766317,
  "n_tokens": 2482,
  "eval_wall_clock_s": 0.11443048600085604,
  "parameter_count": 415872,
  "best_epoch": 10
}
```

## Lectura

- Este baseline es pequeño y CPU-first.
- Sirve como referencia honesta para comparar mejoras posteriores.
- No debe confundirse con un perfil extendido preparado para GPU.

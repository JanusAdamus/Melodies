# Transformer Benchmark Suite

## Proposito

Esta suite convierte el track transformer en una superficie comparable y no solo
 en una coleccion de corridas aisladas. La idea es congelar un conjunto pequeno
 de corridas canonicas, ejecutarlas o consolidarlas de forma uniforme y producir
 una tabla unica de resultados lista para analisis y para integracion en la
 tesis.

## Corridas canonicas

El manifiesto vive en:

- `next_token_experiment/benchmarks/canonical_runs.json`

Las corridas base son:

- `cpu_baseline_smoke`
- `cpu_baseline_full`
- `research_richer_events_smoke`
- `research_richer_events_full`

Cada entrada del manifiesto fija:

- `run_id`
- `profile`
- `purpose`
- `comparison_group`
- `expected_budget_class`
- `run_name`
- `representation`
- `context_length`
- `max_files`
- `max_windows`
- `seed`
- `notes`

## Script de uso

El wrapper de suite vive en:

- `scripts/run_transformer_benchmark_suite.py`

### Solo consolidar resultados existentes

```bash
python scripts/run_transformer_benchmark_suite.py --mode collect
```

### Ejecutar una corrida puntual

```bash
python scripts/run_transformer_benchmark_suite.py \
  --mode execute \
  --run-id cpu_baseline_smoke
```

### Ver configuraciones sin correr nada

```bash
python scripts/run_transformer_benchmark_suite.py \
  --mode execute \
  --only-smoke \
  --dry-run
```

### Filtrar por grupo comparativo

```bash
python scripts/run_transformer_benchmark_suite.py \
  --mode collect \
  --group baseline_vs_research
```

## Artefactos consolidados

La suite escribe en:

- `artifacts/next_token_experiment/results/benchmark_suite/run_manifest_resolved.json`
- `artifacts/next_token_experiment/results/benchmark_suite/run_status.json`
- `artifacts/next_token_experiment/results/benchmark_suite/summary.csv`
- `artifacts/next_token_experiment/results/benchmark_suite/summary.json`
- `artifacts/next_token_experiment/results/benchmark_suite/summary.md`

## Tabla unica de resultados

La tabla consolidada resume por corrida, al menos:

- perfil y representacion;
- longitud de contexto y tamano del modelo;
- uso o no de sesgo posicional relativo;
- numero de piezas y ventanas por split;
- `nll_per_token`, `perplexity`, `accuracy`, `top_3_accuracy`, `top_5_accuracy`;
- tiempos de ajuste, evaluacion y tiempo total;
- numero de parametros;
- dispositivo y precision efectiva.

## Lectura recomendada

Usa la suite en este orden:

1. corre o consolida las `smoke`;
2. valida que las tablas se llenan correctamente;
3. corre las `full`;
4. usa `summary.csv` y `summary.md` como base del primer analisis formal
   baseline vs research;
5. pasa despues al estudio de representacion y de contexto largo.

## Relacion con la tesis

Esta suite existe para alimentar el capitulo comparativo final en
`/home/janusadamuz/Documentos/Tesis`. La pregunta central no es solo quien
obtiene el mejor numero, sino si el costo adicional del transformer produce una
mejora sustancial y defendible frente a HMM y HDP-HMM.

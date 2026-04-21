# Transformer Comparison Analysis

## Estado actual

Ya existe una primera comparacion real entre las corridas:

- `cpu_baseline_smoke`
- `research_richer_events_smoke`

Los resultados consolidados viven en:

- `artifacts/next_token_experiment/results/benchmark_suite/summary.csv`
- `artifacts/next_token_experiment/results/benchmark_suite/summary.json`
- `artifacts/next_token_experiment/results/benchmark_suite/summary.md`

## Lectura metodologica

Estas corridas siguen siendo `smoke`, no `full`. Sirven para validar pipeline,
comparabilidad minima y orden de magnitud del costo. No deben usarse todavia
como evidencia final para la tesis.

Sin embargo, ya permiten responder una pregunta importante:

- el track `research_richer_events` no solo es mas complejo en arquitectura;
- tambien es mas costoso de entrenar aun bajo presupuesto `smoke`;
- y bajo ese mismo presupuesto no supera al baseline en calidad predictiva.

## Resumen numerico inicial

### `cpu_baseline_smoke`

- representacion: `pitch_class`
- contexto: `128`
- parametros: `415,872`
- `nll_per_token`: `2.517699`
- `perplexity`: `12.400031`
- `accuracy`: `0.166016`
- `top_3_accuracy`: `0.430664`
- `top_5_accuracy`: `0.583984`
- `fit_wall_clock_s`: `33.637338`
- `total_wall_clock_s`: `87.663154`

### `research_richer_events_smoke`

- representacion: `event_pitch_duration_metrical`
- contexto efectivo `smoke`: `128`
- sesgo posicional relativo: `true`
- parametros: `4,825,856`
- `nll_per_token`: `5.337212`
- `perplexity`: `207.932204`
- `accuracy`: `0.018066`
- `top_3_accuracy`: `0.049316`
- `top_5_accuracy`: `0.089355`
- `fit_wall_clock_s`: `49.417764`
- `total_wall_clock_s`: `90.323013`

## Comparacion provisional

### Rendimiento predictivo

En esta primera comparacion `smoke`, el baseline supera al track research en
todas las metricas principales:

- menor `nll_per_token`;
- menor `perplexity`;
- mejor `accuracy`;
- mejor `top_3_accuracy`;
- mejor `top_5_accuracy`.

La diferencia es amplia, no marginal.

### Costo de modelo

El `research_richer_events_smoke` usa aproximadamente:

- `11.6x` mas parametros que `cpu_baseline_smoke`.

### Costo de entrenamiento

El `research_richer_events_smoke` exige aproximadamente:

- `1.47x` mas tiempo de ajuste;
- `1.03x` mas tiempo total.

Aunque el tiempo total todavia no explota en `smoke`, el costo de
entrenamiento ya crece de forma visible. La diferencia de costo seria mas
pronunciada si esta corrida se ejecutara con su presupuesto `research` completo.

## Interpretacion provisional

Estos resultados no prueban que el track transformer de investigacion sea una
mala direccion. Lo que si muestran es algo metodologicamente importante:

- una representacion mas rica y un modelo mas grande no producen mejora por si
  solos;
- el costo extra aparece antes que la mejora;
- por lo tanto, la carga de prueba sigue del lado del track research.

La lectura mas razonable hoy es esta:

- el baseline pequeno esta mejor alineado con el presupuesto `smoke`;
- el research track todavia necesita mas trabajo de calibracion y posiblemente
  mas presupuesto de entrenamiento para que la representacion rica pueda
  rendir;
- no seria valido concluir superioridad del transformer solo porque su
  arquitectura es mas potente.

## Hipotesis para la siguiente fase

Las explicaciones mas plausibles del mal desempeno inicial del track research
son:

- vocabulario mucho mas grande y tarea mas dificil;
- muy poco presupuesto de entrenamiento para esa complejidad;
- mayor sensibilidad a representacion y regularizacion;
- necesidad de mas datos efectivos o de un contexto distinto para capitalizar la
  arquitectura.

## Siguiente paso recomendado

El siguiente paso correcto no es escalar mas la arquitectura todavia. Primero
hay que correr:

- `cpu_baseline_full`
- `research_richer_events_full`

y solo despues decidir si:

- la representacion rica empieza a justificarse;
- el sesgo relativo ayuda realmente;
- la mejora observada compensa el aumento de costo frente a HMM y HDP-HMM.

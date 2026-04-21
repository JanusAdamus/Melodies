# Prompt Multiagente para Codex

## Objetivo

Trabaja en dos frentes paralelos y coordinados:

1. `Melodies`: construir infraestructura seria de benchmark para el track
   transformer.
2. `Tesis`: abrir y mantener un capitulo comparativo final sobre costo vs mejora
   entre transformer y HMM/HDP-HMM.

El objetivo no es solo correr modelos, sino producir evidencia defendible para
comparar si la complejidad adicional del transformer justifica una mejora
realmente sustancial.

## Restricciones metodologicas

- No romper la comparabilidad del baseline historico.
- Mantener separados `baseline_tesis` y `research_transformer`.
- No escalar arquitectura o corpus antes de tener una bateria comparable y una
  tabla resumen consolidada.
- Tratar el costo en varias dimensiones:
  - costo de implementacion;
  - costo computacional;
  - costo de experimentacion;
  - interpretabilidad;
  - reproducibilidad;
  - costo de mantenimiento.

## Entregables prioritarios

### En `Melodies`

1. Un manifiesto de corridas canonicas, por ejemplo:
   - `cpu_baseline_smoke`
   - `cpu_baseline_full`
   - `research_richer_events_smoke`
   - `research_richer_events_full`
2. Un script que ejecute o al menos consolide esas corridas.
3. Una tabla unica de resultados en:
   - CSV
   - JSON
   - Markdown para docs
4. Un documento explicando la suite de benchmark y como leer la tabla resumen.

### En `Tesis`

1. Un nuevo capitulo final comparativo sobre transformer vs HMM/HDP-HMM.
2. Tablas base para:
   - comparacion de resultados;
   - comparacion de costo de infraestructura;
   - comparacion de interpretabilidad y reproducibilidad.
3. Un marco explicito para ir actualizando el juicio:
   - si el transformer mejora;
   - cuanto mejora;
   - en que slices mejora;
   - a que costo mejora;
   - si ese costo se justifica frente a HDP-HMM.

## Orden de trabajo obligatorio

1. Bateria de corridas comparables.
2. Agregador de resultados y tabla resumen.
3. Primer analisis formal baseline vs research.
4. Estudio de representacion.
5. Estudio de contexto largo.
6. Solo despues, mas escala o mas arquitectura.

## Bateria de corridas comparables

Construye una suite con manifiesto canonico. Cada corrida debe declarar al
menos:

- `run_id`
- `profile`
- `purpose`
- `representation`
- `max_files`
- `max_windows_train`
- `max_windows_validation`
- `max_windows_test`
- `seed`
- `context_length`
- `comparison_group`
- `expected_budget_class`
- `notes`

Debes soportar por lo menos dos niveles:

- `smoke`: validacion rapida del pipeline
- `full`: corrida ya apta para comparacion metodologica

## Script de benchmark

Implementa un script que pueda correr en dos modos:

- `execute`: ejecuta corridas del manifiesto
- `collect`: consolida resultados ya existentes

Debe soportar filtros tipo:

- `--run-id`
- `--group`
- `--only-smoke`
- `--only-full`
- `--dry-run`

Debe producir artefactos consolidados, por ejemplo:

- `run_manifest_resolved.json`
- `run_status.json`
- `summary.csv`
- `summary.json`
- `summary.md`

## Tabla unica de resultados

La tabla consolidada debe incluir como minimo:

- `run_id`
- `profile`
- `representation`
- `context_length`
- `n_layers`
- `d_model`
- `use_relative_position_bias`
- `n_train_pieces`
- `n_validation_pieces`
- `n_test_pieces`
- `n_train_windows`
- `nll_per_token`
- `perplexity`
- `accuracy`
- `top_3_accuracy`
- `top_5_accuracy`
- `fit_wall_clock_s`
- `evaluation_wall_clock_s`
- `total_wall_clock_s`
- `parameter_count`
- `device`
- `actual_precision`
- `status`
- `notes`

## Analisis formal baseline vs research

Debes dejar lista una primera lectura formal que responda:

- si `research_richer_events` mejora sobre `cpu_baseline`;
- si la mejora aparece en metricas globales y slices;
- si la mejora compensa el costo adicional;
- si la representacion rica aporta mas que el mero aumento de capacidad;
- si la posicion relativa cambia algo sustancial en el rango de contexto actual.

## Capitulo comparativo en `Tesis`

Construye un nuevo capitulo final, idealmente `chapter4`, con secciones como:

1. Objetivo comparativo
2. Costo de desarrollo e infraestructura
3. Costo computacional de entrenamiento e inferencia
4. Comparacion metodologica con HMM finito y HDP-HMM
5. Resultados cuantitativos
6. Interpretabilidad y trazabilidad
7. Discusion final: cuando el costo del transformer si se justifica

Debes dejar tablas base en `chapters/tables/` para:

- `model_comparison_summary`
- `infrastructure_cost_comparison`
- `representation_comparison`

## Delegacion sugerida

### Worker 1: `Melodies`

Ownership:

- `next_token_experiment/benchmarks/*`
- `scripts/*benchmark*`
- `docs/*benchmark*`
- pruebas relacionadas

Responsabilidad:

- manifiesto canonico;
- script de suite;
- agregador;
- tabla resumen;
- documentacion del benchmark.

### Worker 2: `Tesis`

Ownership:

- `chapters/chapter4.tex`
- `chapters/tables/*comparison*.tex`
- ajuste minimo en `itam-thesis.tex`

Responsabilidad:

- capitulo comparativo;
- tablas base;
- marco de costo-beneficio;
- estructura para ir actualizando juicio comparativo.

## Criterios de cierre

No cierres el trabajo como "listo" si todavia no puedes responder, aunque sea de
forma provisional:

- que perfil gana;
- por cuanto gana;
- en que slices gana;
- a que costo gana;
- si esa ganancia justifica la complejidad adicional frente a HDP-HMM.

## Forma esperada de la entrega

1. Cambios concretos en `Melodies`.
2. Cambios concretos en `Tesis`.
3. Validacion ejecutada.
4. Riesgos pendientes.
5. Siguiente paso recomendado.

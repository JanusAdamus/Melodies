# Política de Artefactos y Recursos

## Qué se Versiona

- código fuente en `src/` y `next_token_experiment/`;
- pruebas en `tests/` y `next_token_experiment/tests/`;
- documentación en `docs/`;
- ejemplos pequeños en `examples/`.

## Qué No se Versiona por Defecto

- resultados de corridas;
- checkpoints `.pt`;
- caches y temporales;
- datasets o snapshots de terceros en `external/`.

## Carpetas

- `artifacts/`: contiene salidas locales y debe permanecer fuera del remoto.
- `external/`: contiene recursos recuperables o reinstalables localmente.

## Regla Práctica

Si un archivo se puede regenerar o descargar otra vez, normalmente no debe
entrar al repositorio.

## Referencias Locales a Conservar

Como minimo, conviene preservar estas rutas locales porque ya aparecen en la
documentacion activa o sirven como referencias claras del estado actual:

- `artifacts/next_token_experiment/results/benchmark_suite`
- `artifacts/next_token_experiment/results/cpu_baseline_smoke`
- `artifacts/next_token_experiment/results/cpu_baseline_smoke_v6`
- `artifacts/next_token_experiment/results/research_richer_events_smoke`
- `artifacts/next_token_experiment/results/library_smoke_8_timed`
- `artifacts/outputs/classic_limited_eval_refresh`

## Candidatos Tipicos a Poda Local

Estos artefactos suelen ser buen candidato a borrado local cuando ya existe una
version mas clara o citada:

- variantes intermedias `v1` a `v5` de una misma corrida;
- smoke runs redundantes;
- salidas generadas desde notebooks;
- rutas `tmp/` y `tmp_figs/`;
- carpetas viejas reemplazadas por una version `refresh`.

## Excepción Permitida

Se puede versionar un artefacto puntual si:

- es pequeño;
- aporta evidencia importante;
- no crea problemas de licencias o tamaño;
- queda documentado explícitamente.

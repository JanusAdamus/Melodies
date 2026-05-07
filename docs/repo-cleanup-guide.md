# Guia de limpieza del repo

## Objetivo

Mantener el proyecto navegable sin borrar contexto util.

La regla operativa debe ser:

- pocas rutas activas;
- resultados locales concentrados;
- documentos historicos visibles pero apartados;
- nombres de corridas entendibles;
- ninguna carpeta "misteriosa" sin duenio claro.

## Ruta corta del proyecto

Hoy la ruta corta correcta debe ser esta:

1. `README.md`
2. `docs/index.md`
3. `docs/experiments.md`
4. una de estas dos superficies:
   - `src/`
   - `next_token_experiment/`

Todo lo demas debe leerse como soporte, archivo historico o salida local.

## Superficies activas

### Codigo activo

- `src/`: pipeline principal HMM/HDP-HMM
- `next_token_experiment/`: benchmark de next-token
- `scripts/`: wrappers de ejecucion
- `tests/` y `next_token_experiment/tests/`: pruebas

### Documentacion activa

- `docs/setup-and-reproduction.md`
- `docs/project-structure.md`
- `docs/experiments.md`
- `docs/transformer.md`
- `docs/transformer-benchmark-suite.md`
- `docs/gpu-comparable-execution-plan.md`

### Documentacion de lectura comparativa

- `docs/transformer-comparison-analysis.md`
- `docs/transformer-vs-classical-analysis.md`
- `docs/pdmx-transformer-thesis-report.md`

### Contexto historico

- `docs/reference/`
- `docs/baselines/`
- `notebooks/`

## Politica para `artifacts/`

La carpeta `artifacts/` no debe funcionar como archivo infinito.

### Conservar

- corridas canonicas o citadas en documentos
- resultados usados en tesis
- manifests reproducibles
- summaries agregados

### Candidatos a poda local

- versiones intermedias `v1`, `v2`, `v3`, `v4`, `v5`, `v6` si ya existe una
  corrida consolidada mejor nombrada
- `tmp/`
- `tmp_figs/`
- smoke runs redundantes
- carpetas generadas por notebooks que no alimentan ninguna conclusion actual

### Regla de nombres

Evitar nombres ambiguos como:

- `_smoke_multicorpus`
- `_smoke_multicorpus_v2`
- `analysys_final`
- `smoke_run`
- `smoke_hdp_quiet2`

Preferir nombres de este estilo:

- `library_pitchclass_smoke_2026_05`
- `classic_limited_eval_refresh`
- `transformer_cpu_baseline_smoke`
- `transformer_gpu_comparable_full`

## Politica para `docs/`

No crear un documento nuevo si solo extiende levemente uno existente.

### Crear documento nuevo solo si

- cambia el objetivo;
- cambia la audiencia;
- cambia el protocolo;
- o se necesita una referencia estable y citable.

### Si no, actualizar

- `docs/experiments.md`
- `docs/transformer.md`
- `docs/project-structure.md`

## Politica para scripts

Cada script en `scripts/` debe cumplir al menos una de estas:

- wrapper reproducible de una corrida importante;
- generador de artefactos de tesis;
- benchmark canonico.

Si un script no cumple una de esas, debe absorberse en una CLI o eliminarse.

## Politica para notebooks

Los notebooks deben considerarse exploracion, no interfaz oficial.

Mantener solo notebooks que:

- expliquen una idea importante;
- ayuden a demo;
- o documenten una exploracion no trivial.

## Politica de crecimiento

Antes de agregar una nueva carpeta o familia de corridas, responder:

1. quien es duenio de esta ruta;
2. si es codigo, docs o artefacto local;
3. si reemplaza algo viejo;
4. donde se enlaza desde `README` o `docs/index`;
5. cuando se puede podar.

Si no hay respuesta clara, no deberia nacer una carpeta nueva.

## Limpieza recomendada por etapas

### Etapa 1. Ya

- mantener una sola ruta corta de lectura;
- separar claramente trabajo activo vs historico;
- dejar politica de nombres;
- podar `tmp/` y `tmp_figs/`;
- marcar corridas canonicas.

### Etapa 2. Cuando haya tiempo

- renombrar corridas locales ambiguas;
- consolidar smoke runs redundantes;
- mover resultados realmente citados a una lista canonica;
- podar salidas huerfanas no usadas por docs ni scripts.

### Etapa 3. Antes de tesis final

- congelar un set minimo de resultados defendibles;
- dejar el resto como regenerable o prescindible;
- asegurar que `README` y `docs/index` apunten solo a lo vigente.

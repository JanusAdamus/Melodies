# Estructura del Proyecto

La organizacion del repo sigue una regla simple: codigo versionado arriba,
datos recuperables fuera de Git y artefactos generados concentrados en una sola
zona.

## Arbol Principal

```text
Melodies/
  .github/workflows/       # CI para validar push y pull requests
  artifacts/               # Salidas locales, resultados y temporales
  docs/                    # Documentacion canonicamente mantenida
    baselines/
    reference/
  examples/                # Ejemplos pequenos y reproducibles
  external/                # Datasets y snapshots locales no versionados
  next_token_experiment/   # Experimento de next-token y su suite
  notebooks/               # Exploracion, demos y material historico ligero
    exploratory/
    thesis/
    archive/
  scripts/                 # Wrappers ligeros de ejecucion
  src/                     # Pipeline principal del proyecto
    analysis/
    cli/
    data/
    models/
  tests/                   # Pruebas del pipeline principal
  Makefile                 # Atajos de instalacion, test y demos
  pyproject.toml           # Metadatos del paquete y comandos instalables
  requirements.txt         # Dependencias base conservadas para referencia
  README.md                # Punto de entrada corto
```

## Responsabilidad por Carpeta

- `src/`: implementacion principal del analisis musical con HMM finito y
  HDP-HMM.
- `next_token_experiment/`: experimento acotado de prediccion de siguiente
  token y pruebas asociadas.
- `tests/`: suite del pipeline principal.
- `examples/`: insumos pequenos que permiten validar el pipeline sin depender
  de datasets externos.
- `docs/`: fuente de verdad documental.
- `notebooks/`: exploracion; utiles para trabajar, pero no son la interfaz
  principal del repo.
- `scripts/reproduce_results.py`: entrada minima reproducible del repo.
- `external/`: recursos reinstalables o descargables localmente.
- `artifacts/`: cualquier salida generada que no deba entrar al remoto.

## Ruta Activa Recomendada

Si quieres trabajar sin perder tiempo en carpetas perifericas, quedate en esta
ruta:

1. `README.md`
2. `docs/index.md`
3. `docs/experiments.md`
4. luego elige solo una superficie:
   - `src/`
   - `next_token_experiment/`

Las carpetas `docs/reference/`, `notebooks/` y muchas rutas dentro de
`artifacts/` deben leerse como apoyo o historico, no como punto de entrada.

## Decisiones de Limpieza

- Los comandos de trabajo frecuentes quedaron concentrados en `Makefile`.
- El repositorio ya se puede instalar con `pip install -e .` gracias a
  `pyproject.toml`.
- Las rutas de resultados viven bajo `artifacts/` para mantener limpia la raiz.
- La integracion con GitHub Actions vive en `.github/workflows/`.

## Criterio de Poda

- `artifacts/tmp/` y `artifacts/tmp_figs/` son descartables.
- Las corridas `smoke` redundantes no deben acumularse sin una razon
  documental.
- Los nombres de corridas deben describir corpus, tarea y escala, en vez de
  crecer como variantes opacas.
- Los documentos nuevos deben entrar a `docs/index.md`; si no merecen estar
  ahi, probablemente deban integrarse en otro documento existente.

## Recorrido Recomendado

1. Empieza en [`../README.md`](../README.md).
2. Sigue con [`setup-and-reproduction.md`](setup-and-reproduction.md).
3. Usa [`experiments.md`](experiments.md) para elegir la superficie correcta.
4. Consulta [`transformer.md`](transformer.md) solo si vas a trabajar el
   baseline del transformer.

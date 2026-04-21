# Melodies

Repositorio de tesis para analisis de musica simbolica con dos superficies de
trabajo claramente separadas:

- `src/`: pipeline principal de analisis musical con HMM finito, HDP-HMM,
  reportes y CLIs.
- `next_token_experiment/`: experimento acotado de prediccion de siguiente
  token con HMM, HDP-HMM y un transformer pequeno.

La idea es que el repo se pueda recorrer rapido desde este `README` y que los
detalles vivan en [`docs/`](docs/index.md).

## Inicio Rapido

La forma mas corta de dejar el proyecto operativo es:

```bash
make install
make test
```

Si prefieres hacerlo manualmente:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python -m unittest discover -s tests
python -m unittest discover -s next_token_experiment/tests
```

## Comandos Utiles

```bash
make demo-score
make demo-main
make demo-transformer
```

Tambien quedan disponibles entradas instalables:

```bash
melodies-analyze --input examples/example_score.musicxml --model both
melodies-library --library-dir /ruta/a/musicxml
melodies-multicorpus --include-library /ruta/a/corpus --output-dir artifacts/outputs/multicorpus_batch
melodies-next-token --profile cpu_baseline --run-name smoke
```

## Mapa del Repositorio

```text
Melodies/
  .github/workflows/       # CI para GitHub Actions
  docs/                    # Documentacion canonicamente mantenida
  examples/                # Insumos pequenos y reproducibles
  next_token_experiment/   # Experimento de next-token
  notebooks/               # Exploracion y demos
  scripts/                 # Wrappers ligeros
  src/                     # Pipeline principal
  tests/                   # Suite principal
  external/                # Recursos externos locales
  artifacts/               # Resultados y salidas locales
  Makefile                 # Comandos rapidos de trabajo
  pyproject.toml           # Metadatos de paquete y entry points
```

## Navegacion Recomendada

1. Lee [`docs/index.md`](docs/index.md) para ubicarte.
2. Sigue con [`docs/setup-and-reproduction.md`](docs/setup-and-reproduction.md) si vas a instalar o reproducir.
3. Consulta [`docs/experiments.md`](docs/experiments.md) para distinguir pipeline principal vs experimento next-token.
4. Usa [`docs/transformer.md`](docs/transformer.md) si tu foco es el baseline del transformer.

## Convenciones

- Todo artefacto generado va a `artifacts/`.
- Todo recurso externo recuperable queda en `external/`.
- El repo esta preparado para `pip install -e .`, `make test` y CI en GitHub Actions.
- La licencia final todavia debe confirmarse antes de publicar el repositorio de forma abierta.

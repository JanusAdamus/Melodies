# Melodies

Repositorio de tesis para modelado computacional de musica simbolica, con foco
en cadenas de Markov, HMM finitos, HDP-HMM truncados y un baseline de
prediccion `next-token` con transformer pequeno.

El proyecto esta dividido en dos superficies complementarias:

- `src/`: pipeline principal de analisis musical interpretable.
- `next_token_experiment/`: benchmark acotado para prediccion de siguiente
  token.

La meta no es construir una libreria generalista, sino un codigo de
investigacion reproducible que permita:

- parsear partituras simbolicas;
- extraer secuencias discretas;
- ajustar y comparar HMM finito y HDP-HMM;
- generar tablas, figuras y resúmenes para tesis;
- ejecutar corridas controladas del baseline transformer.

## Inicio Rapido

La forma mas corta de dejar el proyecto operativo es:

```bash
make install
make test
make reproduce-demo
```

Si prefieres hacerlo manualmente:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
pytest tests next_token_experiment/tests
```

## Ejemplo Minimo Reproducible

Para correr una demostracion pequeña sin editar rutas:

```bash
python scripts/reproduce_results.py
```

Eso:

- genera una partitura de ejemplo en `examples/example_score.musicxml`;
- ejecuta el analisis clasico por pieza;
- intenta correr una demo `next-token` si existe `external/library/scores`;
- si ese corpus no existe, falla de forma limpia y solo omite esa parte.

Los outputs de la demo quedan en:

- `artifacts/outputs/reproducibility_demo/`
- `artifacts/next_token_experiment/results/reproducibility_demo_cpu_baseline/`

## Comandos Utiles

```bash
make demo-score
make demo-main
make demo-transformer
make reproduce-demo
```

Tambien quedan disponibles entradas instalables:

```bash
melodies-analyze --input examples/example_score.musicxml --model both
melodies-library --library-dir /ruta/a/musicxml
melodies-multicorpus --include-library /ruta/a/corpus --output-dir artifacts/outputs/multicorpus_pitchclass_batch
melodies-next-token --profile cpu_baseline --run-name smoke
```

## Mapa del Repositorio

```text
Melodies/
  .github/workflows/       # CI para GitHub Actions
  artifacts/               # Resultados locales regenerables
  docs/                    # Documentacion canonicamente mantenida
  examples/                # Insumos pequenos y reproducibles
  next_token_experiment/   # Experimento de next-token
  notebooks/               # Exploracion, demos y material historico ligero
  scripts/                 # Wrappers ligeros
  src/                     # Pipeline principal
  tests/                   # Suite principal
  external/                # Recursos externos locales
  Makefile                 # Comandos rapidos de trabajo
  pyproject.toml           # Metadatos de paquete y entry points
```

## Navegacion Recomendada

1. Lee [`docs/index.md`](docs/index.md) para ubicarte.
2. Sigue con [`docs/setup-and-reproduction.md`](docs/setup-and-reproduction.md) si vas a instalar o reproducir.
3. Consulta [`docs/experiments.md`](docs/experiments.md) para distinguir pipeline principal vs experimento next-token.
4. Usa [`docs/transformer.md`](docs/transformer.md) si tu foco es el baseline del transformer.

Rutas de apoyo:

- [`docs/gpu-comparable-execution-plan.md`](docs/gpu-comparable-execution-plan.md): plan para correr una comparacion GPU realmente comparable.
- [`docs/repo-cleanup-guide.md`](docs/repo-cleanup-guide.md): criterio para mantener el repo limpio y navegable.
- [`docs/reference/`](docs/reference): contexto historico y notas tecnicas largas.

## Datos

El repo no asume que los corpus externos viajen en GitHub.

- coloca datasets externos bajo `external/`;
- usa `examples/` para demos pequenas y seguras;
- trata `artifacts/` como zona de resultados locales regenerables.

Rutas esperadas hoy:

- `external/library/scores`: corpus simbolico pequeño usado en el baseline
  comparable;
- `external/PDMX/mxl`: corpus usado en corridas GPU de escalamiento;
- `examples/example_score.musicxml`: ejemplo minimo incluido en el repo.

## Outputs Generados

El proyecto genera principalmente:

- tablas y figuras del pipeline clasico bajo `artifacts/outputs/`;
- corridas `next-token` bajo `artifacts/next_token_experiment/results/`;
- exportaciones de apoyo a tesis, si se solicitan, bajo
  `artifacts/thesis_export/` por defecto.

## Limitaciones Conocidas

- la comparacion final `HMM` vs `HDP-HMM` vs `Transformer` bajo un mismo
  protocolo `next-token` todavia no esta cerrada;
- algunas corridas GPU actuales deben leerse como escalamiento, no como
  comparacion definitiva contra los modelos clasicos;
- varios documentos en `docs/reference/` conservan contexto historico y no
  representan la interfaz principal del repo;
- los notebooks son material exploratorio y no la ruta oficial de ejecucion.

## Convenciones

- Todo artefacto generado va a `artifacts/`.
- Todo recurso externo recuperable queda en `external/`.
- El repo esta preparado para `pip install -e .[dev]`, `pytest` y CI en GitHub Actions.
- La licencia final todavia debe confirmarse antes de publicar el repositorio de forma abierta.

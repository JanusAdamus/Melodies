# Melodies

Repositorio de tesis para comparar modelos de musica simbolica en tres ejes:
descripcion estructural, prediccion del siguiente evento y costo de ingenieria.
La descripcion de fronteras, segmentos y estados latentes es el objetivo
principal. La NLL de siguiente evento es una tarea secundaria comun. Tiempo,
complejidad y opacidad se reportan por separado, no como un ganador unico.

Las familias principales son HMM finito, HDP-HMM y un Transformer causal
pequeno. El VOMM inspirado en PPM es un control diagnostico opcional: no es
IDyOM ni un aislador causal de memoria. La perplejidad es solo `exp(NLL)` y no
constituye evidencia adicional.

## Inicio rapido

```bash
make install
make test
```

Instalacion manual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python -m unittest discover -s tests
python -m unittest discover -s next_token_experiment/tests
```

## Planificar antes de ejecutar

```bash
melodies-comparacion --plan-only --max-files 12 --run-name audit
```

`--plan-only` escribe configuracion, splits y un plan auditable, y retorna
antes de construir o ajustar modelos. No produce filas de resultados ni afirma
evidencia. `--without-vomm` desactiva el control opcional y
`--structural-annotations annotations.csv` registra la referencia estructural.
Consulta [la guia multidimensional](docs/multidimensional-evaluation.md) para el
protocolo, todos los flags y el contrato de artefactos.

Las divisiones se agrupan por obra canonica. Validacion y test conservan la cola
exacta y puntuan cada evento una vez con BOS como contexto. El soporte comun es
musica+BOS; PAD solo rellena entradas del Transformer y se excluye de NLL,
argmax, top-k y Brier. La inferencia por pares usa obras canonicas, bootstrap
pareado, Wilcoxon y correccion Holm.

Si faltan anotaciones o inferencias estructurales, los artefactos lo declaran
como `not_evaluated`. Una frontera predictiva-costo puede reportarse como
parcial, pero nunca se presenta como una frontera completa de tres ejes.

## Superficies del repositorio

- `src/`: analisis musical y modelos estructurales.
- `Comparacion/`: runner multidimensional, planificacion y artefactos.
- `next_token_experiment/`: datos y Transformer para la tarea predictiva comun.
- `docs/`: documentacion mantenida.
- `tests/` y `next_token_experiment/tests/`: pruebas acotadas.
- `artifacts/`: salidas locales, no resultados versionados.

Comandos adicionales:

```bash
melodies-analyze --input examples/example_score.musicxml --model both
melodies-library --library-dir /ruta/a/musicxml
melodies-multicorpus --include-library /ruta/a/corpus --output-dir artifacts/outputs/multicorpus_batch
melodies-next-token --profile cpu_baseline --run-name smoke
```

## Evidencia y reproduccion

Las pruebas unitarias, demos y smoke tests usan entradas pequenas para validar
software; no son evidencia de tesis. Las corridas canonicas o pesadas, los
conteos de corpus, los tiempos, la cobertura observada y las conclusiones deben
ser ejecutados y documentados por el autor. Este README no afirma que exista ya
una corrida canonica.

Empieza por [docs/index.md](docs/index.md) y despues consulta
[setup-and-reproduction.md](docs/setup-and-reproduction.md),
[experiments.md](docs/experiments.md) y
[multidimensional-evaluation.md](docs/multidimensional-evaluation.md).

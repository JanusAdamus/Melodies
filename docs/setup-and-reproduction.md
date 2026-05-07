# Instalacion y Reproduccion

## Opcion Recomendada

```bash
make install
make test
```

Eso crea `.venv`, instala el proyecto en modo editable y ejecuta ambas suites.

## Opcion Manual

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

## Pruebas

Pipeline principal:

```bash
pytest tests
```

Experimento de siguiente token:

```bash
pytest next_token_experiment/tests
```

Ambas:

```bash
make test
```

## Reproduccion Minima

La entrada mas corta y segura para comprobar que el repo funciona es:

```bash
python scripts/reproduce_results.py
```

Si falta `external/library/scores`, el script ejecuta la demo clasica por pieza
con `examples/example_score.musicxml` y omite la parte `next-token` con un
mensaje claro.

## Ejemplo Pequeno

Genera una partitura de ejemplo:

```bash
make demo-score
```

Luego ejecuta el pipeline principal:

```bash
make demo-main
```

La forma equivalente usando la CLI instalada es:

```bash
melodies-analyze \
  --input examples/example_score.musicxml \
  --obs pitch_class \
  --model both \
  --output-dir artifacts/outputs/demo_compare
```

## Reproduccion del Transformer

Perfil base CPU:

```bash
make demo-transformer
```

Perfil preparado para GPU:

```bash
melodies-next-token \
  --profile gpu_extended \
  --run-name gpu_extended_full
```

Los resultados quedan bajo `artifacts/next_token_experiment/results/`.

## Datos y Recursos Externos

- El proyecto no asume que `external/` viaje en Git.
- Si faltan datasets externos, usa `examples/` o corpus sinteticos para validar
  el pipeline antes de restaurar recursos locales completos.
- Todo resultado regenerable debe quedar en `artifacts/`.

# Guía de Experimentos

El repositorio tiene dos superficies experimentales distintas.

## Ruta Si Eres Nuevo

Si solo quieres entender el repo sin perderte, usa esta secuencia:

1. `README.md`
2. `docs/index.md`
3. este archivo
4. luego elige una sola superficie:
   - `src/`
   - `next_token_experiment/`

## 1. Pipeline Principal en `src/`

Se usa para:

- análisis por pieza;
- catálogo de biblioteca;
- corridas multicorpus;
- comparación HMM finito vs HDP-HMM.

Entradas típicas:

- `python -m src.cli.main`
- `python -m src.cli.library_analysis`
- `python -m src.cli.multicorpus_analysis`
- `melodies-analyze`
- `melodies-library`
- `melodies-multicorpus`

Sus salidas se escriben en `artifacts/outputs/`.

## 2. Experimento en `next_token_experiment/`

Se usa para una comparación acotada de predicción de siguiente token entre:

- HMM finito;
- HDP-HMM truncado;
- Transformer autoregresivo pequeño.

Ahora se divide en dos pistas:

- `cpu_baseline`: comparación conservadora y reproducible.
- `research_richer_events`: primera pista seria para ampliar representación y capacidad del transformer.

Sus resultados se escriben en:

- `artifacts/next_token_experiment/results/`
- Tambien puede correrse via `melodies-next-token`.

## Estado Actual

Hoy la situacion metodologica del repo es esta:

- el pipeline clasico en `src/` ya compara `HMM finito` vs `HDP-HMM`;
- el experimento `next_token_experiment/` ya compara variantes del
  `Transformer`;
- ya existe evidencia nueva de escalamiento GPU para el transformer;
- pero todavia no existe una comparacion final justa entre las tres familias
  bajo el mismo protocolo de evaluacion.

En otras palabras:

- si preguntas "quien gana entre HMM, HDP-HMM y Transformer", la respuesta
  final todavia no esta cerrada;
- si preguntas "que ya sabemos", la respuesta es:
  - `HDP-HMM` supera al `HMM` finito dentro del pipeline clasico actual;
  - el `Transformer` mejora mucho al escalar en GPU dentro de su propia
    familia;
  - aun falta unir ambos mundos con un protocolo compartido.

## Que Falta Para La Comparacion Final

Para cerrar una comparacion defendible de tesis, faltan estas piezas:

1. evaluar `FiniteChordHMM` como predictor `next-token` bajo el mismo split del
   experimento transformer;
2. evaluar `TruncatedHDPHMM` bajo ese mismo protocolo;
3. agregar un perfil `gpu_comparable` donde la GPU cambie solo hardware y no
   corpus, representacion ni arquitectura;
4. consolidar una tabla unica con metricas comparables para:
   - `finite_hmm`
   - `hdp_hmm`
   - `transformer_cpu_comparable`
   - `transformer_gpu_comparable`

Mientras eso no exista, los resultados GPU grandes deben leerse como
`research/scaling`, no como veredicto final contra los modelos clasicos.

## Criterio Metodológico

- El baseline CPU mantiene la comparación conservadora y defendible para tesis.
- El perfil GPU extiende capacidad sin cambiar la tarea ni la representación
  base.
- Las mejoras del transformer deben reportarse contra el baseline histórico, no
  solo contra intuiciones.

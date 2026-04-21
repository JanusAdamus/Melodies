# Protocolo Experimental Minimo

## 1. Alcance

Experimento unico:

- comparacion de prediccion de siguiente token entre HMM finito, HDP-HMM
  truncado y Transformer pequeno.

Queda fuera de alcance en `v1`:

- tareas de generacion libre;
- clasificacion;
- analisis estructural;
- modelado polifonico completo;
- multiples corpus principales;
- busqueda amplia de hiperparametros.

## 2. Corpus

Corpus principal:

- `external/library/scores`

Estado actual observado en el repositorio:

- 69 archivos detectados con extensiones musicales compatibles;
- catalogos previos del repo muestran al menos 1 error de parseo conocido:
  `Mozart_-_Piano_Sonata_No._16_-_Allegro.mxl`.

Regla:

- construir un manifiesto reproducible con archivos incluidos y excluidos.

## 3. Unidad de observacion

Unidad modelada:

- secuencia lineal de eventos musicales discretos por obra.

Preprocesamiento inicial:

- `music21.converter.parse` como lector base;
- `extract_events(..., include_rests=False, prefer_treble=True)`;
- sin transposicion tonal en `v1`;
- sin atributos expresivos;
- sin inyeccion de etiquetas armonicas humanas.

Motivo:

- mantiene el problema en un nivel compatible con HMM, HDP-HMM y Transformer
  sin sesgar el diseno hacia una arquitectura mas rica.

## 4. Representacion

Representacion principal:

- `pitch_class`
- vocabulario musical de 12 simbolos

Mapeo:

- `C, C#, D, D#, E, F, F#, G, G#, A, A#, B`

Representacion alternativa permitida, pero no inicial:

- `pitch_class_duration`
- bins de duracion sugeridos: `0.25, 0.5, 1.0, 2.0, 4.0`

## 5. Tokenizacion

Para `pitch_class`:

- el token musical es el entero `0..11`;
- padding y `BOS` pueden existir solo como detalle interno del Transformer;
- la evaluacion se hace exclusivamente sobre tokens musicales reales.

## 6. Longitud de secuencia

Regla de ventanas:

- longitud maxima: 128 tokens;
- longitud minima retenida: 32 tokens;
- stride de train: 64;
- stride de validacion y test: 128.

Motivo:

- limita costo computacional;
- evita que unas pocas obras largas dominen el entrenamiento;
- mantiene suficiente contexto local para una comparacion sobria.

## 7. Split

Particion fija:

- train: 70%
- validation: 15%
- test: 15%
- semilla: 7

Regla metodologica:

- dividir por grupo de obra canonica, no por archivo.

Motivo:

- el corpus contiene variantes, arreglos y duplicados probables;
- dividir por archivo produciria leakage entre train y test.

## 8. Modelos

### HMM finito

Configuracion recomendada:

- emisiones categoricas sobre el mismo vocabulario;
- grid pequeno: `K in {8, 12, 16}`;
- seleccion por NLL de validacion;
- maximo 50 iteraciones EM;
- tolerancia: `1e-4`.

### HDP-HMM truncado

Configuracion recomendada:

- truncacion: `K_max=20`;
- `n_iters=100`;
- `burn_in=50`;
- misma representacion que los otros modelos;
- seleccion de snapshot por mejor validacion o mejor log-likelihood retenido.

### Transformer pequeno

Configuracion recomendada:

- arquitectura `decoder-only`;
- embeddings discretos sobre el vocabulario musical;
- embeddings posicionales aprendidos;
- `n_layers=3`;
- `n_heads=4`;
- `d_model=128`;
- `ff_dim=256`;
- `dropout=0.1`;
- `lr=3e-4`;
- `batch_size=32`;
- early stopping con paciencia 5 y maximo 25 epocas.

Estimacion de escala:

- alrededor de `0.42M` parametros con `pitch_class`.

## 9. Metricas

Principal:

- `test_nll_per_token`

Secundarias:

- `test_perplexity`
- `test_accuracy`
- `train_wall_clock_s`
- `eval_wall_clock_s`
- `effective_states` para HDP-HMM
- `selected_states` para HMM finito

## 10. Hardware

Objetivo de reproducibilidad:

- CPU-first

Entorno local observado:

- 12 CPUs logicas
- 15 GiB RAM
- sin GPU detectada

Consecuencia:

- cualquier configuracion que requiera GPU para ser viable queda fuera de
  alcance.

## 11. Almacenamiento de resultados

Formato recomendado:

```text
next_token_experiment/results/
  split_manifest.csv
  exclusions.csv
  finite_hmm/
    config.json
    train_log.csv
    validation_metrics.csv
    test_piece_metrics.csv
    test_summary.json
  hdp_hmm/
    config.json
    train_log.csv
    validation_metrics.csv
    test_piece_metrics.csv
    test_summary.json
  transformer/
    config.json
    train_log.csv
    validation_metrics.csv
    test_piece_metrics.csv
    test_summary.json
```

Regla:

- guardar configuracion, split y metricas por separado;
- no sobrescribir resultados sin identificador de corrida;
- registrar exclusiones desde el inicio.

## 12. Criterio de decision despues de la primera comparacion

Solo escalar el Transformer si se cumplen ambas condiciones:

- mejora consistente en `test_nll_per_token`;
- costo computacional compatible con el marco de tesis.

Si eso no ocurre, el Transformer debe quedar como referencia exploratoria y no
como nucleo del trabajo.

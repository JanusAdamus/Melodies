# Experimento Acotado: HMM, HDP-HMM y Transformer Pequeno

## Proposito

Esta carpeta fija el diseno minimo y defendible para una comparacion de
prediccion de siguiente observacion en musica simbolica entre:

- un HMM finito discreto;
- un HDP-HMM truncado;
- un Transformer pequeno autoregresivo.

El objetivo no es abrir una nueva linea amplia de modelado musical, sino
responder una pregunta puntual de tesis con un protocolo sobrio, reproducible
y comparable.

## 1. Pregunta de investigacion precisa

Pregunta propuesta:

> Bajo un corpus unico, una representacion discreta conservadora y un
> presupuesto computacional acotado, un Transformer autoregresivo pequeno
> reduce de manera consistente la perdida predictiva de siguiente token frente
> a un HMM finito y a un HDP-HMM truncado en secuencias de musica simbolica?

Version operativa para la tesis:

- "mejora relevante" se interpretara primero como una reduccion consistente del
  `negative log-likelihood` medio por token en test;
- esa mejora solo se considerara metodologicamente valiosa si no exige un costo
  computacional desproporcionado ni destruye la claridad comparativa con los
  modelos markovianos.

## 2. Tarea exacta a modelar

La tarea unica del experimento sera:

> dado un prefijo discreto de una secuencia musical simbolica, estimar la
> distribucion de probabilidad de la siguiente observacion.

Delimitaciones explicitas:

- No se modelara generacion libre como objetivo principal.
- No se modelara armonizacion.
- No se modelara clasificacion.
- No se modelara estructura formal.
- No se modelara polifonia completa como objeto central.

La unidad de trabajo sera una secuencia lineal de eventos musicales extraida de
cada obra. Para mantener el alcance bajo control, la secuencia se tratara como
un flujo discreto ordenado y comparable entre modelos.

## 3. Representacion principal y alternativa

### Opcion principal recomendada

`pitch_class`

Justificacion metodologica:

- Es la representacion mas conservadora del repositorio actual.
- Ya existe soporte claro en `src/data/observations.py`.
- Tiene vocabulario pequeno y fijo de 12 simbolos.
- Reduce la sparsity y el costo del Transformer.
- Favorece una comparacion justa con HMM y HDP-HMM discretos.
- Mantiene interpretabilidad razonable.

Decision de alcance:

- La version principal del experimento no incorporara octava, dinamica,
  articulacion ni atributos expresivos.
- Tampoco incorporara una rejilla polifonica rica en la primera comparacion.

### Alternativa secundaria permitida

`pitch_class_duration`

Justificacion:

- Introduce una nocion minima de ritmo sin salir del dominio discreto.
- Sigue siendo compatible con HMM/HDP-HMM categoricos.
- Puede servir despues como experimento secundario de robustez.

Condicion:

- Esta alternativa no debe implementarse antes de cerrar y validar la version
  `pitch_class`.

## 4. Metrica principal

Metrica principal:

- `negative log-likelihood` medio por token en test.

Por que conviene:

- Es una metrica probabilistica nativa para los tres modelos.
- Evalua directamente la tarea de prediccion de siguiente token.
- Permite comparar modelos con distinta parametrizacion sin reducir todo a una
  decision dura de clase ganadora.
- Es agregable por pieza, por split y por corrida.

Metricas secundarias permitidas:

- `perplexity`, solo como version interpretable del NLL;
- `accuracy` top-1, solo como complemento;
- tiempo de entrenamiento e inferencia;
- tamano del modelo o numero efectivo de estados, como indicador de costo.

No se recomienda introducir mas metricas en la primera iteracion.

## 5. Protocolo minimo pero valido

### Corpus principal

- Usar un solo corpus en la comparacion principal: `external/library/scores`.
- El catalogo previo del repositorio reporta 69 archivos MusicXML/MXL.
- Existe al menos un archivo con error de parseo conocido y debe excluirse por
  regla explicita, no de manera informal.

### Regla de exclusion inicial

- Excluir archivos que no parseen con `music21`.
- Excluir piezas con menos de 32 eventos validos tras el preprocesamiento.
- Registrar cada exclusion en un manifiesto.

### Preprocesamiento v1

- Parseo con `music21` reutilizando `src/data/parsing.py`.
- Secuencia lineal de eventos con `include_rests=False`.
- Preferencia por partes agudas con `prefer_treble=True`.
- Sin transposicion tonal en la version inicial.
- Sin enriquecimiento con metadatos armonicos en la tarea predictiva.

### Ventanas de secuencia

- Split por obra antes de crear ventanas.
- Ventanas de contexto maximo de 128 tokens.
- `stride=64` para train.
- `stride=128` para validation y test.
- Mantener colas mas cortas si tienen al menos 32 tokens.

### Particion

- `train=0.70`
- `validation=0.15`
- `test=0.15`
- La particion debe ser por grupo de obra canonica, no por archivo individual,
  para evitar fuga entre arreglos o duplicados de una misma pieza.

### Presupuesto de modelos

- HMM finito: pequeno grid de estados, por ejemplo `K in {8, 12, 16}`.
- HDP-HMM truncado: truncacion moderada, por ejemplo `K_max=20`.
- Transformer: `decoder-only`, `d_model=128`, `n_layers=3`, `n_heads=4`,
  `ff_dim=256`, alrededor de `0.42M` parametros.

### Criterio de parada

- Transformer: early stopping por NLL de validacion, paciencia de 5 epocas,
  maximo 25 epocas.
- HMM finito: EM con maximo 50 iteraciones y tolerancia minima.
- HDP-HMM truncado: numero fijo y acotado de iteraciones, por ejemplo 100, con
  `burn_in=50`.

### Hardware objetivo

- CPU como objetivo por defecto.
- Entorno local actual observado: 12 CPUs logicas, 15 GiB de RAM, sin GPU
  detectada.
- El protocolo no debe depender de GPU para considerarse exitoso.

## 6. Maximo nivel de complejidad permitido

El experimento debe permanecer dentro de estos limites:

- un solo corpus principal;
- una sola tarea principal;
- una sola representacion principal;
- un solo Transformer pequeno como referencia moderna;
- nada de busquedas amplias de hiperparametros;
- nada de arquitecturas grandes;
- nada de modelado polifonico rico en la primera comparacion;
- nada de extensiones a tareas secundarias antes de cerrar la comparacion base.

## 7. Esqueleto de modulos de codigo

La estructura propuesta es:

```text
next_token_experiment/
  README.md
  PROTOCOL.md
  __init__.py
  config.py
  protocol.py
  schemas.py
  data/
    __init__.py
    reader.py
    preprocess.py
    tokenizer.py
    dataset.py
    validation.py
  models/
    __init__.py
    base.py
    finite_hmm.py
    hdp_hmm.py
    small_transformer.py
  experiment/
    __init__.py
    metrics.py
    splits.py
    storage.py
    runner.py
  tests/
    test_protocol.py
```

## 8. Orden exacto de implementacion

1. Congelar el protocolo en codigo y validar que el alcance quede acotado.
2. Implementar lector de corpus y manifiesto de exclusiones.
3. Implementar preprocesamiento reutilizando `src/data/parsing.py`.
4. Implementar tokenizacion principal `pitch_class`.
5. Implementar segmentacion en ventanas y dataloaders.
6. Ejecutar validaciones basicas del pipeline de datos.
7. Implementar HMM finito predictivo con la misma interfaz experimental.
8. Implementar wrapper experimental del HDP-HMM truncado.
9. Implementar Transformer pequeno `decoder-only`.
10. Correr un smoke test reducido de punta a punta.
11. Ejecutar la primera comparacion formal en test.
12. Solo despues decidir si vale la pena escalar o detenerse.

## Nota importante sobre reutilizacion del repositorio

Esta carpeta no reemplaza el trabajo ya existente en `src/`. Lo reutiliza.

En particular:

- `src/data/parsing.py` y `src/data/observations.py` son la base natural del
  pipeline de datos;
- `src/models/inference.py` y `src/models/hdp_hmm.py` ofrecen una base concreta
  para el baseline no parametrico;
- el HMM armonico actual del repo se mantiene como antecedente relevante, pero
  la comparacion predictiva de tesis requiere una interfaz experimental nueva y
  mas estricta.

# Reporte para tesis: prediccion de siguiente token en PDMX con Transformer

## Proposito del experimento

Este experimento evalua un Transformer autoregresivo pequeno para prediccion de
siguiente token en musica simbolica. La tarea no consiste en clasificar una obra
completa ni en producir una etiqueta armonica global: en cada posicion de una
secuencia musical, el modelo observa un contexto previo de tokens y estima cual
es el siguiente token mas probable.

En las corridas reportadas aqui, el token musical principal es la clase de
altura (`pitch_class`): una de las doce clases cromaticas, con tokens especiales
para inicio de secuencia y padding. Por tanto, una prediccion correcta significa
que el modelo asigno la mayor probabilidad a la siguiente clase de altura real
dentro de la secuencia simbolica. Las metricas `top_3_accuracy` y
`top_5_accuracy` relajan este criterio: miden si la respuesta correcta aparecio
entre las tres o cinco clases mas probables, respectivamente.

## Datos usados

Las corridas usan archivos MusicXML/MXL del corpus local PDMX ubicado en
`external/PDMX/mxl`. El pipeline recorre el corpus, parsea partituras
simbolicas, descarta archivos no procesables y convierte cada pieza en una
secuencia discreta de tokens. Despues divide las piezas por obra en conjuntos de
entrenamiento, validacion y prueba, y construye ventanas autoregresivas de
contexto para entrenar y evaluar el modelo.

La corrida CPU local se ejecuto en esta computadora para tener una referencia
consistente con el entorno actual. Usa el perfil conservador `cpu_baseline`:
modelo mas pequeno, `fp32`, ejecucion determinista, 4 hilos de CPU y presupuesto
acotado. Las corridas GPU usan el perfil `gpu_extended`: modelo mas grande,
`bf16`, CUDA y mayor presupuesto de datos.

| Configuracion | Piezas preparadas | Exclusiones | Tokens totales | Train windows | Val windows | Test windows | Representacion |
|---|---:|---:|---:|---:|---:|---:|---|
| CPU baseline local (`pdmx_cpu_baseline_full_local`) | 63 | 1 | 11,812 | 101 | 12 | 29 | `pitch_class` |
| GPU 4k (`pdmx_gpu_serious_4096`) | n/d | n/d | n/d | 2,117 | 298 | 512 | `pitch_class` |
| GPU 8k (`pdmx_gpu_serious_8192`) | n/d | n/d | n/d | 5,106 | 649 | 708 | `pitch_class` |
| GPU 16k (`pdmx_gpu_final_16384`) | n/d | n/d | n/d | 10,179 | 1,371 | 1,418 | `pitch_class` |
| GPU overnight (`pdmx_gpu_overnight_131072_e120`) | 31,932 | 836 | 8,048,113 | 80,472 | 10,252 | 10,933 | `pitch_class` |

Nota metodologica: el baseline CPU y la corrida GPU grande no tienen el mismo
presupuesto de datos ni la misma capacidad de modelo. La comparacion sirve para
documentar el costo-beneficio del escalamiento en esta computadora, no para
aislar causalmente una sola variable.

## Resultados

| Corrida | Dispositivo | Parametros | Best epoch | Epocas | Early stop | Val PPL | Test PPL | Test acc | Top-3 | Top-5 | Tokens test |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `pdmx_cpu_baseline_full_local` | CPU | 415,872 | 6 | 10 | si | 7.9353 | 8.6971 | 23.65% | 56.59% | 75.92% | 3,285 |
| `pdmx_gpu_serious_4096` | CUDA | 2,145,280 | 17 | 25 | si | 7.8241 | 7.3095 | 27.03% | 60.30% | 81.38% | 60,495 |
| `pdmx_gpu_serious_8192` | CUDA | 2,145,280 | 28 | 36 | si | 6.7616 | 6.6572 | 31.93% | 65.15% | 83.76% | 79,299 |
| `pdmx_gpu_final_16384` | CUDA | 2,145,280 | 36 | 40 | no | 6.1008 | 4.9252 | 46.10% | 73.53% | 87.94% | 157,557 |
| `pdmx_gpu_overnight_131072_e120` | CUDA | 2,145,280 | 30 | 54 | si | 5.3744 | 4.7073 | 47.27% | 74.67% | 88.49% | 1,209,168 |

La corrida GPU grande redujo la perplexity de prueba de 8.6971 a 4.7073 frente
al baseline CPU local, una reduccion relativa de 45.9%. La exactitud top-1 subio
de 23.65% a 47.27%, una mejora absoluta de 23.62 puntos porcentuales. La mejora
en top-5 fue menor pero consistente: de 75.92% a 88.49%.

## Costo computacional

| Corrida | Preprocesamiento | Entrenamiento | Evaluacion | Tiempo total |
|---|---:|---:|---:|---:|
| `pdmx_cpu_baseline_full_local` | 24.66 s | 8.34 s | 0.08 s | 35.41 s |
| `pdmx_gpu_overnight_131072_e120` | 12,622.46 s | 2,098.35 s | 42.77 s | 14,771.15 s |

El resultado GPU no fue gratuito: uso aproximadamente 5.16 veces mas parametros
que el baseline CPU y entreno sobre un conjunto muchisimo mayor. En la corrida
`overnight`, la mayor parte del tiempo total se gasto en preprocesamiento del
corpus, no en entrenamiento del Transformer. Esto es importante para la tesis:
el cuello de botella practico del escalamiento no es solamente la GPU, sino
tambien la preparacion robusta de datos simbolicos reales.

## Lectura del aprendizaje

La evolucion de las corridas muestra un patron claro. Al pasar de presupuestos
pequenos a medianos, el Transformer mejora de manera sustancial: la exactitud de
test sube de 27.03% en la corrida 4k a 46.10% en la corrida 16k. Sin embargo, al
escalar de 16k a la corrida overnight, la mejora es mas pequena: la exactitud
pasa de 46.10% a 47.27% y la perplexity de 4.9252 a 4.7073.

Esto sugiere que el modelo ya estaba extrayendo regularidades musicales utiles,
pero comenzo a mostrar rendimientos decrecientes con la configuracion actual. La
corrida overnight encontro su mejor validacion en la epoca 30 y termino en la
epoca 54 por early stopping. En otras palabras, despues de la epoca 30 el modelo
continuo entrenando, pero la validacion ya no mejoro de forma sostenida.

## Interpretacion musical de la prediccion

Como la representacion es `pitch_class`, el modelo aprende regularidades de
continuacion melodica y armonica reducidas a clases de altura. Por ejemplo, en
una muestra de generacion de la corrida overnight, el prompt contiene el patron:

```text
D, A, G, F#, A, G, F#, A, G, F#, A, D, A, G, F#, A
```

La continuacion greedy generada por el modelo repite principalmente:

```text
G, F#, A, G, F#, A, G, F#, A, ...
```

Este ejemplo ilustra bien que el Transformer aprendio patrones locales fuertes
de recurrencia. La prediccion no debe interpretarse como "composicion completa"
en sentido estetico, sino como una estimacion probabilistica de continuidad
simbolica. El modelo puede capturar motivos repetidos, centros de altura y
transiciones frecuentes, pero con `pitch_class` pierde informacion sobre octava,
duracion, articulacion, voz e instrumentacion.

Por eso, una accuracy de 47.27% es significativa: ante una tarea de 12 clases
musicales posibles, mas tokens especiales, el modelo acierta casi la mitad de
las siguientes clases de altura exactas y coloca la respuesta correcta entre sus
cinco primeras opciones en 88.49% de los casos. Aun asi, esta metrica no equivale
a calidad compositiva; mide ajuste predictivo local.

## Incidencias de ejecucion

El archivo `artifacts/logs/pdmx_gpu_max_65536_e80.log` no contiene una corrida
fallida por bajo rendimiento del modelo, sino un error operativo de Windows:

```text
OSError: [Errno 22] Invalid argument: ...\Melodies\<stdin>
```

El error aparece al usar `multiprocessing` con codigo ejecutado desde `stdin`.
En Windows, los procesos hijos intentan recargar el modulo principal desde una
ruta real; `<stdin>` no es una ruta valida. Por tanto, esa corrida no aporta
metricas de entrenamiento ni evaluacion. Para repetirla debe lanzarse desde un
archivo `.py` o con `python -m ...`, no mediante codigo pipeado.

## Conclusion para la tesis

Los resultados apoyan tres conclusiones. Primero, el baseline CPU local provee
una referencia reproducible y conservadora en esta computadora: modelo pequeno,
entorno determinista y bajo costo. Segundo, el Transformer extendido en GPU
mejora sustancialmente cuando se le da mas corpus y capacidad, especialmente al
pasar de escalas pequenas a medianas. Tercero, la mejora adicional de la corrida
mas grande es moderada frente al aumento de costo, lo que sugiere que futuros
avances probablemente dependan mas de mejorar la representacion musical
(`duration`, metrica, voces, contexto relativo) que de solo aumentar ventanas de
entrenamiento bajo la misma arquitectura.

En terminos de tesis, estos experimentos muestran que la prediccion de siguiente
token es una prueba util para medir aprendizaje secuencial en musica simbolica:
permite comparar modelos por perplexity y accuracy, documentar costo
computacional y separar dos preguntas distintas: si el modelo predice bien el
siguiente evento local, y si esa capacidad se traduce despues en continuaciones
musicalmente convincentes.

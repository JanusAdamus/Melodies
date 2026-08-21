# Comparacion provisional: Transformer GPU vs modelos clasicos

## Estado de la evidencia

Hoy ya existe evidencia nueva del Transformer entrenado en GPU sobre PDMX en
[`pdmx-transformer-thesis-report.md`](pdmx-transformer-thesis-report.md).
Tambien existen resultados clasicos reproducibles para HMM finito y HDP-HMM en
`artifacts/outputs/classic_limited_eval_refresh/analysis/`.

La comparacion entre familias debe hacerse con cuidado porque no es la misma
tarea experimental:

- el Transformer GPU resuelve prediccion autoregresiva de siguiente token;
- los modelos clasicos resuelven modelado latente no supervisado;
- el Transformer reporta `perplexity`, `accuracy` y metricas `top-k`;
- los clasicos reportan `log_likelihood` y complejidad de estados;
- el Transformer GPU corre sobre `external/PDMX/mxl`;
- el experimento clasico limitado corre sobre `external/library/scores`.

Por tanto, no existe todavia una tabla final de "ganador absoluto" con una
metrica unica y justa para las tres familias.

## Lo que si es comparable

Si hay tres ejes comparables para la tesis:

1. Tipo de problema que resuelve cada familia.
2. Costo de infraestructura y de ejecucion.
3. Balance entre poder predictivo, compacidad e interpretabilidad.

## Resumen por familia

| Familia | Tarea | Representacion visible | Corpus reportado | Metricas principales | Hallazgo principal |
|---|---|---|---|---|---|
| HMM finito | Modelado armonico no supervisado | `pitch_class` | `external/library/scores` | `log_likelihood`, estados activos | Sirve como baseline clasico interpretable, pero pierde frente al HDP-HMM en el subconjunto clasico reportado. |
| HDP-HMM truncado | Modelado armonico no supervisado | `pitch_class` | `external/library/scores` | `log_likelihood`, estados efectivos | Mejora sistematicamente al HMM finito con menos estados efectivos promedio. |
| Transformer GPU | Prediccion de siguiente token | `pitch_class` | `external/PDMX/mxl` | `perplexity`, `accuracy`, `top_3`, `top_5` | Escala mejor en calidad predictiva cuando recibe mucho mas corpus y compute. |

## Comparacion interna de los modelos clasicos

Tomando como referencia
`artifacts/outputs/classic_limited_eval_refresh/analysis/analysis_report.md` y
la tabla `analysis.csv`, el resultado clasico mas solido hoy es:

- se evaluaron `6` obras con observacion `pitch_class`;
- el HDP-HMM truncado mejora la log-verosimilitud en `6/6` obras;
- la ganancia media por obra fue `92.970`;
- la media de log-verosimilitud fue `-267.685` para el HMM finito y `-174.715`
  para el HDP-HMM truncado;
- el HMM finito uso `10.833` estados activos promedio;
- el HDP-HMM uso `6.667` estados efectivos promedio.

Si se agrega la tabla `analysis.csv` por observacion reportada, el resumen
derivado queda asi:

- `648` observaciones totales;
- HMM finito: `-2.479` log-verosimilitud media por observacion;
- HDP-HMM truncado: `-1.618` log-verosimilitud media por observacion.

Esta mejora favorece al HDP-HMM dentro de la familia clasica. No debe leerse
como una comparacion directa contra la `perplexity` del Transformer, porque el
objetivo estadistico y el protocolo de evaluacion no son los mismos.

## Comparacion interna del Transformer CPU vs GPU

Tomando como referencia
[`pdmx-transformer-thesis-report.md`](pdmx-transformer-thesis-report.md), el
salto importante ocurre entre el baseline CPU local y la corrida GPU overnight:

| Corrida | Dispositivo | Parametros | Test PPL | Test acc | Top-5 | Tiempo total |
|---|---|---:|---:|---:|---:|---:|
| `pdmx_cpu_baseline_full_local` | CPU | 415,872 | 8.6971 | 23.65% | 75.92% | 35.41 s |
| `pdmx_gpu_overnight_131072_e120` | CUDA | 2,145,280 | 4.7073 | 47.27% | 88.49% | 14,771.15 s |

Lectura cuantitativa:

- la `perplexity` de prueba baja `45.9%`;
- la `accuracy` top-1 sube `23.62` puntos porcentuales;
- el modelo usa `5.16x` mas parametros;
- el tiempo total de corrida crece aproximadamente `417x`;
- el mayor cuello de botella reportado no es solo entrenamiento, sino
  preprocesamiento de corpus simbolico a gran escala.

## Juicio comparativo provisional

La lectura mas honesta hoy es la siguiente:

- si la pregunta es ajuste armonico interpretable con corpus pequeno y sin
  infraestructura GPU, el HDP-HMM truncado es la opcion mas fuerte entre los
  modelos clasicos disponibles;
- si la pregunta es prediccion local de siguiente token con suficiente corpus y
  presupuesto computacional, el Transformer GPU ya muestra una mejora sustancial
  dentro de su propia familia;
- el Transformer GPU no desplaza todavia a los modelos clasicos como evidencia
  "superior" en sentido general, porque el repo aun no contiene una evaluacion
  HMM/HDP-HMM bajo el mismo protocolo `next-token` ni sobre el mismo split de
  PDMX.

## Conclusiones utiles para tesis

Hoy si se puede sostener algo, pero en forma acotada:

1. Dentro de los modelos clasicos, el HDP-HMM truncado domina al HMM finito en
   ajuste probabilistico del subconjunto clasico reportado y lo hace con una
   representacion latente mas compacta.
2. Dentro del track Transformer, el escalamiento a GPU mejora de forma marcada
   la calidad predictiva, pero a un costo computacional muy alto.
3. Entre familias, la conclusion correcta no es "el Transformer ya gano", sino
   "el Transformer GPU gana en prediccion next-token bajo alto presupuesto,
   mientras que el HDP-HMM sigue ganando en compacidad e interpretabilidad bajo
   un protocolo clasico mas barato".

## Siguiente paso para una comparacion justa

La comparacion que realmente falta es una de estas dos:

- correr HMM finito y HDP-HMM sobre el mismo corpus PDMX con un protocolo de
  evaluacion autoregresiva comparable;
- o construir una tarea comun de evaluacion para las tres familias y reportar
  costo, ajuste y trazabilidad bajo el mismo split.

Hasta que eso exista, la comparacion correcta en el manuscrito debe presentarse
como metodologica y costo-beneficio, no como ranking absoluto por una sola
metrica.

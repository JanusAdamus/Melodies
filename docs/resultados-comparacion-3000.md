# Resultados de la comparación sobre 3000 obras

Reporte de la corrida `tesis_3000_gpu_20260823_1941`. Qué se midió, qué salió, qué
todavía no se puede afirmar y cómo llevarlo a la tesis.

Complementa [`multidimensional-evaluation.md`](multidimensional-evaluation.md), que define
el protocolo, y [`sources-and-methods.md`](sources-and-methods.md), que tiene la
bibliografía completa en §4 y la trazabilidad de cada decisión algorítmica.

Fecha de la corrida: 2026-08-23. Fecha del reporte: 2026-08-24.

---

## 1. Estado: la corrida terminó

Las 15 celdas del diseño se completaron y el proceso escribió su reporte final. No quedó
ningún proceso vivo porque el trabajo acabó, no porque se cayera.

- Última línea de `artifacts/tesis_3000_gpu_20260823_1941.err`: `celda data_seed=3,frac=1.0 lista (15/15)`.
- `artifacts/tesis_3000_gpu_20260823_1941.out` cierra con el bloque JSON `"artifacts"`.
- Salidas en `artifacts/Comparacion/tesis_3000_gpu_20260823_1941/`.

La cancelación previa por temperatura no costó trabajo: `checkpoint.jsonl` permitió
reanudar, y la corrida arrancó con `resuming from cache: 3000 done, 0 pending`.

## 2. Qué se corrió

| Elemento | Valor |
|---|---|
| Corpus | PDMX (Long et al., 2025), `external/PDMX/mxl`, 3000 archivos muestreados con `corpus_sample_seed=7` |
| Preprocesado | 2933 piezas preparadas, 67 exclusiones, 693 754 eventos |
| Partición | test 15%, validación 10%, `split_seed=7`; 424 piezas de test, fija en todas las celdas |
| Fracciones de entrenamiento | 0.10, 0.25, 0.50, 0.75, 1.00 |
| Semillas | `data_seeds=(1,2,3)`, `model_seeds=(1,2)` |
| Celdas | 15 (3 semillas de datos × 5 fracciones), 105 filas en `results_raw.csv` |
| Datos en frac=1.0 | 2173 piezas, 500 959 tokens de entrenamiento |
| Modelos | `vomm` (control), `finite_hmm`, `hdp_hmm`, `transformer` |
| Hardware | transformer en CUDA; los tres modelos clásicos en CPU con 12 procesos |

## 3. Resultado predictivo

Perplejidad media en test, menor es mejor (`results_summary.csv`):

| modelo | frac=0.1 | frac=0.25 | frac=0.5 | frac=0.75 | frac=1.0 | mejora 0.1→1.0 |
|---|---|---|---|---|---|---|
| transformer | 7.640 | 7.287 | 6.878 | 6.259 | **5.823** | −23.8% |
| finite_hmm | 7.655 | 7.447 | 7.403 | 7.363 | 7.356 | −3.9% |
| hdp_hmm | 8.223 | 8.033 | 8.018 | 7.956 | 7.968 | −3.1% |
| vomm | 8.756 | 8.496 | 8.376 | 8.195 | 8.022 | −8.4% |

Tres lecturas:

**Hay un cruce en frac=0.1.** Con ~55 mil tokens el transformer (7.640, sd 0.107) y el HMM
finito (7.655, sd 0.041) son indistinguibles. La ventaja del transformer aparece desde
frac=0.25 y crece de forma monótona. El resultado no es «el transformer gana», es «el
transformer gana a partir de cierto volumen de datos».

**Las pendientes son el hallazgo, no el ranking.** Los tres modelos clásicos están en
meseta a 500 mil tokens; el transformer sigue bajando sin señal de saturación. Ese
contraste de pendientes es la afirmación fuerte, y es la forma que predicen los estudios de
curvas de aprendizaje (Cortes et al., 1994; Perlich et al., 2003; Hestness et al., 2017;
Kaplan et al., 2020).

**El HDP-HMM pierde contra el HMM finito en las cinco fracciones**, con una brecha estable
de ~0.6 de perplejidad, muy por fuera de las desviaciones típicas. El prior no paramétrico
no compra capacidad predictiva en este corpus. Hay que reportarlo explícitamente. La
ocupación efectiva medida es de 14 a 17 estados con `hdp_truncation_level=24`, así que la
truncación no lo está limitando: la desventaja es del modelo, no del recorte.

El HDP-HMM no es monótono entre frac=0.75 (7.9557) y frac=1.0 (7.9677), pero la diferencia
(0.012) es menor que la desviación típica (0.027). Es meseta con ruido, no una regresión.

## 4. Costo

Medias en frac=1.0 (`engineering_costs.csv`, `pareto_summary.json`):

| modelo | parámetros | ajuste (s) | evaluación (s) | NLL |
|---|---|---|---|---|
| transformer | 415 872 | 169 | 2.6 | 1.762 |
| hdp_hmm | 863 | 578 | 0.5 | 2.075 |
| finite_hmm | 2 879 | 628 | 0.6 | 1.996 |
| vomm | 101 201 (tabla de conteos) | 41 | 8.4 | 2.082 |

**El transformer domina a los dos HMM en las dos dimensiones a la vez**: mejor NLL y menos
tiempo de ajuste (169 s contra 578 y 628). Esto invierte el argumento habitual de que los
modelos clásicos compran interpretabilidad a cambio de costo. Frente al transformer no hay
tal canje; el canje real es contra `vomm`, que ajusta en 41 s.

La frontera de Pareto parcial contiene los cuatro modelos, porque `vomm` gana en tiempo de
ajuste y los HMM ganan en tiempo de evaluación. Ninguno domina a otro en los tres ejes
(Miettinen, 1999).

**Advertencia obligatoria**: el transformer corrió en GPU y los clásicos en CPU. Los
tiempos de pared no son comparables entre hardware. La afirmación defendible es «bajo la
configuración de hardware disponible», no «el transformer es intrínsecamente más barato».

## 5. Inferencia estadística

`pairwise_comparisons.json` compara `test_nll` pareado por obra, solo en frac=1.0, sobre
414 obras de test con métrica válida en ambos modelos. Wilcoxon de rangos con signo
(Wilcoxon, 1945), corrección de Holm sobre las 6 comparaciones (Holm, 1979), e intervalos
bootstrap del 95% con 10 000 remuestreos y semilla 17 (Efron, 1979). El emparejamiento por
obra es el diseño que recomienda Demšar (2006) para comparar dos modelos sobre el mismo
conjunto de casos.

| comparación | diferencia media | IC 95% | p (Holm) |
|---|---|---|---|
| finite_hmm − hdp_hmm | −0.1071 | [−0.1184, −0.0959] | 2.5e−52 |
| finite_hmm − transformer | +0.2505 | [+0.2367, +0.2641] | 7.1e−68 |
| finite_hmm − vomm | −0.0734 | [−0.0855, −0.0613] | 9.6e−27 |
| hdp_hmm − transformer | +0.3576 | [+0.3401, +0.3750] | 4.2e−68 |
| hdp_hmm − vomm | +0.0336 | [+0.0164, +0.0513] | 3.5e−04 |
| transformer − vomm | −0.3240 | [−0.3382, −0.3100] | 1.1e−68 |

Negativo favorece al primer modelo. Las 6 comparaciones sobreviven a Holm y ningún
intervalo cruza el cero.

Con 414 obras pareadas, valores p diminutos son esperables y no dicen nada sobre la
magnitud. Lo que hay que reportar es el tamaño del efecto con su intervalo, no el valor p.

## 6. Lo que esta corrida NO establece

Cinco límites. Los cinco deben aparecer en la tesis.

### 6.1 El eje estructural no se evaluó

`structural_evaluation.json` dice:

```json
{"status": "not_evaluated", "reason": "missing_structural_annotations_input",
 "missing_input": "structural_annotations_path"}
```

`config.structural_annotations_path` es `None`, así que no hubo anotaciones de referencia
contra las cuales medir fronteras ni particiones. Por lo mismo,
`pareto_summary.json → full_three_axis_frontier` queda en `not_evaluated`.

Consecuencia directa: **el protocolo se diseñó con tres ejes y esta corrida solo instrumenta
dos** (predicción y costo). La pregunta estructural, que el propio
`multidimensional-evaluation.md` declara primaria, sigue sin respuesta empírica. El código
de métricas existe y está probado (`boundary_f1`, `normalized_mutual_information`,
`adjusted_rand_index` en `Comparacion/structural_metrics.py`), pero le falta la entrada.

### 6.2 El HMM finito está topado por la rejilla

`selected_states` es 48.0 en las 30 corridas de `finite_hmm`, en las cinco fracciones. La
rejilla es `finite_hmm_states=(12, 24, 48)`, así que el modelo elige siempre el techo.

Esto es serio para la lectura del §3: la meseta del HMM finito puede ser un artefacto de
capacidad recortada, no una propiedad de la familia de modelos. **No se puede afirmar que
los HMM finitos saturan hasta correr la selección con una rejilla que no se agote**, por
ejemplo `(24, 48, 96, 192)`. Mientras tanto la redacción honesta es «el HMM finito con
K ≤ 48 satura».

Ni el HDP-HMM (14–17 estados efectivos de 24 disponibles) ni el VOMM (orden 4 de un máximo
de 8) tienen este problema: ambos eligen valores interiores.

### 6.3 En frac=1.0 la varianza de datos es cero por construcción

En frac=1.0 las tres filas de cada modelo por `model_seed` son idénticas dígito a dígito:
mismo `n_train_pieces` (2173), mismo `n_train_tokens` (500 959), mismo `test_ppl`. Con el
conjunto de test fijo, usar toda la fracción de entrenamiento hace que `data_seed` no tenga
ningún efecto.

Se verifica con `vomm`, que es determinista y por tanto reporta `std_test_ppl = 0.0` exacto
con `runs=3`.

Consecuencia: la barra de error del último punto de la curva mide solo variación de
`model_seed`, mientras que las de frac<1.0 mezclan variación de datos y de modelo. **No son
del mismo tipo y no deben dibujarse como si lo fueran.** Dos salidas aceptables: usar solo
`model_seed` como fuente de varianza en toda la curva, o anotar el último punto por
separado en el pie de figura.

Esto no afecta al §5: las pruebas están pareadas por obra, no por semilla, así que no
tratan esas filas repetidas como réplicas independientes.

### 6.4 El ranking hdp_hmm contra vomm cambia de signo según la ponderación

Las medias marginales del §3 ponen a `hdp_hmm` (NLL 2.0754) por delante de `vomm` (2.0822).
La comparación pareada por obra del §5 los invierte: `hdp_hmm − vomm = +0.0336`, a favor de
`vomm`, con intervalo que no cruza el cero.

No es un error: son dos estimandos distintos. La media marginal pondera por tokens dentro
de cada corrida; la comparación pareada promedia sin ponderar sobre obras. Un puñado de
obras largas basta para invertir el orden. **Hay que elegir uno de los dos como criterio
declarado y usarlo de forma consistente**, o reportar ambos y explicar la discrepancia. Lo
que no se puede es citar el §3 para el ranking y el §5 para la significancia.

La diferencia es en todo caso pequeña (0.034 nats) frente a la brecha con el transformer
(0.324 nats). Ninguna de las dos lecturas cambia la conclusión principal.

### 6.5 El VOMM cambia de capacidad a mitad de curva

`selected_order` salta de 2 a 4 entre frac=0.5 y frac=0.75, y la tabla de conteos pasa de
~2000 a ~88 000 entradas. La pendiente de `vomm` (−8.4%, la mayor de los tres clásicos)
refleja en parte ese cambio de capacidad, no escalado suave con los datos.

## 7. Cómo reportarlo

Una figura, dos tablas, cinco frases.

**Figura**: `learning_curve.png`, cuatro curvas de perplejidad contra tokens de
entrenamiento, escala logarítmica en el eje x. En el pie: la fuente de varianza de las
barras de error y la advertencia del §6.3.

**Tabla 1**: perplejidad por modelo y fracción, la del §3, con desviación típica.

**Tabla 2**: comparaciones pareadas del §5, con diferencia media, IC 95% y p corregido por
Holm.

**Las frases defendibles:**

1. Con 500 mil tokens de entrenamiento, un transformer decoder-only de 416 mil parámetros
   alcanza perplejidad 5.82 en test, frente a 7.36 del mejor modelo clásico evaluado; la
   diferencia pareada por obra es de 0.25 nats, IC 95% [0.237, 0.264].
2. La ventaja depende del volumen de datos: en la fracción del 10% ambos son
   indistinguibles.
3. Los modelos clásicos evaluados muestran meseta mientras el transformer no la muestra en
   el rango medido, con la salvedad de que el HMM finito estaba topado por la rejilla de
   selección.
4. El prior no paramétrico del HDP-HMM no compra ventaja predictiva sobre el HMM finito en
   este corpus, con una desventaja estable de ~0.6 de perplejidad.
5. En el hardware disponible el transformer ajusta más rápido que ambos HMM, aunque la
   comparación cruza GPU con CPU y no es hardware-neutral.

**Lo que hay que corregir antes de defender**: la rejilla de estados del §6.2, y decidir si
el eje estructural del §6.1 se instrumenta o se retira del marco declarado.

## 8. Reproducir

```powershell
cd C:\Melodies
$name = "tesis_3000_gpu_20260823_1941"
.\.venv\Scripts\python.exe -m Comparacion.cli `
  --run-name $name `
  --corpus-root external\PDMX\mxl `
  --max-files 3000 --corpus-sample-seed 7 `
  --data-seeds 1,2,3 --model-seeds 1,2 `
  --fractions 0.1,0.25,0.5,0.75,1.0 `
  --n-workers 6 --transformer-device cuda `
  --corpus-cache artifacts\corpus_cache_3000.jsonl
```

`--n-workers 12` junto con `--transformer-device cuda` satura los 12 núcleos mientras la
GPU entrena, y fue la causa del sobrecalentamiento que obligó a cancelar el primer intento.
Con 6 la corrida sigue siendo reanudable desde `checkpoint.jsonl` sin perder trabajo.

## 9. Referencias

Este documento cita por autor y año. La bibliografía completa está en
[`sources-and-methods.md`](sources-and-methods.md), §4, que cubre tanto las referencias de
inferencia y optimización del pipeline como las de corpus, modelos, evaluación y
estadística que respaldan este reporte.

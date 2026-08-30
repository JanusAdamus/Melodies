# Resultados auditados: corrida original y cuatro sensibilidades

Reporte generado desde los CSV y JSON de `artifacts/Comparacion/`, no a mano. Cubre la
corrida de la tesis y las cuatro sensibilidades ejecutadas entre el 2026-08-27 y el
2026-08-29, todas con auditoría `passed`.

Distingue tres cosas que no deben mezclarse: el **resultado original**, la **auditoría** que
lo verificó sin reentrenar, y las **sensibilidades**, que son corridas nuevas con un
parámetro cambiado cada una.

Corridas cubiertas:

| Corrida | Qué cambia respecto de la original | Auditoría |
| --- | --- | --- |
| `tesis_3000_gpu_20260823_1941` | — (resultado original) | `passed` |
| `sens_stride128` | `--train-stride 128` (ventanas sin solape) | `passed` |
| `sens_hmm_grid` | `--finite-hmm-states 48,96,144,192` | `passed` |
| `sens_split17` | `--split-seed 17` | `passed` |
| `sens_split29` | `--split-seed 29` | `passed` |

Todas comparten corpus: `external/PDMX/mxl`, `--max-files 3000`,
`--corpus-sample-seed 7`, 2933 piezas preparadas, 67 exclusiones. Las sensibilidades
usan sólo `--fractions 1.0`; la curva completa existe únicamente en la corrida original.

---

## 1. Denominadores

Idénticos en las cinco corridas: **414 obras canónicas** como unidad de comparación
pareada, con `n_pairs = 414` en las seis comparaciones de cada corrida.

Los archivos de prueba varían con la partición (424 en los splits 7, 423 en el 17 y 428 en
el 29) porque el agrupamiento por obra canónica absorbe 10 archivos que son segundas
copias. No hay descartes: `denominator_audit.json` da `status: ok` y
`explanation: canonicalization_only` en las cinco.

Sigue vigente la salvedad del grupo `after mr`, mal formado, registrada en
[`canonicalizacion-revision-2026-08-24.md`](canonicalizacion-revision-2026-08-24.md). Si se
corrige el identificador, el denominador pasa a 415.

---

## 2. Predicción

Perplejidad de prueba en frac=1.0, media sobre 6 corridas (3 semillas de datos × 2 de
modelo; `vomm` es determinista y aporta 3).

| modelo | original | `sens_stride128` | `sens_hmm_grid` | `sens_split17` | `sens_split29` |
| --- | --- | --- | --- | --- | --- |
| transformer | **5.8225** | 6.4394 | 5.8225 | **5.8562** | **5.4518** |
| `finite_hmm` | 7.3560 | 7.3560 | **6.9835** | 7.3072 | 7.0899 |
| `hdp_hmm` | 7.9677 | 7.9677 | 7.9677 | 7.8784 | 7.7204 |
| `vomm` | 8.0220 | 8.0220 | 8.0220 | 8.0818 | 7.6151 |

Las columnas no son comparables entre sí en horizontal salvo dentro de cada sensibilidad
respecto de la original, porque las particiones tienen distinto tamaño de entrenamiento
(500 959, 529 127 y 548 526 tokens).

---

## 3. Comparaciones pareadas por obra

Diferencia de NLL por token, orientada como `modelo_a − modelo_b`; negativo favorece a
`modelo_a`. Intervalo de confianza al 95% por bootstrap, valor p de Wilcoxon con corrección
de Holm, n=414 obras en todas.

| comparación | original | `sens_stride128` | `sens_hmm_grid` | `sens_split17` | `sens_split29` |
| --- | --- | --- | --- | --- | --- |
| `finite_hmm` − `hdp_hmm` | −0.1071 | −0.1071 | −0.1682 | −0.1058 | −0.1118 |
| `finite_hmm` − transformer | +0.2505 | +0.1521 | +0.1894 | +0.2430 | +0.2590 |
| `finite_hmm` − `vomm` | −0.0734 | −0.0734 | −0.1346 | −0.0776 | −0.0670 |
| `hdp_hmm` − transformer | +0.3576 | +0.2591 | +0.3576 | +0.3488 | +0.3708 |
| `hdp_hmm` − `vomm` | +0.0336 | +0.0336 | +0.0336 | +0.0281 | +0.0449 |
| transformer − `vomm` | −0.3240 | −0.2255 | −0.3240 | −0.3207 | −0.3259 |

**Ningún intervalo cruza el cero, en ninguna de las 30 comparaciones.** El p corregido más
alto de todo el conjunto es 3.5e−03 (`hdp_hmm` − `vomm` en `sens_split17`); los demás caen
entre 1e−04 y 1e−69.

El orden pareado es idéntico en las cinco corridas:

    transformer  <  finite_hmm  <  vomm  <  hdp_hmm

---

## 4. El desacuerdo entre media marginal y comparación pareada queda explicado

La sección 6.4 de [`resultados-comparacion-3000.md`](resultados-comparacion-3000.md) dejó
abierto que las dos lecturas ordenan `vomm` y `hdp_hmm` al revés, con la instrucción de
declarar un criterio. Ya no hace falta elegir a ciegas: el mecanismo es medible.

Diferencia `hdp_hmm − vomm` en la corrida original, por cuartil de longitud de la obra
(positivo favorece a `vomm`):

| cuartil | tokens | mediana | media de la diferencia | obras que gana `vomm` |
| --- | --- | --- | --- | --- |
| 1 | 32–77 | 58 | +0.0226 | 56 / 103 |
| 2 | 77–109 | 91 | +0.0650 | 68 / 103 |
| 3 | 109–158 | 124 | +0.0646 | 68 / 103 |
| 4 | 160–3378 | 406 | **−0.0167** | 45 / 105 |

`vomm` gana en las obras cortas y pierde en las largas. De ahí que los dos estimandos
discrepen de forma sistemática y no aleatoria:

| estimando | valor | ordena |
| --- | --- | --- |
| media por obra, sin ponderar | **+0.0336** | `vomm` mejor |
| media ponderada por tokens | **−0.0076** | `hdp_hmm` mejor |

No son resultados en conflicto: responden a preguntas distintas. «¿Qué modelo predice mejor
una obra tomada al azar?» da `vomm`. «¿Qué modelo predice mejor un evento tomado al azar?»
da `hdp_hmm`, porque las obras largas aportan más eventos.

**Corrección a lo escrito en [`parada-curva-rehecha-2026-08-29.md`](parada-curva-rehecha-2026-08-29.md) §3.5.**
Allí se afirmó que el orden entre `vomm` y `hdp_hmm` «no es decidible», apoyándose en que
las medias marginales se invierten en el split 29. Es incorrecto. La comparación pareada
por obra favorece a `vomm` en las **cinco** corridas, con intervalo que nunca cruza el cero
(+0.0336, +0.0336, +0.0336, +0.0281, +0.0449). Lo que varía con la partición es la media
marginal, no el estimando pareado. La afirmación correcta es que **el orden depende de la
unidad de análisis, y cada unidad da una respuesta estable**.

Recomendación para la tesis: declarar la obra como unidad —es la que usan las pruebas de
significancia y la que evita que un puñado de obras largas domine— y reportar la
ponderación por tokens como lectura secundaria, con esta tabla como justificación.

---

## 5. Exposición de ventanas

`sens_stride128` entrena con desplazamiento igual al contexto, de modo que cada evento se
ve una sola vez por época. `training_exposure_audit.json` confirma `mean_exposure: 1.0` y
`non_overlapping: true` en las tres particiones.

| modelo | stride 64 | stride 128 | Δ |
| --- | --- | --- | --- |
| transformer | 5.8225 | 6.4394 | **+10.6%** |
| `finite_hmm` | 7.3560 | 7.3560 | 0 |
| `hdp_hmm` | 7.9677 | 7.9677 | 0 |
| `vomm` | 8.0220 | 8.0220 | 0 |

Los tres clásicos son idénticos dígito a dígito: entrenan sobre secuencias completas y el
desplazamiento no los toca. El efecto es exclusivo del transformer, y su ventaja pareada
sobre `finite_hmm` cae de 0.2505 a 0.1521 de NLL, un 39% menos.

Esto **no invierte ningún orden**, pero obliga a declarar el desplazamiento al reportar la
ventaja del transformer, porque una fracción no despreciable procede del régimen de
entrenamiento y no de la arquitectura.

---

## 6. Capacidad del HMM finito

| corrida | rejilla | K seleccionados | veredicto |
| --- | --- | --- | --- |
| original | 12, 24, 48 | 48 ×30 | `grid_too_small` |
| `sens_stride128` | 12, 24, 48 | 48 ×6 | `grid_too_small` |
| `sens_split17` | 12, 24, 48 | 48 ×6 | `grid_too_small` |
| `sens_split29` | 12, 24, 48 | 48 ×6 | `grid_too_small` |
| `sens_hmm_grid` | 48, 96, 144, 192 | 144, 192, 144, 192, 144, 192 | `grid_too_small` |

Ampliar la rejilla mejora al HMM finito de 7.3560 a 6.9835 de perplejidad, y su ventaja
pareada sobre `hdp_hmm` crece de −0.1071 a −0.1682. La meseta que reportaba la tesis era el
techo de la rejilla.

El eje se cerró sin resolver el óptimo. El razonamiento, los diagnósticos de K y las
consecuencias para el texto están en
[`parada-curva-rehecha-2026-08-29.md`](parada-curva-rehecha-2026-08-29.md).

**Advertencia sobre las particiones.** `sens_split17` y `sens_split29` corrieron con la
rejilla por defecto (12, 24, 48), así que **su `finite_hmm` está topado en K=48**. Sus
cifras miden sensibilidad a la partición, no capacidad, y no deben compararse con las de
`sens_hmm_grid`.

---

## 7. Sensibilidad a la partición

Perplejidad de prueba en frac=1.0, con la variación entre semillas de la misma partición
como referencia de escala:

| modelo | split 7 | split 17 | split 29 | rango | std entre semillas |
| --- | --- | --- | --- | --- | --- |
| transformer | 5.8225 | 5.8562 | 5.4518 | 0.404 | 0.004–0.016 |
| `finite_hmm` | 7.3560 | 7.3072 | 7.0899 | 0.266 | 0.011–0.047 |
| `hdp_hmm` | 7.9677 | 7.8784 | 7.7204 | 0.248 | 0.024–0.032 |
| `vomm` | 8.0220 | 8.0818 | 7.6151 | 0.467 | 0.000 |

La variación entre particiones supera de 10 a 100 veces la variación entre semillas. Las
bandas de error de `learning_curve.png` provienen de las semillas, así que **subestiman la
incertidumbre real en un orden de magnitud**.

Dos confusores que impiden leer la tabla en horizontal: los tokens de entrenamiento
difieren entre particiones (500 959 / 529 127 / 548 526), y el `finite_hmm` de los splits 17
y 29 está topado en K=48. La comparación válida es dentro de cada partición, y ahí el orden
pareado no se mueve (§3).

---

## 8. Costo

Segundos de ajuste en frac=1.0, media por corrida:

| modelo | original | `sens_stride128` | `sens_hmm_grid` | `sens_split17` |
| --- | --- | --- | --- | --- |
| `finite_hmm` | 628 | 628 | 3843 | 595 |
| `hdp_hmm` | 578 | 590 | 611 | 646 |
| transformer | 169 | 116 | 172 | 171 |
| `vomm` | 41 | 39 | 44 | 36 |

El transformer ajusta en menos tiempo que los dos HMM **y** predice mejor, lo que invierte
el canje habitual entre calidad y costo. La advertencia obligatoria: corrió en GPU (RTX
4060) mientras los clásicos corrieron en CPU, así que la comparación es de configuración
disponible, no de complejidad algorítmica.

La rejilla ampliada multiplica por 6.1 el costo del HMM finito (628 → 3843 s) a cambio de
0.37 de perplejidad.

**`sens_split29` se excluye de esta tabla.** Corrió en paralelo con el diagnóstico de
capacidad, así que sus tiempos de reloj están inflados por contención de CPU
(`finite_hmm` 1072 s y transformer 305 s, contra 595 y 171 en `sens_split17` bajo la misma
configuración). Sus cifras de predicción no se ven afectadas —el ajuste es determinista
dadas las semillas— pero sus cifras de costo no son utilizables. Es una limitación
introducida por el modo de ejecución, no por el experimento.

---

## 9. Diagnósticos del HDP-HMM

| corrida | veredicto | estados activos, cadena 1 | cadena 2 | verosimilitud coincide |
| --- | --- | --- | --- | --- |
| `sens_stride128` | `drift_detected` | 13.79 | 16.85 | sí |
| `sens_hmm_grid` | `drift_detected` | 13.79 | 16.85 | sí |
| `sens_split17` | `drift_detected` | 17.37 | 18.75 | sí |
| `sens_split29` | `drift_detected` | 16.74 | 15.48 | sí |

En las cuatro, ninguna cadena acredita estabilidad de ventana y las dos cadenas discrepan
en el número de estados activos, aunque coincidan en verosimilitud. La corrida original es
anterior a la instrumentación y no tiene el archivo.

Consecuencia: **no se puede reportar el número de estados inferidos por el HDP-HMM como
resultado**. La autocorrelación de retardo uno ronda 0.89–0.91, coherente con cadenas que
aún no mezclan. Su perplejidad sí es utilizable; su interpretación estructural no.

---

## 10. Eje estructural: sigue sin evaluarse

`structural_evaluation.json` da `not_evaluated` con razón
`missing_structural_annotations_input` en las cinco corridas, y `pareto_summary.json`
reporta `full_three_axis_frontier: not_evaluated` por el mismo motivo. La frontera de
Pareto disponible es la parcial de predicción y costo, donde los cuatro modelos quedan en
la frontera porque ninguno domina a otro en los tres ejes a la vez.

Faltan dos piezas, no una, según §6.1 de
[`resultados-comparacion-3000.md`](resultados-comparacion-3000.md): las anotaciones de
referencia, y los productores de segmentación en los evaluadores. Nada de lo ejecutado en
esta ronda las provee.

---

## 11. Qué se puede afirmar, y qué no

**Sostenido por las cinco corridas:**

- El transformer gana a los tres clásicos, en las tres particiones, con ambos
  desplazamientos y contra el HMM finito con cualquier K probado. Ningún intervalo cruza el
  cero en 30 comparaciones.
- El orden pareado `transformer < finite_hmm < vomm < hdp_hmm` es estable.
- El transformer es además el más barato de los tres modelos entrenados, con la salvedad de
  GPU contra CPU.

**Sostenido, con la salvedad que se indica:**

- La ventaja del transformer sobre el HMM finito cae 39% con ventanas sin solape. Declarar
  el desplazamiento.
- `vomm` supera a `hdp_hmm` por obra y pierde por token. Declarar la unidad de análisis.

**No se puede afirmar:**

- Que el HMM finito tenga una meseta de capacidad. Tres rejillas, tres veces el máximo
  elegido.
- Cuántos estados infiere el HDP-HMM. Las cadenas no convergen.
- Nada sobre el eje estructural.
- Que las bandas de error de la figura representen la incertidumbre. Miden semillas, y la
  partición manda un orden de magnitud más.

---

## 12. Procedencia

| Corrida | Auditoría |
| --- | --- |
| `tesis_3000_gpu_20260823_1941` | `artifacts/Comparacion/audits/tesis_3000_gpu_20260823_1941/artifact_audit.json` |
| `sens_stride128` | `artifacts/Comparacion/audits/sens_stride128/artifact_audit.json` |
| `sens_hmm_grid` | `artifacts/Comparacion/audits/sens_hmm_grid/artifact_audit.json` |
| `sens_split17` | `artifacts/Comparacion/audits/sens_split17/artifact_audit.json` |
| `sens_split29` | `artifacts/Comparacion/audits/sens_split29/artifact_audit.json` |

Las cinco dan `status: passed`, con `completed_cells` igual a `expected_cells`.

Diagnósticos de capacidad del HMM finito:
`artifacts/diagnostico_finite_hmm_k.json`,
`artifacts/diagnostico_finite_hmm_k_escalon2.json`,
`artifacts/diagnostico_finite_hmm_convergencia.json`.

`curva_rehecha` está incompleta (8 celdas de 15) y **no se cita en este reporte**.

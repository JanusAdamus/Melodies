# Por qué se detuvo la curva rehecha, y qué implica para la tesis

Decisión del 2026-08-29: se detiene `curva_rehecha` en la celda 9 de 15 y no se relanza.
El eje de capacidad del HMM finito se cierra con lo medido hasta aquí.

Este documento existe para que la parada sea una decisión defendible en la defensa de la
tesis y no un hueco. Registra qué se midió, por qué el gasto restante no compraba
conocimiento, y qué puede y qué no puede afirmar el texto a partir de ahora.

---

## 1. Qué se detuvo

| Campo | Valor |
| --- | --- |
| Corrida | `artifacts/Comparacion/curva_rehecha` |
| Configuración | `--finite-hmm-states 48,96,144,192,288 --finite-hmm-max-iterations 200`, 5 fracciones, 3 semillas de datos, 2 de modelo |
| Avance | 8 celdas completas de 15; la 9 se interrumpió a mitad |
| Consumido | 19.9 h de ajuste |
| Restante estimado | 24.0 h |
| Estado en disco | `checkpoint.jsonl` íntegro (44 MB, 8 celdas). Reanudable con el mismo `--run-name` más `--resume` |

No se generó `results_summary.csv`, así que la corrida **no es auditable ni citable**. Lo
que se conserva es materia prima, no resultado.

---

## 2. Por qué se paró: tres razones, en orden de peso

### 2.1 La corrida no podía responder la pregunta que la lanzó

Se lanzó para determinar si el HMM finito tiene una meseta de capacidad, después de que la
corrida de la tesis eligiera K=48 —el máximo de su rejilla— en las 30 selecciones. Una
rejilla es informativa sólo si el K elegido queda por dentro; si toca el máximo, mide el
techo de la rejilla y no el modelo (`Comparacion` codifica esta regla como
`selection_touching_the_maximum_means_a_short_grid_not_a_model_plateau`).

Las selecciones de las 8 celdas completadas:

| frac | K elegidos (data_seed, model_seed) |
| --- | --- |
| 0.10 | 96, 48, 96, 96 |
| 0.25 | 144, 96, **288**, 144 |
| 0.50 | 144, 144, 144, **288** |
| 0.75 | **288**, **288** |
| 1.00 | **288**, **288** |

Recuento: `{48: 1, 96: 4, 144: 5, 288: 6}`.

Las seis selecciones en 288 —el techo— están concentradas en las fracciones altas, y
frac=1.00 es justamente el punto sobre el que descansa la conclusión del capítulo. Las 24 h
restantes habrían terminado en el mismo veredicto `grid_too_small`, en el mismo lugar, que
ya está documentado tres veces sin gastarlas.

### 2.2 Ampliar la rejilla está fuera de presupuesto, y se midió

El escalón siguiente sería incluir K=384. Su costo está medido, no estimado:

| K | segundos por ajuste | tope de iteraciones |
| --- | --- | --- |
| 144 | 1 044 | 100 |
| 192 | 1 609 | 100 |
| 288 | 6 404 | 100 |
| 384 | 11 019 | 100 |

Fuente: `artifacts/diagnostico_finite_hmm_k_escalon2.json`.

El escalado es peor que cuadrático en K. Una curva completa son 30 ajustes de HMM finito;
con K=384 dentro de la rejilla, el eje solo pasa de días a semanas de cómputo. La
restricción es material, y conviene declararla como tal en el texto en vez de presentarla
como una elección metodológica.

### 2.3 El gasto estaba desbalanceado respecto de lo que la tesis compara

Reparto del tiempo de ajuste en las 8 celdas completadas:

| modelo | horas | proporción |
| --- | --- | --- |
| `finite_hmm` | 17.8 | 89% |
| `hdp_hmm` | 1.7 | 9% |
| `transformer` | 0.3 | 2% |
| `vomm` | 0.1 | 1% |

El 89% del presupuesto se iba en refinar la capacidad de un modelo que ya pierde con
claridad contra el transformer en todas las mediciones disponibles, y cuya posición en el
ranking no depende de ese refinamiento. Ninguna de las conclusiones del capítulo cambia de
signo por saber si el óptimo del HMM finito está en 288, 384 o más allá.

---

## 3. Lo que sí quedó establecido, y con qué evidencia

Todo lo de esta sección está corrido, auditado y es citable.

### 3.1 El HMM finito no tiene meseta; la de la tesis era el techo de la rejilla

| Fuente | K | test_ppl |
| --- | --- | --- |
| Corrida de la tesis, K ≤ 48 | 48 | 7.3560 |
| `sens_hmm_grid`, K ≤ 192 | 144 y 192 | 6.9835 |
| Diagnóstico convergido, tope 400 | 192 | 6.9089 |

Tres rejillas sucesivas, tres veces el K máximo elegido. La afirmación defendible es
**«el HMM finito no satura dentro del presupuesto de cómputo disponible»**, no que tenga
una meseta.

### 3.2 El presupuesto de EM estaba mordiendo, y se corrigió

El ajuste usa parada temprana sobre la NLL de validación con retención del mejor parámetro
(`Comparacion/classical_models.py:298`). Llegar al tope significa que la validación seguía
mejorando cuando se acabó el presupuesto.

| K | val_ppl con tope 100 | val_ppl con tope 400 | iteraciones reales |
| --- | --- | --- | --- |
| 96 | 7.0493 | 7.0031 | 140 |
| 192 | 6.8581 | 6.8110 | 139 |

Dos consecuencias:

- El sesgo por presupuesto fue casi constante entre K (−0.0462 y −0.0471), así que el
  **orden** entre valores de K nunca estuvo distorsionado; sí lo estaban los niveles.
- Las iteraciones necesarias **no crecen con K** (140 contra 139). El costo de K grande
  está en el precio por iteración, no en cuántas hacen falta.

`--finite-hmm-max-iterations` (CLI) y `--max-iterations` (diagnóstico) quedaron expuestos
para que el presupuesto sea un parámetro declarado y no un valor oculto en `config.py`.

### 3.3 Parte de la ventaja del transformer venía de la exposición, no de la arquitectura

`sens_stride128`, con ventanas de entrenamiento sin solape:

| modelo | stride 64 | stride 128 | Δ |
| --- | --- | --- | --- |
| transformer | 5.8225 | 6.4394 | +10.6% |
| `finite_hmm` | 7.3560 | 7.3560 | 0 |
| `hdp_hmm` | 7.9677 | 7.9677 | 0 |
| `vomm` | 8.0220 | 8.0220 | 0 |

Los tres clásicos son idénticos dígito a dígito porque entrenan sobre secuencias completas
y el desplazamiento no los toca. Una décima parte de la ventaja del transformer procede de
ver cada evento unas dos veces por época. El orden no cambia; la magnitud sí, y hay que
reportarla.

### 3.4 La barra de error de la figura mide la fuente de variación equivocada

| modelo | split 7 | split 17 | split 29 | rango | std entre semillas |
| --- | --- | --- | --- | --- | --- |
| transformer | 5.8225 | 5.8562 | 5.4518 | 0.404 | 0.004–0.016 |
| `finite_hmm` | 7.3560 | 7.3072 | 7.0899 | 0.266 | 0.011–0.047 |
| `hdp_hmm` | 7.9677 | 7.8784 | 7.7204 | 0.248 | 0.024–0.032 |
| `vomm` | 8.0220 | 8.0818 | 7.6151 | 0.467 | 0.000 |

La variación entre particiones supera de 10 a 100 veces la variación entre semillas. Las
bandas de `learning_curve.png` provienen de las semillas, así que subestiman la
incertidumbre real en un orden de magnitud.

Confusor a declarar: los tokens de entrenamiento difieren entre particiones (500 959 /
529 127 / 548 526), así que las columnas no son comparables entre sí. La comparación válida
es dentro de cada partición.

### 3.5 El orden entre `vomm` y `hdp_hmm` depende de la unidad de análisis

**Corregido el 2026-08-29.** La primera versión de esta sección afirmaba que el orden entre
`vomm` y `hdp_hmm` «no es decidible», leyendo sólo las medias marginales, que se invierten
en el split 29. Al revisar las comparaciones pareadas resultó falso.

La comparación pareada por obra favorece a `vomm` en las cinco corridas, con intervalo que
nunca cruza el cero (+0.0336, +0.0336, +0.0336, +0.0281, +0.0449). Lo que varía con la
partición es la media marginal, no el estimando pareado.

El mecanismo está medido: `vomm` gana en las obras cortas y pierde en el cuartil largo, de
modo que la media por obra (+0.0336, favorece a `vomm`) y la ponderada por tokens (−0.0076,
favorece a `hdp_hmm`) discrepan de forma sistemática. Cada unidad da una respuesta estable;
lo que hay que declarar es la unidad.

Detalle, tabla por cuartiles y recomendación en §4 de
[`resultados-comparacion-auditada.md`](resultados-comparacion-auditada.md).

### 3.6 El HDP-HMM no acredita convergencia

`hdp_chain_diagnostics.json` de `sens_stride128`: veredicto `drift_detected`, y las dos
cadenas discrepan en estados activos (13.8 contra 16.9) aunque coincidan en
verosimilitud. Cualquier lectura del número de estados inferidos por el HDP tiene que
llevar esta salvedad.

---

## 4. Qué implica para el texto de la tesis

### 4.1 Hay que reescribir la sección 6.2

La redacción provisional «el HMM finito con K ≤ 48 satura» describe la rejilla, no el
modelo, y su evidencia está refutada. Sustitución propuesta:

> El HMM finito no alcanza saturación dentro de la rejilla explorada. En tres ampliaciones
> sucesivas (K ≤ 48, K ≤ 192, K ≤ 384) la selección por validación eligió siempre el mayor
> valor disponible, y la perplejidad de prueba mejoró de 7.3560 a 6.9089. El límite es de
> cómputo y no del modelo: el ajuste escala peor que cuadráticamente en K, con 11 019 s por
> ajuste en K=384 frente a 1 044 s en K=144, de modo que una curva completa con K=384
> excede el presupuesto disponible. En consecuencia, el contraste de pendientes se reporta
> para el HMM finito con K ≤ 288 y no se afirma nada sobre su capacidad asintótica.

### 4.2 El contraste de pendientes pierde a uno de sus tres modelos

La afirmación fuerte del capítulo era que los tres modelos clásicos comparten una pendiente
plana frente a la del transformer. El HMM finito ya no la sostiene: su nivel depende de K y
K no está resuelto. Quedan `vomm` y `hdp_hmm` como clásicos con pendiente medida sobre las
cinco fracciones, ordenables entre sí una vez declarada la unidad de análisis (§3.5).

### 4.3 Hay que declarar tres límites nuevos

1. **Exposición.** El transformer pierde 10.6% con ventanas sin solape; los clásicos son
   insensibles. Reportar con qué desplazamiento se entrenó.
2. **Partición.** Las bandas de la figura son de semillas y subestiman la incertidumbre en
   un orden de magnitud. Reportar el rango entre particiones junto a la figura.
3. **Presupuesto de EM.** El HMM finito converge en ~140 iteraciones para K entre 96 y 192;
   la corrida original usó un tope de 100 y quedó a medio camino, con un sesgo de ~0.047 en
   perplejidad de validación.

### 4.4 Lo que no cambia

El transformer gana en las tres particiones, en ambos desplazamientos y frente al HMM
finito con cualquier K probado. Es el resultado más robusto del capítulo y no depende de
nada de lo que quedó abierto. También se sostiene su ventaja en costo, con la advertencia
ya registrada de que corrió en GPU mientras los clásicos corrieron en CPU.

---

## 5. Qué reabriría el eje

Se retoma sólo si aparece una de estas tres condiciones, y ninguna es esperable en el plazo
de la tesis:

- Hardware que baje el ajuste de K=384 por debajo de ~1 h, lo que vuelve la curva completa
  cuestión de días.
- Una implementación del HMM finito con costo mejor que cuadrático en K.
- Una razón sustantiva, no metodológica, para necesitar el óptimo de capacidad del HMM
  finito. Ordenar los modelos no la exige.

Si se retoma, el punto de partida es `curva_rehecha` con `--resume`: las 8 celdas están
guardadas y no se reajustan.

---

## 6. Inventario de artefactos

| Artefacto | Estado | Sirve para |
| --- | --- | --- |
| `Comparacion/tesis_3000_gpu_20260823_1941` | auditada `passed` | resultado original, intacto |
| `Comparacion/sens_stride128` | auditada `passed` | §3.3, exposición de ventanas |
| `Comparacion/sens_hmm_grid` | auditada `passed` | §3.1, rejilla K ≤ 192 |
| `Comparacion/sens_split17` | auditada `passed` | §3.4 y §3.5, partición |
| `Comparacion/sens_split29` | auditada `passed` | §3.4 y §3.5, partición |
| `Comparacion/curva_rehecha` | **incompleta, no citable** | materia prima; reanudable |
| `diagnostico_finite_hmm_k.json` | completo | K=24,48,96, tope 100 |
| `diagnostico_finite_hmm_k_escalon2.json` | completo | K=144,192,288,384, tope 100 |
| `diagnostico_finite_hmm_convergencia.json` | completo | §3.2, tope 400 |

Las cinco corridas auditadas dan `passed`, con 424 archivos que canonicalizan a 414 obras
por agrupamiento y cero descartes. Sigue vigente la salvedad del grupo `after mr`
registrada en `canonicalizacion-revision-2026-08-24.md`.

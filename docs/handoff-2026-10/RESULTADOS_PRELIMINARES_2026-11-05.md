# Resultados preliminares para el borrador de la tesis (2026-11-05)

Para Codex, que corrige el texto en JanusAdamus/Tesis (`C:\Tesis`). Adrián pidió adelantar el
borrador antes de terminar las sensibilidades de PDMX.

**Toda cifra de este documento es preliminar.** Márquela en el borrador, por ejemplo con un
comentario `% PRELIMINAR`, hasta que se publique la versión final de las tablas.

## Archivos

- Tablas `.tex`, cada una con su procedencia: `C:\Melodies\artifacts\tablas_tesis_prelim_20261105_e270a04\`.
  Están generadas desde los artefactos: no se editan a mano ni se copian cifras de memoria.
  Generador: `scripts/generar_tablas_tesis.py` (rama `feat/generador-tesis`, df59895).
- Sección A, por obra: `artifacts/outputs/porobra_heldout_20261104_19e83b9` (commit 19e83b9).
- Sección B, PDMX: `artifacts/Comparacion/tesis_corregida_20261104_9ee8dda` (commit 9ee8dda).
- Reglas: `docs/preregistro-2026-10-26.md`. Manda sobre cualquier texto anterior.

## Qué entra y cómo se mide

Entra sólo la melodía. Los modelos de estados ocultos la explican con estados que no se
observan; la métrica preregistrada es la NLL por nota de la melodía no vista (menor es mejor).
Decide la media por obra, con IC bootstrap por obra y ajuste de Holm por pasos.

## Sección A: un modelo por obra (164 obras: 64 de MuseTrainer y 100 de SymbTr)

| Modelo | NLL media por obra | D.e. entre obras |
|---|---|---|
| HMM finito | 1.496 | 0.473 |
| Bigrama | 1.555 | 0.346 |
| HDP-HMM | 1.606 | 0.340 |
| Unigrama | 2.085 | 0.212 |

| Comparación (A − B) | Diferencia | IC Holm | Veredicto |
|---|---|---|---|
| HDP-HMM − HMM finito | 0.110 | [0.073, 0.146] | es mejor el otro |
| HDP-HMM − Bigrama | 0.051 | [0.024, 0.081] | es mejor el otro |
| HDP-HMM − Unigrama | −0.479 | [−0.516, −0.441] | mejora |

- Convergencia: 32 de 164 obras (19.5 %) con R̂ de log L < 1.10. Ninguna se excluye (Enmienda 6).
- Sólo con esas 32 (no decide): HDP-HMM − HMM finito = 0.059, IC [−0.042, 0.153], **no hay
  evidencia**. Los otros dos veredictos no cambian.
- Sin las 7 obras expuestas y sin las 3 en el tope de K (no deciden): ningún veredicto cambia.
- Los estados del HDP-HMM no se interpretan como inferencia bayesiana sobre K ni se comparan
  con los del finito (Enmienda 6, punto 4). El K de una obra que tocó el tope tampoco se
  interpreta como número de estados.

## Sección B: PDMX (418 obras de prueba, fracción 1.0, 252180 tokens de entrenamiento)

| Modelo | NLL media por obra | IC bootstrap |
|---|---|---|
| Transformador | 1.762 | [1.745, 1.780] |
| HMM finito | 1.779 | [1.760, 1.799] |
| VOMM | 1.890 | [1.867, 1.912] |
| HDP-HMM | 2.028 | [2.014, 2.043] |

| Comparación (A − B) | Diferencia | IC Holm | Veredicto |
|---|---|---|---|
| HMM finito − HDP-HMM | −0.249 | [−0.264, −0.234] | mejora |
| Transformador − HMM finito | −0.017 | [−0.027, −0.008] | mejora |
| HMM finito − VOMM | −0.110 | [−0.122, −0.098] | mejora |
| Transformador − HDP-HMM | −0.266 | [−0.281, −0.251] | mejora |
| HDP-HMM − VOMM | 0.138 | [0.119, 0.157] | es mejor el otro |
| Transformador − VOMM | −0.128 | [−0.145, −0.111] | mejora |

Transformador − HMM finito por nota (secundaria): −0.009, IC [−0.023, 0.006], incluye el cero.

## Qué se puede y qué no se puede escribir

Se puede:

- El HMM finito, con estados ocultos y sólo melodía, supera al VOMM en PDMX y queda a 0.017
  nats del transformador.
- El HDP-HMM queda por debajo del HMM finito en las dos secciones, y por debajo del bigrama
  (sección A) y del VOMM (sección B).

No se puede:

- **Atribuir el resultado del HDP-HMM a la idea del modelo.** Convergió en 19.5 % de las obras
  de la sección A, y en PDMX ninguna fila declara convergencia. Con sólo las convergidas de la
  sección A no hay evidencia contra el finito. Puede ser la inferencia, no el modelo: dígalo.
- **Afirmar que el HMM finito aprovecha más datos.** 1 de 30 filas tocó K = 768: la curva de
  aprendizaje queda bloqueada (Enmienda 14), y el finito puede estar subestimado en esa fila.
- **Usar tiempos o la frontera de Pareto como costo.** No hay medición aislada todavía; la
  auditoría del generador avisa `costos no aislados`.
- **Afirmar que los estados son armonía** sin una prueba: la NLL mide predicción de la melodía.
- **Reutilizar cifras de la versión anterior** (agosto) como resultado: esa corrida usaba todas
  las partes, no la melodía.

## Pendiente antes de que estas cifras sean finales

1. Recálculo independiente: hecho para medias, diferencias, veredictos y conteos titulares, desde
   los datos por pieza y por obra, sin discrepancias mayores que redondeo. Los IC coinciden dentro
   de ±0.005 con otro bootstrap. Sin comprobar: columnas por nota, Wilcoxon, IC de `pdmx_resultados`,
   diagnósticos del HDP-HMM, costos y Pareto. La fila en K = 768 es de la fracción 0.25; en la
   fracción 1.0 el K elegido va de 144 a 288.
2. Revisión del generador de tablas, todavía sin mezclar.
3. Sensibilidades de PDMX (`--train-stride 128`, `--split-seed 17`, `--split-seed 29`), en curso.
   Dejar sus huecos en la sección «Sensibilidad» del capítulo 4.
4. Medición de costo aislada, sin lanzar. Dejar su hueco.
5. Canonicalización: la auditoría marca 6 de 418 grupos por tener varios compositores. Revisados
   por título: 5 son la misma pieza con otro arreglo o ficha (p. ej., Paddy O'Rafferty). El grupo
   `g graziani acc 250 jazz patterns 1 12` junta 3 archivos que no parecen la misma obra (Vivaldi,
   Graziani y «Natale Mix»); causa sin determinar. Separarlos da 420 obras y mueve poco las medias
   de la fracción 1.0: finito 1.7794 → 1.7774, HDP-HMM 2.0280 → 2.0282, transformador 1.7620 →
   1.7614, VOMM 1.8896 → 1.8864; transformador − finito −0.0174 → −0.0160. Cálculo exploratorio sin
   IC. La cifra oficial sigue con 418 obras; declárelo en una nota.

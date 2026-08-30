# Ejecución pendiente de la auditoría de la comparación

Las once tareas del plan
`docs/superpowers/plans/2026-08-24-auditoria-comparacion-final.md` están cerradas.
Las 1 a 8 se implementaron con pruebas; las 9 y 10 se ejecutaron el 2026-08-29 y la 11
produjo [`resultados-comparacion-auditada.md`](resultados-comparacion-auditada.md).

Queda un punto abierto por decisión y no por ejecución: el óptimo de capacidad del HMM
finito, cerrado sin resolver por costo. Ver
[`parada-curva-rehecha-2026-08-29.md`](parada-curva-rehecha-2026-08-29.md).

El documento se conserva porque registra los argumentos de corpus obligatorios y la
advertencia de `--n-workers`, que siguen aplicando a cualquier corrida nueva.

Ninguna corrida nueva sobrescribe `tesis_3000_gpu_20260823_1941`.

## Lo que ya quedó resuelto sin entrenar

| Pregunta | Respuesta | Evidencia |
| --- | --- | --- |
| ¿Se conservan los artefactos originales? | Sí, la auditoría pasa | `artifacts/Comparacion/audits/tesis_3000_gpu_20260823_1941/artifact_audit.json` |
| ¿424 contra 414? | Sólo canonicalización: 10 obras con dos archivos, cero descartes | `denominator_audit.json` |
| ¿El agrupamiento es seguro? | Nueve grupos correctos, uno mal formado (`after mr`) | [`canonicalizacion-revision-2026-08-24.md`](canonicalizacion-revision-2026-08-24.md) |
| ¿La rejilla del HMM fue informativa? | No: las 30 selecciones eligieron el máximo, 48 | [`hmm-grid-sensitivity.md`](hmm-grid-sensitivity.md) |

## Selección de corpus: obligatoria en cada corrida

El corpus por defecto de la configuración (`external/library/scores`) no existe
en esta máquina. Sin estos cuatro argumentos el runner prepara cero piezas y
falla con `At least one prepared piece is required`:

    --corpus-root external\PDMX\mxl
    --max-files 3000 --corpus-sample-seed 7
    --corpus-cache artifacts\corpus_cache_3000.jsonl

Son los mismos de `tesis_3000_gpu_20260823_1941`, así que cada sensibilidad ve
el mismo corpus (2933 piezas, 67 exclusiones) y el cache evita volver a parsear
PDMX. Un cache no se comparte entre distintos `--max-files`.

## Ejecución (tareas 9 y 10): completada el 2026-08-29

Las cuatro sensibilidades se ejecutaron y auditaron. Resultados en
[`resultados-comparacion-auditada.md`](resultados-comparacion-auditada.md).

| Corrida | Qué midió | Auditoría | Veredicto propio |
| --- | --- | --- | --- |
| `sens_stride128` | ventanas sin solape | `passed` | el transformer pierde 10.6%; los clásicos no cambian |
| `sens_hmm_grid` | rejilla 48,96,144,192 | `passed` | `grid_too_small`: eligió 144 y 192 |
| `sens_split17` | partición alterna | `passed` | orden pareado sin cambios |
| `sens_split29` | partición alterna | `passed` | orden pareado sin cambios; costos no utilizables |

Las pruebas previas pasaron (142 en `tests`, 24 en `next_token_experiment/tests`) y el
`--plan-only` quedó en `artifacts/Comparacion/plan_stride128`.

### El eje de la rejilla se cerró sin resolver el óptimo

El escalón siguiente que preveía este documento (`--finite-hmm-states 144,192`) se ejecutó
como diagnóstico y como corrida completa, y siguió dando `grid_too_small`. La curva
rehecha con rejilla hasta 288 se detuvo en la celda 9 de 15 porque tampoco podía cerrarlo.
El razonamiento completo, con los costos medidos que lo justifican, está en
[`parada-curva-rehecha-2026-08-29.md`](parada-curva-rehecha-2026-08-29.md).

Tres diagnósticos quedaron ejecutados, todos citables:

| Archivo en `artifacts/` | Rejilla | Tope de iteraciones | Resultado |
| --- | --- | --- | --- |
| `diagnostico_finite_hmm_k.json` | 24, 48, 96 | 100 | eligió 96, techo vinculante |
| `diagnostico_finite_hmm_k_escalon2.json` | 144, 192, 288, 384 | 100 | eligió 384, techo vinculante |
| `diagnostico_finite_hmm_convergencia.json` | 96, 192 | 400 | convergió en ~140 iteraciones |

El tercero mostró que el tope de 100 iteraciones estaba mordiendo: el ajuste converge en
~140 y el sesgo era de ~0.047 en perplejidad de validación, casi igual para ambos K.
`--finite-hmm-max-iterations` (CLI) y `--max-iterations` (diagnóstico) se agregaron para
que ese presupuesto sea un parámetro declarado.

### Diagnósticos del HDP-HMM (tarea 10, punto 4)

Las cuatro corridas dan `drift_detected` y cadenas que discrepan en estados activos. No se
ampliaron iteraciones: la perplejidad del HDP-HMM es utilizable y su lectura estructural no,
que es lo que el reporte declara. Ampliar iteraciones queda como trabajo futuro sin fecha.

### Advertencia de método, para no repetirla

`sens_split29` corrió en paralelo con un diagnóstico, y sus tiempos de reloj quedaron
inflados por contención de CPU (`finite_hmm` 1072 s contra 595 s en `sens_split17`, misma
configuración). Sus cifras de predicción no se ven afectadas porque el ajuste es
determinista dadas las semillas, pero **sus cifras de costo no son utilizables**. Las
corridas que midan costo deben ejecutarse solas.

## Traspaso a la tesis (tarea 11)

- ~~Generar `docs/resultados-comparacion-auditada.md` desde CSV y JSON, nunca a
  mano.~~ Hecho el 2026-08-29, extraído de los artefactos.
- ~~Distinguir resultado original, auditoría y sensibilidades.~~ Hecho: §1 del
  reporte separa las tres categorías.
- Actualizar la tesis sólo con artefactos cuya auditoría diga `passed`.
- Citar 414 obras con la salvedad del grupo `after mr`, o rehacer la corrida con
  el identificador corregido y citar 415.

# Revisión del agrupamiento por obra canónica

**Corrida revisada:** `tesis_3000_gpu_20260823_1941` (no se modifica).
**Artefacto:** `artifacts/Comparacion/audits/tesis_3000_gpu_20260823_1941/canonicalization_audit.json`.
**Autor de la revisión:** Adrian Janus, con asistencia de Claude Code.
**Fecha:** 2026-08-24.

## Cómo se agrupan las obras

`canonicalize_work_label` normaliza `"<compositor> <título>"`: pasa por
`Path(...).stem`, sustituye `_`, `.` y `-` por espacios, colapsa espacios y quita
tokens de ruido (`easy`, `arrangement`, `solo`, `piano`, …). El resultado es el
`canonical_work_id` que define la unidad de comparación pareada.

## Resultados sobre los 424 archivos de prueba

- 424 archivos, 414 identificadores canónicos;
- 10 grupos con más de un archivo, que absorben exactamente los 10 archivos de
  diferencia;
- 0 descartes por NLL no finita y 0 archivos de prueba sin puntuar;
- 124 archivos pierden un sufijo final por `Path().stem`;
- 6 identificadores genéricos o de una sola palabra (`th`, `chr`, `am)`, …);
- 0 pares de identificadores distintos que colapsen bajo la clave agresiva.

## Grupos revisados uno por uno

Nueve de los diez grupos son duplicados legítimos: mismo título, compositor
`Unknown`, distinta grafía o mayúsculas (`The House In The Glen` /
`The House in the Glen`, `Miss McLeod's`, `Rakish Paddy`, …). Agruparlos es
correcto: evita contar dos veces la misma obra.

**Un grupo está mal formado.** El identificador `after mr` reúne dos piezas
distintas:

| Archivo | Compositor | Título |
| --- | --- | --- |
| `13\49\QmVVYeb3wT1a6e9YRE9iXcjgurS3r6LWixJWNdXmRbjgPy.mxl` | After Mr. Cronin | "Knocknagow" (jig) 1113 |
| `9\47\QmRUY7rHcfuoGrmw9qmTDsHHKSgn2b9p8Ts27CVxuc41ur.mxl` | After Mr. McFadden | "The Lightning Flash" (reel) 1458 |

Causa: `Path().stem` corta desde el último punto, así que
`"After Mr. Cronin Knocknagow…"` se reduce a `after mr`. Son un jig y un reel
diferentes; el promedio por obra los mezcló en una sola observación pareada.

**Consecuencia:** el denominador correcto sería 415 obras, no 414, y una de las
415 observaciones pareadas de la corrida de agosto promedia dos piezas ajenas.
El efecto es de una observación entre cientos, pero la cifra publicada debe
citarse con esta salvedad.

## Decisiones

1. La corrida `tesis_3000_gpu_20260823_1941` no se recalcula ni se edita.
2. `canonicalize_work_label` no cambia dentro de esta auditoría: cambiarlo
   alteraría retroactivamente el agrupamiento de una corrida ya publicada.
3. Cualquier corrección del identificador (no usar `Path().stem` sobre títulos)
   entra como corrida nueva, con nombre y manifiesto propios, y se compara
   contra ésta.
4. La huella melódica queda como diagnóstico. En esta corrida no hay tokens en
   los CSV guardados, así que no aportó evidencia; nunca decide identidad por sí
   sola.

## Pendiente

- [ ] Decidir si se lanza una corrida nueva con el identificador corregido.
- [ ] Si se lanza, comparar 414 contra 415 obras y verificar que el orden
      predictivo entre modelos no cambie.

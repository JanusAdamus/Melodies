# Sensibilidad de capacidad del HMM finito

## Qué se observó en la corrida de agosto

En `tesis_3000_gpu_20260823_1941` la rejilla fue `(12, 24, 48)` y la selección
por validación eligió **48 estados en las 30 celdas**, es decir, siempre el
máximo de la rejilla.

Eso no es una meseta del modelo. Es una rejilla corta: el criterio de selección
nunca tuvo la oportunidad de rechazar más capacidad. Ninguna afirmación del tipo
"el HMM satura alrededor de 48 estados" puede sostenerse con esta evidencia.

## Regla de decisión

`build_finite_hmm_grid_audit` escribe `finite_hmm_grid_audit.json` en cada
corrida con este veredicto:

| Veredicto | Cuándo | Lectura |
| --- | --- | --- |
| `grid_informative` | la selección deja de tocar el máximo en la mayoría de repeticiones | la rejilla cubrió la capacidad útil |
| `grid_too_small` | la mitad o más de las selecciones caen en el máximo | hay que ampliar la rejilla |
| `grid_limited_by_resources` | igual que arriba, pero con un límite de recursos documentado | el tope es de la máquina, no del modelo |
| `no_selections` | no hubo filas de `finite_hmm` | sin evidencia |

`plateau_claimed` es siempre `false`: este artefacto nunca declara una meseta.

## Escalones preespecificados

1. **Escalón 1:** `48, 72, 96`. Ejecutar primero un piloto de memoria y tiempo
   con una sola celda antes de la corrida completa.
2. **Escalón 2:** sólo si todas las selecciones vuelven a caer en 96, ampliar a
   `144, 192`, otra vez sujeto al piloto de memoria.
3. Si el piloto no cabe en memoria o en tiempo, se registra el límite de
   recursos con su motivo en `finite_hmm_grid_audit.json` y se reporta como
   límite de recursos, nunca como saturación del modelo.

## Cómo ejecutarlo

Piloto (una celda, sin usarlo como evidencia):

    & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name piloto_hmm_grid `
      --finite-hmm-states 48,72,96 --fractions 1.0 --data-seeds 1 --model-seeds 1 --max-files 200

Sensibilidad (corrida nueva, nunca sobre la de agosto):

    & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name sensibilidad_hmm_grid `
      --finite-hmm-states 48,72,96 --fractions 1.0

La rejilla se valida de forma estricta: enteros mayores o iguales a 2, sin
duplicados y en orden creciente.

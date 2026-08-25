# Ejecución pendiente de la auditoría de la comparación

Las tareas 1 a 8 del plan
`docs/superpowers/plans/2026-08-24-auditoria-comparacion-final.md` están
implementadas y con pruebas. Falta lo que sólo puede hacerse en la computadora
principal: las corridas nuevas y el traspaso a la tesis.

Ninguna corrida nueva sobrescribe `tesis_3000_gpu_20260823_1941`.

## Lo que ya quedó resuelto sin entrenar

| Pregunta | Respuesta | Evidencia |
| --- | --- | --- |
| ¿Se conservan los artefactos originales? | Sí, la auditoría pasa | `artifacts/Comparacion/audits/tesis_3000_gpu_20260823_1941/artifact_audit.json` |
| ¿424 contra 414? | Sólo canonicalización: 10 obras con dos archivos, cero descartes | `denominator_audit.json` |
| ¿El agrupamiento es seguro? | Nueve grupos correctos, uno mal formado (`after mr`) | [`canonicalizacion-revision-2026-08-24.md`](canonicalizacion-revision-2026-08-24.md) |
| ¿La rejilla del HMM fue informativa? | No: las 30 selecciones eligieron el máximo, 48 | [`hmm-grid-sensitivity.md`](hmm-grid-sensitivity.md) |

## Ensayo general (tarea 9)

    & '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
    & '.\.venv\Scripts\python.exe' -m unittest discover -s next_token_experiment/tests -v
    git diff --check

Después, un `--plan-only` por cada sensibilidad, con nombre de corrida propio.

## Orden de ejecución (tarea 10)

1. **Exposición de ventanas.** Piloto de una época y luego la sensibilidad:

        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name sens_stride128 `
          --train-stride 128 --fractions 1.0

   Comparar contra `--train-stride 64`. El artefacto es
   `training_exposure_audit.json`.

2. **Rejilla ampliada del HMM.** Piloto de memoria y tiempo, después:

        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name sens_hmm_grid `
          --finite-hmm-states 48,72,96 --fractions 1.0

   Leer `finite_hmm_grid_audit.json`. Si el veredicto vuelve a ser
   `grid_too_small`, ampliar a 144 y 192 sujeto al piloto de memoria.

3. **Particiones adicionales.** Una corrida por semilla de partición:

        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name sens_split17 --split-seed 17 --fractions 1.0
        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name sens_split29 --split-seed 29 --fractions 1.0

   La comparación se hace dentro de cada partición; la variación entre
   particiones se reporta aparte en `split_variation`.

4. **Diagnósticos del HDP-HMM.** Ya salen de cualquier corrida:
   `hdp_trace_SEED.csv` y `hdp_chain_diagnostics.json`. Ampliar iteraciones sólo
   si el veredicto es `drift_detected` o `chains_disagree`.

Para cada corrida:

- guardar salida estándar y de error;
- ejecutar el auditor al terminar:

        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --audit-run artifacts\Comparacion\<corrida>

- copiar la corrida y su manifiesto a almacenamiento persistente;
- no tocar el reporte anterior.

## Traspaso a la tesis (tarea 11)

- Generar `docs/resultados-comparacion-auditada.md` desde CSV y JSON, nunca a
  mano.
- Distinguir resultado original, auditoría y sensibilidades.
- Actualizar la tesis sólo con artefactos cuya auditoría diga `passed`.
- Citar 414 obras con la salvedad del grupo `after mr`, o rehacer la corrida con
  el identificador corregido y citar 415.

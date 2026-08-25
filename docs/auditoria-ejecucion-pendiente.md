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

## Ensayo general (tarea 9)

    & '.\.venv\Scripts\python.exe' -m unittest discover -s tests
    & '.\.venv\Scripts\python.exe' -m unittest discover -s next_token_experiment/tests
    git diff --check

Después, un `--plan-only` por cada sensibilidad, con nombre de corrida propio:

    & '.\.venv\Scripts\python.exe' -m Comparacion.cli --plan-only --run-name plan_stride128 `
      --corpus-root external\PDMX\mxl --max-files 3000 --corpus-sample-seed 7 `
      --corpus-cache artifacts\corpus_cache_3000.jsonl `
      --train-stride 128 --fractions 1.0

## Orden de ejecución (tarea 10)

1. **Exposición de ventanas.**

        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name sens_stride128 `
          --corpus-root external\PDMX\mxl --max-files 3000 --corpus-sample-seed 7 `
          --corpus-cache artifacts\corpus_cache_3000.jsonl `
          --train-stride 128 --fractions 1.0 `
          --n-workers 6 --transformer-device cuda

   Comparar contra `--train-stride 64`. El artefacto es
   `training_exposure_audit.json`.

2. **Rejilla ampliada del HMM.** Primero el piloto de memoria y tiempo:

        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name piloto_hmm_grid `
          --corpus-root external\PDMX\mxl --max-files 400 --corpus-sample-seed 7 `
          --finite-hmm-states 48,72,96 --fractions 1.0 `
          --data-seeds 1 --model-seeds 1 --transformer-max-epochs 1 --transformer-device cuda

   Si pasa, la sensibilidad:

        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name sens_hmm_grid `
          --corpus-root external\PDMX\mxl --max-files 3000 --corpus-sample-seed 7 `
          --corpus-cache artifacts\corpus_cache_3000.jsonl `
          --finite-hmm-states 48,72,96 --fractions 1.0 `
          --n-workers 6 --transformer-device cuda

   Leer `finite_hmm_grid_audit.json`. Si el veredicto vuelve a ser
   `grid_too_small`, siguiente escalón: `--finite-hmm-states 144,192`, otra vez
   con piloto de memoria antes.

3. **Particiones adicionales.** Una corrida por semilla de partición:

        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name sens_split17 `
          --corpus-root external\PDMX\mxl --max-files 3000 --corpus-sample-seed 7 `
          --corpus-cache artifacts\corpus_cache_3000.jsonl `
          --split-seed 17 --fractions 1.0 --n-workers 6 --transformer-device cuda

        & '.\.venv\Scripts\python.exe' -m Comparacion.cli --run-name sens_split29 `
          --corpus-root external\PDMX\mxl --max-files 3000 --corpus-sample-seed 7 `
          --corpus-cache artifacts\corpus_cache_3000.jsonl `
          --split-seed 29 --fractions 1.0 --n-workers 6 --transformer-device cuda

   La comparación se hace dentro de cada partición; la variación entre
   particiones se reporta aparte en `split_variation`.

4. **Diagnósticos del HDP-HMM.** Ya salen de cualquier corrida:
   `hdp_trace_SEED.csv` y `hdp_chain_diagnostics.json`. Ampliar iteraciones sólo
   si el veredicto es `drift_detected` o `chains_disagree`.

`--n-workers 12` junto con `--transformer-device cuda` satura los 12 núcleos
mientras la GPU entrena y provocó el sobrecalentamiento del primer intento de
agosto. Con 6 la corrida sigue siendo reanudable desde `checkpoint.jsonl`.

Si una corrida se interrumpe, se retoma con el mismo `--run-name` más
`--resume`: las celdas ya terminadas se saltan.

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

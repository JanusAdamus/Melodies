# Política de Artefactos y Recursos

## Qué se Versiona

- código fuente en `src/` y `next_token_experiment/`;
- pruebas en `tests/` y `next_token_experiment/tests/`;
- documentación en `docs/`;
- ejemplos pequeños en `examples/`.

## Qué No se Versiona por Defecto

- resultados de corridas;
- checkpoints `.pt`;
- caches y temporales;
- datasets o snapshots de terceros en `external/`.

## Carpetas

- `artifacts/`: contiene salidas locales y debe permanecer fuera del remoto.
- `external/`: contiene recursos recuperables o reinstalables localmente.

## Regla Práctica

Si un archivo se puede regenerar o descargar otra vez, normalmente no debe
entrar al repositorio.

## Excepción Permitida

Se puede versionar un artefacto puntual si:

- es pequeño;
- aporta evidencia importante;
- no crea problemas de licencias o tamaño;
- queda documentado explícitamente.

## Auditoría y Sellado de Corridas

Cada corrida de `Comparacion` se audita en modo sólo lectura antes de citar sus
cifras en la tesis:

    & '.\.venv\Scripts\python.exe' -m Comparacion.cli --audit-run artifacts\Comparacion\<corrida>

El comando nunca escribe dentro de la corrida auditada. Genera, en
`artifacts/Comparacion/audits/<corrida>/`:

- `artifact_manifest.json`: ruta relativa, tamaño y SHA-256 de cada archivo;
- `artifact_audit.json`: estado `passed`, `failed` o `incomplete` por
  comprobación, más conteos de filas, modelos, fracciones, semillas y celdas.

Comprobaciones: presencia de los artefactos obligatorios, JSON legible, celdas
terminadas frente a la rejilla de `config.json`, coincidencia entre
`results_summary.csv` y `results_raw.csv`, correspondencia entre configuración y
filas, y presencia de `run_summary.json` (informativa: las corridas anteriores a
esta auditoría no lo escribían).

Reglas:

- si la carpeta original no aparece, se registra `original_artifacts_not_found`
  y no se reconstruye nada a partir del reporte Markdown;
- la copia inmutable de la corrida va a almacenamiento persistente fuera de Git;
- el manifiesto y la auditoría viven en `artifacts/` porque contienen rutas
  locales; sólo se versionan si se limpian esas rutas.

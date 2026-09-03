# Ejecución del cierre de criterios

## Lo que debe transferirse a la computadora de ejecución

1. Una copia actualizada de `master`.
2. Los cinco directorios originales: corrida principal, `stride128`, `hmm_grid`, `split17` y `split29`.
3. El cache de las 3000 partituras y acceso local al corpus PDMX.
4. Un archivo de tarifas basado en `docs/cost-tariffs.example.json` con valores y fuentes documentados.
5. Un registro basado en `docs/evidence-runs.example.json` con las rutas reales de las cinco corridas.

No copie partituras ni caches al paquete público.

## Preparación

```powershell
git pull origin master
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -e .
```

Antes de medir, cierre entrenamientos, diagnósticos y cargas intensivas ajenas. La declaración `isolated` es parte de la evidencia y no debe utilizarse si otra tarea compite por CPU o GPU.

## Comando único

```powershell
& '.\scripts\run_thesis_closure.ps1' `
  -CorpusRoot 'D:\ruta\PDMX\mxl' `
  -CorpusCache 'D:\ruta\corpus_cache_3000.jsonl' `
  -EvidenceRegistry 'D:\ruta\evidence-runs.json' `
  -Tariffs 'D:\ruta\cost-tariffs.json'
```

El script ejecuta ambas suites, realiza una corrida aislada de la fracción completa con una semilla, calcula escenarios económicos, construye el paquete saneado y genera la matriz formal.

## Resultado esperado

- `artifacts/Comparacion/resource_benchmark_isolated/engineering_costs.csv`;
- `artifacts/Comparacion/resource_benchmark_isolated/cost_scenarios.json`;
- `artifacts/releases/thesis-evidence/` y su ZIP;
- `artifacts/validation/test_report.json`;
- `artifacts/validation/requirements/validation_matrix.json`;
- `artifacts/validation/requirements/validation_matrix.md`.

La primera ejecución puede tardar varias horas. `--resume` sólo evita repetir celdas terminadas; no reanuda un ajuste individual interrumpido. Si la ejecución comparte recursos o se detiene por temperatura, no utilice sus tiempos como evidencia económica: repita el benchmark con un nombre nuevo y condiciones aisladas.

## Publicación posterior

Verifique el ZIP en una copia limpia antes de publicarlo como adjunto de una versión de GitHub. No añada el contenido de `artifacts/` al historial ordinario. La tesis sólo debe actualizarse después de que la matriz y el verificador del paquete hayan terminado sin errores.

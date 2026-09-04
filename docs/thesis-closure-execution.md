# Ejecución del cierre de criterios

## Lo que debe transferirse a la computadora de ejecución

1. Una copia actualizada de `master`.
2. Los cinco directorios originales: corrida principal, `stride128`, `hmm_grid`, `split17` y `split29`.
3. El caché tokenizado de las 3000 partituras que produjo la corrida principal.
4. Un archivo de tarifas basado en `docs/cost-tariffs.example.json` con valores y fuentes documentados.
5. Un registro basado en `docs/evidence-runs.example.json` con las rutas reales de las cinco corridas.
6. La ruta de la corrida principal auditada que contiene la selección de configuraciones.

No copie partituras ni caches al paquete público.

## Preparación

```powershell
git pull origin master
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -e .
```

Antes de medir, cierre entrenamientos, diagnósticos y cargas intensivas ajenas. La declaración `isolated` es parte de la evidencia y no debe utilizarse si otra tarea compite por CPU o GPU.

Obtenga primero la huella portátil del caché. Este valor ignora las rutas de
archivo propias de cada computadora, pero cambia si cambian los tokens, los
metadatos musicales o el conjunto de exclusiones:

```powershell
& '.\.venv\Scripts\python.exe' '.\scripts\fingerprint_corpus_cache.py' `
  --cache 'D:\ruta\corpus_cache_3000.jsonl'
```

Calcule y conserve esta huella a partir del caché archivado con la corrida
principal. Copie el campo `sha256` de la salida; no use `raw_file_sha256`, pues
ese valor puede cambiar si el caché se regenera con otras rutas locales. La
coincidencia posterior sirve como control sólo si el valor esperado se guardó
antes y por separado del caché que se va a medir.

## Comando único

```powershell
& '.\scripts\run_thesis_closure.ps1' `
  -CorpusCache 'D:\ruta\corpus_cache_3000.jsonl' `
  -EvidenceRegistry 'D:\ruta\evidence-runs.json' `
  -Tariffs 'D:\ruta\cost-tariffs.json' `
  -BenchmarkSourceRun 'D:\ruta\corrida_principal' `
  -ExpectedCorpusFingerprint 'PEGUE_AQUI_EL_SHA256_CANONICO'
```

El script ejecuta ambas suites y después repite tres veces cada una de las
cuatro familias con la partición y la configuración elegidas en la corrida
principal. Mide por separado el ajuste y la evaluación, calcula los escenarios
económicos, incorpora el benchmark al paquete saneado y genera la matriz
formal. En el Transformer se fija la arquitectura seleccionada, pero cada
repetición conserva la parada temprana: la corrida original no registró una
época final que pudiera fijarse sin introducir una decisión nueva.

## Resultado esperado

- `artifacts/resource_benchmark/fixed_configuration_split7/resource_benchmark_raw.csv`;
- `artifacts/resource_benchmark/fixed_configuration_split7/resource_benchmark_summary.csv`;
- `artifacts/resource_benchmark/fixed_configuration_split7/resource_benchmark_audit.json`;
- `artifacts/resource_benchmark/fixed_configuration_split7/cost_scenarios.json`;
- `artifacts/releases/thesis-evidence/` y su ZIP;
- `artifacts/validation/test_report.json`;
- `artifacts/validation/requirements/validation_matrix.json`;
- `artifacts/validation/requirements/validation_matrix.md`.

La ejecución puede tardar varias horas y el benchmark no reanuda un ajuste
interrumpido. Si comparte recursos, falla una repetición o se detiene por
temperatura, conserve la salida para diagnóstico, elija un directorio nuevo y
repita el benchmark completo. No presente esos tiempos como evidencia económica.

## Publicación posterior

Verifique el ZIP en una copia limpia antes de publicarlo como adjunto de una versión de GitHub. No añada el contenido de `artifacts/` al historial ordinario. La tesis sólo debe actualizarse después de que la matriz y el verificador del paquete hayan terminado sin errores.

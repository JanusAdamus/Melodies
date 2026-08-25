# Plan de implementación para cerrar la auditoría de la comparación

> **Para agentes o desarrolladores:** HABILIDAD REQUERIDA:
> superpowers:subagent-driven-development o superpowers:executing-plans.
> Ejecutar una tarea por vez y marcar cada casilla sólo después de conservar
> la evidencia indicada.

**Objetivo:** convertir las limitaciones detectadas en la revisión estadística
y de IA en una comparación auditable, sin cambiar retroactivamente los
resultados de la corrida tesis_3000_gpu_20260823_1941.

**Base reproducible:** rama creada desde
b11a2731a6fb3c812dc0bfd0831b7b065a016142. La rama local master divergente no
forma parte de este plan.

**Arquitectura:** primero se recuperan y sellan los artefactos originales.
Después se añaden auditorías puras y mediciones separadas al runner. Las
sensibilidades se ejecutan como corridas nuevas, con nombres y manifiestos
propios; nunca se sobrescribe la corrida de agosto. La tesis sólo se actualiza
cuando una salida estructurada haya pasado la auditoría.

**Tecnologías:** Python 3.12, NumPy, SciPy, pandas, PyTorch, unittest,
PowerShell y Git.

## Restricciones globales

- No editar a mano CSV, JSON, tablas o cifras del reporte.
- No ejecutar la corrida completa antes de aprobar los pilotos.
- Usar TDD: cada cambio de comportamiento comienza con una prueba que falla.
- Conservar la tarea común: BOS es contexto, PAD no se puntúa y cada evento de
  validación y prueba aparece exactamente una vez por modelo.
- Separar costo de selección, ajuste seleccionado, evaluación y memoria.
- No llamar convergencia a una traza estable sin un diagnóstico definido.
- Cada corrida nueva registra commit, configuración, semillas, hardware,
  precisión numérica, exclusiones y hashes de artefactos.

## Fase 0: recuperar la evidencia existente, sin entrenar

### Tarea 1: recuperar y sellar los artefactos primarios

**Archivos:**

- Crear: Comparacion/artifact_audit.py
- Crear: tests/test_artifact_audit.py
- Modificar: Comparacion/cli.py
- Modificar: docs/artifact-policy.md

- [ ] **Paso 1: localizar la carpeta original**

Buscar tesis_3000_gpu_20260823_1941 en la computadora de la corrida sin mover
ni renombrar el original. Si no existe, registrar
original_artifacts_not_found y detener esta tarea; no reconstruir archivos a
partir del reporte Markdown.

- [ ] **Paso 2: escribir pruebas que fallen**

Probar archivo faltante, JSON inválido, corrida no terminada, hash estable y
discrepancia entre filas y resumen.

    & '.\.venv\Scripts\python.exe' -m unittest tests.test_artifact_audit -v

- [ ] **Paso 3: implementar una auditoría de sólo lectura**

El comando python -m Comparacion.cli --audit-run RUTA debe generar, fuera del
original:

- artifact_manifest.json con SHA-256, tamaño y ruta relativa;
- artifact_audit.json con estados passed, failed o incomplete;
- conteos de filas, modelos, fracciones, semillas y celdas;
- correspondencia entre run_summary, configuración y archivos.

Debe revisar results_raw.csv, results_summary.csv, piece_metrics_raw.csv,
engineering_costs.csv, pairwise_comparisons.json, protocol_audit.json,
structural_evaluation.json, pareto_summary.json, config.json,
preprocessing_report.json, exclusions.csv y checkpoint.jsonl.

- [ ] **Paso 4: copiar a almacenamiento persistente**

Conservar una copia inmutable. No añadir PDMX ni artefactos pesados a Git.
Versionar sólo el manifiesto y la auditoría si no revelan rutas personales.

- [ ] **Paso 5: verificar y hacer commit**

    & '.\.venv\Scripts\python.exe' -m unittest tests.test_artifact_audit -v
    git diff --check
    git add Comparacion/artifact_audit.py Comparacion/cli.py tests/test_artifact_audit.py docs/artifact-policy.md
    git commit -m "feat: audit and seal comparison artifacts"

**Criterio de aceptación:** las cifras pueden remontarse a archivos primarios
con hashes, o la pérdida queda registrada sin sustitutos fabricados.

### Tarea 2: reconciliar 424 piezas con 414 obras

**Archivos:**

- Crear: Comparacion/denominator_audit.py
- Crear: tests/test_denominator_audit.py
- Modificar: Comparacion/statistics.py
- Modificar: Comparacion/runner.py

- [ ] Escribir un caso con cuatro archivos, tres obras canónicas y una NLL no
      finita. La salida distingue archivos, obras, variantes agrupadas, obras
      comunes por par y descartes con motivo.
- [ ] Implementar build_denominator_audit desde piece_metrics_raw.csv y los
      artefactos de partición.
- [ ] Generar denominator_audit.json para la corrida original.
- [ ] Comprobar si 424/414 se explica por canonicalización, valores no finitos
      o ambos. No actualizar la tesis antes de obtener la respuesta.
- [ ] Ejecutar pruebas y commit.

    & '.\.venv\Scripts\python.exe' -m unittest tests.test_denominator_audit tests.test_multidimensional_analysis -v
    git add Comparacion/denominator_audit.py Comparacion/statistics.py Comparacion/runner.py tests/test_denominator_audit.py
    git commit -m "feat: make comparison denominators auditable"

### Tarea 3: auditar la canonicalización

**Archivos:**

- Crear: Comparacion/canonicalization_audit.py
- Crear: tests/test_canonicalization_audit.py
- Modificar: next_token_experiment/experiment/splits.py
- Modificar: Comparacion/runner.py

- [ ] Probar títulos genéricos, puntuación distinta, números de movimiento,
      transliteraciones y compositor faltante.
- [ ] Generar canonicalization_audit.json con grupos de varios archivos,
      metadatos vacíos, títulos cercanos separados y colisiones sospechosas.
- [ ] Añadir una huella melódica simple sólo como diagnóstico; no usarla como
      prueba automática de identidad.
- [ ] Revisar manualmente todos los grupos ambiguos de prueba y una muestra
      determinista de entrenamiento. Registrar autor, fecha y justificación.
- [ ] Si cambian asignaciones, conservar intacta la corrida anterior y crear
      una nueva. Si no cambian, conservar el informe.
- [ ] Ejecutar pruebas y commit.

    & '.\.venv\Scripts\python.exe' -m unittest tests.test_canonicalization_audit tests.test_comparacion -v
    git add Comparacion/canonicalization_audit.py Comparacion/runner.py next_token_experiment/experiment/splits.py tests/test_canonicalization_audit.py
    git commit -m "feat: audit canonical work grouping"

## Fase 1: corregir los indicadores de costo

### Tarea 4: separar selección, ajuste final y evaluación

**Archivos:**

- Modificar: Comparacion/classical_models.py
- Modificar: Comparacion/vomm.py
- Modificar: Comparacion/runner.py
- Modificar: next_token_experiment/models/small_transformer.py
- Modificar: tests/test_comparacion.py
- Modificar: tests/test_vomm.py

- [ ] Exigir para cada familia selection_wall_clock_s,
      selected_fit_wall_clock_s, selected_validation_wall_clock_s,
      evaluation_wall_clock_s y total_protocol_wall_clock_s.
- [ ] Instrumentar HMM y HDP-HMM por candidato.
- [ ] Completar el mismo contrato para VOMM.
- [ ] Para el transformador, conservar tiempo por época, validación,
      detención temprana y restauración de la mejor época.
- [ ] Escribir hardware_manifest.json con CPU, GPU, RAM, versiones, procesos,
      precisión numérica y memoria máxima. Usar null con motivo cuando una
      medición no esté disponible.
- [ ] Ejecutar pruebas y commit.

    & '.\.venv\Scripts\python.exe' -m unittest tests.test_comparacion tests.test_vomm -v
    git add Comparacion/classical_models.py Comparacion/vomm.py Comparacion/runner.py next_token_experiment/models/small_transformer.py tests/test_comparacion.py tests/test_vomm.py
    git commit -m "feat: separate model selection and fit costs"

**Criterio de aceptación:** ya no se presenta un único tiempo ambiguo como
costo de entrenar el modelo.

## Fase 2: sensibilidades predictivas

### Tarea 5: medir la exposición de ventanas solapadas

**Archivos:**

- Modificar: next_token_experiment/data/dataset.py
- Modificar: next_token_experiment/config.py
- Modificar: Comparacion/cli.py
- Modificar: Comparacion/runner.py
- Modificar: next_token_experiment/tests/test_protocol.py

- [ ] Probar cuántas veces aparece cada índice como objetivo con
      desplazamientos 64 y 128.
- [ ] Mantener validación y prueba sin solapamiento.
- [ ] Escribir training_exposure_audit.json con objetivos únicos,
      exposiciones totales, multiplicidades, reinicios BOS y ventanas.
- [ ] Exponer --train-stride en la CLI y guardarlo en config.json.
- [ ] Ejecutar un piloto pequeño de una época; no usarlo como evidencia.
- [ ] Ejecutar la sensibilidad principal con partición 7, fracción 1.0 y las
      semillas originales, comparando desplazamientos 64 y 128.
- [ ] Ejecutar pruebas y commit.

    & '.\.venv\Scripts\python.exe' -m unittest next_token_experiment.tests.test_protocol tests.test_comparacion -v
    git add next_token_experiment/data/dataset.py next_token_experiment/config.py next_token_experiment/tests/test_protocol.py Comparacion/cli.py Comparacion/runner.py
    git commit -m "feat: audit transformer training exposure"

### Tarea 6: ampliar la rejilla del HMM finito

**Archivos:**

- Modificar: Comparacion/config.py
- Modificar: Comparacion/cli.py
- Modificar: tests/test_comparacion.py
- Crear: docs/hmm-grid-sensitivity.md

- [ ] Añadir --finite-hmm-states con validación estricta.
- [ ] Ejecutar un piloto de memoria y tiempo con 48, 72 y 96 estados.
- [ ] Si el piloto pasa, ejecutar la sensibilidad con validación independiente.
- [ ] Si todas las selecciones vuelven a elegir 96, no declarar meseta:
      añadir un segundo escalón preespecificado de 144 y 192, sujeto al piloto
      de memoria.
- [ ] Considerar informativa la rejilla sólo cuando la selección deje de tocar
      el máximo en la mayoría de repeticiones o exista un límite de recursos
      documentado. Un límite de recursos no es una meseta del modelo.
- [ ] Ejecutar pruebas y commit.

    & '.\.venv\Scripts\python.exe' -m unittest tests.test_comparacion -v
    git add Comparacion/config.py Comparacion/cli.py tests/test_comparacion.py docs/hmm-grid-sensitivity.md
    git commit -m "feat: parameterize finite HMM capacity sensitivity"

### Tarea 7: repetir la partición

**Archivos:**

- Modificar: Comparacion/config.py
- Modificar: Comparacion/cli.py
- Modificar: Comparacion/runner.py
- Modificar: Comparacion/statistics.py
- Modificar: tests/test_comparacion.py
- Modificar: tests/test_multidimensional_analysis.py

- [ ] Incluir split_seed en cada fila, métrica e identificador de reanudación.
- [ ] Probar que las obras no se traten como observaciones independientes
      entre particiones.
- [ ] Reportar por separado variación entre obras y entre divisiones.
- [ ] Ejecutar primero fracción 1.0, semillas de partición 7, 17 y 29, y las
      dos semillas de modelo.
- [ ] Ampliar la curva sólo si cambia el orden predictivo o el punto de cruce.
- [ ] Ejecutar pruebas y commit.

    & '.\.venv\Scripts\python.exe' -m unittest tests.test_comparacion tests.test_multidimensional_analysis -v
    git add Comparacion/config.py Comparacion/cli.py Comparacion/runner.py Comparacion/statistics.py tests/test_comparacion.py tests/test_multidimensional_analysis.py
    git commit -m "feat: quantify sensitivity to data partitions"

## Fase 3: estabilidad del HDP-HMM

### Tarea 8: conservar trazas y comparar cadenas

**Archivos:**

- Modificar: src/models/hdp_hmm.py
- Modificar: Comparacion/classical_models.py
- Modificar: Comparacion/runner.py
- Crear: Comparacion/hdp_diagnostics.py
- Crear: tests/test_hdp_diagnostics.py
- Modificar: tests/test_hdp_hmm.py

- [ ] Conservar por iteración log-verosimilitud, estados activos y
      hiperparámetros que cambien.
- [ ] Escribir hdp_trace_SEED.csv y un resumen entre cadenas.
- [ ] Reportar estabilidad entre ventanas temprana y tardía, autocorrelación y
      acuerdo entre semillas.
- [ ] Usar diagnostics_inconclusive cuando las cadenas sean demasiado cortas.
- [ ] Ampliar iteraciones sólo si hay deriva o desacuerdo persistente.
- [ ] Ejecutar pruebas y commit.

    & '.\.venv\Scripts\python.exe' -m unittest tests.test_hdp_hmm tests.test_hdp_diagnostics tests.test_comparacion -v
    git add src/models/hdp_hmm.py Comparacion/classical_models.py Comparacion/runner.py Comparacion/hdp_diagnostics.py tests/test_hdp_hmm.py tests/test_hdp_diagnostics.py
    git commit -m "feat: preserve HDP-HMM chain diagnostics"

## Fase 4: corrida y entrega

### Tarea 9: ensayo general

- [ ] Ejecutar ambas suites completas.
- [ ] Ejecutar --plan-only para cada sensibilidad.
- [ ] Confirmar nombres de corrida nuevos.
- [ ] Confirmar espacio, temperatura y límite de procesos.
- [ ] Revisar que ningún artefacto publique rutas personales.

    & '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
    & '.\.venv\Scripts\python.exe' -m unittest discover -s next_token_experiment/tests -v
    git diff --check

### Tarea 10: ejecutar en la computadora principal

Orden:

1. auditoría de la corrida original;
2. auditoría 424/414 y canonicalización;
3. medición de costo separada;
4. sensibilidad de ventanas;
5. rejilla ampliada del HMM;
6. particiones adicionales;
7. diagnósticos y, sólo si hacen falta, extensión del HDP-HMM.

Para cada corrida:

- [ ] guardar salida estándar y de error;
- [ ] vigilar temperatura sin convertirla en una métrica de tesis;
- [ ] ejecutar el auditor al terminar;
- [ ] generar manifiesto SHA-256;
- [ ] copiar artefactos a almacenamiento persistente;
- [ ] no modificar el reporte anterior.

### Tarea 11: transferir resultados a la tesis

- [ ] Generar docs/resultados-comparacion-auditada.md desde CSV y JSON.
- [ ] Distinguir resultado original, auditoría y sensibilidades.
- [ ] Actualizar la tesis sólo con artefactos que hayan pasado.
- [ ] Compilar LaTeX y comprobar referencias, denominadores y cifras.
- [ ] Revisar ambos repositorios antes de fusionar.

## Definición de terminado

El plan termina cuando:

- los artefactos originales están recuperados y sellados, o su pérdida queda
  documentada;
- 424/414 tiene una explicación reproducible;
- la canonicalización tiene un informe de riesgo;
- los costos separan selección, ajuste, evaluación y memoria;
- la exposición solapada tiene una sensibilidad;
- el HMM deja de tocar el borde o se documenta un límite de recursos;
- la ventaja se evalúa en más de una partición;
- el HDP-HMM conserva trazas y una lectura prudente de estabilidad;
- el informe auditado se genera desde artefactos estructurados;
- la tesis no contiene afirmaciones más amplias que esos resultados.

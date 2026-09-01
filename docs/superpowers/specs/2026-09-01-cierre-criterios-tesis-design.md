# Diseño para cerrar los criterios parciales de la tesis

## Propósito

Cerrar, con evidencia verificable y sin reinterpretar retrospectivamente los resultados, cuatro brechas del manuscrito:

1. publicación de los artefactos primarios necesarios para revisar las cifras;
2. medición comparable de recursos y evaluación económica reproducible;
3. validación formal de la solución contra requisitos declarados;
4. evaluación acotada de la viabilidad social del prototipo.

La especificación se escribe antes de implementar los componentes nuevos. No convierte la corrida original en un estudio prerregistrado ni demuestra que el diseño inicial precediera a todo el código. Sí establece una línea de base verificable para la versión final del sistema.

## Estado de la evidencia

- La corrida principal y cuatro sensibilidades están descritas como auditadas, pero sus CSV y JSON primarios no están en `master` ni en esta copia de trabajo.
- La máquina actual conserva artefactos de planificación y una corrida pequeña, no las cinco corridas citadas por la tesis.
- El runner separa selección, ajuste seleccionado, validación y evaluación, y produce un manifiesto de hardware.
- `engineering_costs.csv` declara la memoria máxima como no medida de manera confiable.
- La tesis ya distingue la viabilidad tecnológica observada, la estimación económica condicional y la ausencia de evaluación con usuarios.

La recuperación de los artefactos desde la máquina de ejecución es una dependencia externa. Ningún módulo debe reconstruirlos a partir de tablas, figuras o prosa.

## Alcance y exclusiones

### Incluido

- producir un paquete público saneado de evidencia;
- medir tiempo y memoria con alcances explícitos;
- calcular escenarios monetarios a partir de tarifas proporcionadas y documentadas;
- generar una matriz de validación desde artefactos, no a mano;
- preparar y, si existe autorización institucional, ejecutar una evaluación breve de uso.

### Excluido

- repetir las cinco corridas predictivas completas salvo que la recuperación de artefactos falle;
- estimar impacto ambiental;
- presentar una evaluación pequeña con usuarios como prueba de impacto social general;
- declarar certificación o conformidad con una norma externa;
- reconstruir datos faltantes desde el PDF de la tesis.

## Componente 1: paquete reproducible de evidencia

### Módulo

Crear `Comparacion/evidence_package.py` y pruebas en `tests/test_evidence_package.py`.

### Entrada

Un registro JSON con los nombres y rutas de:

- corrida principal;
- sensibilidad de ventanas sin solapamiento;
- sensibilidad de capacidad del HMM;
- particiones alternativas con semillas 17 y 29.

Cada directorio debe existir y su `artifact_audit.json` debe indicar `passed`. La sensibilidad con semilla 29 se conserva como evidencia predictiva, pero sus tiempos se marcan como no utilizables.

### Salida

Un directorio determinista y un archivo comprimido que contengan, por corrida:

- `config.json`;
- `results_raw.csv` y `results_summary.csv`;
- `piece_metrics_raw.csv`;
- `engineering_costs.csv`;
- `pairwise_comparisons.json`;
- `protocol_audit.json`;
- `structural_evaluation.json`;
- `pareto_summary.json`;
- exclusiones, divisiones y auditorías de denominadores y canonicalización;
- manifiesto de archivos con SHA-256.

El paquete no debe incluir partituras de PDMX, caches del corpus, checkpoints de pesos ni rutas personales. Las rutas absolutas presentes en JSON deben sustituirse por identificadores relativos; el manifiesto debe registrar por separado el hash del archivo saneado. Un diccionario de datos explicará cada columna publicada.

### Interfaz propuesta

```text
python -m Comparacion.cli --export-evidence REGISTRO.json --output DIR
python -m Comparacion.cli --verify-evidence DIR
```

### Criterios de aceptación

- una ruta ausente detiene la exportación;
- una auditoría distinta de `passed` detiene la exportación;
- dos exportaciones de las mismas entradas producen los mismos contenidos y hashes;
- la verificación se ejecuta en una copia limpia sin acceder a las rutas originales;
- ninguna cadena del paquete contiene el directorio personal o la raíz local del corpus;
- las tablas y figuras de la tesis pueden regenerarse desde el paquete, salvo las que estén explícitamente fuera de su alcance.

### Publicación

Publicar el archivo comprimido como adjunto de una versión de GitHub, no dentro del historial ordinario, siempre que la revisión de licencias y rutas sea satisfactoria. Versionar en el repositorio el registro de corridas, el diccionario de datos, el manifiesto público y las instrucciones de verificación.

## Componente 2: medición de recursos y escenarios económicos

### Módulos

Crear:

- `Comparacion/resource_monitor.py`;
- `Comparacion/cost_scenarios.py`;
- `tests/test_resource_monitor.py`;
- `tests/test_cost_scenarios.py`.

### Contrato de medición

`ResourceMonitor` será un administrador de contexto. Durante el bloque medido registrará:

- tiempo de reloj con `perf_counter`;
- máximo de memoria residente del proceso;
- máximo agregado del árbol de procesos cuando pueda medirse;
- máximo de memoria CUDA asignada y reservada cuando corresponda;
- intervalo de muestreo y estado de cada medición.

Las mediciones ausentes conservarán `null` y un motivo. No se sustituirá RAM del sistema por memoria máxima del proceso. Para CUDA se reiniciarán las estadísticas máximas al comenzar cada bloque. Los bloques de selección, ajuste seleccionado y evaluación se medirán por separado.

### Archivo de salida

Ampliar `engineering_costs.csv` con campos cuyo alcance aparezca en el nombre:

```text
selection_peak_process_tree_rss_bytes
selected_fit_peak_process_tree_rss_bytes
evaluation_peak_process_tree_rss_bytes
selection_peak_cuda_allocated_bytes
selected_fit_peak_cuda_allocated_bytes
evaluation_peak_cuda_allocated_bytes
resource_measurement_status
resource_sample_interval_s
```

### Escenarios monetarios

`cost_scenarios.py` recibirá un archivo de tarifas; no tendrá precios codificados. Cada tarifa deberá indicar fuente, fecha, moneda, región, unidad y si representa equipo propio o servicio remoto. El módulo calculará por separado:

- costo de selección y ajuste;
- costo de evaluación;
- costo de un escenario con un número declarado de predicciones;
- puntos de equilibrio entre pares de modelos.

No mezclará segundos, parámetros y NLL en una puntuación única. Tampoco calculará energía o emisiones.

### Corrida mínima necesaria

No es necesario repetir la curva de aprendizaje. Se propone un benchmark aislado sobre la fracción completa y la partición principal:

1. ejecutar cada familia sin cargas concurrentes;
2. conservar los presupuestos de selección ya declarados;
3. registrar una medición completa de selección y ajuste;
4. repetir tres veces sólo el ajuste de la configuración seleccionada y la evaluación para describir variación temporal operativa;
5. documentar hardware, versiones, temperatura o interrupciones y procesos ajenos detectados.

Si el costo de repetir la búsqueda ampliada del HMM resulta desproporcionado, se conservará el tiempo histórico como evidencia de búsqueda y se medirá de nuevo sólo el ajuste seleccionado. Los dos alcances no se presentarán como equivalentes.

### Criterios de aceptación

- las pruebas simulan métricas disponibles y ausentes;
- una medición CPU nunca llena campos CUDA;
- las unidades aparecen en todos los nombres y salidas;
- el cálculo monetario puede reproducirse cambiando únicamente el archivo de tarifas;
- una corrida con contención declarada se excluye automáticamente del resumen económico;
- los resultados distinguen costo de investigación y costo de operación.

## Componente 3: validación formal contra requisitos

### Módulo

Crear `Comparacion/requirements_validation.py`, un registro `docs/engineering-requirements.json` y pruebas en `tests/test_requirements_validation.py`.

### Modelo de requisito

Cada requisito tendrá:

```text
id, tipo, enunciado, decisión, evidencia_requerida,
regla_de_verificación, estado, explicación
```

Los tipos serán `funcional`, `calidad`, `restricción` y `fuera_de_alcance`. Los estados serán `passed`, `partial`, `failed` y `not_applicable`. Un estado manual requerirá una justificación; siempre que sea posible se derivará de JSON o CSV auditados.

### Requisitos mínimos

- R1: las familias usan la misma representación y divisiones por obra;
- R2: se evalúan los mismos eventos y denominadores;
- R3: prueba no participa en la selección;
- R4: procedencia y regeneración desde el paquete público;
- R5: tiempos, memoria, hardware y escenarios monetarios se conservan por separado;
- R6: la ausencia de referencia estructural impide emitir conclusiones estructurales;
- R7: pruebas y auditorías pasan antes de aceptar resultados.

R6 se tratará como restricción epistemológica verificable, no como una promesa de evaluar estructura. Esto permite comprobar que el sistema no genera una conclusión que sus entradas no sostienen.

### Salidas

```text
validation_matrix.json
validation_matrix.md
```

Ambos archivos se generarán desde el mismo registro y los mismos artefactos. La tabla de la tesis se derivará de esta salida.

### Criterios de aceptación

- una evidencia ausente no puede producir `passed`;
- cada requisito apunta a un archivo y una regla concreta;
- el resultado es reproducible en una copia limpia;
- el comando termina con código distinto de cero si hay requisitos `failed`;
- `partial` permanece visible y no se convierte automáticamente en cumplimiento.

## Componente 4: prácticas y estándares de ingeniería

Crear `docs/engineering-practices.md`. El documento relacionará atributos de calidad pertinentes con evidencia del repositorio: corrección funcional, reproducibilidad, mantenibilidad, portabilidad, eficiencia y trazabilidad. Las referencias normativas exactas se verificarán antes de citarlas; el documento no afirmará certificación.

La evidencia incluirá pruebas automatizadas, contratos de CLI, separación de datos y código, gestión de versiones, auditorías, hashes, manejo explícito de faltantes y documentación del entorno. La matriz de requisitos enlazará estas prácticas con decisiones concretas.

## Componente 5: viabilidad social acotada

### Fase documental, sin participantes

Crear `docs/social-viability-protocol.md` con:

- caso de uso: reproducir y revisar una comparación de modelos secuenciales para música simbólica;
- población destinataria: estudiantes o personas investigadoras con experiencia básica en Python;
- tareas: instalación, ejecución del ejemplo pequeño, verificación del paquete e interpretación de un reporte;
- riesgos: barrera de hardware, sesgo del corpus, licencias, sobreinterpretación musical y accesibilidad de la documentación;
- métricas: terminación de tareas, tiempo, errores, solicitudes de ayuda y comprensión de dos limitaciones centrales.

### Fase empírica opcional

Antes de reclutar personas se solicitará al asesor o a la instancia institucional correspondiente una determinación sobre consentimiento y revisión ética. Si se autoriza, realizar entre tres y cinco sesiones sirve para detectar problemas de factibilidad y comprensión, no para generalizar impacto social.

Se conservarán únicamente resultados agregados y notas sin datos personales. El manuscrito deberá presentar la muestra, el procedimiento y las limitaciones. Si no existe autorización o tiempo, la tesis mantendrá la viabilidad social como evaluación documental y no la declarará demostrada.

### Criterios de aceptación

- protocolo y métricas definidos antes de las sesiones;
- consentimiento y tratamiento de datos documentados cuando corresponda;
- resultados negativos y tareas fallidas también se reportan;
- no se infiere impacto educativo, creativo o cultural desde una prueba de instalación;
- el resultado responde únicamente si el prototipo es utilizable y comprensible para el caso de uso declarado.

## Orden de implementación

1. Recuperar las cinco corridas originales desde la máquina de ejecución y copiar sólo en modo lectura.
2. Implementar y probar el paquete de evidencia.
3. Definir el registro de requisitos y generar una primera matriz; R4 y R5 deberán permanecer `partial` hasta contar con evidencia.
4. Implementar la medición de recursos y los escenarios de costo.
5. Ejecutar el benchmark aislado mínimo.
6. Regenerar la matriz; R4 y R5 sólo cambian a `passed` si cumplen sus reglas.
7. Completar la revisión de prácticas de ingeniería.
8. Preparar el protocolo social y decidir con el asesor si se ejecuta la fase empírica.
9. Actualizar la tesis únicamente desde las salidas versionadas.

## Decisión de suficiencia

Para una entrega limpia sin ampliar la pregunta científica principal, el mínimo recomendado es:

- paquete público verificable de las cinco corridas;
- matriz formal generada contra requisitos;
- benchmark aislado de tiempo y memoria;
- escenarios económicos con tarifas documentadas;
- análisis social estructurado y revisión ética explícita, aun si la fase con participantes queda fuera del calendario.

La única brecha que no puede cerrarse por completo mediante ingeniería es la evidencia social empírica. Las demás son abordables sin repetir el entrenamiento principal, siempre que se recuperen los artefactos originales.

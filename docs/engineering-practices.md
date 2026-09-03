# Prácticas de ingeniería y evidencia

Este proyecto no declara certificación ni conformidad con una norma externa. El objetivo de este documento es hacer explícitos los atributos de calidad que importan para la tesis y señalar qué evidencia permite revisarlos.

| Atributo | Decisión de ingeniería | Evidencia |
| --- | --- | --- |
| Corrección funcional | Los cuatro modelos puntúan la misma tarea y el conjunto de prueba no participa en la selección. | Pruebas automatizadas, `protocol_audit.json` y matriz de requisitos. |
| Trazabilidad | Cada corrida conserva configuración, divisiones, exclusiones, versión y artefactos primarios. | Paquete de evidencia, manifiesto SHA-256 y auditoría por corrida. |
| Reproducibilidad | Una copia limpia puede verificar el paquete y regenerar las salidas derivadas incluidas en su alcance. | `evidence_package.py`, diccionario de datos e instrucciones de verificación. |
| Eficiencia | Tiempo, RAM, VRAM y dispositivo se registran sin reducirlos a una unidad común. | `engineering_costs.csv`, monitor de recursos y manifiesto de hardware. |
| Mantenibilidad | Preparación, modelos, evaluación, auditoría y presentación se conservan en componentes separados. | Estructura de módulos, contratos de CLI y suites de pruebas. |
| Portabilidad | Las mediciones ausentes fallan de manera explícita y las rutas locales no entran al paquete público. | Estados de medición, saneamiento de rutas y prueba en copia limpia. |
| Responsabilidad | El sistema no presenta predicción como estructura musical ni costos estimados como gastos observados. | `structural_evaluation.json`, escenarios tarifarios y limitaciones declaradas. |

## Criterio de aceptación

Una evidencia sólo se considera cumplida cuando la regla correspondiente de `docs/engineering-requirements.json` produce `passed`. La presencia de documentación no sustituye una prueba ni una medición. Los estados `partial` y `failed` permanecen visibles en la matriz generada.

## Gestión de cambios

La especificación `docs/superpowers/specs/2026-09-01-cierre-criterios-tesis-design.md` precede a los módulos de cierre. Los cambios posteriores deben conservar pruebas, comandos reproducibles y una explicación de cualquier desviación respecto de esa especificación. Esto documenta el diseño de la versión final; no se utiliza para afirmar que toda la implementación histórica fue diseñada de antemano.

## Revisión normativa

Antes de citar una norma externa en el manuscrito se verificará su edición, alcance y fuente oficial. La correspondencia anterior puede relacionarse con modelos de calidad o prácticas de investigación reproducible, pero no debe describirse como certificación.

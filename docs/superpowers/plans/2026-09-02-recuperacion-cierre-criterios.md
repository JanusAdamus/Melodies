# Plan de recuperación del cierre de criterios de tesis

## Situación

La especificación `docs/superpowers/specs/2026-09-01-cierre-criterios-tesis-design.md` está publicada en `master`. Después de ese punto se inició una implementación local en la unidad Kingston, pero la unidad no está disponible en la sesión del 2 de septiembre de 2026 y esos cambios no aparecen en el remoto.

Este documento conserva el inventario conocido y establece cómo recuperarlo sin presentar como terminado código que todavía no se ha revisado ni ejecutado desde una copia limpia.

## Inventario del trabajo local iniciado

Los cambios que deben buscarse al reconectar la unidad son:

- `Comparacion/resource_monitor.py`;
- `Comparacion/evidence_package.py`;
- `Comparacion/cost_scenarios.py`;
- `Comparacion/requirements_validation.py`;
- integración de los comandos nuevos en `Comparacion/cli.py`;
- integración del monitor de recursos en los ejecutores de la comparación;
- pruebas unitarias de los cuatro componentes;
- registro de requisitos de ingeniería;
- documentación de prácticas de ingeniería;
- protocolo de viabilidad social;
- ejemplos, archivos de tarifas y guion de ejecución;
- dependencia explícita de `psutil`.

Antes de la desconexión se había observado el paso de 16 pruebas nuevas aisladas. Después se realizaron ajustes de revisión y no se alcanzó a repetir la batería; por tanto, ese resultado no acredita el estado actual de los archivos y no debe registrarse como verificación final.

## Estrategia de recuperación

### Ruta A: la unidad conserva los cambios

1. Reconectar Kingston y confirmar la ruta exacta del repositorio.
2. Ejecutar `git status --short` y guardar un parche de respaldo antes de modificar archivos.
3. Comparar el trabajo local con `origin/master` y con la especificación del 1 de septiembre.
4. Descartar únicamente duplicados exactos; no sobrescribir cambios sin inspección.
5. Revisar seguridad, determinismo, tratamiento de rutas personales y estados faltantes.
6. Ejecutar primero las pruebas nuevas y después la batería general.
7. Ejecutar los comandos con datos sintéticos y verificar sus códigos de salida.
8. Dividir los commits por componente y publicar sólo aquello que satisfaga sus criterios de aceptación.

### Ruta B: la unidad no conserva los cambios

Reimplementar desde la especificación en este orden:

1. validación formal contra requisitos;
2. paquete reproducible de evidencia;
3. monitor de recursos;
4. escenarios económicos;
5. documentación de prácticas y viabilidad social;
6. integración de CLI y ejecutores.

Cada módulo se implementará junto con sus pruebas. No se reconstruirán artefactos experimentales desde tablas ni desde el PDF.

## Trabajo que puede cerrarse sin la máquina de ejecución

- esquemas, validaciones y salidas deterministas de los cuatro módulos;
- pruebas con artefactos sintéticos;
- contratos de CLI;
- registro inicial R1--R7;
- documentación de prácticas de ingeniería;
- protocolo social sin participantes;
- archivo de tarifas de ejemplo claramente marcado como no experimental.

## Trabajo que depende de la máquina de ejecución

- recuperar y sanear las cinco corridas auditadas;
- construir el paquete público real;
- repetir el benchmark aislado de tiempo y memoria;
- calcular escenarios económicos con tiempos nuevos y tarifas documentadas;
- regenerar la matriz final con la evidencia recuperada;
- actualizar la tesis desde esas salidas.

## Criterios para publicar la implementación

- todas las pruebas nuevas pasan después de los últimos ajustes;
- la batería existente no presenta regresiones atribuibles a los cambios;
- una evidencia ausente nunca produce `passed`;
- los valores no medidos permanecen como `null` con una explicación;
- CPU y CUDA conservan alcances y unidades separados;
- el paquete de evidencia no contiene rutas personales, datos del corpus ni pesos;
- dos exportaciones equivalentes generan contenidos deterministas;
- los ejemplos se ejecutan desde una copia limpia;
- la documentación no afirma que los artefactos reales ya fueron recuperados;
- el estado de R4 y R5 sólo cambia cuando la evidencia satisface las reglas declaradas.

## Secuencia de commits propuesta

1. `feat: genera la matriz formal de requisitos`
2. `feat: exporta y verifica paquetes de evidencia`
3. `feat: registra recursos por fase experimental`
4. `feat: calcula escenarios económicos parametrizados`
5. `docs: documenta prácticas y protocolo social`
6. `test: verifica el cierre de criterios de tesis`
7. `data: publica manifiestos saneados de evidencia`, únicamente cuando se recuperen las corridas

## Definición de terminado

El trabajo pendiente se considerará recuperado cuando el código esté revisado, todas las pruebas pertinentes pasen y los componentes que no requieren corridas reales estén publicados. El cierre experimental sólo se considerará terminado después de recuperar los artefactos originales y ejecutar el benchmark de recursos en la máquina correspondiente.

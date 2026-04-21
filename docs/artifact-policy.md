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

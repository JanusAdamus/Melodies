# Notebooks

Los notebooks no son la interfaz oficial del proyecto. Se mantienen para
exploracion, demos o contexto historico ligero.

Subcarpetas:

- `exploratory/`: demos y exploracion reproducible.
- `thesis/`: notebooks directamente ligados al manuscrito, si aparecen en el
  futuro.
- `archive/`: material viejo que aun convenga conservar por contexto.

Regla de mantenimiento:

- cualquier flujo estable debe migrar a `src/`, `scripts/` o `docs/`;
- los notebooks deben evitar rutas absolutas y outputs innecesarios.

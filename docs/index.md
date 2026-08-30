# Indice de Documentacion

Esta carpeta es la referencia canonicamente mantenida del proyecto. La ruta
recomendada para orientarte es corta:

1. [`../README.md`](../README.md): panorama rapido del repo.
2. [`setup-and-reproduction.md`](setup-and-reproduction.md): como instalar,
   probar y ejecutar demos.
3. [`project-structure.md`](project-structure.md): mapa de carpetas y criterio
   de organizacion.
4. [`multidimensional-evaluation.md`](multidimensional-evaluation.md): pregunta
   estructural primaria, protocolo predictivo comun, costo, CLI y artefactos.
5. [`experiments.md`](experiments.md): flujo de planificacion y limites de
   evidencia.

## Guias Principales

- [`setup-and-reproduction.md`](setup-and-reproduction.md)
- [`project-structure.md`](project-structure.md)
- [`experiments.md`](experiments.md)
- [`multidimensional-evaluation.md`](multidimensional-evaluation.md)
- [`computational-complexity.md`](computational-complexity.md): costo del pipeline,
  medidas de escalado y tamano de corpus viable.
- [`resultados-comparacion-3000.md`](resultados-comparacion-3000.md): reporte de la
  corrida sobre 3000 obras, con limites de lo que se puede afirmar.
- [`correcciones-2026-08-24.md`](correcciones-2026-08-24.md): que se afirmo, que resulto
  falso o incompleto, y con que evidencia se corrigio.
- [`sources-and-methods.md`](sources-and-methods.md): trazabilidad de cada cambio
  algoritmico a su fuente, metodos descartados con su medicion, y la bibliografia
  unica del proyecto en su seccion 4.
- [`transformer.md`](transformer.md)
- [`transformer-roadmap.md`](transformer-roadmap.md)
- [`codex-multiagent-transformer-benchmark-prompt.md`](codex-multiagent-transformer-benchmark-prompt.md)
- [`transformer-benchmark-suite.md`](transformer-benchmark-suite.md)
- [`transformer-comparison-analysis.md`](transformer-comparison-analysis.md)
- [`artifact-policy.md`](artifact-policy.md)
- [`auditoria-ejecucion-pendiente.md`](auditoria-ejecucion-pendiente.md):
  qué quedó resuelto sin entrenar y qué corridas faltan, con su orden.
- [`canonicalizacion-revision-2026-08-24.md`](canonicalizacion-revision-2026-08-24.md):
  revisión grupo por grupo del agrupamiento por obra canónica.
- [`hmm-grid-sensitivity.md`](hmm-grid-sensitivity.md):
  regla de decisión y escalones de la rejilla de capacidad del HMM finito.
- [`resultados-comparacion-auditada.md`](resultados-comparacion-auditada.md):
  reporte de la corrida original y las cuatro sensibilidades auditadas, con lo
  que se puede y no se puede afirmar de cada eje.
- [`parada-curva-rehecha-2026-08-29.md`](parada-curva-rehecha-2026-08-29.md):
  por qué se detuvo la curva rehecha en la celda 9 de 15, qué quedó establecido
  por las sensibilidades y qué secciones de la tesis hay que reescribir.
- [`superpowers/plans/2026-08-24-auditoria-comparacion-final.md`](superpowers/plans/2026-08-24-auditoria-comparacion-final.md):
  plan ejecutable para recuperar artefactos, auditar denominadores y ejecutar
  las sensibilidades pendientes en la computadora principal.

## Baselines

- [`baselines/transformer-baseline.md`](baselines/transformer-baseline.md):
  baseline historico del transformer usado como punto de comparacion.

## Referencia Historica

Estos documentos conservan contexto tecnico util del trabajo previo. Se
mantienen como material de apoyo y no como punto de entrada para navegar el
repo.

- [`reference/technical-overview.md`](reference/technical-overview.md)
- [`reference/harmonic-extension.md`](reference/harmonic-extension.md)
- [`reference/multicorpus-expansion.md`](reference/multicorpus-expansion.md)
- [`reference/notes.md`](reference/notes.md)
- [`reference/next-token-overview.md`](reference/next-token-overview.md)
- [`reference/next-token-protocol.md`](reference/next-token-protocol.md)

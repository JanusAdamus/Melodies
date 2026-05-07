# Artifacts Locales

Esta carpeta concentra todos los artefactos generados del proyecto:

- salidas de los pipelines de `src/`;
- resultados de `next_token_experiment/`;
- temporales y reportes locales.

Su contenido es local y no debe subirse al repositorio remoto salvo que se
decida versionar un artefacto puntual de forma explícita.

Rutas relevantes:

- `artifacts/outputs/`: salidas de los análisis HMM/HDP-HMM y notebooks.
- `artifacts/next_token_experiment/results/`: corridas del experimento de
  siguiente token.
- `artifacts/tmp/`: temporales descartables.

## Regla Practica

No todo resultado local merece quedarse indefinidamente.

Conservar:

- corridas citadas en `docs/`;
- summaries agregados;
- artefactos necesarios para tesis o reproduccion.

Purgar localmente cuando ya no aporten:

- `tmp/`;
- `tmp_figs/`;
- smoke runs redundantes;
- variantes intermedias sin nombre canonico claro.

# Scripts

Esta carpeta solo debe contener wrappers ligeros y reproducibles.

Estado actual:

- `generate_thesis_chapter3_assets.py`: genera artefactos de apoyo para tesis.
- `reproduce_results.py`: punto de entrada reproducible para una demo minima.
- `run_transformer_benchmark_suite.py`: ejecuta o consolida la suite canónica
  del benchmark de `next_token`.
- `run_transformer_profile.py`: wrapper general para correr perfiles del
  experimento de `next_token`.
- `run_pdmx_gpu_max_verbose.py`: corrida GPU grande para PDMX; pertenece a la
  pista de escalamiento, no a la comparacion base con HMM/HDP-HMM.

Regla de mantenimiento:

- si un script no representa una corrida importante o un generador estable,
  debe integrarse a una CLI o eliminarse.

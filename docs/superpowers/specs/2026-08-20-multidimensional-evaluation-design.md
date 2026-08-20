# Diseño de evaluación multidimensional para Melodies

## Propósito

El repositorio debe pasar de una comparación centrada casi exclusivamente en predicción de siguiente token a un estudio de compromisos entre tres dimensiones: descripción estructural, predicción común y costo de ingeniería. El HMM finito y el HDP-HMM conservan su papel como modelos de estructura latente; el Transformer causal pequeño funciona como referencia neuronal predictiva; un modelo de Markov de orden variable inspirado en PPM funciona como control puente opcional y económico.

La pregunta implementada por el sistema es: bajo los mismos datos observables y divisiones por obra, ¿qué obtiene y qué sacrifica cada familia en capacidad estructural, calidad predictiva y costo?

## Alcance

Esta fase prepara y valida el protocolo. Incluye código, pruebas sintéticas, generación de manifiestos y documentación. No incluye entrenamientos canónicos, búsquedas extensas de hiperparámetros, recorridos masivos de PDMX ni producción de resultados finales para la tesis.

## Familias comparadas

- `finite_hmm`: línea base con número de estados seleccionado por validación.
- `hdp_hmm`: modelo truncado con número efectivo de estados aprendido.
- `transformer`: decodificador causal pequeño, no un sustituto conceptual de los estados latentes.
- `vomm`: control de Markov de orden variable, inspirado en PPM. Debe aparecer como control diagnóstico y no como protagonista de la tesis.

## Invariantes de equidad predictiva

1. Todos los modelos reciben las mismas obras, representación primaria y particiones agrupadas por obra canónica.
2. `BOS` proporciona contexto, pero nunca cuenta como objetivo musical ni contribuye a la NLL reportada. En los HMM la puntuación de un segmento se calcula como `log p([BOS] + x) - log p([BOS])`.
3. Validación y prueba se dividen en segmentos no superpuestos de longitud máxima `max_context_length`. El último segmento se conserva aunque contenga un solo evento.
4. Todos los modelos reinician el contexto en los mismos límites de segmento y puntúan cada evento musical exactamente una vez.
5. El entrenamiento del HDP-HMM no debe introducir transiciones artificiales entre el final de una obra y el inicio de otra.
6. La selección de hiperparámetros utiliza sólo validación. El objeto conservado como mejor ajuste debe corresponder a los hiperparámetros reportados.
7. La representación primaria continúa siendo `pitch_class`. Las representaciones ricas pertenecen a ablaciones internas y no pueden reemplazar el benchmark principal.

## Eje descriptivo primario

El sistema debe aceptar anotaciones estructurales opcionales en CSV con las columnas `piece_id`, `event_index`, `segment_label` y `boundary`. Cuando estén disponibles, debe calcular F1 de fronteras con tolerancia configurable, información mutua normalizada y ARI. Cuando no existan anotaciones, debe producir un artefacto con estado `not_evaluated` y explicar qué insumo falta; no debe fabricar etiquetas ni atribuir significado musical automático a los estados.

Las métricas estructurales deben ser funciones puras y reutilizables. El protocolo puede comparar etiquetas de estados o clusters producidas fuera del módulo, pero no debe presentar pesos de atención como explicaciones.

## Eje predictivo secundario

La medida primaria es NLL por evento musical. La perplejidad es una transformación de la misma cantidad. Siempre que el modelo entregue distribuciones completas se registrarán Brier score y exactitud; ninguna de estas medidas sustituye la NLL. Las curvas de aprendizaje usan subconjuntos anidados y conservan validación y prueba.

## Eje de costo de ingeniería

Por corrida se registran, como mínimo, tiempo de ajuste, tiempo de evaluación, número de parámetros o estados efectivos, dispositivo, número de obras y número de eventos de entrenamiento. La memoria pico y energía se registran sólo cuando el entorno las proporcione de manera fiable; en otro caso quedan como `null` con una razón explícita.

## Inferencia estadística y decisión

La unidad pareada es la obra canónica. Para la fracción completa se generan todos los contrastes por pares disponibles. Cada contraste reporta número de pares, diferencia media y mediana de NLL, intervalo bootstrap pareado del 95 %, prueba de Wilcoxon y valor p ajustado por Holm. La conclusión comparativa se expresa además como frontera de Pareto; no se reduce a un ganador único.

## Artefactos canónicos

Cada corrida debe poder producir:

- `config.json` y manifiestos de corpus y divisiones;
- `protocol_audit.json`, con cobertura de eventos y límites de reinicio;
- `results_raw.csv`, `results_summary.csv` y `piece_metrics_raw.csv`;
- `pairwise_comparisons.json`;
- `engineering_costs.csv`;
- `structural_evaluation.json`;
- `pareto_summary.json`;
- curva de aprendizaje;
- un plan de ejecución generado por `--plan-only`, sin ajustar modelos.

## Criterios de aceptación

- Las dos suites existentes continúan verdes.
- Nuevas pruebas demuestran la conservación de colas cortas, la cobertura exacta de eventos, el condicionamiento correcto en `BOS`, la ausencia de transiciones entre obras en HDP-HMM, la selección consistente del mejor candidato, el backoff del VOMM y el análisis pareado.
- `--plan-only` no llama a `fit` ni recorre el corpus completo cuando recibe `--max-files`.
- Ninguna prueba ejecuta entrenamiento canónico o descarga datos.
- La documentación distingue con claridad lo implementado, lo opcional y lo pendiente de ejecución por el autor.

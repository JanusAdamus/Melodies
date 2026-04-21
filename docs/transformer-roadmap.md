# Transformer Roadmap

## Proposito

Esta hoja de ruta separa de forma explicita dos tracks distintos:

- `baseline_tesis`: comparacion conservadora, CPU-feasible y reproducible.
- `research_transformer`: linea de investigacion para ampliar representacion, contexto y capacidad del modelo.

La idea es evitar un error comun: crecer el transformer sin mejorar antes la representacion y la evaluacion.

## Principios de Diseno

- No romper la comparabilidad del baseline historico.
- Mejorar primero la representacion simbolica, luego el modelo.
- Guardar todas las decisiones metodologicas como artefactos reproducibles.
- Distinguir claramente resultados de tesis y resultados exploratorios.

## Referencias Primarias que Guiaran el Track

- Huang et al., *Music Transformer* (arXiv:1809.04281): muestra que la atencion relativa y el modelado de dependencias largas son especialmente relevantes para musica simbolica.  
  https://arxiv.org/abs/1809.04281
- Huang y Yang, *Pop Music Transformer* (arXiv:2002.00212): muestra que una representacion con estructura metrica mejora el modelado mucho mas que tratar la secuencia como un flujo plano de notas.  
  https://arxiv.org/abs/2002.00212
- Qu et al., *MuPT* (arXiv:2404.06393): apunta a que la escala, la longitud de contexto y el pretraining son una direccion futura plausible cuando el pipeline ya este maduro.  
  https://arxiv.org/abs/2404.06393

## Estado Actual

Lo que hoy existe en el repo sirve como baseline reproducible:

- tarea de next-token prediction;
- representacion `pitch_class` y `pitch_class_duration`;
- transformer pequeno;
- evaluacion con NLL, perplexity y accuracy;
- artefactos de corrida y checkpoints guardados.

Eso no es suficiente todavia para una linea de investigacion fuerte.

## Objetivo Inmediato

Dejar preparado el proyecto para un `research_richer_events` que:

- use una representacion simbolica mas rica;
- pueda incorporar silencios y posicion metrica;
- use una longitud de contexto mayor;
- permita un transformer mas capaz sin tocar el baseline de tesis.

## Fase 1: Endurecer el Baseline

Objetivo:
dejar el baseline historico intocable y bien documentado.

Tareas:

- congelar `cpu_baseline` como referencia metodologica;
- mantener `pitch_class` como comparacion base;
- documentar metricas, splits, limites y supuestos;
- dejar trazabilidad de configuracion, runtime y artefactos.

Salida esperada:

- baseline reproducible;
- docs claros;
- resultados comparables a futuro.

## Fase 2: Mejorar la Representacion

Objetivo:
pasar de una secuencia plana a una secuencia musicalmente mas informativa.

Tareas:

- introducir `event_pitch_duration_metrical`;
- permitir rests cuando el track sea de investigacion;
- codificar al menos pitch, duracion y clase metrica;
- mantener vocabulario determinista a nivel corpus.

Salida esperada:

- representacion nueva lista para entrenar;
- pruebas unitarias de codificacion;
- comparacion directa contra `pitch_class`.

## Fase 3: Track Research

Objetivo:
habilitar un perfil serio sin contaminar el baseline.

Tareas:

- agregar perfil `research_richer_events`;
- ampliar contexto;
- ampliar profundidad y ancho del transformer;
- habilitar entrenamiento `auto` para CPU/GPU sin romper portabilidad.

Salida esperada:

- corridas exploratorias serias;
- capacidad suficiente para notar si la representacion realmente ayuda.

## Fase 4: Evaluacion Mas Fuerte

Objetivo:
dejar de evaluar solo con un numero global.

Tareas:

- metricas por pieza;
- metricas por longitud;
- metricas por rareza de token;
- top-k accuracy;
- muestras de continuacion cualitativas con semillas fijas.

Salida esperada:

- lectura mas fina de errores;
- mejor criterio para decidir si escalar o no.

Estado actual:

- `top_3_accuracy` y `top_5_accuracy` ya estan integradas;
- los resultados ya incluyen slices por longitud de pieza, compositor y rareza de token;
- las corridas ya guardan continuaciones reproducibles para inspeccion cualitativa.

## Fase 5: Arquitectura de Largo Plazo

Objetivo:
resolver el limite real del modelo actual para secuencias largas.

Tareas:

- estudiar una variante con posicion relativa o sesgos relativos;
- evaluar contextos de 256 a 1024 tokens;
- revisar coste memoria/longitud antes de escalar mas.

Salida esperada:

- decision informada sobre seguir con decoder absoluto o migrar a una arquitectura mas adecuada para musica.

Estado actual:

- el track `research_richer_events` ya activa sesgo posicional relativo bucketizado como primer paso practico hacia contexto largo real.

## Fase 6: Escala y Pretraining

Objetivo:
llegar a un transformer que ya no sea solo un baseline ampliado.

Tareas:

- ampliar corpus;
- normalizar representaciones entre fuentes;
- estudiar pretraining autoregresivo;
- considerar multitrack o sincronizacion mas fuerte entre voces.

Salida esperada:

- una linea de investigacion defendible mas alla del experimento pequeño de tesis.

## Prioridad Real

El orden correcto sigue siendo este:

1. representacion
2. evaluacion
3. organizacion experimental
4. arquitectura
5. escala

Si se invierte ese orden, el proyecto puede crecer en parametros pero seguir siendo debil metodologicamente.

## Proximos Entregables del Repo

- perfilar y comparar `cpu_baseline` vs `research_richer_events`;
- agregar reportes por slice y top-k metrics;
- introducir generacion de continuaciones cortas y reproducibles;
- decidir si la siguiente fase exige atencion relativa.

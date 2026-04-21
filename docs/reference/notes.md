# Notes de modelado

## Decisiones principales

- El notebook original se reutilizo solo como fuente de:
  - parser con `music21`,
  - extraccion de `pitchClass`,
  - baseline armonico con Viterbi.
- El baseline armonico ya no usa solo mayor/menor, sino un vocabulario estructurado por `raiz + calidad + extension` con triadas, septimas, novenas, suspendidos y acordes con notas anadidas.
- El HDP-HMM se implemento como una aproximacion truncada con `K` estados potenciales.
- Los estados del HDP-HMM se interpretan a posteriori y no se fuerzan a coincidir con acordes tonales.
- Se introdujo `kappa` como opcion sticky para favorecer permanencia, pero su valor por defecto es `0.0` para no alterar la formulacion base si no se desea.
- El contexto modal se maneja como una capa separada del acorde para evitar multiplicar artificialmente el tamano del espacio de estados por todos los modos posibles.

## Vocabulario armonico elegido

- Se incluyen triadas `maj`, `min`, `dim`, `aug`.
- Se incluyen acordes de septima `maj7`, `7`, `m7`, `m(maj7)`, `ø7`, `dim7`.
- Se incluyen acordes de sexta y notas anadidas `6`, `m6`, `add9`, `madd9`, `6/9`.
- Se incluyen `sus2`, `sus4` y `7sus4`.
- Se incluyen extensiones hasta `maj9`, `9` y `m9`.
- No se incluyen 11, 13 ni alteraciones avanzadas de jazz para mantener interpretabilidad y estabilidad computacional.
- Se mantienen los siete modos diatonicos principales: `ionian`, `dorian`, `phrygian`, `lydian`, `mixolydian`, `aeolian` y `locrian`.

Este equilibrio produce `252` estados armonicos en el baseline, numero lo bastante rico para musica tonal y modal comun, pero aun razonable para Viterbi y para una matriz de transicion heuristica interpretable.

## Honestidad matematica

- El proyecto no pretende implementar un sampler exacto del iHMM infinito.
- La parte exacta corresponde al HMM truncado condicionado a `beta`.
- La parte aproximada aparece en:
  - truncacion weak-limit del espacio de estados,
  - actualizacion de `beta` por MAP en lugar de un sampler HDP completo con tablas auxiliares,
  - fallback aproximado para `beta` cuando `scipy` no esta instalado.

## Limitaciones actuales

- Hay posible label switching entre muestras del HDP-HMM, por lo que la metrica de estabilidad de trayectoria debe leerse con cautela.
- El baseline solo opera sobre `pitch_class`; si se piden otras observaciones, la CLI sigue usando `pitch_class` para ese baseline y deja el tipo solicitado para el HDP-HMM.
- Las emisiones son categoricas; no se modelan aun duraciones continuas ni dependencias jerarquicas entre voces.
- La interpretacion musical es heuristica y esta separada de la inferencia estadistica.
- La capa modal del baseline es local y contextual, no una segunda cadena de Markov acoplada.

## Extensiones razonables

- Sampler HDP mas fiel con cuentas auxiliares tipo Chinese Restaurant Franchise.
- Version sticky HDP-HMM mas desarrollada para segmentacion armonica.
- Emisiones factoriales o multinomiales sobre varios canales simultaneos.
- Alineacion con compases, voces o analisis por ventana.
- Comparacion sistematica entre varias piezas y no una sola secuencia.
- Incorporacion posterior de menor armonica, menor melodica e intercambio modal.

## Expansion multicorpus

- Se anadio una capa de ingestión multicorpus con adaptadores para `SymbTr`, `PDMX` y `JAZZMUS`.
- `SymbTr` aporta `makam`, `usul` y forma desde el nombre canonico de archivo.
- `PDMX` se integra solo cuando el usuario dispone del dataset local real; el repositorio clonado no sustituye al corpus.
- `JAZZMUS` puede requerir exportacion previa desde JSON a `MusicXML`; el proyecto ya automatiza esa preparacion dentro de la CLI multicorpus.
- La comparacion entre corpus debe hacerse siempre conservando la procedencia, porque no todos comparten el mismo sistema musical ni el mismo grado de compatibilidad con un analisis armonico tonal.

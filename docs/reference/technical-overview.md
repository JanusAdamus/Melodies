# Documentacion Tecnica Completa de Melodies

## 1. Vision general

`Melodies` es un proyecto de analisis de musica simbolica construido alrededor de dos modelos de Markov ocultos:

1. un HMM finito de base, con estados armonicos explicitos y semantica musical interpretable;
2. un HDP-HMM truncado, usado como aproximacion computacional a un iHMM, donde los estados ocultos no se fijan como acordes antes del ajuste sino que se aprenden de los datos.

El repositorio no es solo un experimento aislado: ya esta organizado como pipeline reproducible, con parser, representacion de observaciones, modelos, inferencia, interpretacion, visualizacion, analisis por lotes y adaptadores multicorpus.

En terminos practicos, el flujo completo es:

`partitura -> score music21 -> eventos musicales -> observaciones discretas -> inferencia HMM/HDP-HMM -> tablas/figuras/reportes`

El proyecto se apoya en `music21` para leer MusicXML y MIDI, en `numpy` para la parte numerica, en `pandas` para exportes tabulares, y en `matplotlib` / `seaborn` / `networkx` para visualizacion.

## 2. Objetivo cientifico y computacional

El objetivo del repo es formalizar y extender un notebook anterior que ya contenia tres piezas utiles:

1. carga de partituras simbolicas;
2. extraccion de observaciones discretas;
3. un baseline HMM con Viterbi.

La refactorizacion actual hace dos movimientos importantes:

1. convierte ese prototipo en un sistema modular y defendible para trabajo academico;
2. separa claramente:
   - la representacion musical,
   - la inferencia probabilistica,
   - la interpretacion musicologica.

Eso es clave porque el proyecto no mezcla "estado latente aprendido" con "etiqueta armonica humana" como si fueran exactamente lo mismo. El baseline si usa un vocabulario armonico explicito; el HDP-HMM no. El HDP-HMM aprende contextos latentes y luego los interpreta a posteriori.

## 3. Estructura real del repositorio

La estructura funcional central del proyecto es:

```text
Melodies/
  README.md
  NOTES.md
  MULTICORPUS_EXPANSION.md
  requirements.txt
  examples/
    generate_example_score.py
    example_score.musicxml
  notebooks/
    demo_finite_hmm.ipynb
    demo_hdp_hmm.ipynb
    demo_multicorpus.ipynb
    demo_library_corpus.ipynb
  src/
    analysis/
      diagnostics.py
      interpretation.py
      library_batch.py
      library_visualization.py
      visualization.py
    cli/
      main.py
      library_analysis.py
      multicorpus_analysis.py
    data/
      parsing.py
      observations.py
      library_catalog.py
      multicorpus.py
    models/
      harmony.py
      finite_hmm.py
      hdp_hmm.py
      inference.py
      utils.py
  tests/
    test_parsing.py
    test_observations.py
    test_harmony.py
    test_finite_hmm.py
    test_hdp_hmm.py
    test_library_catalog.py
    test_multicorpus.py
  external/
    library/
    SymbTr/
    PDMX/
    ISMIR-Jazzmus/
  outputs/
    ...
```

### Lectura de capas

- `src/data/`: convierte partituras y corpus en estructuras limpias y explotables.
- `src/models/`: contiene teoria armonica, utilidades probabilisticas e implementacion de modelos.
- `src/analysis/`: resume, interpreta, exporta y visualiza resultados.
- `src/cli/`: expone pipelines completos desde linea de comandos.
- `tests/`: fija invariantes tecnicos basicos del sistema.
- `external/`: alojamientos de corpus externos o referencias locales.
- `outputs/`: resultados de corridas ya ejecutadas.

## 4. Dependencias y por que existen

El archivo `requirements.txt` declara:

```text
numpy>=1.26
scipy>=1.13
pandas>=2.2
matplotlib>=3.8
seaborn>=0.13
networkx>=3.2
music21>=9.1
openpyxl>=3.1
jupyter>=1.0
```

Rol de cada dependencia:

- `numpy`: algebra lineal, muestreo Dirichlet/Beta, representacion de matrices.
- `scipy`: optimizacion del update MAP de `beta` en el HDP-HMM.
- `pandas`: tablas, CSV y Excel.
- `matplotlib`: figuras base.
- `seaborn`: heatmaps y graficos estadisticos con mejor legibilidad.
- `networkx`: grafo dirigido de transiciones.
- `music21`: parser de MusicXML/MIDI y analisis de claves/medidas.
- `openpyxl`: escritura de workbooks `.xlsx`.
- `jupyter`: soporte para notebooks.

Si `scipy` no esta disponible, el proyecto no colapsa: el HDP-HMM usa un fallback aproximado para `beta`.

## 5. Flujo de datos de extremo a extremo

El flujo de una corrida individual desde `src/cli/main.py` es:

1. leer argumentos de CLI;
2. cargar la partitura con `parse_score`;
3. extraer eventos temporales con `extract_events`;
4. convertir esos eventos en una secuencia discreta con `build_observation_sequence`;
5. exportar tablas comunes de eventos, observaciones y vocabularios;
6. ejecutar `FiniteChordHMM`, `TruncatedHDPHMM` o ambos;
7. construir tablas diagnosticas;
8. generar figuras;
9. comparar modelos si ambos corrieron;
10. escribir un `summary.txt`.

En batch y multicorpus, el pipeline se envuelve con una capa previa de catalogacion y una capa posterior de agregacion por obra/compositor/fuente.

## 6. Capa de parsing musical (`src/data/parsing.py`)

### 6.1. Entidad central: `MusicalEvent`

Cada evento musical se representa como una instancia inmutable con:

- `index`: indice secuencial dentro de la serie de eventos;
- `offset`: posicion temporal en cuartos;
- `duration`: duracion del evento en cuartos;
- `kind`: `note`, `chord` o `rest`;
- `label`: texto legible del evento;
- `pitch_classes`: pitch classes involucradas;
- `midi_pitches`: alturas MIDI involucradas;
- `representative_pitch_class`: pitch class representativa;
- `representative_midi`: altura MIDI representativa;
- `measure`: compas;
- `beat`: pulso local.

### 6.2. Parseo de partitura

`parse_score(path)` delega directamente en `music21.converter.parse`. Esto significa que la semantica de lectura de formatos depende de `music21`, no de un parser artesanal local.

### 6.3. Seleccion y fusion de partes

La logica de `select_parts(score, prefer_treble=True)` hace algo importante:

- si el score no tiene `parts`, devuelve el propio score como unica fuente;
- si `prefer_treble=False`, conserva todas las partes;
- si `prefer_treble=True`, intenta quedarse con partes que contengan `TrebleClef`;
- si no encuentra ninguna con clave de sol, usa todas.

Esto tiene una consecuencia musical fuerte: el proyecto no modela contrapunto multicanal como varias cadenas acopladas, sino que reduce la partitura a un flujo temporal unificado. Si se activa la preferencia por clave de sol, el material grave puede quedar fuera cuando existan partes agudas suficientes.

`merge_parts` inserta en un `music21.stream.Stream` todos los elementos `notesAndRests` de las partes seleccionadas, respetando `offset`.

### 6.4. Extraccion de eventos

`extract_events(...)` ordena por `(offset, priority)` y recorre notas, acordes y, opcionalmente, silencios.

Detalles importantes:

- por defecto `include_rests=False`, asi que los silencios no entran al modelado;
- una `note.Note` produce un evento con un solo pitch class y un solo MIDI;
- un `music21.chord.Chord` produce:
  - lista ordenada de MIDI,
  - lista ordenada de pitch classes,
  - pitch class representativa basada en la raiz del acorde si `music21` puede estimarla,
  - si no, usa el bajo;
- un silencio solo se exporta si `include_rests=True`.

### 6.5. Pitch representativo en acordes

Para acordes, el sistema necesita una proyeccion a un solo simbolo cuando la observacion no es policanal.

La logica usada es:

1. intentar `chord.root()`;
2. si no existe, usar `chord.bass()`;
3. para MIDI representativo, buscar una altura cuyo `pitchClass` coincida con la raiz;
4. si no aparece, usar el bajo.

Esto no preserva toda la estructura vertical del acorde, pero permite construir observaciones discretas simples y consistentes.

## 7. Construccion de observaciones (`src/data/observations.py`)

### 7.1. Entidad central: `ObservationSequence`

Una secuencia discreta queda descrita por:

- `observation_type`;
- `tokens`: arreglo entero `numpy`;
- `vocabulary`: lista de etiquetas observacionales;
- `decoded`: secuencia tokenizada pero en texto;
- `events`: eventos origen;
- `extra`: metadatos especificos del tipo de observacion.

La tabla exportable incluye `t`, `offset`, `duration`, `measure`, `beat`, `event_kind`, `event_label`, `token` y `observation`.

### 7.2. Tipos de observacion soportados

`available_observation_types()` devuelve:

- `pitch_class`
- `midi_note`
- `interval`
- `pitch_class_duration`

### 7.3. `pitch_class`

- vocabulario fijo de tamano 12;
- tokens = `representative_pitch_class`;
- etiquetas = `C, C#, D, ... , B`.

Formalmente:

`y_t in {0, ..., 11}`

Es la observacion mas compatible con el baseline armonico finito.

### 7.4. `midi_note`

- vocabulario dinamico dependiente de la obra;
- se toman todos los `representative_midi`;
- se ordenan y se indexan.

Si hay alturas MIDI unicas `m_1 < ... < m_V`, entonces:

- `lookup(m_i) = i-1`
- `y_t = lookup(midi_t)`

Esto conserva registro, no solo clase de altura.

### 7.5. `interval`

Modela diferencia respecto del MIDI representativo previo.

La primera observacion siempre es `START`.

Para `t > 1`:

`delta_t = midi_t - midi_(t-1)`

y luego se trunca:

`delta_t_clipped = min(max(delta_t, -max_interval), max_interval)`

Con `max_interval = 12`, el vocabulario es:

- `START`
- `INT_-12` hasta `INT_+12`

Tamano total:

`V = 1 + (2 * max_interval + 1)`

Con el valor por defecto:

`V = 26`

### 7.6. `pitch_class_duration`

Cada evento se proyecta a un par:

`(pitch_class_representativa, duration_bin)`

La duracion se discretiza al bin mas cercano entre, por defecto:

`(0.25, 0.5, 1.0, 2.0, 4.0)`

La etiqueta textual es del tipo:

`C|dur=1`

Este esquema mezcla informacion tonal y ritmica de forma categorial.

### 7.7. Observacion y baseline

Hay una decision de implementacion importante:

- el baseline `FiniteChordHMM` solo acepta `pitch_class`;
- si el usuario pide otra observacion en CLI y tambien solicita `finite_hmm`, la CLI reconstruye internamente una secuencia `pitch_class` solo para ese modelo;
- el HDP-HMM si usa el tipo solicitado.

Eso significa que en modo `both`, los dos modelos pueden estar corriendo sobre representaciones distintas si `--obs != pitch_class`.

## 8. Teoria armonica explicita (`src/models/harmony.py`)

Este modulo es una de las piezas mas importantes del repo porque separa una ontologia armonica interpretable del modelo probabilistico.

### 8.1. Modos diatonicos

Se codifican siete modos:

- `ionian = (0, 2, 4, 5, 7, 9, 11)`
- `dorian = (0, 2, 3, 5, 7, 9, 10)`
- `phrygian = (0, 1, 3, 5, 7, 8, 10)`
- `lydian = (0, 2, 4, 6, 7, 9, 11)`
- `mixolydian = (0, 2, 4, 5, 7, 9, 10)`
- `aeolian = (0, 2, 3, 5, 7, 8, 10)`
- `locrian = (0, 1, 3, 5, 6, 8, 10)`

Son definidos relativos a una tonica base `0`, y luego se transpone por pitch class.

### 8.2. Plantillas de acorde

La clase `ChordTemplate` define:

- `symbol`
- `family`
- `level`
- `intervals`
- `description`

El vocabulario implementado por defecto contiene 21 plantillas:

- triadas: `maj`, `min`, `dim`, `aug`
- septimas: `maj7`, `7`, `m7`, `m(maj7)`, semidisminuido, `dim7`
- acordes con sexta / anadidos: `6`, `m6`, `add9`, `madd9`, `6/9`
- suspendidos: `sus2`, `sus4`, `7sus4`
- novenas: `maj9`, `9`, `m9`

En el codigo, `6/9` se incluye si `include_six_nine=True`, que es el valor por defecto.

### 8.3. Espacio de estados armonicos

Cada `HarmonicState` es una combinacion:

`root in {0,...,11}` x `template in templates`

Por tanto, el tamano del espacio es:

`12 * 21 = 252`

Cada estado tiene:

- `label`, por ejemplo `C:maj7` o `G:9`;
- `chord_tones`, calculado como suma modular `root + interval mod 12`;
- `compatible_modes`, que son los modos cuyos grados contienen todos los intervalos de la plantilla.

### 8.4. Contextos modales

`ModalContext` define:

- `tonic`
- `mode`
- `label`
- `pitch_classes`

El espacio total de contextos modales es:

`12 * 7 = 84`

### 8.5. Distribucion de emision armonica heuristica

`chord_emission_distribution(state, chord_tone_mass=0.72, modal_tone_mass=0.20, outside_mass=0.08)`

construye una distribucion sobre pitch classes de la forma:

1. masa fuerte sobre las notas del acorde;
2. masa secundaria sobre tonos de escalas compatibles que no son notas del acorde;
3. masa residual sobre tonos externos.

Si definimos:

- `C` = conjunto de notas del acorde,
- `M` = union de pitch classes de modos compatibles,
- `S = M \ C`,
- `O = {0,...,11} \ (C union S)`,

entonces:

- para `p in C`: `P(p|z) += 0.72 / |C|`
- para `p in S`: `P(p|z) += 0.20 / |S|`
- para `p in O`: `P(p|z) += 0.08 / |O|`

Si `S` es vacio, su masa pasa al componente `outside`.

No es una distribucion aprendida: es una distribucion de emision disenada a mano para el baseline.

### 8.6. Scoring armonico y modal

Para interpretar estados del HDP-HMM, el modulo define funciones de matching:

- `score_state_against_profile(profile, state)`
- `infer_chord_candidates_from_pitch_classes(...)`
- `score_modal_context(profile, context)`
- `infer_mode_candidates(...)`

#### Scoring acorde-perfil

Si `profile` es una distribucion de pitch classes y `emission(state)` la heuristica del acorde, el score es:

`score = sum_p profile[p] * log(emission[p] + EPSILON)`

Es esencialmente una esperanza de log-probabilidad bajo el perfil observado.

#### Scoring modal

Si `support(context)` es el conjunto de pitch classes del modo:

- `inside = suma de profile[p] para p dentro del modo`
- `outside = 1 - inside`

y el score es:

`inside - 0.6 * outside`

No es una probabilidad normalizada; es una funcion heuristica de compatibilidad.

### 8.7. Proyeccion armonica desde observaciones no tonales

`harmonic_profile_from_observations(...)` permite reinterpretar algunas emisiones del HDP-HMM como perfil de pitch classes:

- si la observacion ya es `pitch_class`, se normaliza directamente;
- si es `pitch_class_duration`, se suman probabilidades por nombre de pitch class;
- si es `midi_note` o `interval`, devuelve `None`.

Eso explica por que la interpretacion musical del HDP-HMM es mas rica cuando las observaciones conservan informacion tonal directa.

## 9. Utilidades numericas (`src/models/utils.py`)

Este modulo concentra primitivas probabilisticas y numericas.

### 9.1. `normalize`

Normaliza vectores o matrices evitando division por cero. Si una fila o columna suma cero, la sustituye por distribucion uniforme.

Esto aparece por todo el proyecto y es una defensa importante contra underflow o filas degeneradas.

### 9.2. `logsumexp`

Implementacion local:

`max + log(sum(exp(values - max)) + EPSILON)`

Sirve para inferencia estable en espacio logaritmico sin depender de `scipy.special`.

### 9.3. Muestreo categorial

- `sample_categorical`
- `sample_categorical_from_log_probs`

La segunda toma log-probabilidades, resta `logsumexp` y exponentia para obtener probabilidades estables.

### 9.4. Stick-breaking truncado

`sample_truncated_stick_breaking(gamma, n_states, rng)` genera:

`v_k ~ Beta(1, gamma)` para `k = 1, ..., K-1`

y luego:

- `beta_1 = v_1`
- `beta_k = v_k * prod_{l<k}(1-v_l)` para `2 <= k <= K-1`
- `beta_K = prod_{l<K}(1-v_l)`

Eso se implementa con `stick_breaking_from_v`.

### 9.5. Conteos

- `count_transitions(states, n_states)` produce matriz `N_ij`;
- `count_emissions(states, observations, n_states, vocab_size)` produce matriz `M_ik`;
- `contiguous_segments(states)` devuelve segmentos `(estado, inicio, fin_exclusivo)`.

Estas funciones alimentan tanto posterior Dirichlet como diagnosticos de segmentacion.

## 10. Inferencia HMM clasica (`src/models/inference.py`)

Este archivo implementa la base algoritmica comun del proyecto.

### 10.1. Emisiones logaritmicas

`compute_emission_log_probs(E, y)` devuelve la matriz:

`log E[:, y_t]`

para cada tiempo `t`.

### 10.2. Forward algorithm

`forward_log_likelihood(...)` implementa:

`alpha_1(j) = log pi_j + log b_j(y_1)`

`alpha_t(j) = log b_j(y_t) + logsumexp_i [alpha_(t-1)(i) + log a_ij]`

y al final:

`log p(y_1:T) = logsumexp_j alpha_T(j)`

### 10.3. FFBS

`ffbs_sample(...)` combina:

1. forward filtering para obtener `alpha`;
2. backward sampling para muestrear una trayectoria entera.

El muestreo hacia atras usa:

`p(z_t=i | z_(t+1)=j, y_1:T) propto exp(alpha_t(i) + log a_ij)`

### 10.4. Viterbi

`viterbi_decode(...)` usa:

`delta_1(j) = log pi_j + log b_j(y_1)`

`delta_t(j) = log b_j(y_t) + max_i [delta_(t-1)(i) + log a_ij]`

`psi_t(j) = argmax_i [delta_(t-1)(i) + log a_ij]`

y luego backtracking sobre `psi`.

## 11. Baseline armonico finito (`src/models/finite_hmm.py`)

### 11.1. Naturaleza del modelo

Este modelo no aprende parametros desde datos. Aunque el metodo se llame `fit_predict`, en realidad:

1. construye una matriz inicial uniforme;
2. construye una matriz de transicion heuristica fija;
3. construye una matriz de emision heuristica fija;
4. ejecuta Viterbi;
5. calcula log-likelihood forward;
6. estima una matriz de transicion empirica sobre la trayectoria decodificada.

Es, por tanto, un baseline experto, interpretable y determinista dado el input.

### 11.2. Estado espacio

Usa `HARMONIC_STATE_SPACE = build_harmonic_state_space()`, o sea 252 estados por defecto.

`CHORD_STATES` es la lista de etiquetas legibles.

### 11.3. Distribucion inicial

`_build_initial_probs()` retorna uniforme:

`pi_j = 1 / S`

con `S = 252`.

### 11.4. Heuristica de transicion

Para cada par `(source, target)`, el score es:

`score = 1.0`

Se suman sesgos:

- `+ stay_bias` si es el mismo estado;
- `+ same_root_bias` si comparten raiz;
- `+ 0.3` extra si, ademas, tienen el mismo `template.level`;
- `+ related_root_bias` si la raiz destino esta a cuarta o quinta (`interval in {5, 7}`);
- `+ shared_mode_bias` si comparten algun modo compatible.

Con los defaults:

- `stay_bias = 2.5`
- `same_root_bias = 0.8`
- `related_root_bias = 0.4`
- `shared_mode_bias = 0.3`

Luego cada fila se normaliza:

`A_i = score_i / sum_j score_ij`

No hay entrenamiento EM ni Bayes aqui.

### 11.5. Matriz de emision

`_build_emission_matrix()` llama a `chord_emission_distribution` para cada estado. Cada fila es una distribucion de dimension 12.

### 11.6. Decodificacion

`fit_predict(observations)` exige `observations.observation_type == "pitch_class"`.

Luego:

1. ejecuta `viterbi_decode`;
2. ejecuta `forward_log_likelihood`;
3. cuenta transiciones observadas en la trayectoria Viterbi;
4. normaliza esas cuentas para obtener `empirical_transition_matrix`;
5. infiere un modo local por ventana temporal.

### 11.7. Modo local

`_infer_modal_labels(states, observations, window_radius=2)` toma, para cada tiempo `t`, una ventana:

`[t-2, t-1, t, t+1, t+2]`

recortada a los limites validos, y estima el contexto modal mas plausible condicionado a la raiz del estado armonico actual.

Esto no es una segunda cadena oculta. Es una etiqueta interpretativa local derivada del vecindario observacional.

### 11.8. Objeto resultado

`FiniteHMMResult` contiene:

- trayectoria latente;
- etiquetas armonicas;
- etiquetas modales;
- estados activos;
- probabilidades iniciales;
- matriz de transicion heuristica;
- matriz de emision heuristica;
- matriz de transicion empirica posterior a Viterbi;
- log-likelihood;
- score Viterbi;
- observaciones;
- estado espacio.

Detalle fino importante:

- las tablas exportadas por `build_finite_tables` usan `empirical_transition_matrix`, no la matriz heuristica original;
- por eso las transiciones exportadas reflejan la trayectoria decodificada, no solo el prior estructural del modelo.

## 12. HDP-HMM truncado (`src/models/hdp_hmm.py`)

### 12.1. Idea general

Este modulo implementa un HMM Bayesiano no parametrico aproximado por truncacion weak-limit.

No implementa un sampler exacto del iHMM infinito. Implementa:

1. una rejilla finita de `K` estados potenciales;
2. pesos globales `beta` tipo stick-breaking;
3. priors Dirichlet sobre distribucion inicial, filas de transicion y emisiones;
4. blocked Gibbs sampling con FFBS para trayectorias;
5. actualizacion aproximada de `beta` por MAP o fallback.

### 12.2. Modelo generativo truncado

Sea:

- `K` = truncacion maxima;
- `V` = tamano del vocabulario de observaciones;
- `alpha` = concentracion de transiciones;
- `alpha0` = concentracion de la distribucion inicial;
- `gamma` = concentracion del stick-breaking global;
- `eta` = masa total del prior de emision;
- `kappa` = sticky mass diagonal opcional.

Entonces:

1. para `k = 1, ..., K-1`:
   `v_k ~ Beta(1, gamma)`
2. `beta = SB(v)` via stick-breaking truncado
3. `pi ~ Dir(alpha0 * beta)`
4. para cada estado `i`:
   `A_i ~ Dir(alpha * beta + kappa * e_i)`
5. para cada estado `i`:
   `phi_i ~ Dir(eta/V, ..., eta/V)`
6. `z_1 ~ Cat(pi)`
7. `z_t | z_(t-1) ~ Cat(A_(z_(t-1)))`
8. `y_t | z_t ~ Cat(phi_(z_t))`

`e_i` es el vector one-hot del estado `i`.

### 12.3. Inicializacion

`_initialize_states` no empieza aleatoriamente del todo. Hace:

1. define `active = min(init_active_states, K, max(2, min(vocab_size, K)))`;
2. mapea cada token observado a uno de esos `active` estados;
3. inyecta ruido aleatorio al 10% de posiciones.

Es una inicializacion pragmatica para evitar empezar desde una particion totalmente ciega.

### 12.4. Posteriores conjugados condicionados en `beta`

#### Emisiones

Si `M_ik` es el conteo de veces que el estado `i` emitio el simbolo `k`, entonces:

`phi_i | z, y ~ Dir(eta/V + M_i1, ..., eta/V + M_iV)`

Eso es exactamente lo que hace `_sample_emissions`.

#### Distribucion inicial

Si `z_1` es el primer estado:

`pi | z, beta ~ Dir(alpha0 * beta + one_hot(z_1))`

Eso es `_sample_initial`.

#### Filas de transicion

Si `N_ij` es el numero de transiciones `i -> j`, entonces:

`A_i | z, beta ~ Dir(alpha * beta + kappa * e_i + N_i1, ..., alpha * beta + kappa * e_i + N_iK)`

Eso es `_sample_transitions`.

### 12.5. Update de `beta`

Esta es la parte mas aproximada del modelo.

#### Modo 1: `map_stick_breaking`

Si `scipy` esta disponible, el proyecto:

1. convierte `beta` a variables `v`;
2. parametriza `v` con logits;
3. minimiza la negativa del log-posterior:
   - prior stick-breaking inducido por `gamma`,
   - log densidad de `pi` dado `alpha0 * beta`,
   - log densidad de cada fila `A_i` dado `alpha * beta + kappa * e_i`.

Es decir, optimiza:

`- log p(v | gamma) - log p(pi | beta) - sum_i log p(A_i | beta)`

con `L-BFGS-B`.

No es muestreo completo de `beta`; es MAP.

#### Modo 2: `approx_dirichlet_fallback`

Si `scipy` no esta disponible o la optimizacion falla:

1. computa `usage = bincount(states)`;
2. forma `pseudo = normalize(usage + gamma * beta + EPSILON)`;
3. samplea:
   `updated ~ Dir(pseudo * K + EPSILON)`

Es un fallback empirico, honesto y explicito, no una derivacion exacta del HDP.

### 12.6. Ciclo de Gibbs bloqueado

En cada iteracion:

1. samplear emisiones;
2. samplear transiciones;
3. samplear distribucion inicial;
4. actualizar `beta`;
5. samplear trayectoria completa con FFBS;
6. registrar log-likelihood, numero de estados activos y entropia de `beta`.

### 12.7. Entropia de `beta`

Se mide como:

`H(beta) = - sum_k beta_k log(beta_k + EPSILON)`

Sirve para diagnosticar cuan dispersa o concentrada esta la masa global entre estados potenciales.

### 12.8. Burn-in, almacenamiento y medias posteriores

Despues de `burn_in`, cada `store_every` iteraciones se acumulan:

- `transition_matrix`
- `emission_matrix`
- `beta`
- `initial_probs`
- `states`

Al final:

- si no hubo muestras posteriores, se usan los mejores parametros por log-likelihood;
- si si hubo, se calculan medias posteriores simples.

### 12.9. Mejor muestra vs media posterior

`HDPHMMResult` devuelve dos tipos de informacion:

1. una "mejor muestra" segun log-likelihood:
   - `latent_states`
   - `transition_matrix`
   - `emission_matrix`
   - `beta`
   - `initial_probs`
2. medias posteriores:
   - `posterior_transition_mean`
   - `posterior_emission_mean`
   - `posterior_beta_mean`
   - `posterior_initial_mean`

Esto es importante porque:

- la trayectoria final corresponde a la mejor iteracion, no a una trayectoria promedio;
- varias tablas y figuras usan medias posteriores, no necesariamente la matriz de la mejor muestra.

### 12.10. Diagnosticos

`HDPHMMDiagnostics` guarda:

- `log_likelihood_history`
- `active_state_history`
- `beta_entropy_history`
- `state_samples`
- `beta_update_mode`
- `best_iteration`

### 12.11. Segmentos y dwell time

`TruncatedHDPHMM.dwell_times(states)` usa segmentos contiguos para estimar duracion de permanencia por estado.

Si un estado `z` aparece en segmentos de longitudes `l_1, ..., l_m`, entonces:

- `mean_dwell = promedio(l_i)`
- `median_dwell = mediana(l_i)`

Estas metricas luego aparecen en la interpretacion de estados.

### 12.12. Complejidad computacional

Por iteracion, ignorando constantes:

- sampleo de emisiones: `O(K * V)`
- sampleo de transiciones: `O(K^2)`
- FFBS: `O(T * K^2)`

Por tanto, el costo dominante suele ser:

`O(n_iters * T * K^2)`

Esto explica por que la CLI batch usa defaults mas pequenos que la CLI individual.

## 13. Diagnosticos tabulares (`src/analysis/diagnostics.py`)

Este modulo traduce resultados de modelos a tablas exportables y metricas agregadas.

### 13.1. Estabilidad de trayectoria

`trajectory_stability(state_samples)`:

1. apila las trayectorias almacenadas post burn-in;
2. para cada posicion temporal, calcula la frecuencia relativa del estado modal;
3. promedia esa frecuencia.

Si en una columna temporal las muestras son `[2,2,2,3,2]`, la estabilidad local es `4/5 = 0.8`.

La estabilidad global es el promedio de esos valores.

Cautela importante: si hay label switching entre muestras, esta metrica puede subestimar estabilidad semantica real.

### 13.2. Estadisticas de segmentacion

`segmentation_statistics(states)` devuelve:

- numero de segmentos;
- longitud media;
- longitud mediana.

Esto permite cuantificar si un modelo segmenta de forma muy atomizada o mas estable.

### 13.3. Tablas del HMM finito

`build_finite_tables(...)` exporta:

- `observations`
- `latent_sequence`
- `summary`
- `transition_matrix_full`
- `transition_matrix_active`
- `emission_matrix_full`
- `harmonic_vocabulary`

### 13.4. Tablas del HDP-HMM

`build_hdp_tables(...)` exporta:

- `observations`
- `latent_sequence`
- `summary`
- `transition_matrix_full`
- `transition_matrix_active`
- `emission_matrix_full`
- `diagnostics`
- `interpretation` si se provee

### 13.5. Comparacion de modelos

`compare_models(...)` arma una tabla con:

- `log_likelihood`
- `effective_states`
- `n_segments`
- `mean_segment_length`
- `trajectory_stability`

En el baseline la estabilidad se fija a `1.0`, porque no hay muestreo posterior ni incertidumbre Monte Carlo.

### 13.6. Exportacion

`export_tables(...)`:

1. guarda cada tabla como CSV;
2. intenta ademas un workbook Excel;
3. limita nombres de hoja a 31 caracteres;
4. exporta con `index=True`, asi que el CSV incluye columna indice adicional.

## 14. Interpretacion musical del HDP-HMM (`src/analysis/interpretation.py`)

Este modulo hace el puente entre estados latentes sin nombre y lectura musical humana.

### 14.1. Idea

Cada estado activo `z_k` se resume por:

- ocupacion;
- permanencia;
- emisiones mas probables;
- sucesores mas probables;
- offsets ejemplo;
- candidatos armonicos;
- candidatos modales;
- una `tentative_label`.

### 14.2. Top emisiones y sucesores

Para cada estado:

- se toman las `top_n_emissions` mayores de `posterior_emission_mean[k]`;
- se toman los `top_n_successors` mayores de `posterior_transition_mean[k]`.

### 14.3. Interpretacion armonica

Si las observaciones pueden proyectarse a pitch classes, se construye un perfil armonico y se buscan:

1. mejores candidatos de acorde entre los 252 estados armonicos;
2. mejores candidatos modales condicionados a la raiz del mejor acorde.

La etiqueta tentativa sale de la forma:

- `C:maj9 en contexto C:ionian`
- o bien `contexto centrado en ...`

Si la observacion no permite proyeccion tonal (`interval`, `midi_note`), la etiqueta cae a:

`contexto latente no tonal directamente interpretable`

### 14.4. Que significa esta capa

Es crucial entender que esta interpretacion:

- no entra al sampler;
- no cambia la inferencia;
- no impone acorde = estado.

Es una lectura posterior de los patrones aprendidos.

## 15. Visualizaciones (`src/analysis/visualization.py`)

Este modulo trabaja en modo headless:

- fija `MPLCONFIGDIR` en `.mplconfig`;
- usa backend `Agg`.

Eso permite generar figuras en servidores o entornos sin GUI.

### 15.1. Figuras principales

- heatmap de transiciones;
- heatmap de emisiones;
- grafo de transiciones via `networkx`;
- timeline de estados latentes;
- para HDP-HMM, historia de log-likelihood y estados activos.

### 15.2. Grafo de transiciones

`plot_transition_graph(...)` agrega una arista `i -> j` solo si:

`A_ij >= threshold`

con `threshold = 0.05` por defecto.

Esto evita grafos saturados imposibles de leer.

### 15.3. Que matrices se usan

- finite: usa transicion empirica activa y emisiones de los estados activos;
- HDP: usa medias posteriores activas.

## 16. Catalogacion de bibliotecas (`src/data/library_catalog.py`)

Esta capa permite analizar corpus grandes y no solo una partitura.

### 16.1. Descubrimiento de archivos

`iter_musicxml_files` busca recursivamente archivos con extensiones:

- `.xml`
- `.musicxml`
- `.mxl`

### 16.2. Inferencia heuristica de metadatos

El modulo contiene diccionarios de patrones para:

- compositor (`COMPOSER_PATTERNS`);
- periodo (`PERIOD_BY_COMPOSER`);
- forma (`FORM_KEYWORDS`).

No es musicologia exhaustiva; es catalogacion practica.

### 16.3. Metricas por obra

`classify_score(path)` intenta parsear la obra y extrae:

- titulo;
- compositor;
- periodo;
- forma;
- tag de arreglo;
- bucket de dificultad;
- numero de partes;
- numero de compases;
- numero de notas;
- numero de acordes;
- duracion total en cuartos;
- promedio de notas por compas;
- numero de pitch classes unicas;
- tonalidad declarada;
- tonalidad estimada por `score.analyze("key")`;
- firmas de compas;
- error, si fallo el parseo.

### 16.4. Dificultad

`infer_difficulty_bucket` mezcla heuristica textual y densidad:

- palabras como `easy` o `beginner` fuerzan `easy`;
- obras con `note_count >= 900` o `avg_notes_per_measure >= 7.5` son `advanced`;
- obras muy pequenas y poco densas caen en `easy`;
- valores intermedios grandes caen en `intermediate-advanced`;
- resto: `intermediate`.

### 16.5. Lectura de clave

Hay dos nociones de tonalidad:

- `declared_key`: leida de `Key` o `KeySignature`;
- `estimated_key`: inferida por `music21`.

Esto es util porque algunos archivos no declaran bien la tonalidad.

## 17. Capa multicorpus (`src/data/multicorpus.py`)

Este modulo generaliza el analisis a varias fuentes heterogeneas.

### 17.1. Entidad `CorpusSource`

Cada fuente tiene:

- `name`
- `source_type`
- `root_dir`

### 17.2. SymbTr

`parse_symbtr_filename` explota la convencion:

`makam--form--usul--title--composer.ext`

y extrae:

- `makam`
- `form`
- `usul`
- `title_hint`
- `composer_hint`

`build_symbtr_catalog` ademas fija:

- `source_name = SymbTr`
- `source_type = symbtr`
- `genre_family = turkish_art_music`
- `style_system = makam`
- `modal_system = makam`

Esto no "tonaliza" SymbTr; solo preserva su procedencia y metadatos.

### 17.3. PDMX

`build_pdmx_catalog` distingue dos escenarios:

1. hay manifiesto CSV compatible:
   - intenta resolver ruta real del MusicXML;
   - filtra preferentemente `subset:no_license_conflict`;
   - injerta metadatos del manifiesto;
2. no hay manifiesto:
   - recurre a descubrimiento recursivo de archivos;
   - marca la limitacion en `ingest_note`.

Esto es importante porque el repositorio clonado de PDMX no equivale automaticamente al dataset completo de Zenodo.

### 17.4. JAZZMUS

`prepare_jazzmus_musicxml` recorre JSON y busca una codificacion `musicxml` dentro de `encodings`.

Si la encuentra:

- escribe `<stem>.musicxml` en un staging dir;
- marca `status = written` o `existing`.

Despues `build_jazzmus_catalog` cataloga esos MusicXML como:

- `genre_family = jazz`
- `style_system = lead_sheet_jazz`

### 17.5. Catalogo generico

`build_generic_catalog` simplemente cataloga MusicXML sin adaptador especializado.

### 17.6. Catalogo combinado

`build_multicorpus_catalog` concatena catalogos por fuente y ordena por:

`source_name, composer, title`

Columnas comunes multicorpus:

- `source_name`
- `source_type`
- `genre_family`
- `style_system`
- `modal_system`
- `ingest_note`

## 18. Analisis batch y multicorpus (`src/analysis/library_batch.py`)

Este modulo conecta catalogacion y modelado.

### 18.1. `analyze_catalog_piece`

Para cada obra del catalogo:

1. parsea el score;
2. extrae eventos;
3. construye observaciones;
4. ejecuta finite, hdp o ambos;
5. compacta resultados en una fila tabular.

Campos derivados que puede agregar:

- `finite_log_likelihood`
- `finite_active_states`
- `finite_top_labels`
- `finite_top_modes`
- `hdp_log_likelihood`
- `hdp_effective_states`
- `hdp_top_interpretations`
- metricas de segmentacion y estabilidad por modelo

### 18.2. `analyze_catalog`

Hace:

1. exporte de catalogo y sus resumenes;
2. filtrado de obras sin errores;
3. orden por `note_count` y `title`;
4. limite opcional;
5. ejecucion por obra;
6. exporte de `analysis.csv` y `analysis_by_composer.csv`;
7. generacion de figuras agregadas;
8. escritura de `analysis_report.md`.

### 18.3. `analyze_library`

Es un wrapper para corpus generico local.

### 18.4. `analyze_multicorpus`

Construye primero el catalogo combinado y luego reusa exactamente el mismo pipeline de `analyze_catalog`.

Eso muestra un buen diseno: una vez que todo se reduce a un `catalog` estandar, el resto del sistema es independiente de la fuente original.

## 19. Visualizacion de catalogos y corpus (`src/analysis/library_visualization.py`)

### 19.1. Figuras de catalogo

`generate_catalog_figures` produce, segun haya datos:

- obras por compositor;
- distribucion de dificultad;
- formas mas frecuentes;
- obras por fuente, si existe `source_name`.

### 19.2. Figuras comparativas de modelos

Si estan ambos modelos:

- scatter de log-likelihood finite vs hdp;
- boxplot de distribucion de log-likelihood;
- ganancia `hdp - finite` por obra;
- scatter de cantidad de estados;
- barras comparativas de complejidad armonica por pieza;
- complejidad promedio por compositor.

### 19.3. Reporte explicativo

`build_analysis_explanation` no solo describe figuras; tambien emite un juicio narrativo basado en:

- numero de obras donde HDP-HMM mejora/empeora;
- ganancia media y mediana;
- medias de log-likelihood;
- medias de numero de estados.

Esto le da al pipeline una salida discursiva, no solo tabular.

## 20. Interfaces de linea de comandos (`src/cli`)

### 20.1. `src/cli/main.py`

Es la CLI principal para una obra individual.

Argumentos clave:

- `--input`
- `--obs`
- `--model`
- `--K`
- `--iters`
- `--burn-in`
- `--alpha`
- `--alpha0`
- `--gamma`
- `--eta`
- `--kappa`
- `--seed`
- `--output-dir`
- `--prefer-treble`

Los parametros probabilisticos solo afectan al HDP-HMM.

### 20.2. `src/cli/library_analysis.py`

Recibe:

- `--library-dir`
- `--output-dir`
- `--obs`
- `--model`
- `--limit`
- `--prefer-treble`
- `--K`
- `--iters`
- `--burn-in`
- `--seed`

### 20.3. `src/cli/multicorpus_analysis.py`

Extiende el batch para multiples fuentes:

- `--include-library`
- `--include-symbtr`
- `--include-pdmx`
- `--include-jazzmus`
- `--sample-per-source`
- `--limit`
- mas los argumentos de modelado.

La funcion `_collect_sources` obliga a especificar al menos una fuente.

## 21. Artefactos de salida

### 21.1. Corrida individual (`src/cli/main.py`)

En `output_dir/common/`:

- `events.csv`
- `observations.csv`
- `harmonic_vocabulary.csv`
- `modal_contexts.csv`
- `common.xlsx`

En `output_dir/finite_hmm/` cuando aplica:

- `observations.csv`
- `latent_sequence.csv`
- `summary.csv`
- `transition_matrix_full.csv`
- `transition_matrix_active.csv`
- `emission_matrix_full.csv`
- `harmonic_vocabulary.csv`
- `finite_hmm.xlsx`
- `finite_transition_heatmap.png`
- `finite_emission_heatmap.png`
- `finite_transition_graph.png`
- `finite_timeline.png`

En `output_dir/hdp_hmm/` cuando aplica:

- `observations.csv`
- `latent_sequence.csv`
- `summary.csv`
- `transition_matrix_full.csv`
- `transition_matrix_active.csv`
- `emission_matrix_full.csv`
- `diagnostics.csv`
- `interpretation.csv`
- `hdp_hmm.xlsx`
- `hdp_transition_heatmap.png`
- `hdp_emission_heatmap.png`
- `hdp_transition_graph.png`
- `hdp_timeline.png`
- `hdp_log_likelihood.png`
- `hdp_active_states.png`

En la raiz de `output_dir/`:

- `comparison.csv` y/o `comparison.xlsx`
- `summary.txt`

### 21.2. Corrida batch / multicorpus

En `catalog/`:

- `catalog.csv`
- resumenes por compositor, forma, dificultad, arreglo, fuente
- `catalog.xlsx`
- figuras de catalogo

En `analysis/`:

- `analysis.csv`
- `analysis_by_composer.csv` si aplica
- `analysis.xlsx`
- figuras comparativas
- `analysis_report.md`

## 22. Ejemplo reproducible

`examples/generate_example_score.py` construye una mini pieza con una secuencia de notas que sugiere:

- `C:maj9`
- `A:m7`
- `D:m9`
- `G:9`
- `C:add9`

No escribe acordes verticales explicitos: escribe una linea melodica/arpegiada que induce esas sonoridades.

Es un detalle importante porque el proyecto no requiere verticalidad densa para hacer inferencia; puede trabajar con secuencias melodicas discretizadas.

## 23. Validacion automatizada

Las pruebas del repositorio cubren varios invariantes:

- parsing correcto de nota y acorde;
- construccion correcta de observaciones;
- existencia del vocabulario armonico extendido;
- deteccion modal y armonica basica;
- ejecucion consistente del HMM finito;
- ejecucion consistente del HDP-HMM truncado;
- inferencia heuristica de catalogo;
- parsing de nombres SymbTr;
- exportacion de MusicXML desde JSON JAZZMUS;
- mezcla de fuentes en catalogo multicorpus.

En el entorno actual del proyecto, la suite se pudo validar con:

```bash
python -m unittest discover -s tests -q
```

resultado:

- `Ran 17 tests`
- `OK`

`pytest` no estaba disponible en el entorno en el momento de esta documentacion, pero no hizo falta porque las pruebas ya estan escritas sobre `unittest`.

## 24. Decisiones de diseno mas importantes

### 24.1. Separar armonia explicita y contexto modal

El proyecto no multiplica el espacio de estados como:

`acorde x modo`

porque eso dispararia el tamano del baseline.

En vez de eso:

- el acorde vive en el estado explicito;
- el modo vive como capa contextual interpretativa.

### 24.2. No forzar semantica tonal sobre estados del HDP-HMM

Esto es una decision epistemicamente sana:

- inferencia estadistica primero;
- interpretacion musical despues.

### 24.3. Reducir partituras polifonicas a flujo temporal comun

Es una simplificacion fuerte, pero hace viable:

- pipeline uniforme;
- observaciones discretas simples;
- HMMs clasicos sin modelado multivoz complejo.

### 24.4. Mantener un baseline fuerte y explicable

El HMM finito no es un strawman. Tiene:

- vocabulario armonico amplio;
- sesgos de transicion musicalmente razonables;
- emisiones estructuradas;
- lectura modal local.

Eso hace que compararlo contra el HDP-HMM tenga mas valor metodologico.

## 25. Limitaciones reales del sistema

### 25.1. El baseline no aprende

Es interpretable, pero no adapta sus parametros al corpus.

### 25.2. El HDP-HMM no es un iHMM exacto

Hay dos aproximaciones claras:

1. truncacion en `K`;
2. update de `beta` por MAP o fallback empirico.

### 25.3. Posible label switching

Como en muchos modelos Bayesianos con estados intercambiables, los labels de estados latentes pueden permutarse entre muestras. Eso afecta lectura directa de estabilidad por posicion.

### 25.4. Reduccion a un solo flujo

El sistema pierde informacion estructural de:

- voces simultaneas;
- orquestacion;
- distribucion vertical detallada.

### 25.5. Observaciones categoricas

No hay:

- emisiones continuas;
- duraciones modeladas por distribuciones parametricas;
- dependencias factoriales entre canales;
- estructura jerarquica por compas/frase/seccion.

### 25.6. Interpretacion musical heuristica

Las etiquetas tentativas del HDP-HMM no son inferencia posterior exacta sobre acordes; son matching posterior contra el vocabulario armonico.

### 25.7. Compatibilidad intercultural limitada

El pipeline puede ingerir `SymbTr`, pero el vocabulario armonico del baseline sigue siendo de corte tonal occidental. En repertorio makam, las etiquetas armonicas deben leerse como aproximaciones descriptivas, no como teoria nativa del sistema musical.

## 26. Complejidad conceptual del proyecto

Una buena forma de entender el repo es pensar en tres capas de abstraccion:

### Capa A: simbolica

- leer notas y acordes;
- convertirlos a eventos;
- tokenizarlos.

### Capa B: probabilistica

- HMM finito con estados armonicos fijos;
- HDP-HMM truncado con estados latentes aprendidos.

### Capa C: musicologica

- acordes candidatos;
- modos candidatos;
- complejidad armonica;
- comparacion por corpus/compositor/fuente.

El valor del proyecto esta precisamente en que esas tres capas estan separadas pero conectadas.

## 27. Como leer una corrida correctamente

### Si corres el baseline

Debes leer:

- `summary.csv` para metricas globales;
- `latent_sequence.csv` para trayectoria armonica;
- `transition_matrix_active.csv` para flujo efectivo entre estados usados;
- `finite_emission_heatmap.png` para entender el sesgo tonal de cada estado.

### Si corres el HDP-HMM

Debes leer:

- `summary.csv` para ajuste y numero de estados efectivos;
- `diagnostics.csv` y `hdp_log_likelihood.png` para mezcla y evolucion;
- `interpretation.csv` para semantica tentativa;
- `hdp_transition_heatmap.png` y `hdp_timeline.png` para dinamica latente.

### Si corres ambos

Debes revisar:

- `comparison.csv`
- diferencia de log-likelihood;
- diferencia de estados efectivos;
- diferencia de segmentacion;
- interpretabilidad relativa de resultados.

## 28. Formula corta de cada pieza del sistema

Si hubiera que resumir matematicamente el proyecto en una sola tabla conceptual:

### Parsing

`score -> events`

### Observaciones

`events -> y_1:T`

### Finite HMM

`P(z_1) = uniforme`

`P(z_t | z_(t-1)) = heuristica musical normalizada`

`P(y_t | z_t) = distribucion acorde/modo heuristica`

`z*_1:T = Viterbi(y_1:T)`

### HDP-HMM truncado

`beta ~ stick-breaking(gamma)`

`pi ~ Dir(alpha0 * beta)`

`A_i ~ Dir(alpha * beta + kappa * e_i)`

`phi_i ~ Dir(eta / V, ..., eta / V)`

`z_1:T ~ FFBS condicionado en (pi, A, phi)`

### Interpretacion

`estado latente -> perfil de emision -> candidatos armonicos/modales`

## 29. En que destaca el proyecto

Tecnicamente, el proyecto destaca porque ya no es un notebook desordenado sino un sistema con:

- modularidad clara;
- pruebas automatizadas;
- soporte de varios corpus;
- separacion entre inferencia y narrativa musical;
- baseline interpretable fuerte;
- modelo Bayesiano mas flexible para comparacion seria.

Metodologicamente, destaca porque evita vender como "exacto" lo que en realidad es una aproximacion truncada. Esa honestidad aparece tanto en `README.md` como en `NOTES.md` y en el propio codigo.

## 30. Resumen ejecutivo final

`Melodies` es un laboratorio serio para analisis de musica simbolica basado en secuencias discretas. Su baseline finito representa armonia explicita por un vocabulario de 252 estados y resuelve la trayectoria con Viterbi. Su modelo principal implementa un HDP-HMM truncado con FFBS, posterior Dirichlet para transiciones y emisiones, y una actualizacion aproximada de pesos globales `beta`. Sobre eso, el proyecto monta una capa completa de interpretacion musical, visualizacion, analisis batch y comparacion multicorpus.

Dicho de otro modo: el repo ya tiene tanto una espina dorsal matematica como una espina dorsal de software. No es solo codigo que corre; es una arquitectura coherente para estudiar patrones armonicos latentes en repertorios simbolicos heterogeneos.

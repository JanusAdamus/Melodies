# Costo computacional del experimento de curva de aprendizaje

Reporte sobre por qué la corrida `canonical_20260820_2102` no terminó, qué la limita, y
qué tamaño de corpus admite realmente la implementación actual.

Fecha: 2026-08-22. Máquina de referencia: 12 núcleos lógicos, 32 GB RAM, Windows 11.

---

## 1. Resumen

La corrida canónica llevaba ~36 h y no había escrito ningún resultado. No estaba colgada:
seguía en la primera etapa, el parseo del corpus, que es **serial** sobre 254,035 archivos.

El diagnóstico importante no es el parseo. Es que la etapa de ajuste de modelos, con la
configuración canónica, cuesta del orden de **480 días de CPU**. La corrida no iba a
terminar con ninguna cantidad de paciencia.

La causa no es el tamaño del corpus ni la complejidad asintótica de los algoritmos, que es
lineal en el número de tokens. Es la **constante por token**: ~77 µs, unas cien veces más
de lo necesario.

El origen está localizado y medido (§4.2): la recursión forward-backward está escrita en
dominio logarítmico, y cada paso temporal aplica `logsumexp` —cinco recorridos con `exp` y
`log`— sobre un bloque de `K²` elementos. Reformularla con reescalado en dominio lineal
convierte ese paso en un producto matricial: **~100× más rápido, con error de 1.4e-08 en la
log-verosimilitud**, sin dependencias nuevas ni cambios al método estadístico.

Implementado y medido de punta a punta: el piloto completo pasó de **110 min a 23.5 min**,
con `finite_hmm` dando perplejidad idéntica a 6e-09 (§7). El corpus completo con malla
reducida queda en ~10 días, contra los ~406 días que costaba antes.

---

## 2. Notación y parámetros del corpus

| Símbolo | Significado | Valor medido |
|---|---|---|
| `N` | partituras descubiertas | 254,035 `.mxl` (PDMX, 1.9 GB) |
| — | tasa de aceptación tras exclusiones | 97.75 % (391/400 en muestra aleatoria) |
| `T̄` | tokens por pieza (media) | 262.8 |
| `L` | tokens totales del corpus | ≈ 65.3 M |
| `L_pool` | tokens del *train pool* (75 % tras test 15 % + val 10 %) | ≈ 48.9 M |
| `V` | vocabulario (`pitch_class` + BOS) | 13 |
| `K` | estados ocultos | finito ∈ {12, 24, 48}; HDP truncado a 40 |
| `I` | iteraciones | EM 100; Gibbs 120 (burn-in 60) |
| `D`, `M`, `G` | `data_seeds`, `model_seeds`, malla de hiperparámetros HDP | 3, 2, 3 |
| `Σf` | suma de `train_fractions` (0.10+0.25+0.50+0.75+1.00) | 2.60 |

Exclusiones observadas en la muestra: 6 `too_short`, 2 `parse_error`, 1
`event_extraction_error`.

---

## 3. Etapa 1 — preparación del corpus

`prepare_corpus` (`next_token_experiment/data/preprocess.py`) parsea cada partitura con
music21 y la tokeniza. El costo es **O(N)** en número de archivos, dominado por
`music21.converter.parse` (`src/data/parsing.py:37`).

Tiempos medidos sobre muestra aleatoria de PDMX:

```
media 0.308 s   mediana 0.122 s   p90 0.715 s   máximo 4.753 s
```

La distribución tiene cola pesada: la media es 2.5× la mediana. Eso importa para el
reparto de carga en paralelo.

### Escalado paralelo medido (2,500 archivos)

| Workers | chunksize | Tiempo | Eficiencia | Corpus completo |
|---|---|---|---|---|
| 1 (serial) | — | 740 s | 100 % | **20.9 h** |
| 6 | 4 | 190.4 s | 65 % | 5.4 h |
| 11 | 16 | 143.6 s | 47 % | 4.1 h |
| 11 | 4 | 138.4 s | 49 % | 3.9 h |
| 11 | 1 | 138.7 s | 49 % | 3.9 h |
| 12 | 4 | 131.2 s | 47 % | **3.7 h** |

La eficiencia se queda en ~47 % en vez de acercarse a 100 %. Tres causas, en orden de
peso: en Windows el arranque usa *spawn*, así que cada worker reimporta music21 desde
cero; los tiempos por archivo son muy dispares y desbalancean el reparto; y la
descompresión de los `.mxl` es I/O.

`chunksize` prácticamente no influye (4, 1 y 16 quedan dentro del ruido), lo que confirma
que el cuello no es el tráfico entre procesos.

### Cambios aplicados

- Paralelización con `ProcessPoolExecutor`, `n_workers` por defecto = núcleos − 1.
- Caché JSONL incremental (`cache_path`): cada partitura se anexa al terminar. Una
  interrupción reanuda desde donde iba en vez de reparsear todo. Medido: 49.8 s en frío
  contra 6.3 s en caliente sobre 400 piezas.
- Warnings de music21 silenciados en los workers. La corrida canónica generó 510 KB de
  `MusicXMLWarning`, cuyo formateo y escritura cuesta tiempo real y no dice nada accionable.
- El `vocabulary` corpus-wide ya no se replica: es idéntico en las 254 k piezas y se
  guardaba una copia por pieza. Los workers lo omiten y el proceso padre comparte una sola
  instancia.
- Muestreo aleatorio con semilla (`sample_seed`). Sin él, `max_files` toma las primeras N
  en orden de ruta, que sigue la estructura de directorios de PDMX y no es una muestra
  representativa del corpus.
- Progreso a stderr. Antes no había forma de distinguir avance de bloqueo.

Resultado: **20.9 h → 3.7 h**, y reanudable.

---

## 4. Etapa 2 — ajuste de modelos

Aquí está el problema real.

### 4.1 Estructura de los bucles

**HDP-HMM** (`src/models/hdp_hmm.py:310`): en cada iteración de Gibbs se hace FFBS sobre
*cada* secuencia. `ffbs_sample` (`src/models/inference.py:52`) hace dos recorridos —
filtrado hacia adelante (`inference.py:28`) y muestreo hacia atrás (`inference.py:46`)—,
ambos como bucles Python sobre los `T` pasos temporales, con una operación numpy de `K×K`
por paso.

**HMM finito** (Baum-Welch, `Comparacion/classical_models.py:193`): por iteración y por
secuencia hay **cuatro** bucles Python sobre `T`: forward (`:208`), backward (`:212`),
acumulación de emisiones (`:217`) y acumulación de transiciones (`:219`).

En ambos casos la complejidad asintótica es `O(I · L · K²)`, que es la esperada. El
problema es el factor constante.

### 4.2 Costo por token medido

`T = 263`, `V = 13`, promedio de 40 repeticiones:

| `K` | forward | FFBS (forward + backward) |
|---|---|---|
| 12 | 18.25 µs/token | 53.59 µs/token |
| 24 | 24.43 µs/token | 60.38 µs/token |
| 40 | 39.59 µs/token | **77.46 µs/token** |
| 48 | 46.76 µs/token | 76.42 µs/token |

Cuadruplicar `K` de 12 a 48 solo multiplica el costo por 2.6, cuando `K²` predice 16×.
Eso sugiere un costo fijo por paso temporal. La hipótesis inmediata —que lo domina el
intérprete de Python— resultó **falsa**, y vale la pena documentar la refutación porque
determina cuál es el arreglo correcto.

### Qué NO es el problema: el bucle Python

Si el costo fuera el intérprete, procesar las secuencias por lotes (un tensor `(B, K)`, con
el bucle recorriendo solo `T`) lo colapsaría. Medido:

| Lote `B` | Por secuencia | Por lotes | Ganancia |
|---|---|---|---|
| 64 | 33.19 µs/token | 38.61 µs/token | 1× |
| 512 | 37.12 µs/token | 37.21 µs/token | 1× |
| 2048 | 36.55 µs/token | 37.84 µs/token | 1× |

**Cero ganancia.** El bucle Python no es el cuello.

### Qué SÍ es el problema: `logsumexp`

El costo está en la formulación en dominio logarítmico. Cada paso temporal construye
`alpha[:, None] + log_transition`, una matriz `B×K×K`, y le aplica `logsumexp` — que
recorre ese arreglo cinco veces y evalúa `exp` y `log` sobre cada elemento. Son funciones
transcendentales: decenas de nanosegundos cada una, sobre `B·K²` elementos por paso.

La aritmética *sí* domina. Pero es aritmética innecesaria.

### La corrección: reescalado en dominio lineal

La formulación estándar de Rabiner (1989) evita el dominio logarítmico: se mantiene `alpha`
en espacio lineal y se renormaliza en cada paso, acumulando el logaritmo de los factores de
escala. El underflow queda controlado igual, pero el paso temporal se vuelve

```python
alpha = (alpha @ transition) * emission[:, observations[:, step]]
scale = alpha.sum(axis=1, keepdims=True)
alpha /= scale
log_likelihood += np.log(scale[:, 0])
```

Un GEMM `(B,K) @ (K,K)` más una normalización. **Ninguna transcendental toca el bloque
`K×K`**; `log` se aplica solo a los `B` factores de escala. BLAS hace el trabajo, con SIMD
y multihilo.

Medido, contra la implementación actual:

| `K` | `B` | Actual | Reescalado | Ganancia | Error máx. en log-verosimilitud |
|---|---|---|---|---|---|
| 40 | 512 | 38.34 µs/tok | 0.342 µs/tok | **112×** | 1.42e-08 |
| 40 | 2048 | 37.29 µs/tok | 0.367 µs/tok | **102×** | 1.42e-08 |
| 48 | 512 | 46.08 µs/tok | 0.363 µs/tok | **127×** | 1.63e-08 |
| 48 | 2048 | 46.62 µs/tok | 0.513 µs/tok | **91×** | 1.64e-08 |

Mismo resultado a 1e-08, dos órdenes de magnitud más rápido, sin dependencias nuevas.

El procesamiento por lotes es **necesario pero no suficiente**: por sí solo no da nada, y
su papel real es que el GEMM sea lo bastante grande para que BLAS lo paralelice.

### 4.3 Proyección para la configuración canónica

Visitas token × iteración, con `L_pool = 48.9 M`:

```
HDP-HMM:     L_pool × Σf(2.60) × D(3) × M(2) × G(3) × I(120) = 2.75e11
HMM finito:  L_pool × Σf(2.60) × D(3) × M(2) × 3 candidatos × I(100) = 2.29e11
```

Multiplicando por el costo medido por token:

| Modelo | Costo estimado |
|---|---|
| HDP-HMM (77.46 µs/token a `K`=40) | ≈ 246 días |
| HMM finito (≈ 268 µs/token sumando los 3 candidatos) | ≈ 237 días |
| **Total** (sin contar VOMM ni Transformer) | **≈ 480 días de CPU** |

Es decir: aunque el parseo hubiera terminado en 21 h, faltaba más de un año de cómputo.
El experimento nunca fue ejecutable con esta configuración.

> El costo del HMM finito lleva una estimación estructural (cuatro recorridos Python por
> token frente a los dos de FFBS). Se está validando contra los `fit_wall_clock_s` reales
> del piloto; ver §7.

### 4.4 Dónde está el desperdicio

En orden de impacto medido:

1. **Reescalado en dominio lineal, por lotes** (§4.2). Sustituye `logsumexp` sobre `B·K²`
   elementos por un GEMM. **~100× medido**, error 1.4e-08. Aplica igual al forward, al
   backward y al muestreo hacia atrás del FFBS, que comparten la misma estructura.
   Es el arreglo, y no requiere dependencias nuevas ni cambiar el método estadístico.

2. **Acumulaciones vectorizables** (`classical_models.py:217-227`). El bucle de emisiones
   es un `np.add.at`; el de transiciones, un `einsum` sobre todos los pasos. Sin esto, tras
   aplicar (1) pasan a dominar el Baum-Welch: quedarían como los únicos recorridos por
   token que sobreviven.

3. **`inference.py:29` recalcula `np.log(transition_matrix + EPSILON)` dentro del bucle de
   pasos temporales.** La matriz no cambia durante el recorrido. Medido a `K`=40: 8.52 µs
   por llamada contra 39.59 µs del paso completo, **21 % del forward desperdiciado**. Queda
   subsumido por (1), que elimina el logaritmo de ese punto por completo; se documenta
   porque explica parte del costo actual.

Con (1)+(2) el corpus completo pasa de inviable a ejecutable. Sin ellas, ninguna cantidad
de paralelismo alcanza: son bucles seriales dentro de cada ajuste.

A esto se suma el muestreo categórico del FFBS, que es un tercio del costo del HDP-HMM y
un problema aparte del `logsumexp`: ver §4.5.

### 4.5 Hallazgos secundarios, medidos

Auditando el resto del código numérico aparecieron seis puntos más. Todos medidos, no
supuestos:

| Punto | Ubicación | Actual → propuesto | Ganancia | Exactitud |
|---|---|---|---|---|
| Muestreo categórico por paso (Gumbel-max) | `utils.py:60` | 6.64 → 0.29 ms | **23×** | exacto |
| Conteo de transiciones (`bincount`) | `utils.py:106` | 18.47 → 0.27 ms | **68×** | idéntico |
| `dirichlet_logpdf` (`gammaln` vectorizado) | `utils.py:94` | 52.52 → 0.05 ms | **1061×** | Δ 5.7e-14 |
| `stick_breaking_from_v` (`cumprod`) | `utils.py:65` | 33.32 → 0.18 ms | **187×** | exacto |
| Bootstrap CI, 10k remuestreos | `statistics.py:68` | 122.29 → 26.89 ms | **5×** | equivalente |
| Muestreo de filas Dirichlet | `hdp_hmm.py:177` | 0.59 → 0.62 ms | **1×** | — |

El más importante es el primero. `sample_categorical_from_log_probs` llama a
`rng.choice(..., p=...)` **una vez por paso temporal** dentro del muestreo hacia atrás del
FFBS (`inference.py:48`). `rng.choice` con `p=` revalida y acumula la distribución en cada
llamada. Medido: 25 µs/token, es decir **un tercio de los 77 µs/token del FFBS**, y es un
costo independiente del `logsumexp` de §4.2.

El truco de Gumbel-max lo elimina de raíz: `argmax(log_probs + gumbel)` es una muestra
categórica exacta, sin normalizar y sin acumular, y se aplica al lote entero de una vez.

```python
states = np.argmax(log_scores + rng.gumbel(size=log_scores.shape), axis=1)
```

Sumando §4.2 y este punto, el FFBS pasa de 77 µs/token a fracciones de µs.

La última fila se incluye porque **la medición contradijo la hipótesis**: parecía que
reemplazar el bucle de `rng.dirichlet` por una sola extracción gamma ganaría, y no gana
nada. `rng.dirichlet` ya es eficiente. No vale la pena tocarlo.

### 4.6 Observaciones numéricas

Ninguna de estas es un error de resultados hoy, pero las tres son deuda que conviene saldar
junto con lo anterior:

1. **Dos implementaciones divergentes de `logsumexp`.** `utils.py:42` hace
   `np.log(summed + EPSILON)`; `classical_models.py:78` hace
   `np.log(np.maximum(summed, EPSILON))`. Misma función, guardas distintas. Además, tras
   restar el máximo, `summed ≥ 1` siempre, así que en la primera el `EPSILON` es inerte.
   El caso que sí difiere es el vector todo `-inf`: la versión de `utils` produce `nan`
   (por `-inf − (-inf)`), la otra devuelve un valor finito. Debería ser una sola función.

2. **Hoisting inconsistente del logaritmo.** `classical_models.py:71` saca
   `np.log(transition_matrix + EPSILON)` fuera del bucle temporal; `inference.py:29` no.
   Es el mismo algoritmo escrito dos veces con criterios distintos.

3. **`EPSILON` dentro de los logaritmos de probabilidad.** `np.log(matrix + 1e-12)` impone
   un piso: una transición de probabilidad cero puntúa −27.6 en vez de −∞. Es suavizado, y
   probablemente deseado, pero hoy es incidental y no está documentado como decisión de
   modelado. Con el reescalado de §4.2 hay que fijarlo explícitamente, porque en dominio
   lineal el piso deja de aplicarse solo.

### 4.7 Sin resultados parciales — resuelto

`run_learning_curve_experiment` acumulaba `raw_rows` en memoria y escribía los CSV al
final. Era el mismo defecto que hizo que 36 h de parseo no dejaran nada recuperable: si la
corrida moría en la celda 14 de 15, se perdían las 14.

Ahora cada celda `(data_seed, fraction)` anexa sus filas a `checkpoint.jsonl` al terminar,
y `--resume` las salta en vez de reajustarlas. Con la malla canónica son 15 celdas, así que
la pérdida máxima baja de la corrida entera a **una celda**.

Es append, nunca reescritura del archivo completo. Una línea truncada por una muerte a
mitad de escritura se descarta al reanudar y esa celda se recalcula.

Granularidad de celda y no de modelo a propósito: bajar a modelo individual reduciría la
pérdida máxima de ~100 a ~15 minutos, pero obliga a reindentar los cuatro bloques del bucle
de ajuste. No compensa.

Verificado contra datos reales, no solo con mocks: se mató una corrida tras 2 de 4 celdas y
al reanudar reportó `reanudando: 2 de 4 celdas ya hechas`, saltando directo a las
pendientes.

---

## 5. Qué tamaño de corpus admite la implementación actual

Del modelo de costo, por token del *train pool* y por celda (`data_seed` × `model_seed` ×
unidad de fracción):

```
HDP-HMM:     G(3) × I(120) × 77.46 µs  = 0.0279 s/token
HMM finito:  I(100) × 268 µs           = 0.0268 s/token
                                  total ≈ 0.0547 s/token
```

Entonces:

```
segundos_totales ≈ L_pool × Σf × D × M × 0.0547
```

Despejando `N` para distintos objetivos:

| Configuración | `D` | `M` | `Σf` | Para 6 h | Para 24 h |
|---|---|---|---|---|---|
| Canónica | 3 | 2 | 2.60 | ~130 partituras | ~510 partituras |
| Reducida (opciones 1+2) | 1 | 1 | 1.75 | **~1,150 partituras** | ~4,600 partituras |

Ese es el veredicto incómodo: **con la implementación actual el experimento soporta del
orden de mil partituras, no 254 mil.**

### Con §4 implementado, medido de punta a punta

El piloto de 400 partituras corrió completo antes y después. **110 min → 23.5 min.**

Extrapolando linealmente desde el piloto (248,320 piezas = 635× el piloto):

| Configuración | Corpus completo (254 k) | En 24 h |
|---|---|---|
| Canónica (`D`=3, `M`=2, `Σf`=2.60) | ≈ 85 días | ~2,900 partituras |
| Reducida (`D`=1, `M`=1, `Σf`=1.75) | **≈ 10 días** | ~26 k partituras |

Corrección honesta: una versión previa de esta nota proyectó 13 h para el corpus completo
con malla reducida. Ese número asumía que el 100× del microbenchmark de §4.2 se trasladaba
de punta a punta, y no lo hizo — la ganancia real es 4.8× sobre el tiempo total de ajuste.
La diferencia está explicada en §7.

La extrapolación lineal es además conservadora: el speedup **crece** con el tamaño (4.2× en
`frac`=0.1 contra 9.0× en `frac`=1.0), porque los lotes amortizan mejor cuantas más
secuencias hay. El corpus completo debería rendir mejor que 10 días.

---

## 6. Plan aplicado (opciones 1 + 2)

**Opción 1 — submuestrear el corpus.** `--max-files` con `--corpus-sample-seed`, para que
la muestra sea aleatoria y reproducible en vez de las primeras N en orden de directorio.

**Opción 2 — recortar la malla.** `--data-seeds 1 --model-seeds 1` y menos fracciones.
Factor de reducción: `D` 3→1, `M` 2→1, `Σf` 2.60→1.75, es decir **8.9×**.

Comando del piloto:

```bash
python -m Comparacion.cli \
  --run-name pilot_400 \
  --corpus-root external/PDMX/mxl \
  --max-files 400 --corpus-sample-seed 7 \
  --data-seeds 1 --model-seeds 1 \
  --fractions 0.1,0.25,0.5,1.0 \
  --n-workers 12 \
  --corpus-cache artifacts/corpus_cache_pdmx.jsonl
```

Costo predicho por el modelo: `L_pool` = 77,060 tokens × Σf(1.85) × 0.0547 ≈ **2.2 h**.

Nota metodológica: reducir `data_seeds` y `model_seeds` a 1 elimina las barras de error del
experimento. Sirve para calibrar tiempos y validar el pipeline; **no** para las cifras
finales de la tesis. Las réplicas hay que recuperarlas una vez que §4.4 abarate cada ajuste.

---

## 7. Implementado

Todo con `tests/test_scaled_inference.py` como prueba de regresión: reproduce el paso E y
la log-verosimilitud del código en dominio logarítmico que reemplaza, con tolerancia
relativa 1e-08.

| Cambio | Archivo | Medido |
|---|---|---|
| FFBS reescalado por lotes + Gumbel-max | `src/models/inference.py` | **38×** aislado, error 1.6e-08 |
| Paso E de Baum-Welch reescalado por lotes | `Comparacion/classical_models.py` | **46×** aislado, conteos a 1e-11 |
| Agrupación por longitud (`length_buckets`) | `src/models/utils.py` | **2.6×** adicional |
| `scaled_forward_log_likelihood` en evaluación | `src/models/inference.py` | 7× sobre esa ruta |
| `count_transitions` / `count_emissions` con `bincount` | `src/models/utils.py` | 68×, idéntico |
| `dirichlet_logpdf` con `gammaln` | `src/models/utils.py` | 1061×, Δ 5.7e-14 |
| `stick_breaking_from_v` con `cumprod` | `src/models/utils.py` | 187×, exacto |

### Resultado end-to-end: el piloto completo, antes y después

Misma configuración, mismo corpus, mismas semillas. **110 min → 23.5 min.**

| Modelo | ppl antes | ppl después | Δ | ajuste antes | después | speedup |
|---|---|---|---|---|---|---|
| `finite_hmm` `frac`=1.0 | 8.639 | 8.639 | **5.9e-09** | 1033.9 s | 115.0 s | **9.0×** |
| `hdp_hmm` `frac`=1.0 | 8.901 | 8.767 | −1.5 % | 2019.9 s | 233.3 s | **8.7×** |
| `transformer` `frac`=1.0 | 8.310 | 8.310 | 0.00 | 216.2 s | 235.3 s | 0.9× |
| `vomm` `frac`=1.0 | 10.054 | 10.054 | 0.00 | 3.7 s | 3.8 s | 1.0× |
| **total ajuste** | | | | **6,579 s** | **1,382 s** | **4.8×** |

`finite_hmm` es determinista dado el inicio y sale idéntico a 6e-09: el reescalado no movió
el resultado. `vomm` y `transformer`, cuyo código no se tocó, salen exactamente iguales, lo
que confirma que no se movió nada más. `hdp_hmm` difiere porque es un muestreador de Gibbs
y Gumbel-max consume el generador distinto: son dos cadenas válidas, no una regresión — y
la nueva da perplejidad **menor** en `frac` 0.5 y 1.0.

El total es 4.8× y no 9× porque el Transformer, que no se tocó, pasó a ser el 37 % del
tiempo de ajuste. Es el siguiente cuello, y es PyTorch en CPU: se ataca con GPU, no
reescribiendo bucles.

### Por qué 9× y no el 100× del microbenchmark

El 100× de §4.2 es el forward aislado. De punta a punta sobrevive menos, por tres razones,
en orden de peso:

1. **Relleno.** Las piezas van de 33 a 4,433 tokens (mediana 94). Un lote plano procesa
   16.1× más celdas de relleno que de datos. Cerrar el lote por presupuesto de celdas
   (`n_secuencias × longitud_máxima ≤ 100k`) tras ordenar por longitud es lo que rescata
   esa pérdida: en `FiniteGlobalHMM` bajó el ajuste de 17.7 s a 6.8 s, **2.6× que no está
   en la formulación sino en cómo se agrupa**. Fue un error propio: el agrupado se aplicó
   primero solo al Baum-Welch y no al FFBS, y con eso el HDP corría más lento que antes.
   Ahora ambos usan `src/models/utils.py:length_buckets`.
2. **La recursión es secuencial en `T`.** Ningún lote elimina el recorrido temporal. Tras
   agrupar, el perfil da 5,086 pasos de bucle para 40,714 tokens: 8× de amortización, no
   infinita.
3. **El paso M y la evaluación** no se reescalan: muestreo de Dirichlet, actualización de
   `beta` por L-BFGS, y la evaluación en validación cada iteración EM.

El speedup crece con el tamaño del problema —4.2× en `frac`=0.1 contra 9.0× en `frac`=1.0—
precisamente porque los lotes amortizan mejor cuantas más secuencias hay.

### Aparte: el audit de protocolo pesaba de más

`_append_protocol_evidence` guardaba `expected_event_indices` y `scored_event_indices` por
pieza, por modelo, por fracción y por semilla. En una corrida que pasa ambas son siempre
`list(range(n))` idénticas: cero información, 8.8 MB con 400 partituras y una sola semilla.
Peor, reescribía la lista entera —que crece— en cada bloque, con costo cuadrático de I/O.

Ahora guarda contadores y los booleanos de mismatch, y los índices completos **solo cuando
algo falla**, que es donde sirven como diagnóstico. Medido: de ~9.5 KB a **410 bytes por
entrada, 23×**. La escritura quedó una sola vez al cerrar, salvo en fallo, donde se adelanta
para que el diagnóstico sobreviva a la excepción.

Proyectado sobre la corrida de 3,000 con malla canónica: de ~440 MB a ~19 MB.

### Aparte: `matplotlib` sin backend

`Comparacion/runner.py` importaba `pyplot` sin fijar backend, mientras
`src/analysis/visualization.py` y `library_visualization.py` sí usan `matplotlib.use("Agg")`.
La curva de aprendizaje se dibuja **al final** de la corrida, así que la corrida canónica
habría fallado con `TclError` después de todo el cómputo. Una línea, y la suite pasa de
99 a 100.

### Descartado a propósito

- **Bootstrap vectorizado** (5×): corre una sola vez al final, 122 ms. No vale el diff.
- **Muestreo de filas Dirichlet**: medido, 1×. `rng.dirichlet` ya es eficiente.

## 8. Pendiente

- **Decidir tamaño de corpus.** Con la malla reducida, 24 h compran ~26 k partituras; el
  corpus completo son ~10 días. Es la decisión abierta.
- **El Transformer es ahora el 37 % del ajuste** y no se tocó. Es PyTorch en CPU
  (`--transformer-device cuda` ya existe en el CLI). Es el siguiente cuello y se ataca con
  hardware, no con código.
- Unificar las dos implementaciones de `logsumexp` y fijar la política de suavizado (§4.6).
  Con el reescalado, el piso de `EPSILON` dentro de los logaritmos dejó de aplicarse solo en
  la ruta caliente; conviene volverlo explícito antes de la corrida final.
- Recuperar `data_seeds` y `model_seeds` > 1 para tener barras de error. El piloto corrió
  con 1 y 1, así que sus cifras no son publicables.

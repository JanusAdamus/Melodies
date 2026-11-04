# Preregistro de las corridas corregidas, 2026-10-26

Se escribe y se sube antes de cualquier corrida de la Fase 2 del plan
[`superpowers/plans/2026-10-26-correccion-tesis.md`](superpowers/plans/2026-10-26-correccion-tesis.md).
Cualquier cambio posterior se registra como enmienda fechada en este archivo,
antes de ver el resultado que afecte.

## Decisiones de Adrián (2026-10-26)

1. Melodía = voz superior de la parte en clave de sol; ligaduras unidas
   (`stripTies`); sin notas de adorno ni cifrados (`harmony.Harmony`).
2. El HDP-HMM se compara contra un HMM finito ajustado (Baum-Welch), un bigrama
   y un unigrama, en held-out.
3. Lo que corre en CPU se ejecuta en la laptop Debian; el transformador en la
   computadora con GPU. Ambos con el mismo commit y documentado.
4. SymbTr: muestra de 100 obras estratificada por makam (semilla 7).

## A. Análisis por obra (capítulo 4, secciones preliminares y dos corpus)

**Corpus.** MuseTrainer completo y la muestra de SymbTr, deduplicados por huella
de contenido. Obras con menos de 20 eventos tras la extracción se excluyen y se
cuentan.

**Partición.** Por obra, contigua: primer 80 % de los eventos para ajustar, último
20 % para evaluar. Dentro del 80 %, el último 20 % sirve de validación para elegir
hiperparámetros; después se reajusta con el 80 % completo.

**Modelos.**
- Unigrama y bigrama con suavizado de Dirichlet α elegido en validación de {0.1, 0.5, 1}.
- HMM finito por Baum-Welch, K elegido en validación de {2, 4, 8, 16, 32}; si se
  elige el máximo, se amplía la rejilla antes de reportar.
- HDP-HMM truncado con β muestreado (Fox et al., 2011), K de truncamiento 20 y
  sensibilidad con 40; 4 cadenas (semillas 1–4), 400 iteraciones, burn-in 200.

**Métrica principal.** NLL por nota en la parte de prueba, en nats:
−log p(x_prueba | x_ajuste) / n_prueba. Para el HDP, promedio de la predictiva
sobre las muestras posteriores al burn-in de las 4 cadenas.

**Comparaciones.** HDP − finito, HDP − bigrama, HDP − unigrama: diferencia
pareada por obra. IC del 95 % por bootstrap por obra (10 000 remuestreos);
Wilcoxon como secundario; Holm sobre las tres.

**Regla de decisión.** «El HDP-HMM mejora a X» sólo si el extremo superior del IC
de (NLL_HDP − NLL_X) es menor que 0 tras Holm. Si el IC cruza 0: «no hay evidencia
de mejora». Si el extremo inferior es mayor que 0: «X es mejor».

**Número de estados.** Para cada obra, media posterior (tras burn-in, sobre
cadenas) del número de estados con ocupación ≥ 2 % de los eventos. Al finito se
le aplica el mismo conteo sobre su posterior de estados. Se reporta media ± d.e.
por obra; no se compara un K del HDP contra un K de Viterbi.

**Convergencia.** R-hat < 1.1 de la log-verosimilitud entre las 4 cadenas. Las obras
que no lo cumplan se reportan aparte y quedan fuera de la métrica principal; si
son más del 10 %, se duplican iteraciones y se registra la enmienda.

**Secundario (no decide).** Log-verosimilitud dentro de muestra en la media
posterior, por nota, para continuidad con la versión anterior.

## B. Comparación PDMX (capítulo 4, comparación y sensibilidad)

**Corpus y partición.** Los de `tesis_3000_gpu_20260823_1941` (3000 partituras,
semilla 7), con agrupamiento por contenido además de metadatos y orden de
partición independiente del sistema operativo. El conjunto de prueba nuevo se
reporta junto con su diferencia respecto del anterior.

**Métrica principal.** NLL por evento en prueba con BOS fuera del soporte
predictivo, frac = 1.0. Diferencia pareada por obra (transformador − cada
clásico), IC bootstrap por obra y Holm sobre las seis comparaciones, como en la
corrida original.

**Regla de decisión.** La misma de A, con «transformador» en lugar de «HDP».

**Curva de aprendizaje.** Sólo se afirma que un modelo «aprovecha más datos» si
su rejilla no tocó el máximo en ninguna fracción y su mejora relativa supera la
del otro con IC que no cruza 0.

**Costo.** Se reportan por separado: búsqueda completa, ajuste seleccionado,
evaluación y memoria, con el dispositivo de cada uno. Ninguna conclusión de
«más barato» compara CPU contra GPU sin declararlo en la misma frase.

**Réplicas.** En frac = 1.0, `data_seed` submuestrea el 90 % del entrenamiento;
3 semillas de datos × 2 de modelo. Se reporta n real de corridas distintas.

## C. Lo que no cambia

- La corrida `tesis_3000_gpu_20260823_1941` y `multicorpus_full` se conservan sin
  tocar y se citan como versión anterior cuando el texto lo necesite.

## Enmienda 1, 2026-10-27 (antes de ver resultados de las corridas corregidas)

Decisión de Adrián. Agrega a la sección B una regla que el texto original no
fijaba; no cambia ninguna de las anteriores.

**Frontera de Pareto (NLL contra costo).** En el eje de NLL, un modelo domina a
otro sólo si su diferencia pareada por obra (NLL ponderada por tokens) tiene IC
tras Holm, por pasos, que excluye 0 a su favor; un par que Holm no rechaza es
empate en ese eje. Un par de modelos con menos de 10 obras de prueba en común no
decide dominancia en NLL con la diferencia pareada y se compara con los
marginales; el reporte lista esos pares. En el eje de costo se usa la
incertidumbre de cada medición de costo, sin el tiempo de evaluación. Esta regla
está implementada en `Comparacion/runner.py` (`PARETO_MIN_PAIRED_WORKS = 10`)
de la rama `fix/auditoria-comparacion`.

## Aclaración 2, 2026-10-27 (antes de ver resultados de las corridas corregidas)

Decisión de Adrián. Precisa el estimando de las reglas de decisión de A y B; no
cambia ninguna regla.

**Estimando que decide.** «Diferencia pareada por obra» es la media, sobre las
obras de prueba, de la diferencia de NLL por nota de cada obra, con el mismo
peso para cada obra. Su IC es bootstrap por obra (remuestreo de obras) ajustado
por Holm por pasos sobre la familia preregistrada: las tres comparaciones de A y
las seis de B. La diferencia ponderada por tokens (cada nota con el mismo peso)
se reporta como secundaria, con su propio IC, y no decide.

**Familia de B.** Las seis comparaciones incluyen al VOMM. Una corrida sin VOMM
no evalúa la regla de B; si se hace, se declara exploratoria.

## Enmienda 3, 2026-10-27 (antes de ver resultados de las corridas corregidas)

Decisión de Adrián. Agrega a la sección A un hiperparámetro que el texto original
no fijaba.

**Concentración de la previa de emisiones del HDP-HMM (η).** Se elige por obra en
la validación interna, de {1, 3, 6}, con el mismo procedimiento que el α de los
n-gramas y el K del finito: se ajusta con el primer 80 % del 80 % de ajuste, se
evalúa la NLL predictiva en el último 20 % de ese 80 %, se elige el η de menor
NLL y se reajusta con el 80 % completo. Se reporta el η elegido por obra.
Motivo, registrado en la auditoría (H10): con datos simulados de 3 estados y
emisiones con ruido moderado, η = 1 da 4.90 estados ocupados y η = 6 da ≈ 3.3.

## Enmienda 4, 2026-10-27 (antes de ver resultados de las corridas corregidas)

Decisión de Adrián. Precisa el eje de costo de la Enmienda 1.

**Incertidumbre del tiempo en la frontera de Pareto.** Para cada modelo, el
intervalo de tiempo es `[mínimo × 0.9, máximo × 1.1]` del tiempo del ajuste
seleccionado entre sus corridas en frac = 1.0 (tolerancia relativa de 10 %).
Dos modelos cuyos intervalos se traslapan empatan en costo. El tiempo de
evaluación se reporta y no decide. Implementado como
`PARETO_TIME_RELATIVE_TOLERANCE = 0.10` en `Comparacion/runner.py`.

## Enmienda 5, 2026-10-28 (antes de ver resultados de las corridas corregidas)

Decisión de Adrián. Agrega a la sección A un hiperparámetro del HMM finito que el
texto original no fijaba.

**Suavizado de Dirichlet del HMM finito (análisis por obra).** Las emisiones y las
transiciones de Baum-Welch se estiman con un seudoconteo α elegido por obra en la
validación interna de {0.1, 0.5, 1}, junto con K (selección conjunta de (K, α) por
menor NLL de validación), con el mismo procedimiento que el α de los n-gramas y el η
del HDP-HMM; se reporta el α elegido por obra. La regla de ampliar la rejilla de K
cuando se elige el máximo se mantiene.
Motivo, medido en una corrida de tiempos de 5 obras de MuseTrainer (sólo cifras de
validación): con el seudoconteo fijo de 0.001, una altura de la validación ausente
en el ajuste (Minueto BWV Anh. 114: do#, 4 de 21 eventos) recibe probabilidad casi
nula, la NLL de validación de K ∈ {2, 4, 8, 16} queda en 3.96–4.70 nats (peor que el
uniforme, 2.485) y la rejilla se amplía hasta K = 512 con 80 eventos de ajuste,
porque los estados sin uso funcionan como suavizado. Con α ∈ {0.1, 0.5, 1} se elige
K entre 8 y 16 con NLL de 2.35–2.40. La comparación PDMX (sección B) no cambia.

## Enmienda 6, 2026-10-28 (antes de ver resultados de prueba de las corridas corregidas)

Decisión de Adrián, tras consultar un consejo de revisión con tres posturas
independientes. Sustituye la regla de convergencia de la sección A.

**Motivo (sólo datos de ajuste y validación).** En una corrida de tiempos con las
primeras 5 obras de MuseTrainer por orden de archivo, el R-hat de log L entre las
4 cadenas fue 2.33, 1.44, 1.27, 1.48 y 1.71 con 400/200 (0 de 5 convergen) y 3.60,
1.19, 1.06, 1.23 y 3.50 con 800/400 (1 de 5): la contingencia preregistrada
(duplicar iteraciones) se probó y no resuelve el problema. Un diagnóstico por cadena
sobre la validación interna (800/400, η = 3) mostró que en las obras largas las
cadenas quedan en modos distintos que cambian la predictiva (rango entre cadenas de
la NLL de validación: 0.056 nats/nota en la obra de 1387 notas y 0.188 en la de
1725; R-hat de la NLL predictiva por muestra 1.33 y 1.73), mientras que en la de
657 notas predicen casi igual (rango 0.013, R-hat 1.06). Excluir obras por la
convergencia del HDP filtraría la comparación con una propiedad de uno solo de los
modelos.

**Regla nueva.**
1. La métrica principal de A incluye todas las obras analizadas; ninguna se
   excluye por la convergencia del HDP-HMM. El HDP-HMM se evalúa como
   procedimiento: 4 cadenas (semillas 1–4), 400 iteraciones, burn-in 200, y la
   predictiva promediada en probabilidad sobre las muestras tras burn-in de las 4
   cadenas («ensamble de cadenas»), como el HMM finito se evalúa con el mejor de sus
   inicios de Baum-Welch.
2. Por obra se reportan: el R-hat de log L, el R-hat de la NLL predictiva por
   muestra, la NLL de prueba de cada cadena por separado y su rango. En el resumen
   se reporta la fracción de obras con R-hat de log L < 1.1.
3. Análisis de sensibilidad (no decide): la regla original, es decir, la misma
   comparación restringida a las obras con R-hat de log L < 1.1.
4. El número de estados del HDP-HMM y su variación se reportan por cadena, con la
   advertencia de que, sin convergencia entre cadenas, no son una media posterior.
   La tesis no interpreta esos números como inferencia bayesiana sobre K.
5. La regla de decisión de A y el estimando de la Aclaración 2 no cambian.

## Aclaración 7, 2026-10-28 (exposición a datos de prueba antes de las Enmiendas 5 y 6)

Para que el lector juzgue las Enmiendas 5 y 6: las corridas de tiempos que las
motivaron (5 primeras obras de MuseTrainer por orden de archivo, 400/200 y 800/400)
calcularon y guardaron la NLL de prueba de esas 5 obras en sus archivos de salida,
fuera del repo. Para decidir las enmiendas se miraron sólo las columnas de
validación, R-hat, η elegido, estados y tiempos, y el diagnóstico por cadena usó
sólo el ajuste interno y la validación; la NLL de prueba de esas obras no se
inspeccionó. Esas 5 obras forman parte del corpus de la corrida definitiva. Antes,
el smoke de PDMX (50 partituras, 7 obras de prueba, 3 épocas, carpeta
`artifacts/Comparacion/smoke_pdmx50_20261028_984dfde`) sí mostró NLL de prueba de la
sección B; no motivó ningún cambio a la sección B.

## Aclaración 8, 2026-10-29 (corrige la Aclaración 7)

La frase de la Aclaración 7 «la NLL de prueba de esas obras no se inspeccionó» no
es exacta. Exposición a cifras de prueba de la sección A antes de la corrida
confirmatoria, según la bitácora y los reportes de la tarea 1.4:
- Arabesque L. 66 n.º 1: cifras de prueba escritas en la bitácora de la rama
  `feat/porobra-heldout` (ff9f998, antes de la Enmienda 5; 700acba, después de la 6).
- Passacaglia: NLL de prueba calculada e impresa por la revisión de la tarea 1.4.
- 12 Variaciones de Twinkle Twinkle, Arabesque y Ave Maria D 839: NLL de prueba de
  un smoke de 40 iteraciones leídas en un reporte el 2026-10-28, después de
  registrar las Enmiendas 5 y 6.
- Minueto BWV Anh. 114 y Tocata BWV 565: NLL de prueba calculada en las corridas de
  tiempos (Aclaración 7), sin leerse.
Las Enmiendas 5 y 6 se motivaron con cifras de validación, R-hat y tiempos. Las seis
obras siguen en la métrica principal. Como análisis adicional, que no decide, el
resumen repite la comparación sin ellas e indica si algún veredicto cambia.

## Aclaración 9, 2026-11-01 (completa la Aclaración 8)

La lista de la Aclaración 8 quedó incompleta. El 2026-10-29, una revisión de la
tarea 1.4 corrió un smoke de 30 iteraciones (burn-in 15) con el Minueto BWV Anh. 114,
Bella Ciao y Passacaglia y leyó cifras derivadas de sus partes de prueba: los
veredictos de la comparación principal y de las sensibilidades, el extremo inferior
del IC de decisión (≈ 0.128), la diferencia ponderada por tokens (≈ 0.413) y la p de
Wilcoxon (0.25). Esas cifras también aparecen en el registro de esa revisión.

Por tanto, Bella Ciao se agrega a la lista y la indicación «sin leerse» del Minueto
ya no vale. Son 7 obras expuestas. Las siete siguen en la métrica principal, y el
análisis adicional sin ellas, que no decide, las excluye a todas. Ninguna regla de
decisión cambia. Desde esta fecha, las pruebas y revisiones previas a la corrida
confirmatoria no leen cifras de prueba de obras reales.

## Aclaración 10, 2026-11-01 (exposición accidental después de la Aclaración 9)

El 2026-11-01, una revisión de la tarea 1.4 hizo una búsqueda de texto en sus
archivos temporales que, por accidente, mostró 3 líneas con un mismo valor numérico
del `summary.json` de una corrida real corta (20 iteraciones, burn-in 10, 2 cadenas;
commit 5f52884). El revisor lo reportó sin repetir el valor, no sabe a qué campo
corresponde y dice no haberlo interpretado ni usado. Pudo ser una cifra derivada de
las partes de prueba, así que se trata como tal. No se volvió a abrir para averiguarlo.

Esa corrida usó sólo 3 obras: 12 Variaciones de Twinkle Twinkle, Arabesque L. 66 n.º 1
y Ave Maria D 839, ya expuestas según la Aclaración 8. Siguen siendo 7 obras expuestas;
ninguna regla de decisión cambia. Desde entonces, las revisiones tampoco hacen
búsquedas de texto sobre las salidas de las corridas.

## Enmienda 11, 2026-11-03 (antes de la corrida confirmatoria)

Decisión de Adrián. Completa la sección A. Sus motivos usan sólo datos de ajuste y
validación y simulaciones.

**Tope de la rejilla del HMM finito.** La ampliación de la rejilla de K termina en
K = 512. Elegir ese K ya no obliga a ampliar y no es una desviación. Motivo: en 3 de
las 63 obras de hasta 300 eventos (Happy Birthday, Preludio Op. 28 n.º 4 y evcara
sofyan, de 25 a 81 eventos), la NLL de validación baja casi siempre al crecer K y se
acerca a la del modelo uniforme (ln 12 ≈ 2.4849), así que la regla de ampliar no
terminaría con ningún techo. Esas obras siguen en la métrica principal y se marcan; su
K no se interpreta como número de estados. Un análisis de sensibilidad, que no decide,
repite la comparación sin ellas (`sensitivity_without_k_ceiling`).

**Mínimo de obras para un veredicto (sección A).** Un bloque con menos de 10 obras
reporta sus cifras, sin veredicto (`no_evaluable`), y eso no cuenta como cambio de
veredicto. Por debajo de 30 obras el bloque lleva un aviso. Motivo: en una simulación
de la revisión, con diferencias sin efecto real, sale algún veredicto en 88 % de los
casos con 2 obras, 35 % con 5, 15 % con 10 y 8 % con 30. Es el mismo mínimo de 10 que la
Enmienda 1 usa en la sección B. La regla de decisión no cambia.

**Valores de implementación que el texto no fijaba.** Se fijan los que usa el código:
5 inicios de Baum-Welch por K (semillas 0 a 4), 200 muestras para contar estados en el
finito, semilla 0 de la estadística e hiperparámetros del HDP-HMM α = 8, α₀ = 4, γ = 2,
κ = 0. El criterio de parada de Baum-Welch es la mejora del objetivo con previa que EM
hace crecer, menor que 10⁻⁴.

**Obras que no se pueden importar.** Una partitura que no se puede importar se excluye
y se cuenta. Hoy es sólo Mozart, Sonata n.º 16, Allegro, por un error de importación
(`MusicXMLImportException`). El destino esperado de los 169 candidatos queda fijado en
el código, y cualquier diferencia es una desviación.

**Dónde corre.** Por el plazo, lo que corre en CPU también se ejecuta en la computadora
con GPU (Windows), no en la laptop Debian; esto sustituye la decisión 3. Todas las
corridas salen de un mismo commit, que queda en cada manifiesto.

## Aclaración 12, 2026-11-03 (precisa la Enmienda 11, antes de cualquier resultado)

Por el plazo, la corrida principal de la sección B arrancó el 2026-11-03 desde el
commit 9fbf714, sin esperar a que se mezclara el código final del análisis por obra.
La frase «un mismo commit» de la Enmienda 11 se precisa así: cada corrida registra su
commit en el manifiesto, y el código de cada sección no cambia entre sus corridas. Las
sensibilidades y la medición de costo de la sección B salen de un commit cuyo
`Comparacion/` y `next_token_experiment/` sean idénticos a los de 9fbf714, y eso se
comprueba con `git diff` antes de lanzarlas. Las corridas de la sección A salen todas
del mismo commit.

## Aclaración 13, 2026-11-04 (cambia el commit de la corrida principal de la sección B, antes de cualquier resultado)

Decisión de Adrián. La corrida que la Aclaración 12 arrancó desde 9fbf714
(`tesis_corregida_20261103_9fbf714`) no cabe en el plazo. Terminó 1 de 15 celdas. En la
celda 2 (fracción 0.25), el HMM finito pasó más de 3 horas sin terminar el primer modelo,
porque la rejilla crece hasta 384 estados y el paso E corre en CPU. Esa corrida se detuvo,
se conserva sin tocar y no se usa. De ella sólo se vieron el registro de avance y metadatos
de la celda 1 (tiempos, rejilla y K elegido), no NLL ni otras cifras de prueba.

La corrida principal de la sección B sale del commit 306652e, con
`--finite-hmm-device cuda`. Ese commit calcula el paso E de Baum-Welch del HMM finito en
la GPU, en float64 y con las mismas operaciones que la versión de numpy. Las pruebas
comparan las dos versiones en CPU y en CUDA: el paso E coincide hasta 10⁻¹⁰ relativo, y
el ajuste completo elige el mismo K, con el mismo número de iteraciones.  Los candidatos con menos de 96 estados siguen en numpy, y cada fila registra dónde corrió. El modelo, la
rejilla, el tope de 384, las semillas, las particiones y la regla de decisión no cambian.
El costo del HMM finito de esta corrida se mide con el paso E en la GPU.

Lo que la Aclaración 12 pedía respecto de 9fbf714 se pide ahora respecto de 306652e:
las sensibilidades y la medición de costo de la sección B salen de un commit cuyo
`Comparacion/` y `next_token_experiment/` sean idénticos a los de 306652e, con la
misma opción, y eso se comprueba con `git diff` antes de lanzarlas.

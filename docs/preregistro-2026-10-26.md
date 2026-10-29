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

## Aclaración 8, 2026-10-28 (corrige el alcance de la Aclaración 7; antes de la corrida completa de A)

Registrada por la revisión de la tarea 1.4; no cambia ninguna regla y queda para
que Adrián la confirme en la Parada 1. La Aclaración 7 se deja como está; ésta
corrige una frase suya con lo que consta en el repo.

**Qué consta.** Cifras de NLL de prueba de una de esas 5 obras (Arabesque L. 66
No. 1) quedaron escritas en la bitácora de la rama `feat/porobra-heldout` antes de
las enmiendas: la entrada de la ronda 1 (ff9f998, 13:04, antes de la Enmienda 5,
fcde76c, 13:54) dice «de prueba 2.03 a 2.11, contra 2.053 del unigrama», y la de la
ronda 2 (700acba, 14:58, después de la Enmienda 5 y de la 6, 220b49f, 14:39) dice
«prueba 2.231». Además, los `per_work.csv` de las corridas de tiempos (13:50 y
14:13, fuera del repo) traen las columnas `*_test_nll` de exactamente esas 5 obras,
y los smokes de 3 obras del CLI (desde las 11:02) también las calcularon. Por eso
la frase «la NLL de prueba de esas obras no se inspeccionó» no vale para Arabesque:
esas cifras se vieron. Que las Enmiendas 5 y 6 citen sólo cifras de validación y de
ajuste sigue siendo cierto.

**Qué se hace.**
1. Las cinco obras siguen en la métrica principal de A (como dice la Aclaración 7).
2. Sensibilidad (no decide): la misma comparación sin esas cinco obras
   (`sensitivity_without_exposed` del resumen; la lista está en
   `TEST_EXPOSED_WORK_IDS` de `src/analysis/porobra.py` y en `test_exposure`).
3. El reporte de resultados nombra estas obras y dice si el veredicto cambia sin
   ellas.

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

# Plan de corrección de la tesis experimental y escrita

> **Para agentes o desarrolladores:** HABILIDAD REQUERIDA:
> superpowers:subagent-driven-development o superpowers:executing-plans.
> Una tarea por vez; una casilla se marca sólo con su evidencia guardada.

**Objetivo:** corregir los hallazgos de
[`../../auditoria-2026-10-26.md`](../../auditoria-2026-10-26.md) en el código,
volver a correr los experimentos afectados y reescribir lo que la tesis afirma
a partir de ellos, sin borrar ni sobrescribir ninguna corrida anterior.

**Base:** rama `fix/auditoria-tesis` desde `master` cbbd8e1 (Melodies) y
`master` 86e8d17 (Tesis). La copia local `Relacionado con Escuela/Tesis` es de
abril y no se usa.

**Arquitectura:** primero se fija qué decide cada resultado (preregistro),
después se corrige el código con pruebas que fallan antes del arreglo, luego se
corre en directorios nuevos y sólo al final se toca el texto, con cifras
generadas por script.

## Restricciones globales

- TDD: cada corrección empieza con una prueba que reproduce el hallazgo y falla.
- Ninguna corrida nueva sobrescribe `multicorpus_full`, `tesis_3000_gpu_*` ni
  `sens_*`. Cada corrida guarda commit, configuración, semillas, hardware y hashes.
- Ninguna cifra de la tesis se escribe a mano: sale de un script que lee artefactos.
- La regla de decisión de cada comparación se escribe y fecha antes de ver el resultado.
- Los hallazgos que la corrección no resuelva se declaran como limitación en el texto.

## Decisiones de Adrián (tomadas el 2026-10-26; ver preregistro)

1. **Qué es «melodía»** (D1): (a) voz superior de la parte en clave de sol, sin
   ligaduras repetidas, sin adornos ni cifrados — coincide con lo que dice el
   capítulo 2; o (b) mantener ambas manos y reescribir la definición.
2. **Línea base por obra** (F1, H4): HMM finito ajustado por Baum-Welch con su K
   elegido por validación, más unigrama y bigrama, todo evaluado en held-out
   por token; o conservar el finito heurístico declarado como referencia uniforme.
3. **Dónde se rehace la comparación PDMX** (C1, C2, C4): en esta máquina (CPU,
   12 núcleos, 15 GB; el caché de 3000 partituras está en el respaldo) o en la
   computadora con GPU de agosto.
4. **Alcance de SymbTr** (D2, D5): muestra estratificada por makam con
   microtonos declarados, o retirar SymbTr del capítulo.

## Fase 0: preparar, sin entrenar

### Tarea 0.1: respaldo fuera de la laptop
- [ ] Subir `~/Respaldos/Melodies-20261026/` al repositorio privado (comando en la conversación del 2026-10-26).
- **Evidencia:** URL del release y `SHA256SUMS` verificados tras descargar.

### Tarea 0.2: preregistro
- [ ] Escribir `docs/preregistro-2026-10-26.md`: métrica principal por obra
  (log-verosimilitud held-out por token), estimando de K (media posterior de
  estados con ocupación ≥ 2 %, igual para todos los modelos), número de cadenas
  y semillas, umbrales de convergencia, regla para declarar que el HDP mejora.
- **Evidencia:** commit fechado antes de cualquier corrida de la Fase 2.

## Fase 1: corregir el código (rama `fix/auditoria-tesis`)

### Tarea 1.1: β del HDP-HMM (H1, H2, H8)
- [ ] Prueba que falla: β cambia entre iteraciones; el K medio sobre 10 semillas
  no se correlaciona con el número de componentes grandes del primer β.
- [ ] Sustituir el paso MAP por el muestreo estándar de Fox et al. (2011):
  mesas auxiliares por CRT, β | m ~ Dir(γ/K + m·k), corrección ρ si κ > 0.
- [ ] `dirichlet_logpdf` deja de ocultar ceros: piso explícito y prueba.
- [ ] El generador aleatorio se crea dentro de `fit_sequences`.
- **Validar:** `pytest tests/test_hdp_hmm.py` y el script `hdp/s5_beta_seed.py` (correlación < 0.3).

### Tarea 1.2: qué reporta el HDP (H5, H6, F2)
- [ ] Log-verosimilitud: media posterior tras burn-in (y held-out en 1.3), nunca el máximo.
- [ ] Alineación de etiquetas (asignación húngara) antes de promediar parámetros.
- [ ] `effective_states` = estimando del preregistro; el mismo para el finito.
- **Validar:** pruebas nuevas + `hdp/s3_report.py` sin selección del máximo.

### Tarea 1.3: análisis por obra comparable (F1, H4, D6)
- [ ] Según la decisión 2: línea base ajustada (Baum-Welch con rejilla de K sin
  techo, unigrama, bigrama) y partición 80/20 por obra.
- [ ] Ganancia reportada por token, y figura con todas las obras.
- **Validar:** el finito heurístico vuelve a dar ≈ −ln 12 por token en la prueba de regresión.

### Tarea 1.4: datos (D1, D3, D4, D5, D7)
- [ ] Extracción de melodía según la decisión 1, con `stripTies`, sin `Harmony` ni adornos.
- [ ] `NoChord` y alturas `None`: error explícito; errores de catálogo y de análisis se cuentan juntos.
- [ ] Deduplicación por huella de contenido.
- [ ] SymbTr según la decisión 4; microtonos con regla única documentada.
- **Validar:** pruebas con el Minueto (65 notas en 16 compases = videos), la Sonata 14 y Feliz cumpleaños.

### Tarea 1.5: comparación PDMX (C1–C7, F3, T1–T3)
- [ ] Agrupación por contenido (contención de n-gramas) además de metadatos; quitar `Path().stem`.
- [ ] Orden de partición independiente del sistema operativo (ruta POSIX normalizada o hash).
- [ ] `data_seed` efectivo en frac = 1.0 (submuestreo o bootstrap del train) y `model_seed` en el DataLoader.
- [ ] Rejillas sin techo: K del HMM y orden del VOMM hasta que la selección deje de tocar el máximo.
- [ ] Costo: ajuste seleccionado contra ajuste seleccionado y selección completa contra selección completa, mismo dispositivo o declarado.
- [ ] Diagnóstico del HDP por celda y sin burn-in; Pareto con tolerancia; IC para cada cifra titular de su propio estimando.
- [ ] BOS fuera del soporte predictivo en todos los modelos; NLL de validación sin label smoothing.
- **Validar:** `pytest` completo + smoke de 58 piezas con auditorías en `passed`.

## Fase 2: volver a correr (directorios nuevos)

### Tarea 2.1: análisis por obra
- [ ] Corpus deduplicado, ≥ 4 cadenas por obra, parámetros del preregistro.
- **Evidencia:** `artifacts/outputs/porobra_<fecha>_<commit>/` con manifiesto.

### Tarea 2.2: comparación PDMX
- [ ] Curva de aprendizaje y las cuatro sensibilidades con el código corregido.
- [ ] Contrastar con `tesis_3000_gpu_20260823_1941`: qué cambia y por qué.
- **Evidencia:** `artifacts/Comparacion/<corrida>_<fecha>/` con `artifact_audit` en `passed`.

### Tarea 2.3: revisión independiente
- [ ] Un revisor que no escribió el código recalcula las cifras titulares desde los artefactos.

## Fase 3: tesis escrita (repo `Tesis`)

### Tarea 3.1: tablas y figuras por script
- [ ] Un generador único lee los artefactos nuevos y escribe todas las tablas `.tex` y figuras; ninguna cifra a mano.

### Tarea 3.2: texto
- [ ] Cap. 2: definición de melodía según la decisión 1.
- [ ] Cap. 1 (metodología): held-out por obra, estimando de K, cadenas, preregistro.
- [ ] Cap. 4: secciones preliminares, extensión a dos corpus, obra por obra,
  sensibilidad y costo, con las nuevas cifras; eliminar «167/167», «uniformemente
  positiva», «el transformador es más barato» y «sólo el transformador aprovecha
  el corpus» si los datos ya no los sostienen; etiquetas de acordes de los estados
  HDP sólo con evidencia.
- [ ] Cap. 5 y resumen: respuesta a la pregunta con las cifras nuevas.
- [ ] Limitaciones: SymbTr, microtonos, reinicio BOS, una sola partición PDMX si aplica.
- [ ] Correcciones menores: «subconjunto clásico», leyendas de figuras, número de compases.

### Tarea 3.3: verificación final
- [ ] Script que extrae cada número del `.tex` y lo compara con los artefactos.
- [ ] Compilación del PDF sin advertencias de referencias.

## Fase 4: videos
- [ ] Rehacer el video 3 con β corregido: estados como contextos aprendidos (no
  acordes), comparación contra la progresión de Viterbi del video 2 y textos en
  pantalla más directos.

## Riesgos

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Con β corregido y held-out, la ventaja del HDP sobre el finito ajustado se reduce o desaparece | media | preregistro antes de correr; la tesis reporta lo que salga |
| La comparación PDMX en CPU tarda días | media | estimar con el smoke de 58 piezas antes; o usar la GPU de agosto |
| El orden VOMM / HDP se invierte | alta | ya es frágil; reportarlo con IC |
| Las correcciones de la asesora (16-oct) chocan con la reescritura | baja | trabajar sobre `master` de Tesis en rama propia |

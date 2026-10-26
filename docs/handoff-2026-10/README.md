# Handoff: cierre de la tesis en la computadora con GPU

Escrito el 2026-10-26 en la laptop Debian. Esta carpeta es todo lo que necesitas.
Tú (Adrián) haces los pasos de este archivo; Claude Code hace el resto siguiendo
[`PROMPT_CLAUDE.md`](PROMPT_CLAUDE.md), que pegas tal cual.

## 1. Situación actual (2026-10-26)

**Qué pasó.** Una auditoría con 5 revisiones independientes encontró 27 problemas
en el código y en el diseño experimental. Registro completo:
[`../auditoria-2026-10-26.md`](../auditoria-2026-10-26.md), con un script de
reproducción por hallazgo en `scripts/auditoria_2026_10_26/`.

Los cuatro que cambian conclusiones de la tesis:
1. **β del HDP-HMM nunca se actualizaba** (H1/H2). El número de estados (K = 8.222)
   lo decidía el primer sorteo de la semilla, no la música.
2. **El HMM finito del análisis por obra equivale al azar** (F1): le gana al
   uniforme por 0.003 nats por nota. «El HDP mejora en 167/167» no demuestra nada.
3. **La «melodía» mezclaba ambas manos, notas ligadas, adornos y cifrados** (D1),
   en el análisis por obra **y también en PDMX** (D1b: corrió con `prefer_treble=True`,
   que no quita ligaduras, adornos ni cifrados).
4. **Costo y curva de aprendizaje del capítulo 4 sesgados** (C1/C2): al HMM se le
   cobraba toda la búsqueda en CPU y al transformador un ajuste en GPU; las rejillas
   de los clásicos tocaban el techo.

Lo que **se sostiene**: la ventaja predictiva del transformador sobre los clásicos
en PDMX (≈ 0.25 nats por obra) y la aritmética de forward, FFBS, Baum-Welch y VOMM.

**Tus decisiones** (registradas en [`../preregistro-2026-10-26.md`](../preregistro-2026-10-26.md)):
melodía = voz superior; el HDP se compara contra un HMM finito ajustado, un bigrama
y un unigrama, en held-out; SymbTr estratificado por makam; todo corre en esta
computadora.

**Estado del código** (todo en GitHub, `JanusAdamus/Melodies`):

| Rama | Contenido | Estado |
|---|---|---|
| `fix/auditoria-tesis` | Rama de integración. Auditoría, plan, preregistro, este handoff, `visualizaciones/`, y ya mezcladas `particiones` y `datos`. Suite: 240 pruebas en verde | base de todo |
| `fix/auditoria-particiones` | Partición por contenido, orden independiente del SO, `data_seed` efectivo (90 % en frac = 1.0), siembra del DataLoader | revisada y mezclada |
| `fix/auditoria-datos` | Melodía = voz superior, ligaduras, adornos y cifrados fuera, errores contados, deduplicación, SymbTr estratificado, microtonos | revisada y mezclada |
| `fix/auditoria-hdp` | β por mesas auxiliares CRT (Fox et al. 2011), Dirichlet sin ceros, reporte posterior, predictiva held-out | **en curso al irme**: 5 commits subidos; puede tener más en la laptop |
| `fix/auditoria-comparacion` | BOS fuera del soporte, NLL sin label smoothing, IC por estimando, Pareto, diagnóstico HDP, rejillas sin techo, costo comparable | **en curso al irme**: 8 commits subidos; puede tener más en la laptop |

**Objetivo final:** tesis experimental y escrita corregidas, verificadas número por
número, `master` de Melodies y de Tesis actualizados, ramas de GitHub limpias, video 3
rehecho y un reporte de cierre.

## 2. Antes de salir de la laptop Debian (2 minutos)

Los dos desarrolladores que seguían trabajando pueden haber dejado commits
locales. Súbelos:

```bash
cd "$HOME/Documentos/Relacionado con Escuela/Melodies-wt-hdp" && git status --short && git push origin fix/auditoria-hdp
cd "$HOME/Documentos/Relacionado con Escuela/Melodies-wt-comparacion" && git status --short && git push origin fix/auditoria-comparacion
```

Si `git status --short` muestra archivos modificados (aparte de `.mplconfig/`),
hay trabajo sin commit: déjalo así y díselo a Claude en la otra computadora. Él
rehará esa parte; no hace falta rescatarla.

Respaldo de los resultados originales (sigue pendiente y sólo existe en la laptop):

```bash
cd ~/Respaldos/Melodies-20261026 && gh repo create JanusAdamus/Melodies-respaldo --private && gh release create respaldo-20261026 --repo JanusAdamus/Melodies-respaldo --title "Respaldo 2026-10-26" --notes "artifacts + rama cleanup-reproducibility" *.tar.gz *.bundle SHA256SUMS
```

## 3. En la computadora con GPU

Requisitos: Windows con PowerShell, Git, GitHub CLI (`gh auth login` hecho),
Python 3.12 o 3.13, controlador NVIDIA con CUDA, Claude Code, y el corpus PDMX en
`C:\Users\AdrianGonzalez\Datasets\PDMX\mxl` (el de agosto). Para la Fase 3 hace falta
una distribución de LaTeX con XeLaTeX (MiKTeX o TeX Live).

```powershell
cd <carpeta donde está tu clon de Melodies>      # el de agosto; si no existe: gh repo clone JanusAdamus/Melodies
git fetch origin --prune
git status                                        # si hay cambios locales sin commit, NO sigas: díselo a Claude
git switch fix/auditoria-tesis
git pull
claude
```

Dentro de Claude Code, pega completo el contenido de
[`PROMPT_CLAUDE.md`](PROMPT_CLAUDE.md) (desde «Eres el responsable…» hasta el final).

## 4. Qué va a pasar y cuándo te va a necesitar

Claude trabaja por fases y **se detiene sólo en estos puntos**, que son tuyos:

| Punto | Qué te pide |
|---|---|
| Antes de la Fase 2 | Que confirmes las estimaciones de tiempo de cada corrida (las mide con smokes) |
| Fin de la Fase 2 | Que veas el resumen de resultados frente a la regla del preregistro |
| Fin de la Fase 3 | Que leas el PDF de la tesis antes de mezclar a `master` de Tesis |
| Ramas sin mezclar de septiembre | Si conviene integrarlas o sólo archivarlas, con su recomendación |

Todo lo demás lo hace solo: integrar ramas, revisar de forma adversarial, correr,
verificar, escribir, commitear y subir.

## 5. Si se corta la sesión o se acaba el uso

Claude deja el avance en `docs/handoff-2026-10/BITACORA.md` después de cada tarea.
Para retomar: abre `claude` en la misma carpeta y pega:

> Lee `docs/handoff-2026-10/PROMPT_CLAUDE.md` y `docs/handoff-2026-10/BITACORA.md` y continúa exactamente desde la última tarea abierta.

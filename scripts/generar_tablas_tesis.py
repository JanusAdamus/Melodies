"""Genera las tablas ``.tex`` de la tesis desde los artefactos de las corridas (tarea 3.2).

Lee, sólo lectura:

* una corrida por obra de ``src/cli/porobra_heldout.py`` (``per_work.csv``,
  ``summary.json``, ``manifest.json``), preregistro sección A;
* una corrida PDMX de ``Comparacion/runner.py`` (``results_summary.csv``,
  ``headline_estimates.json``, ``pairwise_comparisons.json``, ``pareto_summary.json``,
  ``engineering_costs.csv``, ``hdp_chain_diagnostics.json``), preregistro sección B.

Escribe en un directorio de salida nuevo (o vacío) un ``.tex`` por tabla (sólo el
``tabular``: el título y la etiqueta los pone el capítulo) y, junto a cada uno,
``<tabla>.procedencia.json`` con cada número escrito: su texto, su valor sin redondear,
el archivo, el campo o columna y la fila de donde sale, y la operación si la hay
(«media», «negado»...).  Además ``figuras.json`` (figuras de las corridas, copiadas a
``figuras/`` con su sha256) y ``generador_resumen.json`` (corridas, tablas, avisos y
pendientes).

Veredictos: la regla del preregistro (Aclaración 2) sobre el IC de Holm por pasos del
estimando por obra (``work_mean_holm_decision_ci``): «mejora» si el extremo superior es
< 0, «es mejor el otro» si el inferior es > 0, «no hay evidencia» si cruza 0 y «no
evaluable» si Holm no se evaluó (p. ej. menos de 10 obras en común).  En PDMX las
comparaciones con el transformador se orientan «transformador − clásico».

Una corrida que no se da deja sus tablas con ``\\pendiente{...}`` y sin números: nada se
inventa.  Ningún número se teclea aquí; todo sale de un artefacto.

Uso::

    .venv\\Scripts\\python.exe scripts/generar_tablas_tesis.py \\
        --porobra artifacts/outputs/porobra_heldout_<fecha>_<commit7> \\
        --pdmx artifacts/Comparacion/tesis_corregida_<fecha>_<commit7> \\
        --salida <directorio nuevo>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CABECERA = "% Generado por scripts/generar_tablas_tesis.py desde artefactos; no editar a mano.\n"
POROBRA_MODELOS = ("hdp", "finite", "bigram", "unigram")
NOMBRES = {
    "hdp": "HDP-HMM",
    "finite": "HMM finito",
    "bigram": "Bigrama",
    "unigram": "Unigrama",
    "hdp_hmm": "HDP-HMM",
    "finite_hmm": "HMM finito",
    "transformer": "Transformador",
    "vomm": "VOMM",
}
VEREDICTOS_RESUMEN = {
    "mejora": "mejora",
    "no_hay_evidencia": "no hay evidencia",
    "es_mejor_el_otro": "es mejor el otro",
    "no_evaluable": "no evaluable",
}
ESTADOS_HOLM = {
    "too_few_paired_works": "pocas obras en común",
    "not_evaluable": "sin datos pareados",
}
TOLERANCIA_RECALCULO = 1e-9
# results_summary.csv y headline_estimates.json calculan la NLL por nota por caminos
# distintos; en la corrida de humo difieren hasta 1.8e-7 (transformador).
TOLERANCIA_ENTRE_ARTEFACTOS = 1e-5
MIB = 1024 * 1024

TABLAS_POROBRA = (
    "porobra_corpus.tex",
    "porobra_comparaciones.tex",
    "porobra_nll.tex",
    "porobra_estados.tex",
    "porobra_convergencia.tex",
    "porobra_hiperparametros.tex",
    "porobra_por_obra.tex",
)
TABLAS_PDMX = (
    "pdmx_resultados.tex",
    "pdmx_comparaciones.tex",
    "pdmx_costos.tex",
    "pdmx_pareto.tex",
    "pdmx_hdp_diagnosticos.tex",
)
PENDIENTE = {
    "porobra": "corrida por obra de la Fase 2 (tarea 2.1)",
    "pdmx": "corrida PDMX de la Fase 2 (tarea 2.2)",
}


class ArtefactoInconsistente(ValueError):
    """Los artefactos se contradicen entre sí; no se escribe ninguna tabla."""


# --- utilidades ---------------------------------------------------------------------


def veredicto(status: str | None, decision_ci: Sequence[float] | None) -> str:
    """Regla de decisión del preregistro sobre el IC de Holm de (modelo A − modelo B)."""

    if status != "ok" or decision_ci is None:
        return "no evaluable"
    low, high = (float(x) for x in decision_ci)
    if high < 0.0:
        return "mejora"
    if low > 0.0:
        return "es mejor el otro"
    return "no hay evidencia"


def _finito(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _verdadero(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "sí", "si", "yes"}
    return bool(value)


def escapar(text: Any) -> str:
    out = []
    for char in str(text):
        out.append(
            {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}",
            }.get(char, char)
        )
    return "".join(out)


def ident(text: Any) -> str:
    """Identificador (ruta, id de obra): en ``\\texttt``, que el verificador no lee como cifra."""

    return r"\texttt{" + escapar(text) + "}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leer_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Falta el artefacto {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _leer_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Falta el artefacto {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass
class Corrida:
    directorio: Path
    fuentes: dict[str, str] = field(default_factory=dict)

    def archivo(self, nombre: str) -> tuple[Path, str]:
        path = self.directorio / nombre
        etiqueta = f"{self.directorio.name}/{nombre}"
        if path.is_file():
            self.fuentes.setdefault(etiqueta, _sha256(path))
        return path, etiqueta


@dataclass
class Tabla:
    nombre: str
    columnas: str
    encabezado: Sequence[str]
    filas: list[str] = field(default_factory=list)
    cifras: list[dict] = field(default_factory=list)
    notas: list[str] = field(default_factory=list)
    largo: bool = False

    def cifra(
        self,
        valor: Any,
        decimales: int,
        *,
        archivo: str,
        campo: str,
        fila: Mapping[str, Any],
        operacion: str | None = None,
        valor_fuente: Any = None,
    ) -> str:
        number = _finito(valor)
        if number is None:
            return "---"
        texto = f"{number:.{decimales}f}"
        if texto.lstrip("-").strip("0.") == "":
            texto = texto.lstrip("-")  # nunca «-0.000»
        entry = {"texto": texto, "valor": number, "archivo": archivo, "campo": campo, "fila": dict(fila)}
        if operacion:
            entry["operacion"] = operacion
        if valor_fuente is not None:
            entry["valor_fuente"] = valor_fuente
        self.cifras.append(entry)
        return "$-$" + texto[1:] if texto.startswith("-") else texto

    def intervalo(self, ci: Sequence[Any] | None, decimales: int, **kwargs) -> str:
        if ci is None:
            return "---"
        campo = kwargs.pop("campo")
        negar = kwargs.pop("negar", False)
        values = [_finito(x) for x in ci]
        if any(v is None for v in values):
            return "---"
        if negar:
            low, high = -values[1], -values[0]
            low_txt = self.cifra(low, decimales, campo=f"{campo}[1]", operacion="negado", valor_fuente=values[1], **kwargs)
            high_txt = self.cifra(high, decimales, campo=f"{campo}[0]", operacion="negado", valor_fuente=values[0], **kwargs)
        else:
            low_txt = self.cifra(values[0], decimales, campo=f"{campo}[0]", **kwargs)
            high_txt = self.cifra(values[1], decimales, campo=f"{campo}[1]", **kwargs)
        return f"[{low_txt}, {high_txt}]"

    def fila(self, celdas: Iterable[str]) -> None:
        self.filas.append(" & ".join(celdas) + r" \\")

    def tex(self) -> str:
        entorno = "longtable" if self.largo else "tabular"
        lines = [CABECERA, f"% Procedencia de cada número: {self.nombre.replace('.tex', '.procedencia.json')}\n"]
        lines.append(f"\\begin{{{entorno}}}{{{self.columnas}}}\n\\toprule\n")
        lines.append(" & ".join(self.encabezado) + " \\\\\n\\midrule\n")
        if self.largo:
            lines.append("\\endhead\n")
        lines.extend(row + "\n" for row in self.filas)
        lines.append("\\bottomrule\n")
        for nota in self.notas:
            lines.append(f"\\multicolumn{{{len(self.encabezado)}}}{{p{{0.95\\linewidth}}}}{{\\footnotesize {nota}}} \\\\\n")
        lines.append(f"\\end{{{entorno}}}\n")
        return "".join(lines)


# --- corrida por obra (preregistro A) -------------------------------------------------


def _filas_porobra(corrida: Corrida) -> tuple[list[dict], str]:
    path, etiqueta = corrida.archivo("per_work.csv")
    rows = _leer_csv(path)
    for index, row in enumerate(rows, start=2):  # fila 1 = encabezado del CSV
        row["_linea_csv"] = index
    return rows, etiqueta


def _filas_convergidas(rows: list[dict]) -> list[dict]:
    return [row for row in rows if _verdadero(row["hdp_converged"])]


def _verificar_porobra(rows: list[dict], summary: dict) -> None:
    converged = _filas_convergidas(rows)
    for index, comparison in enumerate(summary["comparisons"]["comparisons"]):
        other = comparison["model_b"]
        computed = veredicto(comparison.get("work_mean_holm_status"), comparison.get("work_mean_holm_decision_ci"))
        declared = comparison.get("verdict")
        if declared is not None and VEREDICTOS_RESUMEN.get(declared, declared) != computed:
            raise ArtefactoInconsistente(
                f"summary.json, comparación {index} (hdp − {other}): veredicto declarado {declared!r}, "
                f"la regla sobre work_mean_holm_decision_ci da {computed!r}"
            )
        if comparison.get("status") != "ok":
            continue
        paired = [r for r in converged if _finito(r["hdp_test_nll"]) is not None and _finito(r[f"{other}_test_nll"]) is not None]
        if len(paired) != int(comparison["n_pairs"]):
            raise ArtefactoInconsistente(f"per_work.csv da {len(paired)} obras pareadas hdp − {other}; summary.json dice {comparison['n_pairs']}")
        recomputed = statistics.fmean(float(r["hdp_test_nll"]) - float(r[f"{other}_test_nll"]) for r in paired)
        if abs(recomputed - float(comparison["mean_difference"])) > TOLERANCIA_RECALCULO:
            raise ArtefactoInconsistente(
                f"La media por obra hdp − {other} recalculada desde per_work.csv ({recomputed!r}) no coincide con "
                f"summary.json ({comparison['mean_difference']!r})"
            )


def _comparaciones(tabla: Tabla, comparisons: list[dict], *, archivo: str, prefijo: str, orientar_a: str | None = None) -> None:
    for index, comparison in enumerate(comparisons):
        a, b = comparison["model_a"], comparison["model_b"]
        negar = orientar_a is not None and b == orientar_a
        if negar:
            a, b = b, a
        fila = {"model_a": comparison["model_a"], "model_b": comparison["model_b"]}
        campo = f"{prefijo}[{index}]"
        status = comparison.get("work_mean_holm_status")
        decision = comparison.get("work_mean_holm_decision_ci")
        if decision is not None and negar:
            decision = [-float(decision[1]), -float(decision[0])]
        verdict = veredicto(status, decision)
        rejects = comparison.get("work_mean_holm_rejects_zero")
        if status == "ok" and rejects is not None and bool(rejects) != (verdict in {"mejora", "es mejor el otro"}):
            raise ArtefactoInconsistente(
                f"{archivo}, {campo}: work_mean_holm_rejects_zero={rejects} contradice el IC de decisión {decision}"
            )
        if status != "ok" and status:
            verdict = f"no evaluable ({ESTADOS_HOLM.get(status, escapar(status))})"

        def num(key, decimales, *, signo=True):
            value = comparison.get(key)
            if negar and signo and _finito(value) is not None:
                return tabla.cifra(-float(value), decimales, archivo=archivo, campo=f"{campo}.{key}", fila=fila,
                                   operacion="negado", valor_fuente=float(value))
            return tabla.cifra(value, decimales, archivo=archivo, campo=f"{campo}.{key}", fila=fila)

        tabla.fila(
            [
                NOMBRES.get(a, escapar(a)),
                NOMBRES.get(b, escapar(b)),
                num("n_pairs", 0, signo=False),
                num("mean_difference", 3),
                tabla.intervalo(comparison.get("work_mean_holm_decision_ci"), 3, archivo=archivo,
                                campo=f"{campo}.work_mean_holm_decision_ci", fila=fila, negar=negar),
                verdict,
                num("token_weighted_difference", 3),
                tabla.intervalo(comparison.get("token_weighted_bootstrap_95_ci"), 3, archivo=archivo,
                                campo=f"{campo}.token_weighted_bootstrap_95_ci", fila=fila, negar=negar),
                num("p_value_holm", 3, signo=False),
            ]
        )


ENCABEZADO_COMPARACION = (
    "A",
    "B",
    "Obras",
    r"A $-$ B (media por obra)",
    "IC Holm (decide)",
    "Veredicto",
    r"A $-$ B (por nota)",
    "IC bootstrap",
    "Wilcoxon $p$ (Holm)",
)


def tablas_porobra(corrida: Corrida) -> tuple[list[Tabla], list[str]]:
    rows, csv_label = _filas_porobra(corrida)
    summary_path, summary_label = corrida.archivo("summary.json")
    summary = _leer_json(summary_path)
    _leer_json(corrida.archivo("manifest.json")[0])  # la corrida debe estar completa
    _verificar_porobra(rows, summary)
    avisos: list[str] = []
    tablas: list[Tabla] = []

    # corpus
    corpus = summary["corpus"]
    t = Tabla("porobra_corpus.tex", "lr", ("Concepto", "Obras"))
    for source, count in sorted(corpus["n_by_source_candidates"].items()):
        t.fila([f"Candidatas ({escapar(source)})", t.cifra(count, 0, archivo=summary_label, campo=f"corpus.n_by_source_candidates.{source}", fila={})])
    for key, label in (("n_failures", "Fallas de lectura"), ("n_duplicates", "Duplicadas por contenido"),
                       ("min_events", "Mínimo de eventos por obra"), ("n_excluded_short", "Excluidas por tener menos eventos")):
        t.fila([label, t.cifra(corpus[key], 0, archivo=summary_label, campo=f"corpus.{key}", fila={})])
    for source, count in sorted(corpus["n_by_source_analyzed"].items()):
        t.fila([f"Analizadas ({escapar(source)})", t.cifra(count, 0, archivo=summary_label, campo=f"corpus.n_by_source_analyzed.{source}", fila={})])
    t.fila(["Analizadas (total)", t.cifra(corpus["n_analyzed"], 0, archivo=summary_label, campo="corpus.n_analyzed", fila={})])
    tablas.append(t)

    # comparaciones
    t = Tabla("porobra_comparaciones.tex", "llrrllrrr", ENCABEZADO_COMPARACION)
    _comparaciones(t, summary["comparisons"]["comparisons"], archivo=summary_label, prefijo="comparisons.comparisons")
    t.notas.append("Sólo obras con $\\hat R$ por debajo del umbral. Veredicto: regla del preregistro (sección A, Aclaración 2) "
                   "sobre el IC bootstrap por obra ajustado por Holm por pasos.")
    tablas.append(t)

    # NLL por modelo (media sobre obras convergidas)
    converged = _filas_convergidas(rows)
    t = Tabla("porobra_nll.tex", "llrrr", ("Fuente", "Modelo", "Obras", "NLL media por obra", "D.e. entre obras"))
    grupos = [("todas", "Todas", converged)] + [
        (source, escapar(source), [r for r in converged if r["source"] == source])
        for source in sorted({r["source"] for r in converged})
    ]
    for clave, etiqueta, grupo in grupos:
        if not grupo:
            continue
        for model in POROBRA_MODELOS:
            values = [float(r[f"{model}_test_nll"]) for r in grupo if _finito(r[f"{model}_test_nll"]) is not None]
            fila = {"fuente": clave, "work_ids": [r["work_id"] for r in grupo]}
            t.fila([
                etiqueta,
                NOMBRES[model],
                t.cifra(len(values), 0, archivo=csv_label, campo=f"{model}_test_nll", fila=fila, operacion="conteo"),
                t.cifra(statistics.fmean(values) if values else None, 3, archivo=csv_label, campo=f"{model}_test_nll", fila=fila, operacion="media"),
                t.cifra(statistics.stdev(values) if len(values) > 1 else None, 3, archivo=csv_label, campo=f"{model}_test_nll",
                        fila=fila, operacion="desviacion_estandar"),
            ])
    t.notas.append("NLL por nota en la parte de prueba (nats); sólo obras cuyo HDP-HMM converge.")
    tablas.append(t)

    # estados ocupados
    t = Tabla("porobra_estados.tex", "lrrrrr", ("Fuente", "Obras", "HDP-HMM: media", "HDP-HMM: d.e.", "Finito: media", "Finito: d.e."))
    for clave, etiqueta, grupo in grupos:
        if not grupo:
            continue
        fila = {"fuente": clave, "work_ids": [r["work_id"] for r in grupo]}
        celdas = [etiqueta, t.cifra(len(grupo), 0, archivo=csv_label, campo="work_id", fila=fila, operacion="conteo")]
        for column in ("hdp_states_mean", "finite_states_mean"):
            values = [float(r[column]) for r in grupo if _finito(r[column]) is not None]
            celdas.append(t.cifra(statistics.fmean(values) if values else None, 1, archivo=csv_label, campo=column, fila=fila, operacion="media"))
            celdas.append(t.cifra(statistics.stdev(values) if len(values) > 1 else None, 1, archivo=csv_label, campo=column,
                                  fila=fila, operacion="desviacion_estandar"))
        t.fila(celdas)
    t.notas.append("Estados con ocupación de al menos el umbral del preregistro; media posterior por obra, resumida entre obras que convergen.")
    tablas.append(t)

    # convergencia
    conv = summary["convergence"]
    t = Tabla("porobra_convergencia.tex", "lr", ("Concepto", "Valor"))
    t.fila(["Umbral de $\\hat R$", t.cifra(conv["rhat_threshold"], 2, archivo=summary_label, campo="convergence.rhat_threshold", fila={})])
    t.fila(["Obras analizadas", t.cifra(conv["n_works"], 0, archivo=summary_label, campo="convergence.n_works", fila={})])
    t.fila(["Obras que no convergen", t.cifra(conv["n_not_converged"], 0, archivo=summary_label, campo="convergence.n_not_converged", fila={})])
    t.fila(["Porcentaje que no converge", t.cifra(100 * float(conv["fraction_not_converged"]), 1, archivo=summary_label,
                                                  campo="convergence.fraction_not_converged", fila={}, operacion="por_cien",
                                                  valor_fuente=conv["fraction_not_converged"]) + r"\,\%"])
    tablas.append(t)
    if _verdadero(conv.get("exceeds_10_percent")):
        avisos.append("porobra: más del 10 % de las obras no converge; el preregistro manda duplicar iteraciones antes de reportar.")

    # hiperparámetros elegidos
    hyper = summary["selected_hyperparameters"]
    t = Tabla("porobra_hiperparametros.tex", "lrr", ("Hiperparámetro", "Valor elegido", "Obras"))
    for key, label, decimales in (("hdp_eta", "$\\eta$ del HDP-HMM", 1), ("finite_k", "$K$ del finito", 0),
                                  ("unigram_alpha", "$\\alpha$ del unigrama", 1), ("bigram_alpha", "$\\alpha$ del bigrama", 1)):
        for value, count in sorted(hyper.get(key, {}).items(), key=lambda item: float(item[0])):
            t.fila([label,
                    t.cifra(float(value), decimales, archivo=summary_label, campo=f"selected_hyperparameters.{key}", fila={"clave": value}),
                    t.cifra(count, 0, archivo=summary_label, campo=f"selected_hyperparameters.{key}.{value}", fila={"clave": value})])
    if hyper.get("finite_k_hit_limit"):
        avisos.append(f"porobra: {len(hyper['finite_k_hit_limit'])} obras con la rejilla del finito en el techo.")
    tablas.append(t)

    # por obra (apéndice)
    t = Tabla("porobra_por_obra.tex", "llrrrrrllrrr",
              ("Obra", "Fuente", "Notas prueba", "HDP-HMM", "Finito", "Bigrama", "Unigrama",
               "Estados HDP", "Estados finito", "$\\hat R$", "$\\eta$", "$K$"), largo=True)
    for row in rows:
        fila = {"work_id": row["work_id"], "linea_csv": row["_linea_csv"]}

        def c(column, decimales):
            return t.cifra(row.get(column), decimales, archivo=csv_label, campo=column, fila=fila)

        t.fila([ident(row["work_id"]), escapar(row["source"]), c("n_test", 0),
                *(c(f"{m}_test_nll", 3) for m in POROBRA_MODELOS),
                f"{c('hdp_states_mean', 1)} $\\pm$ {c('hdp_states_sd', 1)}",
                f"{c('finite_states_mean', 1)} $\\pm$ {c('finite_states_sd', 1)}",
                c("hdp_rhat", 2) + ("" if _verdadero(row["hdp_converged"]) else r"\textsuperscript{*}"),
                c("hdp_eta", 1), c("finite_k", 0)])
    t.notas.append("\\textsuperscript{*} No converge: fuera de la métrica principal.")
    tablas.append(t)
    return tablas, avisos


# --- corrida PDMX (preregistro B) -----------------------------------------------------


def _estado_analisis(status: Mapping | None, nombre: str) -> list[str]:
    avisos = []
    status = status or {}
    if status.get("status") != "preregistered" or not status.get("evaluates_preregistered_rule_B", False):
        avisos.append(f"{nombre}: corrida exploratoria, no evalúa la regla B del preregistro ({status.get('reason')}).")
    for gap in status.get("preregistration_gaps") or []:
        avisos.append(f"{nombre}: hueco frente al preregistro: {gap}.")
    return avisos


def tablas_pdmx(corrida: Corrida) -> tuple[list[Tabla], list[str]]:
    tablas: list[Tabla] = []
    summary_path, summary_label = corrida.archivo("results_summary.csv")
    summary_rows = _leer_csv(summary_path)
    headline_path, headline_label = corrida.archivo("headline_estimates.json")
    headline = _leer_json(headline_path)
    pairwise_path, pairwise_label = corrida.archivo("pairwise_comparisons.json")
    pairwise = _leer_json(pairwise_path)
    pareto_path, pareto_label = corrida.archivo("pareto_summary.json")
    pareto = _leer_json(pareto_path)
    costs_path, costs_label = corrida.archivo("engineering_costs.csv")
    costs = _leer_csv(costs_path)
    diag_path, diag_label = corrida.archivo("hdp_chain_diagnostics.json")
    diagnostics = _leer_json(diag_path)

    avisos = _estado_analisis(pairwise.get("analysis_status"), "pdmx")
    exploratoria = any("exploratoria" in aviso for aviso in avisos)
    orientation = pairwise.get("difference_orientation", "model_a_minus_model_b")
    if orientation != "model_a_minus_model_b":
        raise ArtefactoInconsistente(f"{pairwise_label}: orientación desconocida {orientation!r}")

    # resultados por modelo y fracción
    estimates = {(e["model"], float(e["frac"]), e["estimand"]): (i, e) for i, e in enumerate(headline["estimates"])}
    t = Tabla("pdmx_resultados.tex", "lrrrrrlrl",
              ("Modelo", "Fracción", "Obras", "Corridas distintas", "Tokens de entrenamiento",
               "NLL media por obra", "IC bootstrap", "NLL por nota", "IC bootstrap"))
    for line, row in enumerate(summary_rows, start=2):
        model, frac = row["model"], float(row["frac"])
        fila = {"model": model, "frac": row["frac"], "linea_csv": line}
        work = estimates.get((model, frac, "work_mean_nll"))
        token = estimates.get((model, frac, "token_weighted_nll"))
        if work is None or token is None:
            raise ArtefactoInconsistente(f"{headline_label} no tiene los estimandos de {model} en frac = {row['frac']}")
        (wi, we), (ti, te) = work, token
        if abs(float(te["point_estimate"]) - float(row["mean_test_nll"])) > TOLERANCIA_ENTRE_ARTEFACTOS:
            raise ArtefactoInconsistente(f"NLL por nota de {model} frac={row['frac']}: {summary_label} y {headline_label} difieren")
        hfila = {"model": model, "frac": frac}
        t.fila([
            NOMBRES.get(model, escapar(model)),
            t.cifra(frac, 2, archivo=summary_label, campo="frac", fila=fila),
            t.cifra(we["n_works"], 0, archivo=headline_label, campo=f"estimates[{wi}].n_works", fila=hfila),
            t.cifra(row["n_distinct_runs"], 0, archivo=summary_label, campo="n_distinct_runs", fila=fila),
            t.cifra(row["mean_n_train_tokens"], 0, archivo=summary_label, campo="mean_n_train_tokens", fila=fila),
            t.cifra(we["point_estimate"], 3, archivo=headline_label, campo=f"estimates[{wi}].point_estimate", fila={**hfila, "estimand": "work_mean_nll"}),
            t.intervalo(we["bootstrap_95_ci"], 3, archivo=headline_label, campo=f"estimates[{wi}].bootstrap_95_ci", fila={**hfila, "estimand": "work_mean_nll"}),
            t.cifra(te["point_estimate"], 3, archivo=headline_label, campo=f"estimates[{ti}].point_estimate", fila={**hfila, "estimand": "token_weighted_nll"}),
            t.intervalo(te["bootstrap_95_ci"], 3, archivo=headline_label, campo=f"estimates[{ti}].bootstrap_95_ci", fila={**hfila, "estimand": "token_weighted_nll"}),
        ])
    t.notas.append("NLL en nats por evento de prueba. Decide la media por obra; la ponderada por notas es secundaria.")
    tablas.append(t)

    # comparaciones pareadas
    t = Tabla("pdmx_comparaciones.tex", "llrrllrrr", ENCABEZADO_COMPARACION)
    _comparaciones(t, pairwise["comparisons"], archivo=pairwise_label, prefijo="comparisons", orientar_a="transformer")
    if exploratoria:
        t.notas.append("Corrida exploratoria: no evalúa la regla B del preregistro.")
    t.notas.append("Veredicto: regla del preregistro (sección B, Aclaración 2) sobre el IC por obra ajustado por Holm por pasos "
                   "(familia de seis); con el transformador, A es el transformador.")
    tablas.append(t)

    # costos en frac = 1.0, con el dispositivo en la misma fila
    t = Tabla("pdmx_costos.tex", "llrrrrrr",
              ("Modelo", "Dispositivo", "Corridas", "Ajuste seleccionado (s)", "Búsqueda completa (s)",
               "Evaluación (s)", "Memoria máx. (MiB)", "Memoria CUDA máx. (MiB)"))
    full = [(line, row) for line, row in enumerate(costs, start=2) if _finito(row.get("frac")) == 1.0]
    for model in sorted({row["model"] for _, row in full}):
        group = [(line, row) for line, row in full if row["model"] == model]
        lines = [line for line, _ in group]
        fila = {"model": model, "frac": 1.0, "filas_csv": lines}

        def media(column):
            values = [_finito(row.get(column)) for _, row in group]
            values = [v for v in values if v is not None]
            if not values:
                return "no medido"
            return t.cifra(statistics.fmean(values), 2, archivo=costs_label, campo=column, fila=fila, operacion="media")

        def maximo_mib(column):
            values = [_finito(row.get(column)) for _, row in group]
            values = [v for v in values if v is not None]
            return t.cifra(max(values) / MIB if values else None, 0, archivo=costs_label, campo=column, fila=fila,
                           operacion="maximo_en_MiB", valor_fuente=max(values) if values else None)

        devices = sorted({row.get("device") or "?" for _, row in group})
        t.fila([NOMBRES.get(model, escapar(model)), escapar(", ".join(devices)),
                t.cifra(len(group), 0, archivo=costs_label, campo="model", fila=fila, operacion="conteo"),
                media("selected_fit_wall_clock_s"), media("full_search_wall_clock_s"), media("evaluation_wall_clock_s"),
                maximo_mib("selection_peak_process_tree_rss_bytes"), maximo_mib("selection_peak_cuda_allocated_bytes")])
    t.notas.append("Fracción completa del entrenamiento. Cada fila declara su dispositivo: un tiempo en CPU y uno en GPU no se comparan sin decirlo.")
    tablas.append(t)

    # frontera de Pareto (NLL contra costo)
    frontier = pareto.get("predictive_cost_partial_frontier", {})
    on_frontier = {c["model"] for c in frontier.get("frontier", [])}
    t = Tabla("pdmx_pareto.tex", "llrlrll",
              ("Modelo", "Dispositivo", "NLL media por obra", "IC", "Ajuste seleccionado (s)", "Intervalo de tiempo", "En la frontera"))
    for index, candidate in enumerate(frontier.get("candidates", [])):
        base = f"predictive_cost_partial_frontier.candidates[{index}]"
        fila = {"model": candidate["model"]}
        t.fila([
            NOMBRES.get(candidate["model"], escapar(candidate["model"])),
            escapar(", ".join(candidate.get("devices", []))),
            t.cifra(candidate["work_mean_test_nll"], 3, archivo=pareto_label, campo=f"{base}.work_mean_test_nll", fila=fila),
            t.intervalo(candidate.get("work_mean_test_nll_interval"), 3, archivo=pareto_label, campo=f"{base}.work_mean_test_nll_interval", fila=fila),
            t.cifra(candidate["mean_selected_fit_wall_clock_s"], 3, archivo=pareto_label, campo=f"{base}.mean_selected_fit_wall_clock_s", fila=fila),
            t.intervalo(candidate.get("mean_selected_fit_wall_clock_s_interval"), 3, archivo=pareto_label,
                        campo=f"{base}.mean_selected_fit_wall_clock_s_interval", fila=fila),
            "sí" if candidate["model"] in on_frontier else "no",
        ])
    skipped = frontier.get("nll_paired_pairs_skipped") or []
    if skipped:
        avisos.append(f"pdmx: {len(skipped)} pares sin diferencia pareada en la frontera de Pareto (se compararon con marginales).")
        t.notas.append("Pares con menos obras en común que el mínimo de la Enmienda 1: comparados con los marginales.")
    if frontier.get("devices_differ"):
        t.notas.append("Los modelos corren en dispositivos distintos (columna Dispositivo).")
    tablas.append(t)

    # diagnóstico de cadenas del HDP
    t = Tabla("pdmx_hdp_diagnosticos.tex", "rrrlrrl",
              ("Fracción", "Semilla de datos", "Cadenas", "Veredicto", "Estados activos (final)", "Autocorrelación lag-uno", "Convergencia"))
    for index, cell in enumerate(diagnostics.get("cells", [])):
        base = f"cells[{index}]"
        info, diag = cell["cell"], cell["diagnostics"]
        fila = dict(info)
        chains = diag.get("chains", {})
        late = [_finito(c.get("late_mean_active_states")) for c in chains.values()]
        late = [v for v in late if v is not None]
        lag = [_finito(c.get("lag_one_autocorrelation")) for c in chains.values()]
        lag = [v for v in lag if v is not None]
        t.fila([
            t.cifra(info.get("frac"), 2, archivo=diag_label, campo=f"{base}.cell.frac", fila=fila),
            t.cifra(info.get("data_seed"), 0, archivo=diag_label, campo=f"{base}.cell.data_seed", fila=fila),
            t.cifra(diag.get("n_chains"), 0, archivo=diag_label, campo=f"{base}.diagnostics.n_chains", fila=fila),
            ident(diag.get("verdict")),
            t.cifra(statistics.fmean(late) if late else None, 1, archivo=diag_label,
                    campo=f"{base}.diagnostics.chains.*.late_mean_active_states", fila=fila, operacion="media_sobre_cadenas"),
            t.cifra(statistics.fmean(lag) if lag else None, 2, archivo=diag_label,
                    campo=f"{base}.diagnostics.chains.*.lag_one_autocorrelation", fila=fila, operacion="media_sobre_cadenas"),
            "declarada" if diag.get("convergence_claimed") else "no declarada",
        ])
    tablas.append(t)
    return tablas, avisos


# --- figuras y escritura --------------------------------------------------------------


def figuras(corrida: Corrida, salida: Path) -> list[dict]:
    listed = []
    for path in sorted(p for p in corrida.directorio.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".pdf", ".svg"}):
        destino = Path("figuras") / f"{corrida.directorio.name}__{path.name}"
        (salida / destino).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, salida / destino)
        listed.append({"origen": f"{corrida.directorio.name}/{path.name}", "ruta_origen": path.resolve().as_posix(),
                       "sha256": _sha256(path), "destino": destino.as_posix()})
    return listed


def _escribir(salida: Path, tabla: Tabla, fuentes: Mapping[str, str], corrida: Path | None) -> None:
    (salida / tabla.nombre).write_text(tabla.tex(), encoding="utf-8")
    procedencia = {
        "tabla": tabla.nombre,
        "corrida": corrida.resolve().as_posix() if corrida else None,
        "fuentes_sha256": dict(fuentes),
        "cifras": tabla.cifras,
    }
    (salida / tabla.nombre.replace(".tex", ".procedencia.json")).write_text(
        json.dumps(procedencia, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _escribir_pendiente(salida: Path, nombre: str, motivo: str) -> None:
    (salida / nombre).write_text(CABECERA + f"\\pendiente{{{motivo}}}\n", encoding="utf-8")
    (salida / nombre.replace(".tex", ".procedencia.json")).write_text(
        json.dumps({"tabla": nombre, "corrida": None, "fuentes_sha256": {}, "cifras": [], "pendiente": motivo},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tablas .tex de la tesis desde los artefactos, con procedencia de cada número.")
    parser.add_argument("--porobra", type=Path, help="Directorio de una corrida de src.cli.porobra_heldout.")
    parser.add_argument("--pdmx", type=Path, help="Directorio de una corrida de Comparacion (PDMX).")
    parser.add_argument("--salida", type=Path, required=True, help="Directorio nuevo o vacío.")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    salida: Path = args.salida
    if salida.exists() and any(salida.iterdir()):
        raise FileExistsError(f"{salida} no está vacío: el generador nunca sobrescribe.")

    # primero se leen y verifican todos los artefactos; sólo después se escribe
    resultados: list[tuple[str, Corrida | None, list[Tabla], list[str]]] = []
    for clave, directorio, construir, nombres in (("porobra", args.porobra, tablas_porobra, TABLAS_POROBRA),
                                                 ("pdmx", args.pdmx, tablas_pdmx, TABLAS_PDMX)):
        if directorio is None:
            resultados.append((clave, None, [], []))
            continue
        corrida = Corrida(Path(directorio))
        tablas, avisos = construir(corrida)
        assert [t.nombre for t in tablas] == list(nombres)
        resultados.append((clave, corrida, tablas, avisos))

    salida.mkdir(parents=True, exist_ok=True)
    resumen: dict[str, Any] = {"corridas": {}, "tablas": [], "pendientes": [], "avisos": []}
    lista_figuras: list[dict] = []
    for clave, corrida, tablas, avisos in resultados:
        nombres = TABLAS_POROBRA if clave == "porobra" else TABLAS_PDMX
        if corrida is None:
            for nombre in nombres:
                _escribir_pendiente(salida, nombre, PENDIENTE[clave])
                resumen["pendientes"].append({"tabla": nombre, "motivo": PENDIENTE[clave]})
            continue
        for tabla in tablas:
            _escribir(salida, tabla, corrida.fuentes, corrida.directorio)
            resumen["tablas"].append({"tabla": tabla.nombre, "n_cifras": len(tabla.cifras)})
        lista_figuras.extend(figuras(corrida, salida))
        manifest = corrida.directorio / "manifest.json"
        resumen["corridas"][clave] = {
            "directorio": corrida.directorio.resolve().as_posix(),
            "commit": _leer_json(manifest).get("commit") if manifest.is_file() else None,
            "fuentes_sha256": corrida.fuentes,
        }
        resumen["avisos"].extend(avisos)
    (salida / "figuras.json").write_text(json.dumps({"figuras": lista_figuras}, indent=2, ensure_ascii=False), encoding="utf-8")
    (salida / "generador_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Tablas escritas en {salida}: {len(resumen['tablas'])} con cifras, {len(resumen['pendientes'])} pendientes.")
    for aviso in resumen["avisos"]:
        print(f"AVISO: {aviso}")
    return salida


if __name__ == "__main__":
    main()

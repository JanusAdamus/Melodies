"""Genera las tablas ``.tex`` de la tesis desde los artefactos de las corridas (tarea 3.2).

Lee, sólo lectura:

* una corrida por obra de ``src/cli/porobra_heldout.py`` (``per_work.csv``,
  ``summary.json``, ``manifest.json``), preregistro sección A;
* una corrida PDMX de ``Comparacion/runner.py`` (``results_summary.csv``,
  ``headline_estimates.json``, ``pairwise_comparisons.json``, ``pareto_summary.json``,
  ``engineering_costs.csv``, ``hdp_chain_diagnostics.json``), preregistro sección B.

Escribe en un directorio de salida nuevo (o vacío) un ``.tex`` por tabla (sólo el
``tabular``: el título y la etiqueta los pone el capítulo) y, junto a cada uno,
``<tabla>.procedencia.json`` con cada número escrito, en el orden en que aparece: su
``id`` (para citarlo en la prosa con ``\\cifra{<id>}{<número>}``), su texto, su valor
sin redondear, el archivo, el campo o columna y la fila, y cómo recalcularlo: ``ruta``
(claves dentro del JSON; ``*`` recorre los valores de un diccionario) o ``columna`` y
``filas_csv`` (líneas del CSV, la 1 es el encabezado), más la ``operacion`` si la hay
(``negado``, ``por_cien``, ``complemento_por_cien``, ``clave``, ``media``,
``desviacion_estandar``, ``conteo``, ``maximo_en_MiB``, ``media_sobre_cadenas``).
``scripts/verificar_cifras_tesis.py`` recalcula cada cifra con eso.  Además
``figuras.json`` (figuras de las corridas, copiadas a ``figuras/`` con su sha256) y
``generador_resumen.json`` (corridas, tablas, avisos y pendientes).

Veredictos: la regla del preregistro (Aclaración 2) sobre el IC de Holm por pasos del
estimando por obra (``work_mean_holm_decision_ci``): «mejora» si el extremo superior es
< 0, «es mejor el otro» si el inferior es > 0, «no hay evidencia» si cruza 0 y «no
evaluable» si Holm no se evaluó (p. ej. menos de 10 obras en común).  En PDMX las
comparaciones con el transformador se orientan «transformador − clásico».

Por obra rige la Enmienda 6: la métrica principal incluye todas las obras analizadas;
la comparación restringida a obras con R-hat de log L bajo el umbral es un análisis de
sensibilidad que no decide.  El generador reconoce, recalculando desde ``per_work.csv``,
sobre qué obras está cada bloque de comparaciones del resumen:

* ``summary["comparisons"]`` sobre todas las obras → tabla principal;
* ``summary["comparisons"]`` sólo sobre las que convergen (``compare_hdp`` de la rama
  ``feat/porobra-heldout`` en 700acba aún hace eso) → tabla de sensibilidad, y la
  principal queda ``\\pendiente``;
* ``summary["sensitivity_converged_only"]`` (misma forma que ``summary["comparisons"]``)
  → tabla de sensibilidad.

Los estados del HDP-HMM por cadena, el R-hat de la NLL predictiva y la NLL de cada
cadena (Enmienda 6, puntos 2 y 4) aún no los escribe la corrida: quedan ``\\pendiente``.

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
CLAVE_SENSIBILIDAD = "sensitivity_converged_only"
# claves de «fila» que no entran en el id: cambian de una corrida a otra
FILA_SIN_ID = {"linea_csv", "filas_csv", "work_ids"}

TABLAS_POROBRA = (
    "porobra_corpus.tex",
    "porobra_comparaciones.tex",
    "porobra_comparaciones_sensibilidad.tex",
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
PENDIENTE_POR_CADENA = ("estados del HDP-HMM por cadena (Enmienda 6, punto 4): la corrida no los reporta por cadena, "
                        "y sin convergencia entre cadenas su resumen conjunto no es una media posterior")
PENDIENTE_NLL_POR_CADENA = ("por obra, R-hat de la NLL predictiva por muestra, NLL de prueba de cada cadena y su rango "
                            "(Enmienda 6, punto 2): la corrida no los trae")


class ArtefactoInconsistente(ValueError):
    """Los artefactos se contradicen entre sí; no se escribe ninguna tabla."""


# --- utilidades ---------------------------------------------------------------------


def veredicto(status: str | None, decision_ci: Sequence[float] | None) -> str:
    """Regla de decisión del preregistro sobre el IC de Holm de (modelo A − modelo B)."""

    if status != "ok" or decision_ci is None:
        return "no evaluable"
    low, high = (float(x) for x in decision_ci)
    if low > high:
        raise ValueError(f"IC de decisión con el extremo inferior {low!r} sobre el superior {high!r}")
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


def formato(valor: float, decimales: int) -> str:
    texto = f"{valor:.{decimales}f}"
    if texto.lstrip("-").strip("0.") == "":
        texto = texto.lstrip("-")  # nunca «-0.000»
    return texto


def campo_de_ruta(ruta: Sequence[Any]) -> str:
    out = ""
    for part in ruta:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


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
    pendientes: list[str] = field(default_factory=list)
    largo: bool = False
    vacia: bool = False  # toda la tabla es un \pendiente

    def _registrar(self, valor: Any, decimales: int, *, archivo: str, campo: str, fila: Mapping[str, Any] | None,
                   operacion: str | None, valor_fuente: Any, receta: Mapping[str, Any]) -> str:
        number = _finito(valor)
        if number is None:
            return "---"
        fila = dict(fila or {})
        texto = formato(number, decimales)
        base = f"{self.nombre.removesuffix('.tex')}:{campo}" + (f":{operacion}" if operacion else "")
        claves = ",".join(f"{k}={v}" for k, v in fila.items() if k not in FILA_SIN_ID)
        ident_cifra = base + (f":{claves}" if claves else "")
        usados = {entry["id"] for entry in self.cifras}
        candidato, n = ident_cifra, 1
        while candidato in usados:
            n += 1
            candidato = f"{ident_cifra}#{n}"
        entry = {"id": candidato, "texto": texto, "valor": number, "archivo": archivo, "campo": campo, "fila": fila, **receta}
        if operacion:
            entry["operacion"] = operacion
        if valor_fuente is not None:
            entry["valor_fuente"] = valor_fuente
        self.cifras.append(entry)
        return "$-$" + texto[1:] if texto.startswith("-") else texto

    def json(self, valor: Any, decimales: int, *, archivo: str, ruta: Sequence[Any], fila: Mapping[str, Any] | None = None,
             operacion: str | None = None, valor_fuente: Any = None) -> str:
        return self._registrar(valor, decimales, archivo=archivo, campo=campo_de_ruta(ruta), fila=fila, operacion=operacion,
                               valor_fuente=valor_fuente, receta={"ruta": list(ruta)})

    def csv(self, valor: Any, decimales: int, *, archivo: str, columna: str, filas_csv: Sequence[int],
            fila: Mapping[str, Any] | None = None, operacion: str | None = None, valor_fuente: Any = None) -> str:
        return self._registrar(valor, decimales, archivo=archivo, campo=columna, fila=fila, operacion=operacion,
                               valor_fuente=valor_fuente, receta={"columna": columna, "filas_csv": [int(x) for x in filas_csv]})

    def intervalo(self, ci: Sequence[Any] | None, decimales: int, *, archivo: str, ruta: Sequence[Any],
                  fila: Mapping[str, Any] | None = None, negar: bool = False) -> str:
        if ci is None:
            return "---"
        values = [_finito(x) for x in ci]
        if any(v is None for v in values):
            return "---"
        ruta = list(ruta)
        if negar:
            low = self.json(-values[1], decimales, archivo=archivo, ruta=ruta + [1], fila=fila, operacion="negado", valor_fuente=values[1])
            high = self.json(-values[0], decimales, archivo=archivo, ruta=ruta + [0], fila=fila, operacion="negado", valor_fuente=values[0])
        else:
            low = self.json(values[0], decimales, archivo=archivo, ruta=ruta + [0], fila=fila)
            high = self.json(values[1], decimales, archivo=archivo, ruta=ruta + [1], fila=fila)
        return f"[{low}, {high}]"

    def fila(self, celdas: Iterable[str]) -> None:
        self.filas.append(" & ".join(celdas) + r" \\")

    def tex(self) -> str:
        lines = [CABECERA, f"% Procedencia de cada número: {self.nombre.replace('.tex', '.procedencia.json')}\n"]
        if self.vacia:
            lines.extend(f"\\pendiente{{{motivo}}}\n" for motivo in self.pendientes)
            return "".join(lines)
        entorno = "longtable" if self.largo else "tabular"
        lines.append(f"\\begin{{{entorno}}}{{{self.columnas}}}\n\\toprule\n")
        lines.append(" & ".join(self.encabezado) + " \\\\\n\\midrule\n")
        if self.largo:
            lines.append("\\endhead\n")
        lines.extend(row + "\n" for row in self.filas)
        lines.append("\\bottomrule\n")
        notas = self.notas + [f"\\pendiente{{{motivo}}}" for motivo in self.pendientes]
        for nota in notas:
            lines.append(f"\\multicolumn{{{len(self.encabezado)}}}{{p{{0.95\\linewidth}}}}{{\\footnotesize {nota}}} \\\\\n")
        lines.append(f"\\end{{{entorno}}}\n")
        return "".join(lines)


def tabla_pendiente(nombre: str, motivo: str) -> Tabla:
    return Tabla(nombre, "", (), pendientes=[motivo], vacia=True)


# --- corrida por obra (preregistro A) -------------------------------------------------


def _filas_porobra(corrida: Corrida) -> tuple[list[dict], str]:
    path, etiqueta = corrida.archivo("per_work.csv")
    rows = _leer_csv(path)
    for index, row in enumerate(rows, start=2):  # fila 1 = encabezado del CSV
        row["_linea_csv"] = index
    return rows, etiqueta


def _filas_convergidas(rows: list[dict]) -> list[dict]:
    return [row for row in rows if _verdadero(row["hdp_converged"])]


def _poblacion(rows: list[dict], bloque: Mapping, etiqueta: str) -> str | None:
    """Sobre qué obras está un bloque de comparaciones: «todas» o «convergidas».

    Se recalcula desde ``per_work.csv`` el número de pares y la media por obra de cada
    comparación; si nada cuadra, los artefactos se contradicen.  Si todas las obras
    convergen, las dos poblaciones coinciden y se dice «todas».
    """

    poblaciones = {"todas": rows, "convergidas": _filas_convergidas(rows)}
    encontradas = set()
    for index, comparison in enumerate(bloque["comparisons"]):
        other = comparison["model_b"]
        computed = veredicto(comparison.get("work_mean_holm_status"), comparison.get("work_mean_holm_decision_ci"))
        declared = comparison.get("verdict")
        if declared is not None and VEREDICTOS_RESUMEN.get(declared, declared) != computed:
            raise ArtefactoInconsistente(
                f"{etiqueta}, comparación {index} (hdp − {other}): veredicto declarado {declared!r}, "
                f"la regla sobre work_mean_holm_decision_ci da {computed!r}"
            )
        if comparison.get("status") != "ok":
            continue
        cuadran = []
        detalle = []
        for nombre, grupo in poblaciones.items():
            paired = [r for r in grupo if _finito(r["hdp_test_nll"]) is not None and _finito(r[f"{other}_test_nll"]) is not None]
            mean = statistics.fmean(float(r["hdp_test_nll"]) - float(r[f"{other}_test_nll"]) for r in paired) if paired else None
            detalle.append(f"{nombre}: {len(paired)} obras, media {mean!r}")
            if paired and len(paired) == int(comparison["n_pairs"]) and abs(mean - float(comparison["mean_difference"])) <= TOLERANCIA_RECALCULO:
                cuadran.append(nombre)
        if not cuadran:
            raise ArtefactoInconsistente(
                f"{etiqueta}, comparación {index} (hdp − {other}): n_pairs={comparison['n_pairs']} y "
                f"mean_difference={comparison['mean_difference']!r} no salen de per_work.csv ({'; '.join(detalle)})"
            )
        encontradas.add(cuadran[0])
    if len(encontradas) > 1:
        raise ArtefactoInconsistente(f"{etiqueta}: unas comparaciones usan todas las obras y otras sólo las que convergen")
    return encontradas.pop() if encontradas else None


def _comparaciones(tabla: Tabla, comparisons: list[dict], *, archivo: str, prefijo: Sequence[Any], orientar_a: str | None = None) -> None:
    for index, comparison in enumerate(comparisons):
        a, b = comparison["model_a"], comparison["model_b"]
        negar = orientar_a is not None and b == orientar_a
        if negar:
            a, b = b, a
        fila = {"model_a": comparison["model_a"], "model_b": comparison["model_b"]}
        ruta = [*prefijo, index]
        status = comparison.get("work_mean_holm_status")
        decision = comparison.get("work_mean_holm_decision_ci")
        if decision is not None and negar:
            decision = [-float(decision[1]), -float(decision[0])]
        verdict = veredicto(status, decision)
        rejects = comparison.get("work_mean_holm_rejects_zero")
        if status == "ok" and rejects is not None and bool(rejects) != (verdict in {"mejora", "es mejor el otro"}):
            raise ArtefactoInconsistente(
                f"{archivo}, {campo_de_ruta(ruta)}: work_mean_holm_rejects_zero={rejects} contradice el IC de decisión {decision}"
            )
        if status != "ok" and status:
            verdict = f"no evaluable ({ESTADOS_HOLM.get(status, escapar(status))})"

        def num(key, decimales, *, signo=True):
            value = comparison.get(key)
            if negar and signo and _finito(value) is not None:
                return tabla.json(-float(value), decimales, archivo=archivo, ruta=ruta + [key], fila=fila,
                                  operacion="negado", valor_fuente=float(value))
            return tabla.json(value, decimales, archivo=archivo, ruta=ruta + [key], fila=fila)

        tabla.fila(
            [
                NOMBRES.get(a, escapar(a)),
                NOMBRES.get(b, escapar(b)),
                num("n_pairs", 0, signo=False),
                num("mean_difference", 3),
                tabla.intervalo(comparison.get("work_mean_holm_decision_ci"), 3, archivo=archivo,
                                ruta=ruta + ["work_mean_holm_decision_ci"], fila=fila, negar=negar),
                verdict,
                num("token_weighted_difference", 3),
                tabla.intervalo(comparison.get("token_weighted_bootstrap_95_ci"), 3, archivo=archivo,
                                ruta=ruta + ["token_weighted_bootstrap_95_ci"], fila=fila, negar=negar),
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
NOTA_REGLA_A = ("Veredicto: regla del preregistro (sección A, Aclaración 2) sobre el IC bootstrap por obra ajustado por "
                "Holm por pasos.")


def _tabla_comparaciones_porobra(nombre: str, bloque: Mapping, *, archivo: str, prefijo: Sequence[Any], nota: str) -> Tabla:
    t = Tabla(nombre, "llrrllrrr", ENCABEZADO_COMPARACION)
    _comparaciones(t, bloque["comparisons"], archivo=archivo, prefijo=[*prefijo, "comparisons"])
    t.notas.extend([nota, NOTA_REGLA_A])
    return t


def tablas_porobra(corrida: Corrida) -> tuple[list[Tabla], list[str]]:
    rows, csv_label = _filas_porobra(corrida)
    summary_path, summary_label = corrida.archivo("summary.json")
    summary = _leer_json(summary_path)
    _leer_json(corrida.archivo("manifest.json")[0])  # la corrida debe estar completa
    avisos: list[str] = []
    tablas: list[Tabla] = []
    converged = _filas_convergidas(rows)
    todas_convergen = len(converged) == len(rows)

    # comparaciones: principal (todas las obras, Enmienda 6) y sensibilidad (sólo las que convergen)
    principal = sensibilidad = None
    poblacion = _poblacion(rows, summary["comparisons"], f"{summary_label}: comparisons")
    if poblacion is None:  # ninguna comparación evaluable: decide la forma del resumen
        poblacion = "convergidas" if ("n_works_converged" in summary["comparisons"] and not todas_convergen) else "todas"
    if poblacion == "todas":
        principal = (summary["comparisons"], ["comparisons"])
        if todas_convergen:
            sensibilidad = principal
    else:
        sensibilidad = (summary["comparisons"], ["comparisons"])
        avisos.append(f"porobra: {summary_label} compara sólo las obras que convergen ({len(converged)} de {len(rows)}; regla "
                      f"sustituida por la Enmienda 6); se reporta como sensibilidad y la comparación principal queda pendiente.")
    if CLAVE_SENSIBILIDAD in summary:
        if _poblacion(rows, summary[CLAVE_SENSIBILIDAD], f"{summary_label}: {CLAVE_SENSIBILIDAD}") not in (None, "convergidas"):
            raise ArtefactoInconsistente(f"{summary_label}: {CLAVE_SENSIBILIDAD} no está restringida a las obras que convergen")
        sensibilidad = (summary[CLAVE_SENSIBILIDAD], [CLAVE_SENSIBILIDAD])

    # corpus
    corpus = summary["corpus"]
    t = Tabla("porobra_corpus.tex", "lr", ("Concepto", "Obras"))
    for source, count in sorted(corpus["n_by_source_candidates"].items()):
        t.fila([f"Candidatas ({escapar(source)})", t.json(count, 0, archivo=summary_label, ruta=["corpus", "n_by_source_candidates", source])])
    for key, label in (("n_failures", "Fallas de lectura"), ("n_duplicates", "Duplicadas por contenido"),
                       ("min_events", "Mínimo de eventos por obra"), ("n_excluded_short", "Excluidas por tener menos eventos"),
                       ("n_analysis_failures", "Fallas en el análisis")):
        if key in corpus:
            t.fila([label, t.json(corpus[key], 0, archivo=summary_label, ruta=["corpus", key])])
    for source, count in sorted(corpus["n_by_source_analyzed"].items()):
        t.fila([f"Analizadas ({escapar(source)})", t.json(count, 0, archivo=summary_label, ruta=["corpus", "n_by_source_analyzed", source])])
    t.fila(["Analizadas (total)", t.json(corpus["n_analyzed"], 0, archivo=summary_label, ruta=["corpus", "n_analyzed"])])
    tablas.append(t)
    failures = summary.get("analysis_failures") or []
    if len(failures) != int(corpus.get("n_analysis_failures", len(failures))):
        raise ArtefactoInconsistente(f"{summary_label}: corpus.n_analysis_failures no coincide con analysis_failures")
    if failures:
        avisos.append("porobra: obras que fallaron en el análisis y quedan fuera de todo: "
                      + ", ".join(str(f.get("work_id")) for f in failures) + ".")
    if int(corpus["n_analyzed"]) != len(rows):
        raise ArtefactoInconsistente(f"{summary_label}: corpus.n_analyzed={corpus['n_analyzed']} y per_work.csv tiene {len(rows)} obras")

    if principal is None:
        tablas.append(tabla_pendiente("porobra_comparaciones.tex",
                                      f"comparación principal de la sección A sobre las {len(rows)} obras analizadas (Enmienda 6, "
                                      f"punto 1): la corrida {corrida.directorio.name} sólo compara las que convergen"))
    else:
        tablas.append(_tabla_comparaciones_porobra(
            "porobra_comparaciones.tex", principal[0], archivo=summary_label, prefijo=principal[1],
            nota="Todas las obras analizadas: ninguna se excluye por la convergencia del HDP-HMM (Enmienda 6)."))
    if sensibilidad is None:
        tablas.append(tabla_pendiente("porobra_comparaciones_sensibilidad.tex",
                                      "análisis de sensibilidad de la sección A, sólo obras con R-hat de log L bajo el umbral "
                                      f"(Enmienda 6, punto 3): la corrida {corrida.directorio.name} no lo trae"))
    else:
        nota = ("Análisis de sensibilidad (Enmienda 6, punto 3; no decide): sólo obras con $\\hat R$ de log L por debajo del umbral."
                + (" Todas convergen: coincide con la comparación principal." if todas_convergen else ""))
        tablas.append(_tabla_comparaciones_porobra("porobra_comparaciones_sensibilidad.tex", sensibilidad[0], archivo=summary_label,
                                                   prefijo=sensibilidad[1], nota=nota))

    # NLL por modelo, sobre todas las obras analizadas (Enmienda 6)
    grupos = [("todas", "Todas", rows)] + [
        (source, escapar(source), [r for r in rows if r["source"] == source]) for source in sorted({r["source"] for r in rows})
    ]
    t = Tabla("porobra_nll.tex", "llrrr", ("Fuente", "Modelo", "Obras", "NLL media por obra", "D.e. entre obras"))
    for clave, etiqueta, grupo in grupos:
        for model in POROBRA_MODELOS:
            column = f"{model}_test_nll"
            used = [r for r in grupo if _finito(r[column]) is not None]
            values = [float(r[column]) for r in used]
            lines = [r["_linea_csv"] for r in used]
            fila = {"fuente": clave, "modelo": model}
            t.fila([
                etiqueta,
                NOMBRES[model],
                t.csv(len(values), 0, archivo=csv_label, columna=column, filas_csv=lines, fila=fila, operacion="conteo"),
                t.csv(statistics.fmean(values) if values else None, 3, archivo=csv_label, columna=column, filas_csv=lines, fila=fila,
                      operacion="media"),
                t.csv(statistics.stdev(values) if len(values) > 1 else None, 3, archivo=csv_label, columna=column, filas_csv=lines,
                      fila=fila, operacion="desviacion_estandar"),
            ])
    t.notas.append("NLL por nota en la parte de prueba (nats); todas las obras analizadas (Enmienda 6).")
    tablas.append(t)

    # estados ocupados: el finito; el HDP-HMM espera el reporte por cadena
    t = Tabla("porobra_estados.tex", "lrrr", ("Fuente", "Obras", "Finito: media", "Finito: d.e."))
    for clave, etiqueta, grupo in grupos:
        used = [r for r in grupo if _finito(r["finite_states_mean"]) is not None]
        values = [float(r["finite_states_mean"]) for r in used]
        lines = [r["_linea_csv"] for r in used]
        fila = {"fuente": clave}
        t.fila([
            etiqueta,
            t.csv(len(values), 0, archivo=csv_label, columna="finite_states_mean", filas_csv=lines, fila=fila, operacion="conteo"),
            t.csv(statistics.fmean(values) if values else None, 1, archivo=csv_label, columna="finite_states_mean", filas_csv=lines,
                  fila=fila, operacion="media"),
            t.csv(statistics.stdev(values) if len(values) > 1 else None, 1, archivo=csv_label, columna="finite_states_mean",
                  filas_csv=lines, fila=fila, operacion="desviacion_estandar"),
        ])
    t.notas.append("Estados del HMM finito con ocupación de al menos el umbral del preregistro; media posterior por obra, "
                   "resumida entre todas las obras analizadas.")
    t.pendientes.append(PENDIENTE_POR_CADENA)
    tablas.append(t)
    por_cadena = sorted(c for c in rows[0] if "chain" in c) if rows else []
    if por_cadena:
        avisos.append(f"porobra: per_work.csv trae columnas por cadena que el generador aún no lee: {', '.join(por_cadena)}.")

    # convergencia (R-hat de log L entre cadenas; sólo se reporta)
    conv = summary["convergence"]
    if int(conv["n_works"]) != len(rows) or int(conv["n_not_converged"]) != len(rows) - len(converged):
        raise ArtefactoInconsistente(f"{summary_label}: convergence no coincide con hdp_converged de per_work.csv")
    t = Tabla("porobra_convergencia.tex", "lr", ("Concepto", "Valor"))
    t.fila(["Umbral de $\\hat R$", t.json(conv["rhat_threshold"], 2, archivo=summary_label, ruta=["convergence", "rhat_threshold"])])
    t.fila(["Obras analizadas", t.json(conv["n_works"], 0, archivo=summary_label, ruta=["convergence", "n_works"])])
    t.fila(["Obras con $\\hat R \\geq$ umbral", t.json(conv["n_not_converged"], 0, archivo=summary_label,
                                                     ruta=["convergence", "n_not_converged"])])
    t.fila(["Porcentaje con $\\hat R <$ umbral",
            t.json(100 * (1 - float(conv["fraction_not_converged"])), 1, archivo=summary_label,
                   ruta=["convergence", "fraction_not_converged"], operacion="complemento_por_cien",
                   valor_fuente=conv["fraction_not_converged"]) + r"\,\%"])
    t.notas.append("$\\hat R$ de log L entre las cadenas del HDP-HMM. Ninguna obra se excluye por convergencia (Enmienda 6).")
    tablas.append(t)

    # hiperparámetros elegidos en validación
    hyper = summary["selected_hyperparameters"]
    t = Tabla("porobra_hiperparametros.tex", "lrr", ("Hiperparámetro", "Valor elegido", "Obras"))
    for key, label, decimales in (("hdp_eta", "$\\eta$ del HDP-HMM", 1), ("finite_k", "$K$ del finito", 0),
                                  ("finite_alpha", "$\\alpha$ del finito", 1),
                                  ("unigram_alpha", "$\\alpha$ del unigrama", 1), ("bigram_alpha", "$\\alpha$ del bigrama", 1)):
        for value, count in sorted(hyper.get(key, {}).items(), key=lambda item: float(item[0])):
            ruta = ["selected_hyperparameters", key, value]
            t.fila([label,
                    t.json(float(value), decimales, archivo=summary_label, ruta=ruta, operacion="clave"),
                    t.json(count, 0, archivo=summary_label, ruta=ruta)])
    if "finite_alpha" not in hyper:
        t.pendientes.append(f"$\\alpha$ del finito elegido en validación (Enmienda 5): la corrida {escapar(corrida.directorio.name)} no lo trae")
    if hyper.get("finite_k_hit_limit"):
        avisos.append(f"porobra: {len(hyper['finite_k_hit_limit'])} obras con la rejilla del finito en el techo.")
    tablas.append(t)

    # por obra (apéndice)
    con_alpha = bool(rows) and "finite_alpha" in rows[0]
    encabezado = ["Obra", "Fuente", "Notas prueba", "HDP-HMM", "Finito", "Bigrama", "Unigrama", "Estados finito",
                  "$\\hat R$ (log L)", "$\\eta$", "$K$"] + (["$\\alpha$ finito"] if con_alpha else [])
    t = Tabla("porobra_por_obra.tex", "llrrrrrlrrr" + ("r" if con_alpha else ""), tuple(encabezado), largo=True)
    for row in rows:
        fila = {"work_id": row["work_id"]}

        def c(column, decimales):
            return t.csv(row.get(column), decimales, archivo=csv_label, columna=column, filas_csv=[row["_linea_csv"]], fila=fila)

        t.fila([ident(row["work_id"]), escapar(row["source"]), c("n_test", 0),
                *(c(f"{m}_test_nll", 3) for m in POROBRA_MODELOS),
                f"{c('finite_states_mean', 1)} $\\pm$ {c('finite_states_sd', 1)}",
                c("hdp_rhat", 2) + ("" if _verdadero(row["hdp_converged"]) else r"\textsuperscript{*}"),
                c("hdp_eta", 1), c("finite_k", 0), *([c("finite_alpha", 1)] if con_alpha else [])])
    t.notas.append("\\textsuperscript{*} $\\hat R$ de log L por encima del umbral; la obra sigue en la métrica principal (Enmienda 6).")
    t.pendientes.append(PENDIENTE_NLL_POR_CADENA)
    tablas.append(t)

    orden = {nombre: i for i, nombre in enumerate(TABLAS_POROBRA)}
    tablas.sort(key=lambda tabla: orden[tabla.nombre])
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
        fila = {"model": model, "frac": row["frac"]}
        work = estimates.get((model, frac, "work_mean_nll"))
        token = estimates.get((model, frac, "token_weighted_nll"))
        if work is None or token is None:
            raise ArtefactoInconsistente(f"{headline_label} no tiene los estimandos de {model} en frac = {row['frac']}")
        (wi, we), (ti, te) = work, token
        if abs(float(te["point_estimate"]) - float(row["mean_test_nll"])) > TOLERANCIA_ENTRE_ARTEFACTOS:
            raise ArtefactoInconsistente(f"NLL por nota de {model} frac={row['frac']}: {summary_label} y {headline_label} difieren")
        wfila = {**fila, "estimand": "work_mean_nll"}
        tfila = {**fila, "estimand": "token_weighted_nll"}
        t.fila([
            NOMBRES.get(model, escapar(model)),
            t.csv(frac, 2, archivo=summary_label, columna="frac", filas_csv=[line], fila=fila),
            t.json(we["n_works"], 0, archivo=headline_label, ruta=["estimates", wi, "n_works"], fila=wfila),
            t.csv(row["n_distinct_runs"], 0, archivo=summary_label, columna="n_distinct_runs", filas_csv=[line], fila=fila),
            t.csv(row["mean_n_train_tokens"], 0, archivo=summary_label, columna="mean_n_train_tokens", filas_csv=[line], fila=fila),
            t.json(we["point_estimate"], 3, archivo=headline_label, ruta=["estimates", wi, "point_estimate"], fila=wfila),
            t.intervalo(we["bootstrap_95_ci"], 3, archivo=headline_label, ruta=["estimates", wi, "bootstrap_95_ci"], fila=wfila),
            t.json(te["point_estimate"], 3, archivo=headline_label, ruta=["estimates", ti, "point_estimate"], fila=tfila),
            t.intervalo(te["bootstrap_95_ci"], 3, archivo=headline_label, ruta=["estimates", ti, "bootstrap_95_ci"], fila=tfila),
        ])
    t.notas.append("NLL en nats por evento de prueba. Decide la media por obra; la ponderada por notas es secundaria.")
    tablas.append(t)

    # comparaciones pareadas
    t = Tabla("pdmx_comparaciones.tex", "llrrllrrr", ENCABEZADO_COMPARACION)
    _comparaciones(t, pairwise["comparisons"], archivo=pairwise_label, prefijo=["comparisons"], orientar_a="transformer")
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
        fila = {"model": model, "frac": 1.0}

        def medidos(column):
            return [(line, _finito(row.get(column))) for line, row in group if _finito(row.get(column)) is not None]

        def media(column):
            used = medidos(column)
            if not used:
                return "no medido"
            return t.csv(statistics.fmean(v for _, v in used), 2, archivo=costs_label, columna=column,
                         filas_csv=[line for line, _ in used], fila=fila, operacion="media")

        def maximo_mib(column):
            used = medidos(column)
            if not used:
                return "---"
            top = max(v for _, v in used)
            return t.csv(top / MIB, 0, archivo=costs_label, columna=column, filas_csv=[line for line, _ in used], fila=fila,
                         operacion="maximo_en_MiB", valor_fuente=top)

        devices = sorted({row.get("device") or "?" for _, row in group})
        t.fila([NOMBRES.get(model, escapar(model)), escapar(", ".join(devices)),
                t.csv(len(group), 0, archivo=costs_label, columna="model", filas_csv=[line for line, _ in group], fila=fila,
                      operacion="conteo"),
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
        base = ["predictive_cost_partial_frontier", "candidates", index]
        fila = {"model": candidate["model"]}
        t.fila([
            NOMBRES.get(candidate["model"], escapar(candidate["model"])),
            escapar(", ".join(candidate.get("devices", []))),
            t.json(candidate["work_mean_test_nll"], 3, archivo=pareto_label, ruta=base + ["work_mean_test_nll"], fila=fila),
            t.intervalo(candidate.get("work_mean_test_nll_interval"), 3, archivo=pareto_label, ruta=base + ["work_mean_test_nll_interval"], fila=fila),
            t.json(candidate["mean_selected_fit_wall_clock_s"], 3, archivo=pareto_label, ruta=base + ["mean_selected_fit_wall_clock_s"], fila=fila),
            t.intervalo(candidate.get("mean_selected_fit_wall_clock_s_interval"), 3, archivo=pareto_label,
                        ruta=base + ["mean_selected_fit_wall_clock_s_interval"], fila=fila),
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
        base = ["cells", index]
        info, diag = cell["cell"], cell["diagnostics"]
        fila = dict(info)
        chains = diag.get("chains", {})
        late = [_finito(c.get("late_mean_active_states")) for c in chains.values()]
        late = [v for v in late if v is not None]
        lag = [_finito(c.get("lag_one_autocorrelation")) for c in chains.values()]
        lag = [v for v in lag if v is not None]
        t.fila([
            t.json(info.get("frac"), 2, archivo=diag_label, ruta=base + ["cell", "frac"], fila=fila),
            t.json(info.get("data_seed"), 0, archivo=diag_label, ruta=base + ["cell", "data_seed"], fila=fila),
            t.json(diag.get("n_chains"), 0, archivo=diag_label, ruta=base + ["diagnostics", "n_chains"], fila=fila),
            ident(diag.get("verdict")),
            t.json(statistics.fmean(late) if late else None, 1, archivo=diag_label,
                   ruta=base + ["diagnostics", "chains", "*", "late_mean_active_states"], fila=fila, operacion="media_sobre_cadenas"),
            t.json(statistics.fmean(lag) if lag else None, 2, archivo=diag_label,
                   ruta=base + ["diagnostics", "chains", "*", "lag_one_autocorrelation"], fila=fila, operacion="media_sobre_cadenas"),
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
    if tabla.pendientes:
        procedencia["pendientes"] = list(tabla.pendientes)
    (salida / tabla.nombre.replace(".tex", ".procedencia.json")).write_text(
        json.dumps(procedencia, indent=2, ensure_ascii=False), encoding="utf-8"
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
            resultados.append((clave, None, [tabla_pendiente(n, PENDIENTE[clave]) for n in nombres], []))
            continue
        corrida = Corrida(Path(directorio))
        tablas, avisos = construir(corrida)
        assert [t.nombre for t in tablas] == list(nombres)
        resultados.append((clave, corrida, tablas, avisos))

    salida.mkdir(parents=True, exist_ok=True)
    resumen: dict[str, Any] = {"corridas": {}, "tablas": [], "pendientes": [], "avisos": []}
    lista_figuras: list[dict] = []
    for clave, corrida, tablas, avisos in resultados:
        for tabla in tablas:
            _escribir(salida, tabla, corrida.fuentes if corrida else {}, corrida.directorio if corrida else None)
            if tabla.cifras:
                resumen["tablas"].append({"tabla": tabla.nombre, "n_cifras": len(tabla.cifras)})
            resumen["pendientes"].extend({"tabla": tabla.nombre, "motivo": motivo} for motivo in tabla.pendientes)
        if corrida is None:
            continue
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

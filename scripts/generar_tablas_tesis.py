"""Genera las tablas ``.tex`` de la tesis desde los artefactos de las corridas (tarea 3.2).

Lee, sólo lectura:

* una corrida por obra de ``src/cli/porobra_heldout.py`` (``per_work.csv``,
  ``summary.json``, ``manifest.json``), preregistro sección A;
* una corrida PDMX de ``Comparacion/runner.py`` (``results_summary.csv``,
  ``headline_estimates.json``, ``pairwise_comparisons.json``, ``pareto_summary.json``,
  ``engineering_costs.csv``, ``hdp_chain_diagnostics.json``, ``config.json``,
  ``preprocessing_report.json`` y las auditorías ``protocol_audit.json``,
  ``training_exposure_audit.json`` y ``denominator_audit.json``), preregistro sección B.
  Una corrida con el protocolo fallido o con evaluación traslapada no se escribe; una que
  no es la principal de B (``DISENO_B_PRINCIPAL``: el smoke, las sensibilidades) o que es
  exploratoria se escribe con «IC Holm (regla)», la nota y el aviso de que no decide.

Escribe en un directorio de salida nuevo (o vacío) un ``.tex`` por tabla (sólo el
``tabular``: el título y la etiqueta los pone el capítulo; la cabecera nombra la corrida) y,
junto a cada uno, ``<tabla>.procedencia.json`` con el ``.tex`` completo (``tex``: el
verificador compara también rótulos, veredictos, ids y notas) y cada número escrito, en el
orden en que aparece: su
``id`` (para citarlo en la prosa con ``\\cifra{<id>}{<número>}``), su texto, su valor
sin redondear, el archivo, el campo o columna y la fila, y cómo recalcularlo: ``ruta``
(claves dentro del JSON; ``*`` recorre los valores de un diccionario) o ``columna`` y
``filas_csv`` (líneas del CSV, la 1 es el encabezado), más la ``operacion`` si la hay
(``negado``, ``por_cien``, ``complemento_por_cien``, ``clave``, ``media``,
``desviacion_estandar``, ``conteo``, ``longitud``, ``maximo_en_MiB``, ``media_sobre_cadenas``).
``scripts/verificar_cifras_tesis.py`` recalcula cada cifra con eso.  Además
``figuras.json`` (figuras de las corridas, copiadas a ``figuras/`` con su sha256) y
``generador_resumen.json`` (corridas, tablas, avisos y pendientes).

Veredictos: la regla del preregistro (Aclaración 2) sobre el IC de Holm por pasos del
estimando por obra (``work_mean_holm_decision_ci``): «mejora» si el extremo superior es
< 0, «es mejor el otro» si el inferior es > 0, «no hay evidencia» si cruza 0 y «no
evaluable» si Holm no se evaluó (p. ej. menos de 10 obras en común).  En PDMX las
comparaciones con el transformador se orientan «transformador − clásico».

Por obra se lee sólo el formato final de ``src/cli/porobra_heldout.py`` (un formato
anterior se rechaza): ``per_work.csv``, ``summary.json`` (con el bloque ``run``),
``manifest.json`` y ``porobra_audit.json``.  Una corrida cuya auditoría de integridad no
pasó no se escribe; una que no es la confirmatoria (``confirmatory`` falso) se escribe
con la nota y el aviso de que sus veredictos no deciden.  La auditoría debe ser de esta
corrida (``run_dir``), ``per_work.csv`` y ``summary.json`` los que hasheó el manifiesto
(``outputs``), y el bloque ``run`` del resumen debe cuadrar con la auditoría (tipo y
desviaciones) y con el truncamiento del manifiesto.  Las columnas
``hdp_states_pooled_chains_*`` deben estar pero no se escriben (Enmienda 6, punto 4).  Rige la Enmienda 6: la métrica
principal (``comparisons``) incluye todas las obras analizadas; las sensibilidades, que no
deciden, son ``sensitivity_converged_only`` (Enmienda 6, punto 3),
``sensitivity_without_exposed`` (Aclaraciones 8 a 10) y ``sensitivity_without_k_ceiling``
(Enmienda 11), cada una en su tabla con los cambios de veredicto frente a la principal
(«no evaluable» no es cambio).  De cada bloque se recalcula desde ``per_work.csv`` su
población (obras excluidas, sólo convergidas), ``n_works``, ``n_pairs``, la media por
obra, el aviso de menos de 30 obras y el mínimo de 10 obras para un veredicto (Enmienda 11).
Se reportan además el α del finito (Enmienda 5), las obras con K en el tope y con el
reajuste del finito en el tope de iteraciones de EM, y por obra y cadena el R-hat
predictivo, la NLL de cada cadena con su rango y los estados por cadena (Enmienda 6,
puntos 2 y 4), sin la media agregada de cadenas.  Una cifra de una lista guardada en una
celda del CSV lleva ``celda`` en la procedencia (índice, o ``"rango"``: máximo − mínimo).

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
# Enmienda 11: con menos de 10 obras, cifras sin veredicto; con menos de 30, aviso
MIN_OBRAS_VEREDICTO = 10
AVISO_POCAS_OBRAS = 30
# claves de «fila» que no entran en el id: cambian de una corrida a otra
FILA_SIN_ID = {"linea_csv", "filas_csv", "work_ids"}

TABLAS_POROBRA = (
    "porobra_corpus.tex",
    "porobra_comparaciones.tex",
    "porobra_comparaciones_sensibilidad.tex",
    "porobra_sensibilidad_sin_expuestas.tex",
    "porobra_sensibilidad_sin_techo_k.tex",
    "porobra_nll.tex",
    "porobra_estados.tex",
    "porobra_convergencia.tex",
    "porobra_hiperparametros.tex",
    "porobra_por_obra.tex",
    "porobra_por_obra_cadenas.tex",
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
# formato final de src/cli/porobra_heldout.py (Enmiendas 5, 6 y 11; Aclaraciones 8 a 10)
CLAVES_RESUMEN = ("run", "corpus", "analysis_failures", "convergence", "comparisons", "sensitivity_converged_only",
                  "test_exposure", "sensitivity_without_exposed", "sensitivity_without_k_ceiling", "selected_hyperparameters")
CLAVES_HIPERPARAMETROS = ("hdp_eta", "finite_k", "finite_alpha", "finite_k_hit_limit", "finite_em_hit_max_iterations",
                          "unigram_alpha", "bigram_alpha")
COLUMNAS_POR_OBRA = ("work_id", "source", "n_test", "hdp_test_nll", "finite_test_nll", "bigram_test_nll", "unigram_test_nll",
                     "hdp_converged", "hdp_rhat", "hdp_predictive_rhat", "hdp_test_nll_by_chain", "hdp_test_nll_chain_range",
                     "hdp_states_mean_by_chain", "hdp_states_sd_by_chain", "hdp_states_pooled_chains_mean",
                     "hdp_states_pooled_chains_sd", "hdp_eta", "finite_k", "finite_alpha",
                     "finite_k_hit_limit", "finite_em_hit_max_iterations", "finite_states_mean", "finite_states_sd")
ESTADOS_HOLM_POROBRA = {**ESTADOS_HOLM, "too_few_paired_works": r"menos de \estructural{10} obras"}
# (bloque del resumen, tabla, sólo convergidas, obras excluidas, nota)
SENSIBILIDADES_POROBRA = (
    ("sensitivity_converged_only", "porobra_comparaciones_sensibilidad.tex", True, "",
     r"Análisis de sensibilidad (Enmienda 6, punto 3; no decide): sólo obras con $\hat R$ de log L por debajo del umbral."),
    ("sensitivity_without_exposed", "porobra_sensibilidad_sin_expuestas.tex", False, "expuestas",
     "Análisis adicional (Aclaración 8, completada por las Aclaraciones 9 y 10; no decide): sin las obras con exposición a "
     "cifras de prueba antes de la corrida confirmatoria."),
    ("sensitivity_without_k_ceiling", "porobra_sensibilidad_sin_techo_k.tex", False, "techo",
     "Análisis de sensibilidad (Enmienda 11; no decide): sin las obras cuyo HMM finito eligió el tope de la rejilla de $K$."),
)
NOTA_POCAS_OBRAS = (r"Bloque con menos de \estructural{30} obras: léase con cautela; el aviso no cambia la regla ni el "
                    "veredicto (Enmienda 11).")
NOTA_NO_EVALUABLE = (r"No evaluable: con menos de \estructural{10} obras se reportan las cifras sin veredicto, y eso no "
                     "cuenta como cambio de veredicto (Enmienda 11).")
NOTA_ESTADOS_HDP = ("Estados del HDP-HMM por cadena (Enmienda 6, punto 4): sin convergencia entre cadenas no son una media "
                    "posterior y no se interpretan como inferencia bayesiana sobre $K$; no se comparan con los del finito.")
NOTA_TECHO_K = ("La rejilla de $K$ del finito se amplía mientras se elige el máximo y termina en el tope; elegir el tope no es "
                r"desviación (Enmienda 11). Seudoconteo $\alpha$ del finito elegido junto con $K$ en validación (Enmienda 5).")
NOTA_EXPUESTAS = (r"\textsuperscript{\dag} Obra con exposición a cifras de prueba antes de la corrida confirmatoria "
                  "(Aclaraciones 8 a 10); sigue en la métrica principal.")
NOTA_MARCA_TECHO = (r"\textsuperscript{\ddag} El finito eligió el tope de la rejilla de $K$ (Enmienda 11): su $K$ no se "
                    "interpreta como número de estados; la obra sigue en la métrica principal.")
NOTA_MARCA_EM = r"\textsuperscript{\S} El reajuste del finito terminó por el tope de iteraciones de EM, no por la tolerancia."


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
            fila: Mapping[str, Any] | None = None, operacion: str | None = None, valor_fuente: Any = None,
            celda: int | str | None = None) -> str:
        """``celda``: la celda es una lista en JSON; un índice toma ese elemento y ``"rango"``, máximo − mínimo."""

        receta: dict[str, Any] = {"columna": columna, "filas_csv": [int(x) for x in filas_csv]}
        if celda is not None:
            receta["celda"] = celda
            fila = {**(fila or {}), "celda": celda}
        return self._registrar(valor, decimales, archivo=archivo, campo=columna, fila=fila, operacion=operacion,
                               valor_fuente=valor_fuente, receta=receta)

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

    def tex(self, corrida: str | None = None) -> str:
        """``corrida``: nombre de la corrida, en la cabecera (dos corridas escriben tablas del mismo nombre)."""

        de = f" (corrida {corrida})" if corrida else ""
        lines = [CABECERA, f"% Procedencia de cada número: {self.nombre.replace('.tex', '.procedencia.json')}{de}\n"]
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
    faltan = [c for c in COLUMNAS_POR_OBRA if rows and c not in rows[0]]
    if faltan:
        raise ArtefactoInconsistente(f"{etiqueta} no tiene las columnas del formato final de la corrida por obra: {', '.join(faltan)}")
    for index, row in enumerate(rows, start=2):  # fila 1 = encabezado del CSV
        row["_linea_csv"] = index
    return rows, etiqueta


def _lista(row: Mapping, columna: str) -> list[float]:
    return [float(x) for x in json.loads(row[columna])]


def _formato_final(summary: Mapping, etiqueta: str) -> None:
    faltan = [k for k in CLAVES_RESUMEN if k not in summary]
    faltan += [f"convergence.{k}" for k in ("n_converged", "fraction_converged") if k not in summary.get("convergence", {})]
    faltan += [f"selected_hyperparameters.{k}" for k in CLAVES_HIPERPARAMETROS if k not in summary.get("selected_hyperparameters", {})]
    faltan += [f"run.{k}" for k in ("truncation", "run_kind", "deviations") if k not in (summary.get("run") or {})]
    if faltan:
        raise ArtefactoInconsistente(f"{etiqueta}: formato anterior al final de la corrida por obra; falta {', '.join(faltan)}")


# archivos de la corrida con hash en manifest["outputs"] (src/cli/porobra_heldout.py, OUTPUTS)
SALIDAS_CON_HASH = ("per_work.csv", "summary.json")


def _misma_corrida(corrida: Corrida, summary: Mapping, manifest: Mapping, audit: Mapping, *, summary_label: str,
                   manifest_label: str, audit_label: str) -> None:
    """La auditoría, el manifiesto y el resumen deben hablar de estos mismos archivos.

    Un ``porobra_audit.json`` de otra corrida, o un ``per_work.csv`` o ``summary.json``
    rehechos después del manifiesto, darían tablas con el rótulo de una auditoría que no
    las revisó.
    """

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ArtefactoInconsistente(f"{manifest_label}: sin outputs (hashes de los archivos de la corrida)")
    for nombre in SALIDAS_CON_HASH:
        path, _ = corrida.archivo(nombre)
        if outputs.get(nombre) != _sha256(path):
            raise ArtefactoInconsistente(f"{manifest_label}: outputs.{nombre} no es el sha256 de {nombre} "
                                         "(archivo cambiado después de la corrida)")
    run_dir = str(audit.get("run_dir") or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if run_dir != corrida.directorio.resolve().name:
        raise ArtefactoInconsistente(f"{audit_label}: run_dir={audit.get('run_dir')!r} no es esta corrida "
                                     f"({corrida.directorio.name})")
    failures = summary.get("analysis_failures") or []
    if audit.get("status") == "passed" and failures:
        raise ArtefactoInconsistente(f"{audit_label}: status passed con analysis_failures en {summary_label} "
                                     "(la auditoría las marca failed): la auditoría no es de este resumen")
    run = summary["run"]
    prereg = audit.get("preregistration") or {}
    kind = prereg.get("run_kind")
    # el bloque run sólo ve las desviaciones de diseño; la auditoría añade comando, resumen y topes del corpus
    if kind != "no_preregistrada" and run.get("run_kind") != kind:
        raise ArtefactoInconsistente(f"{summary_label}: run.run_kind={run.get('run_kind')!r} y {audit_label} dice {kind!r}")
    if not set(run.get("deviations") or []) <= set(prereg.get("deviations") or {}):
        raise ArtefactoInconsistente(f"{summary_label}: run.deviations={run.get('deviations')!r} no están en las de {audit_label}")
    truncation = (manifest.get("config") or {}).get("truncation")
    if run.get("truncation") != truncation:
        raise ArtefactoInconsistente(f"{summary_label}: run.truncation={run.get('truncation')!r} y {manifest_label} "
                                     f"config.truncation={truncation!r}")


def _verificar_bloque(rows: list[dict], bloque: Mapping, clave: str, etiqueta: str, *, converged_only: bool,
                      excluidas: Sequence[str]) -> None:
    """Un bloque de comparaciones debe salir de ``per_work.csv`` con la población que dice su regla.

    Se recalculan desde el CSV las obras del bloque, el número de pares y la media por obra
    de cada comparación, el aviso de pocas obras, el mínimo de obras para un veredicto
    (Enmienda 11) y que el veredicto declarado sea el de la regla sobre el IC de Holm.
    """

    donde = f"{etiqueta}: {clave}"
    if bool(bloque.get("converged_only")) != converged_only:
        raise ArtefactoInconsistente(f"{donde}: converged_only={bloque.get('converged_only')!r}, su regla pide {converged_only}")
    if sorted(bloque.get("excluded_work_ids") or []) != sorted(excluidas):
        raise ArtefactoInconsistente(f"{donde}: excluded_work_ids={bloque.get('excluded_work_ids')!r}, per_work.csv da {sorted(excluidas)!r}")
    poblacion = [r for r in rows if (_verdadero(r["hdp_converged"]) or not converged_only) and r["work_id"] not in set(excluidas)]
    if int(bloque["n_works"]) != len(poblacion):
        raise ArtefactoInconsistente(f"{donde}: n_works={bloque['n_works']} y per_work.csv da {len(poblacion)} obras")
    if bool(bloque.get("small_n_warning")) != (len(poblacion) < AVISO_POCAS_OBRAS):
        raise ArtefactoInconsistente(f"{donde}: small_n_warning={bloque.get('small_n_warning')!r} con {len(poblacion)} obras")
    for index, comparison in enumerate(bloque["comparisons"]):
        other = comparison["model_b"]
        computed = veredicto(comparison.get("work_mean_holm_status"), comparison.get("work_mean_holm_decision_ci"))
        declared = comparison.get("verdict")
        if declared is not None and VEREDICTOS_RESUMEN.get(declared, declared) != computed:
            raise ArtefactoInconsistente(f"{donde}, comparación {index} (hdp − {other}): veredicto declarado {declared!r}, "
                                         f"la regla sobre work_mean_holm_decision_ci da {computed!r}")
        paired = [r for r in poblacion if _finito(r["hdp_test_nll"]) is not None and _finito(r[f"{other}_test_nll"]) is not None]
        if comparison.get("status") != "ok":
            if paired:
                raise ArtefactoInconsistente(f"{donde}, comparación {index}: sin datos y per_work.csv tiene {len(paired)} pares")
            continue
        mean = statistics.fmean(float(r["hdp_test_nll"]) - float(r[f"{other}_test_nll"]) for r in paired) if paired else None
        if not paired or len(paired) != int(comparison["n_pairs"]) or abs(mean - float(comparison["mean_difference"])) > TOLERANCIA_RECALCULO:
            raise ArtefactoInconsistente(f"{donde}, comparación {index} (hdp − {other}): n_pairs={comparison['n_pairs']} y "
                                         f"mean_difference={comparison['mean_difference']!r} no salen de per_work.csv "
                                         f"({len(paired)} obras, media {mean!r})")
        evaluable = comparison.get("work_mean_holm_status") == "ok"
        if evaluable != (len(paired) >= MIN_OBRAS_VEREDICTO):
            raise ArtefactoInconsistente(f"{donde}, comparación {index}: work_mean_holm_status="
                                         f"{comparison.get('work_mean_holm_status')!r} con {len(paired)} obras (mínimo 10, Enmienda 11)")


def _cambios(principal: Mapping, bloque: Mapping, donde: str) -> tuple[list[dict], list[dict]]:
    """Cambios de veredicto frente a la principal; «no_evaluable» no es un cambio (Enmienda 11)."""

    antes = {c["model_b"]: c["verdict"] for c in principal["comparisons"]}
    pares = [{"model_b": c["model_b"], "principal": antes[c["model_b"]], "sensitivity": c["verdict"]} for c in bloque["comparisons"]]
    no_evaluables = [p for p in pares if "no_evaluable" in (p["principal"], p["sensitivity"])]
    cambios = [p for p in pares if p not in no_evaluables and p["principal"] != p["sensitivity"]]
    if bloque.get("verdict_changes") != cambios or bloque.get("verdict_not_evaluable", no_evaluables) != no_evaluables:
        raise ArtefactoInconsistente(f"{donde}: verdict_changes={bloque.get('verdict_changes')!r} no es lo que dan los veredictos "
                                     f"({cambios!r})")
    return cambios, no_evaluables


def _comparaciones(tabla: Tabla, comparisons: list[dict], *, archivo: str, prefijo: Sequence[Any], orientar_a: str | None = None,
                   estados: Mapping[str, str] = ESTADOS_HOLM) -> None:
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
            verdict = f"no evaluable ({estados.get(status, escapar(status))})"

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


def _encabezado_comparacion(decide: bool) -> tuple[str, ...]:
    return tuple(c.replace("IC Holm (decide)", "IC Holm (regla)") if not decide else c for c in ENCABEZADO_COMPARACION)


def _tabla_comparaciones_porobra(nombre: str, bloque: Mapping, *, archivo: str, prefijo: Sequence[Any], notas: Sequence[str],
                                 decide: bool) -> Tabla:
    t = Tabla(nombre, "llrrllrrr", _encabezado_comparacion(decide))
    _comparaciones(t, bloque["comparisons"], archivo=archivo, prefijo=[*prefijo, "comparisons"], estados=ESTADOS_HOLM_POROBRA)
    t.notas.extend(notas)
    if bloque.get("small_n_warning"):
        t.notas.append(NOTA_POCAS_OBRAS)
    if any(c.get("work_mean_holm_status") == "too_few_paired_works" for c in bloque["comparisons"]):
        t.notas.append(NOTA_NO_EVALUABLE)
    t.notas.append(NOTA_REGLA_A)
    return t


def _nota_cambios(cambios: list[dict], no_evaluables: list[dict]) -> str:
    def v(code):
        return VEREDICTOS_RESUMEN.get(code, escapar(code))

    if cambios:
        texto = "Cambia el veredicto frente a la comparación principal: " + "; ".join(
            f"HDP-HMM $-$ {NOMBRES.get(p['model_b'], escapar(p['model_b']))}: {v(p['principal'])} $\\to$ {v(p['sensitivity'])}"
            for p in cambios) + "."
    else:
        texto = "Ningún veredicto cambia frente a la comparación principal."
    if no_evaluables:
        texto += (" Sin veredicto en " + ", ".join(NOMBRES.get(p["model_b"], escapar(p["model_b"])) for p in no_evaluables)
                  + ": no cuenta como cambio de veredicto (Enmienda 11).")
    return texto


def tablas_porobra(corrida: Corrida) -> tuple[list[Tabla], list[str], dict]:
    rows, csv_label = _filas_porobra(corrida)
    summary_path, summary_label = corrida.archivo("summary.json")
    summary = _leer_json(summary_path)
    _formato_final(summary, summary_label)
    manifest_path, manifest_label = corrida.archivo("manifest.json")
    manifest = _leer_json(manifest_path)
    audit_path, audit_label = corrida.archivo("porobra_audit.json")
    audit = _leer_json(audit_path)
    if audit.get("status") != "passed":
        raise ArtefactoInconsistente(f"{audit_label}: la auditoría de integridad no pasó (status={audit.get('status')!r}); "
                                     "sus cifras no se escriben")
    _misma_corrida(corrida, summary, manifest, audit, summary_label=summary_label, manifest_label=manifest_label,
                   audit_label=audit_label)
    confirmatoria = bool(audit.get("confirmatory"))
    prereg = audit.get("preregistration") or {}
    avisos: list[str] = []
    tablas: list[Tabla] = []
    nota_corrida = []
    if not confirmatoria:
        tipo = prereg.get("run_kind") or (summary.get("run") or {}).get("run_kind")
        desvios = sorted(prereg.get("deviations") or {})
        avisos.append(f"porobra: {corrida.directorio.name} no es la confirmatoria de la sección A (porobra_audit.json: tipo "
                      f"{tipo}{', desviaciones ' + ', '.join(desvios) if desvios else ''}); sus veredictos no deciden.")
        nota_corrida.append(f"Esta corrida no es la confirmatoria (porobra\\_audit.json, tipo {ident(tipo)}): sus veredictos no deciden.")

    # comparaciones: principal (todas las obras, Enmienda 6) y las tres sensibilidades
    exposure = summary["test_exposure"]
    ids = {r["work_id"] for r in rows}
    expuestas = sorted(set(exposure.get("exposed_work_ids") or []) & ids)
    if sorted(exposure.get("exposed_in_run") or []) != expuestas:
        raise ArtefactoInconsistente(f"{summary_label}: test_exposure.exposed_in_run no es la intersección de exposed_work_ids "
                                     "con las obras de per_work.csv")
    techo = [r["work_id"] for r in rows if _verdadero(r["finite_k_hit_limit"])]
    principal = summary["comparisons"]
    _verificar_bloque(rows, principal, "comparisons", summary_label, converged_only=False, excluidas=[])
    tablas.append(_tabla_comparaciones_porobra(
        "porobra_comparaciones.tex", principal, archivo=summary_label, prefijo=["comparisons"], decide=confirmatoria,
        notas=["Todas las obras analizadas: ninguna se excluye por la convergencia del HDP-HMM (Enmienda 6).", *nota_corrida]))
    for clave, nombre, converged_only, excluidas, nota in SENSIBILIDADES_POROBRA:
        bloque = summary[clave]
        excl = {"expuestas": expuestas, "techo": techo, "": []}[excluidas]
        _verificar_bloque(rows, bloque, clave, summary_label, converged_only=converged_only, excluidas=excl)
        cambios, no_evaluables = _cambios(principal, bloque, f"{summary_label}: {clave}")
        tablas.append(_tabla_comparaciones_porobra(nombre, bloque, archivo=summary_label, prefijo=[clave], decide=False,
                                                   notas=[nota, _nota_cambios(cambios, no_evaluables), *nota_corrida]))
        if cambios:
            avisos.append(f"porobra: {clave} cambia {len(cambios)} veredicto(s) frente a la comparación principal.")

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
    t.fila(["Analizadas con exposición a cifras de prueba (Aclaraciones 8 a 10)",
            t.json(len(expuestas), 0, archivo=summary_label, ruta=["test_exposure", "exposed_in_run"], operacion="longitud")])
    tablas.append(t)
    failures = summary.get("analysis_failures") or []
    if len(failures) != int(corpus.get("n_analysis_failures", len(failures))):
        raise ArtefactoInconsistente(f"{summary_label}: corpus.n_analysis_failures no coincide con analysis_failures")
    if failures:
        avisos.append("porobra: obras que fallaron en el análisis y quedan fuera de todo: "
                      + ", ".join(str(f.get("work_id")) for f in failures) + ".")
    if int(corpus["n_analyzed"]) != len(rows):
        raise ArtefactoInconsistente(f"{summary_label}: corpus.n_analyzed={corpus['n_analyzed']} y per_work.csv tiene {len(rows)} obras")

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

    # cadenas: misma cantidad en cada obra y en el manifiesto; el rango declarado, el de las cadenas
    n_cadenas = {len(_lista(r, "hdp_test_nll_by_chain")) for r in rows} | {len(_lista(r, "hdp_states_mean_by_chain")) for r in rows}
    declaradas = (manifest.get("config") or {}).get("n_chains")
    if len(n_cadenas) != 1 or (declaradas is not None and n_cadenas != {int(declaradas)}):
        raise ArtefactoInconsistente(f"{csv_label}: número de cadenas por obra {sorted(n_cadenas)}, el manifiesto dice {declaradas}")
    n_cadenas = n_cadenas.pop()
    for r in rows:
        by_chain = _lista(r, "hdp_test_nll_by_chain")
        if abs(max(by_chain) - min(by_chain) - float(r["hdp_test_nll_chain_range"])) > TOLERANCIA_RECALCULO:
            raise ArtefactoInconsistente(f"{csv_label}: {r['work_id']}: hdp_test_nll_chain_range no es el rango de hdp_test_nll_by_chain")

    # estados ocupados: el finito (media posterior) y, del HDP-HMM, sólo cuánto difieren sus cadenas
    t = Tabla("porobra_estados.tex", "lrrrr", ("Fuente", "Obras", "Finito: media", "Finito: d.e.",
                                               "HDP-HMM: rango entre cadenas (media entre obras)"))
    for clave, etiqueta, grupo in grupos:
        used = [r for r in grupo if _finito(r["finite_states_mean"]) is not None]
        values = [float(r["finite_states_mean"]) for r in used]
        lines = [r["_linea_csv"] for r in used]
        rangos = [max(_lista(r, "hdp_states_mean_by_chain")) - min(_lista(r, "hdp_states_mean_by_chain")) for r in grupo]
        fila = {"fuente": clave}
        t.fila([
            etiqueta,
            t.csv(len(values), 0, archivo=csv_label, columna="finite_states_mean", filas_csv=lines, fila=fila, operacion="conteo"),
            t.csv(statistics.fmean(values) if values else None, 1, archivo=csv_label, columna="finite_states_mean", filas_csv=lines,
                  fila=fila, operacion="media"),
            t.csv(statistics.stdev(values) if len(values) > 1 else None, 1, archivo=csv_label, columna="finite_states_mean",
                  filas_csv=lines, fila=fila, operacion="desviacion_estandar"),
            t.csv(statistics.fmean(rangos) if rangos else None, 1, archivo=csv_label, columna="hdp_states_mean_by_chain",
                  filas_csv=[r["_linea_csv"] for r in grupo], fila=fila, operacion="media", celda="rango"),
        ])
    t.notas.append("Estados del HMM finito con ocupación de al menos el umbral del preregistro; media posterior por obra, "
                   "resumida entre todas las obras analizadas.")
    t.notas.append(NOTA_ESTADOS_HDP)
    tablas.append(t)

    # convergencia (R-hat de log L entre cadenas; sólo se reporta)
    conv = summary["convergence"]
    converged = [r for r in rows if _verdadero(r["hdp_converged"])]
    if (int(conv["n_works"]) != len(rows) or int(conv["n_converged"]) != len(converged)
            or abs(float(conv["fraction_converged"]) - len(converged) / len(rows)) > TOLERANCIA_RECALCULO
            or sorted(conv.get("not_converged_work_ids") or []) != sorted(r["work_id"] for r in rows if r not in converged)):
        raise ArtefactoInconsistente(f"{summary_label}: convergence no coincide con hdp_converged de per_work.csv")
    t = Tabla("porobra_convergencia.tex", "lr", ("Concepto", "Valor"))
    t.fila(["Umbral de $\\hat R$", t.json(conv["rhat_threshold"], 2, archivo=summary_label, ruta=["convergence", "rhat_threshold"])])
    t.fila(["Obras analizadas", t.json(conv["n_works"], 0, archivo=summary_label, ruta=["convergence", "n_works"])])
    t.fila(["Obras con $\\hat R <$ umbral", t.json(conv["n_converged"], 0, archivo=summary_label, ruta=["convergence", "n_converged"])])
    t.fila(["Porcentaje con $\\hat R <$ umbral",
            t.json(100 * float(conv["fraction_converged"]), 1, archivo=summary_label, ruta=["convergence", "fraction_converged"],
                   operacion="por_cien", valor_fuente=conv["fraction_converged"]) + r"\,\%"])
    t.notas.append("$\\hat R$ de log L entre las cadenas del HDP-HMM. Ninguna obra se excluye por convergencia (Enmienda 6).")
    tablas.append(t)

    # hiperparámetros elegidos en validación (Enmiendas 3 y 5) y topes del finito (Enmienda 11)
    hyper = summary["selected_hyperparameters"]
    for key, column in (("hdp_eta", "hdp_eta"), ("finite_k", "finite_k"), ("finite_alpha", "finite_alpha"),
                        ("unigram_alpha", "unigram_alpha"), ("bigram_alpha", "bigram_alpha")):
        declared = {float(k): int(v) for k, v in hyper[key].items()}
        counted: dict[float, int] = {}
        for r in rows:
            counted[float(r[column])] = counted.get(float(r[column]), 0) + 1
        if declared != counted:
            raise ArtefactoInconsistente(f"{summary_label}: selected_hyperparameters.{key} no es la distribución de {column} en per_work.csv")
    em = [r["work_id"] for r in rows if _verdadero(r["finite_em_hit_max_iterations"])]
    for key, esperado in (("finite_k_hit_limit", techo), ("finite_em_hit_max_iterations", em)):
        if sorted(hyper[key]) != sorted(esperado):
            raise ArtefactoInconsistente(f"{summary_label}: selected_hyperparameters.{key} no coincide con per_work.csv")
    tope = (manifest.get("config") or {}).get("max_states")
    if tope is None or any(int(float(r["finite_k"])) != int(tope) for r in rows if r["work_id"] in techo):
        raise ArtefactoInconsistente(f"{manifest_label}: max_states={tope!r} no es el K de las obras con finite_k_hit_limit")
    t = Tabla("porobra_hiperparametros.tex", "lrr", ("Hiperparámetro", "Valor elegido", "Obras"))
    for key, label, decimales in (("hdp_eta", "$\\eta$ del HDP-HMM", 1), ("finite_k", "$K$ del finito", 0),
                                  ("finite_alpha", "$\\alpha$ del finito", 1),
                                  ("unigram_alpha", "$\\alpha$ del unigrama", 1), ("bigram_alpha", "$\\alpha$ del bigrama", 1)):
        for value, count in sorted(hyper[key].items(), key=lambda item: float(item[0])):
            ruta = ["selected_hyperparameters", key, value]
            t.fila([label,
                    t.json(float(value), decimales, archivo=summary_label, ruta=ruta, operacion="clave"),
                    t.json(count, 0, archivo=summary_label, ruta=ruta)])
    t.fila(["Obras con $K$ en el tope de la rejilla",
            t.json(tope, 0, archivo=manifest_label, ruta=["config", "max_states"]),
            t.json(len(techo), 0, archivo=summary_label, ruta=["selected_hyperparameters", "finite_k_hit_limit"], operacion="longitud")])
    t.fila(["Obras con el reajuste del finito en el tope de iteraciones de EM", "---",
            t.json(len(em), 0, archivo=summary_label, ruta=["selected_hyperparameters", "finite_em_hit_max_iterations"],
                   operacion="longitud")])
    t.notas.append(NOTA_TECHO_K)
    tablas.append(t)
    if techo:
        avisos.append(f"porobra: {len(techo)} obras eligieron el tope de K del finito (Enmienda 11: no es desviación; su K no "
                      "se interpreta como número de estados).")
    if em:
        avisos.append(f"porobra: {len(em)} obras terminaron el reajuste del finito por el tope de iteraciones de EM.")

    # por obra (apéndice): NLL e hiperparámetros
    encabezado = ("Obra", "Fuente", "Notas prueba", "HDP-HMM", "Finito", "Bigrama", "Unigrama", "Estados finito",
                  "$\\eta$", "$\\alpha$ finito", "$K$")
    t = Tabla("porobra_por_obra.tex", "llrrrrrlrrr", encabezado, largo=True)
    for row in rows:
        fila = {"work_id": row["work_id"]}

        def c(column, decimales, **kw):
            return t.csv(row.get(column), decimales, archivo=csv_label, columna=column, filas_csv=[row["_linea_csv"]], fila=fila, **kw)

        marca_obra = r"\textsuperscript{\dag}" if row["work_id"] in expuestas else ""
        marca_em = r"\textsuperscript{\S}" if row["work_id"] in em else ""
        marca_k = r"\textsuperscript{\ddag}" if row["work_id"] in techo else ""
        t.fila([ident(row["work_id"]) + marca_obra, escapar(row["source"]), c("n_test", 0),
                c("hdp_test_nll", 3), c("finite_test_nll", 3) + marca_em, c("bigram_test_nll", 3), c("unigram_test_nll", 3),
                f"{c('finite_states_mean', 1)} $\\pm$ {c('finite_states_sd', 1)}",
                c("hdp_eta", 1), c("finite_alpha", 1), c("finite_k", 0) + marca_k])
    t.notas.extend([NOTA_EXPUESTAS, NOTA_MARCA_TECHO, NOTA_MARCA_EM])
    tablas.append(t)

    # por obra (apéndice): diagnóstico por cadena del HDP-HMM (Enmienda 6, puntos 2 y 4)
    encabezado = ("Obra", "$\\hat R$ (log L)", "$\\hat R$ (NLL predictiva)",
                  *(f"NLL $c_{{{i}}}$" for i in range(1, n_cadenas + 1)), "Rango", "Estados por cadena: media", "D.e.")
    t = Tabla("porobra_por_obra_cadenas.tex", "lrr" + "r" * n_cadenas + "rll", encabezado, largo=True)
    for row in rows:
        fila = {"work_id": row["work_id"]}

        def c(column, decimales, **kw):
            return t.csv(row.get(column) if "celda" not in kw else _lista(row, column)[kw["celda"]], decimales, archivo=csv_label,
                         columna=column, filas_csv=[row["_linea_csv"]], fila=fila, **kw)

        t.fila([ident(row["work_id"]),
                c("hdp_rhat", 2) + ("" if _verdadero(row["hdp_converged"]) else r"\textsuperscript{*}"),
                c("hdp_predictive_rhat", 2),
                *(c("hdp_test_nll_by_chain", 3, celda=i) for i in range(n_cadenas)),
                c("hdp_test_nll_chain_range", 3),
                ", ".join(c("hdp_states_mean_by_chain", 1, celda=i) for i in range(n_cadenas)),
                ", ".join(c("hdp_states_sd_by_chain", 1, celda=i) for i in range(n_cadenas))])
    t.notas.extend(["\\textsuperscript{*} $\\hat R$ de log L por encima del umbral; la obra sigue en la métrica principal (Enmienda 6).",
                    "NLL de prueba de cada cadena por separado y su rango; $\\hat R$ de la NLL predictiva por muestra (Enmienda 6, punto 2).",
                    NOTA_ESTADOS_HDP])
    tablas.append(t)

    orden = {nombre: i for i, nombre in enumerate(TABLAS_POROBRA)}
    tablas.sort(key=lambda tabla: orden[tabla.nombre])
    meta = {"confirmatoria": confirmatoria, "decide": confirmatoria, "tipo": prereg.get("run_kind"),
            "auditoria": audit.get("status")}
    return tablas, avisos, meta


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



# --- corrida PDMX (preregistro B) -----------------------------------------------------


#: Diseño de la corrida principal de la sección B en ``config.json`` (``asdict(LearningCurveConfig)``):
#: partición con semilla 7, 3 semillas de datos × 2 de modelo, VOMM en la familia (Aclaración 2),
#: HMM finito en la GPU con tope final de K = 768 (Enmienda 14), extracción de melodía y lo demás
#: como la corrida vigente ``tesis_corregida_20261104_9ee8dda`` (valores por omisión de
#: ``Comparacion/config.py`` y de ``next_token_experiment/config.py``).  Un campo distinto, o
#: ausente, hace que la corrida no decida.  Las sensibilidades de B (sólo frac = 1.0;
#: ``--train-stride 128``; ``--split-seed`` 17 y 29, ``SENSIBILIDADES_B``) y el smoke difieren
#: en alguno; sus diferencias esperadas no tocan a la corrida principal.
DISENO_B_PRINCIPAL = (
    (("split_seed",), 7),
    (("data_seeds",), [1, 2, 3]),
    (("model_seeds",), [1, 2]),
    (("train_fractions",), [0.1, 0.25, 0.5, 0.75, 1.0]),
    (("test_fraction",), 0.15),
    (("validation_fraction",), 0.1),
    (("include_vomm_control",), True),
    (("finite_hmm_max_states",), 768),
    (("finite_hmm_final_ceiling",), True),
    (("finite_hmm_device",), "cuda"),
    (("experiment", "preprocessing", "extraction_mode"), "melody"),
    (("experiment", "windows", "max_context_length"), 128),
    (("experiment", "windows", "train_stride"), 64),
    (("experiment", "windows", "eval_stride"), 128),
    (("experiment", "transformer", "max_epochs"), 25),
)
#: sensibilidades de B (preregistro): campos que pueden diferir, con sus valores; no deciden
SENSIBILIDADES_B = {
    "split_seed": (17, 29),
    "train_fractions": ([1.0],),
    "experiment.windows.train_stride": (128,),
}
COMMIT_PRINCIPAL_B = "9ee8dda"
TOPE_K_B = 768
#: partituras consideradas (``--max-files 3000``): preparadas + excluidas en preprocessing_report.json
PARTITURAS_B = 3000


def _igual(actual: Any, esperado: Any) -> bool:
    if isinstance(esperado, str):
        return actual == esperado
    if isinstance(esperado, bool):
        return actual is esperado
    if isinstance(esperado, list):
        return isinstance(actual, list) and len(actual) == len(esperado) and all(_igual(a, e) for a, e in zip(actual, esperado))
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return float(actual) == float(esperado)


def _diferencias_con_la_principal_b(config: Mapping, preprocessing: Mapping) -> list[str]:
    return [texto for _, texto in _diferencias_b(config, preprocessing)]


def _tipo_de_corrida_b(diferencias: list[tuple[str, str]], config: Mapping) -> str:
    """principal, sensibilidad (sólo difiere en lo que el preregistro permite) u otra."""

    if not diferencias:
        return "principal"
    for campo, _ in diferencias:
        permitidos = SENSIBILIDADES_B.get(campo)
        actual: Any = config
        for key in campo.split("."):
            actual = actual.get(key) if isinstance(actual, Mapping) else None
        if permitidos is None or not any(_igual(actual, valor) for valor in permitidos):
            return "otra"
    return "sensibilidad"


def _diferencias_b(config: Mapping, preprocessing: Mapping) -> list[tuple[str, str]]:
    diferencias = []
    for ruta, esperado in DISENO_B_PRINCIPAL:
        actual: Any = config
        for key in ruta:
            actual = actual.get(key) if isinstance(actual, Mapping) else None
        if not _igual(actual, esperado):
            diferencias.append((".".join(ruta), f"{'.'.join(ruta)}={actual!r} (preregistro: {esperado!r})"))
    try:
        partituras = int(preprocessing["n_prepared_pieces"]) + int(preprocessing["n_exclusions"])
    except (KeyError, TypeError, ValueError):
        partituras = None
    if partituras != PARTITURAS_B:
        diferencias.append(("partituras", f"partituras={partituras!r} (preregistro: {PARTITURAS_B})"))
    return diferencias


def _verdadero_csv(valor: Any) -> bool:
    return str(valor).strip().lower() in {"true", "1", "yes", "si", "sí"}


def _tope_final(corrida: Corrida, config: Mapping) -> dict[str, Any]:
    """Filas del HMM finito que tocaron el tope final de K (Enmienda 14), de ``results_raw.csv`` y de
    ``finite_hmm_grid_audit.json``.  Tocar el tope no detiene la corrida; bloquea la afirmación de la curva de aprendizaje."""

    tope = config.get("finite_hmm_max_states", TOPE_K_B)
    info: dict[str, Any] = {"tope_k": tope, "filas_en_tope_k": None, "filas_finito": None, "auditoria_rejilla": None,
                            "avisos": []}
    raw_path, raw_label = corrida.archivo("results_raw.csv")
    audit_path, audit_label = corrida.archivo("finite_hmm_grid_audit.json")
    tocadas: list[int] = []
    if raw_path.is_file():
        filas = [r for r in _leer_csv(raw_path) if r.get("model") == "finite_hmm"]
        tocadas.append(sum(_verdadero_csv(r.get("hit_final_ceiling")) for r in filas))
        info["filas_finito"] = len(filas)
    if audit_path.is_file():
        audit = _leer_json(audit_path)
        info["auditoria_rejilla"] = audit.get("verdict")
        en_maximo = audit.get("n_at_grid_maximum")
        if isinstance(en_maximo, int) and not isinstance(en_maximo, bool) and audit.get("grid_maximum") in (None, tope):
            tocadas.append(en_maximo)
        if info["filas_finito"] is None and isinstance(audit.get("n_selections"), int):
            info["filas_finito"] = audit["n_selections"]
    if not tocadas:
        if config.get("finite_hmm_final_ceiling") is True:
            info["avisos"].append(f"pdmx: no se pudo comprobar el tope final de K = {tope} del HMM finito ({raw_label} y "
                                  f"{audit_label} no existen); la afirmación de la curva de aprendizaje queda bloqueada.")
            info["bloqueada"] = True
        else:
            info["bloqueada"] = False
        return info
    info["filas_en_tope_k"] = max(tocadas)
    info["bloqueada"] = info["filas_en_tope_k"] > 0
    if info["bloqueada"]:
        total = info["filas_finito"] if info["filas_finito"] is not None else "?"
        info["avisos"].append(
            f"pdmx: {info['filas_en_tope_k']} de {total} filas del HMM finito tocaron el tope final K = {tope} "
            "(Enmienda 14); ahí el HMM finito puede estar subestimado, y la afirmación de la curva de aprendizaje "
            "queda bloqueada.")
    return info


def _nota_tope(info: Mapping[str, Any]) -> str | None:
    if not info.get("bloqueada"):
        return None
    if info.get("filas_en_tope_k") is None:
        return ("No se pudo comprobar el tope final de $K$ del HMM finito: no se afirma que aproveche más datos (curva de "
                "aprendizaje bloqueada, Enmienda 14).")
    total = info["filas_finito"] if info["filas_finito"] is not None else "?"
    return (rf"\estructural{{{info['filas_en_tope_k']}}} de \estructural{{{total}}} filas del HMM finito tocaron el tope final "
            rf"$K = \estructural{{{info['tope_k']}}}$ (Enmienda 14): ahí el HMM finito puede estar subestimado, y no se "
            "afirma que aproveche más datos (curva de aprendizaje bloqueada).")


def _commit_de_la_corrida(directorio: Path) -> tuple[str | None, str | None]:
    """(commit, archivo) del primer archivo de la corrida que lo registra: manifest.json, config.json, execution_plan.json."""

    claves = ("commit", "git_commit", "commit_hash", "git_sha")

    def buscar(dato: Any, profundidad: int = 0) -> str | None:
        if not isinstance(dato, Mapping) or profundidad > 2:
            return None
        for clave in claves:
            valor = dato.get(clave)
            if isinstance(valor, str) and valor.strip():
                return valor.strip()
        for valor in dato.values():
            encontrado = buscar(valor, profundidad + 1)
            if encontrado:
                return encontrado
        return None

    for nombre in ("manifest.json", "config.json", "execution_plan.json"):
        path = directorio / nombre
        if path.is_file():
            commit = buscar(_leer_json(path))
            if commit:
                return commit, nombre
    return None, None


def _condiciones_de_costo(costs: list[dict[str, str]]) -> tuple[bool, str]:
    """(aislados, descripción): sólo cuenta como aislada una fila con condición ``isolated`` y ``resource_cost_usable``."""

    if not costs:
        return False, "unknown"
    condiciones = sorted({row.get("resource_measurement_condition") or "unknown" for row in costs})
    usable = all(_verdadero_csv(row.get("resource_cost_usable")) for row in costs)
    return condiciones == ["isolated"] and usable, ", ".join(condiciones)


def _nota_costos_no_aislados(condiciones: str) -> str:
    return (rf"Costos no aislados (resource\_measurement\_condition: {escapar(condiciones)}): son tiempos medidos con "
            "otra carga en la máquina y no son costos de ingeniería comparables; no se leen como costo de cada modelo.")


def _auditorias_pdmx(corrida: Corrida) -> list[str]:
    """Las auditorías de la corrida PDMX: una que no pasó detiene el generador, como en la sección A."""

    path, label = corrida.archivo("protocol_audit.json")
    protocol = _leer_json(path)
    if protocol.get("status") != "passed":
        raise ArtefactoInconsistente(f"{label}: la auditoría de protocolo no pasó (status={protocol.get('status')!r}); "
                                     "sus cifras no se escriben")
    path, label = corrida.archivo("training_exposure_audit.json")
    exposure = _leer_json(path)
    if exposure.get("evaluation_is_non_overlapping") is not True:
        raise ArtefactoInconsistente(f"{label}: la evaluación de validación o prueba no es sin traslape "
                                     f"(evaluation_is_non_overlapping={exposure.get('evaluation_is_non_overlapping')!r})")
    path, label = corrida.archivo("denominator_audit.json")
    denominators = _leer_json(path)
    if "status" in denominators and denominators["status"] not in ("ok", "passed"):
        raise ArtefactoInconsistente(f"{label}: status={denominators['status']!r}")
    avisos = []
    sin_nll = denominators.get("unscored_test_files") or []
    descartes = denominators.get("discards") or []
    if sin_nll or descartes:
        avisos.append(f"pdmx: {label}: {len(sin_nll)} archivo(s) de prueba sin NLL y {len(descartes)} fila(s) descartada(s); "
                      "las comparaciones usan sólo lo evaluado.")
    return avisos


def _estado_analisis(status: Mapping | None, nombre: str) -> list[str]:
    avisos = []
    status = status or {}
    if status.get("status") != "preregistered" or not status.get("evaluates_preregistered_rule_B", False):
        avisos.append(f"{nombre}: corrida exploratoria, no evalúa la regla B del preregistro ({status.get('reason')}).")
    for gap in status.get("preregistration_gaps") or []:
        avisos.append(f"{nombre}: hueco frente al preregistro: {gap}.")
    return avisos


def tablas_pdmx(corrida: Corrida, costos: Corrida | None = None) -> tuple[list[Tabla], list[str], dict]:
    """``costos``: corrida de costo aparte (opción explícita); sin ella, los costos son los de la corrida."""

    tablas: list[Tabla] = []
    summary_path, summary_label = corrida.archivo("results_summary.csv")
    summary_rows = _leer_csv(summary_path)
    headline_path, headline_label = corrida.archivo("headline_estimates.json")
    headline = _leer_json(headline_path)
    pairwise_path, pairwise_label = corrida.archivo("pairwise_comparisons.json")
    pairwise = _leer_json(pairwise_path)
    de_costos = costos or corrida
    pareto_path, pareto_label = de_costos.archivo("pareto_summary.json")
    pareto = _leer_json(pareto_path)
    costs_path, costs_label = de_costos.archivo("engineering_costs.csv")
    costs = _leer_csv(costs_path)
    diag_path, diag_label = corrida.archivo("hdp_chain_diagnostics.json")
    diagnostics = _leer_json(diag_path)

    avisos = _estado_analisis(pairwise.get("analysis_status"), "pdmx")
    exploratoria = any("exploratoria" in aviso for aviso in avisos)
    avisos += _auditorias_pdmx(corrida)
    config_path, _ = corrida.archivo("config.json")
    prep_path, _ = corrida.archivo("preprocessing_report.json")
    config = _leer_json(config_path)
    diferencias_pares = _diferencias_b(config, _leer_json(prep_path))
    diferencias = [texto for _, texto in diferencias_pares]
    tipo = _tipo_de_corrida_b(diferencias_pares, config)
    tope = _tope_final(corrida, config)
    avisos += tope["avisos"]
    nota_tope = _nota_tope(tope)
    commit, commit_fuente = _commit_de_la_corrida(corrida.directorio)
    if commit is None:
        avisos.append(f"pdmx: sin commit registrado en {corrida.directorio.name} (ni manifest.json, ni config.json, ni "
                      "execution_plan.json): no se puede comprobar contra el commit de la corrida principal "
                      f"({COMMIT_PRINCIPAL_B}).")
    costos_aislados, condiciones = _condiciones_de_costo(costs)
    nota_costos = None if costos_aislados else _nota_costos_no_aislados(condiciones)
    if nota_costos:
        avisos.append(f"pdmx: costos no aislados ({de_costos.directorio.name}: {condiciones}); tiempos y frontera de Pareto "
                      "no se leen como costo de ingeniería.")
    nota_origen_costos = (f"Costos medidos en la corrida aparte {escapar(de_costos.directorio.name)} (opción explícita)."
                          if costos else None)
    if diferencias:
        avisos.append(f"pdmx: {corrida.directorio.name} no es la corrida principal de la sección B (difiere en "
                      f"{'; '.join(diferencias)}); sus veredictos no deciden.")
    decide = not exploratoria and not diferencias
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
    if nota_tope:
        t.notas.append(nota_tope)
    tablas.append(t)

    # comparaciones pareadas
    t = Tabla("pdmx_comparaciones.tex", "llrrllrrr", _encabezado_comparacion(decide))
    _comparaciones(t, pairwise["comparisons"], archivo=pairwise_label, prefijo=["comparisons"], orientar_a="transformer")
    if exploratoria:
        t.notas.append("Corrida exploratoria: no evalúa la regla B del preregistro.")
    nota_no_principal = ("Esta corrida no es la corrida principal de la sección B (detalle en generador\\_resumen.json): "
                         "sus veredictos no deciden.")
    if diferencias:
        t.notas.append(nota_no_principal)
    if nota_tope:
        t.notas.append(nota_tope)
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
    if nota_origen_costos:
        t.notas.append(nota_origen_costos)
    if nota_costos:
        t.notas.append(nota_costos)
    tablas.append(t)

    # frontera de Pareto (NLL contra costo)
    frontier = pareto.get("predictive_cost_partial_frontier") or {}
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
    pareto_evaluado = frontier.get("status") == "ok" and bool(frontier.get("candidates"))
    if not pareto_evaluado:
        avisos.append(f"pdmx: frontera de Pareto no evaluada en {pareto_label} (status={frontier.get('status')!r}).")
        t.notas.append("Frontera de Pareto no evaluada: la tabla no dice qué modelo domina a cuál.")
    excluidos = [str(m) for m in frontier.get("excluded_models") or []]
    if excluidos:
        avisos.append(f"pdmx: modelos excluidos de la frontera de Pareto: {', '.join(excluidos)}.")
        t.notas.append("Modelos excluidos de la frontera (sin tiempo de ajuste seleccionado): "
                       + ", ".join(NOMBRES.get(m, escapar(m)) for m in excluidos) + ".")
    if nota_origen_costos:
        t.notas.append(nota_origen_costos)
    if nota_costos:
        t.notas.append(nota_costos)
    if diferencias:
        t.notas.append(nota_no_principal)
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
    return tablas, avisos, {
        "decide": decide, "tipo": tipo, "diferencias_con_la_principal": diferencias, "tope_k": tope["tope_k"],
        "filas_en_tope_k": tope["filas_en_tope_k"], "filas_finito": tope["filas_finito"],
        "auditoria_rejilla": tope["auditoria_rejilla"], "curva_aprendizaje_bloqueada": tope["bloqueada"],
        "costos_aislados": costos_aislados, "corrida_de_costos": de_costos.directorio.resolve().as_posix() if costos else None,
        "pareto_evaluado": pareto_evaluado, "modelos_excluidos_pareto": excluidos,
        "commit": commit, "commit_fuente": commit_fuente,
    }


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
    tex = tabla.tex(corrida.resolve().name if corrida else None)
    (salida / tabla.nombre).write_text(tex, encoding="utf-8")
    procedencia = {
        "tabla": tabla.nombre,
        "corrida": corrida.resolve().as_posix() if corrida else None,
        "fuentes_sha256": dict(fuentes),
        "cifras": tabla.cifras,
        # el .tex tal como se escribió: el verificador compara también rótulos, veredictos, ids y notas
        "tex": tex,
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
    parser.add_argument("--costos", type=Path,
                        help="Corrida de costo aparte (misma estructura que una corrida PDMX): sus engineering_costs.csv y "
                             "pareto_summary.json sustituyen a los de --pdmx en pdmx_costos.tex y pdmx_pareto.tex.")
    parser.add_argument("--salida", type=Path, required=True, help="Directorio nuevo o vacío.")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    salida: Path = args.salida
    if salida.exists() and any(salida.iterdir()):
        raise FileExistsError(f"{salida} no está vacío: el generador nunca sobrescribe.")

    # primero se leen y verifican todos los artefactos; sólo después se escribe
    resultados: list[tuple[str, Corrida | None, list[Tabla], list[str], dict]] = []
    for clave, directorio, construir, nombres in (("porobra", args.porobra, tablas_porobra, TABLAS_POROBRA),
                                                 ("pdmx", args.pdmx, tablas_pdmx, TABLAS_PDMX)):
        if directorio is None:
            resultados.append((clave, None, [tabla_pendiente(n, PENDIENTE[clave]) for n in nombres], [], {}))
            continue
        corrida = Corrida(Path(directorio))
        if clave == "pdmx" and args.costos is not None:
            tablas, avisos, meta = construir(corrida, Corrida(Path(args.costos)))
        else:
            tablas, avisos, meta = construir(corrida)
        assert [t.nombre for t in tablas] == list(nombres)
        resultados.append((clave, corrida, tablas, avisos, meta))

    salida.mkdir(parents=True, exist_ok=True)
    resumen: dict[str, Any] = {"corridas": {}, "tablas": [], "pendientes": [], "avisos": []}
    lista_figuras: list[dict] = []
    for clave, corrida, tablas, avisos, meta in resultados:
        for tabla in tablas:
            _escribir(salida, tabla, corrida.fuentes if corrida else {}, corrida.directorio if corrida else None)
            if tabla.cifras:
                resumen["tablas"].append({"tabla": tabla.nombre, "n_cifras": len(tabla.cifras)})
            resumen["pendientes"].extend({"tabla": tabla.nombre, "motivo": motivo} for motivo in tabla.pendientes)
        if corrida is None:
            continue
        lista_figuras.extend(figuras(corrida, salida))
        commit, commit_fuente = meta.get("commit"), meta.get("commit_fuente")
        if "commit" not in meta:
            commit, commit_fuente = _commit_de_la_corrida(corrida.directorio)
            if commit is None:
                avisos.append(f"{clave}: sin commit registrado en {corrida.directorio.name}.")
        resumen["corridas"][clave] = {
            "directorio": corrida.directorio.resolve().as_posix(),
            "fuentes_sha256": corrida.fuentes,
            **meta,
            "commit": commit,
            "commit_fuente": commit_fuente,
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

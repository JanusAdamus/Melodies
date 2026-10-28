"""Verifica que cada número de los ``.tex`` de la tesis tenga fuente (tarea 3.4).

Recorre todos los ``.tex`` de un directorio (texto y tablas), extrae cada número y lo
empareja con una fuente:

* los JSON de procedencia que escribe ``scripts/generar_tablas_tesis.py`` (cada cifra con
  su valor sin redondear, archivo, campo y fila), y
* reportes de texto (p. ej. ``docs/resultados-corregidos-<fecha>.md`` o un ``.json`` de
  una corrida), de los que se toma cada número escrito.

Un número empareja si su texto coincide con el de una cifra de procedencia, o si una
fuente redondeada a los decimales escritos da ese número (un entero sólo empareja con
una fuente entera; un número seguido de ``\\%`` también empareja con la fuente × 100).
El signo no cuenta: el texto puede decir «0.07 a favor de» por una diferencia de −0.07.
Un número sin fuente es un error.  Si una tabla generada aparece en la tesis, sus
números deben ser exactamente los de su procedencia; si no, es «tabla_editada».

Reglas que ignoran números estructurales (cada una probada en
``tests/test_verificar_cifras_tesis.py``; el reporte dice qué regla ignoró cada uno):

* ``comentario``: tras un ``%`` no escapado;
* ``preambulo``: antes de ``\\begin{document}`` (si el archivo lo tiene);
* ``pendiente``: dentro de ``\\pendiente{...}``; el marcador se lista aparte;
* ``ecuacion``: matemática en display (``equation``, ``align``, ``gather``, ``multline``,
  ``eqnarray``, ``\\[...\\]``, ``$$...$$``); la matemática en línea SÍ se revisa;
* ``argumento_estructural``: argumentos de ``\\label``, ``\\ref``, ``\\cite``..., rutas
  (``\\input``, ``\\includegraphics``, ``\\url``), identificadores en ``\\texttt``,
  medidas (``\\hspace``, ``\\setlength``...), especificación de columnas de tablas,
  ``\\multicolumn``/``\\multirow``/``\\cmidrule``/``\\cline`` y argumentos opcionales ``[...]``;
* ``subindice``: sub y superíndices (``x_{t-1}``, ``\\beta_2``);
* ``longitud``: un número seguido de unidad (``pt``, ``cm``, ``mm``, ``em``, ``ex``, ``bp``)
  o de ``\\textwidth``/``\\linewidth``...;
* ``fecha``: ``AAAA-MM-DD`` y fechas con nombre de mes («26 de octubre de 2026»);
* ``anio_de_cita``: un año 1500–2099 tras «et al.», «Autor,» o «Autor (»;
* ``referencia_estructural``: tras «capítulo», «sección», «tabla», «figura», «ecuación»,
  «página», «fase», «tarea», «enmienda», «hallazgo», «apéndice»... (también en plural y
  con listas «2 y 3»), y números de catálogo (``BWV``, ``Anh.``, ``Op.``, ``No.``);
* un dígito pegado a letras o a ``_`` es parte de un identificador (``H10``, ``pdmx50``,
  ``984dfde``) y no se lee como número.

Uso::

    .venv\\Scripts\\python.exe scripts/verificar_cifras_tesis.py --tesis <repo Tesis> \\
        --procedencia <salida del generador> --reporte docs/resultados-corregidos-<fecha>.md \\
        --salida verificacion.json

Sale con código 1 si hay algún error (y, con ``--pendientes-son-error``, si queda algún
``\\pendiente``).
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# --- extracción -----------------------------------------------------------------------

NUMERO = re.compile(
    r"(?<![\w.])(?<!\{,\})(?<!\\,)"
    r"(?P<n>\d{1,3}(?:(?:\\,|\{\\,\}|~)\d{3})+(?:(?:\.|\{,\})\d+)?|\d+(?:(?:\.|\{,\})\d+)?)"
    r"(?![\w])(?!\.\d)(?!\{,\}\d)"
)
DIGITOS = re.compile(r"\d+(?:\.\d+)?")
PORCIENTO = re.compile(r"\s*(?:\\,|~|\{\\,\})?\s*\\%")
SEPARADORES_MILES = re.compile(r"\\,|\{\\,\}|~")

# argumentos que se descartan por completo (todos los grupos que siguen)
SIN_ARGUMENTOS = {
    "label", "ref", "eqref", "pageref", "autoref", "cref", "Cref", "nameref", "cite", "citep", "citet", "citeauthor",
    "citeyear", "parencite", "textcite", "autocite", "nocite", "input", "include", "includegraphics", "includepdf",
    "usepackage", "documentclass", "url", "href", "hspace", "vspace", "setlength", "addtolength", "setcounter",
    "addtocounter", "texttt", "bibliography", "bibliographystyle", "addbibresource", "definecolor", "newcommand",
    "renewcommand", "providecommand", "newenvironment", "renewenvironment", "graphicspath", "cline", "hyperref",
    "rule", "raisebox", "scalebox", "verb",
}
# número de grupos {...} que se descartan; el resto del comando se lee
PRIMEROS_ARGUMENTOS = {"multicolumn": 2, "multirow": 2, "resizebox": 2, "color": 1, "textcolor": 1, "colorbox": 1,
                       "cmidrule": 1, "end": 1}
ENTORNOS_TABLA = {"tabular": 1, "tabular*": 2, "tabularx": 2, "tabulary": 2, "longtable": 1, "array": 1,
                  "longtable*": 1, "tabu": 1}
ENTORNOS_ECUACION = r"(?:equation|align|alignat|flalign|gather|multline|eqnarray|displaymath)\*?"
MESES = r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
ESPACIO = r"(?:\s|~)"
LISTA_NUMEROS = rf"\d+(?:\.\d+)*(?:{ESPACIO}*(?:,|y|e|o|a|al|--|–|-){ESPACIO}*\d+(?:\.\d+)*)*"
PALABRAS_ESTRUCTURALES = (
    r"cap[ií]tulos?|cap\.|secci[oó]n(?:es)?|secs?\.|subsecci[oó]n(?:es)?|figuras?|figs?\.|tablas?|cuadros?"
    r"|ecuaci[oó]n(?:es)?|ecs?\.|p[aá]ginas?|pp?\.|fases?|tareas?|enmiendas?|aclaraci[oó]n(?:es)?|hallazgos?"
    r"|ap[eé]ndices?|anexos?|definici[oó]n(?:es)?|teoremas?|lemas?|proposici[oó]n(?:es)?|corolarios?|ejemplos?"
    r"|algoritmos?|l[ií]neas?|versi[oó]n(?:es)?|videos?|paradas?|criterios?|bwv|anh\.|op\.|no\.|n[uú]m\.|hob\.|kv"
)
REGLAS_SPAN = (
    ("fecha", re.compile(r"(?<![\w-])\d{4}-\d{2}-\d{2}(?![\w-])")),
    ("fecha", re.compile(rf"(?<![\w.])\d{{1,2}}{ESPACIO}+de{ESPACIO}+{MESES}(?:{ESPACIO}+del?{ESPACIO}+\d{{4}})?(?!\w)", re.I)),
    ("fecha", re.compile(rf"\b{MESES}{ESPACIO}+(?:del?{ESPACIO}+)?\d{{4}}(?!\w)", re.I)),
    ("referencia_estructural", re.compile(rf"(?<!\w)(?:{PALABRAS_ESTRUCTURALES}){ESPACIO}*{LISTA_NUMEROS}(?![\w])", re.I)),
)
ANIO = r"(?:1[5-9]|20)\d{2}"
REGLAS_ANIO = (
    re.compile(rf"et{ESPACIO}*al\.?,?{ESPACIO}*\(?{ESPACIO}*(?P<y>{ANIO})(?![\w])"),
    re.compile(rf"(?<!\w)[A-ZÁÉÍÓÚÑ][\wáéíóúñü-]+,{ESPACIO}*(?P<y>{ANIO})(?![\w])"),
    re.compile(rf"(?<!\w)[A-ZÁÉÍÓÚÑ][\wáéíóúñü-]+{ESPACIO}*\({ESPACIO}*(?P<y>{ANIO})(?![\w])"),
)
LONGITUD = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?:\s?(?:pt|cm|mm|em|ex|bp)(?![A-Za-z])|\s*\\(?:textwidth|linewidth|columnwidth|textheight|paperwidth|baselineskip|hsize)(?![A-Za-z]))"
)


@dataclass(frozen=True)
class Numero:
    texto: str  # sin separador de miles, con punto decimal y sin signo
    valor: float
    decimales: int
    linea: int
    porcentaje: bool


@dataclass(frozen=True)
class Ignorado:
    texto: str
    regla: str
    linea: int


@dataclass
class Extraccion:
    numeros: list[Numero] = field(default_factory=list)
    ignorados: list[Ignorado] = field(default_factory=list)
    pendientes: list[str] = field(default_factory=list)
    pendientes_lineas: list[int] = field(default_factory=list)


def _normalizar(raw: str) -> tuple[str, float, int]:
    text = SEPARADORES_MILES.sub("", raw).replace("{,}", ".")
    decimals = len(text.split(".", 1)[1]) if "." in text else 0
    return text, float(text), decimals


def _grupo(text: str, start: int, open_char: str) -> int | None:
    """Fin (exclusivo) del grupo balanceado que abre en ``start``; None si no hay grupo."""

    close_char = {"{": "}", "[": "]", "(": ")"}[open_char]
    if start >= len(text) or text[start] != open_char:
        return None
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


class _Lienzo:
    """Texto que se va borrando con espacios (conservando saltos de línea y posiciones)."""

    def __init__(self, text: str):
        self.original = text
        self.chars = list(text)
        self.saltos = [i for i, c in enumerate(text) if c == "\n"]
        self.resultado = Extraccion()

    @property
    def texto(self) -> str:
        return "".join(self.chars)

    def linea(self, pos: int) -> int:
        return bisect.bisect_left(self.saltos, pos) + 1

    def borrar(self, start: int, end: int, regla: str | None) -> None:
        if regla is not None:
            fragment = "".join(self.chars[start:end])
            for match in DIGITOS.finditer(fragment):  # informativo: también los dígitos de identificadores
                self.resultado.ignorados.append(Ignorado(match.group(0), regla, self.linea(start + match.start())))
        for index in range(start, end):
            if self.chars[index] != "\n":
                self.chars[index] = " "


def _comando_con_argumentos(lienzo: _Lienzo) -> None:
    text = lienzo.texto
    for match in re.finditer(r"\\([A-Za-z@]+\*?)", text):
        name = match.group(1)
        base = name.rstrip("*")
        if base not in SIN_ARGUMENTOS and base not in PRIMEROS_ARGUMENTOS and base != "begin":
            continue
        if lienzo.chars[match.start()] == " ":  # ya borrado
            continue
        pos = match.end()
        wanted = None if base in SIN_ARGUMENTOS else PRIMEROS_ARGUMENTOS.get(base, 1)
        taken = 0
        while pos < len(text):
            if text[pos] == "[" or (base == "cmidrule" and text[pos] == "("):
                end = _grupo(text, pos, text[pos])
            elif text[pos] == "{" and (wanted is None or taken < wanted):
                end = _grupo(text, pos, "{")
                if base == "begin" and taken == 0 and end is not None:
                    wanted = 1 + ENTORNOS_TABLA.get(text[pos + 1 : end - 1].strip(), 0)
                taken += 1
            else:
                break
            if end is None:
                break
            pos = end
        lienzo.borrar(match.start(), pos, "argumento_estructural")


def extraer_numeros(tex: str) -> Extraccion:
    lienzo = _Lienzo(tex)
    resultado = lienzo.resultado

    for match in re.finditer(r"(?<!\\)%[^\n]*", lienzo.texto):
        lienzo.borrar(match.start(), match.end(), "comentario")
    text = lienzo.texto
    begin = re.search(r"\\begin\{document\}", text)
    if begin:
        lienzo.borrar(0, begin.end(), "preambulo")
        end = re.search(r"\\end\{document\}", text)
        if end:
            lienzo.borrar(end.start(), len(text), "preambulo")

    text = lienzo.texto
    for match in re.finditer(r"\\pendiente\s*(?=\{)", text):
        close = _grupo(text, match.end(), "{")
        if close is None:
            continue
        resultado.pendientes.append(" ".join(text[match.end() + 1 : close - 1].split()))
        resultado.pendientes_lineas.append(lienzo.linea(match.start()))
        lienzo.borrar(match.start(), close, "pendiente")

    text = lienzo.texto
    for pattern in (rf"\\begin\{{({ENTORNOS_ECUACION})\}}.*?\\end\{{\1\}}", r"(?<!\\)\\\[.*?\\\]", r"(?<!\\)\$\$.*?\$\$"):
        for match in re.finditer(pattern, text, re.S):
            lienzo.borrar(match.start(), match.end(), "ecuacion")
        text = lienzo.texto

    _comando_con_argumentos(lienzo)
    text = lienzo.texto
    for match in re.finditer(r"\\[A-Za-z@]+\*?((?:\[[^\]\n]*\])+)|\\\\\*?(\[[^\]\n]*\])", text):
        group = 1 if match.group(1) else 2
        lienzo.borrar(match.start(group), match.end(group), "argumento_estructural")
    text = lienzo.texto
    for match in LONGITUD.finditer(text):
        lienzo.borrar(match.start(), match.end(), "longitud")
    text = lienzo.texto
    for match in re.finditer(r"(?<!\\)[_^](?:\{[^{}\n]*\}|\d+)", text):
        lienzo.borrar(match.start(), match.end(), "subindice")
    text = lienzo.texto
    for match in re.finditer(r"\\[A-Za-z@]+\*?", text):
        lienzo.borrar(match.start(), match.end(), None)

    text = lienzo.texto
    for regla, pattern in REGLAS_SPAN:
        for match in pattern.finditer(text):
            lienzo.borrar(match.start(), match.end(), regla)
        text = lienzo.texto
    for pattern in REGLAS_ANIO:
        for match in pattern.finditer(text):
            lienzo.borrar(match.start("y"), match.end("y"), "anio_de_cita")
        text = lienzo.texto

    for match in NUMERO.finditer(text):
        texto, valor, decimales = _normalizar(match.group("n"))
        porcentaje = bool(PORCIENTO.match(text, match.end()))
        resultado.numeros.append(Numero(texto, valor, decimales, lienzo.linea(match.start()), porcentaje))
    resultado.numeros.sort(key=lambda n: n.linea)
    return resultado


# --- fuentes y emparejamiento ---------------------------------------------------------

NUMERO_REPORTE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])(?!\.\d)")


@dataclass
class Fuentes:
    textos: dict[str, str] = field(default_factory=dict)
    valores: list[tuple[float, str]] = field(default_factory=list)  # (|valor|, etiqueta), ordenada
    tablas: dict[str, tuple[Path, Counter]] = field(default_factory=dict)

    def agregar(self, valor: float, etiqueta: str, texto: str | None = None) -> None:
        self.valores.append((abs(valor), etiqueta))
        if texto is not None:
            self.textos.setdefault(texto.lstrip("-"), etiqueta)

    def ordenar(self) -> None:
        self.valores.sort()
        self._claves = [v for v, _ in self.valores]

    def buscar(self, numero: Numero) -> str | None:
        if numero.texto in self.textos:
            return self.textos[numero.texto]
        objetivos = [(numero.valor, 1.0)] + ([(numero.valor, 100.0)] if numero.porcentaje else [])
        tolerance = 0.5 * 10 ** (-numero.decimales) + 1e-9
        for target, factor in objetivos:
            low = bisect.bisect_left(self._claves, (target - tolerance) / factor)
            high = bisect.bisect_right(self._claves, (target + tolerance) / factor)
            for valor, etiqueta in self.valores[low:high]:
                scaled = valor * factor
                if numero.decimales == 0 and abs(scaled - round(scaled)) > 1e-9:
                    continue
                if abs(scaled - target) <= tolerance:
                    return etiqueta
        return None


def cargar_fuentes(procedencias: Iterable[Path], reportes: Iterable[Path]) -> Fuentes:
    fuentes = Fuentes()
    for path in procedencias:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        table = data.get("tabla", Path(path).name)
        for entry in data.get("cifras", []):
            label = f"procedencia:{table}:{entry.get('archivo')}:{entry.get('campo')}"
            fuentes.agregar(float(entry["valor"]), label, str(entry["texto"]))
        fuentes.tablas[table] = (Path(path), Counter(str(e["texto"]).lstrip("-") for e in data.get("cifras", [])))
    for path in reportes:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            for match in NUMERO_REPORTE.finditer(line):
                fuentes.agregar(float(match.group(0)), f"reporte:{Path(path).name}:{number}")
    fuentes.ordenar()
    return fuentes


def _expandir(paths: Sequence[Path] | None) -> list[Path]:
    out: list[Path] = []
    for path in paths or []:
        path = Path(path)
        out.extend(sorted(path.rglob("*.procedencia.json")) if path.is_dir() else [path])
    return out


def verificar(tesis: Path, *, procedencias: Sequence[Path] | None = None, reportes: Sequence[Path] | None = None) -> dict:
    tesis = Path(tesis)
    fuentes = cargar_fuentes(_expandir(procedencias), reportes or [])
    errores: list[dict] = []
    pendientes: list[dict] = []
    emparejados: list[dict] = []
    ignorados: Counter = Counter()
    avisos: list[str] = []
    archivos = sorted(tesis.rglob("*.tex"))
    vistos: set[str] = set()
    n_numeros = 0
    for path in archivos:
        relative = path.relative_to(tesis).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        result = extraer_numeros("\n".join(lines))
        ignorados.update(item.regla for item in result.ignorados)
        for text, line in zip(result.pendientes, result.pendientes_lineas):
            pendientes.append({"archivo": relative, "linea": line, "texto": text})
        for numero in result.numeros:
            n_numeros += 1
            source = fuentes.buscar(numero)
            if source is None:
                errores.append({"tipo": "sin_fuente", "archivo": relative, "linea": numero.linea, "texto": numero.texto,
                                "contexto": lines[numero.linea - 1].strip()[:200]})
            else:
                emparejados.append({"archivo": relative, "linea": numero.linea, "texto": numero.texto, "fuente": source})
        if path.name in fuentes.tablas:
            vistos.add(path.name)
            expected = fuentes.tablas[path.name][1]
            written = Counter(n.texto for n in result.numeros)
            if written != expected:
                errores.append({"tipo": "tabla_editada", "archivo": relative, "linea": None, "texto": None,
                                "sobran": sorted((written - expected).elements()),
                                "faltan": sorted((expected - written).elements())})
    for table in sorted(set(fuentes.tablas) - vistos):
        avisos.append(f"La tabla generada {table} no aparece en {tesis}.")
    return {
        "tesis": tesis.resolve().as_posix(),
        "n_archivos": len(archivos),
        "n_numeros": n_numeros,
        "n_numeros_con_fuente": len(emparejados),
        "ignorados_por_regla": dict(sorted(ignorados.items())),
        "errores": errores,
        "pendientes": pendientes,
        "avisos": avisos,
        "emparejados": emparejados,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Empareja cada número de los .tex de la tesis con su fuente.")
    parser.add_argument("--tesis", type=Path, required=True, help="Directorio con los .tex (se recorre completo).")
    parser.add_argument("--procedencia", type=Path, nargs="*", default=[], help="JSON de procedencia o directorios que los contienen.")
    parser.add_argument("--reporte", type=Path, nargs="*", default=[], help="Reportes de texto cuyos números cuentan como fuente.")
    parser.add_argument("--salida", type=Path, help="Escribe aquí el reporte JSON completo.")
    parser.add_argument("--pendientes-son-error", action="store_true", help="Falla si queda algún \\pendiente.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verificar(args.tesis, procedencias=args.procedencia, reportes=args.reporte)
    if args.salida:
        args.salida.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{report['n_archivos']} archivos, {report['n_numeros']} números, {report['n_numeros_con_fuente']} con fuente, "
          f"{len(report['errores'])} errores, {len(report['pendientes'])} pendientes.")
    for error in report["errores"]:
        print(f"ERROR {error['tipo']}: {error['archivo']}:{error['linea']}: {error.get('texto') or ''} "
              f"{error.get('contexto') or error.get('sobran', '')}")
    for item in report["pendientes"]:
        print(f"PENDIENTE {item['archivo']}:{item['linea']}: {item['texto']}")
    for aviso in report["avisos"]:
        print(f"AVISO: {aviso}")
    failed = bool(report["errores"]) or (args.pendientes_son_error and bool(report["pendientes"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

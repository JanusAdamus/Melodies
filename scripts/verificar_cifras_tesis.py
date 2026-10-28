"""Verifica que cada número de los ``.tex`` de la tesis tenga fuente (tarea 3.4).

Recorre todos los ``.tex`` de un directorio (texto y tablas), extrae cada número, con
su signo, y lo empareja con una fuente:

* los JSON de procedencia que escribe ``scripts/generar_tablas_tesis.py`` (cada cifra con
  su ``id``, su texto, su valor sin redondear y cómo recalcularla), y
* reportes de texto (p. ej. ``docs/resultados-corregidos-<fecha>.md``), de los que se toma
  cada número tal como está escrito.

Emparejamiento (un número sin fuente es un error):

* **número suelto**: tiene fuente sólo si su texto, con signo y con los mismos decimales,
  es el de una cifra de procedencia o el de un número de un reporte.  Un número seguido de
  ``\\%`` también empareja con una fuente que, por 100, da exactamente ese número (0.9 →
  90\\,\\%).  No se redondea: «0.5» no toma por fuente un 0.487 cualquiera.
* **número citado**: ``\\cifra{<id>}{<número>}`` (en la tesis, ``\\newcommand{\\cifra}[2]{#2}``)
  empareja sólo con esa cifra: su valor, redondeado a los decimales escritos (y por 100 si
  sigue ``\\%``), debe dar el número con su signo.  ``<id>`` es el ``id`` de la procedencia
  o ``reporte:<archivo>:<línea>``.  Así se escribe una cifra redondeada en la prosa.
* **tablas generadas**: una tabla se reconoce por la cabecera que escribe el generador
  («Procedencia de cada número: X.procedencia.json»), por su nombre de archivo o, en
  cualquier ``tabular``/``longtable``, porque al menos la mitad de sus números son los de
  una tabla generada.  Sus números deben ser, en orden y con signo, los de la procedencia;
  si no, es ``tabla_editada`` (editada, permutada o renombrada).
* **modo estricto** (``--exigir-cita``): todo número fuera de una tabla generada debe ir en
  ``\\cifra``; un número suelto es ``sin_cita`` aunque su texto coincida con alguna cifra.
  Sin él, una cifra inventada puede coincidir por azar con el texto exacto de otra (en el
  humo de 296 cifras, ~5–10 % de números aleatorios de 1–3 decimales); úsese en la
  verificación final.

La procedencia no se cree: cada cifra se **recalcula** desde los artefactos de la
corrida que nombra (``corrida``; con ``--corridas`` se busca una carpeta del mismo nombre
si se movió), tras comprobar el sha256 de cada archivo (``fuentes_sha256``).  Errores:
``corrida_no_encontrada``, ``fuente_modificada``, ``procedencia_no_coincide``,
``procedencia_sin_corrida``.

Reglas que ignoran números estructurales (cada una probada en
``tests/test_verificar_cifras_tesis.py``; el reporte dice qué regla ignoró cada uno):

* ``comentario``: tras un ``%`` no escapado;
* ``preambulo``: antes de ``\\begin{document}`` (si el archivo lo tiene);
* ``pendiente``: dentro de ``\\pendiente{...}``; el marcador se lista aparte;
* ``ecuacion``: matemática en display (``equation``, ``align``, ``gather``, ``multline``,
  ``eqnarray``, ``\\[...\\]``, ``$$...$$``); la matemática en línea SÍ se revisa;
* ``argumento_estructural``: argumentos de ``\\label``, ``\\ref``, ``\\cite``..., rutas
  (``\\input``, ``\\includegraphics``, ``\\url``), el ``id`` de ``\\cifra``, medidas
  (``\\hspace``, ``\\setlength``...), especificación de columnas de tablas,
  ``\\multicolumn``/``\\multirow``/``\\cmidrule``/``\\cline``, argumentos opcionales
  ``[...]`` y ``\\texttt{...}`` con letras (un identificador); ``\\texttt{9.876}`` SÍ se revisa;
* ``subindice``: sub y superíndices (``x_{t-1}``, ``\\beta_2``);
* ``longitud``: un número seguido de unidad (``pt``, ``cm``, ``mm``, ``em``, ``ex``, ``bp``)
  o de ``\\textwidth``/``\\linewidth``...;
* ``fecha``: ``AAAA-MM-DD`` y fechas con nombre de mes («26 de octubre de 2026»);
* ``anio_de_cita``: un año 1500–2099 tras «et al.», en «Autor (AAAA)» o dentro de un
  paréntesis de cita «(Autor, AAAA; Otro, AAAA)»; fuera de paréntesis «Autor, 1999» no;
* ``referencia_estructural``: un número justo tras «capítulo», «sección», «tabla»,
  «figura», «ecuación», «página», «fase», «tarea», «enmienda», «punto», «hallazgo», «apéndice»...
  y números de catálogo (``BWV``, ``Anh.``, ``Op.``, ``No.``).  Sólo tras el plural
  («las tablas 2 y 3», «secciones 4.1, 4.2 y 5.3») sigue una lista, y cada elemento debe
  tener la forma del primero: «tabla 3, 0.42» y «figura 2 y 37 obras» conservan 0.42 y 37;
* un dígito pegado a letras o a ``_`` es parte de un identificador (``H10``, ``pdmx50``,
  ``984dfde``) y no se lee como número.

Signo: un ``-`` (o ``−``, o ``$-$``) justo antes del número lo hace negativo, salvo que
siga a un dígito, ``)``, ``]`` u otro ``-`` (rango «2-3», «0.1--0.3»).

Uso::

    .venv\\Scripts\\python.exe scripts/verificar_cifras_tesis.py --tesis <repo Tesis> \\
        --procedencia <salida del generador> --reporte docs/resultados-corregidos-<fecha>.md \\
        [--corridas artifacts/outputs artifacts/Comparacion] --salida verificacion.json

Sale con código 1 si hay algún error (y, con ``--pendientes-son-error``, si queda algún
``\\pendiente``).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

MIB = 1024 * 1024
TOLERANCIA_RECALCULO = 1e-9

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
    "addtocounter", "bibliography", "bibliographystyle", "addbibresource", "definecolor", "newcommand",
    "renewcommand", "providecommand", "newenvironment", "renewenvironment", "graphicspath", "cline", "hyperref",
    "rule", "raisebox", "scalebox", "verb",
}
# número de grupos {...} que se descartan; el resto del comando se lee
PRIMEROS_ARGUMENTOS = {"multicolumn": 2, "multirow": 2, "resizebox": 2, "color": 1, "textcolor": 1, "colorbox": 1,
                       "cmidrule": 1, "end": 1}
ENTORNOS_TABLA = {"tabular": 1, "tabular*": 2, "tabularx": 2, "tabulary": 2, "longtable": 1, "array": 1,
                  "longtable*": 1, "tabu": 1}
ENTORNOS_ECUACION = r"(?:equation|align|alignat|flalign|gather|multline|eqnarray|displaymath)\*?"
BLOQUE_TABLA = re.compile(r"\\begin\{(tabular\*?|tabularx|tabulary|longtable\*?)\}.*?\\end\{\1\}", re.S)
MESES = r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
ESPACIO = r"(?:\s|~)"
ITEM = r"\d+(?:\.\d+)*"
SEPARADOR_LISTA = re.compile(rf"{ESPACIO}*(?:,|y|e|o|a|al|--|–|-){ESPACIO}*(?={ITEM}(?![\w.]))")
# (raíz, plural): la lista «2 y 3» sólo sigue al plural
PALABRAS_ESTRUCTURALES = (
    ("cap[ií]tulo", "s"), ("secci[oó]n", "es"), ("subsecci[oó]n", "es"), ("figura", "s"), ("tabla", "s"), ("cuadro", "s"),
    ("ecuaci[oó]n", "es"), ("p[aá]gina", "s"), ("fase", "s"), ("tarea", "s"), ("enmienda", "s"), ("aclaraci[oó]n", "es"),
    ("hallazgo", "s"), ("ap[eé]ndice", "s"), ("anexo", "s"), ("definici[oó]n", "es"), ("teorema", "s"), ("lema", "s"),
    ("proposici[oó]n", "es"), ("corolario", "s"), ("algoritmo", "s"), ("l[ií]nea", "s"), ("versi[oó]n", "es"),
    ("video", "s"), ("parada", "s"), ("criterio", "s"), ("punto", "s"), ("inciso", "s"),
)
ABREVIATURAS = r"cap\.|secs?\.|figs?\.|ecs?\.|pp?\.|bwv|anh\.|op\.|no\.|n[uú]m\.|hob\.|kv"
SINGULAR = "|".join(raiz for raiz, _ in PALABRAS_ESTRUCTURALES) + "|" + ABREVIATURAS
PLURAL = "|".join(raiz + plural for raiz, plural in PALABRAS_ESTRUCTURALES) + r"|secs\.|figs\.|ecs\.|pp\."
REFERENCIA_PLURAL = re.compile(rf"(?<!\w)(?:{PLURAL}){ESPACIO}*(?P<item>{ITEM})(?![\w])(?!\.\d)", re.I)
REFERENCIA_SINGULAR = re.compile(rf"(?<!\w)(?:{SINGULAR}){ESPACIO}*{ITEM}(?![\w])(?!\.\d)", re.I)
REGLAS_SPAN = (
    ("fecha", re.compile(r"(?<![\w-])\d{4}-\d{2}-\d{2}(?![\w-])")),
    ("fecha", re.compile(rf"(?<![\w.])\d{{1,2}}{ESPACIO}+de{ESPACIO}+{MESES}(?:{ESPACIO}+del?{ESPACIO}+\d{{4}})?(?!\w)", re.I)),
    ("fecha", re.compile(rf"\b{MESES}{ESPACIO}+(?:del?{ESPACIO}+)?\d{{4}}(?!\w)", re.I)),
)
ANIO = r"(?:1[5-9]|20)\d{2}"
REGLAS_ANIO = (
    re.compile(rf"et{ESPACIO}*al\.?,?{ESPACIO}*\(?{ESPACIO}*(?P<y>{ANIO})(?![\d])"),
    re.compile(rf"(?<!\w)[A-ZÁÉÍÓÚÑ][\wáéíóúñü-]+{ESPACIO}*\({ESPACIO}*(?P<y>{ANIO})[a-z]?{ESPACIO}*(?=[);,])"),
    re.compile(rf"(?<=[(;]){ESPACIO}*[A-ZÁÉÍÓÚÑ][^();\n]{{0,80}}?,{ESPACIO}*(?P<y>{ANIO})[a-z]?{ESPACIO}*(?=[);,])"),
)
LONGITUD = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?:\s?(?:pt|cm|mm|em|ex|bp)(?![A-Za-z])|\s*\\(?:textwidth|linewidth|columnwidth|textheight|paperwidth|baselineskip|hsize)(?![A-Za-z]))"
)
CABECERA_GENERADA = re.compile(r"%\s*Procedencia de cada número:\s*(\S+)\.procedencia\.json")


@dataclass(frozen=True)
class Numero:
    texto: str  # sin separador de miles, con punto decimal y sin signo
    valor: float  # sin signo
    decimales: int
    linea: int
    porcentaje: bool
    negativo: bool = False
    cita: str | None = None  # id de \cifra{id}{...}
    pos: int = 0

    @property
    def con_signo(self) -> str:
        return ("-" if self.negativo else "") + self.texto


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
        self.chars = list(text.replace("−", "-"))  # el signo menos tipográfico mide un carácter
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


def _citas(lienzo: _Lienzo) -> list[tuple[int, int, str]]:
    """``\\cifra{id}{número}``: borra el id y devuelve (inicio, fin, id) del número."""

    spans = []
    text = lienzo.texto
    for match in re.finditer(r"\\cifra\s*(?=\{)", text):
        id_end = _grupo(text, match.end(), "{")
        if id_end is None:
            continue
        value_start = id_end
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        value_end = _grupo(text, value_start, "{")
        if value_end is None:
            continue
        cita = text[match.end() + 1 : id_end - 1].strip()
        lienzo.borrar(match.start(), value_start + 1, "argumento_estructural")
        lienzo.borrar(value_end - 1, value_end, None)
        spans.append((value_start, value_end, cita))
    return spans


def _comando_con_argumentos(lienzo: _Lienzo) -> None:
    text = lienzo.texto
    for match in re.finditer(r"\\([A-Za-z@]+\*?)", text):
        name = match.group(1)
        base = name.rstrip("*")
        if lienzo.chars[match.start()] == " ":  # ya borrado
            continue
        if base == "texttt":
            end = _grupo(text, match.end(), "{")
            if end is None:
                continue
            content = text[match.end() + 1 : end - 1].replace("\\,", "")
            if re.search(r"[A-Za-z_/:@#]", content):  # un identificador
                lienzo.borrar(match.start(), end, "argumento_estructural")
            else:  # un número en máquina de escribir: se revisa
                lienzo.borrar(match.start(), match.end() + 1, None)
                lienzo.borrar(end - 1, end, None)
            continue
        if base not in SIN_ARGUMENTOS and base not in PRIMEROS_ARGUMENTOS and base != "begin":
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


def _referencias(lienzo: _Lienzo) -> None:
    text = lienzo.texto
    for match in REFERENCIA_PLURAL.finditer(text):
        forma = match.group("item").count(".")
        end = match.end()
        while True:
            sep = SEPARADOR_LISTA.match(text, end)
            if sep is None:
                break
            item = re.match(rf"{ITEM}(?![\w])(?!\.\d)", text[sep.end():])
            if item is None or item.group(0).count(".") != forma:
                break
            end = sep.end() + item.end()
        lienzo.borrar(match.start(), end, "referencia_estructural")
    text = lienzo.texto
    for match in REFERENCIA_SINGULAR.finditer(text):
        lienzo.borrar(match.start(), match.end(), "referencia_estructural")


def _negativo(text: str, start: int) -> bool:
    index = start - 1
    while index >= 0 and text[index] in " $~{}":
        index -= 1
    if index < 0 or text[index] != "-":
        return False
    before = index - 1
    while before >= 0 and text[before] in "${":
        before -= 1
    return before < 0 or not (text[before].isdigit() or text[before] in ")]-")


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

    citas = _citas(lienzo)
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
    _referencias(lienzo)
    text = lienzo.texto
    for pattern in REGLAS_ANIO:
        for match in pattern.finditer(text):
            lienzo.borrar(match.start("y"), match.end("y"), "anio_de_cita")
        text = lienzo.texto

    for match in NUMERO.finditer(text):
        texto, valor, decimales = _normalizar(match.group("n"))
        porcentaje = bool(PORCIENTO.match(text, match.end()))
        cita = next((c for start, end, c in citas if start <= match.start() < end), None)
        resultado.numeros.append(Numero(texto, valor, decimales, lienzo.linea(match.start()), porcentaje,
                                        _negativo(text, match.start()), cita, match.start()))
    resultado.numeros.sort(key=lambda n: n.pos)
    return resultado


# --- fuentes y emparejamiento ---------------------------------------------------------

NUMERO_REPORTE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])(?!\.\d)")


def formato(valor: float, decimales: int) -> str:
    texto = f"{valor:.{decimales}f}"
    if texto.lstrip("-").strip("0.") == "":
        texto = texto.lstrip("-")  # nunca «-0.000»
    return texto


def _decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


@dataclass
class Fuentes:
    textos: dict[str, str] = field(default_factory=dict)  # texto con signo → etiqueta
    por_cien: dict[Decimal, str] = field(default_factory=dict)  # fuente × 100, exacta → etiqueta
    ids: dict[str, dict] = field(default_factory=dict)
    lineas_reporte: dict[str, list[float]] = field(default_factory=dict)
    tablas: dict[str, list[str]] = field(default_factory=dict)  # tabla → textos con signo, en orden
    procedencias: list[tuple[Path, dict]] = field(default_factory=list)
    duplicados: list[str] = field(default_factory=list)

    def agregar_texto(self, texto: str, etiqueta: str) -> None:
        self.textos.setdefault(texto, etiqueta)
        number = _decimal(texto)
        if number is not None:
            self.por_cien.setdefault(number * 100, etiqueta)

    def buscar(self, numero: Numero) -> str | None:
        """Fuente de un número suelto: mismo texto con signo, o fuente × 100 exacta si sigue ``\\%``."""

        if numero.con_signo in self.textos:
            return self.textos[numero.con_signo]
        if numero.porcentaje:
            return self.por_cien.get(Decimal(numero.con_signo))
        return None

    def citar(self, numero: Numero) -> tuple[str | None, str | None]:
        """(etiqueta, tipo de error) de un número citado con ``\\cifra``."""

        factor = 100.0 if numero.porcentaje else 1.0
        if numero.cita in self.ids:
            values = [float(self.ids[numero.cita]["valor"])]
        elif numero.cita in self.lineas_reporte:
            values = self.lineas_reporte[numero.cita]
        else:
            return None, "cita_desconocida"
        if any(formato(value * factor, numero.decimales) == numero.con_signo for value in values):
            return numero.cita, None
        return None, "cita_no_coincide"


def cargar_fuentes(procedencias: Iterable[Path], reportes: Iterable[Path]) -> Fuentes:
    fuentes = Fuentes()
    for path in procedencias:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        table = data.get("tabla", Path(path).name)
        fuentes.procedencias.append((Path(path), data))
        for entry in data.get("cifras", []):
            label = f"procedencia:{entry.get('id') or table + ':' + str(entry.get('campo'))}"
            fuentes.agregar_texto(str(entry["texto"]), label)
            if entry.get("id"):
                if entry["id"] in fuentes.ids:
                    fuentes.duplicados.append(entry["id"])
                fuentes.ids[entry["id"]] = entry
        fuentes.tablas[table] = [str(e["texto"]) for e in data.get("cifras", [])]
    for path in reportes:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            label = f"reporte:{Path(path).name}:{number}"
            for match in NUMERO_REPORTE.finditer(line):
                fuentes.agregar_texto(match.group(0), label)
                fuentes.lineas_reporte.setdefault(label, []).append(float(match.group(0)))
    return fuentes


# --- recálculo de la procedencia ------------------------------------------------------


def _resolver(obj: Any, ruta: Sequence[Any]) -> list[Any]:
    """Valores en ``ruta`` (``*`` recorre los valores de un diccionario o lista)."""

    current = [obj]
    for part in ruta:
        nxt = []
        for item in current:
            if part == "*":
                nxt.extend(item.values() if isinstance(item, dict) else item)
            elif isinstance(item, list):
                nxt.append(item[int(part)])
            else:
                nxt.append(item[part])
        current = nxt
    return current


def _recalcular_cifra(entry: dict, archivo: Path, cache: dict) -> float:
    operacion = entry.get("operacion")
    if "ruta" in entry:
        if archivo not in cache:
            cache[archivo] = json.loads(archivo.read_text(encoding="utf-8"))
        ruta = entry["ruta"]
        if operacion == "clave":
            parent = _resolver(cache[archivo], ruta[:-1])
            if len(parent) != 1 or ruta[-1] not in parent[0]:
                raise KeyError(ruta[-1])
            return float(ruta[-1])
        values = _resolver(cache[archivo], ruta)
        if operacion == "media_sobre_cadenas":
            finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
            return statistics.fmean(finite)
        if len(values) != 1:
            raise ValueError(f"la ruta da {len(values)} valores")
        value = float(values[0])
        return {None: value, "negado": -value, "por_cien": 100 * value, "complemento_por_cien": 100 * (1 - value)}[operacion]
    if "columna" in entry:
        if archivo not in cache:
            with archivo.open(newline="", encoding="utf-8") as handle:
                cache[archivo] = list(csv.DictReader(handle))
        rows = [cache[archivo][line - 2] for line in entry["filas_csv"]]
        if operacion == "conteo":
            return float(len(rows))
        values = [float(row[entry["columna"]]) for row in rows]
        if operacion is None:
            if len(values) != 1:
                raise ValueError(f"{len(values)} filas para una cifra sin operación")
            return values[0]
        return {"media": statistics.fmean, "desviacion_estandar": statistics.stdev,
                "maximo_en_MiB": lambda v: max(v) / MIB}[operacion](values)
    raise ValueError("la cifra no dice cómo recalcularse (ni ruta ni columna)")


def _buscar_corrida(corrida: str, raices: Sequence[Path]) -> Path | None:
    path = Path(corrida)
    if path.is_dir():
        return path
    for raiz in raices:
        if (Path(raiz) / path.name).is_dir():
            return Path(raiz) / path.name
        for found in Path(raiz).rglob(path.name):
            if found.is_dir():
                return found
    return None


def recalcular(fuentes: Fuentes, corridas: Sequence[Path]) -> tuple[int, int, list[dict]]:
    """(cifras, cifras recalculadas sin discrepancia, errores)."""

    total = ok = 0
    errores: list[dict] = []
    for path, data in fuentes.procedencias:
        cifras = data.get("cifras", [])
        total += len(cifras)
        if not cifras:
            continue
        base = {"archivo": path.name, "linea": None}
        if not data.get("corrida"):
            errores.append({**base, "tipo": "procedencia_sin_corrida", "texto": None})
            continue
        directorio = _buscar_corrida(data["corrida"], corridas)
        if directorio is None:
            errores.append({**base, "tipo": "corrida_no_encontrada", "texto": data["corrida"]})
            continue
        hashes = data.get("fuentes_sha256") or {}
        malos = set()
        for etiqueta, sha in hashes.items():
            archivo = directorio / etiqueta.split("/", 1)[1]
            actual = hashlib.sha256(archivo.read_bytes()).hexdigest() if archivo.is_file() else None
            if actual != sha:
                malos.add(etiqueta)
                errores.append({**base, "tipo": "fuente_modificada", "texto": etiqueta,
                                "detalle": "falta el archivo" if actual is None else "su sha256 cambió"})
        cache: dict = {}
        for entry in cifras:
            if entry.get("archivo") in malos:
                continue
            try:
                if entry.get("archivo") not in hashes:
                    raise ValueError("su archivo no tiene sha256 en fuentes_sha256")
                value = _recalcular_cifra(entry, directorio / entry["archivo"].split("/", 1)[1], cache)
                problema = None
                if abs(value - float(entry["valor"])) > TOLERANCIA_RECALCULO * max(1.0, abs(value)):
                    problema = f"el artefacto da {value!r}, la procedencia dice {entry['valor']!r}"
                else:
                    texto = str(entry["texto"])
                    decimales = len(texto.split(".", 1)[1]) if "." in texto else 0
                    if formato(value, decimales) != texto:
                        problema = f"el artefacto da {formato(value, decimales)}, la procedencia escribe {texto}"
            except (KeyError, IndexError, ValueError, TypeError, statistics.StatisticsError) as exc:
                problema = f"no se puede recalcular: {exc!r}"
            if problema:
                errores.append({**base, "tipo": "procedencia_no_coincide", "texto": entry.get("texto"),
                                "id": entry.get("id"), "detalle": problema})
            else:
                ok += 1
    return total, ok, errores


# --- verificación ---------------------------------------------------------------------


def _comparar_tabla(table: str, written: list[str], expected: list[str], relative: str, como: str) -> dict | None:
    if written == expected:
        return None
    first = next((i for i, (w, e) in enumerate(zip(written, expected)) if w != e), min(len(written), len(expected)))
    return {"tipo": "tabla_editada", "archivo": relative, "linea": None, "texto": None, "tabla": table, "reconocida_por": como,
            "primera_diferencia": first, "escrito": written[first:first + 3], "esperado": expected[first:first + 3],
            "sobran": sorted((Counter(written) - Counter(expected)).elements()),
            "faltan": sorted((Counter(expected) - Counter(written)).elements())}


def _expandir(paths: Sequence[Path] | None) -> list[Path]:
    out: list[Path] = []
    for path in paths or []:
        path = Path(path)
        out.extend(sorted(path.rglob("*.procedencia.json")) if path.is_dir() else [path])
    return out


def verificar(tesis: Path, *, procedencias: Sequence[Path] | None = None, reportes: Sequence[Path] | None = None,
              corridas: Sequence[Path] | None = None, exigir_cita: bool = False) -> dict:
    tesis = Path(tesis)
    fuentes = cargar_fuentes(_expandir(procedencias), reportes or [])
    n_cifras, n_recalculadas, errores = recalcular(fuentes, corridas or [])
    errores.extend({"tipo": "id_duplicado", "archivo": None, "linea": None, "texto": d} for d in fuentes.duplicados)
    pendientes: list[dict] = []
    emparejados: list[dict] = []
    ignorados: Counter = Counter()
    avisos: list[str] = []
    archivos = sorted(tesis.rglob("*.tex"))
    vistos: set[str] = set()
    n_numeros = 0
    for path in archivos:
        relative = path.relative_to(tesis).as_posix()
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines()
        result = extraer_numeros(original)
        ignorados.update(item.regla for item in result.ignorados)
        for text, line in zip(result.pendientes, result.pendientes_lineas):
            pendientes.append({"archivo": relative, "linea": line, "texto": text})

        # tablas generadas: por cabecera o nombre (todo el archivo) o por contenido (cada tabular)
        en_tabla: list[tuple[int, int]] = []
        header = CABECERA_GENERADA.search(original)
        table = None
        if header and f"{header.group(1)}.tex" in fuentes.tablas:
            table, como = f"{header.group(1)}.tex", "cabecera"
        elif path.name in fuentes.tablas:
            table, como = path.name, "nombre"
        if table is not None:
            vistos.add(table)
            en_tabla.append((0, len(original)))
            error = _comparar_tabla(table, [n.con_signo for n in result.numeros], fuentes.tablas[table], relative, como)
            if error:
                errores.append(error)
        else:
            for block in BLOQUE_TABLA.finditer(original):
                written = [n.con_signo for n in result.numeros if block.start() <= n.pos < block.end()]
                best, overlap = None, 0
                for name, expected in fuentes.tablas.items():
                    common = sum((Counter(written) & Counter(expected)).values())
                    if len(expected) >= 3 and common >= max(3, len(expected) / 2) and common > overlap:
                        best, overlap = name, common
                if best is not None:
                    vistos.add(best)
                    en_tabla.append((block.start(), block.end()))
                    error = _comparar_tabla(best, written, fuentes.tablas[best], relative, "contenido")
                    if error:
                        errores.append(error)

        for numero in result.numeros:
            n_numeros += 1
            if numero.cita is not None:
                source, kind = fuentes.citar(numero)
            elif exigir_cita and not any(start <= numero.pos < end for start, end in en_tabla):
                source, kind = None, "sin_cita"
            else:
                source = fuentes.buscar(numero)
                kind = None if source else "sin_fuente"
            if source is None:
                errores.append({"tipo": kind, "archivo": relative, "linea": numero.linea, "texto": numero.con_signo,
                                "cita": numero.cita, "contexto": lines[numero.linea - 1].strip()[:200]})
            else:
                emparejados.append({"archivo": relative, "linea": numero.linea, "texto": numero.con_signo, "fuente": source})
    for table in sorted(set(fuentes.tablas) - vistos):
        if fuentes.tablas[table]:
            avisos.append(f"La tabla generada {table} no aparece en {tesis}.")
    return {
        "tesis": tesis.resolve().as_posix(),
        "n_archivos": len(archivos),
        "n_numeros": n_numeros,
        "n_numeros_con_fuente": len(emparejados),
        "n_cifras_procedencia": n_cifras,
        "n_cifras_recalculadas": n_recalculadas,
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
    parser.add_argument("--corridas", type=Path, nargs="*", default=[],
                        help="Dónde buscar una corrida por su nombre si ya no está en la ruta de la procedencia.")
    parser.add_argument("--salida", type=Path, help="Escribe aquí el reporte JSON completo.")
    parser.add_argument("--exigir-cita", action="store_true",
                        help="Todo número fuera de una tabla generada debe ir en \\cifra{id}{...} (error «sin_cita»).")
    parser.add_argument("--pendientes-son-error", action="store_true", help="Falla si queda algún \\pendiente.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verificar(args.tesis, procedencias=args.procedencia, reportes=args.reporte, corridas=args.corridas,
                       exigir_cita=args.exigir_cita)
    if args.salida:
        args.salida.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{report['n_archivos']} archivos, {report['n_numeros']} números, {report['n_numeros_con_fuente']} con fuente, "
          f"{report['n_cifras_recalculadas']}/{report['n_cifras_procedencia']} cifras de procedencia recalculadas, "
          f"{len(report['errores'])} errores, {len(report['pendientes'])} pendientes.")
    for error in report["errores"]:
        print(f"ERROR {error['tipo']}: {error['archivo']}:{error['linea']}: {error.get('texto') or ''} "
              f"{error.get('detalle') or error.get('contexto') or error.get('escrito', '')}")
    for item in report["pendientes"]:
        print(f"PENDIENTE {item['archivo']}:{item['linea']}: {item['texto']}")
    for aviso in report["avisos"]:
        print(f"AVISO: {aviso}")
    failed = bool(report["errores"]) or (args.pendientes_son_error and bool(report["pendientes"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

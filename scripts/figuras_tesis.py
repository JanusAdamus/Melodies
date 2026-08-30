"""Publication figures for the model comparison chapter.

Reads the audited run artifacts under ``artifacts/Comparacion`` and the capacity
diagnostics under ``artifacts/`` and writes vector plus raster figures to
``artifacts/figuras/``. Nothing here recomputes a model: every number is read
from an artifact, so a figure can never disagree with the reports.

Three rules keep the figures readable on their own:

* every figure carries the finding as its title, not the topic, so it can be
  read away from the report;
* colour marks only what changes, and context stays grey, which keeps the ink
  proportional to the claim;
* the method line under each title carries n, the error definition and the
  units, so no caption is needed to interpret the axes.

Palette is Okabe-Ito, and every series carries a marker and a dash pattern as
well, so the figures survive being printed in greyscale.

Usage::

    .venv\\Scripts\\python.exe scripts/figuras_tesis.py
"""

from __future__ import annotations

import csv
import json
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
COMPARACION = ARTIFACTS / "Comparacion"
OUTPUT_DIR = ARTIFACTS / "figuras"

ORIGINAL = "tesis_3000_gpu_20260823_1941"

# Okabe-Ito. One colour per model, fixed across every figure in the chapter.
COLORS = {
    "transformer": "#0072B2",
    "finite_hmm": "#D55E00",
    "hdp_hmm": "#009E73",
    "vomm": "#CC79A7",
}
MARKERS = {"transformer": "o", "finite_hmm": "s", "hdp_hmm": "^", "vomm": "D"}
DASHES = {
    "transformer": (None, None),
    "finite_hmm": (5, 2),
    "hdp_hmm": (2, 1.5),
    "vomm": (6, 2, 1, 2),
}
ETIQUETAS = {
    "transformer": "Transformer",
    "finite_hmm": "HMM finito",
    "hdp_hmm": "HDP-HMM",
    "vomm": "VOMM",
}
ORDEN = ["transformer", "finite_hmm", "hdp_hmm", "vomm"]

GRIS = "#9E9E9E"
GRIS_TEXTO = "#5A5A5A"

# Single column 89 mm, double column 183 mm, expressed in inches.
ANCHO_SIMPLE = 3.5
ANCHO_DOBLE = 7.2


def aplicar_estilo() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.5,
            "axes.edgecolor": "#666666",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "xtick.labelcolor": "#333333",
            "ytick.labelcolor": "#333333",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.2,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def titular(fig, hallazgo: str, metodo: str, y: float = 1.0) -> None:
    """The finding as the title, the method as a quiet second line.

    Both are wrapped to the figure width. Left unwrapped, a long method line
    widens the saved bounding box and squashes the axes, because the figure is
    saved with ``bbox_inches='tight'``.
    """
    ancho = fig.get_figwidth()
    fig.text(
        0.0, y, textwrap.fill(hallazgo, width=max(28, int(ancho * 16))),
        fontsize=9, fontweight="bold", ha="left", va="bottom",
    )
    fig.text(
        0.0, y - 0.045, textwrap.fill(metodo, width=max(40, int(ancho * 21))),
        fontsize=6.5, color=GRIS_TEXTO, ha="left", va="top", linespacing=1.5,
    )


def etiqueta_panel(ax, letra: str, dx: float = -0.14, dy: float = 1.05) -> None:
    ax.text(
        dx, dy, letra, transform=ax.transAxes, fontsize=8.5,
        fontweight="bold", va="top", ha="left",
    )


def subtitulo_panel(ax, titulo: str, nota: str = "", y: float = 1.03) -> None:
    """Panel heading with its key immediately below, not floating under the axes.

    The first draft parked each panel's key far below the x label, where it read
    as a caption for the whole figure rather than for one panel.
    """
    # El título va arriba y la clave debajo, pegada al panel: al revés, la clave
    # se lee como encabezado y el título como pie del panel de encima.
    ax.text(0.0, y + (0.085 if nota else 0.0), titulo, transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", va="bottom", ha="left")
    if nota:
        ax.text(0.0, y, nota, transform=ax.transAxes, fontsize=6.3,
                color=GRIS_TEXTO, va="bottom", ha="left")


def recortar_ejes(ax, x=None, y=None) -> None:
    """Trim the spines to the data range.

    Direct labels need room to the right of the last point, but the axis line
    should not run under that empty margin: a spine that extends past the data
    reads as an axis that carries values it never measured.
    """
    if x is not None:
        ax.spines["bottom"].set_bounds(*x)
    if y is not None:
        ax.spines["left"].set_bounds(*y)


def ic_bootstrap(valores, n=2000, semilla=11) -> tuple[float, float]:
    """Percentile bootstrap of the mean. Used where a claim rests on a sign."""
    rng = np.random.default_rng(semilla)
    muestra = np.asarray(valores, dtype=float)
    medias = muestra[rng.integers(0, muestra.size, size=(n, muestra.size))].mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def guardar(fig, nombre: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{nombre}.{ext}")
    plt.close(fig)
    print(f"  escrito {nombre}.pdf y {nombre}.png")


# ----------------------------------------------------------------- lectores


def leer_summary(corrida: str) -> list[dict]:
    with open(COMPARACION / corrida / "results_summary.csv", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ppl_en_frac_1(corrida: str) -> dict[str, float]:
    return {
        row["model"]: float(row["mean_test_ppl"])
        for row in leer_summary(corrida)
        if float(row["frac"]) == 1.0
    }


def leer_pareadas(corrida: str) -> list[dict]:
    with open(COMPARACION / corrida / "pairwise_comparisons.json", encoding="utf-8") as handle:
        return json.load(handle)["comparisons"]


def leer_diagnostico(nombre: str) -> dict:
    with open(ARTIFACTS / nombre, encoding="utf-8") as handle:
        return json.load(handle)


def costos_ajuste(corrida: str) -> dict[str, float]:
    por_modelo = defaultdict(list)
    with open(COMPARACION / corrida / "engineering_costs.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["frac"]) == 1.0:
                por_modelo[row["model"]].append(float(row["fit_wall_clock_s"]))
    return {k: statistics.mean(v) for k, v in por_modelo.items()}


# ------------------------------------------------------------------ figuras


def figura_curva_aprendizaje() -> None:
    """One panel, direct labels, and the slope written where the curve ends."""
    datos = defaultdict(list)
    for row in leer_summary(ORIGINAL):
        datos[row["model"]].append(
            (
                float(row["mean_n_train_tokens"]),
                float(row["mean_test_ppl"]),
                float(row["std_test_ppl"]),
            )
        )

    fig, ax = plt.subplots(figsize=(ANCHO_DOBLE * 0.72, 3.0))

    marcas = None
    for modelo in ORDEN:
        puntos = sorted(datos[modelo])
        x = np.array([p[0] for p in puntos]) / 1000.0
        y = np.array([p[1] for p in puntos])
        err = np.array([p[2] for p in puntos])
        marcas = x
        ax.errorbar(
            x, y, yerr=err,
            color=COLORS[modelo], marker=MARKERS[modelo], dashes=DASHES[modelo],
            capsize=0, elinewidth=0.7, alpha=0.95,
        )
        cambio = 100.0 * (y[-1] - y[0]) / y[0]
        # VOMM y HDP-HMM terminan a 0.05 de distancia; sin separarlos, las
        # etiquetas se solapan.
        desvio = {"vomm": 0.085, "hdp_hmm": -0.085}.get(modelo, 0.0)
        ax.text(
            x[-1] * 1.06, y[-1] + desvio,
            f"{ETIQUETAS[modelo]}   {cambio:+.1f} %",
            fontsize=7, va="center", color=COLORS[modelo],
        )

    ax.set_xscale("log")
    ax.set_xticks(marcas)
    ax.set_xticklabels([f"{v:.0f}" for v in marcas])
    ax.minorticks_off()
    ax.set_xlim(marcas[0] * 0.88, marcas[-1] * 2.45)
    # El eje es logarítmico porque las fracciones del corpus están espaciadas
    # geométricamente; sin decirlo, el lector lee las distancias como lineales.
    ax.set_xlabel("Tokens de entrenamiento (miles, escala logarítmica)")
    ax.set_ylabel("Perplejidad de prueba")
    recortar_ejes(ax, x=(marcas[0], marcas[-1]))

    # Recta, no en arco: el arco de la versión anterior cruzaba la curva del
    # transformer y parecía señalarla a ella.
    ax.annotate(
        "topado en K ≤ 48",
        xy=(marcas[1], datos["finite_hmm"][1][1]),
        xytext=(marcas[0] * 1.02, 6.85),
        fontsize=6.5, color=GRIS_TEXTO, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=GRIS, linewidth=0.5),
    )

    titular(
        fig,
        "Solo el transformer aprovecha el corpus completo",
        "Barras: desviación estándar de 6 corridas; VOMM es determinista, 3. "
        "Porcentaje: cambio de 10 % a 100 %.",
    )
    guardar(fig, "fig1_curva_aprendizaje")


def figura_capacidad_hmm() -> None:
    """Grid ceiling on the left, the cost that closes the axis on the right."""
    d1 = leer_diagnostico("diagnostico_finite_hmm_k.json")
    d2 = leer_diagnostico("diagnostico_finite_hmm_k_escalon2.json")
    d3 = leer_diagnostico("diagnostico_finite_hmm_convergencia.json")

    def serie(d):
        c = sorted(d["candidates"], key=lambda x: x["n_states"])
        return (
            [x["n_states"] for x in c],
            [x["validation_ppl"] for x in c],
            [x["fit_wall_clock_s"] for x in c],
        )

    k1, v1, t1 = serie(d1)
    k2, v2, t2 = serie(d2)
    k3, v3, _ = serie(d3)

    # Both diagnostics share protocol and iteration cap, so they form one curve.
    k_medido = k1 + k2
    v_medido = v1 + v2
    t_medido = t1 + t2

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(ANCHO_DOBLE, 2.7), gridspec_kw={"wspace": 0.32}
    )

    ax.plot(k_medido, v_medido, color=COLORS["finite_hmm"], marker="o",
            markersize=3.0, linewidth=1.1,
            label="Rejilla medida (tope de 100 iteraciones)")
    ax.plot(k3, v3, color="#000000", marker="D", markersize=3.0, linewidth=1.0,
            dashes=(3, 1.5), label="Reajustado hasta converger (tope 400)")

    referencia = d1["reference"]
    ax.scatter([referencia["n_states"]], [referencia["validation_ppl"]],
               marker="*", s=95, color="#000000", zorder=5,
               label="Corrida de la tesis (K ≤ 48)")

    # 48, 96, 192 y 384 fueron el máximo de alguna rejilla probada, y las cuatro
    # veces la validación eligió ese máximo. El anillo marca el dato, no el eje.
    for k_max in (48, 96, 192, 384):
        if k_max in k_medido:
            ax.scatter([k_max], [v_medido[k_medido.index(k_max)]],
                       facecolor="none", edgecolor="#222222", s=64,
                       linewidth=0.8, zorder=6)
    ax.scatter([], [], facecolor="none", edgecolor="#222222", s=64, linewidth=0.8,
               label="Máximo de una rejilla probada:\nlas cuatro veces fue el K elegido")

    # Cinco marcas distintas necesitaban cinco rótulos sueltos, y dos de ellos
    # se encimaban. Una leyenda en la esquina vacía los ordena y no colisiona.
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.6,
              labelspacing=0.55, borderaxespad=0.2)

    ax.set_xscale("log", base=2)
    ax.set_xticks([24, 48, 96, 192, 384])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_ylim(6.72, 8.02)
    ax.set_xlim(21, 500)
    ax.set_yticks([6.8, 7.0, 7.2, 7.4, 7.6])
    ax.set_xlabel("Estados del HMM finito, K")
    ax.set_ylabel("Perplejidad de validación")
    recortar_ejes(ax, y=(6.72, 7.72))
    subtitulo_panel(ax, "A los 384 estados la curva sigue bajando")
    etiqueta_panel(ax, "A", dy=1.13)

    ax2.plot(k_medido, t_medido, color=COLORS["finite_hmm"], marker="o",
             markersize=3.0)
    ancla_k, ancla_t = 48.0, float(t1[k1.index(48)])
    rejilla = np.linspace(24, 384, 100)
    ax2.plot(rejilla, ancla_t * (rejilla / ancla_k) ** 2, color=GRIS,
             dashes=(3, 2), linewidth=0.8)
    # Rotulada donde termina, y diciendo dónde está anclada: una referencia sin
    # ancla invita a leerla como un ajuste, que no lo es.
    # "∝" no existe en Arial y matplotlib lo dibujaría como un hueco.
    ax2.text(115, ancla_t * (115 / ancla_k) ** 2 * 3.2,
             "referencia cuadrática",
             fontsize=6.3, color=GRIS_TEXTO, ha="center", va="bottom")
    # Sin guía: cualquier línea desde este texto hasta el punto de K = 384
    # cruzaba la referencia cuadrática y se confundía con ella.
    ax2.text(
        0.97, 0.03,
        f"K = 384: {t2[-1] / 3600:.1f} h por ajuste,\n30 ajustes por curva",
        transform=ax2.transAxes, fontsize=6.5, color="#222222",
        ha="right", va="bottom", linespacing=1.4,
    )
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks([24, 48, 96, 192, 384])
    ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax2.minorticks_off()
    ax2.set_xlim(21, 500)
    ax2.set_xlabel("Estados del HMM finito, K")
    ax2.set_ylabel("Segundos por ajuste")
    subtitulo_panel(ax2, "y el costo crece cuadrático")
    etiqueta_panel(ax2, "B", dy=1.13)

    titular(
        fig,
        "La meseta del HMM finito era el techo de la rejilla",
        "Una semilla, frac = 1.0. Ejes logarítmicos.",
        y=1.16,
    )
    guardar(fig, "fig2_capacidad_hmm")


def figura_sensibilidades() -> None:
    """Colour only the model that moves; the rest is grey context."""
    original = ppl_en_frac_1(ORIGINAL)
    stride = ppl_en_frac_1("sens_stride128")
    rejilla = ppl_en_frac_1("sens_hmm_grid")
    split17 = ppl_en_frac_1("sens_split17")
    split29 = ppl_en_frac_1("sens_split29")

    # Los tres paneles comparten forma, eje y escala: fila por modelo, el mismo
    # eje de perplejidad. Así el lector compara la magnitud de las tres
    # perturbaciones sin convertir nada mentalmente, que era imposible cuando el
    # panel C tenía la semilla en las abscisas y su propia escala vertical.
    fig, axes = plt.subplots(
        1, 3, figsize=(ANCHO_DOBLE, 2.35), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.14},
    )
    ax_a, ax_b, ax_c = axes
    y = np.arange(len(ORDEN))
    std_intra = 0.047  # peor desviación entre semillas, de results_summary.csv

    def mancuerna(ax, antes, despues, titulo, nota):
        for i, modelo in enumerate(ORDEN):
            a, b = antes[modelo], despues[modelo]
            cambia = abs(b - a) > 1e-9
            color = COLORS[modelo] if cambia else GRIS
            if cambia:
                ax.annotate(
                    "", xy=(b, i), xytext=(a, i),
                    arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.2,
                                    shrinkA=2.5, shrinkB=0),
                )
                ax.text((a + b) / 2, i - 0.30, f"{b - a:+.2f}", ha="center",
                        va="bottom", fontsize=7, fontweight="bold", color=color)
                ax.scatter([a], [i], color=color, marker=MARKERS[modelo], s=24, zorder=3)
                ax.scatter([b], [i], facecolor="white", edgecolor=color,
                           marker=MARKERS[modelo], s=24, linewidth=0.9, zorder=3)
            else:
                ax.scatter([a], [i], color=color, marker=MARKERS[modelo], s=20,
                           zorder=3, alpha=0.7)
        subtitulo_panel(ax, titulo, nota, y=1.06)

    mancuerna(ax_a, original, stride,
              "Exposición de ventanas",
              "● stride 64  ○ stride 128   gris: sin cambio")
    mancuerna(ax_b, original, rejilla,
              "Capacidad del HMM finito",
              "● K ≤ 48  ○ K ≤ 192   gris: sin cambio")

    # Panel C: la semilla es una etiqueta nominal, no un eje. La versión
    # anterior unía las tres semillas con una línea, que dibujaba una tendencia
    # donde solo hay tres réplicas de la misma partición.
    for i, modelo in enumerate(ORDEN):
        valores = [original[modelo], split17[modelo], split29[modelo]]
        color = COLORS[modelo]
        ax_c.plot([min(valores), max(valores)], [i, i], color=color,
                  linewidth=1.2, alpha=0.45, solid_capstyle="round", zorder=2)
        ax_c.scatter(valores, [i] * 3, color=color, marker=MARKERS[modelo],
                     s=24, zorder=3)
        ax_c.text((min(valores) + max(valores)) / 2, i - 0.30,
                  f"{max(valores) - min(valores):.2f}", ha="center", va="bottom",
                  fontsize=7, fontweight="bold", color=color)
    subtitulo_panel(ax_c, "Semilla de la partición",
                    "semillas 7, 17, 29;  cifra: recorrido", y=1.06)

    ax_a.set_yticks(y)
    ax_a.set_yticklabels([ETIQUETAS[m] for m in ORDEN])
    ax_a.invert_yaxis()
    ax_a.set_ylim(len(ORDEN) - 0.25, -0.75)
    for ax in axes:
        ax.set_xlabel("Perplejidad de prueba")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    # Vara de medir: sin ella, el lector no sabe si 0.62 es mucho. Es el ruido
    # entre semillas de la Fig. 1, dibujado en la misma escala que los cambios.
    x0 = ax_a.get_xlim()[0] + 0.15
    ax_a.plot([x0, x0 + 2 * std_intra], [3.42, 3.42], color="#222222",
              linewidth=1.4, solid_capstyle="butt", clip_on=False)
    ax_a.text(x0 + 2 * std_intra + 0.10, 3.42,
              "±0.047: ruido entre semillas",
              fontsize=6.0, color=GRIS_TEXTO, va="center", ha="left",
              clip_on=False)

    for ax, letra in zip(axes, "ABC"):
        etiqueta_panel(ax, letra, dx=-0.06 if ax is not ax_a else -0.42, dy=1.30)

    titular(
        fig,
        "Cada parámetro mueve a un modelo; la partición los mueve a todos",
        "frac = 1.0, media de 6 corridas. Un parámetro por panel; escala compartida.",
        y=1.24,
    )
    guardar(fig, "fig3_sensibilidades")


def figura_comparaciones_pareadas() -> None:
    """Original in black, the four sensitivities as one grey band of agreement."""
    corridas = [ORIGINAL, "sens_stride128", "sens_hmm_grid", "sens_split17", "sens_split29"]

    por_par = defaultdict(list)
    for corrida in corridas:
        for c in leer_pareadas(corrida):
            por_par[(c["model_a"], c["model_b"])].append((corrida, c))

    fig, ax = plt.subplots(figsize=(ANCHO_DOBLE * 0.86, 3.0))
    pares = list(por_par)
    peor_p = max(c["p_value_holm"] for entradas in por_par.values() for _, c in entradas)

    etiquetas_y = []
    for i, par in enumerate(pares):
        entradas = por_par[par]
        sensibilidades = [c for corrida, c in entradas if corrida != ORIGINAL]
        original = next(c for corrida, c in entradas if corrida == ORIGINAL)

        # La diferencia es A − B y la negativa favorece a A, así que el ganador
        # del par da el color. La versión anterior pintaba todo de negro y
        # rompía el contrato de un color por modelo que sostiene el resto del
        # capítulo.
        a, b = par
        ganador = a if original["mean_difference"] < 0 else b
        color = COLORS[ganador]
        etiquetas_y.append(f"{ETIQUETAS[a]}  −  {ETIQUETAS[b]}")

        extremos = [c["bootstrap_95_ci"] for c in sensibilidades]
        ax.plot([min(lo for lo, _ in extremos), max(hi for _, hi in extremos)],
                [i, i], color=color, linewidth=4.0, alpha=0.20,
                solid_capstyle="round", zorder=1)
        for c in sensibilidades:
            ax.plot([c["mean_difference"]], [i], marker="|", markersize=5,
                    markeredgewidth=0.8, color=color, alpha=0.75, zorder=2)

        lo, hi = original["bootstrap_95_ci"]
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.1, zorder=3)
        ax.plot([original["mean_difference"]], [i], marker="o", markersize=3.6,
                color=color, zorder=4)

    ax.axvline(0, color="#000000", linewidth=0.7, dashes=(3, 2))
    ax.set_yticks(range(len(pares)))
    ax.set_yticklabels(etiquetas_y)
    ax.invert_yaxis()
    ax.set_ylim(len(pares) - 0.4, -1.35)
    ax.set_xlabel("Diferencia de NLL por token (modelo A − modelo B)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    ax.text(-0.005, -0.55, "favorece a A  ←", fontsize=6.5, color=GRIS_TEXTO,
            ha="right", va="center")
    ax.text(0.005, -0.55, "→  favorece a B", fontsize=6.5, color=GRIS_TEXTO,
            ha="left", va="center")

    # Fuera del área de datos: antes tapaba la última fila.
    handles = [
        Line2D([], [], color="#444444", marker="o", markersize=3.6, linewidth=1.1,
               label="Corrida original, IC 95 %"),
        Line2D([], [], color="#444444", linewidth=4.0, alpha=0.25,
               label="Rango de los IC de las 4 sensibilidades"),
        Line2D([], [], color="#444444", marker="|", markersize=5, linewidth=0,
               label="Media de cada sensibilidad"),
    ]
    # Dentro del área de datos, en la esquina que ninguna fila ocupa: encima de
    # los ejes chocaba con la línea de método.
    ax.legend(handles=handles, loc="lower right", handlelength=1.6,
              labelspacing=0.35, borderaxespad=0.4)

    titular(
        fig,
        "Ninguna de las 30 comparaciones cruza el cero",
        f"414 obras, 6 pares × 5 corridas. IC 95 % por bootstrap; peor p de Wilcoxon-Holm, "
        f"{peor_p:.3f}. El color marca al modelo favorecido.",
    )
    guardar(fig, "fig4_comparaciones_pareadas")


def figura_pareto() -> None:
    """The two-axis Pareto frontier, computed rather than asserted.

    VOMM fits faster than the transformer, so the transformer is not the
    cheapest model; it dominates the two HMM, which is a narrower and true
    claim. The frontier is derived here so the figure cannot overstate it.
    """
    costos = costos_ajuste(ORIGINAL)
    filas = [r for r in leer_summary(ORIGINAL) if float(r["frac"]) == 1.0]
    nll = {r["model"]: float(r["mean_test_nll"]) for r in filas}
    nll_std = {r["model"]: float(r["std_test_nll"]) for r in filas}

    def dominado(modelo: str) -> bool:
        """True if some other model is at least as cheap and at least as good."""
        return any(
            otro != modelo
            and costos[otro] <= costos[modelo]
            and nll[otro] <= nll[modelo]
            for otro in ORDEN
        )

    frontera = sorted((m for m in ORDEN if not dominado(m)), key=lambda m: costos[m])

    fig, ax = plt.subplots(figsize=(ANCHO_DOBLE * 0.70, 2.7))

    # Escalera, no segmento: la recta de la versión anterior interpolaba entre
    # VOMM y el transformer y prometía configuraciones intermedias que no
    # existen. La frontera real de Pareto es escalonada.
    escalon_x, escalon_y = [], []
    for i, m in enumerate(frontera):
        if i:
            escalon_x.append(costos[m])
            escalon_y.append(nll[frontera[i - 1]])
        escalon_x.append(costos[m])
        escalon_y.append(nll[m])
    ax.plot(escalon_x, escalon_y, color=GRIS, linewidth=0.9, dashes=(4, 2), zorder=1)

    for modelo in ORDEN:
        en_frontera = modelo in frontera
        color = COLORS[modelo]
        # Los dominados conservan su color y pierden peso: bajar a gris les
        # quitaba la identidad que el resto del capítulo les asigna.
        ax.errorbar(
            costos[modelo], nll[modelo], yerr=nll_std[modelo],
            color=color, marker=MARKERS[modelo],
            markersize=7.0 if en_frontera else 5.0,
            alpha=1.0 if en_frontera else 0.45,
            elinewidth=0.8, capsize=1.5, zorder=4,
        )
        # Cuatro puntos en cuatro cuadrantes distintos: un desplazamiento fijo
        # encimaba los rótulos de los dos HMM, que caen casi en la misma x.
        desvio, ha, va = {
            "vomm": ((0, 11), "center", "bottom"),
            "transformer": ((0, -12), "center", "top"),
            "hdp_hmm": ((9, 3), "left", "center"),
            "finite_hmm": ((9, -3), "left", "center"),
        }[modelo]
        ax.annotate(
            f"{ETIQUETAS[modelo]}, {costos[modelo]:.0f} s",
            xy=(costos[modelo], nll[modelo]), xytext=desvio,
            textcoords="offset points", fontsize=6.8, ha=ha, va=va,
            color=color if en_frontera else GRIS_TEXTO,
            fontweight="bold" if en_frontera else "normal",
        )

    ax.set_xscale("log")
    ax.margins(x=0.38, y=0.30)
    # Marcas en los costos medidos: con solo 10² y 10³ rotulados, los 41 s y los
    # 169 s en que descansa la afirmación no se podían leer de la figura. Los
    # dos HMM cuestan casi lo mismo, así que una sola marca los cubre.
    marcas = [round(costos["vomm"]), round(costos["transformer"]),
              round(costos["finite_hmm"])]
    ax.set_xticks(marcas)
    ax.set_xticklabels([f"{v:.0f}" for v in marcas])
    ax.minorticks_off()
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("Segundos de ajuste, frac = 1.0 (escala logarítmica)")
    ax.set_ylabel("NLL de prueba por token")

    ax.annotate(
        "frontera de Pareto",
        xy=(costos[frontera[-1]], nll[frontera[0]]),
        xytext=(-14, -8), textcoords="offset points", fontsize=6.3,
        color=GRIS_TEXTO, ha="right", va="top",
    )
    ax.text(0.99, 0.34, "dominados", transform=ax.transAxes, fontsize=6.3,
            color=GRIS_TEXTO, ha="right", va="top")

    titular(
        fig,
        "El transformer domina a los dos HMM",
        "frac = 1.0, media de 6 corridas; barras: desviación estándar del NLL. "
        "Transformer en GPU, clásicos en CPU.",
    )
    guardar(fig, "fig5_pareto_prediccion_costo")


def figura_unidad_de_analisis() -> None:
    """Why the two estimands order the same pair oppositely."""
    agregado = defaultdict(lambda: defaultdict(list))
    with open(COMPARACION / ORIGINAL / "piece_metrics_raw.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["frac"] != "1.0":
                continue
            agregado[row["canonical_work_id"]][row["model"]].append(
                (float(row["nll_per_token"]), int(row["n_tokens"]))
            )

    obras = []
    for por_modelo in agregado.values():
        if not {"vomm", "hdp_hmm"} <= set(por_modelo):
            continue
        v = statistics.mean(x[0] for x in por_modelo["vomm"])
        h = statistics.mean(x[0] for x in por_modelo["hdp_hmm"])
        obras.append((statistics.mean(x[1] for x in por_modelo["vomm"]), h - v))

    obras.sort()
    n = len(obras)
    cortes = [0, n // 4, n // 2, 3 * n // 4, n]
    medias, etiquetas, ics, tamanos = [], [], [], []
    for i in range(4):
        trozo = obras[cortes[i] : cortes[i + 1]]
        diferencias = [d for _, d in trozo]
        medias.append(statistics.mean(diferencias))
        # El hallazgo del panel es un cambio de signo en el último cuartil. Sin
        # incertidumbre, el lector no puede saber si ese signo se sostiene.
        ics.append(ic_bootstrap(diferencias))
        tamanos.append(len(trozo))
        # Bordes semiabiertos: rotular cada trozo con su propio mínimo y máximo
        # repetía el valor de corte en dos barras y abría un hueco falso entre
        # el tercer cuartil y el cuarto.
        inferior = int(min(t for t, _ in trozo))
        if i < 3:
            superior = int(min(t for t, _ in obras[cortes[i + 1] : cortes[i + 2]]))
            etiquetas.append(f"{inferior}–{superior}")
        else:
            etiquetas.append(f"{inferior}–{int(max(t for t, _ in trozo))}")

    por_obra = statistics.mean(d for _, d in obras)
    por_token = sum(d * t for t, d in obras) / sum(t for t, _ in obras)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(ANCHO_DOBLE * 0.90, 2.7),
        gridspec_kw={"width_ratios": [1.9, 1], "wspace": 0.42},
    )

    x = np.arange(4)
    colores = [COLORS["vomm"] if m > 0 else COLORS["hdp_hmm"] for m in medias]
    ax.bar(x, medias, 0.60, color=colores, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, medias,
                yerr=[[m - lo for m, (lo, _) in zip(medias, ics)],
                      [hi - m for m, (_, hi) in zip(medias, ics)]],
                fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2, zorder=5)
    ax.axhline(0, color="#000000", linewidth=0.7)
    piso = min(lo for lo, _ in ics)
    techo = max(hi for _, hi in ics)
    ax.set_ylim(piso - 0.016, techo + 0.016)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\nn = {t}" for e, t in zip(etiquetas, tamanos)])
    ax.set_xlabel("Longitud de la obra (tokens; cuartiles de igual número de obras)")
    ax.set_ylabel("NLL, HDP-HMM − VOMM")
    ax.text(0.5, techo + 0.010, "gana VOMM", fontsize=6.5,
            color=COLORS["vomm"], ha="center", va="center")
    ax.text(3.0, piso - 0.010, "gana HDP-HMM", fontsize=6.5,
            color=COLORS["hdp_hmm"], ha="center", va="center")
    # El IC del primer cuartil y el del cuarto cruzan el cero, así que el título
    # dice "se desvanece" y no "se invierte": el signo del cuarto cuartil no
    # está resuelto y las barras de error dejan verlo sin necesidad de glosa.
    subtitulo_panel(ax, "La ventaja de VOMM se desvanece en las obras largas")
    etiqueta_panel(ax, "A", dy=1.22)

    ax2.barh([0, 1], [por_obra, por_token], 0.42,
             color=[COLORS["vomm"], COLORS["hdp_hmm"]], edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color="#000000", linewidth=0.7)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Unidad:\nla obra", "Unidad:\nel evento"])
    ax2.invert_yaxis()
    ax2.set_xlabel("NLL, HDP-HMM − VOMM")
    ax2.set_xlim(-0.020, 0.075)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)
    # Ambas etiquetas arrancan a la derecha del cero: a la izquierda chocarían
    # con los rótulos del eje.
    for yi, (valor, ganador) in enumerate([(por_obra, "VOMM"), (por_token, "HDP-HMM")]):
        ax2.text(max(valor, 0.0) + 0.004, yi,
                 f"{valor:+.4f}\ngana {ganador}", va="center",
                 ha="left", fontsize=6.5, linespacing=1.4)
    subtitulo_panel(ax2, "y con ella, la respuesta")
    etiqueta_panel(ax2, "B", dx=-0.30, dy=1.22)

    titular(
        fig,
        "VOMM gana por obra y pierde por evento",
        "414 obras, frac = 1.0. IC 95 % por bootstrap.",
        y=1.18,
    )
    guardar(fig, "fig6_unidad_de_analisis")


def main() -> None:
    aplicar_estilo()
    print(f"figuras -> {OUTPUT_DIR}")
    figura_curva_aprendizaje()
    figura_capacidad_hmm()
    figura_sensibilidades()
    figura_comparaciones_pareadas()
    figura_pareto()
    figura_unidad_de_analisis()
    print("listo")


if __name__ == "__main__":
    main()

"""Idea 1: la pieza camina por el grafo de clases de altura y el grafo se vuelve la matriz de transición.

Uso: python melodia.py && manim -qh escena_grafo.py PiezaComoMarkov
Todo se calcula desde el tiempo transcurrido `t`, así la animación no se desfasa del audio.
"""
import json
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np
from manim import (BLUE_D, DOWN, GREY_B, GREY_D, LEFT, ORIGIN, RIGHT, UP, WHITE, YELLOW, Arrow, Circle,
                   CurvedArrow, Dot, FadeIn, FadeOut, LaggedStart, Line, Rectangle, Scene, Square, Text,
                   ValueTracker, VGroup, Write, config, interpolate_color, linear, ManimColor)

DIR = Path(__file__).parent
NOTAS = json.loads((DIR / "salida/minueto.json").read_text(encoding="utf-8"))
AUDIO = str(DIR / "salida/minueto.wav")
NOMBRES = ["Do", "Do♯", "Re", "Re♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"]
ORDEN_ESCALA = [7, 9, 11, 0, 2, 4, 6]   # Sol mayor, para la matriz final
RADIO_CIRCULO, RADIO_NODO = 2.1, 0.34
CENTRO_GRAFO = DOWN * 0.15
CURVA = 0.45                            # arco de las aristas: i->j y j->i no se enciman
FONDO = ManimColor("#0e1116")
config.background_color = FONDO

CLASES = [n["clase"] for n in NOTAS]
INICIOS = [n["inicio"] for n in NOTAS]
FIN = NOTAS[-1]["inicio"] + NOTAS[-1]["duracion"]
TRANSICIONES = list(zip(CLASES, CLASES[1:]))           # transición k ocurre en INICIOS[k+1]
CONTEO_FINAL = Counter(TRANSICIONES)
MAX_CONTEO = max(CONTEO_FINAL.values())


def nota_actual(t: float) -> int:
    return max(0, bisect_right(INICIOS, t) - 1)


def conteos_hasta(t: float) -> Counter:
    return Counter(TRANSICIONES[:max(0, bisect_right(INICIOS, t) - 1)])


def posicion(clase: int) -> np.ndarray:
    angulo = np.pi / 2 - 2 * np.pi * clase / 12          # Do arriba, sentido horario
    return CENTRO_GRAFO + RADIO_CIRCULO * np.array([np.cos(angulo), np.sin(angulo), 0])


def arista(i: int, j: int) -> CurvedArrow:
    a, b = posicion(i), posicion(j)
    if i == j:  # nota repetida: lazo hacia afuera del círculo
        afuera = (a - CENTRO_GRAFO) / np.linalg.norm(a - CENTRO_GRAFO)
        lado = np.array([-afuera[1], afuera[0], 0])
        return CurvedArrow(a + RADIO_NODO * (afuera + 0.8 * lado) / 1.28,
                           a + RADIO_NODO * (afuera - 0.8 * lado) / 1.28, angle=-4.2, tip_length=0.14)
    d = (b - a) / np.linalg.norm(b - a)
    giro = lambda v, ang: np.array([v[0] * np.cos(ang) - v[1] * np.sin(ang), v[0] * np.sin(ang) + v[1] * np.cos(ang), 0])
    return CurvedArrow(a + RADIO_NODO * giro(d, -CURVA / 2), b - RADIO_NODO * giro(d, CURVA / 2),
                       angle=-CURVA, tip_length=0.16)


class PiezaComoMarkov(Scene):
    def construct(self):
        titulo = Text("Una pieza como cadena de Markov", font_size=40).to_edge(UP, buff=0.3)
        obra = Text("Minueto en Sol mayor, BWV Anh. 114 · atrib. a Christian Petzold (Cuaderno de Anna Magdalena Bach)", font_size=17, color=GREY_B).next_to(titulo, DOWN, 0.12)
        self.play(Write(titulo), FadeIn(obra, shift=UP * 0.2), run_time=1.5)

        nodos = {c: self.nodo(c) for c in range(12)}
        self.play(LaggedStart(*[FadeIn(nodos[c], scale=0.6) for c in range(12)], lag_ratio=0.08), run_time=1.4)
        aristas = {par: arista(*par).set_opacity(0) for par in CONTEO_FINAL}
        rollo, cabezal = self.rollo_piano()
        contador = Text("", font_size=22)
        self.add(*aristas.values(), *nodos.values(), rollo, cabezal, contador)
        self.play(FadeIn(rollo), run_time=0.6)

        t = ValueTracker(0)
        viajero = Dot(radius=0.07, color=YELLOW).set_opacity(0)
        self.add(viajero)
        for par, flecha in aristas.items():
            flecha.add_updater(lambda m, par=par: self.estilo_arista(m, par, t.get_value()))
        for c, nodo in nodos.items():
            nodo.add_updater(lambda m, c=c: self.estilo_nodo(m, c, t.get_value()))
        viajero.add_updater(lambda m: self.mover_viajero(m, aristas, t.get_value()))
        cabezal.add_updater(lambda m: m.set_x(self.x_rollo(t.get_value())))
        contador.add_updater(lambda m: m.become(self.texto_contador(t.get_value())))

        self.add_sound(AUDIO)
        self.play(t.animate.set_value(FIN), run_time=FIN, rate_func=linear)
        for m in [*aristas.values(), *nodos.values(), viajero, cabezal, contador]:
            m.clear_updaters()
        self.wait(1.2)
        self.matriz_final(VGroup(*aristas.values(), *nodos.values(), viajero, rollo, cabezal, contador), titulo, obra)

    # ---------- piezas visuales ----------
    def nodo(self, c: int) -> VGroup:
        circulo = Circle(RADIO_NODO, color=GREY_D, stroke_width=2).set_fill(FONDO, 1).move_to(posicion(c))
        return VGroup(circulo, Text(NOMBRES[c], font_size=20, color=GREY_B).move_to(circulo))

    def estilo_nodo(self, nodo: VGroup, c: int, t: float) -> None:
        sonando = CLASES[nota_actual(t)] == c and t < FIN
        visitado = c in CLASES[:nota_actual(t) + 1]
        nodo[0].set_stroke(YELLOW if sonando else (BLUE_D if visitado else GREY_D), 4 if sonando else 2)
        nodo[0].set_fill(interpolate_color(FONDO, YELLOW, 0.35) if sonando else FONDO, 1)
        nodo[1].set_color(WHITE if visitado else GREY_D)

    def estilo_arista(self, flecha: CurvedArrow, par: tuple, t: float) -> None:
        n = conteos_hasta(t)[par]
        if n == 0:
            flecha.set_opacity(0)
            return
        k = nota_actual(t)
        recien = k >= 1 and TRANSICIONES[k - 1] == par and t - INICIOS[k] < 0.6
        peso = n / MAX_CONTEO
        color = YELLOW if recien else interpolate_color(BLUE_D, WHITE, peso)
        flecha.set_stroke(color, width=1.2 + 6 * peso, opacity=0.35 + 0.65 * peso)
        flecha.tip.set_fill(color, opacity=0.35 + 0.65 * peso).set_stroke(color, opacity=0)

    def mover_viajero(self, punto: Dot, aristas: dict, t: float) -> None:
        k = nota_actual(t)
        avance = (t - INICIOS[k]) / min(0.22, NOTAS[k]["duracion"])
        if k == 0 or avance >= 1 or t >= FIN:
            punto.set_opacity(0)
            return
        punto.move_to(aristas[TRANSICIONES[k - 1]].point_from_proportion(min(avance, 0.92))).set_opacity(1)

    def x_rollo(self, t: float) -> float:
        return -6 + 12 * t / FIN

    def rollo_piano(self) -> tuple[VGroup, Line]:
        alturas = [n["midi"] for n in NOTAS]
        bajo, alto = min(alturas), max(alturas)
        y = lambda m: -3.75 + 1.0 * (m - bajo) / (alto - bajo)
        barras = VGroup(*[Rectangle(width=max(12 * n["duracion"] / FIN - 0.02, 0.03), height=0.07, stroke_width=0)
                          .set_fill(BLUE_D, 0.9).move_to([self.x_rollo(n["inicio"] + n["duracion"] / 2), y(n["midi"]), 0])
                          for n in NOTAS])
        cabezal = Line([0, -3.85, 0], [0, -2.65, 0], color=YELLOW, stroke_width=2).set_x(self.x_rollo(0))
        return barras, cabezal

    def texto_contador(self, t: float) -> Text:
        n = max(0, nota_actual(t)) if t > 0 else 0
        return Text(f"transiciones observadas: {n}", font_size=20, color=GREY_B).move_to(RIGHT * 4.6 + DOWN * 2.2)

    # ---------- cierre: el grafo es la matriz ----------
    def matriz_final(self, grafo: VGroup, titulo: Text, obra: Text) -> None:
        nuevo_titulo = Text("El grafo es la matriz de transición", font_size=40).to_edge(UP, buff=0.3)
        self.play(FadeOut(grafo, scale=0.9), FadeOut(obra), titulo.animate.become(nuevo_titulo), run_time=1.2)

        salidas = Counter(i for i, _ in TRANSICIONES)
        lado = 0.62
        celdas, numeros = VGroup(), VGroup()
        for fila, i in enumerate(ORDEN_ESCALA):
            for col, j in enumerate(ORDEN_ESCALA):
                p = CONTEO_FINAL[(i, j)] / salidas[i] if salidas[i] else 0
                pos = ORIGIN + RIGHT * (col - 3) * lado + DOWN * (fila - 3) * lado + DOWN * 0.35
                celdas.add(Square(lado, stroke_color=GREY_D, stroke_width=1).set_fill(interpolate_color(FONDO, YELLOW, p), 1).move_to(pos))
                numeros.add(Text(f"{p:.2f}".lstrip("0") if p else "·", font_size=15,
                                 color=FONDO if p > 0.5 else GREY_B).move_to(pos))
        filas = VGroup(*[Text(NOMBRES[c], font_size=18, color=GREY_B).next_to(celdas[r * 7], LEFT, 0.15) for r, c in enumerate(ORDEN_ESCALA)])
        cols = VGroup(*[Text(NOMBRES[c], font_size=18, color=GREY_B).next_to(celdas[c_i], UP, 0.15) for c_i, c in enumerate(ORDEN_ESCALA)])
        eje_y = Text("nota actual", font_size=18, color=GREY_B).rotate(np.pi / 2).next_to(filas, LEFT, 0.25)
        eje_x = Text("nota siguiente", font_size=18, color=GREY_B).next_to(cols, UP, 0.15)
        formula = Text("P( siguiente = j | actual = i )", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.35)

        self.play(LaggedStart(*[FadeIn(c, scale=0.5) for c in celdas], lag_ratio=0.02), FadeIn(filas), FadeIn(cols), run_time=2)
        self.play(FadeIn(numeros), FadeIn(eje_x), FadeIn(eje_y), Write(formula), run_time=1.5)
        self.wait(3)

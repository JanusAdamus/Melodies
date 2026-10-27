"""Idea 3: Viterbi en vivo.  Acordes ocultos (I, IV, V) -> melodía -> camino óptimo -> armonía.

Uso: python melodia.py && python audio_viterbi.py && manim -qh escena_viterbi.py ViterbiEnVivo
Como en escena_grafo, todo lo que va con audio se calcula desde el tiempo `t`.
"""
import numpy as np
from manim import (BLUE_C, BLUE_D, DOWN, GREY_B, GREY_D, LEFT, RIGHT, UP, WHITE, YELLOW, Circle, Create,
                   CurvedArrow, FadeIn, FadeOut, Indicate, LaggedStart, Line, ManimColor, Rectangle, Scene,
                   SurroundingRectangle, Text, Transform, ValueTracker, VGroup, Write, config, interpolate_color,
                   linear)

from viterbi import ACORDES, ESTADOS, SEG_POR_COMPAS, TRANSICION, cargar

NOTAS, OBS, R = cargar()
T, N = R.delta.shape
FIN = T * SEG_POR_COMPAS
AUDIO_MELODIA = "salida/melodia_sola.wav"
AUDIO_ARMONIA = "salida/con_armonia.wav"
NOMBRES = ["Do", "Do♯", "Re", "Re♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"]
FONDO = ManimColor("#0e1116")
config.background_color = FONDO

X0, DX = -4.75, 0.72                      # columna t del trellis
FILAS_Y = {0: 1.15, 1: 0.15, 2: -0.85}    # I, IV, V
RADIO_TRELLIS = 0.19
ROLLO_ABAJO, ROLLO_ALTO = -3.45, 1.15
FADE_COLUMNA = 0.35                       # s para que aparezca la columna al cerrar el compás
COLOR_ESTADO = {0: ManimColor("#5fb3f0"), 1: ManimColor("#7fd18b"), 2: ManimColor("#f0a35f")}

# probabilidad relativa dentro de cada columna (lo que se ve como brillo)
PESO = np.exp(R.delta - R.delta.max(axis=1, keepdims=True))
PESO = PESO / PESO.sum(axis=1, keepdims=True)


def x_col(t: int) -> float:
    return X0 + t * DX


def x_tiempo(seg: float) -> float:
    return X0 - DX / 2 + DX * seg / SEG_POR_COMPAS


def nota_actual(seg: float) -> int:
    return max(i for i, n in enumerate(NOTAS) if n["inicio"] <= seg + 1e-9) if seg >= 0 else 0


def nombre_acorde(j: int) -> str:
    raiz = ACORDES[ESTADOS[j]][0]
    return f"{ESTADOS[j]} · {NOMBRES[raiz]}"


class ViterbiEnVivo(Scene):
    def construct(self):
        titulo = Text("Lo que no se escucha: los acordes", font_size=38).to_edge(UP, buff=0.3)
        self.play(Write(titulo), run_time=1.2)
        modelo = self.modelo_oculto()
        self.wait(5)

        filas = VGroup(*[Text(nombre_acorde(j), font_size=20, color=COLOR_ESTADO[j]).move_to([-6.2, FILAS_Y[j], 0])
                         for j in range(N)])
        nodos = VGroup(*[VGroup(*[self.nodo_trellis(t, j) for j in range(N)]) for t in range(T)])
        punteros = {(t, j): Line(self.pos(t - 1, R.psi[t, j]), self.pos(t, j), stroke_width=1.6, color=GREY_B).set_opacity(0)
                     for t in range(1, T) for j in range(N)}
        rollo, cabezal = self.rollo_piano()
        separadores = VGroup(*[Line([x_col(t) - DX / 2, ROLLO_ABAJO, 0], [x_col(t) - DX / 2, -1.45, 0],
                                    stroke_width=1, color=GREY_D).set_opacity(0.5) for t in range(1, T)])
        formula = Text("δₜ(j) = maxᵢ [ δₜ₋₁(i) · aᵢⱼ ] · bⱼ(oₜ)", font_size=24, color=GREY_B).next_to(titulo, DOWN, 0.15)

        self.play(Transform(titulo, Text("Viterbi en vivo: compás por compás", font_size=38).to_edge(UP, buff=0.3)),
                  FadeOut(modelo["flechas"]), FadeOut(modelo["emisiones"]), FadeOut(modelo["nota"]),
                  Transform(modelo["nodos"], filas), run_time=1.6)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in nodos], lag_ratio=0.05),
                  FadeIn(rollo), FadeIn(separadores), FadeIn(formula), run_time=1.6)
        self.add(*punteros.values(), cabezal)

        # ---- 1. la melodía suena y el trellis se llena ----
        reloj = ValueTracker(0)
        for t in range(T):
            for j in range(N):
                nodos[t][j].add_updater(lambda m, t=t, j=j: self.estilo_nodo(m, t, j, reloj.get_value()))
        for (t, j), linea in punteros.items():
            linea.add_updater(lambda m, t=t: m.set_opacity(0.45 * self.aparicion(t, reloj.get_value())))
        rollo.add_updater(lambda m: self.estilo_rollo(m, reloj.get_value()))
        cabezal.add_updater(lambda m: m.set_x(x_tiempo(reloj.get_value())))
        self.add_sound(AUDIO_MELODIA)
        self.play(reloj.animate.set_value(FIN + FADE_COLUMNA), run_time=FIN + FADE_COLUMNA, rate_func=linear)
        for m in [*punteros.values(), rollo, cabezal, *[n for c in nodos for n in c]]:
            m.clear_updaters()
        self.wait(0.8)

        # ---- 2. backtracking: el camino más probable ----
        aviso = Text("Backtracking: del último compás hacia atrás", font_size=24, color=YELLOW).move_to(formula)
        self.play(Transform(formula, aviso), *[l.animate.set_opacity(0.12) for l in punteros.values()], run_time=0.8)
        camino, etiquetas = self.recorrer_camino(nodos)
        self.wait(1.5)

        # ---- 3. se añade la armonía ----
        self.play(Transform(titulo, Text("Ahora, con la armonía descifrada", font_size=38).to_edge(UP, buff=0.3)),
                  FadeOut(formula), run_time=1)
        reloj.set_value(0)
        foco = SurroundingRectangle(VGroup(etiquetas[0], nodos[0]), color=YELLOW, buff=0.08, stroke_width=2)
        foco.add_updater(lambda m: m.move_to([x_col(min(T - 1, int(reloj.get_value() / SEG_POR_COMPAS))), m.get_y(), 0]))
        cabezal.set_x(x_tiempo(0)).add_updater(lambda m: m.set_x(x_tiempo(reloj.get_value())))
        rollo.add_updater(lambda m: self.estilo_rollo(m, reloj.get_value()))
        self.add(foco)
        self.add_sound(AUDIO_ARMONIA)
        self.play(reloj.animate.set_value(FIN), run_time=FIN, rate_func=linear)
        for m in (foco, cabezal, rollo):
            m.clear_updaters()
        self.play(FadeOut(foco), FadeOut(cabezal), run_time=0.6)
        self.wait(2.5)

    # ---------- parte 0: el modelo oculto ----------
    def modelo_oculto(self) -> dict:
        centros = {0: UP * 1.15, 1: LEFT * 2.6 + DOWN * 1.35, 2: RIGHT * 2.6 + DOWN * 1.35}
        nodos = VGroup()
        for j in range(N):
            c = Circle(0.55, color=COLOR_ESTADO[j], stroke_width=3).set_fill(FONDO, 1).move_to(centros[j])
            nodos.add(VGroup(c, Text(nombre_acorde(j), font_size=20, color=COLOR_ESTADO[j]).move_to(c)))
        flechas = VGroup()
        for i in range(N):
            for j in range(N):
                flechas.add(self.flecha_modelo(centros, i, j))
        emisiones = VGroup(*[Text("emite: " + " ".join(NOMBRES[c] for c in ACORDES[ESTADOS[j]]), font_size=18, color=GREY_B)
                             .next_to(nodos[j], DOWN if j else UP, 0.62 if j else 0.12).shift(RIGHT * (1.6 if j == 0 else 0))
                             for j in range(N)])
        nota = VGroup(Text("estado oculto = acorde   ·   lo que oímos = las notas que emite", font_size=22, color=GREY_B),
                      Text("Minueto en Sol mayor, BWV Anh. 114 · atrib. a C. Petzold", font_size=16, color=GREY_B)).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.25)
        self.play(LaggedStart(*[FadeIn(n, scale=0.7) for n in nodos], lag_ratio=0.2), run_time=1.2)
        self.play(LaggedStart(*[Create(f) for f in flechas], lag_ratio=0.08), run_time=2.2)
        self.play(FadeIn(emisiones), FadeIn(nota), run_time=1)
        return {"nodos": nodos, "flechas": flechas, "emisiones": emisiones, "nota": nota}

    def flecha_modelo(self, centros: dict, i: int, j: int) -> VGroup:
        a, b, p = centros[i], centros[j], TRANSICION[i, j]
        if i == j:
            afuera = a / np.linalg.norm(a)
            lado = np.array([-afuera[1], afuera[0], 0])
            arco = CurvedArrow(a + 0.55 * (afuera + 0.7 * lado) / 1.22, a + 0.55 * (afuera - 0.7 * lado) / 1.22,
                               angle=-4.2, tip_length=0.14, color=GREY_B, stroke_width=1.5 + 5 * p)
            etiqueta_pos = a + afuera * 1.35
        else:
            d = (b - a) / np.linalg.norm(b - a)
            arco = CurvedArrow(a + 0.6 * d, b - 0.6 * d, angle=-0.35, tip_length=0.16, color=GREY_B, stroke_width=1.5 + 5 * p)
            normal = np.array([d[1], -d[0], 0])
            etiqueta_pos = (a + b) / 2 + normal * 0.42
        return VGroup(arco, Text(f"{p:.1f}", font_size=18, color=WHITE).move_to(etiqueta_pos))

    # ---------- trellis ----------
    def pos(self, t: int, j: int) -> np.ndarray:
        return np.array([x_col(t), FILAS_Y[j], 0])

    def nodo_trellis(self, t: int, j: int) -> Circle:
        return Circle(RADIO_TRELLIS, color=GREY_D, stroke_width=1.5).set_fill(FONDO, 1).move_to(self.pos(t, j))

    def aparicion(self, t: int, seg: float) -> float:
        return float(np.clip((seg - (t + 1) * SEG_POR_COMPAS) / FADE_COLUMNA, 0, 1))

    def estilo_nodo(self, nodo: Circle, t: int, j: int, seg: float) -> None:
        a = self.aparicion(t, seg)
        escuchando = int(seg / SEG_POR_COMPAS) == t and seg < FIN
        nodo.set_stroke(WHITE if escuchando else interpolate_color(GREY_D, COLOR_ESTADO[j], a), 1.5 + a)
        nodo.set_fill(interpolate_color(FONDO, COLOR_ESTADO[j], a * (0.15 + 0.85 * PESO[t, j])), 1)

    def recorrer_camino(self, nodos: VGroup) -> tuple[list, VGroup]:
        camino = R.camino
        etiquetas = VGroup(*[Text(ESTADOS[s], font_size=22, color=COLOR_ESTADO[s]).move_to([x_col(t), 2.05, 0])
                             for t, s in enumerate(camino)])
        ultimo = nodos[T - 1][camino[-1]]
        self.play(Indicate(ultimo, color=YELLOW), FadeIn(etiquetas[-1]), run_time=0.6)
        self.play(ultimo.animate.set_fill(YELLOW, 1).set_stroke(YELLOW, 3), run_time=0.3)  # después: Indicate restaura el original
        for t in range(T - 1, 0, -1):
            tramo = Line(self.pos(t, camino[t]), self.pos(t - 1, camino[t - 1]), color=YELLOW, stroke_width=4)
            self.add_foreground_mobjects(nodos[t][camino[t]])
            self.play(Create(tramo), nodos[t - 1][camino[t - 1]].animate.set_fill(YELLOW, 1).set_stroke(YELLOW, 3),
                      FadeIn(etiquetas[t - 1], shift=DOWN * 0.15), run_time=0.28)
        self.add_foreground_mobjects(nodos[0][camino[0]])
        return camino, etiquetas

    # ---------- rollo de piano ----------
    def rollo_piano(self) -> tuple[VGroup, Line]:
        alturas = [n["midi"] for n in NOTAS]
        bajo, alto = min(alturas), max(alturas)
        y = lambda m: ROLLO_ABAJO + 0.15 + ROLLO_ALTO * (m - bajo) / (alto - bajo)
        barras = VGroup(*[Rectangle(width=max(DX * n["duracion"] / SEG_POR_COMPAS - 0.02, 0.03), height=0.08, stroke_width=0)
                          .set_fill(BLUE_D, 0.9).move_to([x_tiempo(n["inicio"] + n["duracion"] / 2), y(n["midi"]), 0])
                          for n in NOTAS])
        cabezal = Line([0, ROLLO_ABAJO, 0], [0, -1.45, 0], color=YELLOW, stroke_width=2).set_x(x_tiempo(0))
        return barras, cabezal

    def estilo_rollo(self, barras: VGroup, seg: float) -> None:
        k = nota_actual(seg) if seg < FIN else -1
        for i, b in enumerate(barras):
            b.set_fill(YELLOW if i == k else (BLUE_C if NOTAS[i]["inicio"] <= seg else BLUE_D),
                       1 if i == k else (0.9 if NOTAS[i]["inicio"] <= seg else 0.35))

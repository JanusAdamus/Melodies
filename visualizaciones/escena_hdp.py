"""Video 3: HDP-HMM.  El palo que se rompe -> Gibbs en cámara rápida -> 6 de 20 estados -> cada estado, un instrumento.

Uso: python traza_hdp.py && python audio_hdp.py && manim -qh escena_hdp.py HDPHMM
Los datos salen de la traza real del modelo de la tesis (salida/traza_hdp.json).
"""
import json
from pathlib import Path

import numpy as np
from manim import (DOWN, GREY_B, GREY_D, LEFT, RIGHT, UP, WHITE, YELLOW, Circle, CurvedArrow, FadeIn, FadeOut,
                   LaggedStart, Line, ManimColor, Rectangle, Scene, Text, Transform, ValueTracker, VGroup, VMobject,
                   Write, config, interpolate_color, linear)

from audio_hdp import INSTRUMENTOS, estados_por_uso

DIR = Path(__file__).parent
TRAZA = json.loads((DIR / "salida/traza_hdp.json").read_text(encoding="utf-8"))
NOTAS = json.loads((DIR / "salida/minueto.json").read_text(encoding="utf-8"))
AUDIO = "salida/orquestado.wav"
NOMBRES = ["Do", "Do♯", "Re", "Re♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"]
FONDO = ManimColor("#0e1116")
config.background_color = FONDO

K = TRAZA["params"]["n_states"]
ITERS = TRAZA["params"]["n_iters"]
FIN = NOTAS[-1]["inicio"] + NOTAS[-1]["duracion"]
ORDEN = estados_por_uso(TRAZA)                        # estados finales, más usado primero
LETRAS = "ABCDEF"
PALETA = [ManimColor(c) for c in ("#58C4DD", "#FC6255", "#83C167", "#C59BF0", "#F0A35F", "#F2E9C9")]
COLOR = {k: PALETA[i] for i, k in enumerate(ORDEN)}
BETA_FINAL = np.array(TRAZA["betas"][TRAZA["mejor_iteracion"] - 1])
ANCHO_PALO, X_PALO = 12.0, -6.0
SEG_POR_ITER = 0.2


def color_estado(k: int) -> ManimColor:
    return COLOR.get(k, interpolate_color(GREY_D, GREY_B, (k % 7) / 7))


def nombre_estado(k: int) -> str:
    emision = TRAZA["emision_final"][TRAZA["activos_finales"].index(k)]
    top = [c for c in sorted(range(12), key=lambda c: -emision[c])[:2] if emision[c] > 0.2]
    return f"{LETRAS[ORDEN.index(k)]} · {'/'.join(NOMBRES[c] for c in top)}"


def palo(beta, y: float, alto: float = 0.42, apagar_inactivos: bool = False) -> VGroup:
    piezas, x = VGroup(), X_PALO
    for k, b in enumerate(beta):
        w = ANCHO_PALO * b
        activo = k in COLOR
        opacidad = 0.12 if apagar_inactivos and not activo else 0.95
        piezas.add(Rectangle(width=max(w, 0.001), height=alto, stroke_width=0.6, stroke_color=FONDO)
                   .set_fill(color_estado(k), opacidad).move_to([x + w / 2, y, 0]))
        x += w
    return piezas


def rollo(estados, y0: float = -0.55, alto: float = 1.7) -> VGroup:
    bajo, alto_midi = min(n["midi"] for n in NOTAS), max(n["midi"] for n in NOTAS)
    return VGroup(*[Rectangle(width=max(ANCHO_PALO * n["duracion"] / FIN - 0.02, 0.04), height=0.13, stroke_width=0)
                    .set_fill(color_estado(z), 0.95)
                    .move_to([X_PALO + ANCHO_PALO * (n["inicio"] + n["duracion"] / 2) / FIN,
                              y0 + alto * (n["midi"] - bajo) / (alto_midi - bajo), 0])
                    for n, z in zip(NOTAS, estados)])


class HDPHMM(Scene):
    def construct(self):
        self.pregunta()
        self.romper_palo()
        self.gibbs()
        self.orquesta()
        self.trilogia()

    # ---------- 1. la pregunta ----------
    def pregunta(self) -> None:
        titulo = Text("Video 2: tres acordes ocultos, fijos de antemano", font_size=32, color=GREY_B).to_edge(UP, buff=0.4)
        tres = VGroup(*[Circle(0.4, color=c, stroke_width=3) for c in ("#5fb3f0", "#7fd18b", "#f0a35f")]).arrange(RIGHT, buff=0.8)
        self.play(FadeIn(titulo), LaggedStart(*[FadeIn(c, scale=0.6) for c in tres], lag_ratio=0.15), run_time=1.3)
        pregunta = Text("¿Y si no sabemos cuántos estados hay?", font_size=42).move_to(DOWN * 1.6)
        muchos = VGroup(*[Circle(0.16, color=GREY_B, stroke_width=1.5) for _ in range(K)]).arrange(RIGHT, buff=0.22)
        self.play(Write(pregunta), run_time=1.3)
        self.play(Transform(tres, muchos), run_time=1.2)
        self.wait(1)
        self.play(FadeOut(titulo), FadeOut(tres), FadeOut(pregunta), run_time=0.7)

    # ---------- 2. el palo que se rompe ----------
    def romper_palo(self) -> None:
        titulo = Text("El palo que se rompe", font_size=40).to_edge(UP, buff=0.35)
        formula = Text("βₖ = vₖ · ∏ⱼ₍ₖ (1 − vⱼ),    vₖ ~ Beta(1, γ)", font_size=26, color=GREY_B).next_to(titulo, DOWN, 0.2)
        beta0 = np.array(TRAZA["betas"][0])
        entero = Rectangle(width=ANCHO_PALO, height=0.42, stroke_width=0).set_fill(YELLOW, 0.9).move_to([0, 0.6, 0])
        self.play(Write(titulo), FadeIn(formula), FadeIn(entero), run_time=1.5)
        piezas = palo(beta0, 0.6)
        x = X_PALO
        for k in range(5):
            w = ANCHO_PALO * beta0[k]
            resto = Rectangle(width=max(ANCHO_PALO - (x + w - X_PALO), 0.01), height=0.42, stroke_width=0).set_fill(YELLOW, 0.9)
            resto.move_to([(x + w + X_PALO + ANCHO_PALO) / 2, 0.6, 0])
            etiqueta = Text(f"β{'₁₂₃₄₅'[k]}", font_size=22, color=WHITE).next_to(piezas[k], UP if k % 2 == 0 else DOWN, 0.12)
            self.play(FadeIn(piezas[k]), Transform(entero, resto), FadeIn(etiqueta), run_time=0.75)
            x += w
        self.play(LaggedStart(*[FadeIn(p) for p in piezas[5:]], lag_ratio=0.05), FadeOut(entero), run_time=1.4)
        nota = VGroup(Text(f"γ = 2  ·  truncamos en K = {K} pedazos", font_size=24, color=GREY_B),
                      Text("cada pedazo es un estado posible; su largo, qué tan probable es usarlo", font_size=22, color=GREY_B)
                      ).arrange(DOWN, buff=0.15).move_to(DOWN * 1.2)
        self.play(FadeIn(nota), run_time=1)
        self.wait(2.5)
        self.palo_actual = piezas
        self.play(FadeOut(titulo), FadeOut(formula), FadeOut(nota), *[FadeOut(m) for m in self.mobjects if isinstance(m, Text)], run_time=0.8)

    # ---------- 3. Gibbs en cámara rápida ----------
    def gibbs(self) -> None:
        titulo = Text("Muestreo de Gibbs: el modelo prueba y corrige", font_size=36).to_edge(UP, buff=0.3)
        it = ValueTracker(0)
        i_actual = lambda: min(ITERS - 1, int(it.get_value()))
        barra = palo(TRAZA["betas"][0], 2.35, alto=0.36)
        notas = rollo(TRAZA["estados"][0])
        etiqueta_palo = Text("pesos β del palo (fijos en esta corrida: lo que se mueve son los estados)", font_size=18, color=GREY_B).next_to(barra, DOWN, 0.1).align_to(barra, LEFT)
        etiqueta_rollo = Text("la melodía, coloreada por su estado oculto", font_size=18, color=GREY_B).move_to([-3.6, -0.95, 0])
        contador = Text("", font_size=24)
        curva_ejes = VGroup(Line([-6, -3.5, 0], [-0.6, -3.5, 0], color=GREY_D), Line([-6, -3.5, 0], [-6, -1.6, 0], color=GREY_D))
        curva_titulo = Text("log-verosimilitud", font_size=18, color=GREY_B).next_to(curva_ejes, UP, 0.05).align_to(curva_ejes, LEFT)
        ll = np.array(TRAZA["log_verosimilitud"])
        punto = lambda i: [-6 + 5.4 * i / (ITERS - 1), -3.45 + 1.8 * (ll[i] - ll.min()) / (ll.max() - ll.min()), 0]
        curva = VMobject(color=YELLOW, stroke_width=2.5)

        self.play(Write(titulo), Transform(self.palo_actual, barra), FadeIn(notas), FadeIn(etiqueta_palo), FadeIn(etiqueta_rollo),
                  FadeIn(curva_ejes), FadeIn(curva_titulo), run_time=1.5)
        self.remove(self.palo_actual)
        self.add(barra, curva, contador)
        notas.add_updater(lambda m: m.become(rollo(TRAZA["estados"][i_actual()])))
        curva.add_updater(lambda m: m.set_points_as_corners([punto(i) for i in range(i_actual() + 1)] * (1 if i_actual() else 2)))
        contador.add_updater(lambda m: m.become(VGroup(
            Text(f"iteración {i_actual() + 1} / {ITERS}", font_size=24),
            Text(f"estados en uso: {TRAZA['activos_por_iteracion'][i_actual()]}", font_size=24, color=YELLOW),
            Text(f"burn-in: {TRAZA['params']['burn_in']} iteraciones", font_size=18, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.4, -2.5, 0])))
        self.play(it.animate.set_value(ITERS - 0.01), run_time=ITERS * SEG_POR_ITER, rate_func=linear)
        for m in (barra, notas, curva, contador):
            m.clear_updaters()

        # resultado: 6 de 20
        final = palo(BETA_FINAL, 2.35, alto=0.36, apagar_inactivos=True)
        resultado = Text(f"{len(ORDEN)} estados de {K} posibles: el modelo eligió cuántos", font_size=30, color=YELLOW).move_to([0, -2.5, 0])
        self.play(Transform(barra, final), Transform(notas, rollo(TRAZA["estados_finales"])),
                  FadeOut(contador), FadeOut(curva), FadeOut(curva_ejes), FadeOut(curva_titulo), run_time=1.2)
        self.play(Write(resultado), run_time=1.2)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in (titulo, barra, etiqueta_palo, etiqueta_rollo, resultado)], run_time=0.8)
        self.rollo_final = notas

    # ---------- 4. cada estado, un instrumento ----------
    def orquesta(self) -> None:
        titulo = Text("Escúchalo: cada estado oculto, un instrumento", font_size=36).to_edge(UP, buff=0.3)
        destino = rollo(TRAZA["estados_finales"], y0=1.0, alto=1.3)
        centro, radio = np.array([-3.3, -1.75, 0]), 1.45
        pos = {k: centro + radio * np.array([np.cos(np.pi / 2 - 2 * np.pi * i / len(ORDEN)), np.sin(np.pi / 2 - 2 * np.pi * i / len(ORDEN)), 0])
               for i, k in enumerate(ORDEN)}
        nodos = {k: VGroup(Circle(0.3, color=COLOR[k], stroke_width=3).set_fill(FONDO, 1).move_to(pos[k]),
                           Text(LETRAS[ORDEN.index(k)], font_size=24, color=COLOR[k]).move_to(pos[k])) for k in ORDEN}
        flechas = VGroup()
        trans = np.array(TRAZA["transicion_final"])
        activos = TRAZA["activos_finales"]
        for a in ORDEN:
            for b in ORDEN:
                p = trans[activos.index(a), activos.index(b)]
                if a != b and p > 0.15:
                    d = (pos[b] - pos[a]) / np.linalg.norm(pos[b] - pos[a])
                    flechas.add(CurvedArrow(pos[a] + 0.34 * d, pos[b] - 0.34 * d, angle=-0.4, tip_length=0.13,
                                            color=COLOR[a], stroke_width=1 + 4 * p).set_opacity(0.55))
        leyenda = VGroup(*[VGroup(Rectangle(width=0.3, height=0.22, stroke_width=0).set_fill(COLOR[k], 1),
                                  Text(nombre_estado(k), font_size=20, color=COLOR[k]),
                                  Text(f"→ {INSTRUMENTOS[i][0]}", font_size=20, color=GREY_B)).arrange(RIGHT, buff=0.18)
                           for i, k in enumerate(ORDEN)]).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.2, -1.75, 0])
        cabezal = Line([X_PALO, 0.85, 0], [X_PALO, 2.45, 0], color=WHITE, stroke_width=2)

        self.play(Write(titulo), Transform(self.rollo_final, destino), run_time=1.3)
        self.play(LaggedStart(*[FadeIn(n, scale=0.6) for n in nodos.values()], lag_ratio=0.1), FadeIn(flechas), FadeIn(leyenda), run_time=1.5)
        self.add(cabezal)

        t = ValueTracker(0)
        inicios = [n["inicio"] for n in NOTAS]
        actual = lambda: max(0, int(np.searchsorted(inicios, t.get_value(), side="right")) - 1)
        z_actual = lambda: TRAZA["estados_finales"][actual()] if t.get_value() < FIN else None
        cabezal.add_updater(lambda m: m.set_x(X_PALO + ANCHO_PALO * t.get_value() / FIN))
        self.rollo_final.add_updater(lambda m: [b.set_fill(opacity=1 if i == actual() and t.get_value() < FIN else 0.45)
                                                .set_stroke(WHITE, 1.5 if i == actual() and t.get_value() < FIN else 0)
                                                for i, b in enumerate(m)])
        for k, nodo in nodos.items():
            nodo.add_updater(lambda m, k=k: m[0].set_fill(interpolate_color(FONDO, COLOR[k], 0.85 if z_actual() == k else 0), 1)
                             .set_stroke(width=6 if z_actual() == k else 3))
            nodo[1].add_updater(lambda m, k=k: m.set_color(FONDO if z_actual() == k else COLOR[k]))
        for i, fila in enumerate(leyenda):
            fila.add_updater(lambda m, k=ORDEN[i]: m.set_opacity(1 if z_actual() in (k, None) else 0.35))
        self.add_sound(AUDIO)
        self.play(t.animate.set_value(FIN), run_time=FIN, rate_func=linear)
        for m in [cabezal, self.rollo_final, *nodos.values(), *[n[1] for n in nodos.values()], *leyenda]:
            m.clear_updaters()
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)

    # ---------- 5. la trilogía ----------
    def trilogia(self) -> None:
        filas = VGroup(*[VGroup(Text(n, font_size=40, color=c), Text(t, font_size=26, color=GREY_B)).arrange(RIGHT, buff=0.4)
                         for n, t, c in (("1", "Markov: lo que se oye tiene memoria", "#58C4DD"),
                                         ("2", "HMM: hay algo oculto, y Viterbi lo encuentra", "#83C167"),
                                         ("3", "HDP-HMM: el modelo decide cuántos estados", YELLOW))]
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        pie = Text(f"HDP-HMM truncado de la tesis · K = {K}, {ITERS} iteraciones, burn-in {TRAZA['params']['burn_in']}, "
                   f"semilla {TRAZA['params']['seed']} · melodía, compases 1–16 · BWV Anh. 114 (atrib. C. Petzold)",
                   font_size=14, color=GREY_B).to_edge(DOWN, buff=0.3)
        self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.3) for f in filas], lag_ratio=0.4), FadeIn(pie), run_time=2.4)
        self.wait(3.5)

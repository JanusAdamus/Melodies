# Visualizaciones de la tesis (Manim)

Tres videos con el Minueto en Sol mayor, BWV Anh. 114 (atrib. C. Petzold), compases 1–16:
1. `escena_grafo.py`: cadena de Markov (grafo → matriz de transición).
2. `escena_viterbi.py`: HMM ilustrativo de acordes I/IV/V y Viterbi en vivo. Parámetros puestos a mano, NO los de la tesis.
3. `escena_hdp.py`: HDP-HMM con el código de la tesis (`traza_hdp.py`). **Hay que rehacerlo después de corregir β** (ver `docs/handoff-2026-10/`).

Requisitos: `pip install manim music21 numpy pyfluidsynth pandas scipy`, las librerías de sistema de cairo y pango, fluidsynth y un soundfont General MIDI (variable de entorno `MELODIES_SOUNDFONT`).

Orden, desde esta carpeta: `python melodia.py`, `python audio_viterbi.py`, `python traza_hdp.py`, `python audio_hdp.py`; luego `manim -qh escena_grafo.py PiezaComoMarkov`, `manim -qh escena_viterbi.py ViterbiEnVivo`, `manim -qh escena_hdp.py HDPHMM`. Las salidas van a `salida/` y `media/`, que no se versionan.

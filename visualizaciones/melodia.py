"""Lee una obra del corpus, extrae la melodía y genera el audio sincronizado.

Uso: python melodia.py   (escribe salida/minueto.json y salida/minueto.wav)
"""
import json
import os
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import fluidsynth
import music21 as m21
import numpy as np

OBRA = Path(__file__).resolve().parents[1] / ("external/corpora/"
            "generic_combined/library_scores/Bach_Minuet_in_G_Major_BWV_Anh._114.mxl")
SOUNDFONT = os.environ.get("MELODIES_SOUNDFONT", "/usr/share/sounds/sf2/TimGM6mb.sf2")  # en Windows: ruta a un .sf2 General MIDI
SALIDA = Path(__file__).parent / "salida"
TEMPO_NEGRA = 100          # más lento que el original (126) para que se siga la animación
COMPASES = 16              # primera sección del minueto
MUESTREO = 44100
COLA_SEGUNDOS = 2.0        # que la última nota resuene
VEL_MELODIA, VEL_BAJO = 100, 55
VOLUMEN_PICO = 0.9         # normaliza a -1 dB; fluidsynth sale muy bajo por defecto


@dataclass(frozen=True)
class Nota:
    midi: int
    clase: int        # 0 = Do ... 11 = Si (sin octava)
    inicio: float     # segundos
    duracion: float   # segundos


def _notas(parte: m21.stream.Part, limite_negras: float) -> list[Nota]:
    seg = 60 / TEMPO_NEGRA
    notas = []
    for n in parte.flatten().notes:
        if n.offset >= limite_negras:
            break
        p = max(n.pitches, key=lambda x: x.midi)  # en acordes, la voz superior
        notas.append(Nota(p.midi, p.midi % 12, float(n.offset) * seg, float(n.quarterLength) * seg))
    return notas


def extraer() -> tuple[list[Nota], list[Nota]]:
    obra = m21.converter.parse(OBRA)
    negras_por_compas = obra.recurse().getElementsByClass(m21.meter.TimeSignature)[0].barDuration.quarterLength
    limite = COMPASES * negras_por_compas
    return _notas(obra.parts[0], limite), _notas(obra.parts[1], limite)


def sintetizar(*capas: tuple) -> np.ndarray:
    """Cada capa es (notas, velocidad) o (notas, velocidad, programa GM); cada capa va en su propio canal."""
    canales = [c for c in range(16) if c != 9][:len(capas)]  # el canal 9 es de percusión
    eventos = sorted([(n.inicio, 1, n.midi, capa[1], ch) for capa, ch in zip(capas, canales) for n in capa[0]] +
                     [(n.inicio + n.duracion, 0, n.midi, 0, ch) for capa, ch in zip(capas, canales) for n in capa[0]])
    synth = fluidsynth.Synth(samplerate=MUESTREO)
    sfid = synth.sfload(SOUNDFONT)
    for capa, ch in zip(capas, canales):
        synth.program_select(ch, sfid, 0, capa[2] if len(capa) > 2 else 0)  # 0 = piano acústico
    bloques, t_actual = [], 0.0
    for t, es_inicio, midi, vel, ch in eventos:  # apagar va antes que prender en el mismo instante
        muestras = int(round((t - t_actual) * MUESTREO))
        if muestras > 0:
            bloques.append(synth.get_samples(muestras))
        t_actual += muestras / MUESTREO
        synth.noteon(ch, midi, vel) if es_inicio else synth.noteoff(ch, midi)
    bloques.append(synth.get_samples(int(COLA_SEGUNDOS * MUESTREO)))
    synth.delete()
    audio = np.concatenate(bloques).astype(np.float64)  # estéreo intercalado
    return (audio * (VOLUMEN_PICO * 32767 / np.abs(audio).max())).astype(np.int16)


def guardar_wav(muestras: np.ndarray, ruta: Path) -> None:
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(MUESTREO)
        w.writeframes(muestras.tobytes())


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    melodia, bajo = extraer()
    assert melodia and all(b.inicio >= a.inicio for a, b in zip(melodia, melodia[1:])), "melodía vacía o desordenada"
    (SALIDA / "minueto.json").write_text(json.dumps([asdict(n) for n in melodia]), encoding="utf-8")
    guardar_wav(sintetizar((melodia, VEL_MELODIA), (bajo, VEL_BAJO)), SALIDA / "minueto.wav")
    print(f"Listo: {len(melodia)} notas de melodía, {melodia[-1].inicio + melodia[-1].duracion:.1f} s")


if __name__ == "__main__":
    main()

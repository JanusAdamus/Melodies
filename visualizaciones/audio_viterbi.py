"""Audios del video de Viterbi: la melodía sola y la melodía con los acordes descifrados.

Uso: python audio_viterbi.py   (escribe salida/melodia_sola.wav y salida/con_armonia.wav)
"""
from melodia import SALIDA, VEL_MELODIA, Nota, guardar_wav, sintetizar
from viterbi import ESTADOS, SEG_POR_COMPAS, cargar

# Voces cercanas para que los acordes no salten: I = Sol-Si-Re, IV = Sol-Do-Mi, V = Fa♯-La-Re, más el bajo en la raíz
VOICINGS = {"I": (43, 55, 59, 62), "IV": (48, 55, 60, 64), "V": (50, 54, 57, 62)}
VEL_ACORDES = 58


def acordes(camino: list[int]) -> list[Nota]:
    return [Nota(m, m % 12, t * SEG_POR_COMPAS, SEG_POR_COMPAS * 0.98)
            for t, s in enumerate(camino) for m in VOICINGS[ESTADOS[s]]]


def main() -> None:
    notas, _, r = cargar()
    melodia = [Nota(**n) for n in notas]
    guardar_wav(sintetizar((melodia, VEL_MELODIA)), SALIDA / "melodia_sola.wav")
    guardar_wav(sintetizar((melodia, VEL_MELODIA), (acordes(r.camino), VEL_ACORDES)), SALIDA / "con_armonia.wav")
    print("Listo: melodia_sola.wav y con_armonia.wav")


if __name__ == "__main__":
    main()

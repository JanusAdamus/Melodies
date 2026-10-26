"""Audio del video 3: la melodía orquestada, cada estado oculto del HDP-HMM con su instrumento.

Uso: python traza_hdp.py && python audio_hdp.py   (escribe salida/orquestado.wav)
"""
import json
from pathlib import Path

from melodia import SALIDA, Nota, guardar_wav, sintetizar

DIR = Path(__file__).parent
# programa General MIDI por orden de uso del estado (más usado primero)
INSTRUMENTOS = [("Flauta", 73), ("Vibráfono", 11), ("Piano", 0), ("Clarinete", 71), ("Pizzicato", 45), ("Celesta", 8)]
VELOCIDAD = 100


def estados_por_uso(traza: dict) -> list[int]:
    z = traza["estados_finales"]
    return sorted(traza["activos_finales"], key=lambda k: -z.count(k))


def main() -> None:
    traza = json.loads((DIR / "salida/traza_hdp.json").read_text(encoding="utf-8"))
    notas = [Nota(**n) for n in json.loads((DIR / "salida/minueto.json").read_text(encoding="utf-8"))]
    orden = estados_por_uso(traza)
    assert len(orden) <= len(INSTRUMENTOS), "hay más estados que instrumentos"
    capas = [([n for n, z in zip(notas, traza["estados_finales"]) if z == k], VELOCIDAD, INSTRUMENTOS[i][1])
             for i, k in enumerate(orden)]
    guardar_wav(sintetizar(*capas), SALIDA / "orquestado.wav")
    print("Listo: orquestado.wav ·", ", ".join(f"z{k}→{INSTRUMENTOS[i][0]}" for i, k in enumerate(orden)))


if __name__ == "__main__":
    main()

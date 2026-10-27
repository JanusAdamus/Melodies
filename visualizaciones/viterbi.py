"""HMM ilustrativo: acordes ocultos (I, IV, V de Sol mayor) que emiten las notas de la melodía, decodificado con Viterbi.

Un paso de tiempo = un compás; la observación del compás son sus notas, pesadas por duración.
Los parámetros están puestos a mano para explicar el algoritmo; no son los estimados en la tesis.
"""
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

DIR = Path(__file__).parent
SEG_POR_COMPAS = 3 * 60 / 100          # 3/4 a negra = 100, igual que melodia.py

ESTADOS = ["I", "IV", "V"]
ACORDES = {"I": (7, 11, 2), "IV": (0, 4, 7), "V": (2, 6, 9)}     # Sol-Si-Re, Do-Mi-Sol, Re-Fa♯-La
DIATONICAS = {7, 9, 11, 0, 2, 4, 6}
INICIAL = np.array([0.6, 0.2, 0.2])
TRANSICION = np.array([[0.5, 0.2, 0.3],    # desde I
                       [0.3, 0.3, 0.4],    # desde IV
                       [0.6, 0.1, 0.3]])   # desde V
P_ACORDE, P_DIATONICA = 0.72, 0.26         # masa para notas del acorde / resto de la escala


def emision(estado: str) -> np.ndarray:
    """P(clase de altura | acorde) sobre las 12 clases."""
    tonos = ACORDES[estado]
    otras = DIATONICAS - set(tonos)
    p = np.full(12, (1 - P_ACORDE - P_DIATONICA) / (12 - len(DIATONICAS)))
    p[list(tonos)] = P_ACORDE / len(tonos)
    p[list(otras)] = P_DIATONICA / len(otras)
    return p


EMISION = np.array([emision(e) for e in ESTADOS])


def observaciones(notas: list[dict]) -> list[list[tuple[int, float]]]:
    """Agrupa (clase, duración en negras) por compás."""
    compases = int(round((notas[-1]["inicio"] + notas[-1]["duracion"]) / SEG_POR_COMPAS))
    obs = [[] for _ in range(compases)]
    seg_negra = SEG_POR_COMPAS / 3
    for n in notas:
        negra = round(n["inicio"] / seg_negra * 4) / 4   # 1.7999… s es la negra 3, no la 2.99
        obs[int(negra // 3)].append((n["clase"], n["duracion"] / seg_negra))
    return obs


def log_emision_compas(compas: list[tuple[int, float]]) -> np.ndarray:
    """log b_j(o_t): cada nota cuenta proporcional a su duración (una negra = 1)."""
    return np.array([sum(d * np.log(EMISION[j, c]) for c, d in compas) for j in range(len(ESTADOS))])


@dataclass(frozen=True)
class Resultado:
    delta: np.ndarray       # (T, N) log δ_t(j)
    psi: np.ndarray         # (T, N) mejor predecesor
    camino: list[int]


def viterbi(obs: list[list[tuple[int, float]]]) -> Resultado:
    log_a = np.log(TRANSICION)
    b = np.array([log_emision_compas(o) for o in obs])
    delta = np.zeros_like(b)
    psi = np.zeros(b.shape, dtype=int)
    delta[0] = np.log(INICIAL) + b[0]
    for t in range(1, len(obs)):
        candidatos = delta[t - 1][:, None] + log_a           # [i, j]
        psi[t] = candidatos.argmax(axis=0)
        delta[t] = candidatos.max(axis=0) + b[t]
    camino = [int(delta[-1].argmax())]
    for t in range(len(obs) - 1, 0, -1):
        camino.insert(0, int(psi[t, camino[0]]))
    return Resultado(delta, psi, camino)


def _fuerza_bruta(obs) -> list[int]:
    b = np.array([log_emision_compas(o) for o in obs])
    log_a, log_pi = np.log(TRANSICION), np.log(INICIAL)
    puntaje = lambda s: log_pi[s[0]] + b[0, s[0]] + sum(log_a[s[t - 1], s[t]] + b[t, s[t]] for t in range(1, len(s)))
    return list(max(product(range(len(ESTADOS)), repeat=len(obs)), key=puntaje))


def cargar() -> tuple[list[dict], list, Resultado]:
    notas = json.loads((DIR / "salida/minueto.json").read_text(encoding="utf-8"))
    obs = observaciones(notas)
    return notas, obs, viterbi(obs)


if __name__ == "__main__":
    notas, obs, r = cargar()
    assert np.allclose(EMISION.sum(axis=1), 1) and np.allclose(TRANSICION.sum(axis=1), 1)
    assert viterbi(obs[:8]).camino == _fuerza_bruta(obs[:8]), "Viterbi no coincide con fuerza bruta"
    print(len(obs), "compases:", " ".join(ESTADOS[s] for s in r.camino))

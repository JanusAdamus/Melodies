"""Corre el HDP-HMM de la tesis (código de Melodies, sin modificarlo) sobre la melodía de los videos 1 y 2
y guarda la traza por iteración: pesos β, estados muestreados, estados activos y log-verosimilitud.

Uso: python traza_hdp.py   (escribe salida/traza_hdp.json)
Parámetros = los de la corrida multicorpus_full de la tesis: K=20, 80 iteraciones, burn-in 40, semilla 7.
"""
import json
import sys
from pathlib import Path

import numpy as np

MELODIES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MELODIES))
from src.data.observations import build_observation_sequence  # noqa: E402
from src.data.parsing import extract_events, parse_score  # noqa: E402
from src.models import hdp_hmm  # noqa: E402

DIR = Path(__file__).parent
OBRA = MELODIES / "external/library/scores/Bach_Minuet_in_G_Major_BWV_Anh._114.mxl"
PARAMS = {"n_states": 20, "n_iters": 80, "burn_in": 40, "seed": 7}
NEGRAS_16_COMPASES = 16 * 3


class HDPConTraza(hdp_hmm.TruncatedHDPHMM):
    """Igual al modelo de la tesis; solo anota β y z en cada iteración."""

    def fit(self, observations):
        self.betas, self.estados = [], []
        original_beta, original_ffbs = self._update_beta, hdp_hmm.ffbs_sample

        def beta_anotada(*args, **kwargs):
            beta, modo = original_beta(*args, **kwargs)
            self.betas.append(beta.copy())
            return beta, modo

        def ffbs_anotado(**kwargs):
            z, ll = original_ffbs(**kwargs)
            self.estados.append(z.copy())
            return z, ll

        self._update_beta, hdp_hmm.ffbs_sample = beta_anotada, ffbs_anotado
        try:
            return super().fit(observations)
        finally:
            hdp_hmm.ffbs_sample = original_ffbs


def main() -> None:
    eventos = [e for e in extract_events(parse_score(str(OBRA)), prefer_treble=True) if e.offset < NEGRAS_16_COMPASES]
    melodia = json.loads((DIR / "salida/minueto.json").read_text(encoding="utf-8"))
    assert [e.representative_pitch_class for e in eventos] == [n["clase"] for n in melodia], "no es la misma melodía de los videos"

    modelo = HDPConTraza(**PARAMS)
    r = modelo.fit(build_observation_sequence(eventos, "pitch_class"))
    assert len(modelo.betas) == len(modelo.estados) == PARAMS["n_iters"]
    activos = sorted(int(s) for s in r.active_states)
    salida = {
        "params": PARAMS,
        "betas": [b.round(5).tolist() for b in modelo.betas],
        "estados": [z.tolist() for z in modelo.estados],
        "log_verosimilitud": [round(x, 3) for x in r.diagnostics.log_likelihood_history],
        "activos_por_iteracion": r.diagnostics.active_state_history,
        "mejor_iteracion": r.diagnostics.best_iteration,
        "estados_finales": r.latent_states.tolist(),
        "activos_finales": activos,
        "transicion_final": np.asarray(r.transition_matrix)[np.ix_(activos, activos)].round(4).tolist(),
        "emision_final": np.asarray(r.emission_matrix)[activos].round(4).tolist(),
        "log_verosimilitud_final": round(float(r.log_likelihood), 3),
    }
    (DIR / "salida/traza_hdp.json").write_text(json.dumps(salida), encoding="utf-8")
    print(f"K efectivo = {r.effective_states} de {PARAMS['n_states']}; log L = {r.log_likelihood:.3f}; "
          f"mejor iteración {r.diagnostics.best_iteration}; activos al inicio {r.diagnostics.active_state_history[:5]}")


if __name__ == "__main__":
    main()

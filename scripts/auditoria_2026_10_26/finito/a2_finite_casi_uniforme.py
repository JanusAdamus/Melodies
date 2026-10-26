"""El HMM finito fijo: cuanto se separa del modelo uniforme log(1/12) por token."""
import sys

import numpy as np

sys.path.insert(0, ".")
from src.models.finite_hmm import FiniteChordHMM
from src.models.inference import forward_log_likelihood
from src.data.parsing import extract_events, parse_score
from src.data.observations import build_observation_sequence

m = FiniteChordHMM()
pi, A, B = m.initial_probs, m.transition_matrix, m.emission_matrix
ev = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
print(f"diag(A): min={np.diag(A).min():.4f} max={np.diag(A).max():.4f}; |lambda_2|={ev[1]:.4f}")
print("pi @ B (predictiva del primer token):", np.round(pi @ B, 6))
stat = np.linalg.matrix_power(A.T, 200) @ pi
print("estacionaria @ B:", np.round(stat @ B, 4))


def per_token_gain(x):
    ll = forward_log_likelihood(pi, A, B, np.asarray(x))[0]
    return ll, ll - len(x) * np.log(1 / 12), (ll - len(x) * np.log(1 / 12)) / len(x)


cases = {
    "todo C (T=100)": [0] * 100,
    "escala C mayor x10": [0, 2, 4, 5, 7, 9, 11] * 10,
    "arpegio C-E-G x30": [0, 4, 7] * 30,
    "aleatorio uniforme T=500": list(np.random.default_rng(1).integers(0, 12, 500)),
}
events = extract_events(parse_score("examples/example_score.musicxml"))
cases["examples/example_score.musicxml"] = list(build_observation_sequence(events, "pitch_class").tokens)
for name, x in cases.items():
    ll, g, gt = per_token_gain(x)
    # i.i.d. MLE de referencia (cota de lo que da solo la frecuencia)
    c = np.bincount(x, minlength=12); p = c / c.sum()
    iid = float(np.sum(c[c > 0] * np.log(p[c > 0])))
    print(f"{name:34s} T={len(x):4d} logL_fin={ll:9.2f} uniforme={len(x)*np.log(1/12):9.2f} "
          f"ganancia/token={gt:+.4f} nats | iid-MLE={iid:9.2f}")

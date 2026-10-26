"""library_batch.analyze_catalog_piece: el finito siempre usa pitch_class, el HDP usa obs_type."""
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.analysis.library_batch import analyze_catalog_piece

row = pd.Series({"path": "examples/example_score.musicxml", "title": "ej"})
params = {"n_states": 20, "n_iters": 30, "burn_in": 15, "seed": 7}
for obs in ("pitch_class", "midi_note", "interval"):
    r = analyze_catalog_piece(row, obs_type=obs, model="both", prefer_treble=False, hdp_params=params)
    print(f"obs_type={obs:12s} n_events={r['n_events']} finite_LL={r['finite_log_likelihood']:.2f} hdp_LL={r['hdp_log_likelihood']:.2f}")

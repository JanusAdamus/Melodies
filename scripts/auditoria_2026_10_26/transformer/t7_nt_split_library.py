"""Particiones de next_token_experiment (assign_piece_splits, seed 7) sobre el corpus library: variantes de una obra."""
import json, sys
from next_token_experiment.config import SplitConfig
from next_token_experiment.experiment.splits import assign_piece_splits
from next_token_experiment.schemas import PreparedPiece
P = []
for l in open(sys.argv[1], encoding="utf-8"):
    d = json.loads(l)
    if d["kind"] == "piece": d.pop("kind"); P.append(PreparedPiece(vocabulary=[], **d))
a = assign_piece_splits(P, SplitConfig())
for key in ("Elise", "Canon", "Twinkle", "Ode", "Bella", "Moonlight", "Turkish"):
    fam = [p for p in P if key.lower() in p.piece_id.lower()]
    if len(fam) > 1: print(key, [(p.piece_id.split("\\")[-1][:28], p.canonical_work_id[:22], a[p.canonical_work_id]) for p in fam])

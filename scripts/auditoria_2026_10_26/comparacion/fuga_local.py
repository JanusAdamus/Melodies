"""Fuga por contenido en el smoke local: piezas de prueba/validacion vs entrenamiento (por nombre base)."""
import json, sys, collections
cache, run = sys.argv[1], sys.argv[2]
base = lambda p: p.replace("\\", "/").split("/")[-1]
P = {base(json.loads(l)["piece_id"]): json.loads(l) for l in open(cache, encoding="utf-8")}
S = {"test": json.load(open(f"{run}/splits/test_pieces.json")), "validation": json.load(open(f"{run}/splits/val_pieces.json")),
     "train": json.load(open(f"{run}/splits/train_fractions_seed1.json"))["fractions"][-1]}
split = {base(p): s for s, d in S.items() for p in d["piece_ids"]}
N = 16
def grams(t):
    iv = [(b - a) % 12 for a, b in zip(t, t[1:])]
    return {tuple(iv[i:i+N]) for i in range(len(iv) - N + 1)}
G = {b: grams(P[b]["tokens"]) for b in split}
nll = collections.defaultdict(dict)
import csv
for r in csv.DictReader(open(f"{run}/piece_metrics_raw.csv")):
    nll[base(r["piece_id"])][r["model"]] = float(r["nll_per_token"])
for tgt, pool in (("test", "train"), ("test", "validation"), ("validation", "train")):
    print(f"== {tgt} vs {pool}")
    for b in sorted(x for x in split if split[x] == tgt):
        best = max(((len(G[b] & G[q]) / max(1, len(G[b])), q) for q in split if split[q] == pool), default=(0, None))
        if best[0] >= 0.25:
            extra = {m: round(v, 3) for m, v in nll.get(b, {}).items()}
            print(f"   {best[0]:.2f} {b[:45]:45} <-> {best[1][:40]:40} {extra}")

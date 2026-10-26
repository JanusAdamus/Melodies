"""Fracción de tokens de test evaluados en cortes que no empiezan la obra (BOS a mitad de pieza)."""
import json, sys
cache, plan = sys.argv[1], sys.argv[2]
tok = {}
for l in open(cache, encoding="utf-8"):
    d = json.loads(l)
    if d["kind"] == "piece": tok[d["piece_id"]] = len(d["tokens"])
test = json.load(open(f"{plan}/splits/test_pieces.json"))["piece_ids"]
train = json.load(open(f"{plan}/splits/train_fractions_seed1.json"))["fractions"][-1]["piece_ids"]
C = 128
tot = sum(tok[p] for p in test); later = sum(max(0, tok[p] - C) for p in test)
print(f"test: {tot} tokens; {later/tot:.1%} se puntúan en cortes con BOS a mitad de obra")
# en entrenamiento, stride 64: ventanas que empiezan a mitad de obra
w_mid = sum(len(range(64, max(0, tok[p]-32)+1, 64)) for p in train); w0 = len(train)
print(f"train (stride 64): ~{w_mid} ventanas con BOS a mitad de obra vs {w0} con BOS al inicio")

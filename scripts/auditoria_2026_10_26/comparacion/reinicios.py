"""Cuántos eventos de prueba caen al inicio de un corte BOS a mitad de obra (corrida de tesis)."""
import json, sys
sys.path.insert(0, sys.argv[3])
from next_token_experiment.data.dataset import build_evaluation_slices
pieces = {json.loads(l)["piece_id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
test = json.load(open(f"{sys.argv[2]}/execution_plan.json"))["splits"]["test"]["piece_ids"]
tot = first = mid = 0; early = {1: 0, 8: 0, 16: 0}
for p in test:
    sl = build_evaluation_slices(len(pieces[p]["tokens"]), 128)
    tot += len(pieces[p]["tokens"]); first += 1; mid += len(sl) - 1
    for a, b in sl[1:]:
        for k in early: early[k] += min(k, b - a)
print(f"tokens prueba={tot}; cortes iniciales={first}; cortes a mitad de obra={mid}")
for k, v in early.items(): print(f"  eventos en las primeras {k} posiciones de cortes a mitad de obra: {v} ({v/tot:.1%})")
print("ejemplo de cortes (pieza 300 tokens):", build_evaluation_slices(300, 128))

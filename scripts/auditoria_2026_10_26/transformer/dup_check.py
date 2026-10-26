"""Busca duplicados de contenido (secuencias de tokens) entre train/val/test del plan PDMX."""
import json, sys, collections
cache, plan = sys.argv[1], sys.argv[2]
tok = {}; cw = {}
for line in open(cache, encoding="utf-8"):
    d = json.loads(line)
    if d["kind"] != "piece": continue
    tok[d["piece_id"]] = tuple(d["tokens"]); cw[d["piece_id"]] = d["canonical_work_id"]
def ids(f): return json.load(open(f"{plan}/splits/{f}"))["piece_ids"]
test, val = ids("test_pieces.json"), ids("val_pieces.json")
train = json.load(open(f"{plan}/splits/train_fractions_seed1.json"))["fractions"][-1]["piece_ids"]
print("n train/val/test", len(train), len(val), len(test), "en cache:", sum(p in tok for p in train+val+test))
split_of = {**{p:"train" for p in train}, **{p:"val" for p in val}, **{p:"test" for p in test}}
by_seq = collections.defaultdict(list)
for p, s in split_of.items(): by_seq[tok[p]].append(p)
cross = [g for g in by_seq.values() if len({split_of[p] for p in g}) > 1]
print("grupos con secuencia IDENTICA en >1 particion:", len(cross))
test_exact = sum(1 for p in test if any(split_of[q]=="train" for q in by_seq[tok[p]]))
print("piezas de test con copia exacta en train:", test_exact, "de", len(test))
for g in cross[:5]: print("  ", [(split_of[p], cw[p][:40], len(tok[p])) for p in g])
# near-duplicate: fraccion de 32-gramas de cada pieza de test presentes en train
N=32
train_grams=set()
for p in train:
    t=tok[p]; train_grams.update(t[i:i+N] for i in range(len(t)-N+1))
fr=[]; tot=hit=0
for p in test:
    t=tok[p]; g=[t[i:i+N] for i in range(len(t)-N+1)]
    if not g: continue
    h=sum(x in train_grams for x in g); fr.append(h/len(g)); tot+=len(g); hit+=h
print(f"test: fraccion global de {N}-gramas que aparecen en train = {hit/tot:.3f}")
print("piezas de test con >=50% de 32-gramas en train:", sum(f>=0.5 for f in fr), "| >=90%:", sum(f>=0.9 for f in fr))

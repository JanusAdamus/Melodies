"""Grupos canonicos multi-archivo en la corrida de tesis: ¿son la misma obra?"""
import json, sys, collections
cache, plan = sys.argv[1], sys.argv[2]
pieces = {json.loads(l)["piece_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
e = json.load(open(f"{plan}/execution_plan.json"))
for name in ("train_pool", "validation", "test"):
    g = collections.defaultdict(list)
    for p in e["splits"][name]["piece_ids"]:
        g[pieces[p]["canonical_work_id"]].append(p)
    multi = {k: v for k, v in g.items() if len(v) > 1}
    print(f"== {name}: {len(multi)} grupos multi-archivo, {sum(len(v) for v in multi.values())} archivos")
    if name != "train_pool":
        for k, v in multi.items():
            same = len({tuple(pieces[p]["tokens"]) for p in v}) == 1
            print(f"   [{k}] identicas={same}: " + " | ".join(f"{pieces[p]['title'][:30]!r}({len(pieces[p]['tokens'])})" for p in v))
# ids cortos/genericos
allids = collections.Counter(pieces[p]["canonical_work_id"] for s in e["splits"].values() for p in s["piece_ids"])
short = {k: c for k, c in allids.items() if len(k.split()) < 2}
print("ids de <2 palabras:", len(short), "ejemplos:", list(short.items())[:15])

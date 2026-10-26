"""Fuga por contenido entre particiones de la corrida de tesis (plan_pdmx_canonical_20260825).
Uso: python fuga_contenido.py CACHE.jsonl PLAN_DIR
"""
import json, sys, collections, hashlib
cache, plan = sys.argv[1], sys.argv[2]
pieces = {}
for line in open(cache, encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") == "piece":
        pieces[r["piece_id"]] = r
e = json.load(open(f"{plan}/execution_plan.json"))
split_of = {}
for name in ("train_pool", "validation", "test"):
    for pid in e["splits"][name]["piece_ids"]:
        split_of[pid] = name
missing = [p for p in split_of if p not in pieces]
print("piezas en plan:", len(split_of), "en cache:", len(pieces), "faltan en cache:", len(missing))
for name in ("train_pool", "validation", "test"):
    ids = [p for p in split_of if split_of[p] == name]
    print(name, len(ids), "tokens", sum(len(pieces[p]["tokens"]) for p in ids if p in pieces), "plan_tokens", e["splits"][name]["n_tokens"])

# 1) duplicados exactos de secuencia
by_hash = collections.defaultdict(list)
for pid in split_of:
    if pid in pieces:
        by_hash[hashlib.sha1(bytes(pieces[pid]["tokens"])).hexdigest()].append(pid)
cross_exact = [g for g in by_hash.values() if len({split_of[p] for p in g}) > 1]
print("\n[exacto] grupos de secuencia identica que cruzan particiones:", len(cross_exact))
for g in cross_exact[:8]:
    print("  ", [(split_of[p], pieces[p]["title"][:40], pieces[p]["canonical_work_id"][:30], len(pieces[p]["tokens"])) for p in g])

# 2) cuasi-duplicados: contencion de 16-gramas transpuesto-invariantes (intervalos mod 12)
N = 16
def grams(tok):
    iv = [(b - a) % 12 for a, b in zip(tok, tok[1:])]
    return {tuple(iv[i:i+N]) for i in range(len(iv) - N + 1)}
G = {p: grams(pieces[p]["tokens"]) for p in split_of if p in pieces}
# indice invertido sobre train+val
index = collections.defaultdict(set)
for p, gs in G.items():
    if split_of[p] != "test":
        for g in gs:
            index[g].add(p)
gram_df = {g: len(s) for g, s in index.items()}
def best_match(p, pool_splits):
    gs = G[p]
    if not gs: return None, 0.0
    cnt = collections.Counter()
    for g in gs:
        for q in index.get(g, ()):
            if split_of[q] in pool_splits:
                cnt[q] += 1
    if not cnt: return None, 0.0
    q, c = cnt.most_common(1)[0]
    return q, c / len(gs)
THR = 0.5
out = {}
for target, pool in (("test", {"train_pool"}), ("test", {"validation"}), ("validation", {"train_pool"})):
    hits = []
    ids = [p for p in G if split_of[p] == target]
    for p in ids:
        q, c = best_match(p, pool)
        if q and c >= THR:
            hits.append((p, q, c))
    toks = sum(len(pieces[p]["tokens"]) for p, _, _ in hits)
    tot = sum(len(pieces[p]["tokens"]) for p in ids)
    works = {pieces[p]["canonical_work_id"] for p, _, _ in hits}
    allw = {pieces[p]["canonical_work_id"] for p in ids}
    print(f"\n[{target} vs {sorted(pool)}] piezas con contencion>={THR}: {len(hits)}/{len(ids)}; obras {len(works)}/{len(allw)}; tokens {toks}/{tot} ({toks/tot:.1%})")
    for p, q, c in sorted(hits, key=lambda h: -h[2])[:12]:
        print(f"   {c:.2f}  {pieces[p]['title'][:38]!r:40} [{pieces[p]['canonical_work_id'][:28]}]  <->  {pieces[q]['title'][:38]!r:40} [{pieces[q]['canonical_work_id'][:28]}]")
    out[f"{target}_vs_{'+'.join(sorted(pool))}"] = [[p, q, c] for p, q, c in hits]
json.dump(out, open("fuga_contenido_hits.json", "w"), indent=1)

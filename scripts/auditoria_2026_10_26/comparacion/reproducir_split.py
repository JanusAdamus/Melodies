"""¿El split de la tesis se reproduce desde la cache? (ids Windows tal cual vs ids POSIX)"""
import json, sys
sys.path.insert(0, sys.argv[1])
from next_token_experiment.schemas import PreparedPiece
# El runner usa el constructor compartido de Comparacion.splits (agrupa por metadatos y contenido).
from Comparacion.splits import build_grouped_fixed_splits
import dataclasses
fields = {f.name for f in dataclasses.fields(PreparedPiece)}
rows = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
plan = json.load(open(f"{sys.argv[3]}/execution_plan.json"))["splits"]
excl = set()  # cache contiene 3000 piezas; el plan usa 2933
keep = {p for s in plan.values() for p in s["piece_ids"]}
def build(posix):
    out = []
    for r in rows:
        if r["piece_id"] not in keep: continue
        d = {k: v for k, v in r.items() if k in fields}; d.setdefault("vocabulary", [])
        if posix:
            d["piece_id"] = d["piece_id"].replace("\\", "/"); d["source_path"] = d["source_path"].replace("\\", "/")
        out.append(PreparedPiece(**d))
    return out
for posix in (False, True):
    fs = build_grouped_fixed_splits(build(posix), test_fraction=0.15, validation_fraction=0.10, seed=7)
    got = {p.piece_id.replace("/", "\\") for p in fs.test_pieces}
    ref = set(plan["test"]["piece_ids"])
    print(f"ids_posix={posix}: test={len(got)}  coincide con plan={got == ref}  interseccion={len(got & ref)}")

"""Ubica variantes de la misma obra (Fur Elise, Canon, Bella Ciao) en las particiones."""
import json, sys
cache, plan = sys.argv[1], sys.argv[2]
tok={}; cw={}
for l in open(cache, encoding="utf-8"):
    d=json.loads(l)
    if d["kind"]=="piece": tok[d["piece_id"]]=d["tokens"]; cw[d["piece_id"]]=d["canonical_work_id"]
sp={}
for f,n in [("test_pieces.json","test"),("val_pieces.json","val")]:
    for p in json.load(open(f"{plan}/splits/{f}"))["piece_ids"]: sp[p]=n
for p in json.load(open(f"{plan}/splits/train_fractions_seed1.json"))["fractions"][-1]["piece_ids"]: sp[p]="train"
for key in ["Elise","Canon","Bella"]:
    fam=[p for p in sp if key in p]
    print(key, [(p.split('\\')[-1][:35], sp[p], cw[p][:30]) for p in fam])
    # 16-grama compartido entre variantes de distinta particion
    for a in fam:
        for b in fam:
            if sp[a]=="test" and sp[b]=="train":
                ga={tuple(tok[a][i:i+16]) for i in range(len(tok[a])-15)}
                gb={tuple(tok[b][i:i+16]) for i in range(len(tok[b])-15)}
                print(f"   test {a.split(chr(92))[-1][:25]} vs train {b.split(chr(92))[-1][:25]}: {len(ga&gb)/max(1,len(ga)):.2f} de 16-gramas de test en train")

"""¿Las 3 semillas de datos producen el mismo conjunto de entrenamiento en frac=1.0?"""
import json, sys
d = sys.argv[1]
sets = {}
for s in (1, 2, 3):
    fr = json.load(open(f"{d}/splits/train_fractions_seed{s}.json"))["fractions"]
    sets[s] = {f["frac"]: tuple(f["piece_ids"]) for f in fr}
for frac in sorted(sets[1]):
    same = sets[1][frac] == sets[2][frac] == sets[3][frac]
    print(f"frac={frac}: conjuntos identicos entre data_seed 1,2,3 = {same} (n={len(sets[1][frac])})")

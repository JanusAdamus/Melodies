"""Costo, para el HMM finito, de reiniciar con BOS cada 128 tokens frente a contexto completo (PDMX)."""
import json, sys
from next_token_experiment.schemas import PreparedPiece
from Comparacion.classical_models import FiniteGlobalHMM
cache, plan = sys.argv[1], sys.argv[2]
P = {}
for l in open(cache, encoding="utf-8"):
    d = json.loads(l)
    if d["kind"] == "piece":
        d.pop("kind"); P[d["piece_id"]] = PreparedPiece(vocabulary=[str(i) for i in range(12)], **d)
ids = lambda f: json.load(open(f"{plan}/splits/{f}"))["piece_ids"]
train = [P[p] for p in json.load(open(f"{plan}/splits/train_fractions_seed1.json"))["fractions"][-1]["piece_ids"][:500]]
val, test = [P[p] for p in ids("val_pieces.json")[:80]], [P[p] for p in ids("test_pieces.json")]
h = FiniteGlobalHMM(candidate_num_states=(12,), max_iterations=30, tolerance=1e-4, seed=1, vocab_size=13)
h.fit(train, val, bos_token_id=12, max_context_length=128)
for C in (128, 10**6):
    print(f"contexto={C}: test NLL={h.evaluate(test, bos_token_id=12, max_context_length=C)['summary']['test_nll_per_token']:.4f}")

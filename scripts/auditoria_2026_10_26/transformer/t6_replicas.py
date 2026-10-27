"""En frac=1.0 las data_seeds producen el mismo conjunto y el Transformer da resultados idénticos."""
import random
from dataclasses import replace
from next_token_experiment.protocol import build_default_experiment_config
from next_token_experiment.schemas import PreparedPiece
from Comparacion.config import LearningCurveConfig
from Comparacion.runner import _build_transformer_dataloaders, _build_transformer_model
from Comparacion.splits import build_nested_training_subsets
rng = random.Random(0)
pieces = [PreparedPiece(f"p{i:02d}", "x", "t", "c", f"w{i}", "pitch_class", [str(k) for k in range(12)],
          [rng.randrange(12) for _ in range(rng.randint(40, 200))], 0, {}) for i in range(30)]
pool, val, test = pieces[:20], pieces[20:25], pieces[25:]
subsets = {s: dict(build_nested_training_subsets(pool, fractions=(0.5, 1.0), data_seed=s)) for s in (1, 2, 3)}
for f in (0.5, 1.0):
    print(f"frac={f}: conjuntos idénticos entre data_seed 1,2,3:",
          len({tuple(p.piece_id for p in subsets[s][f]) for s in (1, 2, 3)}) == 1)
exp = build_default_experiment_config()
exp = replace(exp, transformer=replace(exp.transformer, d_model=32, n_layers=1, ff_dim=64, max_epochs=2, batch_size=8),
              hardware=replace(exp.hardware, cpu_threads=1))
cfg = LearningCurveConfig(experiment=exp, results_root="/tmp/x")
res = {}
for ds in (1, 2):
    for ms in (1, 2):
        m = _build_transformer_model(cfg, ms)
        dl = _build_transformer_dataloaders(cfg, subsets[ds][1.0], val, test, model_seed=ms, data_seed=ds)
        m.fit(dl["train"], dl["validation"]); res[(ds, ms)] = m.evaluate(dl["test"])["summary"]["nll_per_token"]
for k, v in res.items(): print(f"data_seed={k[0]} model_seed={k[1]} test_nll={v!r}")

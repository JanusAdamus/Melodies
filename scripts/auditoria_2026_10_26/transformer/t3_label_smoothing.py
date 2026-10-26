"""Brecha entre la 'NLL de validación' con label smoothing (early stopping) y la NLL real, en datos aprendibles."""
import random
from dataclasses import replace
from next_token_experiment.protocol import build_default_experiment_config
from next_token_experiment.schemas import PreparedPiece
from Comparacion.config import LearningCurveConfig
from Comparacion.runner import _build_transformer_dataloaders, _build_transformer_model
rng = random.Random(0)
def piece(i):  # escala ascendente con 10% de ruido: muy predecible
    t, x = [], rng.randrange(12)
    for _ in range(rng.randint(60, 200)):
        x = (x + 1) % 12 if rng.random() > 0.1 else rng.randrange(12); t.append(x)
    return PreparedPiece(f"p{i:02d}", "x", "t", "c", f"w{i}", "pitch_class", [str(k) for k in range(12)], t, len(t), {})
pieces = [piece(i) for i in range(40)]
exp = build_default_experiment_config()
for ls in (0.0, 0.05):
    e = replace(exp, transformer=replace(exp.transformer, d_model=32, n_layers=1, ff_dim=64, max_epochs=8,
                batch_size=8, learning_rate=3e-3, label_smoothing=ls), hardware=replace(exp.hardware, cpu_threads=1))
    cfg = LearningCurveConfig(experiment=e, results_root="/tmp/x")
    m = _build_transformer_model(cfg, 1)
    dl = _build_transformer_dataloaders(cfg, pieces[:30], pieces[30:35], pieces[35:])
    fit = m.fit(dl["train"], dl["validation"])
    real = m.evaluate(dl["validation"])["summary"]["nll_per_token"]
    print(f"label_smoothing={ls}: best_validation_nll reportada={fit['summary']['best_validation_nll']:.4f} "
          f"NLL real val={real:.4f}  diferencia={fit['summary']['best_validation_nll']-real:+.4f}")

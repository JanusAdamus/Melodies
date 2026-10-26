"""Label smoothing en la NLL de validación, orden de datos vs semilla de modelo, truncado max_windows."""
import random, sys
from dataclasses import replace
import torch
sys.path.insert(0, ".")
from next_token_experiment.data.dataset import WindowedSequenceDataset, collate_window_batch
from next_token_experiment.data.tokenizer import SequenceTokenizer
from next_token_experiment.protocol import build_default_experiment_config
from next_token_experiment.profiles import build_profile_config
from next_token_experiment.schemas import PreparedPiece
from Comparacion.config import LearningCurveConfig
from Comparacion.runner import _build_transformer_dataloaders, _build_transformer_model

V = 12
rng = random.Random(0)
def piece(i, n): return PreparedPiece(f"p{i:02d}", "x", "t", "c", f"w{i}", "pitch_class", [str(k) for k in range(V)],
                                     [rng.randrange(V) for _ in range(n)], n, {})
pieces = [piece(i, rng.randint(40, 300)) for i in range(30)]
exp = build_default_experiment_config()
exp = replace(exp, transformer=replace(exp.transformer, d_model=32, n_layers=1, ff_dim=64, max_epochs=2, batch_size=8),
              hardware=replace(exp.hardware, cpu_threads=1))
cfg = LearningCurveConfig(experiment=exp, results_root="/tmp/x")

# A) Orden de lotes de entrenamiento: no depende de model_seed (ni de data_seed)
def first_batches(model_seed):
    _build_transformer_model(cfg, model_seed)  # el runner crea el modelo antes de los loaders
    dl = _build_transformer_dataloaders(cfg, pieces[:20], pieces[20:25], pieces[25:])
    return [tuple(b["piece_ids"]) + tuple(b["start_indices"]) for b in list(dl["train"])[:3]]
print("[seed] orden de lotes idéntico con model_seed=1 y model_seed=2:", first_batches(1) == first_batches(2))
print("[seed] generador del DataLoader sembrado con experiment.split.seed =", exp.split.seed)

# B) Label smoothing: la 'NLL de validación' usada para early stopping y reportada no es NLL
for prof in ("gpu_extended", "research_richer_events"):
    print(f"[ls] perfil {prof}: label_smoothing={build_profile_config(prof).transformer.label_smoothing}")
cfg_ls = replace(cfg, experiment=replace(exp, transformer=replace(exp.transformer, label_smoothing=0.05)))
m = _build_transformer_model(cfg_ls, 1)
dl = _build_transformer_dataloaders(cfg_ls, pieces[:20], pieces[20:25], pieces[25:])
fit = m.fit(dl["train"], dl["validation"])
ev = m.evaluate(dl["validation"])["summary"]["nll_per_token"]
print(f"[ls] best_validation_nll (fit, con smoothing)={fit['summary']['best_validation_nll']:.4f}  NLL real en validación={ev:.4f}")

# C) max_windows trunca tras ordenar por piece_id: no es muestra aleatoria
tok = SequenceTokenizer("pitch_class", tuple(str(k) for k in range(V)))
ds = WindowedSequenceDataset(pieces, tok, "train", 128, 64, 32, max_windows=20)
full = WindowedSequenceDataset(pieces, tok, "train", 128, 64, 32)
print(f"[max_windows] con 20 ventanas: piezas usadas={sorted({e.piece_id for e in ds.examples})} de {len({e.piece_id for e in full.examples})}")
dv = WindowedSequenceDataset(pieces, tok, "test", 128, 128, 32, max_windows=5)
print(f"[max_windows] test con max_windows=5 conserva {len(dv)} ventanas (se ignora en eval)")

"""Verifica máscara causal, desplazamiento de etiquetas, padding y NLL por token del Transformer."""
import math, random, sys
import torch, torch.nn.functional as F
from next_token_experiment.config import HardwareConfig
from next_token_experiment.data.dataset import WindowedSequenceDataset, collate_window_batch
from next_token_experiment.data.tokenizer import SequenceTokenizer
from next_token_experiment.models.small_transformer import SmallTransformerNextTokenModel, SmallTransformerStudySpec
from next_token_experiment.schemas import PreparedPiece

V = 12
tok = SequenceTokenizer("pitch_class", tuple(str(i) for i in range(V)))
def spec(rel, ls=0.0):
    return SmallTransformerStudySpec("decoder_only", 2, 32, 4, 64, 0.0, 1e-2, 0.0, 4, 3, 2, 1, 1.0, ls,
        0.5, 2, 1e-5, True, rel, 8, 32, 1, 4, 4, 1.0, 4)
def model(rel, ls=0.0, seed=1):
    return SmallTransformerNextTokenModel(spec(rel, ls), tok.vocab_size, tok.bos_token_id, tok.pad_token_id, 16,
        hardware=HardwareConfig(cpu_threads=1), seed=seed)
rng = random.Random(0)
pieces = [PreparedPiece(f"p{i}", "x", "t", "c", f"w{i}", "pitch_class", list(tok.musical_vocabulary),
          [rng.randrange(V) for _ in range(rng.randint(20, 45))], 0, {}) for i in range(5)]

# 1) desplazamiento de etiquetas
ds = WindowedSequenceDataset(pieces, tok, "test", 16, 16, 2)
ex = ds.examples[1]
assert ex.input_tokens[0] == tok.bos_token_id and ex.input_tokens[1:] == ex.target_tokens[:-1]
print("[shift] input[t+1]==target[t], input[0]=BOS: OK")

for rel in (False, True):
    m = model(rel); m.model.eval()
    # 2) causalidad: cambiar tokens futuros no cambia logits pasados
    x = torch.randint(0, V, (1, 16)); x[0, 0] = tok.bos_token_id
    mask = torch.ones_like(x, dtype=torch.bool)
    base = m.model(x, mask)
    worst = 0.0
    for t in range(15):
        y = x.clone(); y[0, t + 1:] = torch.randint(0, V, (15 - t,))
        worst = max(worst, (m.model(y, mask)[0, : t + 1] - base[0, : t + 1]).abs().max().item())
    print(f"[causal rel={rel}] max |delta logit| en posiciones <=t al cambiar el futuro: {worst:.2e}")
    # 3) invariancia a padding/lote: NLL de una ventana sola == en lote con relleno
    batch = collate_window_batch([ds[i] for i in range(len(ds))], tok.pad_token_id)
    lp = torch.log_softmax(m._comparison_logits(m.model(batch["input_ids"], batch["attention_mask"])), -1)
    diffs = []
    manual_total, n = 0.0, 0
    for i in range(len(ds)):
        single = collate_window_batch([ds[i]], tok.pad_token_id)
        lps = torch.log_softmax(m._comparison_logits(m.model(single["input_ids"], single["attention_mask"])), -1)
        tg = m._comparison_targets(single["target_ids"])[0]
        nll_single = -lps[0, torch.arange(len(tg)), tg].sum().item()
        L = int(batch["sequence_lengths"][i]); tgb = m._comparison_targets(batch["target_ids"])[i, :L]
        nll_batch = -lp[i, torch.arange(L), tgb].sum().item()
        diffs.append(abs(nll_single - nll_batch)); manual_total += nll_single; n += len(tg)
    print(f"[padding rel={rel}] max |NLL solo - NLL en lote| = {max(diffs):.2e}")
    # 4) evaluate() == NLL manual por token, y n_tokens == tokens reales
    loader = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False,
        collate_fn=lambda b: collate_window_batch(b, tok.pad_token_id))
    ev = m.evaluate(loader)["summary"]
    true_n = sum(len(p.tokens) for p in pieces)
    print(f"[nll rel={rel}] evaluate={ev['nll_per_token']:.6f} manual={manual_total/n:.6f} n_tokens={ev['n_tokens']} reales={true_n}")
    # 5) _run_epoch (usada para early stopping) == evaluate con label_smoothing=0
    re = m._run_epoch(loader, train=False)
    print(f"[run_epoch rel={rel}] run_epoch nll={re['nll_per_token']:.6f}")

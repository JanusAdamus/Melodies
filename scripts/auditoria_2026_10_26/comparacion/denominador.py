"""Mismos tokens y mismo denominador por familia en una corrida; estimando del titular vs del IC."""
import sys, pandas as pd, numpy as np, json
run = sys.argv[1]
pm = pd.read_csv(f"{run}/piece_metrics_raw.csv")
raw = pd.read_csv(f"{run}/results_raw.csv")
pm = pm[pm.frac == 1.0]
ntok = pm.pivot_table(index="piece_id", columns="model", values="n_tokens", aggfunc="first")
print("n_tokens identicos entre familias por pieza:", bool((ntok.nunique(axis=1) == 1).all()), "| total tokens:", ntok.iloc[:, 0].sum())
pm["chk"] = -pm.log_likelihood / pm.n_tokens
print("max |nll - (-ll/n)|:", float((pm.chk - pm.nll_per_token).abs().max()))
for m, g in pm.groupby("model"):
    tw = -g.log_likelihood.sum() / g.n_tokens.sum()
    ww = g.groupby("canonical_work_id").nll_per_token.mean().mean()
    print(f"{m:12} results_raw.test_nll={raw[raw.model==m].test_nll.mean():.6f}  ponderado_por_token={tw:.6f}  media_por_obra={ww:.6f}")

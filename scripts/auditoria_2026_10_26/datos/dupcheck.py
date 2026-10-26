import pandas as pd
a=pd.read_csv("/home/aj/Documentos/Relacionado con Escuela/Melodies/artifacts/outputs/multicorpus_full/analysis/analysis.csv")
c=a.dropna(subset=["finite_log_likelihood"])
d=c[c.duplicated(subset=["n_events","finite_log_likelihood"],keep=False)]
print("exact duplicate sequences (same n_events & finite LL):"); print(d[["source_name","filename","n_events","finite_log_likelihood","hdp_log_likelihood"]].to_string())
c=c.assign(t=c.title.str.lower().str.replace(r"[^a-z]","",regex=True).str[:12])
print(c[c.duplicated("t",keep=False)][["filename","n_events"]].sort_values("filename").to_string())
print("Spearman(gain, n_events)=%.3f"%((c.hdp_log_likelihood-c.finite_log_likelihood).corr(c.n_events,method="spearman")))
print("SymbTr composer '?':",(a[a.source_name=="SymbTr"].composer=="?").sum())

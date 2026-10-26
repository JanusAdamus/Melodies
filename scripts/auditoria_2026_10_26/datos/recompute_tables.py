import pandas as pd, numpy as np
M="/home/aj/Documentos/Relacionado con Escuela/Melodies/artifacts/outputs/"
a=pd.read_csv(M+"multicorpus_full/analysis/analysis.csv")
cat=pd.read_csv(M+"multicorpus_full/catalog/catalog.csv")
print("catalog rows",len(cat), cat.groupby("source_name").size().to_dict())
print("catalog errors:", cat[cat.error.fillna("")!=""][["source_name","filename","error"]].to_string())
print("analysis rows",len(a), a.groupby("source_name").size().to_dict())
print("analysis errors:", a[a.error.notna()][["source_name","filename","error"]].to_string())
missing=set(cat.path)-set(a.path); print("in catalog not in analysis:",missing)
c=a.dropna(subset=["finite_log_likelihood","hdp_log_likelihood"]).copy()
c["gain"]=c.hdp_log_likelihood-c.finite_log_likelihood
def row(d): return [len(d),d.gain.mean(),d.gain.median(),d.finite_log_likelihood.mean(),d.hdp_log_likelihood.mean(),d.finite_active_states.mean(),d.hdp_effective_states.mean(), (d.gain>0).sum(), (d.gain<0).sum()]
for name,d in [("Global",c)]+list(c.groupby("source_name")):
    r=row(d); print(name, r[0], " ".join(f"{x:.3f}" for x in r[1:7]), "mejora",r[7],"empeora",r[8])
# per-token
c["gain_per_event"]=c.gain/c.n_events
print("gain/event mean %.4f median %.4f"%(c.gain_per_event.mean(),c.gain_per_event.median()))
print("note_count==n_events:", (c.note_count==c.n_events).mean(), "rows differ:", (c.note_count!=c.n_events).sum())
print(c.loc[c.note_count!=c.n_events,["filename","note_count","n_events","part_count"]].head(10).to_string())
print("finite states describe", c.finite_active_states.describe().to_dict())
print("hdp states describe", c.hdp_effective_states.describe().to_dict())
print("hdp eff==20 (K cap) count", (c.hdp_effective_states>=20).sum())

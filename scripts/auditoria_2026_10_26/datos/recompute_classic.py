import pandas as pd
M="/home/aj/Documentos/Relacionado con Escuela/Melodies/artifacts/outputs/classic_limited_eval_refresh/"
cat=pd.read_csv(M+"catalog/catalog.csv"); a=pd.read_csv(M+"analysis/analysis.csv")
print(cat[["filename","title","composer","note_count","measure_count"]].to_string())
c=a.dropna(subset=["finite_log_likelihood","hdp_log_likelihood"]).copy(); c["gain"]=c.hdp_log_likelihood-c.finite_log_likelihood
print(c.sort_values("gain")[["title","note_count","finite_log_likelihood","hdp_log_likelihood","gain","finite_active_states","hdp_effective_states"]].to_string())
print("n",len(c),"mean %.3f median %.4f fin %.3f hdp %.3f Kf %.3f Kh %.3f"%(c.gain.mean(),c.gain.median(),c.finite_log_likelihood.mean(),c.hdp_log_likelihood.mean(),c.finite_active_states.mean(),c.hdp_effective_states.mean()))
mc=pd.read_csv("/home/aj/Documentos/Relacionado con Escuela/Melodies/artifacts/outputs/multicorpus_full/analysis/analysis.csv")
m=mc.merge(c[["filename","finite_log_likelihood","hdp_log_likelihood"]],on="filename",suffixes=("_mc","_cl"))
print(m[["filename","finite_log_likelihood_mc","finite_log_likelihood_cl","hdp_log_likelihood_mc","hdp_log_likelihood_cl"]].to_string())

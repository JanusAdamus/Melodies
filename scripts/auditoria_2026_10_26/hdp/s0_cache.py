import pickle, warnings
warnings.filterwarnings("ignore")
from common import *
out = []
for i, r in enumerate(rows()):
    obs = load(r["path"])
    out.append(dict(i=i, path=r["path"], tokens=obs.tokens.astype(int), fin=float(r["finite_log_likelihood"]), hdp=float(r["hdp_log_likelihood"]), K=float(r["hdp_effective_states"]), finK=float(r["finite_active_states"]), vocab=obs.vocabulary))
pickle.dump(out, open("obs_cache.pkl", "wb"))
print(len(out), sum(len(o["tokens"]) for o in out))

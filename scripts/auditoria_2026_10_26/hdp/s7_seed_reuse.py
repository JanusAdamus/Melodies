import pickle, warnings; warnings.filterwarnings("ignore")
from common import REPO
from src.data.observations import ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM
o = pickle.load(open("obs_cache.pkl", "rb"))[50]
s = ObservationSequence("pitch_class", o["tokens"], o["vocab"], [""]*len(o["tokens"]), [], {})
m = TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=7)
a, b = m.fit(s), m.fit(s)
print("misma instancia, seed=7: fit1", round(a.log_likelihood, 3), a.effective_states, "| fit2", round(b.log_likelihood, 3), b.effective_states)

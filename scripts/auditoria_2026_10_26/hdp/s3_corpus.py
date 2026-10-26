"""Por obra: seleccion por maximo, burn-in, K efectivo, medias posteriores, beta, baseline unigram, held-out."""
import pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from common import REPO
from src.data.observations import ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM
from src.models.finite_hmm import FiniteChordHMM
from src.models.inference import scaled_forward_log_likelihood as LL

data = pickle.load(open("obs_cache.pkl", "rb"))
fin = FiniteChordHMM()
def seqobj(t, vocab): return ObservationSequence("pitch_class", t, vocab, [""]*len(t), [], {})

def fit(t, vocab, seed=7):
    m = TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=seed)
    betas = []
    orig = m._update_beta
    def wrapped(beta, *a):
        out = orig(beta, *a); betas.append((beta.copy(), out[0].copy())); return out
    m._update_beta = wrapped
    return m.fit(seqobj(t, vocab)), betas

rows = []
for o in data:
    t, V = o["tokens"], len(o["vocab"])
    res, betas = fit(t, V and o["vocab"])
    d = res.diagnostics; h = np.array(d.log_likelihood_history); act = np.array(d.active_state_history)
    post = h[40:]
    ll_pm = LL(res.posterior_initial_mean, res.posterior_transition_mean, res.posterior_emission_mean, t)
    occ = np.bincount(res.latent_states, minlength=20) / len(t)
    beta_changed_after_first = max(np.abs(b1 - b0).max() for b0, b1 in betas[1:])
    beta_change_first = np.abs(betas[0][1] - betas[0][0]).max()
    cnt = np.bincount(t, minlength=12); p = cnt[cnt > 0] / len(t)
    uni = float((cnt[cnt > 0] * np.log(p)).sum())
    fin_ll = LL(fin.initial_probs, fin.transition_matrix, fin.emission_matrix, t)
    # held-out: entrenar en 80% inicial, evaluar log p(test | train)
    n_tr = int(0.8 * len(t)); tr = t[:n_tr]
    rtr, _ = fit(tr, o["vocab"])
    ho_hdp = LL(rtr.initial_probs, rtr.transition_matrix, rtr.emission_matrix, t) - LL(rtr.initial_probs, rtr.transition_matrix, rtr.emission_matrix, tr)
    ho_hdp_pm = LL(rtr.posterior_initial_mean, rtr.posterior_transition_mean, rtr.posterior_emission_mean, t) - LL(rtr.posterior_initial_mean, rtr.posterior_transition_mean, rtr.posterior_emission_mean, tr)
    ho_fin = LL(fin.initial_probs, fin.transition_matrix, fin.emission_matrix, t) - LL(fin.initial_probs, fin.transition_matrix, fin.emission_matrix, tr)
    c2 = np.bincount(tr, minlength=12) + 0.5; q = c2 / c2.sum()
    ho_uni = float(np.log(q[t[n_tr:]]).sum())
    rows.append(dict(i=o["i"], T=len(t), fin=fin_ll, best=res.log_likelihood, best_it=d.best_iteration,
        post_mean=post.mean(), post_max=post.max(), ll_pm=ll_pm, K=res.effective_states, K_post=act[40:].mean(),
        K_occ2=int((occ >= 0.02).sum()), K_singleton=int((np.bincount(res.latent_states, minlength=20) == 1).sum()),
        uni=uni, dbeta1=beta_change_first, dbeta_rest=beta_changed_after_first,
        ho_n=len(t)-n_tr, ho_hdp=ho_hdp, ho_hdp_pm=ho_hdp_pm, ho_fin=ho_fin, ho_uni=ho_uni))
    print(o["i"], end=" ", flush=True)
pickle.dump(rows, open("s3_rows.pkl", "wb"))

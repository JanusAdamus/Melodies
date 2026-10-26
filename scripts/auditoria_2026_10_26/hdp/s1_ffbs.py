"""FFBS / forward vs enumeracion exacta."""
import itertools, sys
import numpy as np
sys.path.insert(0, "/home/aj/Documentos/Relacionado con Escuela/Melodies-auditoria")
from src.models.inference import ffbs_sample_batch, forward_log_likelihood, scaled_forward_log_likelihood
from scipy.special import logsumexp

rng = np.random.default_rng(0)
K, V = 3, 4
pi = rng.dirichlet(np.ones(K)); A = rng.dirichlet(np.ones(K), size=K); B = rng.dirichlet(np.ones(V), size=K)
seqs = [rng.integers(0, V, size=n) for n in (5, 3, 1, 4)]

def exact(y):
    paths = list(itertools.product(range(K), repeat=len(y)))
    lp = []
    for z in paths:
        v = np.log(pi[z[0]]) + np.log(B[z[0], y[0]])
        for t in range(1, len(y)):
            v += np.log(A[z[t-1], z[t]]) + np.log(B[z[t], y[t]])
        lp.append(v)
    lp = np.array(lp); return paths, lp, logsumexp(lp)

N = 40000
counts = [dict() for _ in seqs]
for it in range(N):
    st, ll = ffbs_sample_batch(pi, A, B, seqs, rng)
    for i, s in enumerate(st):
        counts[i][tuple(s)] = counts[i].get(tuple(s), 0) + 1
for i, y in enumerate(seqs):
    paths, lp, Z = exact(y)
    p = np.exp(lp - Z)
    emp = np.array([counts[i].get(z, 0) for z in paths]) / N
    tv = 0.5 * np.abs(emp - p).sum()
    print(f"seq{i} T={len(y)} logZ exact={Z:.10f} batch={ll[i]:.10f} forward={forward_log_likelihood(pi,A,B,y)[0]:.10f} scaled={scaled_forward_log_likelihood(pi,A,B,y):.10f} TV(emp,exact)={tv:.4f} (MC noise ~{np.sqrt(len(paths)/N)/2:.4f})")

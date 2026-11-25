"""
Optimal-k search for the equal-weight cardinality portfolio  f_k(x) = lam/(2k^2) x'Sigma x - mu'x/k,  x in {0,1}^N, sum(x)=k
Stage A: Monte-Carlo penalty P_k for EVERY k=1..N (Algorithm 1 of the paper, vectorised exactly like your cell 5.3)
Stage B: best f_k for every k with a fast fixed-cardinality swap annealer (numba, parallel restarts)
Stage C: k* = argmin_k f_k  (+ parsimonious k within EPS of the minimum)
Usage in the notebook (after mu, Sigma, LAMBDA exist):
    from k_sweep import run_k_sweep;  res = run_k_sweep(mu, Sigma, LAMBDA)
"""
import time, numpy as np
from numba import njit, prange

# ---------------- Stage A: Monte-Carlo penalty for all k ----------------
def mc_penalty_all_k(mu, Sigma, lam, ks, mc_n=5000, mc_runs=50, seed=7):
    N = len(mu); rng = np.random.default_rng(seed); P = np.zeros((mc_runs, len(ks)))
    for r in range(mc_runs):
        card = rng.integers(0, N + 1, size=mc_n)
        rank = np.argsort(np.argsort(rng.random((mc_n, N)), axis=1), axis=1)
        X = (rank < card[:, None]).astype(float)
        q, lin, s = ((X @ Sigma) * X).sum(1), X @ mu, X.sum(1)
        for j, k in enumerate(ks):
            f = 0.5 * lam / k**2 * q - lin / k; viol = (s - k) ** 2; vmin = viol.min()
            ff = f[viol == vmin].min(); m = (f < ff) & (viol > vmin)
            P[r, j] = ((ff - f[m]) / (viol[m] - vmin)).max() if m.any() else 0.0
    return P.max(axis=0)

# ---------------- Stage B: swap annealer at fixed k (penalty-free, O(1) per proposal) ----------------
@njit(cache=True)
def _anneal(mu, Sg, lam, k, sweeps, T0, T1, seed):
    np.random.seed(seed); N = len(mu)
    perm = np.random.permutation(N); x = np.zeros(N, np.int8); x[perm[:k]] = 1
    g = np.zeros(N)
    for i in range(N):
        if x[i]:
            g += Sg[:, i]
    q = 0.0; lin = 0.0
    for i in range(N):
        if x[i]:
            q += g[i]; lin += mu[i]
    f = 0.5 * lam / k**2 * q - lin / k; best = f; bx = x.copy()
    inn = np.flatnonzero(x == 1); out = np.flatnonzero(x == 0)
    if len(out) == 0:
        return bx, best
    for sw in range(sweeps):
        T = T0 * (T1 / T0) ** (sw / max(sweeps - 1, 1))
        for _ in range(N):
            ia = np.random.randint(len(inn)); ib = np.random.randint(len(out)); a = inn[ia]; b = out[ib]
            dq = -2 * g[a] + 2 * g[b] + Sg[a, a] + Sg[b, b] - 2 * Sg[a, b]
            df = 0.5 * lam / k**2 * dq - (mu[b] - mu[a]) / k
            if df <= 0 or np.random.random() < np.exp(-df / T):
                g += Sg[:, b] - Sg[:, a]; x[a] = 0; x[b] = 1; inn[ia] = b; out[ib] = a; f += df
                if f < best - 1e-15:
                    best = f; bx = x.copy()
    return bx, best

@njit(parallel=True, cache=True)
def _best_of_restarts(mu, Sg, lam, k, sweeps, T0, T1, seeds):
    R = len(seeds); N = len(mu); X = np.zeros((R, N), np.int8); F = np.zeros(R)
    for r in prange(R):
        X[r], F[r] = _anneal(mu, Sg, lam, k, sweeps, T0, T1, seeds[r])
    return X, F

def solve_k(mu, Sigma, lam, k, restarts=32, sweeps=300, seed=0):
    N = len(mu)
    if k >= N:
        x = np.ones(N, np.int8); w = x / N
        return x, 0.5 * lam * w @ Sigma @ w - mu @ w
    # temperature scale: typical |df| of a random swap at this k
    rng = np.random.default_rng(seed); dfs = []
    for _ in range(200):
        S = rng.permutation(N); a, b = S[0], S[k % N] if k < N else S[0]
        w = np.zeros(N); w[S[:k]] = 1 / k; w2 = w.copy(); w2[S[0]] = 0; w2[S[k]] = 1 / k
        dfs.append(abs((0.5 * lam * w2 @ Sigma @ w2 - mu @ w2) - (0.5 * lam * w @ Sigma @ w - mu @ w)))
    T0 = max(np.median(dfs), 1e-12); T1 = T0 * 1e-3
    X, F = _best_of_restarts(mu, np.ascontiguousarray(Sigma), lam, k, sweeps, T0, T1,
                             (np.arange(restarts) + 1000 * k + seed).astype(np.int64))
    j = int(np.argmin(F)); return X[j], float(F[j])

def run_k_sweep(mu, Sigma, lam, ks=None, restarts=32, sweeps=300, eps=1e-4, with_penalty=True):
    N = len(mu); ks = list(range(1, N + 1)) if ks is None else list(ks); t0 = time.time()
    P = mc_penalty_all_k(mu, Sigma, lam, ks) if with_penalty else None
    if with_penalty: print(f"Stage A penalties for {len(ks)} k values: {time.time()-t0:.1f} s")
    rows = []; t1 = time.time()
    for k in ks:
        x, f = solve_k(mu, Sigma, lam, k, restarts, sweeps)
        w = x / x.sum(); r = float(mu @ w); v = float(w @ Sigma @ w)
        rows.append(dict(k=k, objective=f, ret=r, vol=v**.5, sharpe=r / v**.5, P=None if P is None else P[ks.index(k)], x=x))
    print(f"Stage B annealing for {len(ks)} k values: {time.time()-t1:.1f} s")
    import pandas as pd
    df = pd.DataFrame(rows).set_index("k"); fmin = df.objective.min()
    k_star = int(df.objective.idxmin()); k_pars = int(df.index[df.objective <= fmin + eps][0])
    print(f"k* = {k_star} (objective {fmin:.6f}); smallest k within {eps:g} of the minimum: {k_pars}")
    return df, k_star, k_pars

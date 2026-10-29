"""Numerical checks for the extensions of the model (manuscript Sec. VIII, added 2026-10-28).

(1) Theorem 2 (one-step channels): with Kraus operators K_{k,j} of the k-th step, P_kappa equals the sum over
    Kraus-label sequences j of the pair sum with amplitudes phi_j[x] = prod_k <x_k|K_{k,j_k}|x_{k-1}>.
(2) Theorem 3 (records in epoch-dependent bases): the pair formula with mixed-basis paths equals the channel evolution.
(3) Proposition 8 (two records of one epoch in bases B, B'): the composite channel is order independent when B = B'
    and when B, B' are mutually unbiased with injective resolutions (the product of the two dephasings is the
    completely depolarising map); it is order dependent for generic bases and for coarse resolutions.
(4) Proposition 7 (near-Hadamard accessibilities): TV(P_c, uniform) <= (1 - 1/N) prod_k delta_k with
    delta_k = N max_{x,y} |M_k(x|y) - 1/N|, via the Dobrushin contraction TV(M p, M q) <= delta TV(p, q).
(5) Proposition 9 (local records on a product set): the pair factor is prod_i kappa_i^{d_{c_i}} and the latitude bound
    is 1 - prod_i (1 - kappa_i)^{|R|}.
(6) Proposition 10 (growth of the set of alternatives): with X in X' and compatible records,
    TV(P'_kappa, P_kappa (+) 0) <= sum_k eps_k, eps_k = ||W'_k iota - iota W_k|| (operator norm).
Exit code 0 when every check passes."""
import itertools, sys
import numpy as np
from scipy.linalg import expm, logm
rng = np.random.default_rng(20261028)

def haar(N):
    z = (rng.normal(size=(N, N)) + 1j*rng.normal(size=(N, N)))/np.sqrt(2); q, r = np.linalg.qr(z); d = np.diag(r); return q*(d/abs(d))
def dft(N):
    x = np.arange(N); return np.exp(2j*np.pi*np.outer(x, x)/N)/np.sqrt(N)
def Delta_basis(rho, V, c):
    """Dephase in the basis given by the columns of V with the resolution c on basis labels."""
    out = np.zeros_like(rho)
    for a in set(c):
        P = sum(np.outer(V[:, b], V[:, b].conj()) for b in range(V.shape[1]) if c[b] == a); out += P @ rho @ P
    return out
def D_basis(rho, V, c, kappa): return kappa*rho + (1-kappa)*Delta_basis(rho, V, c)
def tv(p, q): return 0.5*abs(np.asarray(p) - np.asarray(q)).sum()
fails = []

# ---------- (1) Theorem 2: one-step channels ----------
def random_channel(N, r):
    """r Kraus operators of a random CPTP map: K_j = (U_j D_j) normalised so that sum K_j^dag K_j = 1."""
    Ks = [rng.normal(size=(N, N)) + 1j*rng.normal(size=(N, N)) for _ in range(r)]
    S = sum(K.conj().T @ K for K in Ks); w, v = np.linalg.eigh(S); Sm = v @ np.diag(w**-0.5) @ v.conj().T
    return [K @ Sm for K in Ks]
w1 = 0.0
for trial in range(6):
    N, T, r = 3, 3, 2; Ks = [random_channel(N, r) for _ in range(T)]
    c = [0, 0, 1] if trial % 2 else list(range(N)); kappa = rng.uniform(); x0 = int(rng.integers(N)); R = list(range(1, T))
    rho = np.zeros((N, N), complex); rho[x0, x0] = 1
    for k in range(T):
        rho = sum(K @ rho @ K.conj().T for K in Ks[k])
        if k in R: rho = D_basis(rho, np.eye(N), c, kappa)
    P_chan = np.diag(rho).real
    P_pair = np.zeros(N)
    for js in itertools.product(range(r), repeat=T):
        amps = {}
        for p in itertools.product(range(N), repeat=T):
            a = 1.0+0j; prev = x0
            for k, x in enumerate(p): a *= Ks[k][js[k]][x, prev]; prev = x
            amps[p] = a
        for xf in range(N):
            ps = [p for p in amps if p[-1] == xf]
            P_pair[xf] += sum((kappa**sum(1 for t in R if c[p[t]] != c[q[t]]))*amps[p]*np.conj(amps[q]) for p in ps for q in ps).real
    w1 = max(w1, abs(P_chan - P_pair).max())
print(f"(1) Theorem 2 (channels): max error {w1:.2e}"); fails += [w1 > 1e-10]

# ---------- (2) Theorem 3: epoch-dependent bases ----------
w2 = 0.0
for trial in range(8):
    N, T = 3, 3; W = [haar(N) @ np.diag(np.exp(1j*rng.uniform(0, 2*np.pi, N))) for _ in range(T)]
    R = [1, 2]; V = {t: (dft(N) if t == 1 else haar(N)) for t in R}            # basis of the record at epoch t
    cs = {t: (list(range(N)) if trial % 2 else [0, 0, 1]) for t in R}; kap = {t: rng.uniform() for t in R}
    x0 = int(rng.integers(N))
    rho = np.zeros((N, N), complex); rho[x0, x0] = 1
    for k in range(T):
        rho = W[k] @ rho @ W[k].conj().T
        if k in R: rho = D_basis(rho, V[k], cs[k], kap[k])
    P_chan = np.diag(rho).real
    # mixed-basis paths: label b_k in the basis V_k after step k for every k in R; final projection on x_f
    P_pair = np.zeros(N)
    for xf in range(N):
        amps = {}
        for b in itertools.product(range(N), repeat=len(R)):
            vec = np.zeros(N, complex); vec[x0] = 1; a = 1.0+0j
            for k in range(T):
                vec = W[k] @ vec
                if k in R:
                    idx = R.index(k); a *= np.vdot(V[k][:, b[idx]], vec); vec = V[k][:, b[idx]].copy()
            a *= vec[xf]; amps[b] = a
        P_pair[xf] = sum((np.prod([kap[t]**(cs[t][b[i]] != cs[t][bb[i]]) for i, t in enumerate(R)]))*amps[b]*np.conj(amps[bb]) for b in amps for bb in amps).real
    w2 = max(w2, abs(P_chan - P_pair).max())
print(f"(2) Theorem 3 (bases per epoch): max error {w2:.2e}"); fails += [w2 > 1e-10]

# ---------- (3) Proposition 8: two frames at one epoch ----------
def commutator_norm(V1, c1, k1, V2, c2, k2, N, trials=20):
    worst = 0.0
    for _ in range(trials):
        psi = rng.normal(size=N) + 1j*rng.normal(size=N); psi /= np.linalg.norm(psi); rho = np.outer(psi, psi.conj())
        a = D_basis(D_basis(rho, V2, c2, k2), V1, c1, k1); b = D_basis(D_basis(rho, V1, c1, k1), V2, c2, k2)
        worst = max(worst, abs(a - b).max())
    return worst
rep = []
for N in (2, 3, 4, 5):
    I, F = np.eye(N), dft(N); inj = list(range(N)); k1, k2 = rng.uniform(size=2)
    # depolarising product for MUB with injective resolutions
    psi = rng.normal(size=N) + 1j*rng.normal(size=N); psi /= np.linalg.norm(psi); rho = np.outer(psi, psi.conj())
    dep = abs(Delta_basis(Delta_basis(rho, I, inj), F, inj) - np.eye(N)/N).max()
    same = commutator_norm(I, inj, k1, I, inj, k2, N); mub = commutator_norm(I, inj, k1, F, inj, k2, N)
    gen = commutator_norm(I, inj, k1, haar(N), inj, k2, N)
    coarse = commutator_norm(I, [0]*(N-1) + [1], k1, F, inj, k2, N) if N > 2 else 0.0
    rep.append((N, dep, same, mub, gen, coarse))
    fails += [dep > 1e-12, same > 1e-12, mub > 1e-12, gen < 1e-6]
for N, dep, same, mub, gen, coarse in rep:
    print(f"(3) N={N}: |Delta_F Delta_I rho - 1/N| = {dep:.1e}; commutator: same basis {same:.1e}, MUB injective {mub:.1e}, generic basis {gen:.3f}, MUB coarse {coarse:.3f}")

# ---------- (4) Proposition 7: near-Hadamard ----------
def markov(A): return abs(A)**2
viol = 0; worst_ratio = 0.0; cases = 0
N = 4; F = dft(N); L = logm(F)
for theta in np.linspace(0.05, 1.0, 12):
    for T in (1, 2, 3, 5):
        A = expm(theta*L)                      # interpolates the identity (theta=0) and the DFT (theta=1)
        Ms = [markov(A @ np.diag(np.exp(1j*rng.uniform(0, 2*np.pi, N)))) for _ in range(T)]
        deltas = [N*abs(M - 1/N).max() for M in Ms]
        p = np.zeros(N); p[0] = 1
        for M in Ms: p = M @ p
        bound = (1 - 1/N)*np.prod(deltas); t = tv(p, np.full(N, 1/N))
        viol += t > bound + 1e-12; worst_ratio = max(worst_ratio, t/bound if bound > 1e-6 else 0); cases += 1
# Dobrushin step: TV(Mp, Mq) <= delta TV(p, q) for random doubly stochastic M from unitaries
for _ in range(300):
    N = int(rng.integers(2, 6)); A = haar(N); M = markov(A); delta = N*abs(M - 1/N).max()
    p = rng.dirichlet(np.ones(N)); q = rng.dirichlet(np.ones(N))
    viol += tv(M @ p, M @ q) > delta*tv(p, q) + 1e-12; cases += 1
print(f"(4) Proposition 7: violations {viol}/{cases}, max TV/bound on the interpolating family {worst_ratio:.3f}"); fails += [viol > 0]

# ---------- (5) Proposition 9: product sets with local records ----------
w5 = 0.0; viol5 = 0
s1, s2 = 2, 3; N = s1*s2; T = 3; R = [1, 2]
c1 = list(range(s1)); c2 = list(range(s2))
for trial in range(6):
    W = [haar(N) @ np.diag(np.exp(1j*rng.uniform(0, 2*np.pi, N))) for _ in range(T)]
    k1, k2 = rng.uniform(size=2); x0 = int(rng.integers(N))
    f1 = [x // s2 for x in range(N)]; f2 = [x % s2 for x in range(N)]          # factors of x = (x1, x2)
    rho = np.zeros((N, N), complex); rho[x0, x0] = 1
    for k in range(T):
        rho = W[k] @ rho @ W[k].conj().T
        if k in R: rho = D_basis(D_basis(rho, np.eye(N), [c1[f1[x]] for x in range(N)], k1), np.eye(N), [c2[f2[x]] for x in range(N)], k2)
    P_chan = np.diag(rho).real
    amps = {}
    for p in itertools.product(range(N), repeat=T):
        a = 1.0+0j; prev = x0
        for k, x in enumerate(p): a *= W[k][x, prev]; prev = x
        amps[p] = a
    P_pair = np.zeros(N)
    for xf in range(N):
        ps = [p for p in amps if p[-1] == xf]
        P_pair[xf] = sum((k1**sum(1 for t in R if f1[p[t]] != f1[q[t]]))*(k2**sum(1 for t in R if f2[p[t]] != f2[q[t]]))*amps[p]*np.conj(amps[q]) for p in ps for q in ps).real
    w5 = max(w5, abs(P_chan - P_pair).max())
    # latitude bound against the fully recorded (kappa = 0) distribution
    rho0 = np.zeros((N, N), complex); rho0[x0, x0] = 1
    for k in range(T):
        rho0 = W[k] @ rho0 @ W[k].conj().T
        if k in R: rho0 = np.diag(np.diag(rho0))
    viol5 += tv(P_chan, np.diag(rho0).real) > 1 - ((1-k1)*(1-k2))**len(R) + 1e-12
print(f"(5) Proposition 9 (product set): max error {w5:.2e}, latitude-bound violations {viol5}/6"); fails += [w5 > 1e-10, viol5 > 0]

# ---------- (6) Proposition 10: growth of the set of alternatives ----------
viol6 = 0; ratios = []
for trial in range(40):
    N, Nn, T = 4, 2, 4; Np = N + Nn; R = [1, 3]; c = [0, 0, 1, 1]; cp = c + [2, 2]   # new alternatives get new categories
    kappa = rng.uniform(); x0 = int(rng.integers(N)); eps_scale = rng.uniform(0.0, 0.6)
    W, Wp, eps = [], [], []
    for k in range(T):
        Wk = haar(N); Uk = haar(Nn); K = rng.normal(size=(Np, Np)) + 1j*rng.normal(size=(Np, Np)); K = (K + K.conj().T)/2
        coupling = expm(1j*eps_scale*K/np.linalg.norm(K, 2))
        Wpk = coupling @ np.block([[Wk, np.zeros((N, Nn))], [np.zeros((Nn, N)), Uk]])
        iota = np.vstack([np.eye(N), np.zeros((Nn, N))])
        W.append(Wk); Wp.append(Wpk); eps.append(np.linalg.norm(Wpk @ iota - iota @ Wk, 2))
    def evolve(Ws, cc, M):
        rho = np.zeros((M, M), complex); rho[x0, x0] = 1
        for k in range(T):
            rho = Ws[k] @ rho @ Ws[k].conj().T
            if k in R: rho = D_basis(rho, np.eye(M), cc, kappa)
        return np.diag(rho).real
    P = np.concatenate([evolve(W, c, N), np.zeros(Nn)]); Pp = evolve(Wp, cp, Np)
    t = tv(Pp, P); bound = sum(eps); viol6 += t > bound + 1e-12; ratios.append(t/bound if bound > 0 else 0)
print(f"(6) Proposition 10 (growth): violations {viol6}/40, max TV/bound {max(ratios):.3f}"); fails += [viol6 > 0]

print("ALL PASS" if not any(fails) else "FAIL")
sys.exit(0 if not any(fails) else 1)

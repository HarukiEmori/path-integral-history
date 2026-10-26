"""Independent check (reviewer) of Theorem 1, Corollaries 1-4, Eq. (channel), Eq. (twostate),
the chirp kernel and the Gaussian overlap of 'A Path Integral Model of History'.
Run: ~/Develop/venv/quantum-env/bin/python check_recording.py
"""
import itertools
import numpy as np

rng = np.random.default_rng(7)
TOL = {}
def rec(name, err):
    TOL[name] = max(TOL.get(name, 0.0), float(err))

def rand_unitary(n):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))

def complete_unitary(cols, dim):
    """Complete an orthonormal family of column vectors to a unitary (Gram-Schmidt on random vectors)."""
    B = list(cols)
    while len(B) < dim:
        v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        for b in B:
            v = v - (b.conj() @ v) * b
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            B.append(v / nv)
    U = np.array(B).T
    assert np.allclose(U.conj().T @ U, np.eye(dim), atol=1e-10)
    return U

def make_model(N, T):
    A = [rand_unitary(N) for _ in range(T)]
    beta = rng.uniform(0, 2, size=T)
    v = [rng.normal(size=N) for _ in range(T)]
    return A, beta, v

def Ws(A, beta, v):
    return [A[k] @ np.diag(np.exp(1j * beta[k] * v[k])) for k in range(len(A))]

def pointers(ncat, kappa):
    """eta_a = sqrt(k)|0> + sqrt(1-k)|a>, probe dim ncat+1."""
    E = np.zeros((ncat, ncat + 1), complex)
    for a in range(ncat):
        E[a, 0] = np.sqrt(kappa); E[a, a + 1] = np.sqrt(1 - kappa)
    return E

def explicit_sim(W, xi, c, ncat, R, kappa, return_state=False):
    """Explicit system (x) probes simulation with a genuine unitary U_rec on H (x) H_P."""
    N = W[0].shape[0]; T = len(W)
    d = ncat + 1
    E = pointers(ncat, kappa)
    # U_rec : |x>|0> -> |x>|eta_c(x)>, completed to a unitary on C^N (x) C^d (index x*d+p)
    cols = []
    for x in range(N):
        vcol = np.zeros(N * d, complex)
        vcol[x * d:(x + 1) * d] = E[c[x]]
        cols.append(vcol)
    Ufull = complete_unitary(cols, N * d)
    # reorder so that column x*d+0 is the image of |x>|0>
    U = np.zeros((N * d, N * d), complex)
    rest = [j for j in range(N * d) if j % d != 0]
    for x in range(N):
        U[:, x * d] = Ufull[:, x]
    for i, j in enumerate(rest):
        U[:, j] = Ufull[:, N + i]
    Rl = sorted(R)
    nP = len(Rl)
    shape = [N] + [d] * nP
    psi = np.zeros(shape, complex)
    psi[(xi,) + (0,) * nP] = 1.0
    for k in range(1, T + 1):
        psi = np.tensordot(W[k - 1], psi, axes=([1], [0]))
        if k in R:
            j = Rl.index(k) + 1
            psi = np.moveaxis(psi, j, 1)                      # (N, d, rest...)
            sh = psi.shape
            psi = (U @ psi.reshape(N * d, -1)).reshape(sh)
            psi = np.moveaxis(psi, 1, j)
    if return_state:
        return psi
    return np.sum(abs(psi) ** 2, axis=tuple(range(1, nP + 1)))

def path_amp(W, path):
    a = 1.0 + 0j
    for k in range(1, len(path)):
        a *= W[k - 1][path[k], path[k - 1]]
    return a

def formula(W, xi, c, R, kappa):
    N = W[0].shape[0]; T = len(W)
    P = np.zeros(N)
    for xf in range(N):
        paths = [(xi,) + mid + (xf,) for mid in itertools.product(range(N), repeat=T - 1)]
        amps = np.array([path_amp(W, p) for p in paths])
        s = 0j
        for i, p in enumerate(paths):
            for j, q in enumerate(paths):
                dc = sum(1 for t in R if c[p[t]] != c[q[t]])
                s += (kappa ** dc if dc > 0 else 1.0) * amps[i] * np.conj(amps[j])
        P[xf] = s.real
        rec('Thm1: imag part of P_kappa', abs(s.imag))
    return P

def channel_sim(W, xi, c, R, kappa):
    N = W[0].shape[0]
    rho = np.zeros((N, N), complex); rho[xi, xi] = 1
    mask = np.array([[1.0 if c[x] == c[y] else kappa for y in range(N)] for x in range(N)])
    for k in range(1, len(W) + 1):
        rho = W[k - 1] @ rho @ W[k - 1].conj().T
        if k in R:
            rho = rho * mask
    return np.real(np.diag(rho))

def Pq(W, xi):
    M = np.eye(W[0].shape[0])
    for w in W:
        M = w @ M
    return abs(M[:, xi]) ** 2

def Pc(W, xi):
    p = np.zeros(W[0].shape[0]); p[xi] = 1
    for w in W:
        p = (abs(w) ** 2) @ p
    return p

# ---------------- (i) Theorem 1 ----------------
n_cases = 0
for N in (2, 3, 4):
    for T in (2, 3, 4):
        if N ** (2 * (T - 1)) * N > 4 ** 7:   # keep double path sums tractable
            pass
        for cmode in ('injective', 'coarse'):
            if cmode == 'injective':
                c = list(range(N)); ncat = N
            else:
                ncat = max(1, N - 1) if N > 2 else 1
                c = list(rng.permutation([i % ncat for i in range(N)]))
                if ncat > 1:
                    assert len(set(c)) == ncat
            for Rmode in ('full', 'partial'):
                full = set(range(1, T))
                if Rmode == 'full':
                    R = full
                else:
                    R = set(t for t in full if rng.random() < 0.5)
                A, beta, v = make_model(N, T)
                W = Ws(A, beta, v)
                xi = int(rng.integers(N))
                for kappa in (0.0, 0.3, 0.7, 1.0):
                    Pe = explicit_sim(W, xi, c, ncat, R, kappa)
                    Pf = formula(W, xi, c, R, kappa)
                    Pch = channel_sim(W, xi, c, R, kappa)
                    rec('Thm1: explicit probe sim vs pair formula', np.max(abs(Pe - Pf)))
                    rec('Eq.(channel): channel composition vs explicit sim', np.max(abs(Pe - Pch)))
                    rec('Thm1: sum P_kappa - 1', abs(Pe.sum() - 1))
                    rec('Thm1: negativity of P_kappa', max(0, -Pe.min()))
                    if kappa == 1.0:
                        rec('Thm1: kappa=1 vs P_q', np.max(abs(Pe - Pq(W, xi))))
                    if kappa == 0.0 and cmode == 'injective' and Rmode == 'full':
                        rec('Thm1: kappa=0, c inj, R full vs P_c', np.max(abs(Pe - Pc(W, xi))))
                    n_cases += 1
print('Theorem 1 cases:', n_cases)

# Prop 1: zero sum and Hadamard uniformity
for _ in range(30):
    N = int(rng.integers(2, 6)); T = int(rng.integers(1, 5))
    A, beta, v = make_model(N, T); W = Ws(A, beta, v); xi = int(rng.integers(N))
    rec('Prop1: sum_xf I(xf)', abs((Pq(W, xi) - Pc(W, xi)).sum()))
    # Hadamard: DFT with random phases (still Hadamard: D1 F D2)
    F = np.exp(2j * np.pi * np.outer(range(N), range(N)) / N) / np.sqrt(N)
    AH = [np.diag(np.exp(1j * rng.uniform(0, 7, N))) @ F @ np.diag(np.exp(1j * rng.uniform(0, 7, N))) for _ in range(T)]
    WH = Ws(AH, beta, v)
    rec('Prop1: Hadamard P_c - 1/N', np.max(abs(Pc(WH, xi) - 1 / N)))

# Cor 1: valuation independence at kappa=0 (inj, full R) and dependence for coarse c
max_dep_inj = 0; min_dep_coarse = []
for _ in range(20):
    N = 3; T = 3
    A, beta, v = make_model(N, T)
    v2 = [rng.normal(size=N) * 3 for _ in range(T)]
    W1 = Ws(A, beta, v); W2 = Ws(A, beta, v2)
    full = {1, 2}
    max_dep_inj = max(max_dep_inj, np.max(abs(channel_sim(W1, 0, [0, 1, 2], full, 0) - channel_sim(W2, 0, [0, 1, 2], full, 0))))
    min_dep_coarse.append(np.max(abs(channel_sim(W1, 0, [0, 0, 1], full, 0) - channel_sim(W2, 0, [0, 0, 1], full, 0))))
rec('Cor1: valuation dependence of P_0 (c inj, R full)', max_dep_inj)
print('Cor1: valuation dependence of P_0 for coarse c (median over 20) = %.3f' % np.median(min_dep_coarse))
# Cor 1: joint law of the records at kappa=0 is the Markov chain
for _ in range(10):
    N = 3; T = 3; A, beta, v = make_model(N, T); W = Ws(A, beta, v)
    psi = explicit_sim(W, 1, [0, 1, 2], 3, {1, 2}, 0.0, return_state=True)  # shape (N, 4, 4)
    for x1 in range(N):
        for x2 in range(N):
            for xf in range(N):
                pj = abs(psi[xf, x1 + 1, x2 + 1]) ** 2   # pointer basis |a+1> at kappa=0
                pm = abs(A[0][x1, 1]) ** 2 * abs(A[1][x2, x1]) ** 2 * abs(A[2][xf, x2]) ** 2
                rec('Cor1: joint law of records vs Markov chain', abs(pj - pm))

# ---------------- (ii) Corollary 2 ----------------
viol = 0; viol_pair = 0; ratios = []
for i in range(50):
    N = int(rng.integers(2, 5)); T = int(rng.integers(2, 5))
    A, beta, v = make_model(N, T); W = Ws(A, beta, v)
    kappa = float(rng.uniform(0, 1))
    if rng.random() < 0.5:
        c = list(range(N))
    else:
        c = list(rng.integers(0, max(1, N - 1), size=N))
    R = set(t for t in range(1, T) if rng.random() < 0.7) or {1}
    xi = int(rng.integers(N))
    Pk = channel_sim(W, xi, c, R, kappa); P0 = channel_sim(W, xi, c, R, 0.0)
    tv = 0.5 * abs(Pk - P0).sum(); bound = 1 - (1 - kappa) ** len(R)
    ratios.append(tv / bound if bound > 0 else 0)
    if tv > bound + 1e-12:
        viol += 1
    # pairs of valuations, injective c and full R: TV(P^v, P^v') <= bound
    Rf = set(range(1, T)); cinj = list(range(N))
    W2 = Ws(A, beta, [rng.normal(size=N) * 5 for _ in range(T)])
    tvp = 0.5 * abs(channel_sim(W, xi, cinj, Rf, kappa) - channel_sim(W2, xi, cinj, Rf, kappa)).sum()
    if tvp > 1 - (1 - kappa) ** len(Rf) + 1e-12:
        viol_pair += 1
print('Cor2: violations %d / 50 ; pair-of-valuations violations %d / 50 ; max TV/bound = %.3f' % (viol, viol_pair, max(ratios)))
# mixture representation Eq. (mixture)
for _ in range(20):
    N = 3; T = 4; A, beta, v = make_model(N, T); W = Ws(A, beta, v)
    c = [0, 0, 1]; R = {1, 2, 3}; kappa = float(rng.uniform())
    mix = np.zeros(N)
    for r in range(len(R) + 1):
        for S in itertools.combinations(sorted(R), r):
            # identity on S, Delta_c on R \ S  == channel_sim with kappa=1 on S and 0 elsewhere
            rho = np.zeros((N, N), complex); rho[0, 0] = 1
            mask0 = np.array([[1.0 if c[x] == c[y] else 0 for y in range(N)] for x in range(N)])
            for k in range(1, T + 1):
                rho = W[k - 1] @ rho @ W[k - 1].conj().T
                if k in R and k not in S:
                    rho = rho * mask0
            mix += kappa ** len(S) * (1 - kappa) ** (len(R) - len(S)) * np.real(np.diag(rho))
    rec('Cor2: mixture representation Eq.(mixture)', np.max(abs(mix - channel_sim(W, 0, c, R, kappa))))

# ---------------- Corollary 3 ----------------
for _ in range(10):
    N = 3; T = 3; A, beta, v = make_model(N, T); W = Ws(A, beta, v); c = [0, 1, 2]; R = {1, 2}
    ks = np.linspace(0, 1, 7)
    Ps = np.array([channel_sim(W, 0, c, R, k) for k in ks])
    coef = np.polynomial.polynomial.polyfit(ks, Ps, len(R))   # shape (deg+1, N)
    F1 = np.zeros(N)
    for xf in range(N):
        paths = [(0,) + m + (xf,) for m in itertools.product(range(N), repeat=T - 1)]
        for p in paths:
            for q in paths:
                if sum(1 for t in R if c[p[t]] != c[q[t]]) == 1:
                    F1[xf] += (path_amp(W, p) * np.conj(path_amp(W, q))).real
    rec('Cor3: F_1 vs linear coefficient of P_kappa', np.max(abs(coef[1] - F1)))
    rec('Cor3: degree <= |R| (residual of degree-|R| fit)', np.max(abs(np.polynomial.polynomial.polyval(ks, coef).T - Ps)))

# ---------------- Corollary 4 (two generations, exact law of records) ----------------
def record_law(A, v, beta, xi, kappa=0.0):
    """Law of (x_1..x_{T-1}, x_f) at kappa=0, c injective, R full, by explicit probes."""
    W = Ws(A, beta, v); N = W[0].shape[0]; T = len(W)
    psi = explicit_sim(W, xi, list(range(N)), N, set(range(1, T)), kappa, return_state=True)
    law = {}
    for mid in itertools.product(range(N), repeat=T - 1):
        for xf in range(N):
            law[mid + (xf,)] = abs(psi[(xf,) + tuple(m + 1 for m in mid)]) ** 2
    return law
for trial in range(5):
    N = 2; T = 3; A, beta, v = make_model(N, T)
    def G(r):  # reinforcement: raise the valuation of the last recorded configuration
        vv = [vk.copy() for vk in v]
        for vk in vv:
            vk[r[-1]] += 5.0 + 3 * r[0]
        return vv
    L1 = record_law(A, v, beta, 0)
    joint_upd = {}; joint_no = {}
    for r1, p1 in L1.items():
        L2u = record_law(A, G(r1), beta, r1[-1]); L2n = record_law(A, v, beta, r1[-1])
        for r2 in L2u:
            joint_upd[(r1, r2)] = p1 * L2u[r2]; joint_no[(r1, r2)] = p1 * L2n[r2]
    rec('Cor4: law of (r1,r2) with valuation update vs without', max(abs(joint_upd[k] - joint_no[k]) for k in joint_upd))
    # controls: the hypotheses are needed. (a) kappa=0.5, c injective: final law depends on v.
    W1 = Ws(A, beta, v); W2 = Ws(A, beta, [vk + np.array([0, 2.0]) for vk in v])
    print('Cor4 control (a) kappa=0.5, c inj, R full: max |P_k^v - P_k^v\'| = %.3e'
          % np.max(abs(channel_sim(W1, 0, [0, 1], {1, 2}, 0.5) - channel_sim(W2, 0, [0, 1], {1, 2}, 0.5))))
    break

# Cor4 control (b): every epoch recorded sharply (kappa=0, R full) but coarse c -> valuation changes law of records
N = 3; T = 3; A, beta, v = make_model(N, T); c = [0, 0, 1]
W1 = Ws(A, beta, v); W2 = Ws(A, beta, [vk + np.array([0, 2.0, 0]) for vk in v])
print('Cor4 control (b) kappa=0, R full, coarse c: max |P_0^v - P_0^v\'| = %.3e'
      % np.max(abs(channel_sim(W1, 0, c, {1, 2}, 0.0) - channel_sim(W2, 0, c, {1, 2}, 0.0))))

# ---------------- (iii) Eq. (twostate) ----------------
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
for _ in range(200):
    b = rng.uniform(0, 3); v0, v1 = rng.normal(size=2) * 2; kappa = rng.uniform()
    W = Ws([H, H], [b, b], [np.array([v0, v1])] * 2)
    P = explicit_sim(W, 0, [0, 1], 2, {1}, kappa)
    ph = np.cos(b * (v1 - v0))
    rec('Eq.(twostate) closed form', max(abs(P[0] - (0.5 + kappa / 2 * ph)), abs(P[1] - (0.5 - kappa / 2 * ph))))
# visibility and distinguishability
for kappa in np.linspace(0, 1, 11):
    E = pointers(2, kappa)
    rho0 = np.outer(E[0], E[0].conj()); rho1 = np.outer(E[1], E[1].conj())
    D = 0.5 * np.sum(abs(np.linalg.eigvalsh(rho0 - rho1)))
    rec('App.B: trace distance vs sqrt(1-k^2)', abs(D - np.sqrt(1 - kappa ** 2)))

# ---------------- chirp kernel and Gaussian overlap ----------------
for N in (2, 4, 6, 8, 10):
    x = np.arange(N)
    A = np.exp(1j * np.pi * (x[:, None] - x[None, :]) ** 2 / N) / np.sqrt(N)
    rec('Eq.(chirp): unitarity, even N', np.max(abs(A.conj().T @ A - np.eye(N))))
odd = []
for N in (3, 5, 7):
    x = np.arange(N)
    A = np.exp(1j * np.pi * (x[:, None] - x[None, :]) ** 2 / N) / np.sqrt(N)
    odd.append(np.max(abs(A.conj().T @ A - np.eye(N))))
print('Eq.(chirp) with odd N (not claimed): unitarity defect =', np.round(odd, 3))
q = np.linspace(-40, 40, 400001); dq = q[1] - q[0]
for sig, lam, ga, gb in [(1.0, 1.0, 0.0, 1.0), (0.7, 2.0, -1.0, 0.5), (2.0, 0.3, 3.0, -2.0)]:
    psi0 = lambda s: (2 * np.pi * sig ** 2) ** -0.25 * np.exp(-s ** 2 / (4 * sig ** 2))
    ov = np.sum(psi0(q - lam * ga) * psi0(q - lam * gb)) * dq
    rec('App.B: Gaussian overlap exp(-l^2 D^2/8s^2)', abs(ov - np.exp(-lam ** 2 * (ga - gb) ** 2 / (8 * sig ** 2))))

# Fig. 1 coordinates (t=0..4, x)
blue = (1, 2, 2, 3, 3); green = (1, 2, 2, 4, 3); red = (1, 0, 1, 2, 3)
dd = lambda a, b: sum(1 for t in (1, 2, 3) if a[t] != b[t])
print('Fig.1: d(blue,green)=%d d(red,blue)=%d d(red,green)=%d' % (dd(blue, green), dd(red, blue), dd(red, green)))

print('\nMaximum errors:')
for k, e in TOL.items():
    print('  %-60s %.2e' % (k, e))

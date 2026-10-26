"""Proposition 6 (circuit realisation) of the manuscript.

(1) The recording gadget G_kappa : |x>|0>|0^m> -> |x> (sqrt(kappa)|0>|0^m> + sqrt(1-kappa)|1>|c(x)>) realises the
    channel D_kappa^c = kappa id + (1-kappa) Delta_c after the pointer is traced out.
(2) The full circuit (layers W_k = A_k diag(e^{i beta v_k}) on the configuration register, one fresh pointer register
    per recorded epoch, pointers never reused) has the configuration statistics P_kappa of Theorem 1, computed
    independently (a) by density-matrix propagation through the channel and (b) by the pair sum over paths.
(3) Measuring the pointers in the basis {|0>|0^m>} u {|1>|a>} gives, at each recorded epoch, "no record" with
    probability kappa and the category with probability 1-kappa, and the joint statistics of pointer outcomes and
    final configuration equals that of the intermittent record (Kraus operators sqrt(kappa) id, sqrt(1-kappa) Pi_a).
(4) Composition of two independent records of the same epoch (Eq. compose in Sec. VII):
    D_kappa^c o D_kappa'^id = kappa kappa' id + (1-kappa) kappa' Delta_c + (1-kappa') Delta_id.
Exit code 0 when all discrepancies are below 1e-10."""
import itertools, sys
import numpy as np
rng = np.random.default_rng(20261026)

def dft(N):
    x = np.arange(N); return np.exp(2j*np.pi*np.outer(x, x)/N)/np.sqrt(N)
def chirp(N):
    x = np.arange(N); return np.exp(1j*np.pi*(x[:, None]-x[None, :])**2/N)/np.sqrt(N)
def haar(N):
    z = (rng.normal(size=(N, N)) + 1j*rng.normal(size=(N, N)))/np.sqrt(2); q, r = np.linalg.qr(z); d = np.diag(r); return q*(d/abs(d))
def Delta(rho, c):
    out = np.zeros_like(rho)
    for a in set(c):
        P = np.diag([1.0 if c[x] == a else 0.0 for x in range(len(c))]); out += P @ rho @ P
    return out
def D(rho, c, kappa): return kappa*rho + (1-kappa)*Delta(rho, c)

worst = 0.0
# ---------- (1) gadget vs channel ----------
for _ in range(30):
    n, m = 3, 1; N = 2**n; c = [x >> (n-m) for x in range(N)]; kappa = rng.uniform()
    psi = rng.normal(size=N) + 1j*rng.normal(size=N); psi /= np.linalg.norm(psi); rho = np.outer(psi, psi.conj())
    # pointer basis index p in 0..2^(m+1)-1: p = flag*2^m + register
    eta = np.zeros((N, 2**(m+1)), complex)
    for x in range(N): eta[x, 0] = np.sqrt(kappa); eta[x, 2**m + c[x]] = np.sqrt(1-kappa)
    out = sum(np.diag(eta[:, p]) @ rho @ np.diag(eta[:, p]).conj().T for p in range(2**(m+1)))
    worst = max(worst, abs(out - D(rho, c, kappa)).max())
print(f"(1) gadget vs D_kappa^c: max error {worst:.2e}")

# ---------- (2),(3) full circuit with pointers ----------
def run_circuit(W, R, c, kappa, x0, m):
    """State vector on configuration register (N) x pointer registers (2^(m+1) each, one per recorded epoch).
    Returns joint probabilities P[xf, o_1, ..., o_|R|] with o = 0 for 'no record' and o = 1 + a for category a."""
    N = W[0].shape[0]; T = len(W); R = sorted(R); P = 2**(m+1)
    state = np.zeros((N,) + (P,)*len(R), complex); state[(x0,) + (0,)*len(R)] = 1.0
    for k in range(T):
        state = np.tensordot(W[k], state, axes=([1], [0]))          # layer on configuration register
        if k in R:
            j = R.index(k); new = np.zeros_like(state)
            for x in range(N):                                   # gadget: pointer j from |0> to |eta_{c(x)}>
                sl = [x] + [slice(None)]*len(R); src = list(sl); src[1+j] = 0
                tgt0 = list(sl); tgt0[1+j] = 0; tgt1 = list(sl); tgt1[1+j] = 2**m + c[x]
                new[tuple(tgt0)] += np.sqrt(kappa)*state[tuple(src)]
                new[tuple(tgt1)] += np.sqrt(1-kappa)*state[tuple(src)]
            state = new
    prob = abs(state)**2
    # collapse pointer index p -> outcome o (0 = no record, 1+a = category a); other p have zero probability
    out = np.zeros((N,) + (1+len(set(c)),)*len(R))
    for idx in itertools.product(range(P), repeat=len(R)):
        o = tuple(0 if p == 0 else 1 + (p - 2**m) for p in idx)
        if any(p != 0 and p < 2**m for p in idx):
            assert abs(prob[(slice(None),)+idx]).max() < 1e-14; continue
        out[(slice(None),)+o] += prob[(slice(None),)+idx]
    return out

def channel_P(W, R, c, kappa, x0):
    N = W[0].shape[0]; rho = np.zeros((N, N), complex); rho[x0, x0] = 1
    for k in range(len(W)):
        rho = W[k] @ rho @ W[k].conj().T
        if k in R: rho = D(rho, c, kappa)
    return np.diag(rho).real

def pairsum_P(W, R, c, kappa, x0):
    N = W[0].shape[0]; T = len(W); amps = {}
    for p in itertools.product(range(N), repeat=T):
        a = 1.0+0j; prev = x0
        for k, x in enumerate(p): a *= W[k][x, prev]; prev = x
        amps[p] = a
    P = np.zeros(N)
    for xf in range(N):
        ps = [p for p in amps if p[-1] == xf]
        P[xf] = sum((kappa**sum(1 for t in R if c[p[t]] != c[q[t]]))*amps[p]*np.conj(amps[q]) for p in ps for q in ps).real
    return P

def intermittent_joint(W, R, c, kappa, x0):
    """Joint distribution of record outcomes and final configuration for the mixture process (Kraus form)."""
    N = W[0].shape[0]; R = sorted(R); cats = sorted(set(c))
    out = np.zeros((N,) + (1+len(cats),)*len(R))
    for o in itertools.product(range(1+len(cats)), repeat=len(R)):
        psi = np.zeros(N, complex); psi[x0] = 1; amp = 1.0
        for k in range(len(W)):
            psi = W[k] @ psi
            if k in R:
                j = R.index(k)
                if o[j] == 0: psi = np.sqrt(kappa)*psi
                else:
                    a = cats[o[j]-1]; psi = np.sqrt(1-kappa)*np.array([psi[x] if c[x] == a else 0 for x in range(N)])
        out[(slice(None),)+o] = abs(psi)**2
    return out

w2 = 0.0; w3 = 0.0; wk = 0.0
for trial in range(12):
    n, m = 2, 1; N = 2**n; T = 4; R = [1, 2]
    c = [x >> (n-m) for x in range(N)] if trial % 2 == 0 else list(rng.integers(0, 2, size=N))
    if len(set(c)) < 2: c[0] = 1 - c[1]
    kappa = rng.uniform(); x0 = int(rng.integers(N))
    A = [dft(N), chirp(N), haar(N), dft(N)]
    W = [A[k] @ np.diag(np.exp(1j*rng.uniform(0, 2*np.pi, size=N))) for k in range(T)]
    joint = run_circuit(W, R, c, kappa, x0, m)
    Pcirc = joint.reshape(N, -1).sum(axis=1)
    w2 = max(w2, abs(Pcirc - channel_P(W, R, c, kappa, x0)).max(), abs(Pcirc - pairsum_P(W, R, c, kappa, x0)).max())
    w3 = max(w3, abs(joint - intermittent_joint(W, R, c, kappa, x0)).max())
    for j in range(len(R)):   # marginal probability of 'no record' at each recorded epoch equals kappa
        marg = joint.sum(axis=tuple(i for i in range(joint.ndim) if i != 1+j))
        wk = max(wk, abs(marg[0] - kappa))
print(f"(2) circuit vs channel vs pair sum: max error {w2:.2e}")
print(f"(3) joint pointer statistics vs intermittent record: max error {w3:.2e}; P(no record) - kappa: {wk:.2e}")

# ---------- (4) composition of two records ----------
w4 = 0.0
for _ in range(30):
    N = 6; c = list(rng.integers(0, 3, size=N)); k1, k2 = rng.uniform(size=2)
    psi = rng.normal(size=N) + 1j*rng.normal(size=N); psi /= np.linalg.norm(psi); rho = np.outer(psi, psi.conj())
    lhs = D(D(rho, list(range(N)), k2), c, k1)
    rhs = k1*k2*rho + (1-k1)*k2*Delta(rho, c) + (1-k2)*np.diag(np.diag(rho))
    w4 = max(w4, abs(lhs - rhs).max())
print(f"(4) composition D_k^c o D_k'^id: max error {w4:.2e}")
worst = max(worst, w2, w3, wk, w4)
print("WORST:", f"{worst:.2e}")
sys.exit(0 if worst < 1e-10 else 1)

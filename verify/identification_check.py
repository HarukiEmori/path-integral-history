"""Checks for Sec. 9 (Contact with Historical Data) of "A Path Integral Model of History".

Proposition 11 (what records determine):
  (i)   complete sharp chronicle: the joint law of the chronicle and the final configuration is the
        product of the transition matrices M_k, it is the same for all models with the same M_k
        (other phases of the accessibilities, other valuations; unitary and open steps), and it
        determines M_k on the reached configurations;
  (ii)  unitary steps give doubly stochastic transition matrices, and every stochastic matrix is
        the matrix M_k of an open step (Kraus operators sqrt(M(x|y)) |x><y|);
  (iii) graded records: P_kappa at |R|+1 known values of kappa determines the coefficients F_d of
        P_kappa = sum_d kappa^d F_d (compared with the direct sums over pairs of paths);
  (iv)  reader's bound: the pointer overlap is at most the Bhattacharyya coefficient of the
        distributions of any reading of the probes, with equality for the intermittent reading,
        and a record that reaches the reader with probability zeta gives the coefficient 1 - zeta.
The blocks below are printed in the order (i), (iii), (iv), (ii).
Also:
  - across one recorded epoch P_kappa - P_0 = kappa (P_1 - P_0) (the defect of the
    Chapman-Kolmogorov relation is the interference term scaled by kappa);
  - the numbers quoted in the text for the reader's bound on the latitude;
  - statements of Secs. 9.2 and 9.4: a coarse record of a unitary model need not have doubly
    stochastic transition frequencies; interference below the resolution of a record survives at
    kappa = 0; two valuations move the realised distribution by at most twice the latitude (and by
    at most 2(1 - zeta^|R|) when a record survives with probability zeta); the record leaves every
    definite configuration unchanged and changes the later statistics of a superposition.

Run: python identification_check.py   (exit code 0 if all checks pass)
"""
import itertools
import sys

import numpy as np

rng = np.random.default_rng(20261123)
TOL = 1e-10
ok_all = True


def report(name, value, tol=TOL):
    global ok_all
    good = bool(value <= tol)
    ok_all = ok_all and good
    print(f"  {'ok ' if good else 'FAIL'} {name}: {value:.3e}")


def haar(n):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def diag_unitary(n):
    return np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, size=n)))


def random_kraus(n, m):
    """m Kraus operators of a random channel on C^n (from an isometry C^n -> C^n (x) C^m)."""
    z = rng.normal(size=(n * m, n)) + 1j * rng.normal(size=(n * m, n))
    q, _ = np.linalg.qr(z)
    return [q[j * n:(j + 1) * n, :] for j in range(m)]


def dephase(rho, cats):
    mask = (cats[:, None] == cats[None, :]).astype(float)
    return rho * mask


def p_kappa_unitary(ws, x_init, rec, kappa, cats):
    n = ws[0].shape[0]
    rho = np.zeros((n, n), dtype=complex)
    rho[x_init, x_init] = 1.0
    for k, w in enumerate(ws, start=1):
        rho = w @ rho @ w.conj().T
        if k in rec:
            rho = kappa * rho + (1 - kappa) * dephase(rho, cats)
    return np.real(np.diag(rho))


# ---------------------------------------------------------------- (i) complete chronicles
print("(i) complete sharp chronicle")
N, steps = 3, 3
A = [haar(N) for _ in range(steps)]
betaV = [rng.uniform(0, 2 * np.pi, size=N) for _ in range(steps)]
W = [A[k] @ np.diag(np.exp(1j * betaV[k])) for k in range(steps)]
M = [np.abs(a) ** 2 for a in A]


def joint_law_unitary(ws, x_init):
    """Law of (x_1, ..., x_N) when every epoch is recorded sharply (projective records)."""
    law = {}
    for path in itertools.product(range(N), repeat=steps):
        amp = 1.0 + 0j
        prev = x_init
        for k, x in enumerate(path):
            amp *= ws[k][x, prev]
            prev = x
        law[path] = abs(amp) ** 2
    return law


law1 = joint_law_unitary(W, 0)
prod_law = {p: np.prod([M[k][p[k], (0 if k == 0 else p[k - 1])] for k in range(steps)]) for p in law1}
report("joint law = product of M_k (unitary steps)", max(abs(law1[p] - prod_law[p]) for p in law1))
report("joint law sums to one", abs(sum(law1.values()) - 1))
# another model with the same moduli: other phases of A, other valuations
A2 = [diag_unitary(N) @ A[k] @ diag_unitary(N) for k in range(steps)]
A2[1] = A2[1].conj()                      # complex conjugation keeps the moduli as well
W2 = [A2[k] @ np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, size=N))) for k in range(steps)]
law2 = joint_law_unitary(W2, 0)
report("same law for other phases and valuations", max(abs(law1[p] - law2[p]) for p in law1))
# the two models differ without records (so the invariance is a property of the complete record)
p_q1 = np.abs((W[2] @ W[1] @ W[0])[:, 0]) ** 2
p_q2 = np.abs((W2[2] @ W2[1] @ W2[0])[:, 0]) ** 2
print(f"       (unrecorded distributions of the two models differ by TV = {0.5 * np.abs(p_q1 - p_q2).sum():.3f})")
# converse: M_k recovered from the joint law on the reached configurations
max_err = 0.0
for k in range(steps):
    for y in range(N):
        den = sum(v for p, v in law1.items() if (0 if k == 0 else p[k - 1]) == y)
        if k == 0 and y != 0:
            continue
        for x in range(N):
            num = sum(v for p, v in law1.items() if (0 if k == 0 else p[k - 1]) == y and p[k] == x)
            max_err = max(max_err, abs(num / den - M[k][x, y]))
report("M_k recovered from the joint law", max_err)

# open steps (Theorem 2): channels with Kraus operators
kraus = [random_kraus(N, 2) for _ in range(steps)]


def joint_law_open(kr, phases, x_init):
    law = {}
    for path in itertools.product(range(N), repeat=steps):
        pr = 1.0
        prev = x_init
        for k, x in enumerate(path):
            pr *= sum(abs((K @ np.diag(np.exp(1j * phases[k])))[x, prev]) ** 2 for K in kr[k])
            prev = x
        law[path] = pr
    return law


M_open = [sum(np.abs(K) ** 2 for K in kraus[k]) for k in range(steps)]
law_o1 = joint_law_open(kraus, betaV, 0)
law_o2 = joint_law_open(kraus, [rng.uniform(0, 2 * np.pi, size=N) for _ in range(steps)], 0)
report("open steps: law = product of M_k", max(abs(law_o1[p] - np.prod([M_open[k][p[k], (0 if k == 0 else p[k - 1])] for k in range(steps)])) for p in law_o1))
report("open steps: law independent of valuations", max(abs(law_o1[p] - law_o2[p]) for p in law_o1))
report("open steps: column sums of M_k equal one", max(abs(m.sum(axis=0) - 1).max() for m in M_open))

# ---------------------------------------------------------------- (ii) graded records
print("(iii) graded records determine the coefficients F_d")
N, steps = 3, 4
A = [haar(N) for _ in range(steps)]
W = [A[k] @ np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, size=N))) for k in range(steps)]
rec = {1, 2, 3}
for cats in (np.array([0, 1, 2]), np.array([0, 0, 1])):
    x_i = 0
    # direct pair sums F_d(x_f)
    F = np.zeros((len(rec) + 1, N))
    inter = list(itertools.product(range(N), repeat=steps - 1))
    for xf in range(N):
        amps = {}
        for mid in inter:
            path = (x_i,) + mid + (xf,)
            a = 1.0 + 0j
            for k in range(steps):
                a *= W[k][path[k + 1], path[k]]
            amps[mid] = a
        for m1 in inter:
            for m2 in inter:
                d = sum(1 for t in rec if cats[m1[t - 1]] != cats[m2[t - 1]])
                F[d, xf] += np.real(amps[m1] * np.conj(amps[m2]))
    kappas = np.array([0.0, 0.3, 0.6, 1.0])
    P = np.array([p_kappa_unitary(W, x_i, rec, kap, cats) for kap in kappas])
    V = np.vander(kappas, N=len(rec) + 1, increasing=True)
    F_fit = np.linalg.solve(V, P)
    report(f"interpolated F_d = pair sums (categories {cats.tolist()})", np.abs(F_fit - F).max(), 1e-9)
    kap = 0.45
    report("P_kappa at a further kappa from the F_d", np.abs(p_kappa_unitary(W, x_i, rec, kap, cats) - sum(kap ** d * F[d] for d in range(len(rec) + 1))).max(), 1e-9)

# one recorded epoch: P_kappa - P_0 = kappa (P_1 - P_0), and at kappa = 0 with a full record the
# Chapman-Kolmogorov product is recovered
N = 4
A1, A2_ = haar(N), haar(N)
W1 = A1 @ np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, size=N)))
W2_ = A2_ @ np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, size=N)))
cats = np.arange(N)
P0 = p_kappa_unitary([W1, W2_], 0, {1}, 0.0, cats)
P1 = p_kappa_unitary([W1, W2_], 0, {1}, 1.0, cats)
ck = (np.abs(A2_) ** 2) @ (np.abs(A1) ** 2)[:, 0]
report("kappa = 0, full record: Chapman-Kolmogorov product", np.abs(P0 - ck).max())
for kap in (0.2, 0.7):
    Pk = p_kappa_unitary([W1, W2_], 0, {1}, kap, cats)
    report(f"defect at kappa = {kap} equals kappa times the interference term", np.abs((Pk - ck) - kap * (P1 - ck)).max())
report("interference term sums to zero", abs((P1 - ck).sum()))

# ---------------------------------------------------------------- (iii) reader's bound
print("(iv) reader's bound: overlap <= Bhattacharyya coefficient of any reading")
worst = -1.0
for trial in range(2000):
    ncat = int(rng.integers(2, 5))
    dim = ncat + 1
    kap = float(rng.uniform(0, 1))
    eta = np.zeros((ncat, dim), dtype=complex)
    for a in range(ncat):
        eta[a, 0] = np.sqrt(kap)
        eta[a, a + 1] = np.sqrt(1 - kap)
    U = haar(dim)
    eta = eta @ U.T                                  # same overlaps, generic vectors
    nout = int(rng.integers(2, 7))
    ops = []
    for _ in range(nout):
        z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        ops.append(z @ z.conj().T)
    S = sum(ops)
    w, v = np.linalg.eigh(S)
    S_inv_half = v @ np.diag(w ** -0.5) @ v.conj().T
    povm = [S_inv_half @ o @ S_inv_half for o in ops]
    p = np.array([[np.real(np.vdot(eta[a], E @ eta[a])) for E in povm] for a in range(ncat)])
    for a in range(ncat):
        for b in range(a + 1, ncat):
            bc = float(np.sum(np.sqrt(np.clip(p[a], 0, None) * np.clip(p[b], 0, None))))
            worst = max(worst, abs(np.vdot(eta[a], eta[b])) - bc)
report("max over 2000 random readings of (overlap - Bhattacharyya coefficient)", max(worst, 0.0))
# equality for the intermittent reading of the standard pointer states
kap = 0.37
ncat = 3
eta = np.zeros((ncat, ncat + 1))
for a in range(ncat):
    eta[a, 0] = np.sqrt(kap)
    eta[a, a + 1] = np.sqrt(1 - kap)
p = eta ** 2
report("intermittent reading: coefficient equals kappa", abs(np.sum(np.sqrt(p[0] * p[1])) - kap))
# a sharp record (kappa = 0) that reaches the reader with probability zeta: coefficient 1 - zeta
zeta = 0.683
p_a = np.array([1 - zeta, zeta, 0.0])
p_b = np.array([1 - zeta, 0.0, zeta])
report("record surviving with probability zeta: coefficient 1 - zeta", abs(np.sum(np.sqrt(p_a * p_b)) - (1 - zeta)))
print("       reader's bound on the latitude 1 - zeta^m for zeta = 0.683:", [round(1 - zeta ** m, 3) for m in (1, 2, 3, 5)])
print("       coefficient for zeta = 0.090:", round(1 - 0.090, 3))

# ---------------------------------------------------------------- (iv) double stochasticity
print("(ii) unitary steps give doubly stochastic transition matrices")
worst_u = 0.0
for _ in range(200):
    n = int(rng.integers(2, 7))
    m = np.abs(haar(n)) ** 2
    worst_u = max(worst_u, abs(m.sum(axis=0) - 1).max(), abs(m.sum(axis=1) - 1).max())
report("row and column sums of |A|^2", worst_u)
m_open = sum(np.abs(K) ** 2 for K in random_kraus(3, 2))
print(f"       (a random channel: row sums {np.round(m_open.sum(axis=1), 3).tolist()}, column sums {np.round(m_open.sum(axis=0), 3).tolist()})")
# two states: doubly stochastic iff M(1|0) = M(0|1)
a, b = 0.191, 0.011
m2 = np.array([[1 - a, b], [a, 1 - b]])
print(f"       (two states with M(1|0) = {a}, M(0|1) = {b}: row sums {np.round(m2.sum(axis=1), 3).tolist()})")

# every stochastic matrix is the matrix M_k of an open step
worst_k, worst_m, worst_law = 0.0, 0.0, 0.0
for _ in range(100):
    n = int(rng.integers(2, 5))
    Ms = []
    krs = []
    for k in range(3):
        m = rng.uniform(size=(n, n))
        m = m / m.sum(axis=0, keepdims=True)                 # columns sum to one: M(x|y)
        kr = []
        for x in range(n):
            for y in range(n):
                K = np.zeros((n, n))
                K[x, y] = np.sqrt(m[x, y])
                kr.append(K)
        worst_k = max(worst_k, np.abs(sum(K.T @ K for K in kr) - np.eye(n)).max())
        for y in range(n):
            rho = np.zeros((n, n)); rho[y, y] = 1.0
            out = sum(K @ rho @ K.T for K in kr)
            worst_m = max(worst_m, np.abs(np.diag(out) - m[:, y]).max())
        Ms.append(m); krs.append(kr)
    # complete chronicle with these steps and arbitrary valuation phases: the Markov chain with the matrices M_k
    phases = [rng.uniform(0, 2 * np.pi, size=n) for _ in range(3)]
    for path in itertools.product(range(n), repeat=3):
        pr, prev = 1.0, 0
        for k, x in enumerate(path):
            pr *= sum(abs((K @ np.diag(np.exp(1j * phases[k])))[x, prev]) ** 2 for K in krs[k])
            prev = x
        chain = np.prod([Ms[k][path[k], (0 if k == 0 else path[k - 1])] for k in range(3)])
        worst_law = max(worst_law, abs(pr - chain))
report("stochastic M: the operators sqrt(M(x|y)) |x><y| satisfy sum K^dagger K = 1", worst_k)
report("stochastic M: <x| A(|y><y|) |x> = M(x|y)", worst_m)
report("stochastic M: complete chronicle of the open steps = Markov chain with M_k", worst_law)

# ---------------------------------------------------------------- statements of Secs. 9.2 and 9.4
print("Secs. 9.2 and 9.4")


def rot(n, i, j, prob):
    """Real rotation in the plane (i, j) that moves the weight `prob` from i to j."""
    u = np.eye(n)
    c, s_ = np.sqrt(1 - prob), np.sqrt(prob)
    u[i, i], u[j, j], u[j, i], u[i, j] = c, c, s_, -s_
    return u


# a coarse record of a unitary model: X = {0, 1, 2}, categories (0, 1, 1), kappa = 0
cats3 = np.array([0, 1, 1])
W1c, W2c = rot(3, 0, 2, 0.105), rot(3, 0, 1, 0.191)
rho = np.zeros((3, 3)); rho[0, 0] = 1.0
rho = dephase(W1c @ rho @ W1c.T, cats3)                       # sharp record of the category after step 1
p_cat1 = rho[1, 1] + rho[2, 2]
joint = np.zeros((2, 2))                                      # joint[b, a]: category a after step 1, b after step 2
for a_cat in (0, 1):
    proj = np.diag((cats3 == a_cat).astype(float))
    cond = proj @ rho @ proj
    after = W2c @ cond @ W2c.T
    joint[0, a_cat] = after[0, 0]
    joint[1, a_cat] = after[1, 1] + after[2, 2]
f01 = joint[1, 0] / joint[:, 0].sum()
f10 = joint[0, 1] / joint[:, 1].sum()
report("coarse record of unitary steps: frequency 0 -> 1 equals 0.191", abs(f01 - 0.191))
report("coarse record of unitary steps: frequency 1 -> 0 equals 0", abs(f10))
print(f"       (first step 0 -> 1: {p_cat1:.3f}; the coarse transition frequencies {f01:.3f} and {f10:.3f} are not doubly stochastic)")

# interference below the resolution survives at kappa = 0
A1s = np.zeros((3, 3)); A1s[:, 0] = [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]; A1s[:, 1] = [0, 1 / np.sqrt(2), -1 / np.sqrt(2)]; A1s[:, 2] = [1, 0, 0]
A2s = np.eye(3); A2s[1:, 1:] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
outs = []
for phi in (0.0, np.pi / 2, np.pi):
    V = np.diag([1, 1, np.exp(1j * phi)])
    outs.append(p_kappa_unitary([A1s, A2s @ V], 0, {1}, 0.0, cats3))
report("valuation below the resolution, kappa = 0, phase 0: (0, 1, 0)", np.abs(outs[0] - np.array([0, 1, 0])).max())
report("valuation below the resolution, kappa = 0, phase pi/2: (0, 1/2, 1/2)", np.abs(outs[1] - np.array([0, 0.5, 0.5])).max())
report("valuation below the resolution, kappa = 0, phase pi: (0, 0, 1)", np.abs(outs[2] - np.array([0, 0, 1])).max())

# two valuations, records of full resolution at every intermediate epoch: TV <= 2 (1 - (1 - kappa)^|R|)
worst_ratio, worst_excess = 0.0, 0.0
for _ in range(3000):
    n = int(rng.integers(2, 5)); st = int(rng.integers(2, 5))
    acc = [haar(n) for _ in range(st)]
    kap = float(rng.uniform(0, 1))
    recs = set(range(1, st))
    d = []
    for _v in range(2):
        ws = [acc[k] @ np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, size=n))) for k in range(st)]
        d.append(p_kappa_unitary(ws, 0, recs, kap, np.arange(n)))
    tv = 0.5 * np.abs(d[0] - d[1]).sum()
    lat = 1 - (1 - kap) ** len(recs)
    worst_excess = max(worst_excess, tv - 2 * lat)
    if lat > 1e-9:
        worst_ratio = max(worst_ratio, tv / lat)
report("two valuations: TV - 2 (1 - (1 - kappa)^|R|) <= 0 over 3000 random models", max(worst_excess, 0.0))
print(f"       (largest ratio of TV to the latitude in these trials: {worst_ratio:.3f})")
zeta, m_rec = 0.9, 2
print(f"       (a record surviving with probability zeta = {zeta}, |R| = {m_rec}: kappa <= {1 - zeta:.2f}, bound 2 (1 - zeta^|R|) = {2 * (1 - zeta ** m_rec):.3f})")

# the record leaves every definite configuration unchanged and changes the later statistics
n = 4
kap = 0.3
worst_def = 0.0
for cats_ in (np.arange(n), np.array([0, 0, 1, 1])):
    for x in range(n):
        rho = np.zeros((n, n)); rho[x, x] = 1.0
        worst_def = max(worst_def, np.abs(kap * rho + (1 - kap) * dephase(rho, cats_) - rho).max())
report("the record channel leaves every definite configuration unchanged", worst_def)
Wa, Wb = haar(n) @ diag_unitary(n), haar(n) @ diag_unitary(n)
with_rec = p_kappa_unitary([Wa, Wb], 0, {1}, 0.0, np.arange(n))
without = p_kappa_unitary([Wa, Wb], 0, {1}, 1.0, np.arange(n))
print(f"       (a sharp record of the intermediate epoch moves the later distribution by TV = {0.5 * np.abs(with_rec - without).sum():.3f})")
# permutation accessibilities: no dependence on kappa or on the valuations
perm = [np.eye(n)[:, rng.permutation(n)] for _ in range(3)]
dists = []
for kap_ in (0.0, 0.5, 1.0):
    ws = [perm[k] @ diag_unitary(n) for k in range(3)]
    dists.append(p_kappa_unitary(ws, 0, {1, 2}, kap_, np.arange(n)))
report("permutation accessibilities: P_kappa independent of kappa and of the valuations", max(np.abs(dists[0] - dists[1]).max(), np.abs(dists[0] - dists[2]).max()))

print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)

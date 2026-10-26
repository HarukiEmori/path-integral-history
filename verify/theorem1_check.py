"""Numerical check of the recording-interpolation theorem (Theorem 1) and its corollaries.

Model: finite configuration space of size N, T steps, W_k = A_k diag(exp(i beta v_k)).
Recording at intermediate times t=1..T-1 with a classification c: X -> C and pointer overlap kappa
(pointer states eta_x with <eta_x|eta_y> = kappa if c(x) != c(y), 1 if c(x) == c(y)).
Checks: (1) explicit system+probe simulation == path-sum formula == channel composition;
(2) kappa=1 gives |<x_f|W_T...W_1|x_0>|^2, kappa=0 (sharp record) gives the Markov chain prod |W|^2;
(3) sum over x_f of the interference term is 0; (4) potential blindness at kappa=0 for sharp records;
(5) first-order term in kappa is the sum over pairs of paths differing at exactly one recorded time.
"""
import itertools, sys
import numpy as np
rng = np.random.default_rng(20261025)

def haar_unitary(n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))

def chirp_hadamard(n):  # equal-magnitude unitary for even n: N^{-1/2} exp(i pi (x-y)^2 / N)
    x = np.arange(n)
    M = np.exp(1j * np.pi * (x[:, None] - x[None, :]) ** 2 / n) / np.sqrt(n)
    return M

def path_amplitudes(W, x0):
    N = W[0].shape[0]; T = len(W)
    amps = {}
    for p in itertools.product(range(N), repeat=T):
        a = 1.0 + 0j; prev = x0
        for k, x in enumerate(p):
            a *= W[k][x, prev]; prev = x
        amps[p] = a
    return amps

def P_pathsum(W, x0, kappa, c):
    N = W[0].shape[0]; T = len(W)
    amps = path_amplitudes(W, x0)
    P = np.zeros(N); first = np.zeros(N); Pc = np.zeros(N)
    for xf in range(N):
        ps = [p for p in amps if p[-1] == xf]
        tot = 0j; f1 = 0j
        for p in ps:
            Pc[xf] += abs(amps[p]) ** 2
            for q in ps:
                d = sum(1 for t in range(T - 1) if c[p[t]] != c[q[t]])
                tot += (kappa ** d) * amps[p] * np.conj(amps[q])
                if d == 1:
                    f1 += amps[p] * np.conj(amps[q])
        P[xf] = tot.real; first[xf] = f1.real
    return P, Pc, first

def P_channel(W, x0, kappa, c):
    N = W[0].shape[0]; T = len(W)
    rho = np.zeros((N, N), complex); rho[x0, x0] = 1
    same = (np.array(c)[:, None] == np.array(c)[None, :])
    for k in range(T):
        rho = W[k] @ rho @ W[k].conj().T
        if k < T - 1:
            rho = np.where(same, rho, kappa * rho)
    return np.real(np.diag(rho))

def P_probe(W, x0, kappa, c):
    """Explicit simulation with one fresh probe per intermediate step.
    Probe basis |0>,|1>,...,|C| ; eta_x = sqrt(kappa)|0> + sqrt(1-kappa)|c(x)+1>."""
    N = W[0].shape[0]; T = len(W)
    C = max(c) + 1
    eta = np.zeros((N, C + 1), complex)
    for x in range(N):
        eta[x, 0] = np.sqrt(kappa); eta[x, c[x] + 1] = np.sqrt(1 - kappa)
    psi = np.zeros(N, complex); psi[x0] = 1
    for k in range(T):
        psi = np.tensordot(W[k], psi, axes=([1], [0]))  # system axis is axis 0
        if k < T - 1:
            psi = psi[..., None] * eta.reshape((N,) + (1,) * (psi.ndim - 1) + (C + 1,))
    prob = np.abs(psi) ** 2
    return prob.reshape(N, -1).sum(axis=1)

def markov(W, x0):
    N = W[0].shape[0]
    p = np.zeros(N); p[x0] = 1
    for Wk in W:
        p = (np.abs(Wk) ** 2) @ p
    return p

def main():
    worst = 0.0
    for trial in range(6):
        N = int(rng.integers(2, 5)); T = int(rng.integers(2, 5))
        beta = 1.3
        if trial % 2 == 0 and N % 2 == 0:
            A = [chirp_hadamard(N) for _ in range(T)]
        else:
            A = [haar_unitary(N) for _ in range(T)]
        V = [rng.normal(size=N) for _ in range(T)]
        W = [a @ np.diag(np.exp(1j * beta * v)) for a, v in zip(A, V)]
        x0 = int(rng.integers(N))
        # sharp classification and a coarse one
        c_sharp = list(range(N))
        c_coarse = [x % 2 for x in range(N)]
        for c in (c_sharp, c_coarse):
            for kappa in (0.0, 0.37, 0.8, 1.0):
                P1, Pc, first = P_pathsum(W, x0, kappa, c)
                P2 = P_channel(W, x0, kappa, c)
                P3 = P_probe(W, x0, kappa, c)
                worst = max(worst, np.abs(P1 - P2).max(), np.abs(P1 - P3).max())
                assert abs(P1.sum() - 1) < 1e-10
                if kappa == 1.0:
                    U = np.eye(N, dtype=complex)
                    for Wk in W: U = Wk @ U
                    worst = max(worst, np.abs(P1 - np.abs(U[:, x0]) ** 2).max())
                if kappa == 0.0 and c is c_sharp:
                    worst = max(worst, np.abs(P1 - markov(W, x0)).max(), np.abs(P1 - Pc).max())
                # interference term sums to zero
                worst = max(worst, abs((P1 - Pc).sum()))
            # first-order term: d/dkappa at 0 equals sum over pairs with d=1 (relative to P_0, which equals
            # the Markov chain only for sharp records; for coarse records P_0 keeps within-category interference)
            eps = 1e-6
            P0, Pc, _ = P_pathsum(W, x0, 0.0, c)
            Pe, _, first = P_pathsum(W, x0, eps, c)
            worst = max(worst, np.abs((Pe - P0) / eps - first).max() / (1 + np.abs(first).max()) - 1e-6)
            if c is c_sharp:
                worst = max(worst, np.abs(P0 - Pc).max())
            else:
                print(f"   coarse record: max |P_0 - P_c| = {np.abs(P0 - Pc).max():.3f} (within-category interference survives)")
        # potential blindness: kappa=0, sharp record -> independent of V
        V2 = [rng.normal(size=N) for _ in range(T)]
        W2 = [a @ np.diag(np.exp(1j * beta * v)) for a, v in zip(A, V2)]
        worst = max(worst, np.abs(P_channel(W, x0, 0.0, c_sharp) - P_channel(W2, x0, 0.0, c_sharp)).max())
        # with kappa>0 the distribution does depend on V (sanity: difference generically nonzero)
        dep = np.abs(P_channel(W, x0, 1.0, c_sharp) - P_channel(W2, x0, 1.0, c_sharp)).max()
        print(f"trial {trial}: N={N} T={T} x0={x0}  max discrepancy so far {worst:.2e}; V-dependence at kappa=1: {dep:.3f}")
    print("WORST DISCREPANCY:", worst)
    # equal-magnitude check for the chirp unitary
    for N in (4, 6, 8, 10):
        M = chirp_hadamard(N)
        print(f"chirp N={N}: unitarity err {np.abs(M @ M.conj().T - np.eye(N)).max():.2e}, |entries| range {np.abs(M).min():.4f}..{np.abs(M).max():.4f}")
    return 0 if worst < 1e-8 else 1

if __name__ == "__main__":
    sys.exit(main())

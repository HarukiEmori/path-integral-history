"""Corollary 2: total-variation distance between P_kappa and the Markov chain P_c is at most 1-(1-kappa)^|R|
(sharp records at all |R| = T-1 intermediate times). Also checks the mixture representation
P_kappa = sum_S kappa^|S| (1-kappa)^{|R|-|S|} P_S, where P_S has coherence kept exactly at the times in S,
and the second statement of the corollary: two choices of the valuations give distributions whose
total-variation distance is at most 1-(1-kappa)^|R| (unitary steps and open steps with random Kraus operators)."""
import itertools, numpy as np
from theorem1_check import haar_unitary, P_channel, markov, rng
def P_subset(W, x0, S):
    N = W[0].shape[0]; T = len(W)
    rho = np.zeros((N, N), complex); rho[x0, x0] = 1
    for k in range(T):
        rho = W[k] @ rho @ W[k].conj().T
        if k < T - 1 and (k + 1) not in S:      # time k+1 recorded sharply unless in S
            rho = np.diag(np.diag(rho))
    return np.real(np.diag(rho))
worst_mix = 0; viol = 0; ratios = []
for trial in range(40):
    N = int(rng.integers(2, 6)); T = int(rng.integers(2, 6)); beta = 2.0
    W = [haar_unitary(N) @ np.diag(np.exp(1j * beta * rng.normal(size=N))) for _ in range(T)]
    x0 = int(rng.integers(N)); R = list(range(1, T)); c = list(range(N))
    for kappa in rng.uniform(0, 1, 4):
        Pk = P_channel(W, x0, kappa, c); Pc = markov(W, x0)
        tv = 0.5 * np.abs(Pk - Pc).sum(); bound = 1 - (1 - kappa) ** len(R)
        if tv > bound + 1e-12: viol += 1
        ratios.append(tv / bound if bound > 0 else 0)
        mix = sum((kappa ** len(S)) * ((1 - kappa) ** (len(R) - len(S))) * P_subset(W, x0, set(S))
                  for r in range(len(R) + 1) for S in itertools.combinations(R, r))
        worst_mix = max(worst_mix, np.abs(mix - Pk).max())
print("mixture representation max err:", worst_mix)
print("TV bound violations:", viol, " max ratio TV/bound:", max(ratios))


def kraus(n, m):
    z = rng.normal(size=(n * m, n)) + 1j * rng.normal(size=(n * m, n))
    q, _ = np.linalg.qr(z)
    return [q[j * n:(j + 1) * n, :] for j in range(m)]


def P_two(steps, phases, x0, kappa):
    """Final distribution for steps given by Kraus operators, valuation phases, and a record of full resolution after every step but the last."""
    N = steps[0][0].shape[0]; T = len(steps)
    rho = np.zeros((N, N), complex); rho[x0, x0] = 1
    for k in range(T):
        V = np.diag(np.exp(1j * phases[k]))
        rho = sum(K @ V @ rho @ V.conj().T @ K.conj().T for K in steps[k])
        if k < T - 1:
            rho = kappa * rho + (1 - kappa) * np.diag(np.diag(rho))
    return np.real(np.diag(rho))


viol2 = 0; worst2 = {"unitary": 0.0, "open": 0.0}
for kind in ("unitary", "open"):
    for trial in range(2000):
        N = int(rng.integers(2, 5)); T = int(rng.integers(2, 5)); kappa = float(rng.uniform(0, 1))
        steps = [[haar_unitary(N)] if kind == "unitary" else kraus(N, 2) for _ in range(T)]
        pa = P_two(steps, [rng.uniform(0, 2 * np.pi, size=N) for _ in range(T)], 0, kappa)
        pb = P_two(steps, [rng.uniform(0, 2 * np.pi, size=N) for _ in range(T)], 0, kappa)
        tv = 0.5 * np.abs(pa - pb).sum(); bound = 1 - (1 - kappa) ** (T - 1)
        if tv > bound + 1e-12: viol2 += 1
        if bound > 1e-9: worst2[kind] = max(worst2[kind], tv / bound)
print("two valuations: violations of TV <= 1-(1-kappa)^|R|:", viol2, " max ratio: unitary %.3f, open %.3f" % (worst2["unitary"], worst2["open"]))
# the bound is attained in the minimal example (N = 2, T = 2, Hadamard accessibility, valuation differences 0 and pi)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
for kappa in (0.2, 0.7, 1.0):
    pa = P_two([[H], [H]], [np.zeros(2), np.array([0.0, 0.0])], 0, kappa)
    pb = P_two([[H], [H]], [np.zeros(2), np.array([0.0, np.pi])], 0, kappa)
    print("minimal example, kappa = %.1f: TV = %.3f, latitude = %.3f" % (kappa, 0.5 * np.abs(pa - pb).sum(), kappa))

"""Corollary: total-variation distance between P_kappa and the Markov chain P_c is at most 1-(1-kappa)^|R|
(sharp records at all |R| = T-1 intermediate times). Also checks the mixture representation
P_kappa = sum_S kappa^|S| (1-kappa)^{|R|-|S|} P_S, where P_S has coherence kept exactly at the times in S."""
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

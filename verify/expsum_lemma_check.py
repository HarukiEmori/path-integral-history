"""Lemma 2 (exponential sums with monotone increments, Kusmin-Landau type) and Proposition 5(ii).
|sum_{n=a}^{b} e^{2 pi i phi_n}| <= cot(pi lambda/2)/2 + 3/2 when the increments delta_n are monotone and lambda <= delta_n <= 1/2.
Also the two-sided tail bound of Proposition 5(ii): |sum_{n0<=|n|<=L} e^{i alpha n^2}| <= cot(alpha n0/2) + 3 <= 2/(alpha n0) + 3 for 2 alpha L <= pi."""
import sys
import numpy as np
rng = np.random.default_rng(20261026)
viol = 0; ratio = 0.0
for _ in range(5000):   # general increments
    m = int(rng.integers(1, 400)); lam = rng.uniform(0.001, 0.5)
    d = np.sort(rng.uniform(lam, 0.5, size=m))
    if rng.random() < 0.5: d = d[::-1]
    phi = np.concatenate([[0.0], np.cumsum(d)]) + rng.uniform()
    S = abs(np.exp(2j*np.pi*phi).sum()); bound = 0.5/np.tan(np.pi*lam/2) + 1.5
    ratio = max(ratio, S/bound); viol += S > bound + 1e-12
print(f"Lemma 2 (general monotone increments): violations {viol}/5000, max ratio sum/bound = {ratio:.4f}")
viol = 0; ratio = 0.0
for _ in range(5000):   # Proposition 5(ii)
    L = int(rng.integers(1, 3000)); alpha = rng.uniform(1e-4, np.pi/(2*L)); n0 = int(rng.integers(1, L+1))
    n = np.arange(n0, L+1); S = 2*abs(np.exp(1j*alpha*n**2).sum())
    b1 = 1/np.tan(alpha*n0/2) + 3; b2 = 2/(alpha*n0) + 3
    ratio = max(ratio, S/b1); viol += (S > b1 + 1e-12) or (b1 > b2 + 1e-12)
print(f"Proposition 5(ii): violations {viol}/5000, max ratio sum/bound = {ratio:.4f}")
print("OK" if viol == 0 else "FAIL")
sys.exit(0 if viol == 0 else 1)

"""Numerical checks for the near-field (Fresnel) statements.
Continuum (Proposition 4): Z_Y = int_Y exp(i a y^2) dy, zones Y_j = {y: j pi <= a y^2 < (j+1) pi}.
 (i)  |Z_R - Z_[-L,L]| <= 1/(a L).
 (ii) Z_{Y_0} = sqrt(2 pi/a) (C(sqrt2) + i S(sqrt2)),  |Z_{R \ Y_0}|^2/|Z_R|^2 = ((1-2C)^2+(1-2S)^2)/2,  |Z_{Y_0}|^2/|Z_R|^2 = 2(C^2+S^2).
 (iii) zone plate K_M = union of Y_0, Y_2, ..., Y_{2M}: Im Z_{K_M} >= (2/sqrt(a pi)) (sqrt(2M+3) - 1).
Lattice (Lemma 1): for 2 a R <= pi, R = max(|a_|,|b_|):
 |sum_{n in Z cap [a_,b_]} e^{i a n^2} - int_a_^b_ e^{i a x^2} dx| <= C0 = 1 + 2/pi + pi/2 (integer endpoints), + 2 for real endpoints.
Kusmin-Landau tail: for 2 a L <= pi and 1 <= n0 <= L: |sum_{n0<=n<=L} e^{i a n^2}| <= cot(a n0/2)."""
import numpy as np
from scipy.special import fresnel
rng = np.random.default_rng(7)
def Zint(a, lo, hi):
    """int_lo^hi exp(i a x^2) dx via Fresnel integrals (scipy convention cos(pi t^2/2))."""
    def F(x):
        u = x*np.sqrt(2*a/np.pi); s, c = fresnel(u); return np.sqrt(np.pi/(2*a))*(c+1j*s)
    return F(hi) - F(lo)
S2, C2 = fresnel(np.sqrt(2))
worst = 0.0
# (i) tail bound
for a in (0.01, 0.1, 1.0, 3.0):
    for L in (1, 2, 5, 20, 100):
        full = np.sqrt(np.pi/a)*np.exp(1j*np.pi/4)
        tail = abs(full - Zint(a, -L, L)); bound = 1/(a*L)
        assert tail <= bound + 1e-9, (a, L, tail, bound)
# (ii) zone-0 constants
for a in (0.01, 0.5, 2.0):
    full = np.sqrt(np.pi/a)*np.exp(1j*np.pi/4)
    z0 = Zint(a, -np.sqrt(np.pi/a), np.sqrt(np.pi/a))
    worst = max(worst, abs(z0 - np.sqrt(2*np.pi/a)*(C2+1j*S2)))
    worst = max(worst, abs(abs(full-z0)**2/abs(full)**2 - ((1-2*C2)**2+(1-2*S2)**2)/2))
    worst = max(worst, abs(abs(z0)**2/abs(full)**2 - 2*(C2**2+S2**2)))
print(f"(ii) constants: C(sqrt2)={C2:.6f} S(sqrt2)={S2:.6f}; removed-central ratio={((1-2*C2)**2+(1-2*S2)**2)/2:.4f}; central-only ratio={2*(C2**2+S2**2):.4f}; max err {worst:.1e}")
# (iii) zone plate lower bound
viol = 0
for a in (0.01, 0.3, 1.0):
    for M in range(0, 40):
        z = sum(Zint(a, np.sqrt(j*np.pi/a), np.sqrt((j+1)*np.pi/a)) + Zint(a, -np.sqrt((j+1)*np.pi/a), -np.sqrt(j*np.pi/a)) for j in range(0, 2*M+1, 2))
        lb = 2/np.sqrt(a*np.pi)*(np.sqrt(2*M+3)-1)
        if z.imag < lb - 1e-9: viol += 1
print("(iii) zone-plate bound violations:", viol)
# Lemma 1: lattice vs continuum, random intervals under 2 a R <= pi
C0 = 1 + 2/np.pi + np.pi/2
worst_int = 0.0; worst_real = 0.0
for trial in range(400):
    R = int(rng.integers(1, 3000)); a = np.pi/(2*R) * rng.uniform(0.05, 1.0)
    m1 = int(rng.integers(-R, R+1)); m2 = int(rng.integers(m1, R+1))
    n = np.arange(m1, m2+1); d = abs(np.exp(1j*a*n**2).sum() - Zint(a, m1, m2))
    worst_int = max(worst_int, d)
    lo = m1 - rng.uniform(0, 1); hi = m2 + rng.uniform(0, 1)
    if max(abs(lo), abs(hi))*2*a <= np.pi:
        n = np.arange(np.ceil(lo), np.floor(hi)+1); d2 = abs(np.exp(1j*a*n**2).sum() - Zint(a, lo, hi))
        worst_real = max(worst_real, d2)
print(f"Lemma 1: max |sum - int| integer endpoints = {worst_int:.3f} (bound {C0:.3f}); real endpoints = {worst_real:.3f} (bound {C0+2:.3f})")
# Kusmin-Landau tail
violKL = 0; ratios = []
for trial in range(300):
    L = int(rng.integers(2, 5000)); a = np.pi/(2*L)*rng.uniform(0.05, 1.0); n0 = int(rng.integers(1, L+1))
    n = np.arange(n0, L+1); s = abs(np.exp(1j*a*n**2).sum()); b = 1/np.tan(a*n0/2)
    if s > b + 1e-9: violKL += 1
    ratios.append(s/b)
print(f"Kusmin-Landau: violations {violKL}/300, max ratio {max(ratios):.3f}")
# lattice near-field ratios converge to the continuum constants as a -> 0 (2 a L = pi)
for L in (100, 1000, 10000):
    a = np.pi/(2*L); n = np.arange(-L, L+1); ph = a*n**2; Z = np.exp(1j*ph)
    z0 = Z[np.abs(n) < np.sqrt(np.pi/a)].sum(); zf = Z.sum()
    print(f"lattice L={L}: removed-central ratio {abs(zf-z0)**2/abs(zf)**2:.4f}, central-only {abs(z0)**2/abs(zf)**2:.4f}, |Z_full - sqrt(pi/a)e^(i pi/4)| = {abs(zf - np.sqrt(np.pi/a)*np.exp(1j*np.pi/4)):.3f} (bound {C0 + 1/(a*L):.3f})")
print("OK")

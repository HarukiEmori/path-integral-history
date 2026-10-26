"""Finite mirror (Fraunhofer, linear phase): S_L(q) = sum_{y=1}^L e^{i q y}.
(i) |S_L(q)|^2 = sin^2(Lq/2)/sin^2(q/2), I(0)=L^2, bound 1/sin^2(q/2) for q not in 2 pi Z, first zero at 2pi/L.
(ii) Grating: P>=3 integer, q0 = 2 pi/P, kept set Y' = {y: cos(q0 y) > 0}; for L = nP the kept sum equals n c_P,
     c_P = 1 + 2 sum_{y=1}^{floor((P-1)/4)} cos(2 pi y/P) > 0 (real), while the full sum at q0 is 0.
Also a Fresnel (quadratic-phase) illustration: shrinking the mirror broadens the reflected distribution."""
import numpy as np
worst = 0
for L in (4, 7, 16, 41):
    for q in np.linspace(0.05, 2 * np.pi - 0.05, 97):
        S = np.exp(1j * q * np.arange(1, L + 1)).sum()
        I = abs(S) ** 2
        worst = max(worst, abs(I - np.sin(L * q / 2) ** 2 / np.sin(q / 2) ** 2))
        assert I <= 1 / np.sin(q / 2) ** 2 + 1e-9
    assert abs(abs(np.exp(1j * 0 * np.arange(1, L + 1)).sum()) ** 2 - L ** 2) < 1e-9
    # first zero at q = 2 pi / L
    assert abs(np.exp(1j * (2 * np.pi / L) * np.arange(1, L + 1)).sum()) < 1e-9
print("(i) closed form max err:", worst)
for P in range(3, 13):
    q0 = 2 * np.pi / P
    cP = 1 + 2 * sum(np.cos(2 * np.pi * y / P) for y in range(1, (P - 1) // 4 + 1))
    cP_direct = sum(np.exp(1j * q0 * y) for y in range(P) if np.cos(q0 * y) > 1e-12)
    for n in (1, 3, 10):
        L = n * P
        ys = np.arange(1, L + 1)
        kept = ys[np.cos(q0 * ys) > 1e-12]
        S_kept = np.exp(1j * q0 * kept).sum()
        S_full = np.exp(1j * q0 * ys).sum()
        worst = max(worst, abs(S_kept - n * cP), abs(S_full))
    print(f"P={P:2d}: c_P = {cP:.4f} (direct {cP_direct.real:+.4f}{cP_direct.imag:+.1e}i), P/pi = {P/np.pi:.4f}, kept fraction {len(kept)/L:.3f}")
print("(ii) max err:", worst)
# Fresnel illustration: points y in [-L, L], source and detector at height h, specular geometry
def fresnel_pattern(L, h=200.0, k=2 * np.pi / 1.0, a=-100.0, n_dir=181):
    ys = np.arange(-L, L + 1, dtype=float)
    out = []
    for b in np.linspace(-300, 100, n_dir):  # detector x-position; specular image at b = +100 (mirror image of a=-100)
        ph = k * (np.hypot(h, ys - a) + np.hypot(h, ys - b))
        out.append(abs(np.exp(1j * ph).sum()) ** 2)
    out = np.array(out); return out / out.max()
for L in (300, 60, 12):
    pat = fresnel_pattern(L)
    half = (pat > 0.5).sum()
    print(f"Fresnel: half-mirror size L={L}: number of detector positions above half max = {half} (broader when smaller)")
print("WORST:", worst)

"""Independent check (reviewer) of Propositions 2 and 3 (finite mirror), Fig. 2 and the c_P table."""
import numpy as np
err = {}
def rec(k, e): err[k] = max(err.get(k, 0.0), float(e))

def S(Y, q):
    Y = np.asarray(Y)
    return np.exp(1j * np.outer(np.atleast_1d(q), Y)).sum(axis=1)

# Prop 2
qs = np.linspace(-np.pi, np.pi, 20001); qs = qs[np.abs(np.sin(qs / 2)) > 1e-6]
for L in range(1, 41):
    Y = np.arange(1, L + 1)
    I = abs(S(Y, qs)) ** 2
    rec('Prop2: I_Y(q) vs sin^2(Lq/2)/sin^2(q/2)', np.max(abs(I - np.sin(L * qs / 2) ** 2 / np.sin(qs / 2) ** 2)))
    rec('Prop2: I_Y(0) - L^2', abs(abs(S(Y, 0.0)[0]) ** 2 - L ** 2))
    viol = np.sum(I > 1 / np.sin(qs / 2) ** 2 + 1e-9)
    rec('Prop2: violations of I <= 1/sin^2(q/2)', viol)
    if L >= 2:
        rec('Prop2: I_Y(2pi/L)', abs(S(Y, 2 * np.pi / L)[0]) ** 2)
        # no zero in (0, 2pi/L)
        qq = np.linspace(1e-3, 2 * np.pi / L * (1 - 1e-3), 2000)
        rec('Prop2: zero of I inside (0, 2pi/L) (1 if found)', float(np.min(abs(S(Y, qq)) ** 2) < 1e-9))
print('Prop2 with L=1: I_Y(q) = 1 for all q, so "vanishes first at 2pi/L" needs L >= 2:',
      np.allclose(abs(S([1], qs)) ** 2, 1))

# Prop 3
def cP(P):
    return 1 + 2 * sum(np.cos(2 * np.pi * m / P) for m in range(1, (P - 1) // 4 + 1))
table = []
for P in range(3, 13):
    q0 = 2 * np.pi / P
    # direct residue check of the floor((P-1)/4) boundary
    kept = [r for r in range(P) if np.cos(q0 * r) > 1e-12]
    mmax = max(min(r, P - r) for r in kept)
    rec('Prop3: largest |m| with cos>0 equals floor((P-1)/4)', abs(mmax - (P - 1) // 4))
    if P % 4 == 0:
        rec('Prop3: P%4==0 residue P/4 has cos = 0 (|cos|)', abs(np.cos(q0 * P / 4)))
        assert P // 4 not in kept
    for n in range(1, 6):
        L = n * P; Y = np.arange(1, L + 1)
        Yp = Y[np.cos(q0 * Y) > 1e-12]
        rec('Prop3: |S_Y(q0)|', abs(S(Y, q0)[0]))
        rec('Prop3: S_Y\'(q0) - n c_P', abs(S(Yp, q0)[0] - n * cP(P)))
        rec('Prop3: I_Y\'(q0) - (c_P/P)^2 L^2', abs(abs(S(Yp, q0)[0]) ** 2 - (cP(P) / P) ** 2 * L ** 2))
        # specular intensity of the pruned array vs intensity in q0
        r = abs(S(Yp, q0)[0]) ** 2 / abs(S(Yp, 0.0)[0]) ** 2
        if n == 1:
            table.append((P, round(cP(P), 3), len(Yp) / L, round(r, 3)))
print('P, c_P, |Y\'|/L, I_Y\'(q0)/I_Y\'(0):')
for t in table: print('  ', t)
listed = [1, 1, 1.618, 2, 2.247, 2.414, 2.879, 3.236, 3.513, 3.732]
rec('App.C: listed c_P values (3 decimals)', max(abs(cP(P) - l) for P, l in zip(range(3, 13), listed)))
for P in (100, 1000, 10000):
    print('c_P/P for P=%d: %.5f  (1/pi = %.5f)' % (P, cP(P) / P, 1 / np.pi))
print('c_8 = %.6f, 1+sqrt2 = %.6f, (c_8/8)^2 = %.4f, 1/pi^2 = %.4f' % (cP(8), 1 + np.sqrt(2), (cP(8) / 8) ** 2, 1 / np.pi ** 2))

# Fig 2b: L=48, P=8
Y = np.arange(1, 49); Yp = Y[np.cos(np.pi / 4 * Y) > 1e-12]
print('Fig2b: removed alternatives =', 48 - len(Yp))
qg = np.linspace(-3.14, 3.14, 6001); qg = qg[abs(np.sin(4 * qg)) > 1e-6]
pg = (3 + 2 * np.cos(6 * qg) + 2 * np.cos(7 * qg) + 2 * np.cos(qg)) * np.sin(24 * qg) ** 2 / (2304 * np.sin(4 * qg) ** 2)
rec('Fig2b: pgfplots formula vs |S_Y\'|^2/L^2', np.max(abs(pg - abs(S(Yp, qg)) ** 2 / 48 ** 2)))
fa = np.sin(3 * qg) ** 2 / (36 * np.sin(qg / 2) ** 2)
rec('Fig2a: pgfplots L=6 formula', np.max(abs(fa - abs(S(np.arange(1, 7), qg)) ** 2 / 36)))
print('Fig2b: pruned array I(0)/L^2 = %.4f, I(q0)/L^2 = %.4f, I(-q0)/L^2 = %.4f'
      % (abs(S(Yp, 0.0)[0]) ** 2 / 2304, abs(S(Yp, np.pi / 4)[0]) ** 2 / 2304, abs(S(Yp, -np.pi / 4)[0]) ** 2 / 2304))
print('\nMaximum errors:')
for k, e in err.items(): print('  %-62s %.2e' % (k, e))

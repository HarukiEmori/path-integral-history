"""Minimal example: N=2, A=Hadamard, V=diag(v0,v1), two steps, record at t=1 with overlap kappa.
Closed form: P_kappa(0) = 1/2 + (kappa/2) cos(beta (v1-v0)), P_kappa(1) = 1/2 - (kappa/2) cos(beta (v1-v0)).
Visibility of P_kappa(0) as a function of the valuation difference equals kappa; distinguishability sqrt(1-kappa^2)."""
import numpy as np
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
worst = 0
for kappa in np.linspace(0, 1, 6):
    vals = []
    for dv in np.linspace(0, 2 * np.pi, 25):
        beta = 1.0; v0 = 0.4; v1 = v0 + dv
        W = H @ np.diag(np.exp(1j * beta * np.array([v0, v1])))
        rho = np.zeros((2, 2), complex); rho[0, 0] = 1
        rho = W @ rho @ W.conj().T
        rho = kappa * rho + (1 - kappa) * np.diag(np.diag(rho))
        rho = W @ rho @ W.conj().T
        p0 = rho[0, 0].real
        worst = max(worst, abs(p0 - (0.5 + 0.5 * kappa * np.cos(beta * (v1 - v0)))))
        vals.append(p0)
    vis = (max(vals) - min(vals)) / (max(vals) + min(vals)) if kappa > 0 else 0
    print(f"kappa={kappa:.2f}: max |P-closed form| = {worst:.2e}, visibility = {vis:.4f}, sqrt(1-kappa^2) = {np.sqrt(1-kappa**2):.4f}, V^2+D^2 = {vis**2 + 1 - kappa**2:.4f}")
print("WORST:", worst)

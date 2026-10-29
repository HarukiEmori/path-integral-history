"""Coarse-graining (refinement) form of Proposition 10 of "A Path Integral Model of History".

Setting.  A coarse model on X (|X| = N) and a fine model on X' with a surjection
rho: X' -> X whose fibres are the refinements of the coarse alternatives.  The
block-uniform embedding iota: C^X -> C^{X'}, iota|x> = |fibre(x)|^{-1/2} sum_{x' in fibre(x)} |x'>,
is an isometry.  The fine records use the pulled-back resolution c' = c o rho, the same
overlap kappa and the same recorded epochs; the fine model starts from iota|x_i>.

Claim (refinement form).  With eps_k = ||W'_k iota - iota W_k|| (operator norm),

    TV( rho_* P'_kappa , P_kappa ) <= sum_k eps_k ,

where rho_* P' is the fine realised distribution pushed forward to X.  The proof is the
one of Proposition 10 (telescoping over the steps, contractivity of the channels, and the
intertwining Delta_{c'} o Ad(iota) = Ad(iota) o Delta_c), followed by the contraction of TV
under the pushforward.  The block form eps_k <= ||iota^+ W'_k iota - W_k|| + ||(1 - iota iota^+) W'_k iota||
also holds.

The script checks (i) the bound for random models and several perturbation sizes,
(ii) that the exactly intertwining case gives TV = 0 to machine precision, (iii) the block
form, together with its sharper form eps_k <= (a^2 + b^2)^{1/2} (the two terms have
orthogonal ranges), (iv) the bound for a fine model that starts from a single configuration
of the fibre of x_i (size m), TV <= sum_k eps_k + (1 - 1/m)^{1/2}, and (v) creation and
refinement together, with rho: X' -> X u {new} and arbitrary categories on the new points.
"""
import numpy as np

rng = np.random.default_rng(7)


def haar(n):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def polar_unitary(m):
    u, _, vh = np.linalg.svd(m)
    return u @ vh


def opnorm(m):
    return np.linalg.norm(m, 2)


def dephase(rho, cats):
    mask = (cats[:, None] == cats[None, :]).astype(float)
    return rho * mask


def evolve(ws, rho0, rec_epochs, kappa, cats):
    rho = rho0
    for k, w in enumerate(ws, start=1):
        rho = w @ rho @ w.conj().T
        if k in rec_epochs:
            rho = kappa * rho + (1.0 - kappa) * dephase(rho, cats)
    return rho


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def embedding(fibres):
    """iota (N' x N) for fibre sizes; rho as an index array of length N'."""
    n = len(fibres)
    nprime = int(sum(fibres))
    iota = np.zeros((nprime, n), dtype=complex)
    rho = np.zeros(nprime, dtype=int)
    pos = 0
    for x, m in enumerate(fibres):
        iota[pos:pos + m, x] = 1.0 / np.sqrt(m)
        rho[pos:pos + m] = x
        pos += m
    return iota, rho


def pushforward(p_fine, rho, n):
    out = np.zeros(n)
    for x_prime, x in enumerate(rho):
        out[x] += p_fine[x_prime]
    return out


def run_case(fibres, cats, rec_epochs, kappa, n_steps, delta, x_init=0):
    n = len(fibres)
    iota, rho = embedding(fibres)
    nprime = iota.shape[0]
    cats = np.asarray(cats)
    cats_fine = cats[rho]
    p_perp = np.eye(nprime) - iota @ iota.conj().T
    ws, ws_fine, eps, eps_block = [], [], [], []
    for _ in range(n_steps):
        w = haar(n)
        v = haar(nprime)
        w_fine0 = iota @ w @ iota.conj().T + p_perp @ v @ p_perp  # exactly intertwining
        g = rng.normal(size=(nprime, nprime)) + 1j * rng.normal(size=(nprime, nprime))
        w_fine = polar_unitary(w_fine0 + delta * g / np.sqrt(nprime))
        ws.append(w)
        ws_fine.append(w_fine)
        eps.append(opnorm(w_fine @ iota - iota @ w))
        a_ = opnorm(iota.conj().T @ w_fine @ iota - w)
        b_ = opnorm(p_perp @ w_fine @ iota)
        eps_block.append(np.sqrt(a_ ** 2 + b_ ** 2))
    rho0 = np.zeros((n, n), dtype=complex)
    rho0[x_init, x_init] = 1.0
    psi0 = iota[:, x_init]
    rho0_fine = np.outer(psi0, psi0.conj())
    p_coarse = np.real(np.diag(evolve(ws, rho0, rec_epochs, kappa, cats)))
    p_fine = np.real(np.diag(evolve(ws_fine, rho0_fine, rec_epochs, kappa, cats_fine)))
    tv_push = tv(pushforward(p_fine, rho, n), p_coarse)
    # single-configuration start inside the fibre of x_init (no bound claimed)
    e = np.zeros(nprime, dtype=complex)
    e[int(np.argmax(np.abs(iota[:, x_init])))] = 1.0
    p_single = np.real(np.diag(evolve(ws_fine, np.outer(e, e.conj()), rec_epochs, kappa, cats_fine)))
    tv_single = tv(pushforward(p_single, rho, n), p_coarse)
    m = fibres[x_init]
    single_bound = float(sum(eps)) + np.sqrt(1.0 - 1.0 / m)
    return tv_push, float(sum(eps)), float(sum(eps_block)), tv_single, single_bound


def main():
    fibres = [2, 1, 3]
    cats = [0, 0, 1]
    rec_epochs = {1, 3}
    kappa = 0.4
    n_steps = 4
    ok = True
    print("fibres", fibres, "categories", cats, "recorded epochs", sorted(rec_epochs), "kappa", kappa, "steps", n_steps)
    head = "TV(rho_*P',P)"
    print(f"{'delta':>7} {head:>16} {'sum eps':>10} {'sum block':>10} {'TV single':>10} {'bound':>10}")
    for delta in [0.0, 0.02, 0.1, 0.3, 1.0]:
        tv_push, s_eps, s_block, tv_single, single_bound = run_case(fibres, cats, rec_epochs, kappa, n_steps, delta)
        flag = tv_push <= s_eps + 1e-12 and s_eps <= s_block + 1e-12 and tv_single <= single_bound + 1e-12
        ok = ok and flag
        if delta == 0.0:
            ok = ok and tv_push < 1e-12
        print(f"{delta:7.3f} {tv_push:16.3e} {s_eps:10.3e} {s_block:10.3e} {tv_single:10.3e} {single_bound:10.3e} {'ok' if flag else 'VIOLATION'}")
    # random fibre structures and resolutions
    viol = 0
    trials = 200
    for _ in range(trials):
        n = int(rng.integers(2, 5))
        fib = [int(rng.integers(1, 4)) for _ in range(n)]
        cat = [int(rng.integers(0, 2)) for _ in range(n)]
        steps = int(rng.integers(1, 6))
        rec = {k for k in range(1, steps + 1) if rng.random() < 0.5}
        kap = float(rng.random())
        dl = float(rng.choice([0.0, 0.05, 0.2, 0.6]))
        tv_push, s_eps, s_block, tv_single, single_bound = run_case(fib, cat, rec, kap, steps, dl, x_init=int(rng.integers(0, n)))
        if tv_push > s_eps + 1e-12 or s_eps > s_block + 1e-12 or tv_single > single_bound + 1e-12:
            viol += 1
    print(f"random trials: {trials}, violations: {viol}")
    ok = ok and viol == 0
    # creation and refinement together: rho maps X' onto X u {new}; new points get arbitrary categories
    viol2 = 0
    trials2 = 200
    for _ in range(trials2):
        n = int(rng.integers(2, 4))
        fib = [int(rng.integers(1, 4)) for _ in range(n)]
        n_new = int(rng.integers(1, 4))
        cat = [int(rng.integers(0, 2)) for _ in range(n)]
        cat_new = [int(rng.integers(0, 3)) for _ in range(n_new)]
        steps = int(rng.integers(1, 5))
        rec = {k for k in range(1, steps + 1) if rng.random() < 0.5}
        kap = float(rng.random())
        dl = float(rng.choice([0.0, 0.05, 0.2]))
        iota, rho = embedding(fib)
        nprime = iota.shape[0] + n_new
        iota_full = np.zeros((nprime, n), dtype=complex)
        iota_full[: iota.shape[0], :] = iota
        cats_fine = np.array(list(np.asarray(cat)[rho]) + cat_new)
        p_perp = np.eye(nprime) - iota_full @ iota_full.conj().T
        ws, ws_fine, eps = [], [], []
        for _k in range(steps):
            w = haar(n)
            v = haar(nprime)
            w0 = iota_full @ w @ iota_full.conj().T + p_perp @ v @ p_perp
            g = rng.normal(size=(nprime, nprime)) + 1j * rng.normal(size=(nprime, nprime))
            wf = polar_unitary(w0 + dl * g / np.sqrt(nprime))
            ws.append(w); ws_fine.append(wf); eps.append(opnorm(wf @ iota_full - iota_full @ w))
        x0 = int(rng.integers(0, n))
        rho0 = np.zeros((n, n), dtype=complex); rho0[x0, x0] = 1.0
        psi0 = iota_full[:, x0]
        p_coarse = np.real(np.diag(evolve(ws, rho0, rec, kap, np.asarray(cat))))
        p_fine = np.real(np.diag(evolve(ws_fine, np.outer(psi0, psi0.conj()), rec, kap, cats_fine)))
        pushed = np.zeros(n)
        for x_prime, x in enumerate(rho):
            pushed[x] += p_fine[x_prime]
        # mass on the new points stays as a defect: compare with P_kappa extended by zero
        tv_val = 0.5 * (np.abs(pushed - p_coarse).sum() + p_fine[iota.shape[0]:].sum())
        if tv_val > sum(eps) + 1e-12:
            viol2 += 1
    print(f"creation+refinement trials: {trials2}, violations: {viol2}")
    ok = ok and viol2 == 0
    print("ALL PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

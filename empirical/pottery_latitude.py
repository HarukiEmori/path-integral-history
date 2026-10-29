"""Toy analysis of the contact between the model and records of technical choices (chaines operatoires).

Setting (after van der Leeuw 1993, "Giving the potter a choice", the paddle-and-anvil tradition of Negros):
the shaping of a pot has three stages (lip, body, base: the invariant sequence of the tradition), and at each stage
the potter makes binary choices among the variable elements of the tradition. Four of them are kept here:
  0 rotation   : discontinuous beating (0) / continuous motion (1)
  1 support    : flat board (0) / inverted fired pot used as a mould (1)
  2 pieces     : single lump (0) / two pieces welded later (1)
  3 drying     : short (0) / long (1) drying between stages
A configuration x = (x0, x1, x2, x3) in X = {0,1}^4 is the current set of choices; an epoch is a stage.
Accessibility: a product of one-qubit rotations (each element can be changed between stages) with a sequence
constraint (a two-piece pot forces late welding: element 2 influences element 1 at the next stage).
Valuation: a potter who values speed (scenario A) or one who values control of the shape (scenario B).
Records: the archaeological record sees the variable elements with different resolutions. Welding traces and
mould contact are visible, rotation marks are partly visible, drying is invisible. Each element i is a local
record with its own blur kappa_i (0 = visible, 1 = invisible), which is Proposition 9 of the manuscript.

The script computes, for several visibility profiles, the total-variation distance between the realised
distributions under the two valuations, and compares it with the latitude bound 1 - prod_i (1-kappa_i)^{|R|}.
It also reports the distance on the visible marginal only, which is what an archaeologist can compare.
"""
import itertools, sys
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(1993)
ELEMENTS = ["rotation", "support", "pieces", "drying"]
n = len(ELEMENTS); N = 2**n; T = 3; R = [0, 1]           # records after the lip stage and the body stage

def bits(x): return [(x >> (n-1-i)) & 1 for i in range(n)]
def local_unitary(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], complex)
def kron_all(mats):
    out = np.array([[1.0+0j]])
    for m in mats: out = np.kron(out, m)
    return out
def accessibility(stage):
    """Each element may change with a stage-dependent angle; then the sequence constraint couples pieces -> support."""
    angles = {0: [0.9, 0.6, 1.1, 0.5], 1: [0.5, 0.8, 0.4, 0.9], 2: [0.3, 0.3, 0.2, 0.7]}[stage]
    A = kron_all([local_unitary(a) for a in angles])
    # constraints as controlled rotations in the (x, x^flip) planes: two pieces push the support towards the mould,
    # and a long drying pushes the potter towards making the pot in two pieces (the halves can be left to harden)
    C = np.eye(N, dtype=complex)
    for ctrl, tgt, ang in ((2, 1, 0.7), (3, 2, 0.9)):
        Ck = np.eye(N, dtype=complex)
        for x in range(N):
            b = bits(x)
            if b[ctrl] == 1:
                y = x ^ (1 << (n-1-tgt))
                Ck[x, x] = np.cos(ang); Ck[y, x] = np.sin(ang) if b[tgt] == 0 else -np.sin(ang)
        C = Ck @ C
    return A @ C
def valuation(scenario, x):
    """Valuations couple the elements: speed is gained by continuous rotation together with the mould, and a long
    drying only pays when the pot is made in two pieces (the joint must harden before welding)."""
    b = bits(x)
    if scenario == "A":   # speed
        return 0.6*b[0] + 0.4*b[1] + 0.9*b[0]*b[1] + 0.5*b[2] - 0.8*b[3] + 0.9*b[2]*b[3]
    else:                 # control of the shape
        return -0.4*b[0] - 0.7*b[1] - 0.6*b[0]*b[1] - 0.5*b[2] + 0.6*b[3] + 0.9*b[2]*b[3]
def step(scenario, k, beta=1.2):
    return accessibility(k) @ np.diag(np.exp(1j*beta*np.array([valuation(scenario, x) for x in range(N)])))
def record(rho, kappas):
    """Local records: element i is dephased with blur kappa_i (Proposition 9)."""
    out = rho.copy()
    for i, kap in enumerate(kappas):
        same = np.array([[bits(x)[i] == bits(y)[i] for y in range(N)] for x in range(N)])
        out = np.where(same, out, kap*out)
    return out
def realised(scenario, kappas, x0):
    rho = np.zeros((N, N), complex); rho[x0, x0] = 1
    for k in range(T):
        W = step(scenario, k); rho = W @ rho @ W.conj().T
        if k in R: rho = record(rho, kappas)
    return np.diag(rho).real
def marginal(P, visible):
    """Marginal on the visible elements."""
    out = {}
    for x in range(N):
        key = tuple(b for i, b in enumerate(bits(x)) if visible[i]); out[key] = out.get(key, 0.0) + P[x]
    return out
def tv(p, q): return 0.5*sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))

# unitarity check of the constructed accessibilities
for k in range(T):
    A = accessibility(k); assert abs(A.conj().T @ A - np.eye(N)).max() < 1e-12, "accessibility not unitary"

profiles = {
    "nothing visible (kappa = 1 everywhere)":            [1.0, 1.0, 1.0, 1.0],
    "welding traces visible (pieces)":                   [1.0, 1.0, 0.0, 1.0],
    "welding and mould contact visible":                 [1.0, 0.0, 0.0, 1.0],
    "welding, mould, rotation marks partly (kappa 0.5)": [0.5, 0.0, 0.0, 1.0],
    "everything visible (kappa = 0 everywhere)":         [0.0, 0.0, 0.0, 0.0],
}
x0 = 0   # start: beating, flat board, single lump, short drying (the most traditional case)
print(f"{'visibility profile':52s} {'TV(A,B) all':>12s} {'TV visible':>11s} {'latitude bound':>15s}")
for name, kap in profiles.items():
    PA = realised("A", kap, x0); PB = realised("B", kap, x0)
    vis = [k < 1.0 for k in kap]
    full = 0.5*abs(PA - PB).sum(); mvis = tv(marginal(PA, vis), marginal(PB, vis)) if any(vis) else 0.0
    bound = 1 - np.prod([(1-k)**len(R) for k in kap])
    print(f"{name:52s} {full:12.3f} {mvis:11.3f} {bound:15.3f}")
    assert full <= bound + 1e-12
print("\nReading: the valuation of the potter (speed against control) moves the realised choices by at most the latitude,")
print("which is one as long as some element is never recorded, and shrinks only when every element leaves a trace.")
print("The distance on the visible marginal is what the archaeological record can show; it can be small while the full")
print("distance is large, which is the signature of latitude hidden in the unrecorded elements.")
print("OK")

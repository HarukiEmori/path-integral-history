"""Emulation of the circuit of Proposition 6 (A Path Integral Model of History) with pytket.

The circuit realises the finite path-integral model with recording: a configuration register of n qubits
(N = 2^n configurations), N layers W_k = A_k diag(e^{i beta v_k}), and at each recorded epoch a fresh
pointer register on which the recording gadget G_kappa acts,
    G_kappa : |x>|0>|0^m>  ->  |x> (sqrt(kappa)|0>|0^m> + sqrt(1-kappa)|1>|c(x)>),
with the category c(x) given by the m most significant bits of x (m = 1 here).

Two emulations are provided.
(a) Exact: the statevector of the pure circuit (pytket's built-in simulator, Circuit.get_statevector);
    tracing out the pointers gives P_kappa, which is compared with the channel evolution of the model.
(b) Shots: the same circuit run on the Aer backend (pytket-qiskit) with and without a depolarising noise
    model; the noise plays the role of the device's own record kappa' of Sec. VII of the manuscript.
Run with the quantum-2026 environment: ~/Develop/venv/quantum-2026/bin/python pytket_emulation.py
"""
import itertools, sys
import numpy as np
from pytket import Circuit, OpType
from pytket.circuit import Unitary1qBox, Unitary2qBox, Qubit

rng = np.random.default_rng(20261029)

# ---------------- model (numpy reference) ----------------
def dft(N):
    x = np.arange(N); return np.exp(2j*np.pi*np.outer(x, x)/N)/np.sqrt(N)
def channel_P(W, R, c, kappa, x0):
    N = W[0].shape[0]; rho = np.zeros((N, N), complex); rho[x0, x0] = 1
    for k in range(len(W)):
        rho = W[k] @ rho @ W[k].conj().T
        if k in R:
            out = kappa*rho
            for a in set(c):
                P = np.diag([1.0 if c[x] == a else 0.0 for x in range(N)]); out += (1-kappa)*P @ rho @ P
            rho = out
    return np.diag(rho).real

# ---------------- circuit construction ----------------
def add_unitary(circ, U, qubits):
    """Append the unitary U (N x N, N = 2 or 4) on the given qubits (pytket uses ILO-BE ordering: the
    first listed qubit is the most significant bit of the basis index)."""
    if len(qubits) == 1: circ.add_unitary1qbox(Unitary1qBox(U), qubits[0])
    elif len(qubits) == 2: circ.add_unitary2qbox(Unitary2qBox(U), qubits[0], qubits[1])
    else: raise ValueError("only N = 2, 4 are emulated here")

def qft_gates(circ, qubits):
    """Textbook QFT on the listed qubits (most significant first) with H and controlled phases, then swaps.
    This is the gate-level form of A = DFT (used instead of the unitary box for N = 4 when asked)."""
    n = len(qubits)
    for i in range(n):
        circ.H(qubits[i])
        for j in range(i+1, n):
            circ.add_gate(OpType.CU1, 1.0/2**(j-i), [qubits[j], qubits[i]])   # angle in half-turns
    for i in range(n//2): circ.SWAP(qubits[i], qubits[n-1-i])

def build_circuit(W, R, c, kappa, x0, n, m=1, gate_level_qft=False):
    """Configuration register q[0..n-1]; pointer registers p_t = (flag, m category qubits) per t in R."""
    N = 2**n; R = sorted(R)
    circ = Circuit()
    q = [Qubit("q", i) for i in range(n)]
    for qi in q: circ.add_qubit(qi)
    ptr = {}
    for t in R:
        regs = [Qubit("f", t)] + [Qubit(f"r{t}", j) for j in range(m)]
        for qi in regs: circ.add_qubit(qi)
        ptr[t] = regs
    # initial configuration x0 (big-endian: q[0] is the most significant bit)
    for i in range(n):
        if (x0 >> (n-1-i)) & 1: circ.X(q[i])
    theta = 2*np.arccos(np.sqrt(kappa))/np.pi                 # Ry angle in half-turns
    for k in range(len(W)):
        if gate_level_qft and W[k][1] is not None:            # W[k] = (full unitary, diagonal phases) for the DFT case
            phases = W[k][1]
            circ.add_unitary2qbox(Unitary2qBox(np.diag(np.exp(1j*phases))), q[0], q[1]) if n == 2 else circ.add_unitary1qbox(Unitary1qBox(np.diag(np.exp(1j*phases))), q[0])
            qft_gates(circ, q)
        else:
            add_unitary(circ, W[k][0] if isinstance(W[k], tuple) else W[k], q)
        if k in R:
            f, r = ptr[k][0], ptr[k][1:]
            circ.Ry(theta, f)                                  # |0> -> sqrt(kappa)|0> + sqrt(1-kappa)|1>
            for j in range(m):                                 # controlled copy of the m most significant bits
                circ.CCX(f, q[j], r[j])
    return circ, q, ptr

def marginal_from_statevector(circ, q, N):
    sv = circ.get_statevector()
    nq = circ.n_qubits; qubits = circ.qubits
    probs = abs(sv)**2
    idx = [qubits.index(qi) for qi in q]                        # positions of the configuration qubits
    P = np.zeros(N)
    for basis in range(2**nq):
        bits = [(basis >> (nq-1-pos)) & 1 for pos in idx]      # ILO-BE: qubit 0 is the most significant bit
        x = int("".join(map(str, bits)), 2); P[x] += probs[basis]
    return P

# ---------------- (a) exact emulation ----------------
print("(a) exact emulation (statevector) vs channel evolution")
worst = 0.0
for trial in range(6):
    n = 1 if trial < 3 else 2; N = 2**n; T = 3; R = [0, 1]
    c = [x >> (n-1) for x in range(N)]                          # most significant bit
    kappa = [0.0, 0.5, 1.0][trial % 3]; x0 = int(rng.integers(N))
    W = []
    for k in range(T):
        phases = rng.uniform(0, 2*np.pi, N); W.append(dft(N) @ np.diag(np.exp(1j*phases)))
    circ, q, ptr = build_circuit(W, R, c, kappa, x0, n)
    P = marginal_from_statevector(circ, q, N); Pref = channel_P(W, R, c, kappa, x0)
    worst = max(worst, abs(P - Pref).max())
    print(f"  N={N} kappa={kappa:.1f} x0={x0}: P_circuit={np.round(P,4)} P_model={np.round(Pref,4)} | gates={circ.n_gates} qubits={circ.n_qubits}")
print(f"  max |P_circuit - P_model| = {worst:.2e}")
# minimal example: N=2, two steps, record at epoch 1, P_kappa(0) = 1/2 + (kappa/2) cos(beta(v1 - v0))
print("(a') minimal example N=2 (Hadamard, beta=1, v=(0.4, 0.4+dv))")
H = np.array([[1, 1], [1, -1]])/np.sqrt(2); err = 0.0
for kappa in (0.0, 0.3, 0.7, 1.0):
    for dv in (0.0, 1.0, 2.5):
        W = [H @ np.diag(np.exp(1j*np.array([0.4, 0.4+dv])))]*2
        circ, q, ptr = build_circuit(W, [0], [0, 1], kappa, 0, 1)
        P = marginal_from_statevector(circ, q, 2); err = max(err, abs(P[0] - (0.5 + 0.5*kappa*np.cos(dv))))
print(f"  max |P_kappa(0) - closed form| = {err:.2e}")

# ---------------- (b) shots on Aer, with and without depolarising noise ----------------
try:
    from pytket.extensions.qiskit import AerBackend
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    print("(b) Aer shots: N=2, three steps, records at epochs 0 and 1, kappa = 0.5")
    n, N, T, R = 1, 2, 3, [0, 1]; c = [0, 1]; kappa = 0.5; x0 = 0
    phases = [[0.3, 2.1], [1.7, 0.4], [0.9, 2.8]]                 # fixed valuations, chosen to give a non-uniform target
    W = [dft(N) @ np.diag(np.exp(1j*np.array(ph))) for ph in phases]
    Pref = channel_P(W, R, c, kappa, x0)
    circ, q, ptr = build_circuit(W, R, c, kappa, x0, n)
    pos = [circ.qubits.index(qi) for qi in q]                   # bit index of each configuration qubit after measure_all
    circ.measure_all()
    def noisy_backend(p1, p2):
        """Depolarising noise on every qubit (1-qubit gates) and every ordered pair (2-qubit gates) of the circuit."""
        nm = NoiseModel()
        nq = circ.n_qubits
        for i in range(nq):
            nm.add_quantum_error(depolarizing_error(p1, 1), ["u1", "u2", "u3", "u", "rz", "sx", "x", "h", "ry"], [i])
        for i in range(nq):
            for j in range(nq):
                if i != j: nm.add_quantum_error(depolarizing_error(p2, 2), ["cx", "cz", "cu1", "cp", "swap"], [i, j])
        return AerBackend(noise_model=nm)
    for label, backend in (("noiseless", AerBackend()), ("depolarising 0.5% (1q), 2% (2q)", noisy_backend(0.005, 0.02)), ("depolarising 2% (1q), 8% (2q)", noisy_backend(0.02, 0.08))):
        compiled = backend.get_compiled_circuit(circ)
        res = backend.run_circuit(compiled, n_shots=20000)
        counts = res.get_counts()
        P = np.zeros(N)
        for bits, cnt in counts.items():
            x = int("".join(str(bits[i]) for i in pos), 2); P[x] += cnt
        P /= P.sum()
        print(f"  {label}: P_shots={np.round(P,3)} P_model={np.round(Pref,3)} TV={0.5*abs(P-Pref).sum():.3f}")
except Exception as e:
    print("(b) skipped:", repr(e)[:300])
print("DONE")

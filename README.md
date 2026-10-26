# path-integral-history

Numerical verification scripts for the manuscript *A Path Integral Model of History* (H. Emori and A. Iriki).
The manuscript lives in the Obsidian vault `Intellect` (`04_paper/Quantum-like modeling/A Path Integral Model of History.md`); this repository holds only the checks.

## Requirements
Python 3.11 with NumPy and SciPy. On the author's machine: `~/Develop/venv/quantum-env/bin/python`.

## Scripts (`verify/`)
| Script | What it checks |
| :-- | :-- |
| `theorem1_check.py` | Recording theorem (Theorem 1): explicit probe simulation, channel composition, and the pair sum agree; corollaries (Markov chain at kappa = 0, zero sum of interference, blindness to the valuation, first-order term); unitarity of the quadratic kernel |
| `tvbound_check.py` | Total-variation bound (Corollary 2) and the mixture representation (imports from `theorem1_check.py`) |
| `twostate_check.py` | Minimal example, visibility and distinguishability duality |
| `mirror_check.py` | Far-field mirror (Propositions 2 and 3), the constants gamma_P (printed as c_P), a Fresnel illustration (not used in the manuscript) |
| `nearfield_check.py` | Near-field statements: tail bound, central-zone constants, zone-plate bound (Proposition 4), lattice-continuum lemma (Lemma 1), tail of the lattice sum (Proposition 5; the sharper constant tested here is the classical Kusmin-Landau form, the manuscript uses the weaker self-contained bound of `expsum_lemma_check.py`) |
| `circuit_check.py` | Circuit realisation (Proposition 6): the recording gadget equals the channel D_kappa^c, the full circuit with pointer registers reproduces P_kappa (channel propagation and pair sum), the pointer statistics equal the intermittent record, and the composition of two records (Eq. for the device's own dephasing in Sec. VII) |
| `expsum_lemma_check.py` | Exponential sums with monotone increments (Lemma 2, Kusmin-Landau type) and the two-sided tail bound of Proposition 5(ii) |
| `check_recording.py`, `check_mirror.py` | Independent re-implementations written by the reviewing agent (kept for reference) |

The scripts use the symbols of the first draft in places (`c_P`, `n`, `S_Y`), which correspond to `gamma_P`, `J`, `Z_Y` in the manuscript.

## Run
```bash
cd verify
for f in theorem1_check.py tvbound_check.py twostate_check.py mirror_check.py nearfield_check.py circuit_check.py expsum_lemma_check.py; do python "$f"; done
```
Every script exits with code 0 when all checks pass; the printed maximum discrepancies are of the order of machine precision.

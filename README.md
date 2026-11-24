# path-integral-history

Numerical checks and data analysis for the manuscript *A Path Integral Model of History* (H. Emori and A. Iriki).
The manuscript formulates a finite path-integral model in which a society has finitely many configurations, a course of events is a path on a lattice of epochs, valuations act as phases, and a chronicle is an indirect measurement of the configuration at each epoch.
This repository holds the scripts that check the statements of the manuscript numerically and the script that computes the counts of its Sec. 9.2 from public data; the manuscript itself is not kept here.

## Requirements
Python 3.11 or later with NumPy and SciPy. The emulation script needs pytket, pytket-qiskit and qiskit-aer.

## Scripts (`verify/`)
| Script | What it checks |
| :-- | :-- |
| `theorem1_check.py` | Recording theorem (Theorem 1): explicit probe simulation, channel composition, and the pair sum agree; corollaries (Markov chain at kappa = 0, zero sum of interference, blindness to the valuation, first-order term); unitarity of the quadratic kernel |
| `tvbound_check.py` | Corollary 2: the total-variation bound, the mixture representation, and the bound for two choices of the valuations, with equal and with unequal overlaps (unitary and open steps); imports from `theorem1_check.py` |
| `twostate_check.py` | Minimal example, visibility and distinguishability duality |
| `mirror_check.py` | Far-field mirror (Propositions 2 and 3), the constants gamma_P (printed as c_P), a Fresnel illustration (not used in the manuscript) |
| `nearfield_check.py` | Near-field statements: tail bound, central-zone constants, zone-plate bound (Proposition 4), lattice-continuum lemma (Lemma 1), tail of the lattice sum (Proposition 5; the sharper constant tested here is the classical Kusmin-Landau form, the manuscript uses the weaker self-contained bound of `expsum_lemma_check.py`) |
| `circuit_check.py` | Circuit realisation (Proposition 6): the recording gadget equals the channel D_kappa^c, the full circuit with pointer registers reproduces P_kappa (channel propagation and pair sum), the pointer statistics equal the intermittent record, and the composition of two records |
| `expsum_lemma_check.py` | Exponential sums with monotone increments (Lemma 2, Kusmin-Landau type) and the two-sided tail bound of Proposition 5(ii) |
| `extensions_check.py` | Extensions (Sec. 8): open steps (Theorem 2), records in several frames (Theorem 3), two frames at one epoch (Proposition 8), near-Hadamard contraction bound (Proposition 7), local records on a product set (Proposition 9), growth of the set of alternatives (Proposition 10) |
| `refinement_check.py` | Coarse-graining form of the growth bound of Proposition 10 |
| `identification_check.py` | What records determine (Proposition 11, Sec. 9): the law of a complete sharp chronicle is the product of the transition matrices and is the same for models with other phases and valuations (unitary and open steps); unitary steps give doubly stochastic transition matrices, and every stochastic matrix is the matrix of an open step; the coefficients F_d are recovered from P_kappa at \|R\|+1 values of kappa; the pointer overlap is at most the Bhattacharyya coefficient of any reading of the probes. Statements of Secs. 9.2 and 9.4: coarse records, interference below the resolution of a record, the bound for two valuations, records in the configuration frame and in another frame, two units that act on one another, shares of surviving records that depend on the category, selective removal with normalised states |
| `check_recording.py`, `check_mirror.py` | Independent re-implementations of the checks of Theorem 1 and of the mirror propositions (kept as a cross-check) |

The scripts use the symbols of an early draft in places (`c_P`, `n`, `S_Y`), which correspond to `gamma_P`, `J`, `Z_Y` in the manuscript.
The numbering of statements follows the revised manuscript (Theorems 1 to 3, Corollaries 1 to 4, Propositions 1 to 11, Lemmas 1 and 2).

## Run
```bash
cd verify
for f in theorem1_check.py tvbound_check.py twostate_check.py mirror_check.py nearfield_check.py circuit_check.py expsum_lemma_check.py extensions_check.py refinement_check.py identification_check.py; do python "$f"; done
```
Every script exits with code 0 when all checks pass; the printed maximum discrepancies are of the order of machine precision.

## Data analysis (`empirical/`)
- `reformation_chain.py`: the counts of Sec. 9.2 of the manuscript, computed from the replication data of J. Rubin, "Printing and Protestants: An empirical test of the role of printing in the Reformation", Rev. Econ. Stat. 96, 270 (2014), available in Harvard Dataverse, https://doi.org/10.7910/DVN/26312 (CC0 1.0). The data file is not kept in this repository. Download `Printing_ReStat.tab` from the Dataverse record and pass its path as the argument:
  ```bash
  python -I empirical/reformation_chain.py /path/to/Printing_ReStat.tab
  ```
  The script prints the number of cities on each course (coded Protestant or not in 1530, 1560, 1600), the transition frequencies, Fisher's exact tests of the Markov property and of double stochasticity, the shares of Protestant cities with and without a printing press by 1500, the territories and present-day countries of the cities, and percentile intervals when territories or countries are resampled in place of cities (200,000 draws, fixed seed; the cities are not independent observations).
  It repeats the counts, tests and intervals for the 753 cities that the article analyses (the selection is checked against the means of Table 3 of the article), and it reproduces the estimation samples of the article (200, 352 and 307 cities for 1530, 1560 and 1600) with the shares of Protestant cities with and without a press in each. These are the numbers of Secs. 9.2 and 9.5 of the manuscript.
- `pottery_latitude.py`: a toy application of Proposition 9 to recorded technical choices (after S. E. van der Leeuw, "Giving the potter a choice", 1993): variable elements of a pottery tradition as factors, archaeological visibility as local overlaps kappa_i, and the latitude of the valuation of the potter. The parameters are illustrative and are not estimated from data.

## Emulation (`emulation/`)
- `pytket_emulation.py`: the circuit of Proposition 6 built with pytket; exact statevector emulation against the channel evolution, and shot-based runs on the Aer backend with and without depolarising noise.

## Licence
MIT (see `LICENSE`).

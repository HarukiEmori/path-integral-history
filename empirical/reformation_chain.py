"""Recorded chain of confessions in the replication data of Rubin (2014).

Data: J. Rubin, "Replication data for: Printing and Protestants: An Empirical Test of the Role of
Printing in the Reformation", Harvard Dataverse, doi:10.7910/DVN/26312 (CC0 1.0), file
Printing_ReStat.tab (873 cities; prot1530, prot1560, prot1600 = 1 if the city is coded Protestant
at that date; press = 1 if the city had a printing press by 1500).  The data file is not kept in
this repository; fetch it from the DOI and pass its path as the first argument.

The script only counts.  It prints
  (1) the number of cities on each of the eight courses (x_1530, x_1560, x_1600),
  (2) the transition frequencies of the three steps 1517 -> 1530 -> 1560 -> 1600,
  (3) Fisher's exact test of the Markov property at 1560 (does the state in 1600 depend on the
      state in 1530, given the state in 1560?),
  (4) Fisher's exact test of M(1|0) = M(0|1), which is double stochasticity for two states,
  (5) the shares of Protestant cities in 1600 with and without a press by 1500.
Run:  python -I reformation_chain.py /path/to/Printing_ReStat.tab
"""
import csv
import math
import sys

from scipy.stats import fisher_exact


def main(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    g = lambda r, c: int(float(r[c]))
    n = len(rows)
    cnt = {}
    for r in rows:
        k = (g(r, "prot1530"), g(r, "prot1560"), g(r, "prot1600"))
        cnt[k] = cnt.get(k, 0) + 1
    c = lambda a, b, d: cnt.get((a, b, d), 0)
    print(f"cities: {n}")
    print("(1) courses (x1530, x1560, x1600): number of cities")
    for a in (0, 1):
        for b in (0, 1):
            for d in (0, 1):
                print(f"    ({a}, {b}, {d}): {c(a, b, d)}")
    assert sum(cnt.values()) == n

    print("(2) transition frequencies (0 = not Protestant, 1 = Protestant)")
    k1 = sum(v for k, v in cnt.items() if k[0] == 1)
    print(f"    1517 -> 1530: 0 -> 1: {k1}/{n} = {k1 / n:.4f}")

    def step(i, j):
        m = [[0, 0], [0, 0]]              # m[to][from]
        for k, v in cnt.items():
            m[k[j]][k[i]] += v
        return m

    out = {}
    for name, (i, j) in (("1530 -> 1560", (0, 1)), ("1560 -> 1600", (1, 2))):
        m = step(i, j)
        n0 = m[0][0] + m[1][0]
        n1 = m[0][1] + m[1][1]
        a01 = m[1][0] / n0
        a10 = m[0][1] / n1
        out[name] = (m, n0, n1, a01, a10)
        print(f"    {name}: 0 -> 1: {m[1][0]}/{n0} = {a01:.4f} (s.e. {math.sqrt(a01 * (1 - a01) / n0):.4f}); "
              f"1 -> 0: {m[0][1]}/{n1} = {a10:.4f} (s.e. {math.sqrt(a10 * (1 - a10) / n1):.4f}); "
              f"row sums {1 - a01 + a10:.4f}, {a01 + 1 - a10:.4f}")

    print("(3) Markov property at 1560")
    t = [[c(0, 1, 1), c(0, 1, 0)], [c(1, 1, 1), c(1, 1, 0)]]
    _, p = fisher_exact(t)
    print(f"    Protestant in 1560: stayed/left by 1600, not Protestant in 1530: {t[0][0]}/{t[0][1]}; "
          f"Protestant in 1530: {t[1][0]}/{t[1][1]}; Fisher exact two-sided p = {p:.4f}")
    t0 = [[c(0, 0, 1), c(0, 0, 0)], [c(1, 0, 1), c(1, 0, 0)]]
    _, p0 = fisher_exact(t0)
    print(f"    not Protestant in 1560: became/stayed by 1600, not Protestant in 1530: {t0[0][0]}/{t0[0][1]}; "
          f"Protestant in 1530: {t0[1][0]}/{t0[1][1]}; Fisher exact two-sided p = {p0:.4f}")

    print("(4) double stochasticity (two states: M(1|0) = M(0|1))")
    for name, (m, n0, n1, a01, a10) in out.items():
        tab = [[m[1][0], m[0][0]], [m[0][1], m[1][1]]]
        _, pp = fisher_exact(tab)
        print(f"    {name}: moved/stayed from 0: {tab[0][0]}/{tab[0][1]}, from 1: {tab[1][0]}/{tab[1][1]}; Fisher exact two-sided p = {pp:.3e}")

    print("(5) shares of Protestant cities")
    for y, col in ((1530, "prot1530"), (1560, "prot1560"), (1600, "prot1600")):
        k = sum(g(r, col) for r in rows)
        print(f"    {y}: {k}/{n} = {k / n:.4f}")
    for pr in (1, 0):
        s = [r for r in rows if g(r, "press") == pr]
        k = sum(g(r, "prot1600") for r in s)
        print(f"    press by 1500 = {pr}: Protestant in 1600: {k}/{len(s)} = {k / len(s):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])

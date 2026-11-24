"""Recorded chain of confessions in the replication data of Rubin (2014).

Data: J. Rubin, "Replication data for: Printing and Protestants: An Empirical Test of the Role of
Printing in the Reformation", Harvard Dataverse, doi:10.7910/DVN/26312 (CC0 1.0), file
Printing_ReStat.tab (873 cities; prot1530, prot1560, prot1600 = 1 if the city is coded Protestant
at that date; press = 1 if the city had a printing press by 1500).  The data file is not kept in
this repository; fetch it from the DOI and pass its path as the first argument.

The script only counts.  It prints
  (1) the number of cities on each of the eight courses (x_1530, x_1560, x_1600),
  (2) the transition frequencies of the steps 1530 -> 1560 -> 1600, and the share of cities coded
      Protestant in 1530, which is the frequency of the step 1517 -> 1530 under the ASSUMPTION
      that every city has the code 0 in 1517 (the data contain no code for 1517),
  (3) Fisher's exact test of the Markov property at 1560 (does the state in 1600 depend on the
      state in 1530, given the state in 1560?),
  (4) Fisher's exact test of M(1|0) = M(0|1), which is double stochasticity for two states,
  (5) the shares of Protestant cities in 1600 with and without a press by 1500,
  (6) the territories of the cities (the cities of a territory are not independent observations)
      and percentile intervals of the differences used in (3) and (4) when territories are
      resampled with replacement in place of cities (10,000 draws, fixed seed; a city without a
      territory in the data is its own unit),
  (7) the courses by present-day country (variable currcountry, with the three Italian regions of
      the file counted as one country): the countries all of whose cities share one course, the
      countries that hold the changes of each step, and the intervals of (6) when the countries
      are resampled in place of the territories.
Run:  python -I reformation_chain.py /path/to/Printing_ReStat.tab
"""
import collections
import csv
import math
import sys

import numpy as np
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
    print(f"    1517 -> 1530 (assuming the code 0 for every city in 1517): 0 -> 1: {k1}/{n} = {k1 / n:.4f}")

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

    print("(6) territories and resampling of territories")
    units = collections.defaultdict(list)
    for i, r in enumerate(rows):
        name = r["territory"].strip('"').strip()
        key = name if name else f"(no territory) {i}"
        units[key].append((g(r, "prot1530"), g(r, "prot1560"), g(r, "prot1600")))
    named = [k for k in units if not k.startswith("(no territory)")]
    print(f"    territories named in the data: {len(named)}; cities without a territory: {len(units) - len(named)}; units: {len(units)}")
    sizes = sorted((len(v) for v in units.values()), reverse=True)
    print(f"    largest units: {sizes[:6]}; units with one city: {sum(1 for x in sizes if x == 1)}")

    def diffs(sample):
        cc = collections.Counter(sample)
        n0_2 = sum(v for k, v in cc.items() if k[0] == 0)
        n1_2 = sum(v for k, v in cc.items() if k[0] == 1)
        n0_3 = sum(v for k, v in cc.items() if k[1] == 0)
        n1_3 = sum(v for k, v in cc.items() if k[1] == 1)
        new_n = cc[(0, 1, 1)] + cc[(0, 1, 0)]
        old_n = cc[(1, 1, 1)] + cc[(1, 1, 0)]
        if not (n0_2 and n1_2 and n0_3 and n1_3 and new_n and old_n):
            return None
        d2 = sum(v for k, v in cc.items() if k[0] == 0 and k[1] == 1) / n0_2 - sum(v for k, v in cc.items() if k[0] == 1 and k[1] == 0) / n1_2
        d3 = sum(v for k, v in cc.items() if k[1] == 0 and k[2] == 1) / n0_3 - sum(v for k, v in cc.items() if k[1] == 1 and k[2] == 0) / n1_3
        mk = cc[(1, 1, 1)] / old_n - cc[(0, 1, 1)] / new_n
        return d2, d3, mk

    keys = sorted(units)
    obs = diffs([c3 for k in keys for c3 in units[k]])
    rng = np.random.default_rng(20261123)
    draws = []
    while len(draws) < 10000:
        pick = rng.integers(0, len(keys), size=len(keys))
        d = diffs([c3 for i in pick for c3 in units[keys[i]]])
        if d is not None:
            draws.append(d)
    draws = np.array(draws)
    labels = ("M(1|0) - M(0|1), 1530 -> 1560", "M(1|0) - M(0|1), 1560 -> 1600", "share staying Protestant to 1600: Protestant since 1530 minus changed after 1530")
    for j, lab in enumerate(labels):
        lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
        print(f"    {lab}: observed {obs[j]:+.4f}; 95% interval over resampled territories [{lo:+.3f}, {hi:+.3f}]")
    left = [(r["city"].strip('"'), r["territory"].strip('"')) for r in rows if g(r, "prot1560") == 1 and g(r, "prot1600") == 0]
    print(f"    cities coded Protestant in 1560 and not in 1600: {left}")

    print("(7) present-day countries")

    def country(r):
        name = r["currcountry"].strip('"').strip()
        return "Italy" if name.endswith("Italy") else name

    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[country(r)][(g(r, "prot1530"), g(r, "prot1560"), g(r, "prot1600"))] += 1
    single = sorted((k for k, v in by.items() if len(v) == 1), key=lambda k: -sum(by[k].values()))
    print(f"    countries: {len(by)}; countries all of whose cities share one course: {len(single)}, with {sum(sum(by[k].values()) for k in single)} cities")
    print("    " + ", ".join(f"{k} {''.join(map(str, next(iter(by[k]))))} ({sum(by[k].values())})" for k in single))

    def where(pred, label):
        cc = collections.Counter()
        for k, v in by.items():
            n = sum(x for c3, x in v.items() if pred(c3))
            if n:
                cc[k] = n
        print(f"    {label}: {sum(cc.values())} cities: " + ", ".join(f"{k} {n}" for k, n in cc.most_common()))

    where(lambda c3: c3[0] == 1, "Protestant in 1530")
    where(lambda c3: c3[0] == 0 and c3[1] == 1, "0 -> 1, 1530 -> 1560")
    where(lambda c3: c3[1] == 0 and c3[2] == 1, "0 -> 1, 1560 -> 1600")
    where(lambda c3: c3[0] == 1 and c3[1] == 0, "1 -> 0, 1530 -> 1560")
    where(lambda c3: c3[1] == 1 and c3[2] == 0, "1 -> 0, 1560 -> 1600")
    ckeys = sorted(by)
    cunits = {k: [c3 for c3, x in by[k].items() for _ in range(x)] for k in ckeys}
    rng = np.random.default_rng(20261123)
    draws = []
    while len(draws) < 10000:
        pick = rng.integers(0, len(ckeys), size=len(ckeys))
        d = diffs([c3 for i in pick for c3 in cunits[ckeys[i]]])
        if d is not None:
            draws.append(d)
    draws = np.array(draws)
    for j, lab in enumerate(labels[:2]):
        lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
        print(f"    {lab}: 95% interval over resampled countries [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1])

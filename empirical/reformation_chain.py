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
      resampled with replacement in place of cities (200,000 draws, fixed seed; a city without a
      territory in the data is its own unit; the limits are stable to about 0.002 across seeds),
  (7) the courses by present-day country (variable currcountry, with the three Italian regions of
      the file counted as one country): the countries all of whose cities share one course, the
      countries that hold the changes of each step, and the intervals of (6) when the countries
      are resampled in place of the territories,
  (8) the same counts, tests and intervals for the 753 cities that the published article analyses
      (the rows with a recorded population in 1500, without Mainz, Wittenberg and Zurich, from
      which the article measures distances); the selection is checked against the means of
      Table 3 of the article,
  (9) the estimation samples of the article (Tables 6 and 7: 200, 352 and 307 cities in 102, 157
      and 147 territories for 1530, 1560 and 1600), obtained from the 753 cities by leaving out
      every nation or Imperial Circle of the file whose cities all have the same code at the
      date, and the shares of Protestant cities with and without a press in each.
Run:  python -I reformation_chain.py /path/to/Printing_ReStat.tab
"""
import collections
import csv
import math
import sys

import numpy as np
from scipy.stats import fisher_exact

LABELS = ("M(1|0) - M(0|1), 1530 -> 1560", "M(1|0) - M(0|1), 1560 -> 1600",
          "share staying Protestant to 1600: Protestant since 1530 minus changed after 1530")
NATIONS = ("scotdum", "engdum", "iredum", "denmarkdum", "findum", "swededum", "nordum", "frandum", "prusdum",
           "italiandum", "italianhre", "poldum", "portdum", "espdum", "belghredum", "nethdum", "switzdum")
CIRCLES = ("electorate", "upperrhencirc", "lowersaxoncirc", "swabiancirc", "franccirc", "westphalcirc",
           "bavariancirc", "bohemiancirc", "austriancirc", "burgcirc")


def g(r, c):
    return int(float(r[c]))


def course(r):
    return (g(r, "prot1530"), g(r, "prot1560"), g(r, "prot1600"))


def country(r):
    name = r["currcountry"].strip('"').strip()
    return "Italy" if name.endswith("Italy") else name


def territory_units(rows):
    """The cities grouped by territory; a city without a territory in the data is its own unit."""
    units = collections.defaultdict(list)
    for i, r in enumerate(rows):
        name = r["territory"].strip('"').strip()
        units[name if name else f"(no territory) {i}"].append(course(r))
    return units


def country_units(rows):
    units = collections.defaultdict(list)
    for r in rows:
        units[country(r)].append(course(r))
    return units


def course_table(groups):
    """One row per unit: the number of its cities on each of the eight courses (index 4a + 2b + c)."""
    keys_ = sorted(groups)
    tab = np.zeros((len(keys_), 8))
    for i, k in enumerate(keys_):
        for c3 in groups[k]:
            tab[i, 4 * c3[0] + 2 * c3[1] + c3[2]] += 1
    return tab


def diffs(counts):
    """counts[..., 8] -> the three differences used in (3) and (4)."""
    def n(a=None, b=None, c=None):
        return sum(counts[..., 4 * x + 2 * y + z] for x in (0, 1) for y in (0, 1) for z in (0, 1)
                   if (a is None or x == a) and (b is None or y == b) and (c is None or z == c))
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = n(0, 1) / n(0) - n(1, 0) / n(1)
        d3 = n(None, 0, 1) / n(None, 0) - n(None, 1, 0) / n(None, 1)
        mk = n(1, 1, 1) / n(1, 1) - n(0, 1, 1) / n(0, 1)
    return np.stack([d2, d3, mk], axis=-1)


def resample(tab, draws=200000, seed=20261123):
    """Percentile intervals when the units are resampled with replacement (multinomial weights)."""
    rng = np.random.default_rng(seed)
    m = tab.shape[0]
    out = []
    for _ in range(draws // 10000):
        w = rng.multinomial(m, np.ones(m) / m, size=10000).astype(float)
        out.append(diffs(w @ tab))
    out = np.concatenate(out)
    return [np.percentile(col[np.isfinite(col)], [2.5, 97.5]) for col in out.T]


def press_shares(rows, col="prot1600"):
    out = []
    for pr in (1, 0):
        s = [r for r in rows if g(r, "press") == pr]
        out.append((sum(g(r, col) for r in s), len(s)))
    return out


def main(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    n = len(rows)
    cnt = collections.Counter(course(r) for r in rows)
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
    for pr, (k, m) in zip((1, 0), press_shares(rows)):
        print(f"    press by 1500 = {pr}: Protestant in 1600: {k}/{m} = {k / m:.4f}")

    print("(6) territories and resampling of territories")
    units = territory_units(rows)
    named = [k for k in units if not k.startswith("(no territory)")]
    print(f"    territories named in the data: {len(named)}; cities without a territory: {len(units) - len(named)}; units: {len(units)}")
    sizes = sorted((len(v) for v in units.values()), reverse=True)
    print(f"    largest units: {sizes[:6]}; units with one city: {sum(1 for x in sizes if x == 1)}")
    tab = course_table(units)
    obs = diffs(tab.sum(axis=0))
    for j, (lab, (lo, hi)) in enumerate(zip(LABELS, resample(tab))):
        print(f"    {lab}: observed {obs[j]:+.4f}; 95% interval over resampled territories [{lo:+.3f}, {hi:+.3f}]")
    left = [(r["city"].strip('"'), r["territory"].strip('"')) for r in rows if g(r, "prot1560") == 1 and g(r, "prot1600") == 0]
    print(f"    cities coded Protestant in 1560 and not in 1600: {left}")

    print("(7) present-day countries")
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[country(r)][course(r)] += 1
    single = sorted((k for k, v in by.items() if len(v) == 1), key=lambda k: -sum(by[k].values()))
    print(f"    countries: {len(by)}; countries all of whose cities share one course: {len(single)}, with {sum(sum(by[k].values()) for k in single)} cities")
    print("    " + ", ".join(f"{k} {''.join(map(str, next(iter(by[k]))))} ({sum(by[k].values())})" for k in single))

    def where(pred, label):
        cc = collections.Counter()
        for k, v in by.items():
            m = sum(x for c3, x in v.items() if pred(c3))
            if m:
                cc[k] = m
        print(f"    {label}: {sum(cc.values())} cities: " + ", ".join(f"{k} {m}" for k, m in cc.most_common()))

    where(lambda c3: c3[0] == 1, "Protestant in 1530")
    where(lambda c3: c3[0] == 0 and c3[1] == 1, "0 -> 1, 1530 -> 1560")
    where(lambda c3: c3[1] == 0 and c3[2] == 1, "0 -> 1, 1560 -> 1600")
    where(lambda c3: c3[0] == 1 and c3[1] == 0, "1 -> 0, 1530 -> 1560")
    where(lambda c3: c3[1] == 1 and c3[2] == 0, "1 -> 0, 1560 -> 1600")
    ctab = course_table(country_units(rows))
    for lab, (lo, hi) in list(zip(LABELS, resample(ctab)))[:2]:
        print(f"    {lab}: 95% interval over resampled countries [{lo:+.3f}, {hi:+.3f}]")

    print("(8) the 753 cities analysed in the published article")
    sub = [r for r in rows if r["pop1500"].strip() != "" and r["city"].strip('"').strip() not in ("Mainz", "Wittenberg", "Zürich")]
    m = len(sub)
    means = [sum(g(r, col) for r in sub) / m for col in ("prot1530", "prot1560", "prot1600", "press")]
    print(f"    rows with a recorded population in 1500, without Mainz, Wittenberg and Zurich: {m}")
    print("    means of prot1530, prot1560, prot1600, press: " + ", ".join(f"{x:.3f}" for x in means) + " (Table 3 of the article: 0.110, 0.292, 0.337, 0.234 for 753 cities)")
    assert m == 753 and [round(x, 3) for x in means] == [0.110, 0.292, 0.337, 0.234]
    sc = collections.Counter(course(r) for r in sub)
    s = lambda a=None, b=None, d=None: sum(v for k, v in sc.items() if (a is None or k[0] == a) and (b is None or k[1] == b) and (d is None or k[2] == d))
    print(f"    1517 -> 1530: 0 -> 1: {s(1)}/{m} = {s(1) / m:.4f}")
    for name, (n01, n0, n10, n1) in (("1530 -> 1560", (s(0, 1), s(0), s(1, 0), s(1))),
                                      ("1560 -> 1600", (s(None, 0, 1), s(None, 0), s(None, 1, 0), s(None, 1)))):
        _, pp = fisher_exact([[n01, n0 - n01], [n10, n1 - n10]])
        print(f"    {name}: 0 -> 1: {n01}/{n0} = {n01 / n0:.4f}; 1 -> 0: {n10}/{n1} = {n10 / n1:.4f}; Fisher exact two-sided p = {pp:.3e}")
    _, pm = fisher_exact([[s(0, 1, 1), s(0, 1, 0)], [s(1, 1, 1), s(1, 1, 0)]])
    print(f"    Markov property at 1560: Protestant in 1600 among those Protestant in 1560: changed after 1530: {s(0, 1, 1)}/{s(0, 1)}; "
          f"Protestant since 1530: {s(1, 1, 1)}/{s(1, 1)}; Fisher exact two-sided p = {pm:.4f}")
    stab, sctab = course_table(territory_units(sub)), course_table(country_units(sub))
    sobs = diffs(stab.sum(axis=0))
    for j, (lab, (lo, hi), (clo, chi)) in enumerate(zip(LABELS, resample(stab), resample(sctab))):
        print(f"    {lab}: observed {sobs[j]:+.4f}; 95% interval over resampled territories ({stab.shape[0]} units) [{lo:+.3f}, {hi:+.3f}]"
              + (f", over resampled countries ({sctab.shape[0]}) [{clo:+.3f}, {chi:+.3f}]" if j < 2 else ""))
    for pr, (k, mm) in zip((1, 0), press_shares(sub)):
        print(f"    press by 1500 = {pr}: Protestant in 1600: {k}/{mm} = {k / mm:.4f}")

    print("(9) the estimation samples of the published article")
    for col, target, clusters in (("prot1530", 200, 102), ("prot1560", 352, 157), ("prot1600", 307, 147)):
        kept = list(sub)
        dropped = []
        changed = True
        while changed:                    # leave out every nation or Imperial Circle whose cities all have the same code
            changed = False
            for d in NATIONS + CIRCLES:
                grp = [r for r in kept if g(r, d) == 1]
                if grp and len({g(r, col) for r in grp}) == 1:
                    kept = [r for r in kept if g(r, d) == 0]
                    dropped.append(f"{d} ({len(grp)}, code {g(grp[0], col)})")
                    changed = True
        nterr = len({r["territory"] for r in kept})
        (kp, mp), (kn, mn) = press_shares(kept, col)
        print(f"    {col}: {len(kept)} cities in {nterr} territories (article: {target} in {clusters}); "
              f"press by 1500: {kp}/{mp} = {kp / mp:.4f}; no press: {kn}/{mn} = {kn / mn:.4f}")
        print("        left out: " + ", ".join(dropped))
        assert (len(kept), nterr) == (target, clusters)


if __name__ == "__main__":
    main(sys.argv[1])

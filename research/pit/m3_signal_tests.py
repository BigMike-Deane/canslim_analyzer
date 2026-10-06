# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M3: the pre-registered signal tests (docs/phase2-pit-backtest-plan.md,
"M3 pre-registration"). Reads META_DIR/m3_panel.csv.gz. Implements the rules
as written; reporting-only extras are labelled as such.

Excess = stock return - equal-weight universe return on that date.
Non-overlapping 20-session windows = every 2nd panel date (panel step 10).
Returns are winsorized at [-90%, +300%] to stop data glitches dominating means.
"""
import itertools

import numpy as np
import pandas as pd

from common import META_DIR

TRAIN = ("2016-01-01", "2021-12-31")
TEST = ("2022-01-01", "2026-12-31")
COMP = ["c", "a", "n", "s", "l", "i"]


def load():
    d = pd.read_csv(META_DIR / "m3_panel.csv.gz", parse_dates=["date"])
    for h in (10, 20, 60):
        d[f"r{h}"] = d[f"r{h}"].clip(-0.9, 3.0)
        d[f"x{h}"] = d[f"r{h}"] - d.groupby("date")[f"r{h}"].transform("mean")
    dates = sorted(d.date.unique())
    d["nonoverlap"] = d.date.isin(dates[::2])
    return d


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan


def quintile_spread(d, col, h=20):
    """Per-date top-minus-bottom quintile mean excess, on non-overlapping dates."""
    g = d[d.nonoverlap & d[f"x{h}"].notna()].copy()
    g["q"] = g.groupby("date")[col].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
    per = g.groupby(["date", "q"])[f"x{h}"].mean().unstack()
    return (per[4] - per[0]).dropna()


def window(s, lo_hi):
    return s[(s.index >= lo_hi[0]) & (s.index <= lo_hi[1])]


def summarize(name, s):
    full, tr, te = s, window(s, TRAIN), window(s, TEST)
    print(f"  {name:28s} full {full.mean()*1e4:+7.0f} bps (t {tstat(full):+.2f}, n {len(full)}) | "
          f"2016-21 {tr.mean()*1e4:+6.0f} (t {tstat(tr):+.2f}) | 2022-26 {te.mean()*1e4:+6.0f} (t {tstat(te):+.2f})")
    return full, tr, te


def main():
    d = load()
    print(f"panel: {len(d):,} rows, {d.date.nunique()} dates, {d.cik.nunique():,} companies, "
          f"median {d.groupby('date').size().median():.0f} stocks/date; "
          f"20d windows cut short by delisting: {d.cut20.mean():.2%}")

    print("\nH1 (GATE): top-minus-bottom quintile of TOTAL score, 20-session excess")
    full, tr, te = summarize("total score", quintile_spread(d, "total"))
    h1 = full.mean() > 0 and tstat(full) >= 2.0 and te.mean() > 0
    print(f"  => H1 {'PASS' if h1 else 'FAIL'} (needs full mean > 0 with t >= 2.0 AND 2022-26 mean > 0)")
    print("  reporting only:")
    summarize("total, 10-session", quintile_spread(d, "total", 10))
    summarize("total, 60-session", quintile_spread(d, "total", 60))
    for c in COMP:
        summarize(f"component {c.upper()}", quintile_spread(d, c))
    band = d[d.nonoverlap & d.x20.notna()]
    b72 = band[band.total >= 72].groupby("date").x20.mean()
    summarize("72+ band vs universe", b72)
    spy_x = (band.groupby("date").r20.mean() - band.groupby("date").spy20.first())
    summarize("universe (eq-wt) vs SPY", spy_x)

    print("\nH2: 60-72 band minus 72+ band, 20-session excess")
    g = band.assign(b=np.select([band.total >= 72, band.total >= 60], ["72+", "60-72"], "lo"))
    per = g[g.b != "lo"].groupby(["date", "b"]).x20.mean().unstack().dropna()
    full, tr, te = summarize("60-72 minus 72+", per["60-72"] - per["72+"])
    h2 = tr.mean() > 0 and tstat(tr) >= 2 and te.mean() > 0
    print(f"  => H2 {'SUPPORTED' if h2 else 'NOT supported'} (needs 2016-21 > 0, t >= 2, AND 2022-26 > 0)")

    print("\nH3: reweight C/A/N/S/L/I (integer 0-3) on 2016-21, one look at 2022-26")
    tr_d = d[(d.date >= TRAIN[0]) & (d.date <= TRAIN[1])]
    best, best_w = -np.inf, None
    for w in itertools.product(range(4), repeat=6):
        if sum(w) == 0:
            continue
        sc = sum(wi * tr_d[c] for wi, c in zip(w, COMP))
        m = quintile_spread(tr_d.assign(w=sc), "w").mean()
        if m > best:
            best, best_w = m, w
    te_d = d[(d.date >= TEST[0]) & (d.date <= TEST[1])]
    tuned = quintile_spread(te_d.assign(w=sum(wi * te_d[c] for wi, c in zip(best_w, COMP))), "w")
    base = quintile_spread(te_d.assign(w=sum(te_d[c] for c in COMP)), "w")
    print(f"  best train weights {dict(zip([c.upper() for c in COMP], best_w))} (train spread {best*1e4:+.0f} bps)")
    print(f"  2022-26: tuned {tuned.mean()*1e4:+.0f} bps (t {tstat(tuned):+.2f}) vs current {base.mean()*1e4:+.0f} bps (t {tstat(base):+.2f})")
    print(f"  => H3 {'SUPPORTED' if tuned.mean() > base.mean() else 'NOT supported'}")

    print("\nH4: momentum (L, N, S) spread by SPY vs its 50-day MA on the score date")
    from common import load_prices
    spy = load_prices("SPY").Close
    above = (spy > spy.rolling(50).mean()).rename("above")
    for c in ["l", "n", "s"]:
        sp = quintile_spread(d, c)
        reg = above.reindex(sp.index)
        diff_tr = window(sp[reg], TRAIN).mean() - window(sp[~reg], TRAIN).mean()
        diff_te = window(sp[reg], TEST).mean() - window(sp[~reg], TEST).mean()
        # t for a difference of means (Welch)
        a, b = window(sp[reg], TRAIN), window(sp[~reg], TRAIN)
        t = diff_tr / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 2 and len(b) > 2 else np.nan
        ok = diff_tr > 0 and t >= 2 and np.sign(diff_te) == np.sign(diff_tr)
        print(f"  {c.upper()}: above-minus-below 2016-21 {diff_tr*1e4:+.0f} bps (t {t:+.2f}), 2022-26 {diff_te*1e4:+.0f} bps "
              f"=> {'SUPPORTED' if ok else 'not supported'}")


if __name__ == "__main__":
    main()

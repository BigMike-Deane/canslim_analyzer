# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
"""Batch 2 / Score v2 tests (pre-registered in docs/phase2-pit-backtest-plan.md,
"Batch 2 / Score v2 pre-registration").

Universe: M3 panel v2, top 1,000 by market cap per date.
(a) top - bottom quintile 60-session excess, every panel date, Newey-West t
    (6 lags) >= 3 full, mean > 0 in 2016-21 and 2022-26
(b) b1 top-quintile EW - SPY - cost, or b2 cap-weighted ex-bottom-quintile - SPY
    - cost, > 0 in both halves (turnover between dates 6 apart x 19 bps)
PASS = (a) and (b).
"""
import numpy as np
import pandas as pd

from common import META_DIR

TRAIN = ("2016-01-01", "2021-12-31")
TEST = ("2022-01-01", "2026-12-31")
RT_COST, LAG, TOP_N = 0.0019, 6, 1000


def nw_t(s, lags=LAG):
    x = pd.Series(s).dropna().to_numpy()
    n = len(x)
    if n < lags + 3:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(var / n) if var > 0 else np.nan


def win(s, lo_hi):
    return s[(s.index >= lo_hi[0]) & (s.index <= lo_hi[1])]


def line(name, s, nw=True):
    tr, te = win(s, TRAIN), win(s, TEST)
    t = nw_t(s) if nw else s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
    print(f"  {name:34s} full {s.mean()*1e4:+7.0f} bps (t {t:+.2f}, n {len(s)}) | "
          f"2016-21 {tr.mean()*1e4:+6.0f} | 2022-26 {te.mean()*1e4:+6.0f}")
    return s.mean(), t, tr.mean(), te.mean()


def quint(d, col):
    g = d[d[col].notna()].copy()
    g["q"] = g.groupby("date")[col].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
    return g


def spread(g, h):
    per = g[g[f"x{h}"].notna()].groupby(["date", "q"])[f"x{h}"].mean().unstack()
    return (per[4] - per[0]).dropna()


def turnover(members, weights=None):
    """Share of the portfolio replaced vs LAG panel dates earlier (one rebalance)."""
    dates = list(members.index)
    out = {}
    for i, dt in enumerate(dates):
        if i < LAG:
            continue
        now, before = members.iloc[i], members.iloc[i - LAG]
        if weights is None:
            out[dt] = 1 - len(now & before) / len(now)
        else:
            w = weights.iloc[i]
            out[dt] = sum(w.get(c, 0) for c in now - before)
    return pd.Series(out)


def long_only(g):
    top = g[(g.q == 4) & g.r60.notna()]
    b1_gross = top.groupby("date").r60.mean() - top.groupby("date").spy60.first()
    b1 = (b1_gross - turnover(top.groupby("date").cik.apply(set)) * RT_COST).dropna()
    keep = g[(g.q > 0) & g.r60.notna()].copy()
    keep["w"] = keep.mcap / keep.groupby("date").mcap.transform("sum")
    cw = (keep.w * keep.r60).groupby(keep.date).sum() - keep.groupby("date").spy60.first()
    wts = keep.groupby("date")[["cik", "w"]].apply(lambda x: dict(zip(x.cik, x.w)))
    b2 = (cw - turnover(keep.groupby("date").cik.apply(set), wts) * RT_COST).dropna()
    return b1, b2


def main():
    from m3_signal_tests import load
    d = load()
    d = d[d.groupby("date").mcap.rank(ascending=False, method="first") <= TOP_N].copy()
    for h in (20, 60):
        d[f"x{h}"] = d[f"r{h}"] - d.groupby("date")[f"r{h}"].transform("mean")
    d = d.merge(pd.read_csv(META_DIR / "p3_signals.csv.gz", parse_dates=["date"]), on=["cik", "date"], how="left")
    d = d.merge(pd.read_csv(META_DIR / "p4_signals.csv.gz", parse_dates=["date"]), on=["cik", "date"], how="left")
    rk = lambda c: d.groupby("date")[c].rank(pct=True)  # noqa: E731
    d["V"] = pd.concat([rk("ep"), rk("bm")], axis=1).mean(axis=1)
    d["LV"] = d.lv
    d["Q"] = pd.concat([rk("roe"), rk("nacc")], axis=1).mean(axis=1)
    parts = pd.concat([rk("V"), rk("LV"), rk("Q"), rk("s1"), rk("s3")], axis=1)
    d["SCORE_v2"] = parts.mean(axis=1).where(parts.notna().sum(axis=1) >= 4)
    d["logcap"] = np.log(d.mcap)

    print(f"universe: top {TOP_N} by mcap/date; {len(d):,} rows, {d.date.nunique()} dates, "
          f"{d.cik.nunique():,} companies")
    names = {"V": "V value (E/P + B/M)", "LV": "LV low volatility", "Q": "Q quality (ROE + low accruals)",
             "SCORE_v2": "SCORE v2 (V, LV, Q, PEAD, issuance)"}
    verdict = {}
    for c, name in names.items():
        g = quint(d, c)
        rc = d[[c, "logcap"]].corr("spearman").iloc[0, 1]
        print(f"\n{name}  (coverage {d[c].notna().mean():.0%}, rank corr with log mcap {rc:+.2f})")
        m, t, tr, te = line("(a) top-bottom 60d (NW t)", spread(g, 60))
        a_ok = m > 0 and t >= 3.0 and tr > 0 and te > 0
        b1, b2 = long_only(g)
        _, _, b1tr, b1te = line("(b1) top quintile EW - SPY, net", b1)
        _, _, b2tr, b2te = line("(b2) cap-wt ex-bottom - SPY, net", b2)
        b_ok = (b1tr > 0 and b1te > 0) or (b2tr > 0 and b2te > 0)
        print("  reporting only:")
        line("top-bottom 20d (NW t)", spread(g, 20))
        verdict[name] = "PASS" if a_ok and b_ok else f"FAIL ((a) {'ok' if a_ok else 'no'}, (b) {'ok' if b_ok else 'no'})"
        print(f"  => {verdict[name]}")

    print("\nReporting only (seen in phase 3, not evidence): top-bottom 60d in this universe")
    for c, name in {"s1": "S1 PEAD", "s2": "S2 gross profitability", "s3": "S3 net issuance",
                    "s4": "S4 momentum 12-1"}.items():
        line(name, spread(quint(d, c), 60))
    print("\nSummary:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print("\n=> BATCH 2 " + ("has a passing signal" if any(v == "PASS" for v in verdict.values())
                             else "FAIL: nothing passes -> recommend index funds, research closes"))


if __name__ == "__main__":
    main()

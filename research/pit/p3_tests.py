# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
"""Phase 3 signal-lab tests (pre-registered in docs/phase2-pit-backtest-plan.md,
"Phase 3 signal-lab pre-registration"). Reads m3_panel.csv.gz + p3_signals.csv.gz.

(a) top-bottom quintile 20d excess spread: full t >= 3, > 0 in 2016-21 and 2022-26
(b) top-quintile eq-wt 20d return - SPY 20d - turnover x 19 bps > 0 in both halves
PASS = (a) and (b).
"""
import numpy as np
import pandas as pd

from common import META_DIR
from m3_signal_tests import load, quintile_spread, summarize, tstat

SIGS = {"s1": "S1 PEAD", "s2": "S2 gross profitability", "s3": "S3 net issuance (buybacks +)",
        "s4": "S4 momentum 12-1", "s5": "S5 composite"}
RT_COST = 0.0019


def top_minus_spy(d, col, bench="spy20"):
    g = d[d.nonoverlap & d.x20.notna() & d[col].notna()].copy()
    g["q"] = g.groupby("date")[col].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
    top = g[g.q == 4]
    ret = top.groupby("date").r20.mean() - top.groupby("date")[bench].first()
    members = top.groupby("date").cik.apply(set)
    prev, turn = None, {}
    for dt, s in members.items():
        turn[dt] = 1.0 if prev is None else 1 - len(s & prev) / len(s)
        prev = s
    turn = pd.Series(turn)
    return ret - turn * RT_COST, turn


def main():
    d = load()
    s = pd.read_csv(META_DIR / "p3_signals.csv.gz", parse_dates=["date"])
    d = d.merge(s, on=["cik", "date"], how="left")
    ranks = d.groupby("date")[["s1", "s2", "s3", "s4"]].rank(pct=True)
    d["s5"] = ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= 3)
    d["logcap"] = np.log(d.mcap)

    print(f"panel {len(d):,} rows; coverage " + " ".join(f"{c} {d[c].notna().mean():.0%}" for c in SIGS))
    verdict = {}
    for c, name in SIGS.items():
        sub = d[d[c].notna()]
        sub = sub[sub.groupby("date")[c].transform("size") >= 50]
        print(f"\n{name}  (rank corr with log mcap {sub[[c, 'logcap']].corr('spearman').iloc[0, 1]:+.2f})")
        full, tr, te = summarize("(a) top-bottom quintile 20d", quintile_spread(sub, c))
        a_ok = full.mean() > 0 and tstat(full) >= 3.0 and tr.mean() > 0 and te.mean() > 0
        net, turn = top_minus_spy(sub, c)
        _, ntr, nte = summarize("(b) top quintile - SPY, net", net)
        b_ok = ntr.mean() > 0 and nte.mean() > 0
        print(f"      top-quintile turnover per 20d: {turn.iloc[1:].mean():.0%}")
        print("  reporting only:")
        summarize("top-bottom 10d", quintile_spread(sub, c, 10))
        summarize("top-bottom 60d", quintile_spread(sub, c, 60))
        summarize("top quintile - IWM, net", top_minus_spy(sub, c, "iwm20")[0])
        verdict[name] = "PASS" if a_ok and b_ok else f"FAIL ((a) {'ok' if a_ok else 'no'}, (b) {'ok' if b_ok else 'no'})"
        print(f"  => {verdict[name]}")
    print("\nSummary:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print("\n=> LAB " + ("has a signal for M4" if any(v == "PASS" for v in verdict.values())
                         else "FAIL: no signal passes -> recommend index funds"))


if __name__ == "__main__":
    main()

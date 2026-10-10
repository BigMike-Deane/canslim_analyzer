# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Live CANSLIM 2.0 vs the research definitions (parity check; owner rule: test logic must
match what trades live).

Recomputes the six v5b signals with the RESEARCH code (m2_adapter.surprise_asof,
p4_signals.fundamentals ROE, p3_signals.issuance, p8_analyst n_brokers, FINRA short
interest) as of D for every live ticker, then compares with the live inputs
(backend/canslim2.py) per signal and on the combined score, both ranked on the common set.

  python3 c2_parity.py live_scores.csv [--asof 2026-10-05]
live_scores.csv: ticker, score, beat_streak, surprise_pct, roe, s3, dtc, n_brokers (export of canslim2_scores)
"""
import argparse

import numpy as np
import pandas as pd

import m2_adapter as m
import p3_signals as p3
import p4_signals as p4
import p8_analyst as p8
from common import META_DIR

FEATS = {"beat_streak": 1, "surprise_pct": 1, "roe": 1, "s3": 1, "dtc": -1, "n_brokers": 1}


def research_features(tickers, d):
    t = pd.read_csv(META_DIR / "v3_table.csv.gz", usecols=["cik", "date", "symbol"], parse_dates=["date"])
    sym2cik = t.sort_values("date").drop_duplicates("symbol", keep="last").set_index("symbol").cik.to_dict()
    f = pd.concat([p4._read(META_DIR / "sec_facts.csv.gz", ["NetIncomeLoss", "StockholdersEquity"]),
                   p4._read(META_DIR / "sec_facts_q.csv.gz", ["Assets", "NetCashProvidedByUsedInOperatingActivities"])])
    f["days"] = (f.end - f.start).dt.days
    p4.FUND = {cik: g for cik, g in f.groupby("cik")}
    si = pd.read_csv(META_DIR / "short_interest.csv.gz", parse_dates=["settle", "known"])
    si = si[si.known <= d].sort_values("settle").drop_duplicates("symbol", keep="last").set_index("symbol")
    rows = []
    for tk in tickers:
        cik = sym2cik.get(tk)
        if cik is None:
            continue
        sp, bs = m.surprise_asof(tk, d)
        dtc = si.dtc.get(tk.replace("-", ".")) if tk.replace("-", ".") in si.index else si.dtc.get(tk)
        rows.append({"ticker": tk, "cik": cik, "beat_streak": bs, "surprise_pct": sp,
                     "roe": p4.fundamentals(cik, d, 1.0)["roe"], "s3": p3.issuance(cik, m._shares_table(cik), d),
                     "dtc": dtc if dtc is not None and dtc < 999 else np.nan})
    r = pd.DataFrame(rows)
    ev, _ = p8.load_events(set(r.cik))
    nb = p8.am_signal(pd.DataFrame({"cik": r.cik, "date": d}), ev)[["cik", "n_brokers"]]
    return r.merge(nb, on="cik", how="left")


def score(df):
    return sum(sg * (df[f].rank(pct=True) - 0.5).fillna(0) for f, sg in FEATS.items()) / len(FEATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("live")
    ap.add_argument("--asof", default="2026-10-05")
    a = ap.parse_args()
    d = pd.Timestamp(a.asof)
    live = pd.read_csv(a.live)
    m.identity(); m._facts()
    from common import _segments
    _segments()
    res = research_features(sorted(live.ticker), d)
    j = live.merge(res, on="ticker", suffixes=("_live", "_res"))
    print(f"live {len(live):,} tickers; matched to research CIKs {len(j):,}; research as of {d.date()}")
    print("\nper signal (Spearman on stocks where both sides have a value):")
    for f in FEATS:
        a_, b_ = j[f + "_live"], j[f + "_res"]
        ok = a_.notna() & b_.notna()
        rho = a_[ok].corr(b_[ok], method="spearman")
        print(f"  {f:13s} n={ok.sum():5d}  rho {rho:+.3f}  | live-only {int((a_.notna() & b_.isna()).sum())} "
              f"research-only {int((a_.isna() & b_.notna()).sum())}")
    sl = score(j.rename(columns={f + "_live": f for f in FEATS}))
    sr = score(j.rename(columns={f + "_res": f for f in FEATS}))
    rho = sl.corr(sr, method="spearman")
    top_l, top_r = set(j.ticker[sl.rank(pct=True) > 0.9]), set(j.ticker[sr.rank(pct=True) > 0.9])
    print(f"\ncombined score: Spearman {rho:+.3f} | top-decile overlap {len(top_l & top_r) / max(len(top_r), 1):.0%}")
    j["d_score"] = (sl.rank(pct=True) - sr.rank(pct=True)).abs()
    cols = ["ticker"] + [c for f in FEATS for c in (f + "_live", f + "_res")]
    print("\nlargest score disagreements:")
    print(j.sort_values("d_score", ascending=False)[cols].head(12).to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()

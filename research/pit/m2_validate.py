# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M2 gate: does the PIT score track the live score on the overlap window?

Live sample: meta/live_scores_sample.csv (7 dates Jul-14 -> Oct-2, every
scanned stock, exported read-only from prod stock_scores).
Gate (docs/phase2-pit-backtest-plan.md): per-date Spearman(total) >= 0.80 and
top-quintile overlap >= 60%. Per-component agreement explains any gap.

Note the live score is an intraday scan; PIT scores use D's close. Some
disagreement in N/S/L (price-driven) is expected from that alone.
"""
import pandas as pd

import m2_adapter as m

COMP = ["c", "a", "n", "s", "l", "i", "m"]


def main():
    live = pd.read_csv(m.META_DIR / "live_scores_sample.csv", parse_dates=["date"])
    ident = m.identity().reset_index()
    sym2cik = ident.dropna(subset=["symbol"]).drop_duplicates("symbol").set_index("symbol").cik
    live["cik"] = live.ticker.map(sym2cik)
    print(f"live rows {len(live):,}; mapped to a CIK {live.cik.notna().mean():.1%}", flush=True)

    out = []
    for d, g in live.dropna(subset=["cik"]).groupby("date"):
        mscore = m.market_score_asof(d)
        for r in g.itertuples():
            s = m.score_asof(int(r.cik), d, m_score=mscore)
            if s is None:
                continue
            out.append({"ticker": r.ticker, "date": d, "total": s.total_score,
                        **{k: getattr(s, f"{k}_score") for k in COMP}})
        print(f"  {d.date()}: scored {sum(o['date'] == d for o in out):,}/{len(g):,}", flush=True)
    pit = pd.DataFrame(out)
    j = live.merge(pit, on=["ticker", "date"], suffixes=("_live", "_pit"))
    j.to_csv(m.META_DIR / "m2_validation.csv.gz", index=False)

    rows = []
    for d, g in j.groupby("date"):
        q_live = g.total_live >= g.total_live.quantile(0.8)
        q_pit = g.total_pit >= g.total_pit.quantile(0.8)
        rows.append({"date": d.date(), "n": len(g),
                     "spearman_total": g.total_live.corr(g.total_pit, method="spearman"),
                     "top20_overlap": (q_live & q_pit).sum() / max(q_live.sum(), 1),
                     **{f"rho_{k}": g[f"{k}_live"].corr(g[f"{k}_pit"], method="spearman") for k in COMP}})
    rep = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(rep.round(2).to_string(index=False))
    print("\nmean |live - pit| per component:")
    print({k: round((j[f"{k}_live"] - j[f"{k}_pit"]).abs().mean(), 2) for k in COMP + ["total"]})
    ok = (rep.spearman_total >= 0.80).all() and (rep.top20_overlap >= 0.60).all()
    print(f"\nM2 GATE: {'PASS' if ok else 'FAIL'} (Spearman >= 0.80 and top-20% overlap >= 60% on every date)")


if __name__ == "__main__":
    main()

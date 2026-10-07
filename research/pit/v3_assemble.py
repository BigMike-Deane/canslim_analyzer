# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""Score v3 modeling table (docs/score-v3-plan.md): one row per (cik, panel date)
in the v3 universe with every pre-registered raw feature and the target.

Universe: close > $5, mcap >= $1B (glitch rows dropped by load()), 20d $ volume >= $5M.
Target y = forward 60-session stock return - SPY 60-session TOTAL return (dividend-
adjusted), both from D's close; winsorized per date at 1/99% later, in the model.
Stock returns are split-adjusted price returns (no dividends) -> slightly
conservative for the portfolio; irrelevant for ranking.

Output: META_DIR/v3_table.csv.gz
"""
import glob
import gzip
import json

import numpy as np
import pandas as pd

from common import DATA_DIR, META_DIR
from m3_signal_tests import load

FEATURES = [
    # CANSLIM inputs
    "c", "a", "n", "s", "l", "i", "eps_growth", "annual_cagr", "surprise_pct", "beat_streak", "inst_pct",
    # earnings / fundamentals
    "s1", "s2", "s3", "ep", "bm", "roe", "nacc",
    # price / volume
    "ret1m", "mom3", "mom6", "s4", "dist52", "vol60", "max21", "beta252", "logcap", "logdvol", "voltrend",
    # setups
    "pct_from_pivot", "breaking_out",
    # analyst
    "am", "n_brokers",
    # insider
    "ins_buyers90", "ins_net90",
    # short interest
    "si_pct", "dtc", "si_chg3m",
]
CONTEXT = ["spy_200", "spy_3m"]  # same for every stock on a date (tree model only)


def spy_tr():
    d = json.load(gzip.open(DATA_DIR / "timing" / "SPY.json.gz"))["adj"]
    return pd.Series({pd.Timestamp(r["date"]): r["adjClose"] for r in d}).sort_index()


def add_insider(t):
    ins = pd.read_csv(META_DIR / "insider_trades.csv.gz", parse_dates=["filed"])
    ins = ins[ins.issuer_cik.isin(set(t.cik))]
    out = []
    for d, g in t.groupby("date"):
        w = ins[(ins.filed < d) & (ins.filed >= d - pd.Timedelta(days=90)) & ins.issuer_cik.isin(g.cik)]
        buyers = w[w.code == "P"].groupby("issuer_cik").owner_cik.nunique()
        net = w.assign(sv=np.where(w.code == "P", w.value, -w.value)).groupby("issuer_cik").sv.sum()
        x = g[["cik", "date", "mcap"]].copy()
        x["ins_buyers90"] = x.cik.map(buyers).fillna(0)
        x["ins_net90"] = x.cik.map(net).fillna(0) / x.mcap
        out.append(x.drop(columns="mcap"))
    return t.merge(pd.concat(out), on=["cik", "date"], how="left")


def add_short_interest(t):
    si = pd.read_csv(META_DIR / "short_interest.csv.gz", parse_dates=["settle", "known"])
    si = si[si.symbol.isin(set(t.symbol))].sort_values("known")
    first = si.known.min()
    base = t[["cik", "date", "symbol", "close", "mcap"]].sort_values("date")
    now = pd.merge_asof(base, si[["symbol", "known", "si", "dtc"]], left_on="date", right_on="known",
                        by="symbol", direction="backward", tolerance=pd.Timedelta(days=45))
    then = pd.merge_asof(base.assign(d3=base.date - pd.Timedelta(days=91)).sort_values("d3"),
                         si[["symbol", "known", "si"]].rename(columns={"si": "si_then"}),
                         left_on="d3", right_on="known", by="symbol", direction="backward",
                         tolerance=pd.Timedelta(days=45))
    now = now.merge(then[["cik", "date", "si_then"]], on=["cik", "date"], how="left")
    shares = now.mcap / now.close
    now["si_pct"] = now.si / shares
    now["si_chg3m"] = np.log((now.si + 1) / (now.si_then + 1))
    now["dtc"] = now.dtc.where(now.dtc < 999)  # FINRA codes "no volume" as 999.99
    now.loc[now.date < first, ["si_pct", "dtc", "si_chg3m"]] = np.nan
    return t.merge(now[["cik", "date", "si_pct", "dtc", "si_chg3m"]], on=["cik", "date"], how="left")


def main():
    t = load()[["cik", "date", "symbol", "close", "mcap", "c", "a", "n", "s", "l", "i", "r20", "r60", "cut60"]]
    pf = pd.read_csv(META_DIR / "v3_price_features.csv.gz", parse_dates=["date"])
    t = t.merge(pf, on=["cik", "date"], how="left")
    t = t[(t.close > 5) & (t.mcap >= 1e9) & (t.dvol20 >= 5e6)].copy()
    print(f"universe: {len(t):,} rows, {t.cik.nunique():,} companies, {t.date.nunique()} dates, "
          f"median {t.groupby('date').size().median():.0f} names/date", flush=True)

    dates = set(t.date)
    cols = ["cik", "date", "eps_growth", "annual_cagr", "surprise_pct", "beat_streak", "inst_pct", "sector"]
    daily = pd.concat((pd.read_csv(f, usecols=cols, parse_dates=["date"])
                       for f in sorted(glob.glob(str(META_DIR / "daily" / "*.csv.gz")))), ignore_index=True)
    t = t.merge(daily[daily.date.isin(dates)], on=["cik", "date"], how="left")
    for f, keep in (("p3_signals.csv.gz", ["s1", "s2", "s3", "s4"]), ("p4_signals.csv.gz", ["ep", "bm", "roe", "nacc"]),
                    ("p5_setups.csv.gz", ["pct_from_pivot", "breaking_out"])):
        x = pd.read_csv(META_DIR / f, parse_dates=["date"], usecols=["cik", "date"] + keep)
        t = t.merge(x, on=["cik", "date"], how="left")
    t["breaking_out"] = t.breaking_out.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    import p8_analyst as p8
    ev, _ = p8.load_events(set(t.cik))
    t = t.merge(p8.am_signal(t[["cik", "date"]], ev), on=["cik", "date"], how="left")
    t = add_insider(t)
    t = add_short_interest(t)
    t["logcap"], t["logdvol"] = np.log(t.mcap), np.log(t.dvol20)

    spy = spy_tr()
    sess = spy.index
    def fwd_spy(d, h=60):
        i = sess.searchsorted(d)
        return spy.iloc[min(i + h, len(spy) - 1)] / spy.iloc[i] - 1
    def ctx(d):
        s = spy[:d]
        return pd.Series({"spy_200": s.iloc[-1] / s.iloc[-200:].mean() - 1, "spy_3m": s.iloc[-1] / s.iloc[-64] - 1,
                          "spy60_tr": fwd_spy(d), "spy20_tr": fwd_spy(d, 20)})
    c = pd.DataFrame({d: ctx(d) for d in sorted(dates)}).T.rename_axis("date").reset_index()
    t = t.merge(c, on="date", how="left")
    t["y"] = t.r60 - t.spy60_tr

    keep = ["cik", "date", "symbol", "sector", "close", "mcap", "dvol20", "r20", "r60", "cut60", "spy20_tr", "spy60_tr", "y"]
    t = t[keep + FEATURES + CONTEXT]
    t.to_csv(META_DIR / "v3_table.csv.gz", index=False)
    cov = t[FEATURES].notna().mean().sort_values()
    print("feature coverage (lowest 12):")
    print("  " + "  ".join(f"{k} {v:.0%}" for k, v in cov.head(12).items()))
    print(f"wrote {len(t):,} rows; y present {t.y.notna().mean():.0%}", flush=True)


if __name__ == "__main__":
    main()

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
"""International replication of A1 / A4 / A5 (docs/exposure-plan.md). Price indexes + a constant
assumed dividend yield; local cash rates; same engine/costs/delay as a_exposure.

  python3 a_intl.py
"""
import gzip
import json

import numpy as np
import pandas as pd

import a_exposure as A
import a_exposure2 as A2
from common import DATA_DIR

T = DATA_DIR / "timing"
MARKETS = {  # name: (file, dividend yield, rate pieces [(fred_id, until)])
    "Japan Nikkei 225": ("N225", 0.015, [("INTDSRJPM193N", "1985-06-30"), ("IRSTCI01JPM156N", None)]),
    "UK FTSE 100": ("FTSE", 0.035, [("IR3TIB01GBM156N", None)]),
    "Hong Kong Hang Seng": ("HSI", 0.030, [("TB3MS", "1953-12-31"), ("DTB3", None)]),
    "Euro Stoxx 50": ("STOXX50E", 0.030, [("IR3TIB01DEM156N", "1993-12-31"), ("IR3TIB01EZM156N", None)]),
}


def fred(fid):
    f = pd.read_csv(T / "fred" / f"{fid}.csv")
    s = pd.to_numeric(f.iloc[:, 1], errors="coerce")
    return pd.Series(s.to_numpy(), index=pd.to_datetime(f.iloc[:, 0])).dropna() / 100


def load(file, dy, pieces):
    g = pd.DataFrame(json.load(gzip.open(T / "intl" / f"{file}.json.gz")))
    px = pd.Series(g.price.to_numpy(), index=pd.to_datetime(g.date)).sort_index()
    px = px[px > 0]
    parts, prev_end = [], None
    for fid, until in pieces:
        s = fred(fid)
        if prev_end is not None:
            s = s[s.index > prev_end]
        if until:
            s = s[s.index <= until]
            prev_end = pd.Timestamp(until)
        parts.append(s)
    rate = pd.concat(parts).sort_index()
    rate = rate[~rate.index.duplicated()]
    first_rate = rate.index.min()
    rate = rate.reindex(px.index, method="ffill")
    px = px[px.index >= first_rate]
    rate = rate.reindex(px.index)
    return px, px.pct_change().fillna(0) + dy / 252, rate


def evaluate(px, tr, rate, start):
    rules = {"A1": 2 * (px > px.rolling(200).mean()).astype(float), "A4": A2.a4_exposure(px),
             "A5": A2.a5_exposure(px, tr, rate)}
    end = px.index[-1]
    out = {"bh": A.stats(tr, rate, start, end)}
    for k, ex in rules.items():
        r, e = A.run(px, tr, rate, ex)
        out[k] = A.stats(r, rate, start, end, e)
        out[k]["series"] = r
    return out


def main():
    tally = {k: {"beat": 0, "dd": 0} for k in ("A1", "A4", "A5")}
    for name, (file, dy, pieces) in MARKETS.items():
        px, tr, rate = load(file, dy, pieces)
        start = px.index[min(260, len(px) - 1)]
        res = evaluate(px, tr, rate, start)
        bh = res["bh"]
        print(f"\n{name} ({start.date()} -> {px.index[-1].date()}, dividend {dy:.1%}):")
        print(f"  buy & hold: CAGR {bh['cagr']:.2%} | max DD {bh['dd']:.1%} | Sharpe {bh['sharpe']:.2f} | worst yr {bh['worst_yr']:+.1%}")
        for k in ("A1", "A4", "A5"):
            s = res[k]
            b, d = s["cagr"] > bh["cagr"], s["dd"] < bh["dd"]
            tally[k]["beat"] += b
            tally[k]["dd"] += d
            sens = []
            for adj in (-0.01, 0.01):
                _, tr2, _ = load(file, dy + adj, pieces)
                r2 = evaluate(px, tr2, rate, start)
                sens.append(f"div {dy + adj:.1%}: {r2[k]['cagr'] - r2['bh']['cagr']:+.2%}")
            print(f"  {k}: CAGR {s['cagr']:.2%} ({s['cagr'] - bh['cagr']:+.2%}) | max DD {s['dd']:.1%} | Sharpe {s['sharpe']:.2f} | "
                  f"worst yr {s['worst_yr']:+.1%} | invested {s['invested']:.0%} | excess at {', '.join(sens)}")
        if file == "N225":
            lo, hi = "1990-01-01", "2012-12-31"
            sb = A.stats(tr, rate, lo, hi)
            print(f"  Japan 1990-2012: buy & hold {sb['cagr']:.2%} (DD {sb['dd']:.1%}) | " +
                  " | ".join(f"{k} {A.stats(res[k]['series'], rate, lo, hi)['cagr']:.2%} (DD {A.stats(res[k]['series'], rate, lo, hi)['dd']:.1%})"
                             for k in ("A1", "A4", "A5")))
    print("\nTally (markets beaten / shallower drawdown, of 4):")
    for k, v in tally.items():
        ok = v["beat"] >= 3 and v["dd"] >= 3
        print(f"  {k}: CAGR beat {v['beat']}/4, DD shallower {v['dd']}/4 => {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()

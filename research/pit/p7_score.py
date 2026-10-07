# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Score the H7 vintages against the pre-registered rule (plan doc "H7 pre-registration").

PASS (all required): the median vintage's CAGR > SPY total-return CAGR over 2016-21, over 2022-26
and over the full period. SPY TR = FMP dividend-adjusted closes, measured over the SAME dates as
the vintage it is compared with (each vintage starts on its own offset session). The median
vintage is picked per window by CAGR; the median of per-vintage excess CAGR is reported beside it.

Also reported (not gating): Sharpe (daily, annualized, rf = 0), max drawdown vs SPY, trades,
win rate (closed sells), and the share of total profit from the single best trade -- both the
best single sell and the best position (all realized gains on one ticker between flat points).

Usage: python3 p7_score.py [--min-vintages 5]   -> meta/h7/h7_score.json + printed table
"""
import argparse
import gzip
import json
import statistics as st

import numpy as np
import pandas as pd

from common import DATA_DIR, META_DIR

WINDOWS = {"2016-21": ("2016-01-01", "2021-12-31"), "2022-26": ("2022-01-01", "2026-12-31"),
           "full": ("2016-01-01", "2026-12-31")}


def spy_tr():
    d = json.load(gzip.open(DATA_DIR / "timing" / "SPY.json.gz"))["adj"]
    return pd.Series({r["date"]: r["adjClose"] for r in d}).sort_index()


def cagr(s):
    yrs = (pd.Timestamp(s.index[-1]) - pd.Timestamp(s.index[0])).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1


def sharpe(s):
    r = s.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def max_dd(s):
    return float((1 - s / s.cummax()).max())


def window(s, lo, hi):
    return s[(s.index >= lo) & (s.index <= hi)]


def best_trade_share(trades, profit):
    sells = [t for t in trades if t["action"] == "SELL" and t["gain"] is not None]
    best_sell = max((t["gain"] for t in sells), default=0.0)
    # position = realized gains on one ticker between flat points (partial sells + final exit)
    pos, held, open_gain = [], {}, {}
    for t in sorted(trades, key=lambda t: t["date"]):
        k = t["ticker"]
        if t["action"] in ("BUY", "PYRAMID"):
            held[k] = held.get(k, 0.0) + t["shares"]
        elif t["action"] == "SELL":
            held[k] = held.get(k, 0.0) - t["shares"]
            open_gain[k] = open_gain.get(k, 0.0) + (t["gain"] or 0.0)
            if held[k] <= 1e-6:
                pos.append((k, t["date"], open_gain.pop(k)))
                held.pop(k)
    best_pos = max(pos, key=lambda p: p[2], default=(None, None, 0.0))
    wins = sum(1 for t in sells if t["gain"] > 0)
    return {"sells": len(sells), "win_rate": wins / len(sells) if sells else None,
            "best_sell": best_sell, "best_sell_share": best_sell / profit if profit > 0 else None,
            "best_position": {"ticker": best_pos[0], "closed": best_pos[1], "gain": best_pos[2]},
            "best_position_share": best_pos[2] / profit if profit > 0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-vintages", type=int, default=5)
    ap.add_argument("--variant", default="none", help="flat_reset = labeled diagnostic (meta/h7_flat_reset), never the verdict")
    a = ap.parse_args()
    spy = spy_tr()
    src = META_DIR / ("h7" if a.variant == "none" else f"h7_{a.variant}")
    vint = []
    for o in (0, 10, 20, 30, 40):
        f = src / f"vintage_{o}.json"
        if f.exists():
            vint.append(json.load(open(f)))
    if len(vint) < a.min_vintages:
        raise SystemExit(f"only {len(vint)} vintages finished (need {a.min_vintages})")

    rows = []
    for v in vint:
        eq = pd.Series(dict(v["equity"])).sort_index()
        sp = spy.reindex(eq.index).ffill()
        r = {"offset": v["offset"], "start": eq.index[0], "end": eq.index[-1], "trades": len(v["trades"]),
             "final": float(eq.iloc[-1])}
        for w, (lo, hi) in WINDOWS.items():
            e, s = window(eq, lo, hi), window(sp, lo, hi)
            r[w] = {"cagr": cagr(e), "spy_cagr": cagr(s), "excess": cagr(e) - cagr(s),
                    "sharpe": sharpe(e), "spy_sharpe": sharpe(s), "max_dd": max_dd(e), "spy_max_dd": max_dd(s)}
        r.update(best_trade_share(v["trades"], r["final"] - 25000))
        rows.append(r)

    verdict = {}
    for w in WINDOWS:
        ranked = sorted(rows, key=lambda r: r[w]["cagr"])
        med = ranked[len(ranked) // 2] if len(ranked) % 2 else None
        if med is None:  # even count (partial run): average the middle two for display only
            m1, m2 = ranked[len(ranked) // 2 - 1], ranked[len(ranked) // 2]
            mc, ms = (m1[w]["cagr"] + m2[w]["cagr"]) / 2, (m1[w]["spy_cagr"] + m2[w]["spy_cagr"]) / 2
        else:
            mc, ms = med[w]["cagr"], med[w]["spy_cagr"]
        verdict[w] = {"median_cagr": mc, "spy_cagr": ms, "pass": mc > ms,
                      "median_excess": st.median(r[w]["excess"] for r in rows),
                      "vintages_beating_spy": sum(r[w]["excess"] > 0 for r in rows)}
    final = all(verdict[w]["pass"] for w in WINDOWS) and len(rows) == 5

    print(f"H7{'' if a.variant == 'none' else ' DIAGNOSTIC (' + a.variant + ', not the verdict)'} — {len(rows)} vintages\n")
    print(f"{'off':>4} {'window':>8} {'CAGR':>7} {'SPY':>7} {'excess':>7} {'Sharpe':>6} {'SPYsh':>6} "
          f"{'maxDD':>6} {'SPYdd':>6}")
    for r in rows:
        for w in WINDOWS:
            x = r[w]
            print(f"{r['offset']:>4} {w:>8} {x['cagr']:7.1%} {x['spy_cagr']:7.1%} {x['excess']:+7.1%} "
                  f"{x['sharpe']:6.2f} {x['spy_sharpe']:6.2f} {x['max_dd']:6.1%} {x['spy_max_dd']:6.1%}")
        bp = r["best_position"]
        print(f"     trades {r['trades']}, sells {r['sells']}, win {r['win_rate'] or 0:.0%}, final ${r['final']:,.0f}, "
              f"best position {bp['ticker']} ${bp['gain']:,.0f} = {r['best_position_share'] or 0:.0%} of profit")
    print()
    for w, x in verdict.items():
        print(f"{w:>8}: median CAGR {x['median_cagr']:.1%} vs SPY TR {x['spy_cagr']:.1%} -> "
              f"{'PASS' if x['pass'] else 'FAIL'} (median excess {x['median_excess']:+.1%}, "
              f"{x['vintages_beating_spy']}/{len(rows)} vintages beat SPY)")
    print(f"\nH7 {'VERDICT' if a.variant == 'none' else 'diagnostic outcome'}: {'PASS' if final else 'FAIL'}")
    json.dump({"vintages": rows, "verdict": verdict, "pass": final},
              open(src / "h7_score.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

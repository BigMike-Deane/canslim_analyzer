# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""Big-winner study Part 3 (docs/big-winner-plan.md, pre-registered 2026-10-08 9:29 AM CT):
momentum entries + the app's exits, vs hold-126 (M2) and random entries (R, 20 seeds).

Standalone daily-close simulation on the stitched PIT prices (delisted names included).
Decisions on day t use data through close t-1 and execute at close t; exits use close t.

  python3 b3_momentum.py
Output: META_DIR/bigwin/b3_results.json + printed tables.
"""
import json
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from b2_score import YEARS, spy_tr, yearly_excess
from common import META_DIR

COST = 0.00095
STOP, SLOTS, COOLDOWN, MIN_DVOL = 7.0, 8, 10, 5e6
TRAIL = ((50, 25), (30, 18), (20, 12), (10, 6), (5, 4))          # (peak gain >=, trail % below peak)
PARTIALS = ((50, 0.75), (40, 0.50), (25, 0.25))                   # (gain >=, cumulative fraction sold)
VINTAGES, SEEDS = (0, 20, 40), 20
GATE = {"big_year_pp": 20.0, "min_big_years": 3, "worst_year_pp": -15.0, "random_pct": 90}


def build():
    p = pd.read_csv(META_DIR / "m3_panel.csv.gz", usecols=["cik", "date"], parse_dates=["date"])
    spy = m.prices("SPY")
    days = spy.loc["2015-06-01":"2026-10-05"].index
    ciks = sorted(p.cik.unique())
    close = np.full((len(days), len(ciks)), np.nan)
    dvol = np.full_like(close, np.nan)
    sym = np.zeros(close.shape, dtype=np.int32)
    codes: dict = {}
    for j, cik in enumerate(ciks):
        px = m.cik_prices(cik)
        if px.empty:
            continue
        px = px[~px.index.duplicated(keep="last")].reindex(days)
        close[:, j] = px.Close.to_numpy(float)
        dvol[:, j] = (px.Close * px.Volume).rolling(20, min_periods=15).mean().to_numpy(float)
        sym[:, j] = [codes.setdefault(s, len(codes) + 1) if isinstance(s, str) else 0 for s in px.symbol]
    members = {d: set(g.cik) for d, g in p.groupby("date")}
    col = {c: j for j, c in enumerate(ciks)}
    return days, ciks, col, close, dvol, sym, members, spy.Close.reindex(days).to_numpy(float)


def candidate_lists(days, col, close, dvol, members, rng=None):
    """{day index of panel date: [column indices, best first]} -- momentum, or random if rng."""
    out = {}
    pos = {d: i for i, d in enumerate(days)}
    for d, cs in members.items():
        i = pos.get(d)
        if i is None or i < 126:
            continue
        js = np.array([col[c] for c in cs if c in col])
        ok = js[(dvol[i, js] >= MIN_DVOL) & np.isfinite(close[i, js])]
        if rng is not None:
            out[i] = list(rng.permutation(ok))
            continue
        mom = close[i - 21, ok] / close[i - 126, ok] - 1
        good = np.isfinite(mom)
        ok, mom = ok[good], mom[good]
        top = ok[mom >= np.quantile(mom, 0.90)] if len(mom) else ok
        out[i] = list(top[np.argsort(-(close[i - 21, top] / close[i - 126, top]))])
    return out


def simulate(days, close, sym, spy, cands, start, mode="app", slots=SLOTS, stop=STOP, idle_spy=False):
    ma50 = pd.Series(spy).rolling(50).mean().to_numpy()
    spy_ret = np.r_[0, spy[1:] / spy[:-1] - 1]
    panel_days = sorted(cands)
    cash, pos, recent, eq, trades = 25000.0, {}, {}, [], []
    t0 = int(np.searchsorted(days, pd.Timestamp(start)))
    last = None

    def value(t):
        v = cash
        for j, p in pos.items():
            v += p["sh"] * p["last"]
        return v

    def sell(t, j, frac_of_pos, price, why):
        nonlocal cash
        p = pos[j]
        sh = p["sh"] * frac_of_pos
        cash += sh * price * (1 - COST)
        p["realized"] += sh * (price * (1 - COST) - p["cost"])
        p["sh"] -= sh
        if p["sh"] <= 1e-9:
            trades.append({"entry": str(days[p["t"]].date()), "exit": str(days[t].date()), "why": why,
                           "pct": (p["realized"] + 0.0) / (p["orig"] * p["cost"]) * 100, "gain": p["realized"]})
            del pos[j]
            recent[j] = t

    for t in range(t0, len(days)):
        if idle_spy:
            cash *= 1 + spy_ret[t]
        # ---- exits on close t
        for j in list(pos):
            p, c = pos[j], close[t, j]
            if not np.isfinite(c) or sym[t, j] != p["sym"]:
                p["stale"] += 1
                if sym[t, j] not in (0, p["sym"]) or p["stale"] >= 10:      # ticker change / delisted
                    sell(t, j, 1.0, p["last"], "delist/ticker change")
                continue
            p["stale"], p["last"] = 0, c
            p["peak"] = max(p["peak"], c)
            g = (c / p["cost"] - 1) * 100
            if mode == "hold":
                if t - p["t"] >= 126:
                    sell(t, j, 1.0, c, "hold 126")
                continue
            if g <= -stop:
                sell(t, j, 1.0, c, "stop")
                continue
            pg = (p["peak"] / p["cost"] - 1) * 100
            trail = next((tr for lvl, tr in TRAIL if pg >= lvl), None)
            if trail is not None and c <= p["peak"] * (1 - trail / 100):
                sell(t, j, 1.0, c, "trailing")
                continue
            tgt = next((f for lvl, f in PARTIALS if g >= lvl), 0.0)
            if tgt > p["sold"] + 1e-9:
                frac = (tgt - p["sold"]) * p["orig"] / p["sh"]
                p["sold"] = tgt
                sell(t, j, min(frac, 1.0), c, "partial")
        # ---- buys at close t, decided on data through t-1
        k = np.searchsorted(panel_days, t - 1, side="right") - 1
        if k >= 0:
            last = cands[panel_days[k]]
        gate = t >= 1 and np.isfinite(ma50[t - 1]) and spy[t - 1] > ma50[t - 1]
        if gate and last is not None and len(pos) < slots:
            eqv = value(t)
            for j in last:
                if len(pos) >= slots or cash < 100:
                    break
                if j in pos or (j in recent and t - recent[j] <= COOLDOWN):
                    continue
                c = close[t, j]
                if not np.isfinite(c) or c <= 0 or sym[t, j] == 0:
                    continue
                amt = min(eqv / slots, cash)
                sh = amt / (c * (1 + COST))
                cash -= amt
                pos[j] = {"sh": sh, "orig": sh, "cost": c * (1 + COST), "peak": c, "last": c, "t": t, "sold": 0.0,
                          "sym": sym[t, j], "stale": 0, "realized": 0.0}
        eq.append((days[t], value(t)))
    s = pd.Series(dict(eq))
    return s, trades


def summarize(eq, trades, spy_tr_s):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    sp = spy_tr_s.reindex(eq.index, method="ffill")
    tr = pd.DataFrame(trades)
    big = tr[tr.pct >= 50] if len(tr) else tr
    return dict(cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, spy_cagr=(sp.iloc[-1] / sp.iloc[0]) ** (1 / yrs) - 1,
                dd=float((1 - eq / eq.cummax()).max()), years=yearly_excess(eq, spy_tr_s), n=len(tr),
                n50=int((tr.pct >= 50).sum()) if len(tr) else 0, n100=int((tr.pct >= 100).sum()) if len(tr) else 0,
                big_share=float(big.gain.sum() / tr.gain[tr.gain > 0].sum()) if len(tr) and (tr.gain > 0).any() else 0.0,
                win_rate=float((tr.gain > 0).mean()) if len(tr) else 0.0)


def gate(per_vintage):
    yr = pd.DataFrame({o: r["years"] for o, r in per_vintage.items()}).median(axis=1)
    full = yr.loc[[y for y in YEARS if y in yr.index]]
    a, b, c = full.mean() > 0, int((full >= GATE["big_year_pp"]).sum()) >= GATE["min_big_years"], \
        full.min() >= GATE["worst_year_pp"]
    return yr, full, bool(a), bool(b), bool(c)


def main():
    t0 = time.time()
    days, ciks, col, close, dvol, sym, members, spy = build()
    spy_tr_s = spy_tr()
    print(f"matrix {close.shape}, {len(members)} panel dates; build {time.time() - t0:.0f}s", flush=True)
    mom = candidate_lists(days, col, close, dvol, members)
    sess = [d for d in sorted(members) if d >= pd.Timestamp("2016-01-04")]
    starts = {o: m.prices("SPY").loc["2016-01-04":].index[o] for o in VINTAGES}
    res = {}
    for name, kw in (("M1", {}), ("M2", {"mode": "hold"})):
        res[name] = {o: summarize(*simulate(days, close, sym, spy, mom, starts[o], **kw), spy_tr_s) for o in VINTAGES}
    rand = []
    for seed in range(SEEDS):
        rc = candidate_lists(days, col, close, dvol, members, rng=np.random.default_rng(seed))
        rand.append(summarize(*simulate(days, close, sym, spy, rc, starts[0]), spy_tr_s))
    sens = {k: summarize(*simulate(days, close, sym, spy, mom, starts[0], **kw), spy_tr_s)
            for k, kw in (("slots20", {"slots": 20}), ("stop10", {"stop": 10.0}), ("idle_spy", {"idle_spy": True}))}
    del sess
    rc_cagr = np.array([r["cagr"] for r in rand])
    p90 = float(np.percentile(rc_cagr, GATE["random_pct"]))
    out = {}
    print("\nMedian yearly excess vs SPY TR (pp):")
    for name in ("M1", "M2"):
        yr, full, a, b, c = gate(res[name])
        d = res[name][0]["cagr"] > p90
        verdict = "PASS" if (a and b and c and d) else "FAIL"
        out[name] = dict(years=yr.round(1).to_dict(), mean=round(full.mean(), 2), big_years=int((full >= 20).sum()),
                         worst=round(full.min(), 1), a=a, b=b, c=c, d=bool(d),
                         verdict=verdict if name == "M1" else "reported",
                         vintages={o: {k: v for k, v in r.items() if k != "years"} for o, r in res[name].items()})
        print(f"  {name}: " + " ".join(f"{y}:{v:+.1f}" for y, v in yr.items())
              + f" | mean {full.mean():+.1f}, big years {int((full >= 20).sum())}, worst {full.min():+.1f}"
              + (f" | a {a} b {b} c {c} d {bool(d)} -> {verdict}" if name == "M1" else ""))
        for o, r in res[name].items():
            print(f"     v{o}: CAGR {r['cagr']:.2%} vs SPY {r['spy_cagr']:.2%} | DD {r['dd']:.1%} | trades {r['n']} "
                  f"| >=+50% {r['n50']} | >=+100% {r['n100']} | big-winner profit share {r['big_share']:.0%} "
                  f"| win rate {r['win_rate']:.0%}")
    print(f"\n  Random entries + app exits (20 seeds, offset 0): CAGR median {np.median(rc_cagr):.2%}, "
          f"90th pct {p90:.2%}, max {rc_cagr.max():.2%}; M1 offset 0 = {res['M1'][0]['cagr']:.2%}")
    print("  Sensitivities (offset 0, reporting only): " + " | ".join(
        f"{k}: CAGR {v['cagr']:.2%}, DD {v['dd']:.1%}" for k, v in sens.items()))
    out["random"] = dict(cagr=[round(x, 4) for x in rc_cagr], p90=p90)
    out["sensitivities"] = {k: {kk: vv for kk, vv in v.items() if kk != "years"} for k, v in sens.items()}
    json.dump(out, open(META_DIR / "bigwin" / "b3_results.json", "w"), indent=1, default=str)
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

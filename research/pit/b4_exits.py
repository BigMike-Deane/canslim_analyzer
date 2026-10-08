# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""Big-winner study Part 4 (docs/big-winner-plan.md, pre-registered 2026-10-08 10:15 AM CT):
exit rules for momentum entries. Companies split by sha256(CIK) mod 2 BEFORE any run;
select the best of 8 variants on half A, confirm the single choice once on half B.

  python3 b4_exits.py
Output: META_DIR/bigwin/b4_results.json + printed tables.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from b2_score import YEARS, spy_tr, yearly_excess
from b3_momentum import COOLDOWN, COST, MIN_DVOL, PARTIALS, SLOTS, STOP, TRAIL, build
from common import META_DIR

VINTAGES, SEEDS = (0, 20, 40), 20
SELECTABLE = [(x, idle) for x in ("X1", "X2", "X3", "X4") for idle in (False, True)]


def half(cik) -> int:
    return int(hashlib.sha256(str(int(cik)).encode()).hexdigest(), 16) % 2


def rank_lists(days, col, close, dvol, members, rng=None):
    """Per panel day: (entry candidates best-first, set of top-30% 'leaders')."""
    pos = {d: i for i, d in enumerate(days)}
    out = {}
    for d, cs in members.items():
        i = pos.get(d)
        if i is None or i < 126:
            continue
        js = np.array([col[c] for c in cs if c in col])
        ok = js[(dvol[i, js] >= MIN_DVOL) & np.isfinite(close[i, js])]
        mom = close[i - 21, ok] / close[i - 126, ok] - 1
        good = np.isfinite(mom)
        ok, mom = ok[good], mom[good]
        if not len(ok):
            continue
        leaders = set(ok[mom >= np.quantile(mom, 0.70)])
        if rng is not None:
            cand = list(rng.permutation(ok))
        else:
            top = mom >= np.quantile(mom, 0.90)
            cand = list(ok[top][np.argsort(-mom[top])])
        out[i] = (cand, leaders)
    return out


def simulate(days, close, sym, spy_px, spy_trr, lists, start, exit_rule, idle_spy):
    ma50 = pd.Series(spy_px).rolling(50).mean().to_numpy()
    panel_days = sorted(lists)
    cash, pos, recent, eq, trades = 25000.0, {}, {}, [], []
    t0 = int(np.searchsorted(days, pd.Timestamp(start)))

    def value():
        return cash + sum(p["sh"] * p["last"] for p in pos.values())

    def sell(t, j, frac, price, why):
        nonlocal cash
        p = pos[j]
        sh = p["sh"] * frac
        cash += sh * price * (1 - COST)
        p["realized"] += sh * (price * (1 - COST) - p["cost"])
        p["sh"] -= sh
        if p["sh"] <= 1e-9:
            trades.append({"why": why, "pct": p["realized"] / (p["orig"] * p["cost"]) * 100, "gain": p["realized"]})
            del pos[j]
            recent[j] = t

    for t in range(t0, len(days)):
        if idle_spy:
            cash *= 1 + spy_trr[t]
        k = np.searchsorted(panel_days, t - 1, side="right") - 1
        pday = panel_days[k] if k >= 0 else None
        for j in list(pos):
            p, c = pos[j], close[t, j]
            if not np.isfinite(c) or sym[t, j] != p["sym"]:
                p["stale"] += 1
                if sym[t, j] not in (0, p["sym"]) or p["stale"] >= 10:
                    sell(t, j, 1.0, p["last"], "delist/ticker change")
                continue
            p["stale"], p["last"] = 0, c
            p["peak"] = max(p["peak"], c)
            g = (c / p["cost"] - 1) * 100
            held = t - p["t"]
            if exit_rule == "X0":
                if g <= -STOP:
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
            elif exit_rule in ("X1", "X2"):
                if exit_rule == "X2" and g <= -25:
                    sell(t, j, 1.0, c, "disaster stop")
                elif held >= 126:
                    sell(t, j, 1.0, c, "hold 126")
            elif exit_rule == "X3":
                if c <= p["peak"] * 0.75:
                    sell(t, j, 1.0, c, "trail 25")
            elif exit_rule == "X4":
                if pday is not None and pday != p["pday"] and pday > p["t"] - 1:
                    p["pday"] = pday
                    if j not in lists[pday][1]:
                        sell(t, j, 1.0, c, "left top 30%")
        gate = t >= 1 and np.isfinite(ma50[t - 1]) and spy_px[t - 1] > ma50[t - 1]
        if gate and pday is not None and len(pos) < SLOTS:
            eqv = value()
            for j in lists[pday][0]:
                if len(pos) >= SLOTS or cash < 100:
                    break
                if j in pos or (j in recent and t - recent[j] <= COOLDOWN):
                    continue
                c = close[t, j]
                if not np.isfinite(c) or c <= 0 or sym[t, j] == 0:
                    continue
                amt = min(eqv / SLOTS, cash)
                sh = amt / (c * (1 + COST))
                cash -= amt
                pos[j] = {"sh": sh, "orig": sh, "cost": c * (1 + COST), "peak": c, "last": c, "t": t, "sold": 0.0,
                          "sym": sym[t, j], "stale": 0, "realized": 0.0, "pday": pday}
        eq.append((days[t], value()))
    return pd.Series(dict(eq)), trades


def summarize(eq, trades, spy_tr_s):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    sp = spy_tr_s.reindex(eq.index, method="ffill")
    tr = pd.DataFrame(trades)
    pos_gain = tr.gain[tr.gain > 0].sum() if len(tr) else 0
    return dict(cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, spy_cagr=(sp.iloc[-1] / sp.iloc[0]) ** (1 / yrs) - 1,
                dd=float((1 - eq / eq.cummax()).max()), years=yearly_excess(eq, spy_tr_s), n=len(tr),
                n100=int((tr.pct >= 100).sum()) if len(tr) else 0,
                big_share=float(tr.gain[tr.pct >= 50].sum() / pos_gain) if pos_gain else 0.0)


def score(runs):
    yr = pd.DataFrame({o: r["years"] for o, r in runs.items()}).median(axis=1)
    full = yr.loc[[y for y in YEARS if y in yr.index]]
    med_cagr = float(np.median([r["cagr"] for r in runs.values()]))
    med_spy = float(np.median([r["spy_cagr"] for r in runs.values()]))
    return yr, full, med_cagr, med_spy


def main():
    t0 = time.time()
    days, ciks, col, close, dvol, sym, members, spy_px = build()
    spy_tr_s = spy_tr()
    spy_trr = spy_tr_s.reindex(days, method="ffill").pct_change().fillna(0).to_numpy()
    starts = {o: m.prices("SPY").loc["2016-01-04":].index[o] for o in VINTAGES}
    halves = {h: {d: {c for c in cs if half(c) == h} for d, cs in members.items()} for h in (0, 1)}
    print(f"companies: half A {sum(half(c) == 0 for c in ciks):,}, half B {sum(half(c) == 1 for c in ciks):,}; "
          f"build {time.time() - t0:.0f}s", flush=True)

    def run(h, rule, idle, lists=None):
        lists = lists or rank_lists(days, col, close, dvol, halves[h])
        return {o: summarize(*simulate(days, close, sym, spy_px, spy_trr, lists, starts[o], rule, idle), spy_tr_s)
                for o in VINTAGES}

    lists_a = rank_lists(days, col, close, dvol, halves[0])
    out = {"A": {}}
    print("\nHALF A (selection):")
    for rule, idle in [("X0", False)] + SELECTABLE:
        runs = run(0, rule, idle, lists_a)
        yr, full, mc, ms = score(runs)
        name = f"{rule}{'+SPY' if idle else ''}"
        out["A"][name] = dict(cagr=mc, spy=ms, mean=float(full.mean()), big=int((full >= 20).sum()),
                              worst=float(full.min()), dd=float(np.median([r["dd"] for r in runs.values()])),
                              n100=int(np.median([r["n100"] for r in runs.values()])),
                              big_share=float(np.median([r["big_share"] for r in runs.values()])))
        v = out["A"][name]
        print(f"  {name:<8} CAGR {mc:.2%} (SPY {ms:.2%}) | yearly excess mean {v['mean']:+.1f}, years>=+20 {v['big']}, "
              f"worst {v['worst']:+.1f} | DD {v['dd']:.0%} | doublers {v['n100']} | big-winner profit {v['big_share']:.0%}")
    choice = max((n for n in out["A"] if n != "X0"), key=lambda n: out["A"][n]["cagr"])
    rule, idle = choice.replace("+SPY", ""), choice.endswith("+SPY")
    print(f"\n=> chosen on half A: {choice}")

    runs_b = run(1, rule, idle)
    yr, full, mc, ms = score(runs_b)
    lists_b_rand = [rank_lists(days, col, close, dvol, halves[1], rng=np.random.default_rng(s)) for s in range(SEEDS)]
    rand = np.array([summarize(*simulate(days, close, sym, spy_px, spy_trr, lb, starts[0], rule, idle), spy_tr_s)["cagr"]
                     for lb in lists_b_rand])
    p90 = float(np.percentile(rand, 90))
    a, b, c = full.mean() > 0, int((full >= 20).sum()) >= 3, full.min() >= -15
    d, e = runs_b[0]["cagr"] > p90, mc > ms
    verdict = "PASS" if all((a, b, c, d, e)) else ("PASS EXCEPT DRAWDOWN TOLERANCE" if all((a, b, d, e)) else "FAIL")
    print(f"\nHALF B (confirmation, {choice}):")
    print("  yearly excess (median vintage): " + " ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))
    for o, r in runs_b.items():
        print(f"  v{o}: CAGR {r['cagr']:.2%} vs SPY {r['spy_cagr']:.2%} | DD {r['dd']:.1%} | trades {r['n']} | "
              f"doublers {r['n100']} | big-winner profit {r['big_share']:.0%}")
    print(f"  random entries + same exit (20 seeds, offset 0): median {np.median(rand):.2%}, 90th pct {p90:.2%}")
    print(f"  (a) mean {full.mean():+.1f} {a} | (b) years>=+20 {int((full >= 20).sum())} {b} | (c) worst {full.min():+.1f} {c} "
          f"| (d) {runs_b[0]['cagr']:.2%} > {p90:.2%} {d} | (e) {mc:.2%} > SPY {ms:.2%} {e}")
    print(f"  => {verdict}")
    out["choice"] = choice
    out["B"] = dict(years=yr.round(1).to_dict(), cagr=mc, spy=ms, a=bool(a), b=bool(b), c=bool(c), d=bool(d), e=bool(e),
                    verdict=verdict, random=[round(x, 4) for x in rand], p90=p90,
                    vintages={o: {k: v for k, v in r.items() if k != "years"} for o, r in runs_b.items()})
    json.dump(out, open(META_DIR / "bigwin" / "b4_results.json", "w"), indent=1, default=str)
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

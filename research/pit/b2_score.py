# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Big-winner study Part 2 scoring (docs/big-winner-plan.md): calendar-year excess vs SPY
total return per exit variant (median across vintages 0/20/40), the owner's-terms gate,
and "does capping help?" (full-period CAGR, all 3 vintages same sign).

  python3 b2_score.py
"""
import gzip
import json

import numpy as np
import pandas as pd

from common import DATA_DIR, META_DIR

VARIANTS = ["E0", "E1", "E2", "E3", "E4", "E5"]
VINTAGES = [0, 20, 40]
YEARS = list(range(2016, 2026))
GATE = {"min_big_years": 3, "big_year_pp": 20.0, "worst_year_pp": -15.0}
COMPARE = {"E1": "E0", "E2": "E0", "E3": "E0", "E4": "E5"}


def spy_tr():
    rows = json.load(gzip.open(DATA_DIR / "timing" / "SPY.json.gz"))["adj"]
    return pd.Series({pd.Timestamp(r["date"]): r["adjClose"] for r in rows}).sort_index()


def load(e, o):
    p = META_DIR / "bigwin" / e / f"vintage_{o}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return pd.Series({pd.Timestamp(a): b for a, b in d["equity"]}).sort_index(), d["trades"]


def yearly_excess(eq, spy):
    sp = spy.reindex(eq.index, method="ffill")
    out = {}
    for y in sorted(set(eq.index.year)):
        a, b = eq[eq.index.year == y], sp[sp.index.year == y]
        # each year runs from the prior year's last close (first year: from the start)
        prev = eq[eq.index.year < y]
        a0, b0 = (prev.iloc[-1], sp[sp.index.year < y].iloc[-1]) if len(prev) else (a.iloc[0], b.iloc[0])
        out[y] = ((a.iloc[-1] / a0) - (b.iloc[-1] / b0)) * 100
    return out


def cagr(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1


def main():
    spy = spy_tr()
    res, cg = {}, {}
    for e in VARIANTS:
        per = {}
        for o in VINTAGES:
            r = load(e, o)
            if r is None:
                continue
            eq, trades = r
            per[o] = yearly_excess(eq, spy)
            sp = spy.reindex(eq.index, method="ffill")
            big = [t for t in trades if t["action"] == "SELL" and t["gain"] is not None]
            cg[(e, o)] = dict(cagr=cagr(eq), spy=cagr(sp), dd=float((1 - eq / eq.cummax()).max()),
                              n_trades=len(trades), n_caps=sum(1 for t in trades if (t["reason"] or "").startswith("CAP")),
                              realized=sum(t["gain"] for t in big))
        if len(per) < len(VINTAGES):
            print(f"{e}: {len(per)}/{len(VINTAGES)} vintages done -- not scored")
            continue
        yr = pd.DataFrame(per)
        med = yr.median(axis=1)
        full = med.loc[[y for y in YEARS if y in med.index]]
        a = full.mean() > 0
        b = int((full >= GATE["big_year_pp"]).sum()) >= GATE["min_big_years"]
        c = full.min() >= GATE["worst_year_pp"]
        res[e] = dict(years=med.round(1).to_dict(), mean=round(full.mean(), 2), big_years=int((full >= GATE["big_year_pp"]).sum()),
                      worst=round(full.min(), 1), best=round(full.max(), 1), a=bool(a), b=bool(b), c=bool(c),
                      verdict="PASS" if a and b and c else "FAIL",
                      cagr={o: round(cg[(e, o)]["cagr"] * 100, 2) for o in VINTAGES},
                      spy_cagr=round(np.median([cg[(e, o)]["spy"] for o in VINTAGES]) * 100, 2),
                      max_dd={o: round(cg[(e, o)]["dd"] * 100, 1) for o in VINTAGES},
                      caps={o: cg[(e, o)]["n_caps"] for o in VINTAGES})
    print("Median yearly excess vs SPY TR (pp), vintages 0/20/40:")
    yrs = sorted({y for r in res.values() for y in r["years"]})
    print("        " + " ".join(f"{y:>6}" for y in yrs) + "   mean  big  worst  verdict")
    for e, r in res.items():
        print(f"  {e:<5} " + " ".join(f"{r['years'].get(y, np.nan):>+6.1f}" for y in yrs)
              + f"  {r['mean']:>+5.1f}  {r['big_years']:>3}  {r['worst']:>+5.1f}  {r['verdict']} (a {r['a']}, b {r['b']}, c {r['c']})")
    print("\nFull-period CAGR by vintage (SPY TR median {}%):".format(next(iter(res.values()))["spy_cagr"] if res else "-"))
    for e, r in res.items():
        print(f"  {e}: {r['cagr']} | max DD {r['max_dd']} | cap exits {r['caps']}")
    print("\nDoes capping help? (better in all 3 vintages = helps, worse in all 3 = hurts):")
    for e, base in COMPARE.items():
        if e in res and base in res:
            d = [res[e]["cagr"][o] - res[base]["cagr"][o] for o in VINTAGES]
            v = "helps" if all(x > 0 for x in d) else "hurts" if all(x < 0 for x in d) else "mixed"
            print(f"  {e} vs {base}: CAGR diff {[round(x, 2) for x in d]} pp -> {v}")
            res[e]["vs_" + base] = v
    json.dump(res, open(META_DIR / "bigwin" / "b2_results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

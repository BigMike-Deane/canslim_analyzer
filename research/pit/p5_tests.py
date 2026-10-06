# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
"""H5 big-winner odds + H6 setup signal (pre-registered in
docs/phase2-pit-backtest-plan.md, "Test it the way it trades").
Reads m3_panel.csv.gz (v2) + p5_setups.csv.gz.

H5 per group: per-date big-winner rate (r120 >= +50%) minus universe rate;
   NW t (12 lags) >= 3, > 0 in both halves, net tail edge
   ((big - blowup) group minus universe; blowup = r120 <= -30%) >= 0 both halves.
H6 per group: per-date mean 60d excess; NW t (6 lags) >= 3, > 0 both halves,
   AND group EW 60d - SPY - 19 bps x turnover > 0 both halves.
"""
import pandas as pd

from common import META_DIR
from m3_signal_tests import load
from p4_tests import RT_COST, line, nw_t, turnover, win

TRAIN = ("2016-01-01", "2021-12-31")
TEST = ("2022-01-01", "2026-12-31")


def halves(s):
    return win(s, TRAIN).mean(), win(s, TEST).mean()


def main():
    d = load().merge(pd.read_csv(META_DIR / "p5_setups.csv.gz", parse_dates=["date"]), on=["cik", "date"], how="inner")
    d["big"] = (d.r120 >= 0.5).astype(float).where(d.r120.notna())
    d["huge"] = (d.r120 >= 1.0).astype(float).where(d.r120.notna())
    d["blow"] = (d.r120 <= -0.3).astype(float).where(d.r120.notna())
    d["x120"] = d.r120 - d.groupby("date").r120.transform("mean")
    groups = {
        "G72 (score >= 72)": d.total >= 72,
        "GSET (pre-breakout setup)": d.entry_type == "pre-breakout",
        "GSET65 (pre-breakout & score >= 65)": (d.entry_type == "pre-breakout") & (d.total >= 65),
    }
    u = d.groupby("date")[["big", "huge", "blow"]].mean()
    print(f"rows {len(d):,}; universe per-date rates: big {u.big.mean():.2%}, +100% {u.huge.mean():.2%}, "
          f"blowup {u.blow.mean():.2%}; entry types {d.entry_type.value_counts(normalize=True).round(3).to_dict()}")

    verdict = {}
    print("\nH5 big-winner odds (r120 >= +50%)")
    for name, mask in groups.items():
        g = d[mask].groupby("date")[["big", "huge", "blow", "x120"]].mean()
        n = d[mask].groupby("date").size()
        diff = (g.big - u.big).dropna()
        net = ((g.big - g.blow) - (u.big - u.blow)).dropna()
        print(f"\n{name}  (median {n.median():.0f} stocks/date)")
        m, t, tr, te = line("big-winner rate - universe", diff, nw=False)
        t = nw_t(diff, 12)
        print(f"      NW t (12 lags) {t:+.2f}")
        ntr, nte = halves(net)
        print(f"  net tail edge (big-blowup vs universe): 2016-21 {ntr:+.2%} | 2022-26 {nte:+.2%}")
        ok = m > 0 and t >= 3 and tr > 0 and te > 0 and ntr >= 0 and nte >= 0
        print("  reporting only:")
        print(f"      +100% rate - universe: {(g.huge - u.huge).mean():+.2%}; blowup rate - universe: "
              f"{(g.blow - u.blow).mean():+.2%}; mean r120 excess: {g.x120.mean():+.2%}")
        verdict[f"H5 {name}"] = "PASS" if ok else "FAIL"
        print(f"  => {verdict[f'H5 {name}']}")

    print("\nH6 setup signal (60-session excess)")
    groups6 = {**{k: v for k, v in groups.items() if k != "G72 (score >= 72)"},
               "BRK (breaking out)": d.breaking_out.astype(bool)}
    for name, mask in groups6.items():
        sub = d[mask & d.x60.notna()]
        ex = sub.groupby("date").x60.mean()
        m, t, tr, te = line("group 60d excess (NW t)", ex)
        net_ret = sub.groupby("date").r60.mean() - sub.groupby("date").spy60.first()
        net = (net_ret - turnover(sub.groupby("date").cik.apply(set)) * RT_COST).dropna()
        _, _, ntr, nte = line("group EW - SPY, net", net)
        ok = m > 0 and t >= 3 and tr > 0 and te > 0 and ntr > 0 and nte > 0
        verdict[f"H6 {name}"] = "PASS" if ok else "FAIL"
        print(f"  (median {sub.groupby('date').size().median():.0f} stocks/date) => {verdict[f'H6 {name}']}")

    print("\nSummary:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

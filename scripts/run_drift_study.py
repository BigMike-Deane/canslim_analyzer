"""Predictive-drift study — can anything FORECAST chop-bleed days?

Follow-up to scripts/run_chop_lab.py (2026-08-27): every chop lever is
concurrent-regime insurance that pays trend-day premiums. This asks the
sharper question: does any end-of-day-t signal predict the book's day-t+1
excess vs SPY, better than the concurrent chop-band flag the arms use?

Series: user-1 daily portfolio snapshots vs SPY from market_snapshots —
the SAME pairing the edge meter uses (instrument parity, no replay).

Pre-declared predictors (declared before looking, to bound snooping):
  gap      SPY % above its 50MA (level, end of day t)
  dgap5    5-day change in gap (trajectory: sagging vs climbing)
  vol10    SPY 10d realized vol (std of daily returns, bps)
  spyret5  SPY 5d return, %
  qgap     QQQ % above its 50MA
  mscore   market_score composite (the M-score the scanner stamps)
  prevx    day-t book excess itself (does drift cluster?)
  band     concurrent chop-band flag 0<=gap<=1.5 (the arms' trigger --
           BASELINE to beat); meter = gap<=1.5 variant also shown

Stats per predictor: Spearman rho vs next-day excess; p from a
circular-shift permutation null (5,000 shifts, min offset 10 — preserves
both series' autocorrelation, unlike naive shuffling); tercile means;
split-half sign stability. EXPLORATORY: 8 predictors, n~95 — a single
p<0.05 is weak evidence; demand sign-stable halves + a mechanism before
proposing any arm.

Run:  DATABASE_URL=postgresql://canslim:<pw>@<host>:5432/canslim \\
          python3 scripts/run_drift_study.py
"""
import os
import random
import sys

import pandas as pd
from sqlalchemy import create_engine, text

N_SHIFT = 5000
MIN_OFFSET = 10
BAND_PCT = 1.5


def load():
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("Set DATABASE_URL (read-only queries only).")
    eng = create_engine(url)
    with eng.connect() as cx:
        port = pd.read_sql(text(
            "SELECT DISTINCT ON (date) date, total_value "
            "FROM ai_portfolio_snapshots WHERE user_id=1 "
            "ORDER BY date, timestamp DESC"), cx)
        mkt = pd.read_sql(text(
            "SELECT DISTINCT ON (date) date, spy_price, spy_50_ma, "
            "qqq_price, qqq_50_ma, market_score "
            "FROM market_snapshots WHERE date >= '2026-03-01' "
            "ORDER BY date, timestamp DESC"), cx)
        first = pd.read_sql(text(
            "SELECT min(executed_at)::date AS d FROM ai_portfolio_trades "
            "WHERE user_id=1"), cx).d.iloc[0]
    port["date"] = pd.to_datetime(port["date"]).dt.date
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.date
    out = port.merge(mkt, on="date").sort_values("date").reset_index(drop=True)
    # pre-launch snapshots are cash-only: excess = -spy_ret exactly, which
    # would fake a risk-on correlation — drop everything before the first
    # trade settles (predictor warmups may still reach back via mkt only)
    return out[out.date > first].reset_index(drop=True)


def spearman(x, y):
    return x.rank().corr(y.rank())


def shift_pval(x, y, rho_obs, rng):
    """Circular-shift null: p = share of shifted |rho| >= |rho_obs|."""
    n, hits = len(x), 0
    xv = list(x)
    for _ in range(N_SHIFT):
        k = rng.randint(MIN_OFFSET, n - MIN_OFFSET)
        xs = pd.Series(xv[k:] + xv[:k])
        if abs(spearman(xs, y.reset_index(drop=True))) >= abs(rho_obs):
            hits += 1
    return hits / N_SHIFT


def main():
    df = load()
    df["spy_ret"] = df.spy_price.pct_change()
    df["port_ret"] = df.total_value.pct_change()
    df["excess"] = (df.port_ret - df.spy_ret) * 1e4          # bps
    df["gap"] = (df.spy_price / df.spy_50_ma - 1) * 100
    df["qgap"] = (df.qqq_price / df.qqq_50_ma - 1) * 100
    df["dgap5"] = df.gap - df.gap.shift(5)
    df["vol10"] = df.spy_ret.rolling(10).std() * 1e4          # bps
    df["spyret5"] = (df.spy_price / df.spy_price.shift(5) - 1) * 100
    df["mscore"] = df.market_score
    df["prevx"] = df.excess
    df["band"] = ((df.gap >= 0) & (df.gap <= BAND_PCT)).astype(float)
    df["meter"] = (df.gap <= BAND_PCT).astype(float)
    df["y"] = df.excess.shift(-1)                             # next-day target

    cols = ["gap", "dgap5", "vol10", "spyret5", "qgap", "mscore",
            "prevx", "band", "meter"]
    d = df.dropna(subset=cols + ["y"]).reset_index(drop=True)
    n = len(d)
    half = n // 2
    rng = random.Random(20260827)

    print("=" * 76)
    print(f"PREDICTIVE-DRIFT STUDY  {d.date.iloc[0]} → {d.date.iloc[-1]}   "
          f"n={n} day-pairs (target: next-day excess bps)")
    print(f"target base rate: mean {d.y.mean():+.1f} bps/d, "
          f"std {d.y.std():.0f}, drift days (y<-20bps): "
          f"{(d.y < -20).mean() * 100:.0f}%")
    print("-" * 76)
    print(f"{'predictor':9} {'rho':>6} {'p(shift)':>9} "
          f"{'T1':>7} {'T2':>7} {'T3':>7}   halves(rho)")
    rows = []
    for c in cols:
        rho = spearman(d[c], d.y)
        p = shift_pval(d[c], d.y, rho, rng)
        if c in ("band", "meter"):
            g0 = d.y[d[c] == 0].mean()
            g1 = d.y[d[c] == 1].mean()
            terc = (f"{'out:':>3}{g0:+6.1f} {'in:':>4}{g1:+6.1f} {'':7}")
        else:
            t = pd.qcut(d[c], 3, labels=False, duplicates="drop")
            means = [d.y[t == i].mean() for i in sorted(t.dropna().unique())]
            terc = " ".join(f"{m:+7.1f}" for m in means)
        h1 = spearman(d[c].iloc[:half], d.y.iloc[:half])
        h2 = spearman(d[c].iloc[half:], d.y.iloc[half:])
        stable = "STABLE" if (h1 * h2 > 0 and abs(h1) > .05
                              and abs(h2) > .05) else "flips"
        print(f"{c:9} {rho:+6.2f} {p:9.3f} {terc}   "
              f"{h1:+.2f}/{h2:+.2f} {stable}")
        rows.append((c, rho, p, stable))
    print("-" * 76)
    sig = [r for r in rows if r[2] < 0.05 and r[3] == "STABLE"]
    bonf = [r for r in rows if r[2] < 0.05 / len(cols)]
    print(f"pass screen (p<.05 AND sign-stable halves): "
          f"{[r[0] for r in sig] or 'NONE'}")
    print(f"survive Bonferroni p<{0.05 / len(cols):.4f}: "
          f"{[r[0] for r in bonf] or 'NONE'}")
    print("Exploratory screen — any hit needs a mechanism story and a "
          "pre-registered forward arm before it touches trading.")


if __name__ == "__main__":
    main()

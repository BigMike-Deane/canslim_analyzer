# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Score v8 CANSLIM letter variants for every (cik, date) row of the v3 table
(docs/score-v3-plan.md, "Score v8 pre-registration", committed 6b6efb2 before any number).

C eps_accel, sue, rev_growth, rev_accel | A eps_stable | S ud_vol50 | I inst_chg, breadth_chg
Everything as known on D (SEC rows filed <= D, 13F known_date <= D, prices <= D's close).

  python3 v8_features.py [--workers 10] [--limit N]
Output: META_DIR/v8_features.csv.gz
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from common import META_DIR, split_factor

REV = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
FEATS = ["eps_accel", "sue", "rev_growth", "rev_accel", "eps_stable", "ud_vol50", "inst_chg", "breadth_chg"]
PANEL = None
REVT = None


def setup():
    global PANEL, REVT
    PANEL = pd.read_csv(META_DIR / "v3_table.csv.gz", usecols=["cik", "date"], parse_dates=["date"])
    f = pd.read_csv(META_DIR / "sec_facts.csv.gz", low_memory=False,
                    usecols=["cik", "concept", "unit", "start", "end", "val", "filed"])
    f = f[f.concept.isin(REV) & (f.unit == "USD") & f.cik.isin(set(PANEL.cik))]
    for c in ("start", "end", "filed"):
        f[c] = pd.to_datetime(f[c], errors="coerce")
    f = f.dropna(subset=["start", "end", "filed", "val"])
    f = f[(f.end - f.start).dt.days.between(80, 100)]
    f["pri"] = f.concept.map({c: i for i, c in enumerate(REV)})
    REVT = {cik: g for cik, g in f.groupby("cik")}
    m._facts(); m.identity(); m._inst()
    from common import _segments
    _segments()


def _known_q(t, d):
    """Quarterly values known on D: one value per period end (latest filing), newest first."""
    t = t[t.filed <= d].sort_values("filed").drop_duplicates("end", keep="last")
    return t.sort_values("end", ascending=False)[["end", "val"]].reset_index(drop=True)


def _yago(q, end):
    """Value of the quarter ending ~1 year before `end` (350-380 days)."""
    h = q[(q.end <= end - pd.Timedelta(days=350)) & (q.end >= end - pd.Timedelta(days=380))]
    return h.val.iloc[0] if len(h) else np.nan


def _prev(q, end):
    h = q[(q.end <= end - pd.Timedelta(days=80)) & (q.end >= end - pd.Timedelta(days=100))]
    return h.iloc[0] if len(h) else None


def _g(e0, e4):
    if np.isnan(e0) or np.isnan(e4):
        return np.nan
    return float(np.clip((e0 - e4) / max(abs(e4), 0.05), -3, 3))


def eps_feats(cik, d):
    out = {"eps_accel": np.nan, "sue": np.nan, "eps_stable": np.nan}
    t = m._eps_table(cik)
    if t.empty:
        return out
    q = _known_q(t[t.kind == "Q"], d)
    if len(q) and q.end.iloc[0] >= d - pd.Timedelta(days=200):
        e0, end0 = q.val.iloc[0], q.end.iloc[0]
        p = _prev(q, end0)
        if p is not None:
            g0, g1 = _g(e0, _yago(q, end0)), _g(p.val, _yago(q, p.end))
            if not (np.isnan(g0) or np.isnan(g1)):
                out["eps_accel"] = g0 - g1
        ch = [r.val - _yago(q, r.end) for r in q.head(8).itertuples()]
        ch = [c for c in ch if not np.isnan(c)]
        cur = e0 - _yago(q, end0)
        if len(ch) >= 6 and not np.isnan(cur):
            sd = float(np.std(ch, ddof=1))
            if sd > 0:
                out["sue"] = float(np.clip(cur / sd, -10, 10))
    fy = _known_q(t[t.kind == "FY"], d)
    if len(fy) >= 4 and fy.end.iloc[0] >= d - pd.Timedelta(days=550):
        v = fy.val.to_numpy()
        out["eps_stable"] = float(sum(v[i] > v[i + 1] for i in range(3)))
    return out


def rev_feats(cik, d):
    out = {"rev_growth": np.nan, "rev_accel": np.nan}
    t = REVT.get(cik)
    if t is None:
        return out
    t = t[t.filed <= d]
    # per period end: the highest-priority concept; within it the latest filing
    t = t.sort_values(["end", "pri", "filed"], ascending=[True, True, False]).drop_duplicates("end", keep="first")
    q = t.sort_values("end", ascending=False)[["end", "val"]].reset_index(drop=True)
    if not len(q) or q.end.iloc[0] < d - pd.Timedelta(days=200):
        return out

    def rg(val, end):
        y = _yago(q, end)
        return float(np.clip(val / y - 1, -1, 5)) if y and y > 0 and not np.isnan(y) else np.nan
    g0 = rg(q.val.iloc[0], q.end.iloc[0])
    out["rev_growth"] = g0
    p = _prev(q, q.end.iloc[0])
    if p is not None and not np.isnan(g0):
        g1 = rg(p.val, p.end)
        if not np.isnan(g1):
            out["rev_accel"] = g0 - g1
    return out


def ud_vol(px, k):
    if k < 51:
        return np.nan
    c, v = px.Close.to_numpy()[k - 50:k + 1], px.Volume.to_numpy()[k - 50:k + 1]
    dc = np.diff(c)
    up, dn = np.nansum(v[1:][dc > 0]), np.nansum(v[1:][dc < 0])
    return float(np.log(up / dn)) if up > 0 and dn > 0 else np.nan


def inst_feats(cik, d, px, k):
    out = {"inst_chg": np.nan, "breadth_chg": np.nan}
    ident = m.identity()
    if cik not in ident.index or not isinstance(ident.loc[cik, "cusips"], str):
        return out
    by = m._inst()
    parts = [by[c] for c in ident.loc[cik, "cusips"].split() if c in by]
    if not parts:
        return out
    i = pd.concat(parts)
    i = i[i.known_date <= d]
    per = sorted(i.period.unique())
    if len(per) < 2:
        return out
    a, b = i[i.period == per[-1]], i[i.period == per[-2]]
    n0, n1 = a.n_filers.sum(), b.n_filers.sum()
    if n0 >= 5 and n1 >= 5:
        out["breadth_chg"] = float(np.clip(n0 / n1 - 1, -1, 3))
    sh = m._shares_table(cik)
    sh = sh[sh.filed <= d]
    shares = sh.val.iloc[-1] if not sh.empty and sh.val.iloc[-1] else None
    if shares is None and k >= 0:                       # same fallback as inst_pct_asof
        sym = px.symbol.iloc[k]
        close = px.Close.iloc[k] * split_factor(sym, d)
        fm = m.fmp_mcap_asof(sym, d)
        shares = fm / close if fm and close > 0 else None
    if shares:
        out["inst_chg"] = float(np.clip((a.inst_shares.sum() - b.inst_shares.sum()) / shares * 100, -50, 50))
    return out


def work(ciks):
    rows = []
    for cik in ciks:
        try:
            px = m.cik_prices(cik)
            for d in PANEL.date[PANEL.cik == cik]:
                k = px.index.searchsorted(d, side="right") - 1 if not px.empty else -1
                r = {"cik": cik, "date": d, **eps_feats(cik, d), **rev_feats(cik, d),
                     "ud_vol50": ud_vol(px, k) if k >= 0 else np.nan, **inst_feats(cik, d, px, k)}
                rows.append(r)
        except Exception as e:
            rows.append({"cik": cik, "date": None, "error": repr(e)[:200]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    setup()
    ciks = sorted(PANEL.cik.unique())
    if a.limit:
        ciks = ciks[:: max(1, len(ciks) // a.limit)][: a.limit]
    chunks = [ciks[i:i + 20] for i in range(0, len(ciks), 20)]
    out = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(work, chunks), 1):
            out += rows
            if i % 25 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    errs = df[df["error"].notna()] if "error" in df else df.iloc[:0]
    df = df[df.date.notna()].drop(columns=["error"], errors="ignore")
    name = "v8_features_sample.csv.gz" if a.limit else "v8_features.csv.gz"
    df.to_csv(META_DIR / name, index=False)
    print(f"wrote {len(df):,} rows; errors {len(errs)}; {time.time() - t0:.0f}s", flush=True)
    print("coverage: " + "  ".join(f"{f} {df[f].notna().mean():.0%}" for f in FEATS))
    print(df[FEATS].describe().T[["mean", "std", "min", "50%", "max"]].to_string(float_format=lambda v: f"{v:.3f}"))
    if len(errs):
        print(errs.error.value_counts().head(5).to_string())


if __name__ == "__main__":
    main()

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportReturnType=false
"""M2: score any company on any past date with the REAL live scorer.

Builds the StockData the scanner would have built on date D, using only data
available at D's close, and passes it to canslim_scorer.CANSLIMScorer. No
scoring logic is copied (the old backtester's inline copies drifted).

Inputs as of D (see docs/phase2-pit-backtest-plan.md):
  price_history  last 252 sessions <= D, Close + Volume only (as live)
  high_52w       max High over those sessions (FMP quote yearHigh is intraday)
  avg_volume_50d mean of last 50 non-zero volumes; current_volume = D's volume
  quarterly/annual EPS  SEC diluted (basic fallback), latest value filed <= D;
                 fiscal Q4 = FY - Q1..Q3, dated to the 10-K
  institutional  13F shares (period known <= D) / SEC shares outstanding (filed <= D)
  surprise/streak  FMP earnings reports dated <= D, same rule as the live fetcher
  sector         FMP profile (current; rarely changes)
  M              data_fetcher.calculate_index_m_score on SPY/QQQ/DIA <= D
  L              SPY closes <= D via a fetcher stub
Snapshot-only inputs with no history are left neutral (0):
  eps_estimate_revision_pct, earnings_growth_estimate.
"""
import gzip
import json
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from common import FMP_DIR, META_DIR, REPO, load_cik_prices, load_prices, split_factor

sys.path.insert(0, str(REPO))
from canslim_scorer import CANSLIMScorer  # noqa: E402
from data_fetcher import MARKET_INDEX_WEIGHTS, StockData, calculate_index_m_score  # noqa: E402

LOOKBACK = 252


# ---------- prices ----------
@lru_cache(maxsize=None)
def prices(symbol: str) -> pd.DataFrame:
    return load_prices(symbol)


@lru_cache(maxsize=None)
def cik_prices(cik: int) -> pd.DataFrame:
    return load_cik_prices(cik)


def market_score_asof(d: pd.Timestamp) -> float:
    total_w = m_sum = 0.0
    for ticker, w in MARKET_INDEX_WEIGHTS.items():
        c = prices(ticker).Close.loc[:d].dropna()
        if len(c) < 50:
            continue
        ma200 = c.iloc[-200:].mean() if len(c) >= 200 else c.mean()
        m_sum += calculate_index_m_score(c.iloc[-1], c.iloc[-50:].mean(), ma200) * w
        total_w += w
    return round(m_sum / total_w * 15.0, 1) if total_w else 7.5


class _AsOfFetcher:
    """Stands in for DataFetcher: the scorer's L reads get_sp500_history()."""

    def __init__(self, d):
        self.d = d

    def get_sp500_history(self):
        return prices("SPY").loc[: self.d, ["Close"]].iloc[-LOOKBACK:]

    def get_market_direction(self):  # fallback path; unused when M is preset
        raise RuntimeError("market score must be preset")


# ---------- SEC fundamentals ----------
@lru_cache(maxsize=1)
def _facts() -> pd.DataFrame:
    f = pd.read_csv(META_DIR / "sec_facts.csv.gz", low_memory=False,
                    usecols=["cik", "concept", "unit", "start", "end", "val", "filed", "form"])
    f = f[f.concept.isin(["EarningsPerShareDiluted", "EarningsPerShareBasic",
                          "EntityCommonStockSharesOutstanding",
                          "WeightedAverageNumberOfDilutedSharesOutstanding"])]
    for c in ("start", "end", "filed"):
        f[c] = pd.to_datetime(f[c], errors="coerce")
    f["days"] = (f.end - f.start).dt.days
    return f


@lru_cache(maxsize=None)
def _eps_table(cik: int) -> pd.DataFrame:
    """One row per (period_end, filed) with kind Q/FY; diluted, basic fallback.
    Derived Q4 rows are added per 10-K filing."""
    f = _facts()
    f = f[(f.cik == cik) & f.concept.isin(["EarningsPerShareDiluted", "EarningsPerShareBasic"])]
    if f.empty:
        return f
    # prefer diluted per (end, filed); basic only where diluted is missing
    f = f.assign(pref=(f.concept != "EarningsPerShareDiluted").astype(int))
    f = f.sort_values("pref").drop_duplicates(["start", "end", "filed"])
    q = f[f.days.between(80, 100)].assign(kind="Q")
    fy = f[f.days.between(350, 380)].assign(kind="FY")
    derived = []
    for r in fy.itertuples():
        # quarters inside this fiscal year, as known when the 10-K was filed
        inside = q[(q.end > r.start) & (q.end < r.end - pd.Timedelta(days=20)) & (q.filed <= r.filed)]
        inside = inside.sort_values("filed").drop_duplicates("end", keep="last")
        if len(inside) == 3:
            derived.append({"end": r.end, "filed": r.filed, "val": r.val - inside.val.sum(), "kind": "Q"})
    cols = ["end", "filed", "val", "kind"]
    out = pd.concat([q[cols], fy[cols], pd.DataFrame(derived, columns=cols)], ignore_index=True)
    return out.dropna(subset=["end", "filed", "val"])


def eps_asof(cik: int, d: pd.Timestamp, kind: str, n: int) -> list[float]:
    t = _eps_table(cik)
    if t.empty:
        return []
    t = t[(t.kind == kind) & (t.filed <= d)]
    latest = t.sort_values("filed").drop_duplicates("end", keep="last").sort_values("end", ascending=False)
    # drop stale periods: a quarter more than ~2 years before D isn't "recent"
    return latest.val.head(n).tolist()


@lru_cache(maxsize=None)
def _shares_table(cik: int) -> pd.DataFrame:
    """dei shares outstanding (cover page). Multi-class filers (META, DELL,
    Alphabet) often report it only per class, which companyfacts omits; ~500
    CIKs had none and silently left every panel. Fallback: quarterly/annual
    weighted-average diluted shares (all classes), dated by filing."""
    f = _facts()
    f = f[f.cik == cik]
    s = f[f.concept == "EntityCommonStockSharesOutstanding"]
    wa = f[(f.concept == "WeightedAverageNumberOfDilutedSharesOutstanding") & f.days.between(80, 380)]
    if not s.empty:
        # dei may start late (Alphabet: cover-page total only from 2024) -> fill earlier filings from wa
        wa = wa[wa.filed < s.filed.min()]
    s = pd.concat([s, wa])
    return s.sort_values("filed")[["filed", "end", "val"]]


# ---------- market cap (FMP daily series primary; 2026-10-07 repair) ----------
@lru_cache(maxsize=1)
def _fmp_mcap() -> dict:
    """symbol -> daily market cap (FMP historical-market-capitalization, m1_mcap.py).
    ~6% of companies had SEC-derived caps off by > 2x (per-class / ADR-ratio share counts)."""
    p = META_DIR / "fmp_mcap.csv.gz"
    if not p.exists():
        return {}
    m = pd.read_csv(p, parse_dates=["date"])
    return {sym: g.set_index("date").mcap.sort_index() for sym, g in m.groupby("symbol")}


def fmp_mcap_asof(symbol: str, d: pd.Timestamp):
    s = _fmp_mcap().get(symbol)
    if s is None:
        return None
    s = s[:d]
    if s.empty or s.index[-1] < d - pd.Timedelta(days=7):
        return None
    return float(s.iloc[-1])


def mcap_asof(cik: int, symbol: str, d: pd.Timestamp, close: float):
    """Market cap on D: FMP's series for the symbol traded on D, else actual close x SEC
    shares as known (None if neither)."""
    fm = fmp_mcap_asof(symbol, d)
    if fm is not None:
        return fm
    sh = _shares_table(cik)
    sh = sh[sh.filed <= d]
    return float(close * sh.val.iloc[-1]) if not sh.empty and sh.val.iloc[-1] else None


# ---------- 13F institutional ----------
@lru_cache(maxsize=1)
def _inst() -> dict:
    """cusip -> its 13F rows (period, known_date, inst_shares)."""
    i = pd.read_csv(META_DIR / "inst_13f.csv.gz", dtype={"cusip": str},
                    parse_dates=["period", "known_date"])
    return {c: g for c, g in i.groupby("cusip")}


@lru_cache(maxsize=1)
def identity() -> pd.DataFrame:
    return pd.read_csv(META_DIR / "identity.csv.gz", parse_dates=["valid_from", "valid_to"]).set_index("cik")


def inst_pct_asof(cik: int, d: pd.Timestamp) -> float:
    ident = identity()
    if cik not in ident.index or not isinstance(ident.loc[cik, "cusips"], str):
        return 0.0
    cusips = ident.loc[cik, "cusips"].split()
    by_cusip = _inst()
    parts = [by_cusip[c] for c in cusips if c in by_cusip]
    if not parts:
        return 0.0
    i = pd.concat(parts)
    i = i[i.known_date <= d]
    if i.empty:
        return 0.0
    latest = i[i.period == i.period.max()]
    sh = _shares_table(cik)
    sh = sh[sh.filed <= d]
    shares = sh.val.iloc[-1] if not sh.empty and sh.val.iloc[-1] else None
    if shares is None:  # no SEC count (multi-class before 2024 etc.): FMP cap / actual close
        px = cik_prices(cik)
        k = px.index.searchsorted(d, side="right") - 1 if not px.empty else -1
        if k >= 0:
            sym = px.symbol.iloc[k]
            close = px.Close.iloc[k] * split_factor(sym, d)
            fm = fmp_mcap_asof(sym, d)
            shares = fm / close if fm and close > 0 else None
    if not shares:
        return 0.0
    return float(latest.inst_shares.sum() / shares * 100)


# ---------- FMP earnings + profile ----------
def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def surprise_asof(symbol: str, d: pd.Timestamp) -> tuple[float, int]:
    """Mirrors data_fetcher's FMP earnings walk, restricted to reports <= D."""
    data = _json(FMP_DIR / "earnings" / f"{symbol}.json") or []
    past = sorted((x for x in data if x.get("date") and x["date"] <= d.strftime("%Y-%m-%d")
                   and x.get("epsActual") is not None), key=lambda x: x["date"], reverse=True)
    surprise, streak = None, 0
    for x in past:
        a, e = x["epsActual"], x.get("epsEstimated")
        if e is None:
            continue
        if surprise is None and e != 0:
            surprise = max(-200.0, min(200.0, (a - e) / abs(e) * 100))
        if a > e:
            streak += 1
        else:
            break
    return (surprise or 0.0), streak


def sector_of(symbol: str) -> str:
    p = _json(FMP_DIR / "profile" / f"{symbol}.json") or []
    return (p[0].get("sector") or "") if p else ""


# ---------- assemble ----------
def stock_data_asof(cik: int, d: pd.Timestamp) -> StockData | None:
    ident = identity()
    if cik not in ident.index:
        return None
    row = ident.loc[cik]
    if not (row.valid_from <= d <= row.valid_to):
        return None
    px = cik_prices(cik)
    px = px.loc[row.valid_from: d].iloc[-LOOKBACK:]
    if len(px) < 50 or px.index[-1] < d - pd.Timedelta(days=5):
        return None  # no fresh price on D: not tradeable/scannable that day
    sd = StockData(row.symbol)
    sd.price_history = px[["Close", "Volume"]]
    sd.current_price = float(px.Close.iloc[-1])
    sd.high_52w = float(px.High.max())
    sd.low_52w = float(px.Low.min())
    vols = [v for v in px.Volume.iloc[-50:] if v]
    sd.avg_volume_50d = float(np.mean(vols)) if vols else 0.0
    sd.current_volume = float(px.Volume.iloc[-1] or 0)
    sd.quarterly_earnings = eps_asof(cik, d, "Q", 8)
    sd.annual_earnings = eps_asof(cik, d, "FY", 5)
    sd.institutional_holders_pct = inst_pct_asof(cik, d)
    sd.earnings_surprise_pct, streak = surprise_asof(row.symbol, d)
    sd.eps_beat_streak = sd.earnings_beat_streak = streak
    sd.sector = sector_of(row.symbol)
    sd.is_valid = True
    return sd


def score_asof(cik: int, d, m_score: float | None = None):
    d = pd.Timestamp(d)
    sd = stock_data_asof(cik, d)
    if sd is None:
        return None
    scorer = CANSLIMScorer(_AsOfFetcher(d))
    scorer._market_score = market_score_asof(d) if m_score is None else m_score
    scorer._market_detail = "pit"
    return scorer.score_stock(sd)

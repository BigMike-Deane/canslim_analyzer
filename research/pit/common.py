"""Shared helpers for the point-in-time (PIT) research pipeline.

Research-only: nothing here is imported by the app, and research/ is not
copied into the Docker image. Raw API responses are cached as JSON under
DATA_DIR so every step is resumable and auditable.
See docs/phase2-pit-backtest-plan.md.
"""
import json
import os
import threading
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PIT_DATA_DIR", Path.home() / "canslim_pit_data"))
SEC_DIR, FMP_DIR, META_DIR = DATA_DIR / "sec", DATA_DIR / "fmp", DATA_DIR / "meta"
for _d in (SEC_DIR, FMP_DIR, META_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# SEC requires automated clients to declare a name + contact email. The owner
# approved theirs (2026-10-05); it lives in the gitignored .env, not in code.
SEC_HEADERS = {"User-Agent": dotenv_values(REPO / ".env").get(
    "SEC_USER_AGENT", "canslim-pit-research/0.1 research-pipeline")}
FMP_BASE = "https://financialmodelingprep.com/stable/"


class Throttle:
    """Minimum spacing between calls, shared across threads."""

    def __init__(self, per_minute: float):
        self.gap = 60.0 / per_minute
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                time.sleep(self.next_at - now)
            self.next_at = max(now, self.next_at) + self.gap


class ScannerAwareThrottle(Throttle):
    """FMP pacing that yields to the live scanner, which shares the plan's
    300/min budget and scans every 90 min around the clock (~50 min each,
    limiter target 250/min). Slow while it scans, faster in the gaps.
    Status is polled at most once a minute; unreachable -> assume scanning."""

    STATUS_URL = "http://100.104.189.36:8001/api/scanner/status"

    def __init__(self, busy_per_min: float, idle_per_min: float):
        super().__init__(busy_per_min)
        self.busy_gap, self.idle_gap = 60.0 / busy_per_min, 60.0 / idle_per_min
        self.checked_at = 0.0

    def wait(self):
        if time.monotonic() - self.checked_at > 60:
            self.checked_at = time.monotonic()
            try:
                # Same READ-ONLY token the canslim-api MCP uses (mcp/api_server.py).
                tok = Path("~/.config/canslim/readonly_token").expanduser().read_text().strip()
                scanning = requests.get(self.STATUS_URL, timeout=5,
                                        headers={"Authorization": f"Bearer {tok}"}
                                        ).json().get("is_scanning", True)
            except Exception:
                scanning = True
            self.gap = self.busy_gap if scanning else self.idle_gap
        super().wait()


_sec_throttle = Throttle(per_minute=8 * 60)  # SEC fair access: <= 10 req/s
if "PIT_FMP_PER_MIN" in os.environ:  # explicit fixed rate wins
    _fmp_throttle = Throttle(per_minute=float(os.environ["PIT_FMP_PER_MIN"]))
else:
    _fmp_throttle = ScannerAwareThrottle(busy_per_min=40, idle_per_min=150)


def _get_json(url, params, headers, throttle, retries=4):
    for attempt in range(retries):
        throttle.wait()
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * 2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {url} {params}")


def cached(path: Path, fetch):
    """Return cached JSON at path, else call fetch() and cache its result.
    None results are cached too (as JSON null) so misses aren't re-fetched."""
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
    return data


def sec_get(url):
    return _get_json(url, None, SEC_HEADERS, _sec_throttle)


def fmp_get(endpoint, **params):
    params["apikey"] = dotenv_values(REPO / ".env")["FMP_API_KEY"]
    return _get_json(FMP_BASE + endpoint, params, None, _fmp_throttle)


_PRICE_COLS = ["date", "Open", "High", "Low", "Close", "Volume"]


def _load_rows(path: Path):
    import gzip
    import pandas as pd
    empty = pd.DataFrame(columns=_PRICE_COLS[1:], index=pd.DatetimeIndex([], name="date"), dtype=float)
    if not path.exists():
        return empty
    with gzip.open(path, "rt") as fh:
        rows = json.load(fh)
    if not rows:
        return empty
    df = pd.DataFrame(rows, columns=_PRICE_COLS)
    df["date"] = pd.to_datetime(df.date)
    return df.set_index("date").sort_index().astype(float)


def load_fmp_prices(symbol: str):
    return _load_rows(FMP_DIR / "prices" / f"{symbol}.json.gz")


def load_prices(symbol: str):
    """Split-adjusted daily OHLCV: FMP, with Alpaca SIP bars filling dates FMP
    lacks (delisted stocks FMP dropped; 2016+). See m1_alpaca_fill.py."""
    import pandas as pd
    fmp = load_fmp_prices(symbol)
    alp = _load_rows(DATA_DIR / "alpaca" / f"{symbol}.json.gz")
    if alp.empty:
        return fmp
    if fmp.empty:
        return alp
    return pd.concat([fmp, alp[~alp.index.isin(fmp.index)]]).sort_index()


def load_cik_prices(cik: int):
    """A company's price series stitched across every ticker it used
    (meta/symbol_segments.csv.gz). Where segments overlap, the ticker seen
    most often in fails-to-deliver wins; the identity ticker fills the rest."""
    import pandas as pd
    seg = _segments().get(cik)
    if seg is None:
        return _load_rows(Path("/nonexistent"))
    parts = []
    for s in seg.sort_values("n_obs").itertuples():  # low priority first
        px = load_prices(s.symbol)
        parts.append(px[(px.index >= s.seg_from) & (px.index <= s.seg_to)].assign(symbol=s.symbol))
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


_SPLITS: dict = {}


def split_factor(symbol: str, d) -> float:
    """Multiply a split-adjusted price on date d by this to get the actual
    price then: product of every split ratio dated after d (FMP /splits)."""
    import pandas as pd
    if symbol not in _SPLITS:
        p = FMP_DIR / "splits" / f"{symbol}.json"
        rows = json.loads(p.read_text()) if p.exists() else None
        _SPLITS[symbol] = [(pd.Timestamp(r["date"]), r["numerator"] / r["denominator"])
                           for r in (rows or []) if r.get("numerator") and r.get("denominator")]
    f = 1.0
    for when, ratio in _SPLITS[symbol]:
        if when > d:
            f *= ratio
    return f


_SEG = None


def _segments():
    global _SEG
    if _SEG is None:
        import pandas as pd
        s = pd.read_csv(META_DIR / "symbol_segments.csv.gz", parse_dates=["seg_from", "seg_to"])
        _SEG = {c: g for c, g in s.groupby("cik")}
    return _SEG

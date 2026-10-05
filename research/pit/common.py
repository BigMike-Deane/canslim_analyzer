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


_sec_throttle = Throttle(per_minute=8 * 60)  # SEC fair access: <= 10 req/s
# The live scanner shares the FMP plan's 300/min budget. Default to 60/min;
# raise via PIT_FMP_PER_MIN only after market hours.
_fmp_throttle = Throttle(per_minute=float(os.environ.get("PIT_FMP_PER_MIN", 60)))


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

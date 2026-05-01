#!/usr/bin/env python3
"""Round 2 fetcher tuning bench.

Tests two additional knobs on top of Round 1 (yahoo_sem=5, fmp=12):
  - _yahoo_delay: 0.6s → 0.3s (per-call sleep gate)
  - _yahoo_max_retries: 4 → 2 (no-data retry budget)

The retry knob shows up only on flaky tickers, so each variant samples 80
tickers and we accept that absolute timings depend on how many flaky
tickers happened to land in each sample. The 429 count is the firm
safety signal — any nonzero value vetoes the variant.

Run inside the canslim container:

    docker exec canslim-analyzer python3 /app/scripts/bench_fetch_round2.py

Reverse-order pass to control for drift.
"""
import asyncio
import random
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/backend")

import async_data_fetcher as adf  # noqa: E402
import fmp_rate_limiter  # noqa: E402
from sp500_tickers import get_all_tickers  # noqa: E402

# All variants run on top of Round 1 (yahoo_sem=5, fmp=12) which is already
# the deployed prod config — _init_async_primitives sets these.
adf.save_scan_progress = lambda *a, **kw: None
adf.load_scan_progress = lambda *a, **kw: []
adf.clear_scan_progress = lambda *a, **kw: None


def _stats():
    s = fmp_rate_limiter.get_rate_limit_stats()
    return {"requests": s.get("total_requests", 0), "errors_429": s.get("errors_429", 0)}


async def bench(label, yahoo_delay, yahoo_max_retries, tickers):
    adf._yahoo_delay = yahoo_delay
    adf._yahoo_max_retries = yahoo_max_retries

    pre = _stats()
    t0 = time.time()
    results = await adf.fetch_stocks_batch_async(
        tickers, batch_size=50, resume_from_checkpoint=False
    )
    elapsed = time.time() - t0
    post = _stats()

    valid = sum(1 for r in results if r and getattr(r, "is_valid", False))
    print(
        f"{label:48s} | {elapsed:6.1f}s | "
        f"valid={valid:3d}/{len(tickers)} | "
        f"req={post['requests'] - pre['requests']:4d} | "
        f"429={post['errors_429'] - pre['errors_429']}",
        flush=True,
    )


async def main():
    rng = random.Random(43)  # different seed than round 1 to avoid sample overlap
    tickers = list(get_all_tickers())
    rng.shuffle(tickers)
    SAMPLE = 80
    samples = [tickers[i * SAMPLE : (i + 1) * SAMPLE] for i in range(5)]
    assert all(len(s) == SAMPLE for s in samples)

    print(
        f"Universe: {len(tickers)} tickers; using 5×{SAMPLE} disjoint random samples\n"
        f"Round 1 baseline already deployed (yahoo_sem=5, fmp=12).\n"
        f"{'CONFIG':48s} | {'TIME':>6} | {'VALID':>11} | {'REQ':>4} | {'429':>3}\n"
        + "-" * 96,
        flush=True,
    )

    # baseline = current prod (delay=0.6, retries=4)
    await bench("baseline_1  delay=0.6 retries=4", 0.6, 4, samples[0])
    # combined: aggressive on both knobs
    await bench("combined    delay=0.3 retries=2", 0.3, 2, samples[1])
    # isolate retries
    await bench("retries_only delay=0.6 retries=2", 0.6, 2, samples[2])
    # isolate delay
    await bench("delay_only  delay=0.3 retries=4", 0.3, 4, samples[3])
    # baseline drift control
    await bench("baseline_2  delay=0.6 retries=4", 0.6, 4, samples[4])


if __name__ == "__main__":
    asyncio.run(main())

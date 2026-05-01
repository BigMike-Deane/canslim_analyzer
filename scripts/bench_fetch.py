#!/usr/bin/env python3
"""Benchmark fetcher concurrency settings.

Runs ``fetch_stocks_batch_async`` with five (label, yahoo_sem, fmp_sem,
batch_size) configurations on disjoint random ticker samples, reporting
wall-clock, FMP request count, FMP 429s, and valid-fetch rate per run.

Run inside the canslim container, e.g.:

    docker exec canslim-analyzer python3 /app/scripts/bench_fetch.py

The two baseline runs bracket the variants so you can detect drift caused
by concurrent production scanning. A variant is "safe" only if it shows
no 429s above the baseline floor.
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

# --- Patch _init so we can override _yahoo_semaphore size after the original
# init runs (the original hardcodes Semaphore(3) at line 243).
_original_init = adf._init_async_primitives
_yahoo_target = 3


async def _patched_init():
    await _original_init()
    adf._yahoo_semaphore = asyncio.Semaphore(_yahoo_target)


adf._init_async_primitives = _patched_init

# --- Disable checkpoint persistence so we don't corrupt the production scan's
# checkpoint file (same scan_id = same date).
adf.save_scan_progress = lambda *a, **kw: None
adf.load_scan_progress = lambda *a, **kw: []
adf.clear_scan_progress = lambda *a, **kw: None


def _stats():
    s = fmp_rate_limiter.get_rate_limit_stats()
    return {"requests": s.get("total_requests", 0), "errors_429": s.get("errors_429", 0)}


async def bench(label, yahoo_sem, fmp_sem, batch_size, tickers):
    global _yahoo_target
    _yahoo_target = yahoo_sem
    adf.MAX_CONCURRENT_REQUESTS = fmp_sem

    pre = _stats()
    t0 = time.time()
    results = await adf.fetch_stocks_batch_async(
        tickers, batch_size=batch_size, resume_from_checkpoint=False
    )
    elapsed = time.time() - t0
    post = _stats()

    valid = sum(1 for r in results if r and getattr(r, "is_valid", False))
    print(
        f"{label:44s} | {elapsed:6.1f}s | "
        f"valid={valid:3d}/{len(tickers)} | "
        f"req={post['requests'] - pre['requests']:4d} | "
        f"429={post['errors_429'] - pre['errors_429']}",
        flush=True,
    )


async def main():
    rng = random.Random(42)
    tickers = list(get_all_tickers())
    rng.shuffle(tickers)
    SAMPLE = 60
    samples = [tickers[i * SAMPLE : (i + 1) * SAMPLE] for i in range(5)]
    assert all(len(s) == SAMPLE for s in samples), "not enough universe for 5x60"

    print(
        f"Universe: {len(tickers)} tickers; using 5×{SAMPLE} disjoint random samples\n"
        f"{'CONFIG':44s} | {'TIME':>6} | {'VALID':>11} | {'REQ':>4} | {'429':>3}\n"
        + "-" * 92,
        flush=True,
    )

    # Reverse-order run to confirm rankings aren't drift-induced.
    await bench("baseline_1  yahoo=3 fmp=8  batch=50",  3, 8,  50,  samples[0])
    await bench("variant_C   yahoo=3 fmp=8  batch=100", 3, 8,  100, samples[1])
    await bench("variant_B   yahoo=3 fmp=12 batch=50",  3, 12, 50,  samples[2])
    await bench("variant_A   yahoo=5 fmp=8  batch=50",  5, 8,  50,  samples[3])
    await bench("baseline_2  yahoo=3 fmp=8  batch=50",  3, 8,  50,  samples[4])


if __name__ == "__main__":
    asyncio.run(main())

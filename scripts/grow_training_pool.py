#!/usr/bin/env python3
"""Grow the deduped ML training pool by issuing a curated sweep of backtests.

The dedup rule is: latest backtest per (start_date, end_date), trade-level
dedup on (ticker, buy_date) keeps newest. So overlapping date ranges with
different strategies maximize unique-trade yield without wasting compute on
identical (start, end, strategy) configs already in the pool.

May 2026 context: pool sits at ~755 deduped trades; v18 isotonic calibration
overfit at this size. Target is 2000+ to either retry isotonic or — sooner —
support a less data-hungry sigmoid retrain (see ml/trainer.py
calibration_method="sigmoid").

Usage:
    # Dry-run (default): print the planned sweep, do not call the API
    python3 scripts/grow_training_pool.py

    # Issue the backtests
    python3 scripts/grow_training_pool.py --execute

    # Override defaults
    python3 scripts/grow_training_pool.py --execute \\
        --api-base https://canslim.duckdns.org \\
        --starting-cash 25000 \\
        --stock-universe all \\
        --token "$ADMIN_TOKEN"
"""

import argparse
import sys
from typing import Optional

import requests


# Curated sweep — overlapping windows × both live strategies. ~10 entries,
# each yielding ~50-150 unique deduped trades. Date ranges are deliberately
# offset by 6 months so trade-level dedup on (ticker, buy_date) lets distinct
# market regimes contribute fresh samples instead of collapsing.
SWEEP = [
    # nostate_optimized — D config, ML bonus + veto
    {"start_date": "2020-01-01", "end_date": "2024-01-01", "strategy": "nostate_optimized"},
    {"start_date": "2020-06-01", "end_date": "2024-06-01", "strategy": "nostate_optimized"},
    {"start_date": "2021-01-01", "end_date": "2025-01-01", "strategy": "nostate_optimized"},
    {"start_date": "2021-06-01", "end_date": "2025-06-01", "strategy": "nostate_optimized"},
    {"start_date": "2022-01-01", "end_date": "2026-01-01", "strategy": "nostate_optimized"},

    # nostate_cs_bear — C config, ML veto-only
    {"start_date": "2020-01-01", "end_date": "2024-01-01", "strategy": "nostate_cs_bear"},
    {"start_date": "2020-06-01", "end_date": "2024-06-01", "strategy": "nostate_cs_bear"},
    {"start_date": "2021-01-01", "end_date": "2025-01-01", "strategy": "nostate_cs_bear"},
    {"start_date": "2021-06-01", "end_date": "2025-06-01", "strategy": "nostate_cs_bear"},
    {"start_date": "2022-01-01", "end_date": "2026-01-01", "strategy": "nostate_cs_bear"},
]

# Rough per-backtest yield, post-dedup. Anchored to the May 2026 pool snapshot
# where 22+ overlapping backtests deduped down to ~755 trades. Bounds are
# conservative — actual yield depends on cache state + strategy config drift.
ESTIMATED_YIELD_LOW = 50
ESTIMATED_YIELD_HIGH = 150

# Rough per-backtest runtime (4-yr window, universe="all" on the VPS).
EST_RUNTIME_MIN = 12
EST_RUNTIME_MAX = 18


def build_payload(
    entry: dict, starting_cash: float, stock_universe: str
) -> dict:
    return {
        "start_date": entry["start_date"],
        "end_date": entry["end_date"],
        "starting_cash": starting_cash,
        "stock_universe": stock_universe,
        "strategy": entry["strategy"],
    }


def issue_backtest(
    api_base: str, payload: dict, token: Optional[str] = None
) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{api_base.rstrip('/')}/api/backtests"
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def print_dry_run(
    sweep: list, starting_cash: float, stock_universe: str, api_base: str
) -> None:
    n = len(sweep)
    runtime_lo = n * EST_RUNTIME_MIN
    runtime_hi = n * EST_RUNTIME_MAX
    yield_lo = n * ESTIMATED_YIELD_LOW
    yield_hi = n * ESTIMATED_YIELD_HIGH

    print(f"Planned sweep: {n} backtests against {api_base}")
    print(f"  starting_cash = ${starting_cash:,.0f}")
    print(f"  stock_universe = {stock_universe}")
    print()
    print(f"{'#':>3} {'strategy':<20} {'start':<12} {'end':<12}")
    print(f"{'-' * 3:>3} {'-' * 20:<20} {'-' * 12:<12} {'-' * 12:<12}")
    for i, entry in enumerate(sweep, 1):
        print(
            f"{i:>3} {entry['strategy']:<20} "
            f"{entry['start_date']:<12} {entry['end_date']:<12}"
        )
    print()
    print(
        f"Estimated wall time: {runtime_lo}-{runtime_hi} min "
        f"(serial; parallel runs cut this proportionally)"
    )
    print(
        f"Estimated unique-trade yield: ~{yield_lo}-{yield_hi} deduped trades "
        f"(very rough — depends on cache state + strategy overlap)"
    )
    print()
    print("Re-run with --execute to actually issue these backtests.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Issue a curated sweep of backtests to grow the ML training pool. "
            "Defaults to dry-run."
        )
    )
    parser.add_argument(
        "--api-base", default="https://canslim.duckdns.org",
        help="Base URL for the CANSLIM Analyzer API (default: production)",
    )
    parser.add_argument(
        "--starting-cash", type=float, default=25000.0,
        help="Starting cash per backtest (default: 25000)",
    )
    parser.add_argument(
        "--stock-universe", default="all",
        help="Stock universe per backtest (default: all)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually issue the backtests. Without this flag, prints the plan only.",
    )
    parser.add_argument(
        "--token", default=None,
        help="Admin bearer token. Required only if /api/backtests is admin-gated in your env.",
    )
    args = parser.parse_args()

    if not args.execute:
        print_dry_run(SWEEP, args.starting_cash, args.stock_universe, args.api_base)
        return 0

    print(f"Issuing {len(SWEEP)} backtests against {args.api_base} ...")
    print()
    issued = []
    failed = []
    for i, entry in enumerate(SWEEP, 1):
        payload = build_payload(entry, args.starting_cash, args.stock_universe)
        label = f"{entry['strategy']} {entry['start_date']}..{entry['end_date']}"
        try:
            resp = issue_backtest(args.api_base, payload, args.token)
            bt_id = resp.get("id") or resp.get("backtest_id") or "?"
            issued.append((bt_id, label))
            print(f"[{i}/{len(SWEEP)}] queued bt={bt_id}  {label}")
        except requests.HTTPError as e:
            failed.append((label, f"HTTP {e.response.status_code}: {e.response.text[:200]}"))
            print(f"[{i}/{len(SWEEP)}] FAILED  {label}  ({e})")
        except requests.RequestException as e:
            failed.append((label, str(e)[:200]))
            print(f"[{i}/{len(SWEEP)}] FAILED  {label}  ({e})")

    print()
    print(f"Done: {len(issued)} queued, {len(failed)} failed.")
    if issued:
        print("Backtest IDs:", ", ".join(str(bt_id) for bt_id, _ in issued))
    if failed:
        print()
        print("Failures (re-run later):")
        for label, err in failed:
            print(f"  {label}  ->  {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

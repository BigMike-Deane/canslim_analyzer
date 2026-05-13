#!/usr/bin/env python3
"""Grow the deduped ML training pool by issuing a curated sweep of backtests.

The dedup rule is: latest backtest per (start_date, end_date), trade-level
dedup on (ticker, buy_date) keeps newest. So overlapping date ranges with
different strategies maximize unique-trade yield without wasting compute on
identical (start, end, strategy) configs already in the pool.

May 2026 v2: queries the live pool via /api/admin/backtest-pairs and only
issues backtests whose (start_date, end_date) pair is NOT already in the
pool — re-running an existing pair just replaces the older run via dedup
and yields ZERO new training samples (lesson from 2026-05-13 sweep: 10
backtests, 3hr compute, 0 new rows because all 10 collided).

Usage:
    # Dry-run (default): print the planned fresh-only sweep
    python3 scripts/grow_training_pool.py --token "$ADMIN_TOKEN"

    # Issue the backtests
    python3 scripts/grow_training_pool.py --execute --token "$ADMIN_TOKEN"

    # Override defaults
    python3 scripts/grow_training_pool.py --execute \\
        --api-base https://canslim.duckdns.org \\
        --starting-cash 25000 \\
        --stock-universe all \\
        --max-per-strategy 6 \\
        --token "$ADMIN_TOKEN"
"""

import argparse
import sys
from datetime import date
from typing import Optional

import requests


# Strategies to grow the pool for. Each gets `max_per_strategy` fresh
# date-range entries, chosen from the candidate grid below.
STRATEGIES = ["nostate_optimized", "nostate_cs_bear"]

# Candidate grid: quarterly start months × years 2018-2022 × window lengths
# {3, 4}. Generates 5 * 4 * 2 = 40 candidates per strategy before filtering.
# Plenty of diversity to find fresh combinations even when the pool grows.
CANDIDATE_START_YEARS = [2018, 2019, 2020, 2021, 2022]
CANDIDATE_START_MONTHS = [1, 4, 7, 10]
CANDIDATE_WINDOW_YEARS = [3, 4]

# Rough per-backtest yield, post-dedup.
ESTIMATED_YIELD_LOW = 50
ESTIMATED_YIELD_HIGH = 150

# Rough per-backtest runtime (4-yr window, universe="all" on the VPS).
EST_RUNTIME_MIN = 12
EST_RUNTIME_MAX = 18


def _add_years(d: date, years: int) -> date:
    """Add `years` to a date, capping Feb 29 → Feb 28 on non-leap years."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        # Feb 29 + N where target year is non-leap: bump to Feb 28.
        return d.replace(year=d.year + years, day=28)


def generate_candidate_grid(strategies: list = None) -> list[dict]:
    """Cartesian-product candidate grid. Pre-filter to windows that end on
    or before today (we can't backtest into the future)."""
    strategies = strategies or STRATEGIES
    today = date.today()
    out = []
    for strategy in strategies:
        for year in CANDIDATE_START_YEARS:
            for month in CANDIDATE_START_MONTHS:
                start = date(year, month, 1)
                for wyrs in CANDIDATE_WINDOW_YEARS:
                    end = _add_years(start, wyrs)
                    if end > today:
                        continue
                    out.append({
                        "start_date": start.isoformat(),
                        "end_date": end.isoformat(),
                        "strategy": strategy,
                    })
    return out


def filter_fresh(candidates: list[dict], existing: set[tuple]) -> list[dict]:
    """Drop candidates whose (start, end, strategy) is already in the pool."""
    return [
        c for c in candidates
        if (c["start_date"], c["end_date"], c["strategy"]) not in existing
    ]


def fetch_existing_pairs(api_base: str, token: Optional[str] = None) -> set[tuple]:
    """Query the live pool for completed-backtest (start, end, strategy) triples.

    Returns a set of (start_date_iso, end_date_iso, strategy) tuples for
    O(1) lookup during filtering.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{api_base.rstrip('/')}/api/admin/backtest-pairs"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    return {
        (p["start_date"], p["end_date"], p["strategy"])
        for p in body.get("pairs", [])
    }


def build_sweep(api_base: str, token: Optional[str], max_per_strategy: int) -> list[dict]:
    """Generate the full candidate grid, filter against the live pool, then
    pick the first `max_per_strategy` fresh entries per strategy.
    """
    existing = fetch_existing_pairs(api_base, token)
    candidates = generate_candidate_grid()
    fresh = filter_fresh(candidates, existing)
    # Group by strategy and pick first N each. Preserves grid order
    # (older-year/Q1 first) for deterministic output.
    out = []
    for strategy in STRATEGIES:
        picked = [c for c in fresh if c["strategy"] == strategy][:max_per_strategy]
        out.extend(picked)
    return out


def build_payload(entry: dict, starting_cash: float, stock_universe: str) -> dict:
    return {
        "start_date": entry["start_date"],
        "end_date": entry["end_date"],
        "starting_cash": starting_cash,
        "stock_universe": stock_universe,
        "strategy": entry["strategy"],
    }


def issue_backtest(api_base: str, payload: dict, token: Optional[str] = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{api_base.rstrip('/')}/api/backtests"
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def print_dry_run(sweep: list, existing_count: int, candidate_count: int,
                  starting_cash: float, stock_universe: str, api_base: str) -> None:
    n = len(sweep)
    runtime_lo = n * EST_RUNTIME_MIN
    runtime_hi = n * EST_RUNTIME_MAX
    yield_lo = n * ESTIMATED_YIELD_LOW
    yield_hi = n * ESTIMATED_YIELD_HIGH

    print(f"Planned sweep: {n} fresh backtests against {api_base}")
    print(f"  starting_cash  = ${starting_cash:,.0f}")
    print(f"  stock_universe = {stock_universe}")
    print(f"  pool already has {existing_count} unique (start, end, strategy) triples")
    print(f"  candidate grid generated {candidate_count} candidates, "
          f"{candidate_count - n} filtered as already-in-pool")
    print()
    if not sweep:
        print("No fresh date ranges left in the candidate grid — widen "
              "CANDIDATE_START_YEARS / CANDIDATE_WINDOW_YEARS / "
              "CANDIDATE_START_MONTHS to generate more.")
        return
    print(f"{'#':>3} {'strategy':<20} {'start':<12} {'end':<12}")
    print(f"{'-' * 3:>3} {'-' * 20:<20} {'-' * 12:<12} {'-' * 12:<12}")
    for i, entry in enumerate(sweep, 1):
        print(
            f"{i:>3} {entry['strategy']:<20} "
            f"{entry['start_date']:<12} {entry['end_date']:<12}"
        )
    print()
    print(f"Estimated wall time: {runtime_lo}-{runtime_hi} min (serial)")
    print(f"Estimated unique-trade yield: ~{yield_lo}-{yield_hi} deduped trades")
    print()
    print("Re-run with --execute to actually issue these backtests.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Issue a curated sweep of backtests to grow the ML training pool. "
            "Only issues date ranges NOT already in the pool. Defaults to dry-run."
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
        "--max-per-strategy", type=int, default=5,
        help="Max fresh backtests to issue per strategy (default: 5)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually issue the backtests. Without this flag, prints the plan only.",
    )
    parser.add_argument(
        "--token", default=None,
        help="Admin bearer token. Required: /api/admin/backtest-pairs is admin-gated.",
    )
    args = parser.parse_args()

    try:
        existing = fetch_existing_pairs(args.api_base, args.token)
    except requests.HTTPError as e:
        print(f"Failed to fetch existing pairs: HTTP {e.response.status_code}: {e.response.text[:200]}", file=sys.stderr)
        return 1
    except requests.RequestException as e:
        print(f"Failed to fetch existing pairs: {e}", file=sys.stderr)
        return 1

    candidates = generate_candidate_grid()
    fresh = filter_fresh(candidates, existing)

    sweep = []
    for strategy in STRATEGIES:
        picked = [c for c in fresh if c["strategy"] == strategy][:args.max_per_strategy]
        sweep.extend(picked)

    if not args.execute:
        print_dry_run(sweep, len(existing), len(candidates),
                      args.starting_cash, args.stock_universe, args.api_base)
        return 0

    if not sweep:
        print("No fresh date ranges to issue. Run without --execute to see the candidate analysis.")
        return 0

    print(f"Issuing {len(sweep)} fresh backtests against {args.api_base} ...")
    print()
    issued = []
    failed = []
    for i, entry in enumerate(sweep, 1):
        payload = build_payload(entry, args.starting_cash, args.stock_universe)
        label = f"{entry['strategy']} {entry['start_date']}..{entry['end_date']}"
        try:
            resp = issue_backtest(args.api_base, payload, args.token)
            bt_id = resp.get("id") or resp.get("backtest_id") or "?"
            issued.append((bt_id, label))
            print(f"[{i}/{len(sweep)}] queued bt={bt_id}  {label}")
        except requests.HTTPError as e:
            failed.append((label, f"HTTP {e.response.status_code}: {e.response.text[:200]}"))
            print(f"[{i}/{len(sweep)}] FAILED  {label}  ({e})")
        except requests.RequestException as e:
            failed.append((label, str(e)[:200]))
            print(f"[{i}/{len(sweep)}] FAILED  {label}  ({e})")

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

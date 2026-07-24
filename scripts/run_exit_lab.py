#!/usr/bin/env python3
"""
Exit Lab runner — end-to-end harness around backend/exit_lab.py.

Usage:
    python3 scripts/run_exit_lab.py --trades trades.json [--out results.json]

``trades.json`` is a JSON array of ai_portfolio_trades rows with at least:
user_id, ticker, action, shares, price, executed_at, signal_factors.
Export via psql/MCP:
    SELECT user_id, ticker, action, shares, price, executed_at, signal_factors
    FROM ai_portfolio_trades ORDER BY executed_at;

Prices are fetched from Yahoo (daily OHLC) per distinct ticker, from each
ticker's earliest entry to today, so slow variants can ride past the actual
exit date. Run from the repo root.

NOTE: scripts/ is not in the Docker image (by design) — this is an analysis
tool, not a runtime component. It never touches live positions or config.
"""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.exit_lab import (  # noqa: E402
    default_variant_grid,
    reconstruct_episodes,
    run_grid,
    stop_slippage,
)


def fetch_prices(tickers: dict) -> dict:
    """{ticker: earliest_entry_date} -> {ticker: [daily bar dicts]} via yfinance."""
    import yfinance as yf

    prices = {}
    for ticker, start in sorted(tickers.items()):
        yf_ticker = ticker.replace("-", "-")  # yfinance uses '-' for class shares already
        try:
            df = yf.Ticker(yf_ticker).history(
                start=str(start - timedelta(days=5)), end=None,
                interval="1d", auto_adjust=False,
            )
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  ! {ticker}: fetch failed ({e})", file=sys.stderr)
            continue
        if df is None or df.empty:
            print(f"  ! {ticker}: no price data", file=sys.stderr)
            continue
        prices[ticker] = [
            {
                "date": idx.date().isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }
            for idx, row in df.iterrows()
        ]
    return prices


def fmt_pct(x: float) -> str:
    return f"{x:+.1f}%"


def print_report(out: dict) -> None:
    summary = out["summary"]
    baseline = out["baseline"]
    print(f"\n=== Exit Lab: {summary[baseline]['n']} episodes replayed "
          f"({out['skipped_no_prices']} skipped, no prices) ===\n")
    header = (f"{'variant':<22}{'mean ret':>10}{'95% CI':>18}{'vs base':>10}"
              f"{'sig':>5}{'win%':>7}{'effcy':>7}{'hold_d':>8}{'open':>6}")
    print(header)
    print("-" * len(header))
    ranked = sorted(summary.items(),
                    key=lambda kv: kv[1].get("mean_return_pct", -1e9), reverse=True)
    for name, s in ranked:
        if s.get("n", 0) == 0:
            continue
        ci = s["ci95"]
        vs = s.get("vs_baseline", {})
        diff = fmt_pct(vs["mean_diff_pp"]) if vs.get("n") else "—"
        sig = "*" if vs.get("significant") else ""
        print(f"{name:<22}{fmt_pct(s['mean_return_pct']):>10}"
              f"{'[' + fmt_pct(ci[0]) + ',' + fmt_pct(ci[1]) + ']':>18}"
              f"{diff:>10}{sig:>5}"
              f"{s['win_rate'] * 100:>6.0f}%"
              f"{s['mean_efficiency']:>7.2f}"
              f"{s['mean_holding_days']:>8.1f}"
              f"{s['censored']:>6}")
    print("\n('sig' = paired bootstrap 95% CI on the per-episode diff excludes 0. "
          "Early-exit variants are understated: replay has no capital recycling.)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", required=True, help="JSON array of trade rows")
    ap.add_argument("--prices", help="optional pre-fetched prices JSON (ticker -> bars)")
    ap.add_argument("--out", help="write full results JSON here")
    ap.add_argument("--include-open", action="store_true",
                    help="also replay still-open episodes (censored at last close); "
                         "closed-only excludes surviving winners and favors early exits")
    args = ap.parse_args()

    trades = json.loads(Path(args.trades).read_text())
    episodes = reconstruct_episodes(trades)
    closed = [e for e in episodes if e.closed]
    pool = episodes if args.include_open else closed
    print(f"Reconstructed {len(episodes)} episodes ({len(closed)} closed, "
          f"{len(episodes) - len(closed)} open/censored)")

    slippage = stop_slippage(trades)
    if slippage.get("n"):
        print(f"Stop-loss slippage: n={slippage['n']}, "
              f"mean +{slippage['mean_slippage_pp']:.1f}pp beyond configured stop "
              f"(max +{slippage['max_slippage_pp']:.1f}pp)")

    if args.prices:
        prices = json.loads(Path(args.prices).read_text())
    else:
        needed: dict = {}
        for e in pool:
            prev = needed.get(e.ticker)
            needed[e.ticker] = min(prev, e.entry_date) if prev else e.entry_date
        print(f"Fetching daily OHLC for {len(needed)} tickers...")
        prices = fetch_prices(needed)

    out = run_grid(pool, prices, default_variant_grid(), include_open=args.include_open)
    print_report(out)

    if args.out:
        serializable = {
            "baseline": out["baseline"],
            "skipped_no_prices": out["skipped_no_prices"],
            "summary": out["summary"],
            "slippage": slippage,
            "generated": date.today().isoformat(),
        }
        Path(args.out).write_text(json.dumps(serializable, indent=2, default=str))
        print(f"\nFull summary written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

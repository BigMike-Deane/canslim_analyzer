"""
Exit Lab — counterfactual replay of exit-rule variants over historical trades.

Answers "would exit rule X have done better?" by replaying alternate exit
configs (stop width, trailing bands, partial-profit ladder) over the daily
price paths of REAL episodes reconstructed from ai_portfolio_trades. This is
valid for exit rules specifically because they are price-mechanical: unlike
scorer changes (which frozen-score backtest replay cannot honestly evaluate),
a trailing band or stop width operates only on the price path, which is fully
known historically.

Scope and honesty limits (screening tool, NOT a promotion verdict):
  - Single-lot model: each episode is replayed as one unit bought at the
    initial entry price. Pyramids affect only the trailing-widening rule
    (via the episode's recorded pyramid_count), not the basis. Actual live
    returns are capital-weighted against a blended basis, so absolute levels
    differ; CROSS-VARIANT comparison is the point (biases cancel — every
    variant replays the same simplified model on the same paths).
  - No capital recycling: an early exit here does not free cash for the next
    buy, so variants that exit early are UNDERSTATED relative to a live
    portfolio. Treat "early-exit variant ties baseline" as a win for it.
  - Score gates on the ladder and partial-trailing are assumed to pass
    (replay has no score history); pre-earnings tightening and score-crash
    sells are not modeled.
  - Daily-bar approximation matches live cadence deliberately: trailing
    stops evaluate at the CLOSE (production trailing_cadence.daily_only fires
    them once in the close window) while hard stops check the intraday LOW
    with gap-through pricing (live hard stops stay intraday).

Trigger semantics mirror ai_trader.evaluate_sells / trading_engine as of
2026-07-24 (order: hard stop -> trailing [partial-on-trailing for 2+ pyramid
positions, peak reset] -> partial-profit ladder). tests/test_exit_lab.py
contains a parity test pinning the default bands to
trading_engine.get_trailing_stop_pct — if the live tiers change, that test
fails rather than this module silently drifting.

Pure module: no DB, no config_loader, no network. The __main__ runner reads
JSON exports (see scripts/run_exit_lab.py for the end-to-end harness).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Episode reconstruction from ai_portfolio_trades rows
# ---------------------------------------------------------------------------

_SHARE_EPS = 1e-6


@dataclass
class Episode:
    """One position lifecycle: first BUY to the SELL that closes it."""
    user_id: int
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date | None = None       # None while still open (censored)
    closed: bool = False
    pyramid_count: int = 0
    total_buy_cost: float = 0.0
    total_sell_proceeds: float = 0.0
    open_shares: float = 0.0
    sell_reasons: list = field(default_factory=list)

    @property
    def actual_return_pct(self) -> float | None:
        """Capital-weighted realized return (closed episodes only)."""
        if not self.closed or self.total_buy_cost <= 0:
            return None
        return (self.total_sell_proceeds / self.total_buy_cost - 1.0) * 100.0


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def reconstruct_episodes(trades: list[dict]) -> list[Episode]:
    """Rebuild position lifecycles from trade rows.

    ``trades`` are dicts with at least: user_id, ticker, action
    (BUY/PYRAMID/SELL), shares, price, executed_at. Rows are processed in
    executed_at order per (user_id, ticker); an episode closes when open
    shares return to ~0, and a later BUY on the same key starts a fresh one.
    SELL rows with no open episode (pre-history partials) are ignored.
    """
    by_key: dict = {}
    for t in trades:
        key = (t["user_id"], t["ticker"])
        by_key.setdefault(key, []).append(t)

    episodes: list[Episode] = []
    for (user_id, ticker), rows in by_key.items():
        rows.sort(key=lambda r: str(r["executed_at"]))
        current: Episode | None = None
        for r in rows:
            action = r["action"]
            shares = float(r["shares"] or 0)
            price = float(r["price"] or 0)
            when = _as_date(r["executed_at"])
            if action in ("BUY", "SEED"):
                if current is None:
                    current = Episode(
                        user_id=user_id, ticker=ticker,
                        entry_date=when, entry_price=price,
                    )
                    episodes.append(current)
                current.open_shares += shares
                current.total_buy_cost += shares * price
            elif action == "PYRAMID":
                if current is None:
                    continue  # pyramid into pre-history position: unusable
                current.pyramid_count += 1
                current.open_shares += shares
                current.total_buy_cost += shares * price
            elif action == "SELL":
                if current is None:
                    continue
                current.open_shares -= shares
                current.total_sell_proceeds += shares * price
                sf = r.get("signal_factors") or {}
                current.sell_reasons.append(sf.get("sell_reason") or r.get("reason", ""))
                if current.open_shares <= max(_SHARE_EPS, abs(current.open_shares) * 1e-9) \
                        or current.open_shares < 0.01:
                    current.exit_date = when
                    current.closed = True
                    current = None
    return episodes


# ---------------------------------------------------------------------------
# Exit-rule variant configuration
# ---------------------------------------------------------------------------

# (min_peak_gain_pct, trailing_band_pct) — highest tier first. Defaults mirror
# the live champion profile (nostate_optimized / nostate_cs_bear).
CHAMPION_BANDS: tuple = ((50, 25.0), (30, 18.0), (20, 12.0), (10, 6.0), (5, 4.0))

# (gain_pct, cumulative_sell_pct) — highest tier first. Mirrors
# ai_trader.partial_profits defaults.
CHAMPION_LADDER: tuple = ((50, 75.0), (40, 50.0), (25, 25.0))


@dataclass(frozen=True)
class ExitConfig:
    name: str
    stop_loss_pct: float = 7.0
    trailing_bands: tuple = CHAMPION_BANDS
    band_scale: float = 1.0             # multiplies every band (0.5 = twice as tight)
    pyramid_widening: bool = True       # +2%/pyramid, max +6 (trading_engine)
    partial_on_trailing: bool = True    # 2+ pyramids: sell half, reset peak
    partial_sell_pct: float = 50.0
    ladder: tuple = CHAMPION_LADDER
    ladder_enabled: bool = True


def trailing_band_for(peak_gain_pct: float, cfg: ExitConfig, pyramid_count: int) -> float | None:
    """Band (in %) for the given peak gain, or None below the minimum tier.

    Mirrors trading_engine.get_trailing_stop_pct + apply_pyramid_widening;
    pinned by the parity test in tests/test_exit_lab.py.
    """
    band = None
    for threshold, pct in cfg.trailing_bands:
        if peak_gain_pct >= threshold:
            band = pct * cfg.band_scale
            break
    if band is None:
        return None
    if cfg.pyramid_widening and pyramid_count > 0:
        band += min(pyramid_count * 2.0, 6.0)
    return band


def ladder_action(gain_pct: float, taken_pct: float, cfg: ExitConfig) -> float | None:
    """Additional percent-of-position to sell now, or None.

    Mirrors trading_engine.get_partial_profit_action tier walk (highest tier
    first, cumulative targets), with score gates assumed to pass.
    """
    if not cfg.ladder_enabled:
        return None
    for gain_threshold, target_cum in cfg.ladder:
        if gain_pct >= gain_threshold:
            take = target_cum - taken_pct
            return take if take > 0 else None
    return None


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    config_name: str
    ticker: str
    user_id: int
    entry_date: date
    weighted_return_pct: float      # sum of leg_fraction * leg_return
    exit_date: date | None          # date the last share left (None if censored)
    holding_days: int
    mfe_pct: float                  # max favorable excursion (intraday high)
    mae_pct: float                  # max adverse excursion (intraday low)
    exit_efficiency: float          # (1 + ret) / (1 + mfe), 1.0 = sold the top
    censored: bool                  # ran out of price path with shares open
    legs: list = field(default_factory=list)  # (date, fraction, return_pct, kind)


def replay_episode(
    entry_price: float,
    bars: list[dict],
    cfg: ExitConfig,
    pyramid_count: int = 0,
    ticker: str = "",
    user_id: int = 0,
) -> ReplayResult | None:
    """Replay one exit config over an episode's daily bars.

    ``bars``: chronological dicts with date/open/high/low/close, the first bar
    being the entry day. Triggers start on the bar AFTER entry (the live
    system buys intraday; same-day peak starts at the entry price).
    """
    if entry_price <= 0 or len(bars) < 2:
        return None

    entry_day = _as_date(bars[0]["date"])
    frac = 1.0                  # fraction of the original position still open
    taken_pct = 0.0             # cumulative partial-profit percentage (live semantics)
    peak = entry_price
    legs: list = []
    mfe = 0.0
    mae = 0.0
    stop_price = entry_price * (1.0 - cfg.stop_loss_pct / 100.0)

    def ret(price: float) -> float:
        return (price / entry_price - 1.0) * 100.0

    for bar in bars[1:]:
        day = _as_date(bar["date"])
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        opn = float(bar["open"])
        mfe = max(mfe, ret(high))
        mae = min(mae, ret(low))

        # 1) Hard stop — intraday, gap-aware: a gap below the stop fills at
        #    the open, not at the stop price (this is where live slippage
        #    beyond the configured width comes from).
        if low <= stop_price:
            fill = min(opn, stop_price)
            legs.append((day, frac, ret(fill), "STOP"))
            frac = 0.0
            break

        peak = max(peak, high)

        # 2) Trailing stop — evaluated at the close (daily_only cadence).
        peak_gain = (peak / entry_price - 1.0) * 100.0
        band = trailing_band_for(peak_gain, cfg, pyramid_count)
        if band is not None and peak > 0:
            drop = (peak - close) / peak * 100.0
            if drop >= band:
                if cfg.partial_on_trailing and pyramid_count >= 2 and frac > 0.5 * _SHARE_EPS:
                    sell_frac = frac * (cfg.partial_sell_pct / 100.0)
                    legs.append((day, sell_frac, ret(close), "PARTIAL_TRAIL"))
                    frac -= sell_frac
                    taken_pct += cfg.partial_sell_pct
                    peak = close  # live resets peak after a partial trail
                    continue
                legs.append((day, frac, ret(close), "TRAIL"))
                frac = 0.0
                break

        # 3) Partial-profit ladder — at the close, cumulative tiers.
        gain = ret(close)
        take = ladder_action(gain, taken_pct, cfg)
        if take is not None and frac > 0:
            sell_frac = frac * (take / 100.0)
            legs.append((day, sell_frac, gain, f"LADDER_{int(take)}"))
            frac -= sell_frac
            taken_pct += take

    censored = frac > _SHARE_EPS
    if censored:
        last = bars[-1]
        legs.append((_as_date(last["date"]), frac, ret(float(last["close"])), "OPEN"))

    weighted = sum(f * r for _, f, r, _ in legs)
    last_exit = legs[-1][0] if legs else entry_day
    return ReplayResult(
        config_name=cfg.name,
        ticker=ticker,
        user_id=user_id,
        entry_date=entry_day,
        weighted_return_pct=weighted,
        exit_date=None if censored else last_exit,
        holding_days=(last_exit - entry_day).days,
        mfe_pct=mfe,
        mae_pct=mae,
        exit_efficiency=(1.0 + weighted / 100.0) / (1.0 + mfe / 100.0) if mfe > -100 else 0.0,
        censored=censored,
        legs=legs,
    )


# ---------------------------------------------------------------------------
# Aggregation & bootstrap
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(values: list[float], n_boot: int = 2000, seed: int = 42,
                      lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Deterministic via seed."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(n)) / n
        for _ in range(n_boot)
    )
    def pct(p: float) -> float:
        idx = min(n_boot - 1, max(0, int(round(p / 100.0 * (n_boot - 1)))))
        return means[idx]
    return (pct(lo), pct(hi))


def aggregate(results: list[ReplayResult]) -> dict:
    """Summary stats for one variant across episodes."""
    rets = [r.weighted_return_pct for r in results]
    if not rets:
        return {"n": 0}
    mean = sum(rets) / len(rets)
    ci_lo, ci_hi = bootstrap_mean_ci(rets)
    return {
        "n": len(rets),
        "mean_return_pct": mean,
        "median_return_pct": sorted(rets)[len(rets) // 2],
        "ci95": (ci_lo, ci_hi),
        "win_rate": sum(1 for r in rets if r > 0) / len(rets),
        "mean_efficiency": sum(r.exit_efficiency for r in results) / len(results),
        "mean_holding_days": sum(r.holding_days for r in results) / len(results),
        "censored": sum(1 for r in results if r.censored),
        "total_return_sum": sum(rets),
    }


def paired_diff(variant: list[ReplayResult], baseline: list[ReplayResult],
                n_boot: int = 2000, seed: int = 42) -> dict:
    """Per-episode paired comparison (same episodes, two rules).

    Pairing by (user_id, ticker, entry_date) removes cross-episode variance —
    this is why replay reaches significance orders of magnitude faster than
    comparing two live portfolios over calendar time.
    """
    key = lambda r: (r.user_id, r.ticker, r.entry_date)
    base_map = {key(r): r for r in baseline}
    diffs = [
        r.weighted_return_pct - base_map[key(r)].weighted_return_pct
        for r in variant if key(r) in base_map
    ]
    if not diffs:
        return {"n": 0}
    mean = sum(diffs) / len(diffs)
    ci = bootstrap_mean_ci(diffs, n_boot=n_boot, seed=seed)
    return {
        "n": len(diffs),
        "mean_diff_pp": mean,
        "ci95": ci,
        "pct_episodes_better": sum(1 for d in diffs if d > 0) / len(diffs),
        "significant": not (ci[0] <= 0.0 <= ci[1]),
    }


# ---------------------------------------------------------------------------
# Variant grid
# ---------------------------------------------------------------------------

def default_variant_grid() -> list[ExitConfig]:
    """The standing screening grid. baseline_champion replays today's live rules."""
    base = ExitConfig(name="baseline_champion")
    return [
        base,
        # Tighter / looser trailing bands across the board
        replace(base, name="bands_x0.6", band_scale=0.6),
        replace(base, name="bands_x0.8", band_scale=0.8),
        replace(base, name="bands_x1.25", band_scale=1.25),
        replace(base, name="bands_x1.5", band_scale=1.5),
        # Flat bands (ignore gain tiers)
        replace(base, name="flat_trail_8", trailing_bands=((0, 8.0),)),
        replace(base, name="flat_trail_15", trailing_bands=((0, 15.0),)),
        # Stop width
        replace(base, name="stop_6", stop_loss_pct=6.0),
        replace(base, name="stop_8", stop_loss_pct=8.0),
        replace(base, name="stop_10", stop_loss_pct=10.0),
        # Realize-earlier ladders (the "are we holding too long" axis)
        replace(base, name="ladder_off", ladder_enabled=False),
        replace(base, name="ladder_early",
                ladder=((40, 75.0), (30, 50.0), (20, 25.0))),
        replace(base, name="ladder_aggressive",
                ladder=((30, 90.0), (20, 60.0), (15, 30.0))),
        replace(base, name="take_all_at_25", ladder=((25, 100.0),)),
        # Structure levers
        replace(base, name="no_pyramid_widening", pyramid_widening=False),
        replace(base, name="no_partial_trailing", partial_on_trailing=False),
    ]


def run_grid(
    episodes: list[Episode],
    prices: dict,
    variants: list[ExitConfig] | None = None,
    include_open: bool = False,
) -> dict:
    """Replay every variant over every episode with available prices.

    ``prices``: {ticker: [bar, ...]} chronological daily OHLC covering each
    episode's window. Bars are sliced from the entry date to the end of the
    supplied path (beyond the actual exit, so slower variants can keep
    riding and censoring is explicit rather than silent).

    ``include_open``: also replay still-open episodes (censored legs valued
    at the last supplied close). Closed-only excludes the surviving winners
    — the let-it-run tail — so it is biased TOWARD early-exit variants;
    include_open is the fairer view for "are we holding too long" questions
    (every variant is marked to the same final date).
    """
    variants = variants or default_variant_grid()
    grid: dict = {v.name: [] for v in variants}
    skipped = 0
    for ep in episodes:
        if not ep.closed and not include_open:
            continue
        bars = prices.get(ep.ticker) or []
        window = [b for b in bars if _as_date(b["date"]) >= ep.entry_date]
        if len(window) < 2:
            skipped += 1
            continue
        for v in variants:
            result = replay_episode(
                ep.entry_price, window, v,
                pyramid_count=ep.pyramid_count,
                ticker=ep.ticker, user_id=ep.user_id,
            )
            if result is not None:
                grid[v.name].append(result)
    baseline_name = variants[0].name
    summary = {
        name: {
            **aggregate(results),
            **({"vs_baseline": paired_diff(results, grid[baseline_name])}
               if name != baseline_name else {}),
        }
        for name, results in grid.items()
    }
    return {"summary": summary, "results": grid, "skipped_no_prices": skipped,
            "baseline": baseline_name}


# ---------------------------------------------------------------------------
# Stop-slippage diagnostic (actual trades, no replay)
# ---------------------------------------------------------------------------

def stop_slippage(trades: list[dict]) -> dict:
    """How far beyond the configured stop the actual STOP LOSS fills landed."""
    slips = []
    for t in trades:
        sf = t.get("signal_factors") or {}
        if t.get("action") == "SELL" and sf.get("sell_reason") == "STOP LOSS":
            gain = sf.get("gain_pct")
            # `or` (not a .get default): exports may carry an explicit null
            stop = sf.get("stop_pct") or 8.0
            if gain is not None:
                slips.append(abs(float(gain)) - float(stop))
    if not slips:
        return {"n": 0}
    return {
        "n": len(slips),
        "mean_slippage_pp": sum(slips) / len(slips),
        "max_slippage_pp": max(slips),
        "ci95": bootstrap_mean_ci(slips),
    }

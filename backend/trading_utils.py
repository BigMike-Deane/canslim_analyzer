"""
Shared Trading Utilities

Common functions and constants used by both ai_trader.py and backtester.py.
Extracted to eliminate duplication and ensure these stay in sync.
"""

import sys
import os

# Add parent directory to path for config_loader import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import config


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _nan_safe(val, default=0):
    """Convert None/NaN to a safe default. float('nan') is truthy and passes `or 0`."""
    if val is None:
        return default
    try:
        if val != val:  # NaN != NaN per IEEE 754
            return default
    except (TypeError, ValueError):
        pass
    return val


def get_strategy_profile(strategy_name: str = "balanced") -> dict:
    """Load strategy profile from YAML config, falling back to balanced defaults."""
    profiles = config.get('strategy_profiles', {})
    profile = profiles.get(strategy_name, profiles.get('balanced', {}))
    return profile


# ---------------------------------------------------------------------------
# ML Signal Layer ordinal maps (must match ml/feature_extractor.py)
# ---------------------------------------------------------------------------

ENTRY_TYPE_MAP_ML = {"breakout": 0, "pre-breakout": 1, "standard": 2}
REGIME_MAP_ML = {"bearish": 0, "neutral": 1, "bullish": 2}


# ---------------------------------------------------------------------------
# Trading allocation limits - loaded from config with fallbacks
# ---------------------------------------------------------------------------

MIN_CASH_RESERVE_PCT = config.get('ai_trader.allocation.min_cash_reserve_pct', default=0.10)
MAX_SECTOR_ALLOCATION = config.get('ai_trader.allocation.max_sector_allocation', default=0.30)
MAX_STOCKS_PER_SECTOR = config.get('ai_trader.allocation.max_stocks_per_sector', default=4)
MAX_POSITION_ALLOCATION = config.get('ai_trader.allocation.max_single_position', default=0.15)


# ---------------------------------------------------------------------------
# Sync helpers — pure functions inline-duplicated between ai_trader and
# backtester. Extracted so a future refactor of either side cannot silently
# desync. Past sync drift cost 25-100pp on the 4-yr backtest each time
# (NaN poisoning rounds 1+2; partial-profit 25% tier delta). These helpers
# close that structural risk by giving both call sites a single source of
# truth.
# ---------------------------------------------------------------------------

# Minimum allocation room (as fraction of portfolio) below which a sector
# is treated as full instead of being squeezed into a tiny remaining slot.
SECTOR_MIN_ROOM_FRACTION = 0.02


def should_take_partial_on_trailing_stop(
    pyramid_count: int,
    score: float,
    partial_on_trailing: bool = True,
    partial_min_pyramid_count: int = 2,
    partial_min_score: float = 65,
) -> bool:
    """Decide whether a trailing-stop hit should sell partial vs full.

    High-conviction positions (pyramided into 2+ times AND still scoring
    strong) sell only a fraction (default 50%) and keep the remainder on a
    fresh, wider trailing stop. Everyone else takes the full sell.

    Pure function — no I/O, no globals. Both call sites read the same
    YAML keys (`ai_trader.trailing_stops.partial_*`) and pass them in.

    Behavior contract:
      - All four config knobs gate the decision; any falsy/sub-threshold
        value bypasses the partial path.
      - Pass `score=0` (or any value < `partial_min_score`) when the
        underlying score is missing/unavailable. The live trader uses a
        `score_available = score > 0` precondition; passing 0 here
        reproduces that semantically.

    Call sites:
      - backend/ai_trader.py::evaluate_sells (around the trailing-stop block)
      - backend/backtester.py::Backtester._evaluate_sells (trailing-stop block)

    Args:
        pyramid_count: How many times this position has been pyramided.
        score: Current effective CANSLIM/Growth score for the ticker. Pass
            0 if the score is unavailable.
        partial_on_trailing: Master kill-switch from YAML. False ⇒ always full.
        partial_min_pyramid_count: Min pyramids required for partial.
        partial_min_score: Min score required for partial.

    Returns:
        True if a partial sell is appropriate; False ⇒ take a full sell.
    """
    if not partial_on_trailing:
        return False
    if pyramid_count < partial_min_pyramid_count:
        return False
    if score < partial_min_score:
        return False
    return True


def apply_sector_allocation_cap(
    requested_position_value: float,
    current_sector_value: float,
    portfolio_value: float,
    max_sector_allocation: float = MAX_SECTOR_ALLOCATION,
    min_room_fraction: float = SECTOR_MIN_ROOM_FRACTION,
) -> float:
    """Cap a buy amount so the resulting sector allocation stays under the limit.

    Mirrors the live trader's `check_sector_limit` allocation arm and the
    backtester's inline allocation check. The count-based cap
    (MAX_STOCKS_PER_SECTOR) is checked separately at both call sites and is
    NOT part of this helper — keeping concerns separate makes the math
    here easy to test in isolation.

    Algorithm:
      1. If portfolio_value <= 0, no cap can be computed → return as-is.
         (Both call sites only call this branch when portfolio_value > 0.)
      2. Compute new_alloc = (current_sector_value + requested) / portfolio_value.
      3. If new_alloc <= max_sector_allocation, no cap needed → return as-is.
      4. Else compute remaining_room = max_sector_allocation - current_alloc.
         - If remaining_room <= min_room_fraction (default 2%), reject (return 0)
           — squeezing into a sliver isn't worth the bookkeeping.
         - Otherwise scale down to remaining_room * portfolio_value.

    Pure function — no I/O, no globals.

    Call sites:
      - backend/ai_trader.py::check_sector_limit (live)
      - backend/backtester.py::Backtester._evaluate_buys (sector cap block)

    Args:
        requested_position_value: Dollar size of the buy being considered.
        current_sector_value: Total current $ value held in this sector.
        portfolio_value: Total portfolio value (cash + positions).
        max_sector_allocation: Cap as fraction of portfolio (default 0.30).
        min_room_fraction: Below this fraction of room remaining, treat
            sector as full (default 0.02 = 2%).

    Returns:
        Adjusted dollar amount: original (no cap), reduced (squeezed to fit),
        or 0.0 (sector full → caller should skip the buy).
    """
    if portfolio_value <= 0:
        return requested_position_value

    current_alloc = current_sector_value / portfolio_value
    new_alloc = current_alloc + (requested_position_value / portfolio_value)

    if new_alloc <= max_sector_allocation:
        return requested_position_value

    remaining_room = max_sector_allocation - current_alloc
    if remaining_room <= min_room_fraction:
        return 0.0

    return remaining_room * portfolio_value


def select_effective_stop_loss_pct(
    profile: dict,
    stop_loss_config: dict,
    default_normal_pct: float,
    is_bearish_market: bool,
    vix_proxy: float | None = None,
    vix_config: dict | None = None,
) -> float:
    """Resolve the effective stop-loss percentage for a position.

    Combines the bearish-market gate (use bearish_stop_loss_pct when SPY is
    below its 50-day MA) with the optional VIX-regime multiplier in a single
    pure function. Three call sites previously inlined this same shape:

      - backend/ai_trader.py::check_stop_losses (live, persisted-config default)
      - backend/ai_trader.py::evaluate_sells   (live, persisted-config default)
      - backend/backtester.py::Backtester._evaluate_sells (backtest-config default)

    The fallback for `normal_stop_loss_pct` is intentionally a parameter:
      - Live trader passes the user's `AIPortfolioConfig.stop_loss_pct`
      - Backtester passes `BacktestRun.stop_loss_pct`
    Same semantic role ("the runtime's configured normal stop"), different
    concrete sources. Bearish fallback (7.0) is hard-coded — consistent across
    all three sites historically.

    VIX adjustment is opt-in: pass `vix_proxy=None` (the default) to skip,
    e.g. when the data provider is unavailable or the lookup raised. The
    helper does NOT fetch VIX — that I/O lives at each call site (lazy-loaded
    HistoricalDataProvider in live, injected `data_provider` in backtester).

    Resolution order for normal_pct (highest priority first):
      1. profile['stop_loss_pct']
      2. stop_loss_config['normal_stop_loss_pct']
      3. default_normal_pct (caller-supplied)
    Same chain for bearish_pct, with hard-coded fallback 7.0.

    Pure function — no I/O, no globals.

    Args:
        profile: Strategy profile dict (may override stop_loss_pct + bearish).
        stop_loss_config: YAML `ai_trader.stops` section.
        default_normal_pct: Caller's runtime default (e.g. live portfolio's
            stop_loss_pct or backtest run's stop_loss_pct).
        is_bearish_market: True if SPY < 50MA — selects bearish stop.
        vix_proxy: VIX-proxy reading for `current_date`. None ⇒ no VIX adjust.
        vix_config: YAML `vix_stops` section. None or `{enabled: false}` ⇒
            no VIX adjust even if vix_proxy is provided.

    Returns:
        Effective stop-loss percentage (e.g. 7.0 = 7%). NaN-safe in practice
        because NaN comparisons short-circuit both vix branches → returns the
        unmultiplied base.
    """
    normal_pct = profile.get(
        'stop_loss_pct',
        stop_loss_config.get('normal_stop_loss_pct', default_normal_pct),
    )
    bearish_pct = profile.get(
        'bearish_stop_loss_pct',
        stop_loss_config.get('bearish_stop_loss_pct', 7.0),
    )
    effective = bearish_pct if is_bearish_market else normal_pct

    if vix_proxy is None or vix_config is None:
        return effective
    if not vix_config.get('enabled', True):
        return effective

    low_vix = vix_config.get('low_vix_threshold', 15)
    high_vix = vix_config.get('high_vix_threshold', 25)
    if vix_proxy < low_vix:
        effective *= vix_config.get('low_vix_stop_tighten', 0.80)
    elif vix_proxy > high_vix:
        effective *= vix_config.get('high_vix_stop_widen', 1.20)
    return effective

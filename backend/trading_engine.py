"""
Common Trading Engine - Shared trading logic between ai_trader.py and backtester.py

This module extracts the core trading DECISION logic that must stay in sync
between live trading (ai_trader.py) and historical backtesting (backtester.py).

Builds on trading_utils.py (basic utilities/constants) by adding higher-level
trading patterns: trailing stops, score crash detection, partial profit tiers,
composite scoring, position sizing, entry signal detection.

CRITICAL: Any change to trading logic here affects BOTH live and backtest.
This is intentional — the CLAUDE.md rule "AI Trader <> Backtester MUST stay
in sync" is enforced by having a single source of truth.
"""

import logging
import math

from config_loader import config
from backend.trading_utils import _nan_safe  # noqa: F401 - re-export for convenience

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trailing stop calculation
# ---------------------------------------------------------------------------

def get_trailing_stop_pct(peak_gain_pct: float, profile: dict) -> float | None:
    """
    Determine the trailing stop percentage based on peak gain level.
    Uses dynamic thresholds from strategy profile.

    Args:
        peak_gain_pct: How much the position was up at its peak (from cost basis)
        profile: Strategy profile dict with optional 'trailing_stops' overrides

    Returns:
        Trailing stop percentage, or None if peak gain is below minimum threshold
    """
    profile_trailing = profile.get('trailing_stops', {})
    if peak_gain_pct >= 50:
        return profile_trailing.get('gain_50_plus', 25)
    elif peak_gain_pct >= 30:
        return profile_trailing.get('gain_30_to_50', 18)
    elif peak_gain_pct >= 20:
        return profile_trailing.get('gain_20_to_30', 12)
    elif peak_gain_pct >= 10:
        return profile_trailing.get('gain_10_to_20', 8)
    elif peak_gain_pct >= 5:
        return profile_trailing.get('gain_5_to_10', 4)
    return None


def apply_pyramid_widening(trailing_stop_pct: float, pyramid_count: int) -> float:
    """
    Widen trailing stop for pyramided (high-conviction) positions.
    +2% per pyramid, max +6%.

    Args:
        trailing_stop_pct: Base trailing stop percentage
        pyramid_count: Number of times position has been pyramided

    Returns:
        Adjusted trailing stop percentage
    """
    if pyramid_count > 0:
        pyramid_widening = min(pyramid_count * 2.0, 6.0)
        return trailing_stop_pct + pyramid_widening
    return trailing_stop_pct


def apply_new_position_guard(
    current_stop_pct: float,
    *,
    guard_config: dict,
    holding_days: int,
    pyramid_count: int,
) -> float:
    """Tighten a new position's stop to ``guard_stop_pct`` during its first
    ``guard_days``, so a fast-falling fresh buy can't ride a pro-cyclically
    widening ATR stop toward the 20% cap (jun-09 bug: FSLR held to -18%).

    This is the SHARED DECISION used by the live trader
    (ai_trader: ``_check_and_execute_stop_losses_impl`` and ``evaluate_sells``)
    and the simulator (backtester ``_evaluate_sells``). Each caller sources its
    own inputs — live uses ``date.today()`` and yaml-only config; the
    backtester uses the simulation ``current_date`` and profile-overridable
    config — but the threshold/skip/min logic lives here so the three sites
    cannot drift (which is exactly how the jun-09 parity bug happened).

    Args:
        current_stop_pct: The stop% computed so far (e.g. the ATR-adjusted stop).
        guard_config: The ``new_position_guard`` config dict (already sourced).
        holding_days: Calendar days the position has been held.
        pyramid_count: How many times the position has been pyramided.

    Returns:
        The (possibly tightened) stop percentage. Unchanged when the guard is
        disabled, the position is past the window, or it's a skipped pyramid.
    """
    if not guard_config.get('enabled', False):
        return current_stop_pct
    guard_days = guard_config.get('guard_days', 21)
    guard_stop_pct = guard_config.get('guard_stop_pct', 8.0)
    skip_if_pyramided = guard_config.get('skip_if_pyramided', True)
    if holding_days <= guard_days and not (skip_if_pyramided and pyramid_count > 0):
        return min(current_stop_pct, guard_stop_pct)
    return current_stop_pct


def trailing_stops_allowed_now(yaml_config=None, now_et=None) -> bool:
    """Trailing-stop cadence gate (June-19 exit-parity item 2).

    When ``ai_trader.trailing_cadence.daily_only`` is enabled, live trailing
    stops only fire inside the close window (default: the final hour before the
    16:00 ET close), i.e. once per day — matching the backtester's daily
    cadence and removing the intraday-churn drag (measured −1..−17pp per 2yr
    window, 8/8 arms; live-validated: trailing exits held 9.2d / 50% WR / −0.1%
    avg vs 81d / 76% / +3.6% modeled). HARD stops (stop-loss, score-crash) are
    NOT gated by this — they stay intraday for crash protection.

    Default-OFF: returns True (always allow) when the lever is disabled, so
    behavior is unchanged until the lever is flipped. The backtester does not
    call this — it already evaluates once daily at the close, so enabling the
    live lever is what RESTORES trader↔backtester trailing-cadence parity.

    Args:
        yaml_config: config object (defaults to global config).
        now_et: override for the current ET time (tests); defaults to now in
            America/New_York.
    """
    if yaml_config is None:
        yaml_config = config
    cfg = yaml_config.get('ai_trader.trailing_cadence', {})
    if not cfg.get('daily_only', False):
        return True

    if now_et is None:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))

    start_hour = cfg.get('window_start_hour_et', 15)
    start_minute = cfg.get('window_start_minute_et', 0)
    close_hour = cfg.get('market_close_hour_et', 16)
    window_start = now_et.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    market_close = now_et.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    return window_start <= now_et <= market_close


# ---------------------------------------------------------------------------
# ATR-stop last-known cache (D3)
# ---------------------------------------------------------------------------
# Live calculate_atr_stop() does a per-cycle Yahoo fetch and silently returns
# the BASE stop on any failure (throttle/404/timeout) — so a position's stop
# snaps from (say) 18% to 7% for that cycle, risking a random premature
# stop-out, then snaps back. These let the live path reuse the most recent good
# ATR stop on a failed fetch instead. Process-local (the trading cycle is one
# process); off until calculate_atr_stop() calls them (June-19).

_atr_stop_cache: dict = {}  # ticker -> (date, effective_stop_pct)


def cache_atr_stop(ticker: str, effective_stop_pct: float, today=None) -> None:
    """Record a successfully-computed ATR stop for `ticker` (call on success)."""
    from datetime import date as _date
    _atr_stop_cache[ticker] = (today or _date.today(), float(effective_stop_pct))


def atr_stop_fallback(ticker: str, base_stop_pct: float, max_age_days: int = 5, today=None) -> float:
    """Effective stop to use when the live ATR fetch FAILS: the last good ATR
    stop for `ticker` if cached within `max_age_days`, else `base_stop_pct`.
    Avoids snapping a wide volatility stop down to base on a transient Yahoo
    failure (D3)."""
    from datetime import date as _date, timedelta
    entry = _atr_stop_cache.get(ticker)
    if entry is None:
        return base_stop_pct
    cached_date, stop = entry
    if ((today or _date.today()) - cached_date) <= timedelta(days=max_age_days):
        return stop
    return base_stop_pct


# ---------------------------------------------------------------------------
# Score stability / blip detection
# ---------------------------------------------------------------------------

def dedup_scores_to_daily(score_rows, lookback: int) -> list:
    """Reduce scan-frequency score rows to one per calendar day — the latest
    scan of each day — most-recent day first, capped at ``lookback`` days (D2).

    Live ``check_score_stability`` reads the last N *scans* (every ~90 min), so
    "3 consecutive low" can confirm a score crash in a single bad afternoon,
    while the backtester appends once per *trading day*. Feeding the live
    history through this first aligns the two clocks. Input must be ordered
    newest-first; each row needs a ``.timestamp`` (datetime). Off until
    ``check_score_stability`` calls it (June-19).
    """
    seen = {}
    for row in score_rows:
        ts = getattr(row, "timestamp", None)
        day = ts.date() if ts is not None else None
        if day is None or day in seen:
            continue
        seen[day] = row          # first-seen per day == latest scan (desc order)
        if len(seen) >= lookback:
            break
    return list(seen.values())


def check_score_stability_from_history(
    recent_scores: list,
    current_score: float,
    threshold: float = 50,
    recent_low_window: int = 3,
) -> dict:
    """
    Check if a low score is consistent across recent scans (not a one-time blip).
    Prevents whipsaw sells on transient data issues.

    This is the pure logic — callers provide the score history.
    ai_trader fetches from StockScore DB table; backtester uses in-memory history.

    Args:
        recent_scores: List of recent scores in CHRONOLOGICAL order (oldest first,
                       most recent last). Order matters: consecutive_low walks
                       reversed(recent_scores) from the newest, and recent_low_count
                       takes recent_scores[-recent_low_window:] (the newest N). Passing
                       reverse-ordered input would count the wrong scans for both the
                       "consecutive low" and "2 of last 3" gates. Callers must sort
                       oldest-first (ai_trader reverses DB desc rows; backtester appends
                       in date order).
        current_score: Current score to evaluate
        threshold: Score threshold below which we consider "low"

    Returns:
        dict with:
        - is_stable: True if score has been consistently low (not a blip)
        - recent_scores: the scores used
        - avg_score: average of recent scores
        - consecutive_low: count of consecutive low scores from most recent
    """
    if len(recent_scores) < 2:
        low = 1 if current_score < threshold else 0
        return {
            "is_stable": True,
            "recent_scores": [current_score] if not recent_scores else recent_scores,
            "avg_score": current_score,
            "consecutive_low": low,
            "recent_low_count": low,
        }

    avg_score = sum(recent_scores) / len(recent_scores)

    # Count consecutive low scores from most recent
    consecutive_low = 0
    for score in reversed(recent_scores):
        if score < threshold:
            consecutive_low += 1
        else:
            break

    # Count total low scores in last N scans (for "2 of last 3" gate)
    last_n = recent_scores[-recent_low_window:] if len(recent_scores) >= recent_low_window else recent_scores
    recent_low_count = sum(1 for s in last_n if s < threshold)

    # Check if current score is significantly lower than average (potential blip)
    score_variance = abs(current_score - avg_score)

    # If current score is much lower than recent average AND only 1 low scan, it might be a blip
    # Require consecutive_low < 2 to flag as blip — if 2+ consecutive low, it's a real drop
    is_blip = (current_score < threshold and
               avg_score > threshold + 10 and
               score_variance > 15 and
               consecutive_low < 2)

    return {
        "is_stable": not is_blip,
        "recent_scores": recent_scores,
        "avg_score": avg_score,
        "consecutive_low": consecutive_low,
        "recent_low_count": recent_low_count
    }


# ---------------------------------------------------------------------------
# Partial profit tier evaluation
# ---------------------------------------------------------------------------

def get_partial_profit_action(
    gain_pct: float,
    current_score: float,
    partial_taken: float,
    yaml_config=None
) -> dict | None:
    """
    Evaluate whether a position qualifies for partial profit taking.
    Checks three tiers (highest first): 50%/40%/25% gain thresholds.

    Args:
        gain_pct: Current gain percentage
        current_score: Current effective score
        partial_taken: Cumulative percentage already taken as partial profits
        yaml_config: Config object (defaults to global config)

    Returns:
        dict with {tier, take_pct, target_sell_pct, reason, priority} or None
    """
    if yaml_config is None:
        yaml_config = config

    partial_profit_config = yaml_config.get('ai_trader.partial_profits', {})

    pp_50_gain = partial_profit_config.get('threshold_50pct', {}).get('gain_pct', 50)
    pp_50_sell = partial_profit_config.get('threshold_50pct', {}).get('sell_pct', 75)
    pp_50_min_score = partial_profit_config.get('threshold_50pct', {}).get('min_score', 55)

    pp_40_gain = partial_profit_config.get('threshold_40pct', {}).get('gain_pct', 40)
    pp_40_sell = partial_profit_config.get('threshold_40pct', {}).get('sell_pct', 50)
    pp_40_min_score = partial_profit_config.get('threshold_40pct', {}).get('min_score', 60)

    pp_25_gain = partial_profit_config.get('threshold_25pct', {}).get('gain_pct', 25)
    pp_25_sell = partial_profit_config.get('threshold_25pct', {}).get('sell_pct', 25)
    pp_25_min_score = partial_profit_config.get('threshold_25pct', {}).get('min_score', 60)

    # Check highest tier first
    if gain_pct >= pp_50_gain and current_score >= pp_50_min_score and partial_taken < pp_50_sell:
        take_pct = pp_50_sell - partial_taken
        if take_pct > 0:
            return {
                "tier": 50,
                "take_pct": take_pct,
                "target_sell_pct": pp_50_sell,
                "reason": f"PARTIAL PROFIT {pp_50_sell}%: Up {gain_pct:.1f}%, score {current_score:.0f} - protecting big winner",
                "priority": 3,
            }

    elif gain_pct >= pp_40_gain and current_score >= pp_40_min_score and partial_taken < pp_40_sell:
        take_pct = pp_40_sell - partial_taken
        if take_pct > 0:
            return {
                "tier": 40,
                "take_pct": take_pct,
                "target_sell_pct": pp_40_sell,
                "reason": f"PARTIAL PROFIT {pp_40_sell}%: Up {gain_pct:.1f}%, score {current_score:.0f} still strong",
                "priority": 4,
            }

    elif gain_pct >= pp_25_gain and current_score >= pp_25_min_score and partial_taken < pp_25_sell:
        take_pct = pp_25_sell - partial_taken
        if take_pct > 0:
            return {
                "tier": 25,
                "take_pct": take_pct,
                "target_sell_pct": pp_25_sell,
                "reason": f"PARTIAL PROFIT {pp_25_sell}%: Up {gain_pct:.1f}%, score {current_score:.0f} still strong",
                "priority": 5,
            }

    return None


# ---------------------------------------------------------------------------
# Pre-breakout / breakout / extended detection + base quality scoring
# ---------------------------------------------------------------------------

def calculate_base_quality_bonus(base_type: str, weeks_in_base: int) -> int:
    """
    Calculate base pattern quality bonus (up to 15 points).

    Args:
        base_type: Pattern type (cup_with_handle, cup, double_bottom, flat, none)
        weeks_in_base: Number of weeks in the base pattern

    Returns:
        Bonus points for base quality
    """
    has_base = base_type not in ('none', '', None)
    if not has_base:
        return 0

    bonus = 0
    if base_type == "cup_with_handle":
        bonus = 10
    elif base_type == "cup":
        bonus = 8
    elif base_type == "double_bottom":
        bonus = 7
    elif base_type == "flat":
        bonus = 6

    # Extra bonus for longer consolidation (max +5)
    if weeks_in_base >= 8:
        bonus += 5
    elif weeks_in_base >= 6:
        bonus += 3
    elif weeks_in_base >= 5:
        bonus += 1

    return bonus


def calculate_entry_signals(
    current_price: float,
    week_52_high: float,
    pivot_price: float,
    base_type: str,
    weeks_in_base: int,
    is_breaking_out: bool,
    breakout_volume_ratio: float,
    volume_ratio: float,
    effective_score: float,
) -> dict:
    """
    Calculate momentum score, breakout bonus, pre-breakout bonus, and extended penalty.
    This is the core entry signal evaluation shared by ai_trader and backtester.

    Returns:
        dict with momentum_score, breakout_bonus, pre_breakout_bonus, extended_penalty,
        entry_type, pct_from_pivot
    """
    has_base = base_type not in ('none', '', None)
    momentum_score = 0
    breakout_bonus = 0
    pre_breakout_bonus = 0
    extended_penalty = 0
    pct_from_pivot = 0

    if not week_52_high or week_52_high <= 0 or not current_price:
        return {
            "momentum_score": 0,
            "breakout_bonus": 0,
            "pre_breakout_bonus": 0,
            "extended_penalty": 0,
            "entry_type": "standard",
            "pct_from_pivot": 0,
        }

    pct_from_high = ((week_52_high - current_price) / week_52_high) * 100

    if pivot_price > 0:
        pct_from_pivot = ((pivot_price - current_price) / pivot_price) * 100

    # PRE-BREAKOUT: 5-15% below pivot with valid base pattern
    if has_base and pivot_price > 0 and 5 <= pct_from_pivot <= 15:
        pre_breakout_bonus = 40
        momentum_score = 35
        if volume_ratio >= 1.3:
            pre_breakout_bonus += 5
        if weeks_in_base >= 10:
            pre_breakout_bonus += 5

    # AT PIVOT ZONE: 0-5% below pivot with base pattern
    elif has_base and pivot_price > 0 and 0 <= pct_from_pivot < 5:
        pre_breakout_bonus = 35
        momentum_score = 30
        if volume_ratio >= 1.5:
            momentum_score += 5

    # BREAKOUT STOCKS - buying AFTER the pivot point
    elif is_breaking_out:
        breakout_bonus = 10
        if breakout_volume_ratio >= 2.0:
            breakout_bonus += 5
        momentum_score = 15

    # EXTENDED: More than 5% above pivot
    elif has_base and pivot_price > 0 and pct_from_pivot < -5:
        if pct_from_pivot < -10:
            extended_penalty = -20
            momentum_score = 5
        else:
            extended_penalty = -10
            momentum_score = 10

    # NO BASE PATTERN: Use 52-week high with penalties
    elif not has_base:
        if pct_from_high <= 2:
            if effective_score < 85:
                extended_penalty = -15
                momentum_score = 5
            else:
                momentum_score = 12
        elif pct_from_high <= 10:
            momentum_score = 15
            if volume_ratio >= 1.5:
                momentum_score += 3
        elif pct_from_high <= 25:
            momentum_score = 8
        else:
            momentum_score = -5

    # Determine entry type
    if is_breaking_out:
        entry_type = "breakout"
    elif pre_breakout_bonus >= 15:
        entry_type = "pre-breakout"
    else:
        entry_type = "standard"

    return {
        "momentum_score": momentum_score,
        "breakout_bonus": breakout_bonus,
        "pre_breakout_bonus": pre_breakout_bonus,
        "extended_penalty": extended_penalty,
        "entry_type": entry_type,
        "pct_from_pivot": pct_from_pivot,
    }


# ---------------------------------------------------------------------------
# Composite buy scoring
# ---------------------------------------------------------------------------

def calculate_composite_score(
    growth_projection: float,
    effective_score: float,
    momentum_score: float,
    breakout_bonus: float,
    pre_breakout_bonus: float,
    base_quality_bonus: float,
    extended_penalty: float,
    coiled_spring_bonus: float = 0,
    rs_line_bonus: float = 0,
    earnings_drift_bonus: float = 0,
    estimate_revision_bonus: float = 0,
    profile: dict = None,
) -> float:
    """
    Calculate the composite buy score — the weighted blend that ranks candidates.

    Returns:
        Composite score (higher = better buy candidate)
    """
    if profile is None:
        profile = {}

    scoring_weights = profile.get('scoring_weights', {})
    w_growth = scoring_weights.get('growth_projection', 0.25)
    w_score = scoring_weights.get('canslim_score', 0.25)
    w_momentum = scoring_weights.get('momentum', 0.20)
    w_breakout = scoring_weights.get('breakout', 0.20)
    w_base = scoring_weights.get('base_quality', 0.10)

    composite = (
        (growth_projection * w_growth) +
        (effective_score * w_score) +
        (momentum_score * w_momentum) +
        ((breakout_bonus + pre_breakout_bonus) * w_breakout) +
        (base_quality_bonus * w_base) +
        extended_penalty +
        coiled_spring_bonus +
        rs_line_bonus +
        earnings_drift_bonus +
        estimate_revision_bonus
    )

    return composite


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def calculate_position_size_pct(
    composite_score: float,
    max_positions: int,
    profile: dict,
    regime_max_pct: float = 12.0,
    yaml_config=None,
) -> float:
    """
    Calculate position size as percentage of portfolio using conviction-based sizing.

    Returns:
        Position size as percentage of portfolio (before multipliers)
    """
    if yaml_config is None:
        yaml_config = config

    conv_config = yaml_config.get('ai_trader.conviction_sizing', {})
    profile_conv = profile.get('conviction_sizing', {})

    if conv_config.get('enabled', False):
        conv_min = profile_conv.get('min_pct', conv_config.get('min_pct', 8.0))
        conv_max = profile_conv.get('max_pct', conv_config.get('max_pct', 20.0))
        score_floor = profile_conv.get('score_floor', conv_config.get('score_floor', 30))
        score_ceiling = profile_conv.get('score_ceiling', conv_config.get('score_ceiling', 75))
        score_ratio = max(0, min(1, (composite_score - score_floor) / max(1, score_ceiling - score_floor)))
        position_pct = conv_min + score_ratio * (conv_max - conv_min)
    else:
        min_position_pct = 90.0 / max_positions
        conviction_multiplier = min(composite_score / 50, 1.5)
        conviction_pct = min_position_pct + (conviction_multiplier * (regime_max_pct - min_position_pct) / 1.5)
        position_pct = max(min_position_pct, conviction_pct)

    return position_pct


def apply_position_size_multipliers(
    position_pct: float,
    pre_breakout_bonus: float,
    has_base: bool,
    is_breaking_out: bool,
    breakout_volume_ratio: float,
    is_coiled_spring: bool,
    in_soft_zone: bool,
    soft_zone_mult: float,
    is_correction_zone: bool,
    cz_position_mult: float,
    heat_penalty_active: bool,
    market_state_multiplier: float,
    profile: dict,
    yaml_config=None,
) -> float:
    """
    Apply all position size multipliers (pre-breakout, CS, soft zone, CZ, heat, market state).

    Returns:
        Adjusted position size percentage (before regime/profile cap)
    """
    if yaml_config is None:
        yaml_config = config

    # Half-size positions when portfolio heat is elevated
    if heat_penalty_active:
        position_pct *= 0.50

    # Market state position sizing
    if market_state_multiplier != 1.0:
        position_pct *= market_state_multiplier

    # PREDICTIVE POSITION SIZING: Pre-breakout stocks get largest positions
    pre_breakout_mult = profile.get('pre_breakout_multiplier', 1.40)
    if pre_breakout_bonus >= 35 and has_base:
        position_pct *= pre_breakout_mult
    elif pre_breakout_bonus >= 25 and has_base:
        position_pct *= (pre_breakout_mult * 0.93)
    elif is_breaking_out and breakout_volume_ratio >= 1.5:
        position_pct *= 1.0  # No boost

    # Coiled Spring position boost
    if is_coiled_spring:
        cs_config = yaml_config.get('coiled_spring', {})
        cs_multiplier = cs_config.get('position_multiplier', 1.25)
        position_pct *= cs_multiplier

    # SOFT ZONE: Apply graduated position multiplier
    if in_soft_zone and soft_zone_mult < 1.0:
        position_pct *= soft_zone_mult

    # CORRECTION ZONE: Reduced position size
    if is_correction_zone:
        position_pct *= cz_position_mult

    # Cap at profile max
    profile_max_pct = profile.get('max_single_position_pct', 25)
    position_pct = min(position_pct, profile_max_pct)

    return position_pct


# ---------------------------------------------------------------------------
# Sell reason categorization
# ---------------------------------------------------------------------------

def categorize_sell_reason(reason: str) -> str:
    """Extract sell category from reason string for trade attribution."""
    reason_upper = reason.upper()
    if "PARTIAL TRAILING STOP" in reason_upper:
        return "PARTIAL TRAILING"
    elif "TRAILING STOP" in reason_upper:
        return "TRAILING STOP"
    elif "STOP LOSS" in reason_upper:
        return "STOP LOSS"
    elif "SCORE CRASH" in reason_upper:
        return "SCORE CRASH"
    elif "TAKE PROFIT" in reason_upper:
        return "TAKE PROFIT"
    elif "PARTIAL PROFIT" in reason_upper:
        return "PARTIAL PROFIT"
    elif "PROTECT GAINS" in reason_upper:
        return "PROTECT GAINS"
    elif "WEAK POSITION" in reason_upper:
        return "WEAK POSITION"
    elif "CIRCUIT BREAKER" in reason_upper:
        return "CIRCUIT BREAKER"
    elif "PRE-EARNINGS" in reason_upper:
        return "PRE-EARNINGS"
    return "OTHER"


# ---------------------------------------------------------------------------
# Pre-earnings trailing stop tightening
# ---------------------------------------------------------------------------

def get_tightened_trailing_stop(
    peak_gain_pct: float,
    profile: dict,
    tighten_factor: float = 0.50,
) -> float | None:
    """
    Calculate tightened trailing stop for positions approaching earnings.
    Non-CS positions get tighter stops to protect gains before earnings risk.

    Returns:
        Tightened trailing stop percentage, or None if below minimum threshold
    """
    base_stop = get_trailing_stop_pct(peak_gain_pct, profile)
    if base_stop is not None:
        return base_stop * tighten_factor
    return None


# ---------------------------------------------------------------------------
# Score crash evaluation
# ---------------------------------------------------------------------------

def evaluate_score_crash(
    current_score: float,
    purchase_score: float,
    gain_pct: float,
    stability: dict,
    profile: dict,
    yaml_config=None,
) -> dict | None:
    """
    Evaluate whether a position should be sold due to score crash.
    Includes stability check, consecutive requirement, and profitability exception.

    Returns:
        dict with {should_sell, reason, consecutive_low, score_drop} or None if no crash
    """
    if yaml_config is None:
        yaml_config = config

    score_crash_config = yaml_config.get('ai_trader.score_crash', {})
    consecutive_required = score_crash_config.get('consecutive_required', 3)
    score_threshold = score_crash_config.get('threshold', 50)
    drop_required = profile.get('score_crash_drop_required', score_crash_config.get('drop_required', 20))
    ignore_if_profitable_pct = profile.get('score_crash_ignore_if_profitable', score_crash_config.get('ignore_if_profitable_pct', 10))

    score_drop = purchase_score - current_score
    if score_drop <= drop_required or current_score >= score_threshold:
        return None

    if gain_pct >= ignore_if_profitable_pct:
        return {"should_sell": False, "reason": f"SKIP score crash - profitable (+{gain_pct:.1f}%)"}

    if not stability["is_stable"]:
        return {"should_sell": False, "reason": f"Possible blip: current {current_score:.0f} vs avg {stability['avg_score']:.0f}"}

    consecutive_low = stability.get("consecutive_low", 0)
    recent_low_count = stability.get("recent_low_count", 0)

    # "2 of last 3" gate: trigger if 2+ of the last 3 scans were below threshold,
    # even if they weren't consecutive (one bounce no longer resets the clock).
    # Falls back to the legacy consecutive gate for backwards compatibility.
    recent_low_required = score_crash_config.get('recent_low_required', 2)
    recent_low_window = score_crash_config.get('recent_low_window', 3)

    meets_consecutive = consecutive_low >= consecutive_required
    meets_recent_low = recent_low_count >= recent_low_required

    if not meets_consecutive and not meets_recent_low:
        return {
            "should_sell": False,
            "reason": f"Only {consecutive_low} consecutive / {recent_low_count} of last {recent_low_window} low score(s)",
            "consecutive_low": consecutive_low,
        }

    trigger_method = "consecutive" if meets_consecutive else f"{recent_low_count}-of-{recent_low_window}"
    return {
        "should_sell": True,
        "reason": f"SCORE CRASH: {purchase_score:.0f} → {current_score:.0f} "
                  f"(avg: {stability['avg_score']:.0f}, {trigger_method} low scans)",
        "consecutive_low": consecutive_low,
        "score_drop": score_drop,
    }


# ---------------------------------------------------------------------------
# Sanitize signal factors for JSON serialization
# ---------------------------------------------------------------------------

def sanitize_signal_factors(sf: dict) -> dict:
    """Sanitize signal_factors dict so it's valid JSON (NaN/Inf are not valid JSON)."""
    if not sf:
        return sf
    clean = {}
    for k, v in sf.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = 0
        else:
            clean[k] = v
    return clean

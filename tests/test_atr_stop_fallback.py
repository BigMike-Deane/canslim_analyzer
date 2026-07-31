"""Freeze-safe half of June-19 item D3 — ATR-stop last-known cache.

`trading_engine.atr_stop_fallback` lets the live `calculate_atr_stop` reuse the
most recent good ATR stop when its Yahoo fetch fails, instead of silently
snapping the stop down to base (which causes random premature stop-outs). The
ai_trader.py hookup is staged in docs/june19-minor-items-playbook.md.
"""

from datetime import date

from backend.trading_engine import (
    cache_atr_stop,
    atr_stop_fallback,
    get_cached_atr_stop,
    _atr_stop_cache,
)


def setup_function():
    _atr_stop_cache.clear()


def test_fallback_returns_base_when_uncached():
    assert atr_stop_fallback("NEW", 7.0) == 7.0


def test_fallback_returns_recent_cached_stop():
    cache_atr_stop("FSLR", 18.0, today=date(2026, 6, 12))
    # same day and within the 5-day window → reuse the wide stop, not base
    assert atr_stop_fallback("FSLR", 7.0, today=date(2026, 6, 12)) == 18.0
    assert atr_stop_fallback("FSLR", 7.0, today=date(2026, 6, 15)) == 18.0


def test_fallback_to_base_when_cache_stale():
    cache_atr_stop("FSLR", 18.0, today=date(2026, 6, 1))
    # 11 days old > max_age_days(5) → don't trust a stale ATR, use base
    assert atr_stop_fallback("FSLR", 7.0, today=date(2026, 6, 12)) == 7.0


def test_cache_overwrites_with_latest():
    cache_atr_stop("AMD", 15.0, today=date(2026, 6, 10))
    cache_atr_stop("AMD", 11.0, today=date(2026, 6, 12))
    assert atr_stop_fallback("AMD", 7.0, today=date(2026, 6, 12)) == 11.0


# get_cached_atr_stop — read-only view used by the Exit Plan display (returns
# None instead of a base fallback, so callers can tell "no fresh reading" apart
# from "ATR happens to equal base").

def test_get_cached_returns_none_when_uncached():
    assert get_cached_atr_stop("NEW") is None


def test_get_cached_returns_fresh_value():
    cache_atr_stop("PANW", 12.4, today=date(2026, 7, 29))
    assert get_cached_atr_stop("PANW", today=date(2026, 7, 31)) == 12.4


def test_get_cached_returns_none_when_stale():
    cache_atr_stop("PANW", 12.4, today=date(2026, 7, 1))
    assert get_cached_atr_stop("PANW", today=date(2026, 7, 31)) is None

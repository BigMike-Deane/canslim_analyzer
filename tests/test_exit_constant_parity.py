"""Source-level constant parity: ai_trader.py ≡ backtester.py exit/stop literals.

The behavioral parity harnesses (test_evaluate_sells_engine_parity.py,
test_backtester_trading_parity.py) drive both engines on matched scenarios and
assert the same DECISION. But a one-sided edit to a shared *constant* only
trips those if some scenario happens to straddle the moved boundary — otherwise
the live trader and the backtest simulator silently diverge, corrupting both
real exits and the backtest-replay eval signal the strategy decisions rest on.

This file closes that gap statically. It parses both modules and asserts the
exit/stop/sell-risk constants resolve to the SAME fallback in each. A profile
that omits a key must fall back to the same number live and in simulation.

Two layers:
  1. YAML-fallback parity — every `X.get(key, <number>)` default for an
     exit-critical key must match across the two files (AST-extracted, so
     reformatting/whitespace can't fool it).
  2. Bare-literal parity — the inline thresholds with no config key
     (winner gate, weak-position cut, take-profit score-decline, pre-earnings
     partial cap) must be present in BOTH sell-evaluation paths.

When you legitimately add a NEW shared exit constant, add its key to
EXIT_CRITICAL_KEYS (layer 1) or a pattern to BARE_LITERALS (layer 2) so the
guard keeps covering it.

NOT covered on purpose: buy-side / position-sizing keys whose fallbacks
legitimately differ by call-site context (e.g. ``max_positions``,
``min_score``, ``min_l_score`` each appear with different conservative vs
champion defaults). Those are not exit-parity-critical; folding them in would
make this a false-positive factory and the guard would get ignored.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1] / "backend"
AI_TRADER = _BACKEND / "ai_trader.py"
BACKTESTER = _BACKEND / "backtester.py"
TRADING_ENGINE = _BACKEND / "trading_engine.py"


def _numeric(node: ast.AST):
    """Return a float for an int/float literal (or its unary-minus), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _numeric(node.operand)
        return None if inner is None else -inner
    return None


def _numeric_get_fallbacks(path: Path) -> dict[str, set[float]]:
    """Map config-key -> set of numeric defaults across all `X.get(key, num)` calls.

    A set (not a single value) because the same key is read at several call
    sites; we compare the full set so a drift at ANY site is caught.
    """
    tree = ast.parse(path.read_text())
    out: dict[str, set[float]] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and len(node.args) == 2):
            key, default = node.args
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                val = _numeric(default)
                if val is not None:
                    out.setdefault(key.value, set()).add(val)
    return out


# Parsed once at import — the source files don't change mid-run.
_AI_FALLBACKS = _numeric_get_fallbacks(AI_TRADER)
_BT_FALLBACKS = _numeric_get_fallbacks(BACKTESTER)
_TE_FALLBACKS = _numeric_get_fallbacks(TRADING_ENGINE)


# Exit / stop / sell-risk constants that MUST resolve identically in the live
# trader and the simulator. All currently match; this list is the contract.
EXIT_CRITICAL_KEYS = [
    # Hard stops & stop sizing
    "max_stop_pct",
    "normal_stop_loss_pct",
    # NOTE: guard_stop_pct / guard_days are NOT here — they were centralized
    # into trading_engine.apply_new_position_guard (one source of truth, so
    # they physically cannot drift). See test_new_position_guard_constants_centralized.
    "stop_loss_days",
    "stop_tighten_factor",
    "atr_multiplier",
    "atr_period",
    # Trailing
    "trailing_stop_days",
    # Profit taking
    "take_profit_pct",
    "partial_sell_pct",
    "partial_min_score",
    "partial_min_pyramid_count",
    "min_gain_for_partial",
    "add_pct",
    # Portfolio-level risk / circuit breaker
    "liquidate_all_pct",
    "halt_new_buys_pct",
    "max_heat_pct",
    "max_single_position_pct",
    # Score-crash exit
    "consecutive_required",
    "recovery_pct",
]


@pytest.mark.parametrize("key", EXIT_CRITICAL_KEYS)
def test_exit_constant_fallback_parity(key):
    """Each exit-critical key falls back to the same number(s) in both engines."""
    assert key in _AI_FALLBACKS, (
        f"'{key}' is no longer a numeric .get() fallback in ai_trader.py — "
        f"if it was renamed/removed, update EXIT_CRITICAL_KEYS to match.")
    assert key in _BT_FALLBACKS, (
        f"'{key}' is no longer a numeric .get() fallback in backtester.py — "
        f"if it was renamed/removed, update EXIT_CRITICAL_KEYS to match.")
    assert _AI_FALLBACKS[key] == _BT_FALLBACKS[key], (
        f"EXIT-CONSTANT DRIFT on '{key}': ai_trader.py falls back to "
        f"{sorted(_AI_FALLBACKS[key])} but backtester.py falls back to "
        f"{sorted(_BT_FALLBACKS[key])}. Live and backtested exits will diverge "
        f"silently. Set BOTH files to the same value.")


# Inline thresholds with no config key — (description, regex). Each must appear
# in BOTH files' sell-evaluation code. Regexes tolerate the variable-name
# differences between the engines (e.g. `score` vs `current_score`).
BARE_LITERALS = {
    "winner gate (gain_pct >= 20)": r"gain_pct >= 20\b",
    "weak-position cut (gain_pct < 10)": r"gain_pct < 10\b",
    "take-profit score-decline (purchase_score - 15)": r"purchase_score - 15\b",
    "pre-earnings partial cap (partial...taken < 25)": r"partial[_a-z]*\s*<\s*25\b",
}


@pytest.mark.parametrize("desc,pattern", list(BARE_LITERALS.items()))
def test_bare_exit_literal_parity(desc, pattern):
    """Bare exit thresholds (no config key) exist in both sell paths."""
    ai_src = AI_TRADER.read_text()
    bt_src = BACKTESTER.read_text()
    assert re.search(pattern, ai_src), (
        f"ai_trader.py is missing the exit literal for {desc} ({pattern!r}). "
        f"If the threshold moved, mirror it in backtester.py and update this guard.")
    assert re.search(pattern, bt_src), (
        f"backtester.py is missing the exit literal for {desc} ({pattern!r}). "
        f"If the threshold moved, mirror it in ai_trader.py and update this guard.")


# Exit constants that were de-duplicated into a single shared helper. Their
# canonical fallback now lives ONLY in trading_engine.py, so cross-file parity
# is replaced by single-source-of-truth. This guard makes sure they stay
# centralized (not silently re-inlined back into the two engines, which is how
# the original drift bug was born).
_CENTRALIZED_GUARD_CONSTANTS = {"guard_days": 21.0, "guard_stop_pct": 8.0}


@pytest.mark.parametrize("key,expected", list(_CENTRALIZED_GUARD_CONSTANTS.items()))
def test_new_position_guard_constants_centralized(key, expected):
    """guard_days / guard_stop_pct live canonically in trading_engine and are
    NOT re-duplicated into ai_trader.py or backtester.py."""
    assert _TE_FALLBACKS.get(key) == {expected}, (
        f"'{key}' should resolve to {expected} in trading_engine."
        f"apply_new_position_guard; got {_TE_FALLBACKS.get(key)}.")
    assert key not in _AI_FALLBACKS, (
        f"'{key}' was re-inlined into ai_trader.py — keep it centralized in "
        f"trading_engine.apply_new_position_guard so the three stop sites can't drift.")
    assert key not in _BT_FALLBACKS, (
        f"'{key}' was re-inlined into backtester.py — keep it centralized in "
        f"trading_engine.apply_new_position_guard so the three stop sites can't drift.")

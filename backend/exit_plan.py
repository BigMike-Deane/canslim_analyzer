"""Read-only exit-plan derivation for AI Portfolio positions.

Surfaces, for a held position, the *price levels and conditions that would
trigger a SELL* — the same triggers ``backend/ai_trader.py::evaluate_sells``
acts on, but computed WITHOUT importing or modifying ``ai_trader.py`` (it is on
the June-18 eval freeze). Instead this reuses the shared threshold helpers the
trader itself calls:

  - ``get_strategy_profile``          (backend/trading_utils.py)
  - ``select_effective_stop_loss_pct``(backend/trading_utils.py)
  - ``get_trailing_stop_pct`` / ``apply_pyramid_widening`` (backend/trading_engine.py)

so the levels shown to the user are produced by the *identical* code path the
live trader runs — they cannot drift from reality the way a re-implemented copy
(e.g. the hard-coded 15/12/10/8 tiers inline in the endpoint) would.

Pure: no DB, no I/O, no globals. Input is plain position scalars + config
values; output is a JSON-friendly dict. Trivially unit-testable; zero risk to
the trading path.

Scope / honest caveats baked into the trigger ``note`` fields:
  - The hard stop is the *base* (normal/bearish) percentage. The live trader may
    *widen* it per-position via ATR for volatile names — that only ever gives a
    position MORE room, so the displayed level is a conservative floor.
  - Take-profit is shown as the price target, but the trader only takes the full
    profit when the score is *also* fading — noted on the trigger.
  - Score-based exits (crash / weak-position) are conditions, not prices.
"""

from __future__ import annotations

from typing import Optional

from backend.trading_engine import get_trailing_stop_pct, apply_pyramid_widening
from backend.trading_utils import get_strategy_profile, select_effective_stop_loss_pct


def _pct(numer: float, denom: float) -> Optional[float]:
    if not denom:
        return None
    return (numer / denom - 1.0) * 100.0


def compute_exit_plan(
    *,
    cost_basis: Optional[float],
    current_price: Optional[float],
    peak_price: Optional[float] = None,
    gain_loss_pct: Optional[float] = None,
    current_score: Optional[float] = None,
    purchase_score: Optional[float] = None,
    pyramid_count: int = 0,
    strategy: str = "balanced",
    sell_score_threshold: Optional[float] = None,
    stop_loss_pct: Optional[float] = None,
    stop_loss_config: Optional[dict] = None,
    is_bearish_market: bool = False,
) -> dict:
    """Derive the active sell triggers for one position.

    Returns a dict::

        {
          "triggers": [ {kind, label, price|None, distance_pct|None,
                         direction, note, ...}, ... ],
          "nearest_kind": str | None,   # price trigger closest to firing
        }

    ``direction`` is ``"down"`` for protective stops (price below current) and
    ``"up"`` for the profit target. ``distance_pct`` is the signed move from the
    current price to the trigger (always reported as a positive magnitude with
    ``direction`` giving the sense): for a stop it is the downside cushion, for
    the target it is the upside still to go.
    """
    profile = get_strategy_profile(strategy)
    triggers: list[dict] = []

    has_price = (
        cost_basis is not None and cost_basis > 0
        and current_price is not None and current_price > 0
    )

    # ── 1. Hard stop loss ────────────────────────────────────────────
    # Mirrors evaluate_sells: effective stop = profile/config normal (or bearish
    # when SPY<50MA). VIX + ATR adjustments are intentionally skipped here — VIX
    # needs live data and ATR only widens, so the base is a safe floor.
    if has_price:
        stop_pct = select_effective_stop_loss_pct(
            profile=profile,
            # YAML ai_trader.stops, passed in by the caller (kept as a param
            # so this module stays pure). The live trader resolves through
            # this same rung (ai_trader.py evaluate_sells/check_stop_losses);
            # omitting it here skipped one level of the resolution chain.
            stop_loss_config=stop_loss_config or {},
            default_normal_pct=stop_loss_pct if stop_loss_pct is not None else 8.0,
            is_bearish_market=is_bearish_market,
            vix_proxy=None,
            vix_config=None,
        )
        stop_price = cost_basis * (1 - stop_pct / 100.0)
        triggers.append({
            "kind": "stop_loss",
            "label": "Stop loss",
            "price": round(stop_price, 2),
            "direction": "down",
            "distance_pct": round(abs(_pct(stop_price, current_price) or 0.0), 1),
            "note": (
                f"{stop_pct:.0f}% below entry"
                + (" · bearish-market stop" if is_bearish_market else "")
                + " · may widen for volatile names"
            ),
        })

    # ── 2. Trailing stop ─────────────────────────────────────────────
    # Only active once the position has run up enough from cost basis to reach a
    # tier (get_trailing_stop_pct returns None below +5% peak gain). Anchored at
    # the peak, not the entry.
    if has_price and peak_price and peak_price > 0:
        peak_gain_pct = (peak_price / cost_basis - 1.0) * 100.0
        trail_pct = get_trailing_stop_pct(peak_gain_pct, profile)
        if trail_pct and pyramid_count:
            trail_pct = apply_pyramid_widening(trail_pct, pyramid_count)
        if trail_pct:
            trail_price = peak_price * (1 - trail_pct / 100.0)
            pyr_note = f" (+{min(pyramid_count*2,6):.0f}% pyramid room)" if pyramid_count else ""
            triggers.append({
                "kind": "trailing_stop",
                "label": "Trailing stop",
                "price": round(trail_price, 2),
                "direction": "down",
                "distance_pct": round(abs(_pct(trail_price, current_price) or 0.0), 1),
                "note": f"{trail_pct:.0f}% off peak ${peak_price:,.2f}{pyr_note}",
            })

    # ── 3. Take-profit target ────────────────────────────────────────
    # Literal 75.0 fallback mirrors ai_trader.py evaluate_sells (and the
    # backtester): the trader deliberately does NOT fall back to the user's
    # AIPortfolioConfig.take_profit_pct column (stale default 25.0), so
    # neither can the displayed plan — that was exactly the drift this
    # module promises not to have.
    tp_pct = profile.get("take_profit_pct", 75.0)
    if has_price and tp_pct:
        tp_price = cost_basis * (1 + tp_pct / 100.0)
        # If already past the target, it does NOT auto-sell — the full-profit
        # rule also requires the score to fade — so the position is held as a
        # runner. Flag that explicitly so the UI doesn't show "X% to go" or
        # treat a passed target as the nearest *approaching* trigger.
        to_go = _pct(tp_price, current_price)
        reached = current_price >= tp_price
        triggers.append({
            "kind": "take_profit",
            "label": "Take profit",
            "price": round(tp_price, 2),
            "direction": "up",
            "distance_pct": round(abs(to_go) if to_go is not None else 0.0, 1),
            "reached": reached,
            "note": (
                f"+{tp_pct:.0f}% target passed · held while score holds up"
                if reached
                else f"+{tp_pct:.0f}% target · fires when score also fades"
            ),
        })

    # ── 4. Score exit (condition, not a price) ───────────────────────
    # The trader only acts on score < threshold when the position is below
    # +10% (WEAK POSITION) or at/above +20% (PROTECT GAINS) — a position
    # sitting in the +10–20% band is HELD regardless of score
    # (ai_trader.evaluate_sells gain gates). Surface that so the card
    # doesn't predict a sale the trader would never make.
    if sell_score_threshold is not None:
        in_hold_band = (
            gain_loss_pct is not None and 10 <= gain_loss_pct < 20
        )
        triggers.append({
            "kind": "score_exit",
            "label": "Score exit",
            "price": None,
            "direction": "score",
            "distance_pct": None,
            "threshold": round(sell_score_threshold, 0),
            "current_score": round(current_score, 0) if current_score is not None else None,
            "active": not in_hold_band,
            "note": (
                (
                    f"paused while +10–20% — holds even below {sell_score_threshold:.0f}"
                    if in_hold_band
                    else f"sells if score falls below {sell_score_threshold:.0f}"
                )
                + (f" (now {current_score:.0f})" if current_score is not None else "")
            ),
        })

    # ── Nearest price-based trigger (closest to firing) ──────────────
    # Exclude an already-reached target: a passed take-profit that's being held
    # for the score is not an *approaching* trigger, so the nearest actionable
    # exit for a runner is its trailing stop, not the target behind it.
    priced = [
        t for t in triggers
        if t.get("price") is not None and t.get("distance_pct") is not None and not t.get("reached")
    ]
    nearest_kind = min(priced, key=lambda t: t["distance_pct"])["kind"] if priced else None

    return {"triggers": triggers, "nearest_kind": nearest_kind}

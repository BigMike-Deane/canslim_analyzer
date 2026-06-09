"""Stop-guard breach monitor — freeze-safe alerting for a known live bug.

Background (2026-06-09): the automated trading loop's sell path,
``backend/ai_trader.py::evaluate_sells`` (~line 1623), is MISSING the
``new_position_guard`` clamp that BOTH the manual stop-loss path
(``_check_and_execute_stop_losses_impl``, ai_trader.py:1304-1314) AND the
backtester (``backtester.py::_evaluate_sells``) apply. For a fast-falling
new buy, the ATR stop widens pro-cyclically toward the 20% cap and the
8% new-position clamp never kicks in, so the live loop holds the position
to ~-20% instead of ~-8% (observed: FSLR -18% with a 20% stop, AMD -13.7%
with a 19.2% stop).

``ai_trader.py`` is frozen until the June-18 C-score-cap eval completes, so
the fix (a ~6-line guard port into ``evaluate_sells``) cannot ship yet.
This module is the freeze-safe mitigation: it does NOT sell anything — it
watches every AI Portfolio position that the guard *would* have stopped out
and raises an urgent in-app + push notification so the owner can decide to
exit manually (e.g. via the guarded manual stop-loss endpoint).

Mirrors the guard semantics of ai_trader.py:1304-1314 exactly (window
``holding_days <= guard_days``, ``skip_if_pyramided`` on ``pyramid_count >
0``) without modifying that file. Imports from ``backend.ai_trader`` are
read-only and already an established scheduler pattern (scheduler.py:445,
720, 1575, 1703).

DELETE this module once the guard port ships post-freeze — it exists only
to cover the gap.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

NOTIFICATION_KIND = "stop_guard_breach"

# One alert per position per trading day. 20h (not 24h) so a job tick that
# lands slightly earlier than yesterday's doesn't suppress today's alert.
DEDUP_WINDOW_HOURS = 20


def guard_would_fire(
    *,
    gain_loss_pct: Optional[float],
    purchase_date,
    pyramid_count: int,
    today: date,
    guard_enabled: bool,
    guard_days: int,
    guard_stop_pct: float,
    skip_if_pyramided: bool,
) -> bool:
    """True iff the new_position_guard clamp would stop this position out.

    Pure mirror of the guard block in ai_trader.py:1304-1314 plus the stop
    trigger ``gain_pct <= -position_stop_pct`` with the clamped stop. Because
    the clamp is ``min(atr_stop, guard_stop_pct)``, a loss at or past
    ``guard_stop_pct`` is sufficient to know the guarded stop would fire,
    regardless of what the ATR stop is.
    """
    if not guard_enabled or purchase_date is None:
        return False

    purchase_dt = purchase_date.date() if hasattr(purchase_date, "date") else purchase_date
    holding_days = (today - purchase_dt).days
    if holding_days > guard_days:
        return False

    pyramid_count = pyramid_count or 0
    if skip_if_pyramided and pyramid_count > 0:
        return False

    gain_pct = gain_loss_pct or 0
    return gain_pct <= -guard_stop_pct


def _estimate_live_stop_pct(ticker: str, current_price: float, user_config) -> Optional[float]:
    """Best-effort estimate of the stop the UNGUARDED live loop is using.

    Same pipeline as evaluate_sells: profile/bearish-aware base stop, then
    ATR widening (read-only import from ai_trader). Skips the VIX adjustment
    (vix_proxy=None) — VIX only nudges the base, and ATR widening dominates
    for exactly the volatile names this bug bites. Returns None on any
    failure; the alert still fires, just without the live-stop figure.
    """
    try:
        from config_loader import config as yaml_config
        from backend.trading_utils import get_strategy_profile, select_effective_stop_loss_pct
        from backend.ai_trader import calculate_atr_stop
        from data_fetcher import get_cached_market_direction

        is_bearish_market = False
        market_data = get_cached_market_direction() or {}
        if market_data.get("success"):
            spy = market_data.get("indexes", {}).get("SPY", {})
            spy_price, spy_ma_50 = spy.get("price", 0), spy.get("ma_50", 0)
            is_bearish_market = spy_price < spy_ma_50 if spy_price > 0 and spy_ma_50 > 0 else False

        profile = get_strategy_profile(getattr(user_config, "strategy", "balanced") or "balanced")
        base_stop = select_effective_stop_loss_pct(
            profile=profile,
            stop_loss_config=yaml_config.get("ai_trader.stops", {}),
            default_normal_pct=getattr(user_config, "stop_loss_pct", 15.0) or 15.0,
            is_bearish_market=is_bearish_market,
            vix_proxy=None,
            vix_config=yaml_config.get("vix_stops", {}),
        )
        return calculate_atr_stop(ticker, current_price, base_stop)
    except Exception as e:
        logger.debug(f"Live-stop estimate failed for {ticker}: {e}")
        return None


def _already_alerted(db, user_id: int, ticker: str, now: datetime) -> bool:
    """True if this user already got a breach alert for this ticker recently.

    Notification.data is JSON, so filter the (few) recent rows of this kind
    in Python rather than relying on dialect-specific JSON operators.
    """
    from backend.database import Notification

    cutoff = now - timedelta(hours=DEDUP_WINDOW_HOURS)
    recent = (
        db.query(Notification)
        .filter(
            Notification.user_id == user_id,
            Notification.kind == NOTIFICATION_KIND,
            Notification.created_at >= cutoff,
        )
        .all()
    )
    return any((n.data or {}).get("ticker") == ticker for n in recent)


def check_stop_guard_breaches(db=None) -> list:
    """Scan all users' AI Portfolio positions and alert on guard breaches.

    Returns the list of breach dicts (also used by tests). Notifications are
    in-app + web push via create_notification; no email, no trades.
    """
    from config_loader import config as yaml_config

    guard_config = yaml_config.get("ai_trader.new_position_guard", {})
    if not guard_config.get("enabled", False):
        return []

    guard_days = guard_config.get("guard_days", 21)
    guard_stop_pct = guard_config.get("guard_stop_pct", 8.0)
    skip_if_pyramided = guard_config.get("skip_if_pyramided", True)

    from backend.database import SessionLocal, AIPortfolioConfig, AIPortfolioPosition
    from backend.email_utils import create_notification

    owns_session = db is None
    if owns_session:
        db = SessionLocal()

    breaches = []
    try:
        positions = db.query(AIPortfolioPosition).all()
        today = date.today()
        now = datetime.now(timezone.utc)
        config_by_user = {}

        for position in positions:
            price = position.current_price
            if not price or (isinstance(price, float) and math.isnan(price)):
                continue  # mirror live loop's None/NaN skip

            if not guard_would_fire(
                gain_loss_pct=position.gain_loss_pct,
                purchase_date=position.purchase_date,
                pyramid_count=position.pyramid_count,
                today=today,
                guard_enabled=True,
                guard_days=guard_days,
                guard_stop_pct=guard_stop_pct,
                skip_if_pyramided=skip_if_pyramided,
            ):
                continue

            user_id = position.user_id or 1  # legacy rows belong to the owner
            ticker = position.ticker
            if _already_alerted(db, user_id, ticker, now):
                continue

            if position.user_id not in config_by_user:
                config_by_user[position.user_id] = (
                    db.query(AIPortfolioConfig)
                    .filter(AIPortfolioConfig.user_id == position.user_id)
                    .first()
                )
            live_stop_pct = _estimate_live_stop_pct(ticker, price, config_by_user[position.user_id])

            gain_pct = position.gain_loss_pct or 0
            purchase_dt = (
                position.purchase_date.date()
                if hasattr(position.purchase_date, "date")
                else position.purchase_date
            )
            holding_days = (today - purchase_dt).days

            if live_stop_pct is not None:
                stop_note = (
                    f"Its widened ATR stop won't fire until about -{live_stop_pct:.1f}%."
                )
            else:
                stop_note = "Its ATR stop has likely widened well past the guard."

            breach = {
                "ticker": ticker,
                "user_id": user_id,
                "gain_loss_pct": round(gain_pct, 1),
                "guard_stop_pct": guard_stop_pct,
                "live_stop_pct": round(live_stop_pct, 1) if live_stop_pct is not None else None,
                "holding_days": holding_days,
            }
            breaches.append(breach)

            create_notification(
                user_id=user_id,
                kind=NOTIFICATION_KIND,
                title=f"{ticker} down {abs(gain_pct):.1f}% — past the {guard_stop_pct:.0f}% new-buy guard",
                body=(
                    f"{ticker} was bought {holding_days} days ago and is down "
                    f"{abs(gain_pct):.1f}%. The {guard_stop_pct:.0f}% new-position guard "
                    f"would have sold it, but the automated loop is missing that clamp "
                    f"(known bug, fix frozen until June 18). {stop_note} "
                    f"Review the position for a manual exit."
                ),
                priority="urgent",
                tags=["rotating_light"],
                data=breach,
            )
            logger.warning(
                f"Stop-guard breach: {ticker} (user {user_id}) at {gain_pct:.1f}%, "
                f"guard {guard_stop_pct:.0f}%, live stop ~{live_stop_pct}"
            )
    finally:
        if owns_session:
            db.close()

    return breaches

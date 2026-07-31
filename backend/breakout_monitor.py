"""
Intraday Breakout Monitor

Lightweight 5-min check for stocks crossing breakout pivots.
Only runs during market hours. Pushes ntfy alerts on confirmed breakouts.

Filters match AI trader criteria so every alert is actionable:
- Score >= 72, C >= 10, L >= 8 (same as nostate_optimized)
- Only BREAKOUT and BREAKOUT+VOLUME alerts (no "near pivot" noise)
- DB-backed 24h per-ticker cooldown (one alert per stock per day max)
- DB-backed daily cap (default 10 alerts/day) to prevent broad-market spam

The cooldown + cap live in `BreakoutAlert` rows so they survive container
restarts. Pre-fix (May 2026) they lived in a Python module-level dict that
reset on every redeploy, causing repeat alerts for the same ticker after
each deploy.
"""

import logging
from datetime import datetime, timezone, timedelta, date

logger = logging.getLogger(__name__)

ALERT_COOLDOWN_HOURS = 24

# Match AI trader quality filters
MIN_SCORE = 72
MIN_C_SCORE = 10
MIN_L_SCORE = 8

# Default daily cap if config not set. Breakouts are more common than CS
# catalysts, so 10/day is a reasonable noise floor (was effectively unlimited
# pre-fix when the in-memory cooldown got wiped on each restart).
DEFAULT_DAILY_CAP = 10


def _get_daily_cap() -> int:
    """Read configured daily cap, falling back to DEFAULT_DAILY_CAP."""
    try:
        from config_loader import config
        cap = config.get('breakout_monitor.daily_cap', default=DEFAULT_DAILY_CAP)
        # Defensive: guard against None / non-int from a malformed YAML
        if cap is None:
            return DEFAULT_DAILY_CAP
        return int(cap)
    except Exception:
        return DEFAULT_DAILY_CAP


def _was_alerted_recently(db, ticker: str, now: datetime) -> bool:
    """Check whether ticker has a BreakoutAlert row within the cooldown window."""
    from backend.database import BreakoutAlert
    cutoff = now - timedelta(hours=ALERT_COOLDOWN_HOURS)
    existing = db.query(BreakoutAlert).filter(
        BreakoutAlert.ticker == ticker,
        BreakoutAlert.created_at >= cutoff,
    ).first()
    return existing is not None


def _todays_alert_count(db, today: date) -> int:
    """Count BreakoutAlert rows fired today (by alert_date)."""
    from backend.database import BreakoutAlert
    return db.query(BreakoutAlert).filter(
        BreakoutAlert.alert_date == today,
    ).count()


def _record_alert(db, ticker: str, label: str, pivot_price, current_price,
                  vol_ratio, now: datetime) -> None:
    """Insert a BreakoutAlert row. Caller is responsible for the surrounding commit."""
    from backend.database import BreakoutAlert
    alert = BreakoutAlert(
        ticker=ticker,
        alert_date=now.date(),
        created_at=now,
        pivot_price=pivot_price,
        current_price=current_price,
        vol_ratio=vol_ratio,
        label=label,
    )
    db.add(alert)


def check_intraday_breakouts():
    """Check top candidates near pivot for intraday breakouts.
    Designed to run every 5 min during market hours."""
    from backend.ai_trader import is_market_open
    from backend.database import SessionLocal, Stock
    from backend.email_utils import send_webhook_notification, broadcast_notification

    if not is_market_open():
        return

    daily_cap = _get_daily_cap()

    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        today = now.date()

        # Daily cap pre-check — short-circuit if already at the limit.
        if _todays_alert_count(db, today) >= daily_cap:
            logger.debug(f"Breakout monitor: daily cap ({daily_cap}) reached, skipping cycle")
            return

        # Find stocks with pivot prices that meet trading quality thresholds
        candidates = db.query(Stock).filter(
            Stock.pivot_price != None,
            Stock.pivot_price > 0,
            Stock.canslim_score != None,
            Stock.canslim_score >= MIN_SCORE,
            Stock.current_price != None,
            Stock.current_price > 0,
        ).all()

        # Apply quality filters (C and L scores) to match AI trader criteria
        qualified = []
        for s in candidates:
            c_score = getattr(s, 'c_score', None)
            l_score = getattr(s, 'l_score', None)
            if (c_score is not None and c_score >= MIN_C_SCORE and
                    l_score is not None and l_score >= MIN_L_SCORE):
                qualified.append(s)

        # Filter to stocks within 5% of pivot (pre-breakout zone)
        near_pivot = []
        for s in qualified:
            pct = ((s.pivot_price - s.current_price) / s.pivot_price) * 100
            if -3 <= pct <= 5:  # Between 3% above pivot and 5% below
                near_pivot.append((s, pct))

        if not near_pivot:
            return

        # Fetch fresh quotes for candidates near pivot
        tickers = [s.ticker for s, _ in near_pivot]
        fresh_prices = _fetch_quick_quotes(tickers)

        if not fresh_prices:
            return

        alerts_sent = 0

        for stock, old_pct in near_pivot:
            price = fresh_prices.get(stock.ticker)
            if not price or price <= 0:
                continue

            pct_from_pivot = ((stock.pivot_price - price) / stock.pivot_price) * 100

            # Per-ticker 24h cooldown (DB-backed — survives restart)
            if _was_alerted_recently(db, stock.ticker, now):
                continue

            # Re-check daily cap inline (an earlier loop iteration may have
            # bumped today's total to the cap).
            if _todays_alert_count(db, today) >= daily_cap:
                logger.info(f"Breakout monitor: hit daily cap ({daily_cap}) mid-cycle, stopping")
                break

            # Alert only on confirmed breakouts (no "near pivot" noise)
            alert = None
            vol_ratio = getattr(stock, 'volume_ratio', None)
            if pct_from_pivot <= 0 and old_pct > 0:
                # Just crossed above pivot — BREAKOUT
                if vol_ratio and vol_ratio > 1.5:
                    alert = ("BREAKOUT + VOLUME", "high", ["fire", "chart_with_upwards_trend"])
                else:
                    alert = ("BREAKOUT", "high", ["rotating_light", "chart_with_upwards_trend"])
            elif pct_from_pivot <= 0 and pct_from_pivot > -3:
                # Already above pivot — only alert if volume confirms
                if vol_ratio and vol_ratio > 1.5:
                    alert = ("BREAKOUT + VOLUME", "high", ["fire", "chart_with_upwards_trend"])

            if alert:
                label, priority, tags = alert
                base_info = f"{stock.base_type or 'base'} {stock.weeks_in_base or '?'}w" if stock.base_type else ""
                title = f"{label}: {stock.ticker} ${price:.2f}"
                message = (
                    f"Score: {stock.canslim_score:.0f} | Pivot: ${stock.pivot_price:.2f} "
                    f"({pct_from_pivot:+.1f}%)\n"
                    f"{base_info} | {stock.sector or ''}"
                )
                # COMMIT the cooldown row BEFORE any push leaves the building:
                # pushes fired first, and a failed end-of-cycle commit rolled
                # the cooldown rows back → the same ticker re-alerted every
                # 5-min tick until a commit succeeded (the exact spam loop the
                # DB-backed cooldown was built to prevent). A committed row
                # with a failed push loses one alert once — the right trade.
                _record_alert(db, stock.ticker, label, stock.pivot_price,
                              price, vol_ratio, now)
                db.commit()
                # In-app: broadcast to every active user (system event, not portfolio-scoped).
                broadcast_notification(
                    kind="breakout", title=title, body=message,
                    priority=priority, tags=tags,
                    coalesce_daily=True, digest_label="Breakouts",
                    digest_url="/breakouts",
                    data={"ticker": stock.ticker, "price": price,
                          "pivot_price": stock.pivot_price,
                          "pct_from_pivot": pct_from_pivot,
                          "score": stock.canslim_score,
                          "sector": stock.sector,
                          "base_type": stock.base_type,
                          "weeks_in_base": stock.weeks_in_base,
                          "label": label},
                )
                # ntfy push (legacy global URL — phone alerts).
                send_webhook_notification(title=title, message=message, kind="breakout",
                                          priority=priority, tags=tags)
                alerts_sent += 1

            # Update the stock's current price in DB while we have fresh data
            stock.current_price = price

        if alerts_sent > 0:
            db.commit()
            logger.info(f"Breakout monitor: {alerts_sent} alerts sent from {len(near_pivot)} candidates")
        else:
            # Still commit any current_price updates that happened during the loop.
            db.commit()

    except Exception as e:
        logger.error(f"Breakout monitor error: {e}")
        db.rollback()
    finally:
        db.close()


def _fetch_quick_quotes(tickers: list) -> dict:
    """Fetch current prices for a list of tickers via yfinance (fast batch)."""
    try:
        import pandas as pd
        import yfinance as yf
        data = yf.download(tickers, period="1d", interval="1m", progress=False, group_by='ticker')
        prices = {}
        if data is None or data.empty:
            return prices
        # yfinance >= ~0.2.51 returns MultiIndex (TICKER, field) columns even
        # for a single ticker (multi_level_index default flipped). The old
        # len(tickers)==1 special case did data['Close'] → KeyError, swallowed
        # below → {} → the monitor was a silent no-op whenever exactly one
        # stock sat near pivot. Branch on the actual column shape instead.
        is_multi = isinstance(data.columns, pd.MultiIndex)
        for ticker in tickers:
            try:
                series = data[ticker]['Close'] if is_multi else data['Close']
                col = series.dropna()
                if not col.empty:
                    prices[ticker] = float(col.iloc[-1])
            except (KeyError, IndexError):
                pass
        return prices
    except Exception as e:
        logger.error(f"Quick quote fetch failed: {e}")
        return {}


def get_breakout_monitor_status() -> dict:
    """Return monitor state for health dashboard."""
    from backend.database import SessionLocal, BreakoutAlert
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=ALERT_COOLDOWN_HOURS)
        active = db.query(BreakoutAlert).filter(
            BreakoutAlert.created_at >= cutoff,
        ).all()
        active_tickers = sorted({a.ticker for a in active})
        today_count = _todays_alert_count(db, now.date())
        return {
            "active_cooldowns": len(active_tickers),
            "cooldown_tickers": active_tickers,
            "cooldown_hours": ALERT_COOLDOWN_HOURS,
            "alerts_today": today_count,
            "daily_cap": _get_daily_cap(),
            "min_score": MIN_SCORE,
            "min_c_score": MIN_C_SCORE,
            "min_l_score": MIN_L_SCORE,
        }
    finally:
        db.close()

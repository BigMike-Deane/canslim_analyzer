"""
Intraday Breakout Monitor

Lightweight 5-min check for stocks crossing breakout pivots.
Only runs during market hours. Pushes ntfy alerts on confirmed breakouts.

Filters match AI trader criteria so every alert is actionable:
- Score >= 72, C >= 10, L >= 8 (same as nostate_optimized)
- Only BREAKOUT and BREAKOUT+VOLUME alerts (no "near pivot" noise)
- 24h per-ticker cooldown (one alert per stock per day max)
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Track recent alerts to avoid spam (ticker -> last alert timestamp)
_recent_alerts = {}
ALERT_COOLDOWN_HOURS = 24

# Match AI trader quality filters
MIN_SCORE = 72
MIN_C_SCORE = 10
MIN_L_SCORE = 8


def check_intraday_breakouts():
    """Check top candidates near pivot for intraday breakouts.
    Designed to run every 5 min during market hours."""
    from backend.ai_trader import is_market_open
    from backend.database import SessionLocal, Stock
    from backend.email_utils import send_webhook_notification, broadcast_notification

    if not is_market_open():
        return

    db = SessionLocal()
    try:
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

        now = datetime.now(timezone.utc)
        alerts_sent = 0

        for stock, old_pct in near_pivot:
            price = fresh_prices.get(stock.ticker)
            if not price or price <= 0:
                continue

            pct_from_pivot = ((stock.pivot_price - price) / stock.pivot_price) * 100

            # Check cooldown (24h — one alert per stock per day)
            last_alert = _recent_alerts.get(stock.ticker)
            if last_alert and (now - last_alert) < timedelta(hours=ALERT_COOLDOWN_HOURS):
                continue

            # Alert only on confirmed breakouts (no "near pivot" noise)
            alert = None
            if pct_from_pivot <= 0 and old_pct > 0:
                # Just crossed above pivot — BREAKOUT
                vol_ratio = getattr(stock, 'volume_ratio', None)
                if vol_ratio and vol_ratio > 1.5:
                    alert = ("BREAKOUT + VOLUME", "high", ["fire", "chart_with_upwards_trend"])
                else:
                    alert = ("BREAKOUT", "high", ["rotating_light", "chart_with_upwards_trend"])
            elif pct_from_pivot <= 0 and pct_from_pivot > -3:
                # Already above pivot — only alert if volume confirms
                vol_ratio = getattr(stock, 'volume_ratio', None)
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
                # In-app: broadcast to every active user (system event, not portfolio-scoped).
                broadcast_notification(
                    kind="breakout", title=title, body=message,
                    priority=priority, tags=tags,
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
                send_webhook_notification(title=title, message=message,
                                          priority=priority, tags=tags)
                _recent_alerts[stock.ticker] = now
                alerts_sent += 1

            # Update the stock's current price in DB while we have fresh data
            stock.current_price = price

        if alerts_sent > 0:
            db.commit()
            logger.info(f"Breakout monitor: {alerts_sent} alerts sent from {len(near_pivot)} candidates")

        # Cleanup old cooldowns
        cutoff = now - timedelta(hours=ALERT_COOLDOWN_HOURS * 2)
        expired = [t for t, ts in _recent_alerts.items() if ts < cutoff]
        for t in expired:
            del _recent_alerts[t]

    except Exception as e:
        logger.error(f"Breakout monitor error: {e}")
    finally:
        db.close()


def _fetch_quick_quotes(tickers: list) -> dict:
    """Fetch current prices for a list of tickers via yfinance (fast batch)."""
    try:
        import yfinance as yf
        data = yf.download(tickers, period="1d", interval="1m", progress=False, group_by='ticker')
        prices = {}
        if len(tickers) == 1:
            if not data.empty:
                prices[tickers[0]] = float(data['Close'].dropna().iloc[-1])
        else:
            for ticker in tickers:
                try:
                    col = data[ticker]['Close'].dropna()
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
    return {
        "active_cooldowns": len(_recent_alerts),
        "cooldown_tickers": list(_recent_alerts.keys()),
        "cooldown_hours": ALERT_COOLDOWN_HOURS,
        "min_score": MIN_SCORE,
        "min_c_score": MIN_C_SCORE,
        "min_l_score": MIN_L_SCORE,
    }

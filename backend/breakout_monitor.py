"""
Intraday Breakout Monitor

Lightweight 5-min check for stocks approaching or crossing breakout pivots.
Only runs during market hours. Pushes ntfy alerts on breakouts.
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Track recent alerts to avoid spam (ticker -> last alert timestamp)
_recent_alerts = {}
ALERT_COOLDOWN_HOURS = 4


def check_intraday_breakouts():
    """Check top candidates near pivot for intraday breakouts.
    Designed to run every 5 min during market hours."""
    from backend.ai_trader import is_market_open
    from backend.database import SessionLocal, Stock
    from email_utils import send_webhook_notification

    if not is_market_open():
        return

    db = SessionLocal()
    try:
        # Find stocks with pivot prices, good scores, not already breaking out
        candidates = db.query(Stock).filter(
            Stock.pivot_price != None,
            Stock.pivot_price > 0,
            Stock.canslim_score != None,
            Stock.canslim_score >= 65,
            Stock.current_price != None,
            Stock.current_price > 0,
        ).all()

        # Filter to stocks within 5% of pivot (pre-breakout zone)
        near_pivot = []
        for s in candidates:
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

            # Check cooldown
            last_alert = _recent_alerts.get(stock.ticker)
            if last_alert and (now - last_alert) < timedelta(hours=ALERT_COOLDOWN_HOURS):
                continue

            # Alert conditions
            alert = None
            if pct_from_pivot <= 0 and old_pct > 0:
                # Just crossed above pivot — BREAKOUT
                alert = ("BREAKOUT", "high", ["rotating_light", "chart_with_upwards_trend"])
            elif 0 < pct_from_pivot <= 1.5:
                # Within 1.5% of pivot — approaching
                alert = ("NEAR PIVOT", "default", ["eyes", "chart_with_upwards_trend"])
            elif pct_from_pivot <= 0 and pct_from_pivot > -3:
                # Check volume confirmation on breakout
                vol_ratio = getattr(stock, 'volume_ratio', None)
                if vol_ratio and vol_ratio > 1.5:
                    alert = ("BREAKOUT + VOLUME", "high", ["fire", "chart_with_upwards_trend"])

            if alert:
                label, priority, tags = alert
                base_info = f"{stock.base_type or 'base'} {stock.weeks_in_base or '?'}w" if stock.base_type else ""
                send_webhook_notification(
                    title=f"{label}: {stock.ticker} ${price:.2f}",
                    message=(
                        f"Score: {stock.canslim_score:.0f} | Pivot: ${stock.pivot_price:.2f} "
                        f"({pct_from_pivot:+.1f}%)\n"
                        f"{base_info} | {stock.sector or ''}"
                    ),
                    priority=priority,
                    tags=tags,
                )
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
    }

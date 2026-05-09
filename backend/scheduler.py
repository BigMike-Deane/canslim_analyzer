"""
CANSLIM Continuous Scanning Scheduler

Runs automatic scans at configurable intervals to keep stock data fresh.
Stays within FMP API rate limits (300 calls/min).
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from collections import deque
from datetime import datetime, timezone
from sqlalchemy.orm import Session
import logging
import os
import random
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scheduler instance
scheduler = BackgroundScheduler()

# Friendly labels for the per-phase progress UI. Map phase keys (used in
# scheduler internals + by the async scanner progress callback) to short
# user-facing strings rendered in the CommandCenter / SystemHealth scanner card.
_PHASE_LABELS = {
    "scanning": "Fetching stock data",
    "stocks": "Fetching stock data",
    "insider_short": "Fetching insider/short data",
    "p1_data": "Fetching P1 data",
    "saving": "Saving to database",
    "coiled_spring": "Checking Coiled Spring",
    "earnings_audit": "Earnings audit",
    "industry_groups": "Industry group rankings",
    "bear_base": "Bear base scan",
    "ai_trading": "AI trading cycle",
    "cleanup": "Cleanup",
}

# State tracking
# - stocks_scanned / total_stocks remain the Phase-1 stock-fetch counters
#   (existing consumers depend on this contract; do not overload).
# - current_phase / phase_current / phase_total / phase_label give the
#   broader per-phase progress that keeps updating after Phase 1 ends.
_scan_config = {
    "enabled": False,
    "source": "sp500",  # sp500, top50, russell, all
    "interval_minutes": 15,
    "last_scan_start": None,
    "last_scan_end": None,
    "stocks_scanned": 0,
    "total_stocks": 0,
    "is_scanning": False,
    "phase": None,  # Current phase: "scanning", "p1_data", "saving", "ai_trading", None
    "phase_detail": None,  # Additional detail like "Fetching earnings calendar..."
    "current_phase": None,  # Active phase key (mirrors `phase`, kept for explicit UI use)
    "phase_current": 0,  # Items processed in current phase (e.g. 300)
    "phase_total": 0,  # Total items in current phase (e.g. 1426)
    "phase_label": None,  # Friendly label like "Fetching P1 data"
}

# System health tracking for error alerting
_system_health = {
    "last_successful_scan": None,
    "last_scan_error": None,
    "consecutive_scan_failures": 0,
    "last_successful_trade_cycle": None,
    "last_trade_cycle_error": None,
    "consecutive_trade_failures": 0,
    "last_backup": None,
    "last_backup_error": None,
    "errors_today": deque(maxlen=50),  # Rolling list of recent errors (bounded)
}

# Lock for thread-safe access to global state dicts
_state_lock = threading.Lock()

_HEALTH_REDIS_KEY = "canslim:system_health"


def _persist_health_to_redis():
    """Save system health to Redis for persistence across restarts."""
    try:
        from redis_cache import get_redis_client
        import json
        client = get_redis_client()
        if client:
            client.set(_HEALTH_REDIS_KEY, json.dumps(_system_health))
    except Exception:
        pass  # Redis persistence is best-effort


def _restore_health_from_redis():
    """Restore system health from Redis on startup."""
    try:
        from redis_cache import get_redis_client
        import json
        client = get_redis_client()
        if client:
            data = client.get(_HEALTH_REDIS_KEY)
            if data:
                saved = json.loads(data)
                _system_health.update(saved)
                logger.info(f"Restored system health from Redis (last scan: {saved.get('last_successful_scan', 'N/A')})")
    except Exception as e:
        logger.debug(f"Could not restore health from Redis: {e}")


def get_system_health():
    """Return system health state for dashboard."""
    with _state_lock:
        return dict(_system_health)


def _record_success(task_name: str):
    """Record a successful task execution."""
    now = datetime.now(timezone.utc).isoformat()
    with _state_lock:
        if task_name == "scan":
            _system_health["last_successful_scan"] = now
            _system_health["consecutive_scan_failures"] = 0
            _system_health["last_scan_error"] = None
        elif task_name == "trade_cycle":
            _system_health["last_successful_trade_cycle"] = now
            _system_health["consecutive_trade_failures"] = 0
            _system_health["last_trade_cycle_error"] = None
        elif task_name == "backup":
            _system_health["last_backup"] = now
            _system_health["last_backup_error"] = None
    _persist_health_to_redis()


def _record_failure(task_name: str, error: str):
    """Record a task failure and alert if consecutive failures exceed threshold."""
    from backend.email_utils import send_webhook_notification as _send_alert

    now = datetime.now(timezone.utc).isoformat()
    error_entry = {"task": task_name, "error": error[:200], "timestamp": now}

    with _state_lock:
        _system_health["errors_today"].append(error_entry)

        if task_name == "scan":
            _system_health["last_scan_error"] = error_entry
            _system_health["consecutive_scan_failures"] += 1
            failures = _system_health["consecutive_scan_failures"]
        elif task_name == "trade_cycle":
            _system_health["last_trade_cycle_error"] = error_entry
            _system_health["consecutive_trade_failures"] += 1
            failures = _system_health["consecutive_trade_failures"]
        else:
            failures = 1

    _persist_health_to_redis()

    # Alert on first failure and every 3rd consecutive failure
    if failures == 1 or failures % 3 == 0:
        _send_alert(
            title=f"CANSLIM {task_name.replace('_', ' ').title()} Failed",
            message=f"Error ({failures}x consecutive): {error[:150]}",
            priority="high" if failures >= 3 else "default",
            tags=["warning", "chart_with_downwards_trend"],
        )


def cleanup_old_stock_scores(db: Session, days_to_keep: int = 30):
    """Delete StockScore records older than N days to prevent database bloat."""
    from datetime import timedelta
    from sqlalchemy import text
    from backend.database import StockScore

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

    # Count before delete
    count_before = db.query(StockScore).filter(StockScore.timestamp < cutoff_date).count()

    if count_before == 0:
        logger.info("StockScore cleanup: No old records to delete")
        return 0

    logger.info(f"StockScore cleanup: Deleting {count_before} records older than {days_to_keep} days")

    # Delete in batches to avoid memory issues
    batch_size = 5000
    total_deleted = 0
    while True:
        # Use subquery to get IDs first, then delete by ID (more compatible)
        subquery = db.query(StockScore.id).filter(
            StockScore.timestamp < cutoff_date
        ).limit(batch_size).subquery().select()

        deleted = db.query(StockScore).filter(
            StockScore.id.in_(subquery)
        ).delete(synchronize_session='fetch')

        db.commit()
        total_deleted += deleted
        if deleted < batch_size:
            break

    # VACUUM to reclaim disk space (SQLite only)
    from backend.database import DATABASE_URL
    if DATABASE_URL.startswith("sqlite"):
        try:
            db.execute(text("VACUUM"))
            db.commit()
        except Exception as e:
            logger.debug(f"VACUUM skipped: {e}")

    logger.info(f"StockScore cleanup complete: {total_deleted} records deleted")
    return total_deleted


_last_price_cache_vacuum: datetime | None = None


def cleanup_price_cache(vacuum_interval_days: int = 7):
    """Delete expired entries from the SQLite price_cache; VACUUM at most once per N days.

    The price cache grew to 11GB in production because cleanup_expired() was never called.
    Running this every scan keeps the entry count bounded; VACUUM is throttled because it
    rewrites the whole DB and holds an exclusive lock.
    """
    global _last_price_cache_vacuum
    try:
        from backend.price_cache import get_price_cache
        from datetime import timedelta
        import sqlite3

        cache = get_price_cache()
        deleted = cache.cleanup_expired()
        if deleted:
            logger.info(f"price_cache cleanup: deleted {deleted} expired entries")

        now = datetime.now(timezone.utc)
        if _last_price_cache_vacuum is None or (now - _last_price_cache_vacuum) > timedelta(days=vacuum_interval_days):
            with sqlite3.connect(cache.db_path) as conn:
                conn.execute("VACUUM")
            _last_price_cache_vacuum = now
            logger.info("price_cache VACUUM complete")
    except Exception as e:
        logger.error(f"price_cache cleanup failed: {e}")


def check_watchlist_alerts():
    """Check watchlist items against target_price and alert_score thresholds"""
    from backend.database import SessionLocal, Watchlist, Stock
    from config_loader import config
    from datetime import timedelta

    # Check if alerts are enabled
    watchlist_config = config.get('watchlist', {}).get('alerts', {})
    if not watchlist_config.get('enabled', True):
        logger.debug("Watchlist alerts disabled in config")
        return

    cooldown_hours = watchlist_config.get('cooldown_hours', 24)

    db = SessionLocal()
    alerts_sent = 0

    try:
        # Get watchlist items with alerts set (either target_price or alert_score)
        items = db.query(Watchlist).filter(
            (Watchlist.target_price != None) | (Watchlist.alert_score != None)
        ).all()

        if not items:
            logger.debug("No watchlist items with alerts configured")
            return

        logger.info(f"Checking {len(items)} watchlist items for alerts...")

        # Batch-fetch all stocks for watchlist items (avoids N+1 queries)
        item_tickers = [item.ticker for item in items]
        stocks = db.query(Stock).filter(Stock.ticker.in_(item_tickers)).all()
        stocks_by_ticker = {s.ticker: s for s in stocks}

        for item in items:
            stock = stocks_by_ticker.get(item.ticker)
            if not stock or not stock.current_price:
                continue

            triggered = False
            reasons = []

            # Check price alert (crosses above target)
            if item.target_price and stock.current_price:
                if stock.current_price >= item.target_price:
                    # Only trigger if we didn't already know it was above target
                    if not item.last_check_price or item.last_check_price < item.target_price:
                        triggered = True
                        reasons.append(f"Price ${stock.current_price:.2f} >= target ${item.target_price:.2f}")

            # Check score alert (crosses above threshold)
            if item.alert_score and stock.canslim_score:
                if stock.canslim_score >= item.alert_score:
                    triggered = True
                    reasons.append(f"CANSLIM score {stock.canslim_score:.0f} >= target {item.alert_score:.0f}")

            # Update last check price
            item.last_check_price = stock.current_price

            if triggered:
                # Check cooldown - don't re-alert if recently sent
                if item.alert_triggered_at:
                    alert_at = item.alert_triggered_at
                    if alert_at.tzinfo is None:
                        alert_at = alert_at.replace(tzinfo=timezone.utc)
                    time_since_last = datetime.now(timezone.utc) - alert_at
                    if time_since_last < timedelta(hours=cooldown_hours):
                        logger.debug(f"Skipping {item.ticker} alert - within {cooldown_hours}h cooldown")
                        continue

                # Send the alert
                try:
                    from backend.email_utils import send_watchlist_alert_email

                    # Record trigger time before sending to prevent retry storms
                    item.alert_triggered_at = datetime.now(timezone.utc)
                    item.alert_sent = True

                    # Gate on email.automated_enabled (May 7 2026 — user opted out
                    # of routine emails; in-app notification bell still records
                    # the trigger, so signal isn't lost).
                    if not config.get('email.automated_enabled', False):
                        logger.debug(f"Watchlist alert for {item.ticker} skipped: email.automated_enabled=false")
                    elif send_watchlist_alert_email(item, stock, reasons):
                        alerts_sent += 1
                        logger.info(f"Watchlist alert sent for {item.ticker}: {', '.join(reasons)}")
                    else:
                        logger.warning(f"Failed to send watchlist alert email for {item.ticker}")
                except Exception as e:
                    logger.error(f"Error sending watchlist alert for {item.ticker}: {e}")
                    # Still mark as triggered to prevent retry storms
                    item.alert_triggered_at = datetime.now(timezone.utc)

        db.commit()

        if alerts_sent > 0:
            logger.info(f"Sent {alerts_sent} watchlist alerts")

    except Exception as e:
        logger.error(f"Watchlist alert check failed: {e}")
        db.rollback()
    finally:
        db.close()


def _set_phase(phase, detail=None, current=0, total=0):
    """Update the broader per-phase progress fields used by the UI.

    Keeps the existing `phase` / `phase_detail` strings working while also
    populating `current_phase` / `phase_current` / `phase_total` / `phase_label`
    so the frontend can render a phase-tagged progress bar that updates through
    insider/short, p1_data, saving, etc. (not just Phase 1).
    """
    with _state_lock:
        _scan_config["phase"] = phase
        _scan_config["current_phase"] = phase
        _scan_config["phase_current"] = current
        _scan_config["phase_total"] = total
        _scan_config["phase_label"] = _PHASE_LABELS.get(phase) if phase else None
        if detail is not None:
            _scan_config["phase_detail"] = detail


def get_scan_status():
    """Get current scheduler status"""
    next_run = None
    job = scheduler.get_job("continuous_scan")
    if job and job.next_run_time:
        # APScheduler returns timezone-aware datetime, isoformat() includes offset
        next_run = job.next_run_time.isoformat()

    with _state_lock:
        config_snapshot = dict(_scan_config)
    return {
        **config_snapshot,
        "scheduler_running": scheduler.running,
        "next_run": next_run
    }


def auto_record_coiled_spring_alerts():
    """
    Automatically check for and record Coiled Spring alerts after each scan.
    Uses the same criteria as the CS detection but auto-records to the alert table.
    """
    from backend.database import SessionLocal, Stock, CoiledSpringAlert
    from backend.ai_trader import calculate_coiled_spring_score_for_stock, record_coiled_spring_alert
    from config_loader import config
    from datetime import date, datetime, timedelta

    cs_config = config.get('coiled_spring', {})
    if not cs_config.get('alerts', {}).get('enabled', True):
        logger.info("CS auto-recording disabled in config")
        return

    db = SessionLocal()
    try:
        # Get thresholds from config (relaxed for pre-breakout detection)
        thresholds = cs_config.get('pre_breakout_thresholds', cs_config.get('thresholds', {}))
        min_weeks = thresholds.get('min_weeks_in_base', 15)
        min_beat_streak = thresholds.get('min_beat_streak', 3)
        min_c_score = thresholds.get('min_c_score', 5)
        min_total_score = thresholds.get('min_total_score', 48)
        max_inst_pct = cs_config.get('thresholds', {}).get('max_institutional_pct', 75)

        # Query candidates with basic filters
        candidates = db.query(Stock).filter(
            Stock.weeks_in_base >= min_weeks,
            Stock.earnings_beat_streak >= min_beat_streak,
            Stock.c_score >= min_c_score,
            Stock.canslim_score >= min_total_score,
            Stock.days_to_earnings != None,
            Stock.days_to_earnings > 0,
            Stock.days_to_earnings <= 14  # Within 2 weeks of earnings
        ).all()

        if not candidates:
            logger.info("No CS candidates found in this scan")
            return

        logger.info(f"Found {len(candidates)} potential CS candidates, checking each...")

        recorded = 0
        for stock in candidates:
            # Full CS check with all criteria
            cs_result = calculate_coiled_spring_score_for_stock(stock)

            if cs_result and cs_result.get('is_coiled_spring'):
                # Try to record (respects daily limits and cooldown)
                if record_coiled_spring_alert(db, stock.ticker, cs_result, stock):
                    recorded += 1
                    logger.info(f"Auto-recorded CS alert: {stock.ticker} ({cs_result.get('cs_details', '')})")

        if recorded > 0:
            logger.info(f"Auto-recorded {recorded} Coiled Spring alerts")
        else:
            logger.info("No new CS alerts to record (limits may have been reached)")

    except Exception as e:
        logger.error(f"CS auto-recording error: {e}")
    finally:
        db.close()


def update_coiled_spring_outcomes():
    """
    Automatically update outcomes for CS alerts that have passed earnings.
    Runs after each scan to track success/failure of CS predictions.

    Design note (#5, philosophical): the outcome bucket measures whether the
    stock is up X% from `price_at_alert` at evaluation time (>= check_days
    after earnings). It is a SIGNAL-QUALITY metric for the CS detector — it
    is NOT a simulated trader P&L. A real trader holding through earnings
    sees gap mechanics, intraday volatility, and exit timing that this
    metric intentionally ignores. The AI trader simulates that separately;
    do not try to fold trader behavior into this evaluator.
    """
    from backend.database import SessionLocal, Stock, CoiledSpringAlert, MarketSnapshot
    from config_loader import config
    from datetime import date, timedelta

    cs_config = config.get('coiled_spring', {})
    outcome_config = cs_config.get('outcome_tracking', {})

    if not outcome_config.get('enabled', True):
        logger.debug("CS outcome tracking disabled in config")
        return

    db = SessionLocal()
    try:
        # Get thresholds from config
        check_days = outcome_config.get('check_days_after_earnings', 3)
        thresholds = outcome_config.get('thresholds', {})
        big_win_pct = thresholds.get('big_win_pct', 15)
        win_pct = thresholds.get('win_pct', 5)
        loss_pct = thresholds.get('loss_pct', -5)

        # Find alerts without outcomes that are old enough to evaluate.
        # Design note (#2, intentional): we ONLY consider rows where
        # `outcome IS NULL`. The first evaluation wins permanently — a later
        # choppy day cannot flip "win" → "flat". Re-evaluating on every scan
        # would create infinite mutability and confuse users ("why did my
        # win flip to a loss overnight?"). If the snapshot timing turns out
        # to be a real problem, fix it by extending the wait window, NOT by
        # re-bucketing already-evaluated alerts.
        cutoff_date = date.today() - timedelta(days=check_days)

        alerts_to_update = db.query(CoiledSpringAlert).filter(
            CoiledSpringAlert.outcome.is_(None),
            CoiledSpringAlert.alert_date <= cutoff_date
        ).all()

        if not alerts_to_update:
            logger.debug("No CS alerts to update outcomes for")
            return

        logger.info(f"Checking outcomes for {len(alerts_to_update)} CS alerts...")

        # Batch-fetch all stocks for alerts (avoids N+1 queries)
        alert_tickers = list(set(a.ticker for a in alerts_to_update))
        alert_stocks = db.query(Stock).filter(Stock.ticker.in_(alert_tickers)).all()
        stocks_by_ticker = {s.ticker: s for s in alert_stocks}

        # SPY is used for alpha-relative bucketing (fix #4): subtract SPY's
        # same-window % change from the alert's price change so a brutal
        # market day doesn't directly bucket a stock-level alert as a loss.
        # Falls back to absolute price change if SPY data is missing.
        # `current_price` from the stocks table is the latest scan; the
        # historical SPY-at-alert-date price comes from the daily
        # MarketSnapshot table (one row per market day).
        spy_stock = db.query(Stock).filter(Stock.ticker == 'SPY').first()
        spy_current_price = (
            spy_stock.current_price
            if spy_stock and spy_stock.current_price and spy_stock.current_price > 0
            else None
        )

        # Batch-fetch SPY snapshot prices for all unique alert_dates so we
        # don't issue an N+1 of MarketSnapshot lookups inside the loop.
        spy_at_alert_by_date: dict = {}
        if spy_current_price is not None:
            unique_alert_dates = list({a.alert_date for a in alerts_to_update})
            spy_snapshots = (
                db.query(MarketSnapshot)
                .filter(MarketSnapshot.date.in_(unique_alert_dates))
                .all()
            )
            spy_at_alert_by_date = {
                snap.date: snap.spy_price
                for snap in spy_snapshots
                if snap.spy_price and snap.spy_price > 0
            }

        updated = 0
        for alert in alerts_to_update:
            # Get current stock price
            stock = stocks_by_ticker.get(alert.ticker)
            if not stock or not stock.current_price:
                continue

            # Determine the actual wait window. If the stock's
            # `next_earnings_date` is available, compute the wait from the
            # actual earnings date — handles reschedules / pre-announcements
            # / surprise reports where the snapshotted DTE is now wrong.
            # Falls back to the snapshotted DTE when next_earnings_date is
            # null (legacy rows / detector ran without earnings data).
            actual_earnings_date = getattr(stock, 'next_earnings_date', None)
            days_since_alert = (date.today() - alert.alert_date).days
            if actual_earnings_date is not None:
                days_alert_to_earnings = max(
                    0, (actual_earnings_date - alert.alert_date).days
                )
                days_needed = days_alert_to_earnings + check_days
            else:
                days_needed = (alert.days_to_earnings or 7) + check_days

            if days_since_alert < days_needed:
                # Not enough time has passed
                continue

            # Calculate price change
            if not alert.price_at_alert or alert.price_at_alert <= 0:
                continue

            raw_price_change_pct = (
                (stock.current_price - alert.price_at_alert) / alert.price_at_alert
            ) * 100

            # Alpha-relative bucketing (fix #4): subtract SPY's same-window
            # change so a market-wide drop on eval day doesn't unfairly
            # bucket a stock-specific alert. Falls back to raw change when
            # SPY data isn't reliably available (preserves prior behavior).
            spy_change_pct = None
            if spy_current_price is not None:
                spy_at_alert = spy_at_alert_by_date.get(alert.alert_date)
                if spy_at_alert and spy_at_alert > 0:
                    spy_change_pct = (
                        (spy_current_price - spy_at_alert) / spy_at_alert
                    ) * 100

            if spy_change_pct is not None:
                price_change_pct = raw_price_change_pct - spy_change_pct
            else:
                price_change_pct = raw_price_change_pct

            # Determine outcome
            if price_change_pct >= big_win_pct:
                outcome = 'big_win'
            elif price_change_pct >= win_pct:
                outcome = 'win'
            elif price_change_pct <= loss_pct:
                outcome = 'loss'
            else:
                outcome = 'flat'

            # Update alert (fix #1: pin outcome_updated_at for audit trail)
            alert.price_after_earnings = stock.current_price
            alert.price_change_pct = round(price_change_pct, 2)
            alert.outcome = outcome
            alert.outcome_updated_at = datetime.now(timezone.utc)
            updated += 1

            logger.info(f"CS outcome {alert.ticker}: {outcome} ({price_change_pct:+.1f}%)")

        if updated > 0:
            db.commit()
            logger.info(f"Updated {updated} CS alert outcomes")

    except Exception as e:
        logger.error(f"CS outcome tracking error: {e}")
        db.rollback()
    finally:
        db.close()


# Morning briefing dedup is persisted in SystemState (key="last_briefing_date")
# so a redeploy mid-morning doesn't re-trigger today's briefing. Pre-fix this
# was a module-level global that reset on every restart.
LAST_BRIEFING_DATE_KEY = "last_briefing_date"


def send_morning_briefing_if_due():
    """Send morning briefing email once per day (first scan after configured time)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config_loader import config as yaml_config
    from backend.database import SessionLocal, Stock, AIPortfolioPosition, get_system_state, set_system_state

    briefing_config = yaml_config.get('morning_briefing', {})
    if not briefing_config.get('enabled', True):
        return

    now = datetime.now(timezone.utc)
    today = now.date() if hasattr(now, 'date') else now
    today_iso = today.isoformat()

    # Only send once per day — read previous from SystemState (survives restart)
    dedup_db = SessionLocal()
    try:
        last_sent_iso = get_system_state(dedup_db, LAST_BRIEFING_DATE_KEY)
    finally:
        dedup_db.close()
    if last_sent_iso == today_iso:
        return

    # Check if it's after the configured send time (default 8:00 AM CT)
    from zoneinfo import ZoneInfo
    ct_now = datetime.now(ZoneInfo("America/Chicago"))
    send_hour = briefing_config.get('send_time_hour', 8)
    send_minute = briefing_config.get('send_time_minute', 0)
    if ct_now.hour < send_hour or (ct_now.hour == send_hour and ct_now.minute < send_minute):
        return

    # Don't send on weekends
    if ct_now.weekday() > 4:
        return

    logger.info("Sending morning briefing email...")
    db = SessionLocal()
    try:
        from backend.ai_trader import get_portfolio_value, get_market_regime, evaluate_buys, get_effective_score
        from backend.email_utils import send_morning_briefing_email

        # Portfolio summary
        portfolio = get_portfolio_value(db)
        market_regime = get_market_regime(db)

        # Current positions with stop distances and earnings warnings
        from backend.trading_engine import get_trailing_stop_pct, apply_pyramid_widening
        positions = db.query(AIPortfolioPosition).all()

        # Load strategy profile for stop calculation
        profile = yaml_config.get(f'strategy_profiles.{config.strategy}', {}) if hasattr(config, 'strategy') else {}
        base_stop = profile.get('stop_loss_pct', 7.0)

        # Pre-load earnings dates for positions
        pos_tickers = [p.ticker for p in positions]
        pos_stocks = db.query(Stock).filter(Stock.ticker.in_(pos_tickers)).all() if pos_tickers else []
        stocks_map = {s.ticker: s for s in pos_stocks}

        pos_data = []
        near_stop_positions = []
        earnings_warnings = []
        for p in positions:
            gain_pct = p.gain_loss_pct or 0
            trail_pct = get_trailing_stop_pct(gain_pct, profile)
            trail_pct = apply_pyramid_widening(trail_pct, p.pyramid_count or 0)

            if gain_pct > 10 and p.peak_price:
                stop_price = p.peak_price * (1 - trail_pct / 100)
            else:
                stop_price = p.cost_basis * (1 - base_stop / 100)

            pct_to_stop = ((p.current_price - stop_price) / p.current_price * 100) if p.current_price and stop_price else 0
            days_held = (datetime.now(timezone.utc) - p.purchase_date).days if p.purchase_date else 0

            pos_data.append({
                "ticker": p.ticker,
                "price": p.current_price or 0,
                "gain_pct": gain_pct,
                "score": get_effective_score(p, use_current=True),
                "pct_to_stop": round(pct_to_stop, 1),
                "stop_price": round(stop_price, 2),
                "days_held": days_held,
            })

            # Flag positions within 3% of stop
            if pct_to_stop < 3:
                near_stop_positions.append({"ticker": p.ticker, "pct_to_stop": round(pct_to_stop, 1)})

            # Flag positions with earnings this week
            stock = stocks_map.get(p.ticker)
            if stock and hasattr(stock, 'next_earnings_date') and stock.next_earnings_date:
                from datetime import date
                try:
                    if isinstance(stock.next_earnings_date, str):
                        earnings_date = date.fromisoformat(stock.next_earnings_date)
                    else:
                        earnings_date = stock.next_earnings_date if isinstance(stock.next_earnings_date, date) else stock.next_earnings_date.date()
                    days_to_earnings = (earnings_date - date.today()).days
                    if 0 <= days_to_earnings <= 5:
                        earnings_warnings.append({"ticker": p.ticker, "days": days_to_earnings, "date": str(earnings_date)})
                except (ValueError, TypeError):
                    pass

        # Top candidates (evaluate without executing)
        try:
            buys = evaluate_buys(db)[:5]
            candidates = [{
                "ticker": b["stock"].ticker,
                "price": b["stock"].current_price or 0,
                "score": b.get("effective_score", 0),
                "reason": b.get("reason", "")
            } for b in buys]
        except Exception as e:
            logger.warning(f"Failed to evaluate buy candidates for briefing: {e}")
            candidates = []

        # Market timing info
        timing_info = {"state": "N/A", "can_buy": True}
        try:
            from backend.historical_data import HistoricalDataProvider
            from datetime import date, timedelta
            provider = HistoricalDataProvider(['SPY'])
            provider.preload_data(date.today() - timedelta(days=60), date.today())
            timing_info = provider.get_follow_through_day_status(date.today())
        except Exception as e:
            logger.warning(f"Failed to load market timing info: {e}")

        # Portfolio heat
        heat = 0.0
        stop_cfg = yaml_config.get('ai_trader.stops', {})
        base_stop = stop_cfg.get('normal_stop_loss_pct', 8.0)
        pv = portfolio.get("total_value", 0)
        if pv > 0:
            for p in positions:
                if p.current_price and p.current_price > 0 and p.cost_basis and p.cost_basis > 0:
                    pos_pct = ((p.current_value or 0) / pv) * 100
                    g_pct = ((p.current_price - p.cost_basis) / p.cost_basis) * 100
                    heat += pos_pct * ((base_stop + g_pct) / 100)

        briefing_data = {
            "portfolio": portfolio,
            "market_regime": market_regime,
            "positions": pos_data,
            "top_candidates": candidates,
            "market_timing": timing_info,
            "portfolio_heat": heat,
            "near_stop_positions": near_stop_positions,
            "earnings_warnings": earnings_warnings,
        }

        # Gate email path on email.automated_enabled (May 7 2026 — push
        # notification path below stays on; that's the channel the user wants).
        # NOTE: This function imports config_loader as `yaml_config` (line ~632).
        if yaml_config.get('email.automated_enabled', False):
            success = send_morning_briefing_email(briefing_data)
            if success:
                set_system_state(db, LAST_BRIEFING_DATE_KEY, today_iso)
                db.commit()
                logger.info("Morning briefing sent successfully")
            else:
                logger.warning("Morning briefing email failed to send")
        else:
            # Email path disabled — still mark as fired so we don't re-attempt
            # the briefing later in the day via the push path.
            set_system_state(db, LAST_BRIEFING_DATE_KEY, today_iso)
            db.commit()
            logger.debug("Morning briefing email skipped: email.automated_enabled=false (push still sent)")

        # Also send compact push notification (always on — push is the
        # preferred channel; user disabled email separately).
        try:
            from backend.email_utils import send_morning_briefing_push
            send_morning_briefing_push(briefing_data)
        except Exception as e:
            logger.error(f"Morning briefing push failed: {e}")

    except Exception as e:
        logger.error(f"Morning briefing generation error: {e}")
    finally:
        db.close()


def run_continuous_scan():
    """Execute a scan of the configured stock universe"""
    from backend.database import SessionLocal, Stock
    from sp500_tickers import get_sp500_tickers, get_russell2000_tickers, get_all_tickers
    import time

    with _state_lock:
        if _scan_config["is_scanning"]:
            logger.info("Scan already in progress, skipping...")
            return
        _scan_config["is_scanning"] = True
        _scan_config["last_scan_start"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        _scan_config["stocks_scanned"] = 0
        _scan_config["total_stocks"] = 0
        _scan_config["phase"] = "scanning"
        _scan_config["phase_detail"] = "Initializing scan..."
        _scan_config["current_phase"] = "scanning"
        _scan_config["phase_current"] = 0
        _scan_config["phase_total"] = 0
        _scan_config["phase_label"] = _PHASE_LABELS.get("scanning")

    logger.info(f"Starting continuous scan ({_scan_config['source']})...")

    # Get tickers based on source
    # Always include portfolio tickers first (they're most important)
    from sp500_tickers import get_portfolio_tickers
    portfolio_tickers = get_portfolio_tickers()

    source = _scan_config["source"]
    if source == "top50":
        base_tickers = get_sp500_tickers()[:50]
    elif source == "sp500":
        base_tickers = get_sp500_tickers()
    elif source == "russell":
        base_tickers = get_russell2000_tickers()
    elif source == "all":
        base_tickers = get_all_tickers(include_portfolio=False)  # Portfolio added separately
    else:
        base_tickers = get_sp500_tickers()

    # Combine: portfolio first (priority), then base tickers, deduplicated
    seen = set()
    tickers = []
    for t in portfolio_tickers + base_tickers:
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    # Shuffle ONLY the non-portfolio tickers to avoid Yahoo warmup issues
    # Keep portfolio tickers at the front (they're most important)
    num_portfolio = len(portfolio_tickers)
    portfolio_section = tickers[:num_portfolio]
    rest_section = tickers[num_portfolio:]
    random.shuffle(rest_section)
    tickers = portfolio_section + rest_section

    logger.info(f"Including {len(portfolio_tickers)} portfolio tickers in scan")

    _scan_config["total_stocks"] = len(tickers)
    logger.info(f"Scanning {len(tickers)} stocks...")

    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from canslim_scorer import CANSLIMScorer, GrowthModeScorer, TechnicalAnalyzer
    from data_fetcher import (
        DataFetcher, get_cached_market_direction,
        fetch_fmp_insider_trading, fetch_short_interest,
        fetch_fmp_earnings_calendar, fetch_fmp_analyst_estimates,
        is_data_fresh, mark_data_fetched
    )
    from growth_projector import GrowthProjector

    # IMPORTANT: Fetch market direction FIRST, before any stock analysis
    # This ensures the M score uses fresh data and avoids rate limiting during stock scans
    logger.info("Fetching market direction data (SPY, QQQ, DIA)...")
    market_data = get_cached_market_direction(force_refresh=True)
    if market_data.get("success"):
        logger.info(f"Market direction: {market_data.get('market_trend')} "
                   f"(score: {market_data.get('market_score')}, "
                   f"signal: {market_data.get('weighted_signal', 0):.2f})")
    else:
        logger.warning(f"Failed to fetch market direction: {market_data.get('error')}")

    data_fetcher = DataFetcher()
    canslim_scorer = CANSLIMScorer(data_fetcher)
    growth_mode_scorer = GrowthModeScorer(data_fetcher, canslim_scorer)
    growth_projector = GrowthProjector(data_fetcher)

    # Pre-load industry group ranks from DB (from previous scan cycle)
    _industry_group_ranks = {}
    try:
        ig_preload_db = SessionLocal()
        ig_rows = ig_preload_db.query(Stock.ticker, Stock.industry_group_rank).filter(
            Stock.industry_group_rank != None
        ).all()
        _industry_group_ranks = {t: r for t, r in ig_rows}
        ig_preload_db.close()
        if _industry_group_ranks:
            logger.info(f"Loaded industry group ranks for {len(_industry_group_ranks)} stocks")
    except Exception as e:
        logger.debug(f"Industry group rank preload failed: {e}")

    def analyze_stock(ticker: str) -> dict:
        """Analyze a single stock"""
        try:
            stock_data = data_fetcher.get_stock_data(ticker)
            if not stock_data or not stock_data.is_valid:
                return None

            # Pass industry group rank from previous cycle (bootstrap: first scan has no ranks)
            ig_rank = _industry_group_ranks.get(ticker)
            canslim_result = canslim_scorer.score_stock(stock_data, industry_group_rank=ig_rank)
            projection = growth_projector.project_growth(
                stock_data=stock_data,
                canslim_score=canslim_result
            )

            # Extract RS values for persistence
            rs_values = canslim_scorer.extract_rs_values(stock_data)

            # Calculate Growth Mode score if applicable
            growth_mode_result = None
            is_growth_stock = growth_mode_scorer.should_use_growth_mode(stock_data)
            if is_growth_stock:
                growth_mode_result = growth_mode_scorer.score_stock(stock_data)

            # Technical analysis
            base_pattern = TechnicalAnalyzer.detect_base_pattern(stock_data.weekly_price_history)
            volume_ratio = TechnicalAnalyzer.calculate_volume_ratio(stock_data)
            is_breaking_out, breakout_vol = TechnicalAnalyzer.is_breaking_out(stock_data, base_pattern)

            # Insider trading signals (fetch weekly)
            insider_data = {}
            if not is_data_fresh(ticker, "insider_trading"):
                insider_data = fetch_fmp_insider_trading(ticker)
                if insider_data:
                    mark_data_fetched(ticker, "insider_trading")

            # Short interest data (fetch daily)
            short_data = {}
            if not is_data_fresh(ticker, "short_interest"):
                short_data = fetch_short_interest(ticker)
                if short_data:
                    mark_data_fetched(ticker, "short_interest")

            # P1 Feature: Earnings calendar (fetch weekly)
            earnings_calendar_data = {}
            if not is_data_fresh(ticker, "earnings_calendar"):
                earnings_calendar_data = fetch_fmp_earnings_calendar(ticker)
                if earnings_calendar_data:
                    mark_data_fetched(ticker, "earnings_calendar")

            # P1 Feature: Analyst estimates (fetch every 3 days)
            analyst_estimates_data = {}
            if not is_data_fresh(ticker, "analyst_estimates"):
                analyst_estimates_data = fetch_fmp_analyst_estimates(ticker)
                if analyst_estimates_data:
                    mark_data_fetched(ticker, "analyst_estimates")

            return {
                "ticker": ticker,
                "company_name": stock_data.name,
                "sector": stock_data.sector,
                "industry": stock_data.industry or None,
                "current_price": stock_data.current_price,
                "market_cap": stock_data.market_cap,
                "canslim_score": canslim_result.total_score,
                "c_score": canslim_result.c_score,
                "a_score": canslim_result.a_score,
                "n_score": canslim_result.n_score,
                "s_score": canslim_result.s_score,
                "l_score": canslim_result.l_score,
                "i_score": canslim_result.i_score,
                "m_score": canslim_result.m_score,
                "score_details": {
                    "c": {
                        "summary": canslim_result.c_detail,
                        "quarterly_eps": stock_data.quarterly_earnings[:4] if stock_data.quarterly_earnings else [],
                        "earnings_surprise_pct": stock_data.earnings_surprise_pct,
                    },
                    "a": {
                        "summary": canslim_result.a_detail,
                        "annual_eps": stock_data.annual_earnings[:3] if stock_data.annual_earnings else [],
                        "roe": stock_data.roe,
                    },
                    "n": {
                        "summary": canslim_result.n_detail,
                        "current_price": stock_data.current_price,
                        "week_52_high": stock_data.high_52w,
                        "pct_from_high": round(((stock_data.high_52w - stock_data.current_price) / stock_data.high_52w) * 100, 1) if stock_data.high_52w and stock_data.current_price else None,
                    },
                    "s": {
                        "summary": canslim_result.s_detail,
                        "volume_ratio": volume_ratio,
                        "avg_volume": stock_data.avg_volume_50d,
                        "shares_outstanding": stock_data.shares_outstanding,
                    },
                    "l": {
                        "summary": canslim_result.l_detail,
                        "relative_strength": stock_data.relative_strength if hasattr(stock_data, 'relative_strength') else None,
                        "rs_12m": rs_values.get("rs_12m"),
                        "rs_3m": rs_values.get("rs_3m"),
                    },
                    "i": {
                        "summary": canslim_result.i_detail,
                        "institutional_pct": stock_data.institutional_holders_pct,
                    },
                    "m": {
                        "summary": canslim_result.m_detail,
                    },
                },
                "projected_growth": projection.projected_growth_pct,
                "confidence": projection.confidence,
                "analyst_target": stock_data.analyst_target_price,
                "pe_ratio": stock_data.trailing_pe,
                "week_52_high": stock_data.high_52w,
                "week_52_low": stock_data.low_52w,
                "relative_strength": None,  # Could parse from l_detail if needed
                "institutional_ownership": stock_data.institutional_holders_pct,
                # RS values for persistence
                "rs_12m": rs_values.get("rs_12m"),
                "rs_3m": rs_values.get("rs_3m"),
                # Growth Mode scoring
                "is_growth_stock": is_growth_stock,
                "growth_mode_score": growth_mode_result.total_score if growth_mode_result else None,
                "growth_mode_details": {
                    "r": growth_mode_result.r_detail,
                    "f": growth_mode_result.f_detail,
                    "n": growth_mode_result.n_detail,
                    "s": growth_mode_result.s_detail,
                    "l": growth_mode_result.l_detail,
                    "i": growth_mode_result.i_detail,
                    "m": growth_mode_result.m_detail,
                } if growth_mode_result else None,
                # Enhanced earnings
                "eps_acceleration": len(stock_data.quarterly_earnings) >= 5 and canslim_result.c_detail and "+accel" in canslim_result.c_detail,
                "earnings_surprise_pct": stock_data.earnings_surprise_pct,
                "revenue_growth_pct": _calc_revenue_growth(stock_data),
                # Technical analysis
                "volume_ratio": volume_ratio,
                "weeks_in_base": base_pattern.get("weeks", 0),
                "base_type": base_pattern.get("type", "none"),
                "pivot_price": base_pattern.get("pivot_price", 0),
                "is_breaking_out": is_breaking_out,
                "breakout_volume_ratio": breakout_vol if is_breaking_out else None,
                # Insider trading signals
                "insider_buy_count": insider_data.get("buy_count"),
                "insider_sell_count": insider_data.get("sell_count"),
                "insider_net_shares": insider_data.get("net_shares"),
                "insider_sentiment": insider_data.get("sentiment"),
                # Short interest
                "short_interest_pct": short_data.get("short_interest_pct"),
                "short_ratio": short_data.get("short_ratio"),
                # P1 Feature: Earnings Calendar
                "next_earnings_date": earnings_calendar_data.get("next_earnings_date"),
                "days_to_earnings": earnings_calendar_data.get("days_to_earnings"),
                "earnings_beat_streak": earnings_calendar_data.get("earnings_beat_streak"),
                # P1 Feature: Analyst Estimate Revisions
                "eps_estimate_current": analyst_estimates_data.get("eps_estimate_current"),
                "eps_estimate_prior": analyst_estimates_data.get("eps_estimate_prior"),
                "eps_estimate_revision_pct": analyst_estimates_data.get("eps_estimate_revision_pct"),
                "estimate_revision_trend": analyst_estimates_data.get("estimate_revision_trend"),
                # P1 Feature: Insider Value Tracking (from enhanced insider_data)
                "insider_buy_value": insider_data.get("buy_value"),
                "insider_sell_value": insider_data.get("sell_value"),
                "insider_net_value": insider_data.get("net_value"),
                "insider_largest_buy": insider_data.get("largest_buy"),
                "insider_largest_buyer_title": insider_data.get("largest_buyer_title"),
                # Raw earnings data (needed for database save)
                "quarterly_earnings": stock_data.quarterly_earnings,
                "annual_earnings": stock_data.annual_earnings,
                "quarterly_revenue": stock_data.quarterly_revenue,
            }
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None

    def _calc_revenue_growth(stock_data) -> float:
        """Calculate YoY revenue growth percentage"""
        if stock_data.quarterly_revenue and len(stock_data.quarterly_revenue) >= 5:
            current = stock_data.quarterly_revenue[0]
            prior = stock_data.quarterly_revenue[4]
            if prior > 0:
                return round(((current - prior) / prior) * 100, 1)
        return None

    def save_stock_to_db(db, analysis: dict):
        """Save analysis to database with historical tracking"""
        from backend.database import Stock, StockScore
        from datetime import date

        stock = db.query(Stock).filter(Stock.ticker == analysis["ticker"]).first()
        if not stock:
            stock = Stock(ticker=analysis["ticker"])
            db.add(stock)
            db.flush()  # Get the ID

        # Track score change
        old_score = stock.canslim_score
        new_score = analysis.get("canslim_score")

        # SAFEGUARD: Detect potential data blips before saving
        # If score dropped dramatically AND key data is missing, keep the old score
        if old_score is not None and new_score is not None:
            score_drop = old_score - new_score

            # Check for signs of incomplete data
            has_earnings_data = bool(analysis.get("quarterly_earnings"))
            has_price_data = analysis.get("current_price") is not None and analysis.get("current_price") > 0
            has_52w_high = analysis.get("week_52_high") is not None and analysis.get("week_52_high") > 0

            # Count how many component scores are 0 (suspicious if many are 0)
            component_scores = [
                analysis.get("c_score", 0),
                analysis.get("a_score", 0),
                analysis.get("n_score", 0),
                analysis.get("s_score", 0),
                analysis.get("l_score", 0),
                analysis.get("i_score", 0),
            ]
            zero_components = sum(1 for s in component_scores if s == 0)

            # Check score details for "Insufficient data" or "No data" indicators
            score_details = analysis.get("score_details", {})
            data_issues = []
            for component, details in score_details.items():
                if isinstance(details, dict):
                    summary = details.get("summary", "")
                    if any(x in summary.lower() for x in ["insufficient", "no data", "no price", "no volume"]):
                        data_issues.append(component.upper())

            # BLIP DETECTION: Score dropped >25 points AND looks like missing data
            is_likely_blip = (
                score_drop > 25 and  # Big drop
                (zero_components >= 3 or  # Too many zero components
                 not has_earnings_data or  # Missing earnings
                 len(data_issues) >= 2)  # Multiple data issues
            )

            if is_likely_blip:
                logger.warning(f"DATA BLIP DETECTED for {analysis['ticker']}: "
                              f"Score would drop {old_score:.0f} → {new_score:.0f} (-{score_drop:.0f}). "
                              f"Issues: {data_issues}, Zero components: {zero_components}. "
                              f"KEEPING OLD SCORE.")
                # Keep the old score - don't update
                new_score = old_score
                analysis["canslim_score"] = old_score
                # Also preserve component scores if they were non-zero before
                if stock.c_score and analysis.get("c_score") == 0:
                    analysis["c_score"] = stock.c_score
                if stock.a_score and analysis.get("a_score") == 0:
                    analysis["a_score"] = stock.a_score
                if stock.n_score and analysis.get("n_score") == 0:
                    analysis["n_score"] = stock.n_score
                if stock.s_score and analysis.get("s_score") == 0:
                    analysis["s_score"] = stock.s_score
                if stock.l_score and analysis.get("l_score") == 0:
                    analysis["l_score"] = stock.l_score
                if stock.i_score and analysis.get("i_score") == 0:
                    analysis["i_score"] = stock.i_score

            stock.previous_score = old_score
            stock.score_change = round(new_score - old_score, 2)
        else:
            stock.previous_score = None
            stock.score_change = None

        # Update stock data (only overwrite with non-None values to prevent
        # API failures from wiping existing good data)
        stock.name = analysis.get("company_name") or stock.name
        stock.sector = analysis.get("sector") or stock.sector
        stock.industry = analysis.get("industry") or stock.industry
        stock.current_price = analysis.get("current_price") or stock.current_price
        stock.market_cap = analysis.get("market_cap") or stock.market_cap
        stock.canslim_score = new_score
        stock.c_score = analysis.get("c_score")
        stock.a_score = analysis.get("a_score")
        stock.n_score = analysis.get("n_score")
        stock.s_score = analysis.get("s_score")
        stock.l_score = analysis.get("l_score")
        stock.i_score = analysis.get("i_score")
        stock.m_score = analysis.get("m_score")
        stock.score_details = analysis.get("score_details")
        stock.projected_growth = analysis.get("projected_growth")
        stock.growth_confidence = analysis.get("confidence")
        stock.week_52_high = analysis.get("week_52_high") or stock.week_52_high
        stock.week_52_low = analysis.get("week_52_low") or stock.week_52_low
        stock.last_updated = datetime.now(timezone.utc)

        # RS values for momentum confirmation
        stock.rs_12m = analysis.get("rs_12m")
        stock.rs_3m = analysis.get("rs_3m")

        # Growth Mode scoring
        stock.growth_mode_score = analysis.get("growth_mode_score")
        stock.growth_mode_details = analysis.get("growth_mode_details")
        stock.is_growth_stock = analysis.get("is_growth_stock", False)

        # Enhanced earnings analysis
        stock.eps_acceleration = analysis.get("eps_acceleration")
        stock.earnings_surprise_pct = analysis.get("earnings_surprise_pct")
        stock.revenue_growth_pct = analysis.get("revenue_growth_pct")
        stock.quarterly_earnings = analysis.get("quarterly_earnings")
        stock.annual_earnings = analysis.get("annual_earnings")
        stock.quarterly_revenue = analysis.get("quarterly_revenue")

        # Technical analysis
        stock.volume_ratio = analysis.get("volume_ratio")
        stock.weeks_in_base = analysis.get("weeks_in_base")
        stock.base_type = analysis.get("base_type")
        stock.pivot_price = analysis.get("pivot_price")
        stock.is_breaking_out = analysis.get("is_breaking_out", False)
        stock.breakout_volume_ratio = analysis.get("breakout_volume_ratio")
        stock.volume_dry_up = analysis.get("volume_dry_up", False)
        stock.volume_dry_up_score = analysis.get("volume_dry_up_score", 0)
        stock.institutional_accumulation = analysis.get("institutional_accumulation", False)
        if analysis.get("ma_21") is not None:
            stock.ma_21 = analysis.get("ma_21")
        if analysis.get("ma_50") is not None:
            stock.ma_50 = analysis.get("ma_50")
        if analysis.get("atr_pct") is not None:
            stock.atr_pct = analysis.get("atr_pct")

        # Insider trading signals (only update if we have data)
        if analysis.get("insider_sentiment"):
            stock.insider_buy_count = analysis.get("insider_buy_count")
            stock.insider_sell_count = analysis.get("insider_sell_count")
            stock.insider_net_shares = analysis.get("insider_net_shares")
            stock.insider_sentiment = analysis.get("insider_sentiment")
            stock.insider_updated_at = datetime.now(timezone.utc)
            # P1 Feature: Insider value tracking
            stock.insider_buy_value = analysis.get("insider_buy_value")
            stock.insider_sell_value = analysis.get("insider_sell_value")
            stock.insider_net_value = analysis.get("insider_net_value")
            stock.insider_largest_buy = analysis.get("insider_largest_buy")
            stock.insider_largest_buyer_title = analysis.get("insider_largest_buyer_title")

        # Short interest (only update if we have data)
        if analysis.get("short_interest_pct") is not None:
            stock.short_interest_pct = analysis.get("short_interest_pct")
            stock.short_ratio = analysis.get("short_ratio")
            stock.short_updated_at = datetime.now(timezone.utc)

        # P1 Feature: Earnings calendar (only update if we have data)
        if analysis.get("next_earnings_date") or analysis.get("earnings_beat_streak"):
            # Convert string date to Python date object for SQLite
            next_earnings_str = analysis.get("next_earnings_date")
            if next_earnings_str:
                stock.next_earnings_date = datetime.strptime(next_earnings_str, '%Y-%m-%d').date()
            stock.days_to_earnings = analysis.get("days_to_earnings")
            stock.earnings_beat_streak = analysis.get("earnings_beat_streak")
            stock.earnings_calendar_updated_at = datetime.now(timezone.utc)

        # P1 Feature: Analyst estimate revisions (only update if we have data)
        if analysis.get("eps_estimate_current") is not None:
            stock.eps_estimate_current = analysis.get("eps_estimate_current")
            stock.eps_estimate_prior = analysis.get("eps_estimate_prior")
            stock.eps_estimate_revision_pct = analysis.get("eps_estimate_revision_pct")
            stock.estimate_revision_trend = analysis.get("estimate_revision_trend")
            stock.analyst_estimates_updated_at = datetime.now(timezone.utc)

        # Save historical score (one per scan for granular backtesting data)
        today = date.today()
        historical_score = StockScore(
            stock_id=stock.id,
            timestamp=datetime.now(timezone.utc),
            date=today,
            total_score=new_score,
            c_score=analysis.get("c_score"),
            c_score_uncapped=analysis.get("c_score_uncapped"),
            a_score=analysis.get("a_score"),
            n_score=analysis.get("n_score"),
            s_score=analysis.get("s_score"),
            l_score=analysis.get("l_score"),
            i_score=analysis.get("i_score"),
            m_score=analysis.get("m_score"),
            projected_growth=analysis.get("projected_growth"),
            current_price=analysis.get("current_price"),
            week_52_high=analysis.get("week_52_high")
        )
        db.add(historical_score)

        # NOTE: Don't commit here - batched commits happen in the main loop for performance
        return stock

    try:
        # ASYNC SCANNING: Use async batch processing for 10x performance boost
        # Fetches 50 stocks concurrently, then saves to database
        # Performance: ~10 stocks in 3s vs 30-50s with old method
        logger.info(f"Starting ASYNC scan of {len(tickers)} stocks (batch_size=100)...")

        from async_scanner import run_async_scan

        # Phase 1: Scanning stocks
        _set_phase("scanning", detail="Fetching stock data & P1 metrics...", current=0, total=len(tickers))

        # Progress callback to update frontend in real-time
        def update_progress(current, total, phase="stocks"):
            # stocks_scanned / total_stocks remain Phase-1 only — insider_short
            # and p1_data run over a different (smaller) population, so writing
            # them into stocks_scanned would regress the existing UI.
            if phase == "stocks":
                _scan_config["stocks_scanned"] = current
                # Also update total if it differs (e.g., from checkpoint resume)
                if _scan_config["total_stocks"] != total:
                    _scan_config["total_stocks"] = total

            # Build the phase_detail string + drive the broader per-phase
            # progress fields so the UI keeps moving after Phase 1 ends.
            if phase == "stocks":
                detail = f"Fetching stock data ({current}/{total})..."
            elif phase == "insider_short":
                detail = f"Fetching insider/short data ({current}/{total})..."
            elif phase == "p1_data":
                detail = f"Fetching P1 data ({current}/{total})..."
            else:
                detail = f"{phase} ({current}/{total})..."

            _set_phase(phase, detail=detail, current=current, total=total)

            if total and current % 100 == 0:  # Log every 100 items
                logger.info(f"Progress ({phase}): {current}/{total} ({current/total*100:.1f}%)")

        # Fetch and analyze all stocks asynchronously (this is the fast part!)
        start_time = time.time()
        analysis_results = run_async_scan(tickers, batch_size=100, progress_callback=update_progress)
        fetch_time = time.time() - start_time

        logger.info(f"✓ Async fetching complete: {len(analysis_results)} stocks in {fetch_time:.1f}s ({fetch_time/len(analysis_results):.2f}s per stock)")

        # Phase 2: Saving to database
        _set_phase(
            "saving",
            detail=f"Saving {len(analysis_results)} stocks to database...",
            current=0,
            total=len(analysis_results),
        )
        _scan_config["stocks_scanned"] = 0  # Reset for save progress

        # Save results to database with BATCHED COMMITS for performance
        # Uses savepoints so one bad stock doesn't lose the entire batch
        BATCH_SIZE = 50
        db = SessionLocal()
        successful = 0
        failed = 0
        try:
            for i, analysis in enumerate(analysis_results):
                nested = None
                try:
                    # Use savepoint so a single failure doesn't rollback the batch
                    nested = db.begin_nested()
                    save_stock_to_db(db, analysis)
                    nested.commit()
                    successful += 1
                    # Update progress in real-time
                    _scan_config["stocks_scanned"] = successful
                    _scan_config["phase_current"] = successful

                    # Batch commit every BATCH_SIZE stocks
                    if successful % BATCH_SIZE == 0:
                        db.commit()
                        logger.debug(f"DB batch commit: {successful} stocks saved")
                except Exception as e:
                    failed += 1
                    logger.error(f"Error saving {analysis.get('ticker', 'unknown')}: {e}")
                    if nested:
                        nested.rollback()  # Only rollback this one stock's savepoint

            # Final commit for remaining stocks
            try:
                db.commit()
                if failed:
                    logger.warning(f"Stock save: {successful} succeeded, {failed} failed")
            except Exception as e:
                logger.error(f"Final commit error: {e}")
                db.rollback()
        finally:
            db.close()

        save_time = time.time() - start_time - fetch_time
        total_time = time.time() - start_time
        _scan_config["stocks_scanned"] = successful
        logger.info(f"Continuous scan complete: {successful}/{len(tickers)} stocks in {total_time:.1f}s total")
        logger.info(f"  Fetch: {fetch_time:.0f}s | Save: {save_time:.0f}s | Per stock: {total_time/successful:.2f}s")

        # Log rate limit stats and cache stats
        try:
            from data_fetcher import get_rate_limit_stats, reset_rate_limit_stats, get_cache_stats, get_cache_hit_stats
            stats = get_rate_limit_stats()
            error_rate = (stats['errors_429'] / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
            logger.info(f"Rate limit stats: {stats['errors_429']} 429 errors / {stats['total_requests']} requests ({error_rate:.1f}%)")

            cache_stats = get_cache_stats()
            hit_stats = get_cache_hit_stats()
            hit_rate = (hit_stats['hits'] / (hit_stats['hits'] + hit_stats['misses']) * 100) if (hit_stats['hits'] + hit_stats['misses']) > 0 else 0
            logger.info(f"Cache stats: {hit_stats['hits']} hits, {hit_stats['misses']} misses ({hit_rate:.1f}% hit rate)")
            mem_stats = cache_stats.get('memory', cache_stats)  # Handle both nested and flat formats
            logger.info(f"Cache size: {mem_stats.get('tickers_tracked', 0)} tickers, {mem_stats.get('cached_data_entries', 0)} data entries")

            reset_rate_limit_stats()  # Reset for next scan
        except Exception as e:
            logger.error(f"Rate limit stats error: {e}")

        # Update market snapshot (SPY price, MAs, trend)
        market_db = None
        try:
            from backend.main import update_market_snapshot
            market_db = SessionLocal()
            update_market_snapshot(market_db)
            logger.info("Market snapshot updated")
        except Exception as e:
            logger.error(f"Market snapshot error: {e}")
        finally:
            if market_db:
                market_db.close()

        # Phase 2.5: Auto-record Coiled Spring alerts
        phase_start = time.time()
        _set_phase("coiled_spring", detail="Checking for Coiled Spring candidates...")

        try:
            auto_record_coiled_spring_alerts()
        except Exception as e:
            logger.error(f"Coiled Spring auto-record error: {e}")

        # Phase 2.6: Update CS alert outcomes
        _set_phase("coiled_spring", detail="Updating CS alert outcomes...")

        try:
            update_coiled_spring_outcomes()
        except Exception as e:
            logger.error(f"CS outcome tracking error: {e}")

        # Phase 2.7: Earnings Audit (deep FMP fundamental check for top candidates)
        _set_phase("earnings_audit", detail="Running earnings audit on top candidates...")

        try:
            from backend.earnings_audit import run_earnings_audit
            audit_db = SessionLocal()
            try:
                audit_results = run_earnings_audit(audit_db)
                if audit_results:
                    logger.info(f"Earnings audit: {len(audit_results)} candidates audited")
            finally:
                audit_db.close()
        except Exception as e:
            logger.error(f"Earnings audit error: {e}")

        # Phase 2.8: Update industry group strength rankings
        _set_phase("industry_groups", detail="Computing industry group rankings...")

        try:
            from backend.industry_group import compute_industry_group_rankings, update_stock_group_ranks
            ig_db = SessionLocal()
            try:
                rankings = compute_industry_group_rankings(ig_db)
                if rankings:
                    updated = update_stock_group_ranks(ig_db, rankings)
                    logger.info(f"Industry group rankings: {len(rankings)} groups, {updated} stocks updated")
            finally:
                ig_db.close()
        except Exception as e:
            logger.error(f"Industry group ranking error: {e}")

        # Phase 2.9: Update bear market base-building watchlist (only when SPY < 50MA)
        try:
            from data_fetcher import get_cached_market_direction
            mkt = get_cached_market_direction()
            spy_info = mkt.get('indexes', {}).get('SPY', {}) if mkt else {}
            spy_px = spy_info.get('price', 0)
            spy_50 = spy_info.get('ma_50', 0)
            if spy_px and spy_50 and spy_px < spy_50:
                _set_phase("bear_base", detail="Scanning bear market base candidates...")
                from backend.bear_base import update_bear_base_candidates
                bb_db = SessionLocal()
                try:
                    result = update_bear_base_candidates(bb_db)
                    logger.info(f"Bear base watchlist: {result.get('total', 0)} candidates")
                finally:
                    bb_db.close()
        except Exception as e:
            logger.error(f"Bear base watchlist error: {e}")

        # Phase 3: AI Trading
        _set_phase("ai_trading", detail="Running AI trading cycle...")

        # Run AI trading cycle after scan completes (only during market hours)
        # Iterates over ALL active users' portfolios
        ai_trading_result = {}
        ai_db = None
        try:
            from backend.ai_trader import run_ai_trading_cycle, take_portfolio_snapshot, is_market_open
            from backend.database import AIPortfolioConfig
            ai_db = SessionLocal()
            active_configs = ai_db.query(AIPortfolioConfig).filter(AIPortfolioConfig.is_active == True).all()
            if not active_configs:
                logger.info("No active AI portfolio configs found")
            for cfg in active_configs:
                uid = cfg.user_id or 1
                try:
                    if is_market_open():
                        logger.info(f"Running AI trading cycle for user {uid}...")
                        result = run_ai_trading_cycle(ai_db, user_id=uid)
                        logger.info(f"AI trading (user {uid}): {len(result.get('buys_executed', []))} buys, {len(result.get('sells_executed', []))} sells")
                        ai_trading_result = result  # Keep last for backward compat
                        _record_success("trade_cycle")
                    else:
                        logger.info("Market closed - skipping AI trading and snapshot")
                except Exception as ue:
                    logger.error(f"AI trading error for user {uid}: {ue}")
                    _record_failure("trade_cycle", f"User {uid}: {ue}")
            # For inactive configs, still take snapshot during market hours
            inactive_configs = ai_db.query(AIPortfolioConfig).filter(AIPortfolioConfig.is_active == False).all()
            for cfg in inactive_configs:
                uid = cfg.user_id or 1
                try:
                    if is_market_open():
                        take_portfolio_snapshot(ai_db, user_id=uid)
                except Exception as ue:
                    logger.error(f"Snapshot error for user {uid}: {ue}")
        except Exception as e:
            logger.error(f"AI trading error: {e}")
        finally:
            if ai_db:
                ai_db.close()

        # Phase 3.5: Shadow paper-trading
        # Forward-only telemetry collection for candidate scoring stacks.
        # Wrapped so a shadow failure cannot break the live save path; sandbox
        # rolls back any incidental writes (only ShadowTrade rows survive).
        # See backend/shadow_trader.py + docs/shadow-paper-trading-design.md.
        shadow_db = None
        try:
            from backend.shadow_trader import run_shadow_strategies
            shadow_db = SessionLocal()
            shadow_summary = run_shadow_strategies(shadow_db, analysis_results)
            if shadow_summary.get("strategies_run"):
                logger.info(
                    f"Shadow paper-trading: {shadow_summary['strategies_run']} stack(s), "
                    f"{shadow_summary['total_shadow_trades']} virtual trade(s), "
                    f"{shadow_summary['errors']} error(s)"
                )
        except Exception as e:
            logger.error(f"Shadow paper-trading failed: {e}", exc_info=True)
        finally:
            if shadow_db:
                shadow_db.close()

        # Phase 4: Cleanup old StockScore records
        _set_phase("cleanup", detail="Cleaning up old score records...")

        cleanup_db = None
        try:
            cleanup_db = SessionLocal()
            cleanup_old_stock_scores(cleanup_db, days_to_keep=30)
        except Exception as e:
            logger.error(f"StockScore cleanup failed: {e}")
        finally:
            if cleanup_db:
                cleanup_db.close()

        cleanup_price_cache()

        # Phase 4.5: Post-earnings gap-up detection
        _set_phase("cleanup", detail="Checking post-earnings gap-ups...")

        try:
            from backend.earnings_gapup import find_earnings_gapups, send_gapup_alert
            gu_db = SessionLocal()
            try:
                gapups = find_earnings_gapups(gu_db, days_lookback=3, min_gap_pct=5.0)
                if gapups:
                    send_gapup_alert(gapups)
                    logger.info(f"Post-earnings gap-ups: {len(gapups)} found")
            finally:
                gu_db.close()
        except Exception as e:
            logger.error(f"Gap-up detection error: {e}")

        # Phase 5: Check watchlist alerts
        _set_phase("cleanup", detail="Checking watchlist alerts...")

        try:
            check_watchlist_alerts()
        except Exception as e:
            logger.error(f"Watchlist alert check failed: {e}")

        # Phase 6: Morning briefing email (first scan of the day only)
        try:
            send_morning_briefing_if_due()
        except Exception as e:
            logger.error(f"Morning briefing error: {e}")

        # Phase 7: Scan completion logging
        post_scan_time = time.time() - phase_start
        full_time = time.time() - start_time
        logger.info(f"Scan complete: {successful}/{len(tickers)} stocks in {full_time:.0f}s "
                    f"(fetch={fetch_time:.0f}s save={save_time:.0f}s post={post_scan_time:.0f}s)")
        _record_success("scan")

    except Exception as e:
        logger.error(f"Scan error: {e}")
        _record_failure("scan", str(e))
    finally:
        with _state_lock:
            _scan_config["is_scanning"] = False
            _scan_config["last_scan_end"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            _scan_config["phase"] = None
            _scan_config["phase_detail"] = None
            _scan_config["current_phase"] = None
            _scan_config["phase_current"] = 0
            _scan_config["phase_total"] = 0
            _scan_config["phase_label"] = None


def _refresh_portfolio_prices():
    """Lightweight job: refresh position prices + take snapshot (runs every 15 min during market hours).
    Iterates over all active users' portfolios."""
    try:
        from backend.ai_trader import is_market_open, refresh_ai_portfolio
        from backend.database import SessionLocal, AIPortfolioConfig

        if not is_market_open():
            return

        db = SessionLocal()
        try:
            active_configs = db.query(AIPortfolioConfig).filter(AIPortfolioConfig.is_active == True).all()
            for cfg in active_configs:
                uid = cfg.user_id or 1
                try:
                    result = refresh_ai_portfolio(db, user_id=uid)
                    logger.info(f"Portfolio price refresh (user {uid}): {result.get('message', 'done')}")
                except Exception as ue:
                    logger.error(f"Portfolio price refresh error for user {uid}: {ue}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Portfolio price refresh error: {e}")


def _sync_shadow_strategies_at_boot():
    """Reconcile shadow_strategy_profiles YAML → shadow_strategies table.

    Wraps the helper with logging + error containment so a malformed YAML
    entry (e.g. unknown parent_strategy) never kills the scheduler boot —
    the live scanner must still come up. Failures are logged loud; the
    rest of the harness reads the live DB state regardless.
    """
    try:
        from backend.database import SessionLocal
        from backend.shadow_strategy_sync import sync_shadow_strategies_from_yaml
        logger.info("Syncing shadow_strategy_profiles...")
        db = SessionLocal()
        try:
            result = sync_shadow_strategies_from_yaml(db)
            logger.info(
                "shadow_strategy_profiles sync complete: "
                f"inserted={result['inserted']} updated={result['updated']} "
                f"archived={result['archived']} skipped={result['skipped']}"
            )
        finally:
            db.close()
    except ValueError as ve:
        # Bad YAML (e.g. unknown parent_strategy). Log + skip; existing rows
        # in the DB keep working, scheduler boot continues.
        logger.error(f"shadow_strategy_profiles sync rejected: {ve}")
    except Exception as e:
        logger.error(f"shadow_strategy_profiles sync failed: {e}", exc_info=True)


def start_continuous_scanning(source: str = "sp500", interval_minutes: int = 15):
    """Start continuous scanning with specified interval"""
    _restore_health_from_redis()
    _sync_shadow_strategies_at_boot()
    with _state_lock:
        _scan_config["enabled"] = True
        _scan_config["source"] = source
        _scan_config["interval_minutes"] = interval_minutes

    # Remove existing jobs if any
    if scheduler.get_job("continuous_scan"):
        scheduler.remove_job("continuous_scan")
    if scheduler.get_job("portfolio_price_refresh"):
        scheduler.remove_job("portfolio_price_refresh")

    # Add the scan job
    scheduler.add_job(
        run_continuous_scan,
        IntervalTrigger(minutes=interval_minutes),
        id="continuous_scan",
        name=f"Continuous CANSLIM Scan ({source})",
        replace_existing=True
    )

    # Add portfolio price refresh job (every 5 min for chart data points, matches frontend)
    scheduler.add_job(
        _refresh_portfolio_prices,
        IntervalTrigger(minutes=5),
        id="portfolio_price_refresh",
        name="Portfolio Price Refresh",
        replace_existing=True
    )

    if not scheduler.running:
        scheduler.start()

    # Also start the weekly email job if configured
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config_loader import config
        if config.get('notifications.weekly_email.enabled', True):
            start_weekly_email_job()
    except Exception as e:
        logger.warning(f"Failed to start weekly email job: {e}")

    # Start daily database backup job
    try:
        start_backup_job()
    except Exception as e:
        logger.warning(f"Failed to start backup job: {e}")

    # Start daily ml_predictions outcome backfill job
    try:
        start_ml_backfill_job()
    except Exception as e:
        logger.warning(f"Failed to start ML backfill job: {e}")

    # Start intraday breakout monitor (every 5 min during market hours)
    try:
        start_breakout_monitor_job()
    except Exception as e:
        logger.warning(f"Failed to start breakout monitor: {e}")

    # Start weekly A/B eval snapshot email (Mon 9 AM UTC)
    try:
        start_ab_eval_email_job()
    except Exception as e:
        logger.warning(f"Failed to start A/B eval email job: {e}")

    logger.info(f"Continuous scanning started: {source} every {interval_minutes} minutes")

    # Run first scan immediately
    from threading import Thread
    Thread(target=run_continuous_scan).start()

    return get_scan_status()


def stop_continuous_scanning():
    """Stop continuous scanning"""
    with _state_lock:
        _scan_config["enabled"] = False

    if scheduler.get_job("continuous_scan"):
        scheduler.remove_job("continuous_scan")
    if scheduler.get_job("portfolio_price_refresh"):
        scheduler.remove_job("portfolio_price_refresh")

    logger.info("Continuous scanning stopped")
    return get_scan_status()


def update_scan_config(source: str = None, interval_minutes: int = None):
    """Update scan configuration"""
    with _state_lock:
        if source:
            _scan_config["source"] = source
        if interval_minutes:
            _scan_config["interval_minutes"] = interval_minutes
        enabled = _scan_config["enabled"]
        current_source = _scan_config["source"]
        current_interval = _scan_config["interval_minutes"]

    # If enabled, restart with new config
    if enabled:
        return start_continuous_scanning(current_source, current_interval)

    return get_scan_status()


def send_weekly_performance_email():
    """
    Send weekly performance summary email comparing portfolio to SPY.
    Scheduled to run every Saturday at 9 AM.

    Disabled by default May 7 2026 via email.automated_enabled config flag.
    The cron job still fires (cheaper than removing+re-adding it) but the
    function early-returns when the flag is false.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import config
    if not config.get('email.automated_enabled', False):
        logger.debug("Weekly performance email skipped: email.automated_enabled=false")
        return
    from backend.database import SessionLocal, AIPortfolioSnapshot, AIPortfolioTrade
    from backend.email_utils import send_email
    from datetime import timedelta

    # Check if weekly email is enabled
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import config

    email_config = config.get('notifications.weekly_email', {})
    if not email_config.get('enabled', True):
        logger.info("Weekly email disabled in config")
        return

    db = SessionLocal()
    try:
        # Get snapshots for the past week
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)

        # Get first and last snapshots of the week
        first_snapshot = db.query(AIPortfolioSnapshot).filter(
            AIPortfolioSnapshot.timestamp >= one_week_ago
        ).order_by(AIPortfolioSnapshot.timestamp.asc()).first()

        last_snapshot = db.query(AIPortfolioSnapshot).order_by(
            AIPortfolioSnapshot.timestamp.desc()
        ).first()

        if not first_snapshot or not last_snapshot:
            logger.warning("Not enough snapshots for weekly email")
            return

        # Calculate portfolio return
        start_value = first_snapshot.total_value
        end_value = last_snapshot.total_value
        portfolio_return = ((end_value - start_value) / start_value) * 100 if start_value > 0 else 0

        # Get SPY return for comparison (from last snapshot if available)
        spy_return = getattr(last_snapshot, 'spy_weekly_return', None)
        if spy_return is None:
            # Try to calculate from market data
            try:
                import yfinance as yf
                spy = yf.Ticker("SPY")
                hist = spy.history(period="7d")
                if len(hist) >= 2:
                    spy_return = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            except Exception as e:
                logger.warning(f"Failed to fetch SPY return for weekly report: {e}")
                spy_return = None

        # Get trades from the week
        trades = db.query(AIPortfolioTrade).filter(
            AIPortfolioTrade.executed_at >= one_week_ago
        ).order_by(AIPortfolioTrade.executed_at.desc()).all()

        # Calculate stats
        buy_trades = [t for t in trades if t.action == "BUY"]
        sell_trades = [t for t in trades if t.action == "SELL"]
        total_realized = sum(t.realized_gain or 0 for t in sell_trades)
        winning_sells = len([t for t in sell_trades if (t.realized_gain or 0) > 0])
        win_rate = (winning_sells / len(sell_trades) * 100) if sell_trades else 0

        # Signal attribution for the week
        entry_type_stats = {}
        for t in sell_trades:
            factors = t.signal_factors if hasattr(t, 'signal_factors') and isinstance(t.signal_factors, dict) else {}
            et = factors.get("entry_type", "unknown")
            if et not in entry_type_stats:
                entry_type_stats[et] = {"trades": 0, "wins": 0, "pnl": 0}
            entry_type_stats[et]["trades"] += 1
            entry_type_stats[et]["pnl"] += t.realized_gain or 0
            if (t.realized_gain or 0) > 0:
                entry_type_stats[et]["wins"] += 1

        # Build email content
        vs_spy = ""
        if spy_return is not None:
            diff = portfolio_return - spy_return
            diff_color = "green" if diff > 0 else "red"
            vs_spy = f"""
            <div style="margin: 10px 0;">
                <span style="color: #666;">SPY Return:</span>
                <span style="font-weight: bold;">{spy_return:+.2f}%</span>
            </div>
            <div style="margin: 10px 0;">
                <span style="color: #666;">vs SPY:</span>
                <span style="font-weight: bold; color: {diff_color};">{diff:+.2f}%</span>
            </div>
            """

        trades_html = ""
        if trades:
            trades_html = """
            <h3>Recent Trades</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr style="background: #f5f5f5;">
                    <th style="padding: 8px; text-align: left;">Ticker</th>
                    <th style="padding: 8px; text-align: left;">Action</th>
                    <th style="padding: 8px; text-align: right;">Shares</th>
                    <th style="padding: 8px; text-align: right;">P/L</th>
                </tr>
            """
            for trade in trades[:10]:  # Limit to 10 most recent
                pl_str = ""
                if trade.realized_gain:
                    color = "green" if trade.realized_gain > 0 else "red"
                    pl_str = f'<span style="color: {color};">${trade.realized_gain:+,.0f}</span>'
                trades_html += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px;">{trade.ticker}</td>
                    <td style="padding: 8px;">{trade.action}</td>
                    <td style="padding: 8px; text-align: right;">{trade.shares:.0f}</td>
                    <td style="padding: 8px; text-align: right;">{pl_str}</td>
                </tr>
                """
            trades_html += "</table>"

        # Build attribution rows for email
        attribution_html = ""
        if entry_type_stats:
            attr_rows = ""
            for et, stats in sorted(entry_type_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
                wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
                pnl_color = "green" if stats["pnl"] > 0 else "red"
                attr_rows += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 6px 8px; text-transform: capitalize;">{et}</td>
                    <td style="padding: 6px 8px; text-align: center;">{stats['trades']}</td>
                    <td style="padding: 6px 8px; text-align: center;">{wr:.0f}%</td>
                    <td style="padding: 6px 8px; text-align: right; color: {pnl_color};">${stats['pnl']:+,.0f}</td>
                </tr>"""
            attribution_html = f"""
            <h3>Signal Attribution</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr style="background: #f5f5f5;">
                    <th style="padding: 6px 8px; text-align: left;">Entry Type</th>
                    <th style="padding: 6px 8px; text-align: center;">Trades</th>
                    <th style="padding: 6px 8px; text-align: center;">Win Rate</th>
                    <th style="padding: 6px 8px; text-align: right;">P/L</th>
                </tr>
                {attr_rows}
            </table>"""

        portfolio_color = "green" if portfolio_return > 0 else "red"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .stats {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
                .stat-value {{ font-size: 1.5em; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2 style="margin: 0;">Weekly Performance Summary</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">AI Portfolio Report</p>
            </div>

            <div class="stats">
                <div style="margin: 10px 0;">
                    <span style="color: #666;">Portfolio Value:</span>
                    <span class="stat-value">${end_value:,.0f}</span>
                </div>
                <div style="margin: 10px 0;">
                    <span style="color: #666;">Weekly Return:</span>
                    <span class="stat-value" style="color: {portfolio_color};">{portfolio_return:+.2f}%</span>
                </div>
                {vs_spy}
                <div style="margin: 10px 0;">
                    <span style="color: #666;">Realized P/L:</span>
                    <span style="font-weight: bold;">${total_realized:+,.0f}</span>
                </div>
                <div style="margin: 10px 0;">
                    <span style="color: #666;">Win Rate:</span>
                    <span>{win_rate:.0f}% ({winning_sells}/{len(sell_trades)} trades)</span>
                </div>
            </div>

            {trades_html}

            {attribution_html}

            <p style="color: #666; font-size: 0.9em; margin-top: 20px;">
                Generated by CANSLIM Analyzer
            </p>
        </body>
        </html>
        """

        text_content = f"""Weekly AI Portfolio Performance Summary

Portfolio Value: ${end_value:,.0f}
Weekly Return: {portfolio_return:+.2f}%
{f'SPY Return: {spy_return:+.2f}%' if spy_return else ''}
Realized P/L: ${total_realized:+,.0f}
Win Rate: {win_rate:.0f}%

Trades: {len(buy_trades)} buys, {len(sell_trades)} sells
"""

        subject = f"CANSLIM Weekly: {portfolio_return:+.2f}%{f' (vs SPY {spy_return:+.2f}%)' if spy_return else ''}"

        if send_email(subject, html_content, text_content):
            logger.info(f"Weekly performance email sent: {portfolio_return:+.2f}%")
        else:
            logger.warning("Failed to send weekly performance email")

    except Exception as e:
        logger.error(f"Weekly performance email failed: {e}")
    finally:
        db.close()


def send_weekly_bear_market_report():
    """
    Send weekly bear market report (only when SPY < 50MA).
    Includes: top bases forming, sector rotation shifts, ready-to-buy list.
    """
    from backend.database import SessionLocal
    from data_fetcher import get_cached_market_direction

    # Check if we're in a bear market
    try:
        mkt = get_cached_market_direction()
        spy_info = mkt.get('indexes', {}).get('SPY', {}) if mkt else {}
        spy_px = spy_info.get('price', 0)
        spy_50 = spy_info.get('ma_50', 0)
        if not spy_px or not spy_50 or spy_px >= spy_50:
            logger.info("Bear market report: SPY above 50MA, skipping")
            return
    except Exception as e:
        logger.error(f"Bear market report: market data check failed: {e}")
        return

    db = SessionLocal()
    try:
        from backend.bear_base import get_bear_base_list
        from backend.industry_group import get_group_rotation_summary
        from backend.email_utils import send_bear_market_report_push

        # Gather data
        candidates = get_bear_base_list(db, limit=20)
        rotation = get_group_rotation_summary(db)

        report_data = {
            "bases_forming": len(candidates),
            "improving_groups": rotation.get('improving', [])[:5],
            "deteriorating_groups": rotation.get('deteriorating', [])[:5],
            "top_ready": candidates[:5],
            "spy_price": spy_px,
            "spy_ma50": spy_50,
        }

        # Send push notification
        send_bear_market_report_push(report_data)

        # Send email with more detail
        _send_bear_report_email(report_data, candidates)

        logger.info(f"Bear market report sent: {len(candidates)} bases, "
                    f"{len(rotation.get('improving', []))} improving groups")

    except Exception as e:
        logger.error(f"Bear market report failed: {e}")
    finally:
        db.close()


def _send_bear_report_email(report_data: dict, candidates: list):
    """Send detailed bear market report via email.

    Gated on email.automated_enabled (May 7 2026) — early-return when off.
    Push channel for bear-market readiness lives elsewhere (send_bear_market_report_push).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import config
    if not config.get('email.automated_enabled', False):
        logger.debug("Bear report email skipped: email.automated_enabled=false")
        return
    from backend.email_utils import send_email

    spy_px = report_data.get('spy_price', 0)
    spy_50 = report_data.get('spy_ma50', 0)
    improving = report_data.get('improving_groups', [])
    deteriorating = report_data.get('deteriorating_groups', [])

    # Build candidate rows
    candidate_rows = ""
    for c in candidates[:15]:
        color = "#16a34a" if (c.get('readiness_score', 0)) >= 60 else "#ca8a04" if (c.get('readiness_score', 0)) >= 40 else "#94a3b8"
        candidate_rows += f"""
        <tr>
            <td style="padding:6px 10px;">{c.get('ticker','')}</td>
            <td style="padding:6px 10px;color:{color};font-weight:bold;">{c.get('readiness_score',0):.0f}</td>
            <td style="padding:6px 10px;">{c.get('canslim_score',0):.0f}</td>
            <td style="padding:6px 10px;">{c.get('base_type','')}</td>
            <td style="padding:6px 10px;">{c.get('weeks_in_base',0)}w</td>
            <td style="padding:6px 10px;">{c.get('sector','')[:15]}</td>
        </tr>"""

    # Build rotation rows
    improving_rows = "".join(
        f"<li style='color:#16a34a;'>{g.get('industry','')} (rank {g.get('rank',0)})</li>"
        for g in improving[:5]
    )
    deteriorating_rows = "".join(
        f"<li style='color:#dc2626;'>{g.get('industry','')} (rank {g.get('rank',0)})</li>"
        for g in deteriorating[:5]
    )

    subject = f"Bear Market Report: {len(candidates)} bases forming | SPY ${spy_px:.2f}"

    html_content = f"""
    <!DOCTYPE html><html><head><style>
        body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #dc2626, #7c3aed); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #e2e8f0; padding: 8px 10px; text-align: left; font-size: 0.9em; }}
    </style></head><body>
        <div class="header">
            <h2 style="margin:0;">Weekly Bear Market Report</h2>
            <p style="margin:5px 0 0;opacity:0.9;">SPY ${spy_px:.2f} (50MA ${spy_50:.2f}) | {len(candidates)} bases forming</p>
        </div>
        <div class="card">
            <h3 style="margin-top:0;">Top Base-Building Candidates</h3>
            <table><tr><th>Ticker</th><th>Ready</th><th>Score</th><th>Base</th><th>Weeks</th><th>Sector</th></tr>
            {candidate_rows}</table>
        </div>
        {"<div class='card'><h3 style='margin-top:0;'>Improving Groups</h3><ul>" + improving_rows + "</ul></div>" if improving_rows else ""}
        {"<div class='card'><h3 style='margin-top:0;'>Deteriorating Groups</h3><ul>" + deteriorating_rows + "</ul></div>" if deteriorating_rows else ""}
        <p style="color:#666;font-size:0.85em;">Generated by CANSLIM Analyzer</p>
    </body></html>"""

    text_content = f"""Bear Market Report — SPY ${spy_px:.2f} (50MA ${spy_50:.2f})
{len(candidates)} base-building candidates

Top Candidates:
""" + "\n".join(
        f"  {c.get('ticker','')} ready={c.get('readiness_score',0):.0f} "
        f"score={c.get('canslim_score',0):.0f} {c.get('base_type','')} {c.get('weeks_in_base',0)}w"
        for c in candidates[:10]
    )

    send_email(subject, html_content, text_content)


def start_weekly_email_job():
    """Start the weekly performance email job (Saturday 9 AM) and bear market report (Sunday 6 PM)"""
    from apscheduler.triggers.cron import CronTrigger

    job_id = "weekly_performance_email"

    # Remove existing job if any
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    # Add the job - Saturday at 9 AM
    scheduler.add_job(
        send_weekly_performance_email,
        CronTrigger(day_of_week='sat', hour=9, minute=0),
        id=job_id,
        name="Weekly Performance Email",
        replace_existing=True
    )

    # Bear market report - Sunday at 6 PM (to prepare for Monday)
    bear_job_id = "weekly_bear_market_report"
    if scheduler.get_job(bear_job_id):
        scheduler.remove_job(bear_job_id)

    scheduler.add_job(
        send_weekly_bear_market_report,
        CronTrigger(day_of_week='sun', hour=18, minute=0),
        id=bear_job_id,
        name="Weekly Bear Market Report",
        replace_existing=True
    )

    if not scheduler.running:
        scheduler.start()

    logger.info("Weekly jobs scheduled: performance email (Sat 9 AM), bear report (Sun 6 PM)")


def _run_scheduled_backup():
    """Wrapper for scheduled backup with health tracking."""
    try:
        from backend.backup import perform_backup
        result = perform_backup()
        if result["status"] == "success":
            _record_success("backup")
        else:
            _record_failure("backup", result.get("error", "unknown"))
    except Exception as e:
        logger.error(f"Scheduled backup failed: {e}")
        _record_failure("backup", str(e))


def start_backup_job():
    """Schedule daily database backup at 2 AM UTC."""
    from apscheduler.triggers.cron import CronTrigger

    job_id = "daily_database_backup"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    scheduler.add_job(
        _run_scheduled_backup,
        CronTrigger(hour=2, minute=0),
        id=job_id,
        name="Daily Database Backup",
        replace_existing=True,
    )

    if not scheduler.running:
        scheduler.start()

    logger.info("Daily backup job scheduled (2 AM UTC)")


def _run_ml_backfill():
    """Wrapper for APScheduler — owns its DB session and swallows errors.

    Runs realized backfill first (matches predictions to real buy/sell pairs),
    then counterfactual (synthesizes outcomes for vetoed/un-bought predictions).
    """
    try:
        from backend.ml_backfill import backfill_actual_outcomes, counterfactual_backfill
        realized = backfill_actual_outcomes()
        logger.info(f"ML backfill (realized): {realized}")
        cf = counterfactual_backfill()
        logger.info(f"ML backfill (counterfactual): {cf}")
    except Exception as e:
        logger.error(f"ML backfill job failed: {e}", exc_info=True)


def start_ml_backfill_job():
    """Schedule daily ml_predictions outcome backfill at 3 AM UTC.

    Runs after the 2 AM backup, before US market open. Idempotent — only
    updates rows where actual_outcome IS NULL, so re-runs are cheap.
    Counterfactual fills the rest (vetoed predictions) so the next training
    pool isn't biased toward what the current model already approves of.
    """
    from apscheduler.triggers.cron import CronTrigger

    job_id = "ml_predictions_backfill"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    scheduler.add_job(
        _run_ml_backfill,
        CronTrigger(hour=3, minute=0),
        id=job_id,
        name="ML Predictions Outcome Backfill",
        replace_existing=True,
    )

    if not scheduler.running:
        scheduler.start()

    logger.info("ML backfill job scheduled (3 AM UTC)")


_AB_EVAL_LIVE_STRATEGY = 'nostate_cs_bear'
_AB_EVAL_CUTOFF_DATE = '2026-05-07'


def _run_weekly_ab_eval_email():
    """Wrapper for APScheduler — owns its DB session, swallows errors so a
    single failure doesn't kill the scheduler.

    Fans out one email per delivery target: first the live experiment
    (Approach 2, cutoff 2026-05-07, nostate_cs_bear — the live trading
    strategy for both users), then one per active ShadowStrategy. Shadow
    fan-out is failure-isolated — a single shadow snapshot blowing up
    must not kill the live email or the other shadows.

    Cutoff is hard-coded to the live Approach 2 cutoff for both live and
    shadow during the 2026-05-07 → 2026-06-18 evaluation window; this
    keeps shadow verdicts directly comparable to the live verdict. Step 7
    revisits per-stack cutoffs once real candidate stacks land.

    Disabled when CANSLIM_ENV=test so the test suite doesn't try to send
    real mail through smtp.gmail.com."""
    if os.environ.get("CANSLIM_ENV") == "test":
        logger.debug("Weekly A/B eval email skipped: CANSLIM_ENV=test")
        return
    try:
        from backend.ab_eval_email import send_ab_eval_snapshot
        from backend.database import SessionLocal, ShadowStrategy
        db = SessionLocal()
        try:
            # Live first — this is the verdict-driver for the June 18 read.
            try:
                result = send_ab_eval_snapshot(
                    strategy=_AB_EVAL_LIVE_STRATEGY,
                    cutoff_date=_AB_EVAL_CUTOFF_DATE,
                    db=db,
                )
                logger.info(
                    f"Weekly A/B eval email (live): sent={result['sent']} "
                    f"decision={result['decision']} post_sells={result['post_sell_count']}"
                )
            except Exception as e:
                logger.error(f"Live A/B eval email failed: {e}", exc_info=True)

            # Shadow fan-out — one email per active stack, failure-isolated.
            shadows = db.query(ShadowStrategy).filter(
                ShadowStrategy.archived_at.is_(None)
            ).order_by(ShadowStrategy.id).all()
            for shadow in shadows:
                try:
                    result = send_ab_eval_snapshot(
                        strategy=shadow.name,
                        cutoff_date=_AB_EVAL_CUTOFF_DATE,
                        db=db,
                        source='shadow',
                    )
                    logger.info(
                        f"Weekly A/B eval email (shadow {shadow.name}): "
                        f"sent={result['sent']} decision={result['decision']} "
                        f"post_sells={result['post_sell_count']}"
                    )
                except Exception as e:
                    logger.error(
                        f"Shadow A/B eval email failed for '{shadow.name}': {e}",
                        exc_info=True,
                    )
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Weekly A/B eval email outer failure: {e}", exc_info=True)


def start_ab_eval_email_job():
    """Schedule weekly A/B eval snapshot email — Monday 9 AM UTC.

    Monday morning lands the email before US market open, so the operator
    can react to a REVERT or MARGINAL verdict before the next session
    accumulates more trades under the experiment."""
    from apscheduler.triggers.cron import CronTrigger

    job_id = "weekly_ab_eval_email"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    scheduler.add_job(
        _run_weekly_ab_eval_email,
        CronTrigger(day_of_week='mon', hour=9, minute=0),
        id=job_id,
        name="Weekly A/B Eval Snapshot Email",
        replace_existing=True,
    )

    if not scheduler.running:
        scheduler.start()

    logger.info("Weekly A/B eval email scheduled (Mon 9 AM UTC)")


def start_breakout_monitor_job():
    """Schedule intraday breakout checks every 5 minutes during market hours."""
    job_id = "intraday_breakout_monitor"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    scheduler.add_job(
        _run_breakout_monitor,
        IntervalTrigger(minutes=5),
        id=job_id,
        name="Intraday Breakout Monitor",
        replace_existing=True,
    )

    if not scheduler.running:
        scheduler.start()

    logger.info("Intraday breakout monitor scheduled (every 5 min)")


def _run_breakout_monitor():
    """Wrapper for breakout monitor with error tracking."""
    try:
        from backend.breakout_monitor import check_intraday_breakouts
        check_intraday_breakouts()
    except Exception as e:
        logger.error(f"Breakout monitor error: {e}")

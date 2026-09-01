"""
Post-Earnings Gap-Up Detector

Identifies stocks that gapped up significantly after reporting strong earnings.
These are among the highest-probability CANSLIM entries because they combine:
1. Fundamental catalyst (earnings beat)
2. Institutional validation (gap-up on volume = big money buying)
3. Breakout confirmation (price breaking to new levels on news)

O'Neil research: Many of the biggest winners in market history made their
initial breakout move on an earnings gap-up.
"""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import List
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Per-ticker cooldown to prevent re-alerting the same gap-up every scan cycle.
# In-memory (resets on container restart) — mirrors backend/breakout_monitor.py.
_recent_gapup_alerts: dict = {}
GAPUP_COOLDOWN_HOURS = 24

# Per-ticker gap-check cache: (ticker, earnings_date) -> (fetched_at, raw gap data).
# Stores threshold-free results so any min_gap_pct query reuses them; only
# successful fetches are cached (a failed Yahoo call is never cached as empty).
# The scheduler's scan-cycle run pre-warms this for the UI.
_gap_check_cache: dict = {}
GAP_CACHE_TTL_HOURS = 6
GAP_FETCH_WORKERS = 8


def find_earnings_gapups(
    db: Session,
    days_lookback: int = 7,
    min_gap_pct: float = 5.0,
) -> List[dict]:
    """
    Find stocks that gapped up after earnings within the lookback window.

    Criteria:
    - Reported earnings within last N days
    - Price gapped up >= min_gap_pct% on the earnings day (or next trading day)
    - Volume on gap day was above average
    - Has a positive earnings beat streak

    Args:
        db: Database session
        days_lookback: How many days back to search
        min_gap_pct: Minimum gap-up percentage to qualify

    Returns:
        List of gap-up candidate dicts, sorted by gap size
    """
    from backend.database import Stock

    cutoff_date = date.today() - timedelta(days=days_lookback)

    # Find stocks that recently had earnings (days_to_earnings is small or negative)
    # Stocks with next_earnings_date in the recent past had earnings recently
    stocks = db.query(Stock).filter(
        Stock.current_price > 0,
        Stock.canslim_score != None,
        Stock.next_earnings_date != None,
        Stock.next_earnings_date >= cutoff_date,
        Stock.next_earnings_date <= date.today(),
        Stock.earnings_beat_streak >= 1,  # Must have beaten estimates
    ).all()

    if not stocks:
        return []

    # Fetch gap data concurrently — one Yahoo call per uncached ticker. Done
    # sequentially this took 30s+ for a typical earnings week; the endpoint
    # handler must stay a sync `def` so this runs in FastAPI's threadpool
    # rather than on the event loop.
    from concurrent.futures import ThreadPoolExecutor

    def _safe_check(stock):
        try:
            return _check_gap_up_cached(stock)
        except Exception as e:
            logger.debug(f"Gap-up check failed for {stock.ticker}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=min(GAP_FETCH_WORKERS, len(stocks))) as pool:
        gap_results = list(pool.map(_safe_check, stocks))

    gapups = []

    for stock, gap_data in zip(stocks, gap_results):
        try:
            if gap_data and gap_data['gap_pct'] >= min_gap_pct:
                gapups.append({
                    'ticker': stock.ticker,
                    'name': stock.name,
                    'sector': stock.sector,
                    'industry': stock.industry,
                    'earnings_date': stock.next_earnings_date.isoformat() if stock.next_earnings_date else None,
                    'gap_pct': gap_data['gap_pct'],
                    'volume_ratio': gap_data['volume_ratio'],
                    'current_price': stock.current_price,
                    'canslim_score': stock.canslim_score,
                    'c_score': stock.c_score,
                    'a_score': stock.a_score,
                    'l_score': stock.l_score,
                    'beat_streak': stock.earnings_beat_streak,
                    'earnings_surprise_pct': stock.earnings_surprise_pct,
                    'base_type': stock.base_type,
                    'weeks_in_base': stock.weeks_in_base,
                    'industry_group_rank': stock.industry_group_rank,
                    'rs_3m': stock.rs_3m,
                    'is_actionable': _is_actionable(stock, gap_data),
                })
        except Exception as e:
            logger.debug(f"Gap-up check failed for {stock.ticker}: {e}")

    # Sort by gap size (biggest gaps first)
    gapups.sort(key=lambda g: g['gap_pct'], reverse=True)

    if gapups:
        logger.info(f"Found {len(gapups)} post-earnings gap-ups "
                    f"(>={min_gap_pct}%, last {days_lookback}d)")

    return gapups


def _check_gap_up_cached(stock) -> dict:
    """
    TTL-cached wrapper around _check_gap_up, keyed by (ticker, earnings_date).

    Calls the fetcher with -inf so the cached result is threshold-free (raw
    gap_pct, even negative); callers apply their own min_gap_pct. None results
    (fetch failure / unusable data) are not cached, so transient Yahoo errors
    retry on the next request.
    """
    key = (stock.ticker, stock.next_earnings_date)
    now = datetime.now(timezone.utc)

    hit = _gap_check_cache.get(key)
    if hit and (now - hit[0]) < timedelta(hours=GAP_CACHE_TTL_HOURS):
        return hit[1]

    gap_data = _check_gap_up(stock, float("-inf"))
    if gap_data is not None:
        _gap_check_cache[key] = (now, gap_data)
        # Prune expired entries so the dict stays bounded by recent-earnings names
        cutoff = now - timedelta(hours=GAP_CACHE_TTL_HOURS)
        for k in [k for k, (ts, _) in _gap_check_cache.items() if ts < cutoff]:
            del _gap_check_cache[k]
    return gap_data


def _check_gap_up(stock, min_gap_pct: float) -> dict:
    """
    Check if a stock gapped up on/after its earnings date.
    Uses Yahoo Finance for intraday gap detection.

    Returns dict with gap_pct and volume_ratio, or None.
    """
    import requests

    ticker = stock.ticker
    earnings_date = stock.next_earnings_date

    try:
        # Fetch 10 days of data around earnings date
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1d", "range": "15d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=5)

        if resp.status_code != 200:
            return None

        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None

        timestamps = result[0].get("timestamp", [])
        quote = result[0].get("indicators", {}).get("quote", [{}])[0]
        opens = quote.get("open", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        if not timestamps or not opens or not closes:
            return None

        # Convert timestamps to dates
        from datetime import datetime as dt
        dates = [dt.fromtimestamp(ts).date() for ts in timestamps]

        # Find the trading day on or right after earnings
        earnings_idx = None
        for i, d in enumerate(dates):
            if d >= earnings_date:
                earnings_idx = i
                break

        if earnings_idx is None or earnings_idx == 0:
            return None

        # The reaction gap can land on either day: BMO reporters gap at the
        # open of day E (vs close E-1), but AMC reporters — the MAJORITY of US
        # companies — report after E's close, so the gap is the open of E+1
        # (vs close E). Measuring only day E systematically missed every AMC
        # beat. Evaluate BOTH candidate days and keep the larger gap.
        candidates_idx = [earnings_idx]
        if earnings_idx + 1 < len(opens):
            candidates_idx.append(earnings_idx + 1)

        best = None  # (gap_pct, gap_idx)
        for gi in candidates_idx:
            prior_close = closes[gi - 1]
            gap_open = opens[gi]
            if not prior_close or not gap_open or prior_close <= 0:
                continue
            g = ((gap_open - prior_close) / prior_close) * 100
            if best is None or g > best[0]:
                best = (g, gi)

        if best is None:
            return None

        gap_pct, gap_idx = best
        earnings_open = opens[gap_idx]
        prior_close = closes[gap_idx - 1]  # the close the winning gap is measured against
        earnings_volume = volumes[gap_idx] if gap_idx < len(volumes) else 0

        # Calculate volume ratio (gap-day volume vs average of prior 5 days)
        prior_volumes = [v for v in volumes[max(0, gap_idx-5):gap_idx] if v and v > 0]
        avg_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else 1
        volume_ratio = earnings_volume / avg_volume if avg_volume > 0 and earnings_volume else 1.0

        if gap_pct >= min_gap_pct:
            return {
                'gap_pct': round(gap_pct, 1),
                'volume_ratio': round(volume_ratio, 1),
                'earnings_open': round(earnings_open, 2),
                'prior_close': round(prior_close, 2),
            }

    except Exception as e:
        logger.debug(f"Gap-up detection for {ticker}: {e}")

    return None


def _is_actionable(stock, gap_data: dict) -> bool:
    """
    Determine if a gap-up is actionable (worth buying into).

    Actionable = strong fundamentals + not too extended after gap.
    """
    # Must have decent CANSLIM score
    if (stock.canslim_score or 0) < 55:
        return False
    # Volume confirmation (gap on above-average volume)
    if gap_data.get('volume_ratio', 0) < 1.3:
        return False
    # Multiple beats = pattern, not fluke
    if (stock.earnings_beat_streak or 0) < 2:
        return False
    return True


def send_gapup_alert(gapups: list) -> bool:
    """Push notification for notable post-earnings gap-ups (owner web push +
    in-app row; ntfy retired 2026-09-01).

    Only fires during market hours (gap-ups aren't actionable overnight / on
    weekends). Applies a 24h per-ticker cooldown so each gap-up alerts once.
    """
    if not gapups:
        return False

    try:
        from backend.ai_trader import is_market_open
        from backend.email_utils import create_notification

        if not is_market_open():
            return False

        actionable = [g for g in gapups if g.get('is_actionable')]
        if not actionable:
            return False

        now = datetime.now(timezone.utc)
        cooldown = timedelta(hours=GAPUP_COOLDOWN_HOURS)

        fresh = []
        for g in actionable:
            last = _recent_gapup_alerts.get(g['ticker'])
            if last is None or (now - last) >= cooldown:
                fresh.append(g)

        if not fresh:
            return False

        title = f"Earnings Gap-Up: {len(fresh)} actionable"
        lines = []
        for g in fresh[:5]:
            lines.append(
                f"{g['ticker']} +{g['gap_pct']:.1f}% gap, "
                f"{g['volume_ratio']:.1f}x vol, "
                f"score {g['canslim_score']:.0f}"
            )
        message = "\n".join(lines)

        sent = create_notification(
            1,  # owner — was the global ntfy topic's de-facto audience
            kind="earnings_gapup", title=title, body=message,
            priority="high", tags=["rocket", "chart_with_upwards_trend"],
            data={"url": f"/stock/{fresh[0]['ticker']}" if len(fresh) == 1
                  else "/notifications"},
        )

        if sent:
            for g in fresh:
                _recent_gapup_alerts[g['ticker']] = now
            # Cleanup stale cooldowns so the dict doesn't grow unbounded
            cutoff = now - (cooldown * 2)
            for t in [t for t, ts in _recent_gapup_alerts.items() if ts < cutoff]:
                del _recent_gapup_alerts[t]

        return sent
    except Exception as e:
        logger.warning(f"Gap-up alert failed: {e}")
        return False

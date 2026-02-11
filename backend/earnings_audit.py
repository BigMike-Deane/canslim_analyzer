"""
Earnings Audit: Deep FMP fundamental analysis for buy candidates.

Runs between scan and AI trading phases. Fetches richer FMP data
(analyst targets, earnings beat quality, financial health, insider
conviction, estimate revisions) for top candidates, computes a
fundamental_confidence score (0-100), and stores in EarningsAudit table.

The evaluate_buys() function reads the most recent audit for each
candidate and applies a composite score bonus/penalty.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import aiohttp
from sqlalchemy.orm import Session
from sqlalchemy import desc

from config_loader import config as yaml_config

logger = logging.getLogger(__name__)

# FMP API config (reuse from existing data fetcher)
import os
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

if not FMP_API_KEY:
    try:
        from data_fetcher import FMP_API_KEY as _fmp_key
        FMP_API_KEY = _fmp_key
    except ImportError:
        pass


async def _fetch_json(session: aiohttp.ClientSession, url: str, timeout: int = 15) -> Optional[list]:
    """Fetch JSON from URL with error handling."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.json()
            elif resp.status == 429:
                logger.warning("FMP rate limit hit during earnings audit")
                await asyncio.sleep(2)
            return None
    except Exception as e:
        logger.debug(f"Earnings audit fetch error: {e}")
        return None


async def _audit_single_ticker(
    session: aiohttp.ClientSession,
    ticker: str,
    current_price: float,
) -> Optional[Dict]:
    """
    Fetch and compute fundamental confidence for a single ticker.

    Makes 3 FMP API calls in parallel:
    1. Price target consensus (analyst targets)
    2. Key metrics (ROE, debt/equity, FCF)
    3. Earnings surprises (beat streak, magnitude)

    Plus we reuse insider data already in the Stock model.
    """
    if not FMP_API_KEY:
        return None

    # Parallel FMP fetches
    targets_url = f"{FMP_BASE_URL}/price-target-consensus?symbol={ticker}&apikey={FMP_API_KEY}"
    metrics_url = f"{FMP_BASE_URL}/key-metrics?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
    earnings_url = f"{FMP_BASE_URL}/earnings-surprises?symbol={ticker}&apikey={FMP_API_KEY}"

    targets_data, metrics_data, earnings_data = await asyncio.gather(
        _fetch_json(session, targets_url),
        _fetch_json(session, metrics_url),
        _fetch_json(session, earnings_url),
        return_exceptions=True,
    )

    result = {"ticker": ticker, "price_at_audit": current_price}

    # --- Parse analyst targets ---
    if isinstance(targets_data, list) and targets_data:
        pt = targets_data[0]
        avg_target = pt.get("targetConsensus", 0) or 0
        result["analyst_avg_target"] = avg_target
        result["analyst_high_target"] = pt.get("targetHigh", 0) or 0
        result["analyst_low_target"] = pt.get("targetLow", 0) or 0
        result["analyst_num"] = pt.get("numberOfAnalysts", 0) or 0
        if current_price and current_price > 0 and avg_target > 0:
            result["analyst_upside_pct"] = ((avg_target - current_price) / current_price) * 100
        else:
            result["analyst_upside_pct"] = 0
    else:
        result["analyst_upside_pct"] = 0

    # --- Parse key metrics (financial health) ---
    if isinstance(metrics_data, list) and metrics_data:
        m = metrics_data[0]
        result["roe"] = m.get("returnOnEquity", 0) or 0
        result["debt_to_equity"] = m.get("debtToEquity", 0) or 0
        result["free_cash_flow_per_share"] = m.get("freeCashFlowPerShare", 0) or 0
        result["current_ratio"] = m.get("currentRatio", 0) or 0
    else:
        result["roe"] = 0
        result["debt_to_equity"] = 0
        result["free_cash_flow_per_share"] = 0
        result["current_ratio"] = 0

    # --- Parse earnings surprises ---
    if isinstance(earnings_data, list) and earnings_data:
        # Beat streak
        beat_streak = 0
        beat_magnitudes = []
        for record in earnings_data[:8]:
            estimated = record.get("estimatedEarning", 0)
            actual = record.get("actualEarningResult", 0)
            if estimated and actual and actual > estimated:
                beat_streak += 1
                if abs(estimated) > 0.01:
                    beat_magnitudes.append(((actual - estimated) / abs(estimated)) * 100)
            else:
                break

        result["beat_streak"] = beat_streak
        result["avg_beat_magnitude"] = sum(beat_magnitudes) / len(beat_magnitudes) if beat_magnitudes else 0

        # Most recent beat
        if earnings_data:
            latest = earnings_data[0]
            est = latest.get("estimatedEarning", 0) or 0
            act = latest.get("actualEarningResult", 0) or 0
            if abs(est) > 0.01:
                result["last_beat_pct"] = ((act - est) / abs(est)) * 100
            else:
                result["last_beat_pct"] = 0
    else:
        result["beat_streak"] = 0
        result["avg_beat_magnitude"] = 0
        result["last_beat_pct"] = 0

    return result


def compute_fundamental_confidence(audit_data: Dict, stock_data: Dict = None) -> tuple:
    """
    Compute fundamental_confidence score (0-100) from audit data.

    Components (weighted):
    - Analyst upside potential:  20pts (>20% upside = max)
    - Earnings beat quality:    25pts (long streak + large beats)
    - Financial health:         20pts (ROE, debt, FCF)
    - Insider conviction:       15pts (cluster buys + net value)
    - Estimate revision trend:  20pts (upward EPS/revenue revisions)

    Returns (confidence_score, breakdown_dict).
    """
    config = yaml_config.get('ai_trader.earnings_audit', {})
    weights = config.get('weights', {})

    w_analyst = weights.get('analyst_upside', 0.20)
    w_beat = weights.get('beat_quality', 0.25)
    w_health = weights.get('financial_health', 0.20)
    w_insider = weights.get('insider_conviction', 0.15)
    w_revisions = weights.get('estimate_revisions', 0.20)

    breakdown = {}

    # 1. Analyst upside (0-100 sub-score)
    upside = audit_data.get("analyst_upside_pct", 0) or 0
    num_analysts = audit_data.get("analyst_num", 0) or 0
    if upside >= 30:
        analyst_score = 100
    elif upside >= 20:
        analyst_score = 80
    elif upside >= 10:
        analyst_score = 60
    elif upside >= 5:
        analyst_score = 40
    elif upside > 0:
        analyst_score = 20
    else:
        analyst_score = 0
    # Penalize if very few analysts
    if num_analysts < 3:
        analyst_score *= 0.5
    breakdown["analyst_upside"] = round(analyst_score, 1)

    # 2. Earnings beat quality (0-100 sub-score)
    beat_streak = audit_data.get("beat_streak", 0) or 0
    avg_beat_mag = audit_data.get("avg_beat_magnitude", 0) or 0
    # Streak score: 0 = 0, 1 = 15, 2 = 30, 3 = 50, 4 = 70, 5+ = 85
    streak_scores = {0: 0, 1: 15, 2: 30, 3: 50, 4: 70}
    streak_score = streak_scores.get(beat_streak, 85)
    # Magnitude bonus (big beats matter)
    if avg_beat_mag >= 20:
        mag_bonus = 15
    elif avg_beat_mag >= 10:
        mag_bonus = 10
    elif avg_beat_mag >= 5:
        mag_bonus = 5
    else:
        mag_bonus = 0
    beat_score = min(streak_score + mag_bonus, 100)
    breakdown["beat_quality"] = round(beat_score, 1)

    # 3. Financial health (0-100 sub-score)
    roe = audit_data.get("roe", 0) or 0
    debt_eq = audit_data.get("debt_to_equity", 0) or 0
    fcf_ps = audit_data.get("free_cash_flow_per_share", 0) or 0

    # ROE component (0-50): 17%+ = great for CANSLIM
    if roe >= 0.25:
        roe_score = 50
    elif roe >= 0.17:
        roe_score = 40
    elif roe >= 0.10:
        roe_score = 25
    elif roe > 0:
        roe_score = 10
    else:
        roe_score = 0

    # Debt/Equity component (0-30): lower is better
    if debt_eq <= 0.3:
        debt_score = 30
    elif debt_eq <= 0.5:
        debt_score = 25
    elif debt_eq <= 1.0:
        debt_score = 15
    elif debt_eq <= 2.0:
        debt_score = 5
    else:
        debt_score = 0

    # FCF component (0-20): positive is good
    if fcf_ps > 5:
        fcf_score = 20
    elif fcf_ps > 2:
        fcf_score = 15
    elif fcf_ps > 0:
        fcf_score = 10
    else:
        fcf_score = 0

    health_score = min(roe_score + debt_score + fcf_score, 100)
    breakdown["financial_health"] = round(health_score, 1)

    # 4. Insider conviction (0-100 sub-score)
    # Use stock model data if available (already tracked by scanner)
    insider_net_val = 0
    insider_clusters = 0
    if stock_data:
        insider_net_val = stock_data.get("insider_net_value", 0) or 0
        buy_count = stock_data.get("insider_buy_count", 0) or 0
        insider_clusters = buy_count  # Each distinct insider buy

    if insider_net_val > 500000:
        insider_score = 100
    elif insider_net_val > 100000:
        insider_score = 70
    elif insider_net_val > 10000:
        insider_score = 40
    elif insider_net_val > 0:
        insider_score = 20
    else:
        insider_score = 0
    # Cluster bonus
    if insider_clusters >= 3:
        insider_score = min(insider_score + 20, 100)
    breakdown["insider_conviction"] = round(insider_score, 1)

    # Store insider data on audit result
    audit_data["insider_net_value"] = insider_net_val
    audit_data["insider_cluster_buys"] = insider_clusters

    # 5. Estimate revisions (0-100 sub-score)
    # Use stock model revision data if available
    eps_rev = 0
    rev_trend = "neutral"
    if stock_data:
        eps_rev = stock_data.get("eps_estimate_revision_pct", 0) or 0
        rev_trend = stock_data.get("estimate_revision_trend", "neutral") or "neutral"

    if eps_rev >= 10:
        revision_score = 100
    elif eps_rev >= 5:
        revision_score = 75
    elif eps_rev > 0:
        revision_score = 50
    elif eps_rev > -5:
        revision_score = 25
    else:
        revision_score = 0
    # Trend confirmation
    if rev_trend == "strong_up":
        revision_score = min(revision_score + 15, 100)
    elif rev_trend == "up":
        revision_score = min(revision_score + 5, 100)
    breakdown["estimate_revisions"] = round(revision_score, 1)

    audit_data["eps_revision_pct"] = eps_rev
    audit_data["revenue_revision_pct"] = 0  # Not separately tracked yet

    # Weighted composite
    confidence = (
        analyst_score * w_analyst +
        beat_score * w_beat +
        health_score * w_health +
        insider_score * w_insider +
        revision_score * w_revisions
    )

    return round(confidence, 1), breakdown


def run_earnings_audit(db: Session) -> List[Dict]:
    """
    Main entry point: audit top buy candidates with deep FMP data.

    Called from scheduler between scan and AI trading phases.
    Returns list of audit results for logging.
    """
    from backend.database import Stock, EarningsAudit

    config = yaml_config.get('ai_trader.earnings_audit', {})
    if not config.get('enabled', True):
        logger.info("Earnings audit disabled in config")
        return []

    if not FMP_API_KEY:
        logger.warning("Earnings audit skipped: no FMP_API_KEY")
        return []

    max_candidates = config.get('max_candidates', 30)
    freshness_hours = config.get('freshness_hours', 24)

    # Get top candidates by CANSLIM score (same pool evaluate_buys will consider)
    candidates = db.query(Stock).filter(
        Stock.canslim_score >= 60,  # Wider net than min_score (72) to catch soft zone
        Stock.current_price > 0,
    ).order_by(desc(Stock.canslim_score)).limit(max_candidates).all()

    if not candidates:
        logger.info("Earnings audit: no candidates above threshold")
        return []

    # Filter out recently audited tickers
    cutoff = datetime.now(timezone.utc) - timedelta(hours=freshness_hours)
    recent_audits = db.query(EarningsAudit.ticker).filter(
        EarningsAudit.audited_at >= cutoff,
    ).distinct().all()
    recently_audited = {r.ticker for r in recent_audits}

    tickers_to_audit = [
        s for s in candidates
        if s.ticker not in recently_audited
    ]

    if not tickers_to_audit:
        logger.info(f"Earnings audit: all {len(candidates)} candidates recently audited, skipping")
        return []

    logger.info(f"Earnings audit: auditing {len(tickers_to_audit)} candidates "
                f"({len(recently_audited)} already fresh)")

    # Build stock data lookup for insider/revision data
    stock_data_map = {}
    for stock in tickers_to_audit:
        stock_data_map[stock.ticker] = {
            "insider_net_value": getattr(stock, 'insider_net_value', 0) or 0,
            "insider_buy_count": getattr(stock, 'insider_buy_count', 0) or 0,
            "eps_estimate_revision_pct": getattr(stock, 'eps_estimate_revision_pct', 0) or 0,
            "estimate_revision_trend": getattr(stock, 'estimate_revision_trend', None),
        }

    # Run async audits
    audit_inputs = [(s.ticker, s.current_price or 0) for s in tickers_to_audit]

    async def _run_all():
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                _audit_single_ticker(session, ticker, price)
                for ticker, price in audit_inputs
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

    try:
        loop = asyncio.new_event_loop()
        raw_results = loop.run_until_complete(_run_all())
        loop.close()
    except Exception as e:
        logger.error(f"Earnings audit async error: {e}")
        return []

    # Process results and save to DB
    saved = []
    for result in raw_results:
        if isinstance(result, Exception) or result is None:
            continue

        ticker = result["ticker"]
        stock_data = stock_data_map.get(ticker, {})
        confidence, breakdown = compute_fundamental_confidence(result, stock_data)

        audit = EarningsAudit(
            ticker=ticker,
            analyst_avg_target=result.get("analyst_avg_target"),
            analyst_high_target=result.get("analyst_high_target"),
            analyst_low_target=result.get("analyst_low_target"),
            analyst_num=result.get("analyst_num"),
            analyst_upside_pct=result.get("analyst_upside_pct"),
            beat_streak=result.get("beat_streak"),
            avg_beat_magnitude=result.get("avg_beat_magnitude"),
            last_beat_pct=result.get("last_beat_pct"),
            roe=result.get("roe"),
            debt_to_equity=result.get("debt_to_equity"),
            free_cash_flow_per_share=result.get("free_cash_flow_per_share"),
            current_ratio=result.get("current_ratio"),
            insider_net_value=result.get("insider_net_value"),
            insider_cluster_buys=result.get("insider_cluster_buys"),
            eps_revision_pct=result.get("eps_revision_pct"),
            revenue_revision_pct=result.get("revenue_revision_pct"),
            fundamental_confidence=confidence,
            confidence_breakdown=breakdown,
            price_at_audit=result.get("price_at_audit"),
        )
        db.add(audit)
        saved.append({
            "ticker": ticker,
            "confidence": confidence,
            "breakdown": breakdown,
        })

    try:
        db.commit()
        logger.info(f"Earnings audit: saved {len(saved)} audits "
                    f"(avg confidence: {sum(s['confidence'] for s in saved) / len(saved):.1f})" if saved else "")
    except Exception as e:
        db.rollback()
        logger.error(f"Earnings audit DB commit error: {e}")
        return []

    return saved


def get_latest_audit(db: Session, ticker: str, max_age_hours: int = 24) -> Optional[Dict]:
    """
    Get the most recent earnings audit for a ticker (if fresh enough).

    Used by evaluate_buys() to get the audit bonus.
    Returns dict with fundamental_confidence and breakdown, or None.
    """
    from backend.database import EarningsAudit

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    audit = db.query(EarningsAudit).filter(
        EarningsAudit.ticker == ticker,
        EarningsAudit.audited_at >= cutoff,
    ).order_by(desc(EarningsAudit.audited_at)).first()

    if not audit:
        return None

    return {
        "fundamental_confidence": audit.fundamental_confidence,
        "breakdown": audit.confidence_breakdown or {},
        "audited_at": audit.audited_at.isoformat() + "Z" if audit.audited_at else None,
        "analyst_upside_pct": audit.analyst_upside_pct,
        "beat_streak": audit.beat_streak,
        "roe": audit.roe,
    }


def get_audit_bonus(confidence: Optional[float]) -> int:
    """
    Convert fundamental_confidence to composite score bonus.

    >= 70 → +10
    >= 50 → +5
    <  30 → -5
    None  → 0 (graceful degradation)
    """
    if confidence is None:
        return 0

    config = yaml_config.get('ai_trader.earnings_audit', {})
    thresholds = config.get('bonus_thresholds', {})

    high_conf = thresholds.get('high_confidence', 70)
    high_bonus = thresholds.get('high_bonus', 10)
    med_conf = thresholds.get('medium_confidence', 50)
    med_bonus = thresholds.get('medium_bonus', 5)
    low_conf = thresholds.get('low_confidence', 30)
    low_penalty = thresholds.get('low_penalty', -5)

    if confidence >= high_conf:
        return high_bonus
    elif confidence >= med_conf:
        return med_bonus
    elif confidence < low_conf:
        return low_penalty
    return 0

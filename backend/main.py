"""
CANSLIM Analyzer Web API

FastAPI backend wrapping the existing CANSLIM analysis modules.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text, case
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional, List
import logging

# Add parent directory to path for importing existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import (
    init_db, get_db, Stock, StockScore, PortfolioPosition,
    Watchlist, AnalysisJob, MarketSnapshot,
    AIPortfolioConfig, AIPortfolioPosition, AIPortfolioTrade, AIPortfolioSnapshot,
    BacktestRun, BacktestSnapshot, BacktestTrade, CoiledSpringAlert,
    EarningsAudit
)
from backend.config import settings
from pydantic import BaseModel

# Import existing CANSLIM modules
from canslim_scorer import CANSLIMScorer
from data_fetcher import DataFetcher, get_cached_market_direction, fetch_market_direction_data
from growth_projector import GrowthProjector
from sp500_tickers import get_sp500_tickers, get_russell2000_tickers, get_all_tickers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Request Models ==============

from pydantic import Field, field_validator
import re

def get_latest_market_snapshot(db: Session):
    """Get the most recent market snapshot."""
    return db.query(MarketSnapshot).order_by(desc(MarketSnapshot.date)).first()


def validate_ticker_param(ticker: str) -> str:
    """Validate and normalize a ticker path parameter."""
    ticker = ticker.upper().strip()
    if not ticker or not re.match(r'^[A-Z0-9.\-]{1,10}$', ticker):
        raise HTTPException(status_code=400, detail=f"Invalid ticker format: {ticker}")
    return ticker


class PositionCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    shares: float = Field(..., gt=0, le=1_000_000_000)
    cost_basis: Optional[float] = Field(None, ge=0, le=1_000_000)
    notes: Optional[str] = Field(None, max_length=500)

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.upper().strip()
        if not re.match(r'^[A-Z0-9.\-]+$', v):
            raise ValueError('Ticker must contain only letters, numbers, dots, or hyphens')
        return v

class PositionUpdate(BaseModel):
    shares: Optional[float] = Field(None, gt=0, le=1_000_000_000)
    cost_basis: Optional[float] = Field(None, ge=0, le=1_000_000)
    notes: Optional[str] = Field(None, max_length=500)

class WatchlistCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    notes: Optional[str] = Field(None, max_length=500)
    target_price: Optional[float] = Field(None, gt=0, le=1_000_000)
    alert_score: Optional[float] = Field(None, ge=0, le=100)

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.upper().strip()
        if not re.match(r'^[A-Z0-9.\-]+$', v):
            raise ValueError('Ticker must contain only letters, numbers, dots, or hyphens')
        return v

class ScanRequest(BaseModel):
    tickers: Optional[List[str]] = Field(None, max_length=500)

    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        return [t.upper().strip() for t in v if t and len(t) <= 10]


def get_valid_strategy_names() -> list:
    """Return list of valid strategy profile names from config."""
    from config_loader import config as yaml_config
    profiles = yaml_config.get('strategy_profiles', {})
    return list(profiles.keys())


def validate_strategy_name(strategy: str) -> str:
    """Validate strategy name exists in config. Returns the name or raises HTTPException."""
    valid = get_valid_strategy_names()
    if strategy not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{strategy}'. Valid strategies: {', '.join(sorted(valid))}"
        )
    return strategy


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown"""
    logger.info("Starting CANSLIM Analyzer API...")
    init_db()

    # Start backtest queue worker
    from backend.backtest_queue import backtest_queue
    backtest_queue.start()

    # Re-queue orphaned backtests (pending/running from previous crash)
    from backend.database import SessionLocal
    orphan_db = SessionLocal()
    try:
        orphans = orphan_db.query(BacktestRun).filter(
            BacktestRun.status.in_(["pending", "running"])
        ).order_by(BacktestRun.created_at).all()
        for bt in orphans:
            bt.status = "pending"  # Reset running → pending
            bt.progress_pct = 0
            orphan_db.commit()
            backtest_queue.enqueue(bt.id)
        if orphans:
            logger.info(f"Re-queued {len(orphans)} orphaned backtests: {[bt.id for bt in orphans]}")
    except Exception as e:
        logger.error(f"Failed to re-queue orphaned backtests: {e}")
    finally:
        orphan_db.close()

    # Auto-start scanner after a short delay to allow full startup
    import asyncio
    if os.environ.get("DISABLE_SCHEDULER") == "true":
        logger.info("Scheduler disabled via DISABLE_SCHEDULER env var")
    else:
        async def auto_start_scanner():
            await asyncio.sleep(5)  # Wait for app to fully initialize
            try:
                from backend.scheduler import start_continuous_scanning
                logger.info("Auto-starting scanner: source=all, interval=30 minutes")
                start_continuous_scanning(source="all", interval_minutes=30)
                logger.info("Scanner auto-started successfully")
            except Exception as e:
                logger.error(f"Failed to auto-start scanner: {e}")

        asyncio.create_task(auto_start_scanner())

    yield

    # Shutdown: stop queue worker
    backtest_queue.stop()
    logger.info("Shutting down...")


# ============== App Setup ==============

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Analysis Helpers ==============

# Initialize analysis components
data_fetcher = DataFetcher()
canslim_scorer = CANSLIMScorer(data_fetcher)
growth_projector = GrowthProjector(data_fetcher)


# ============== Shared Constants & Helpers ==============

# Duplicate ticker groups (same company, different share classes)
DUPLICATE_TICKERS = [
    {'GOOGL', 'GOOG'},  # Alphabet Class A vs Class C
    # Add more pairs here if needed (e.g., BRK.A/BRK.B)
]


def adjust_score_for_market(stock, current_m_score: float) -> Optional[float]:
    """Adjust CANSLIM score using current market M score instead of stored M score"""
    if stock.canslim_score is None:
        return None
    stored_m = stock.m_score or 0
    # Replace stored M score with current market M score
    return round(stock.canslim_score - stored_m + current_m_score, 1)


def get_data_freshness(last_updated: datetime) -> dict:
    """
    Calculate data freshness information for display.
    Returns dict with age_minutes, age_text, and is_stale flag.
    """
    if not last_updated:
        return {"age_minutes": None, "age_text": "Unknown", "is_stale": True}

    # Handle timezone-naive datetimes from database (assume UTC)
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)

    age_seconds = (datetime.now(timezone.utc) - last_updated).total_seconds()
    age_minutes = int(age_seconds / 60)

    # Determine staleness (>4 hours = stale based on SCORE_CACHE_HOURS)
    is_stale = age_seconds > settings.SCORE_CACHE_HOURS * 3600

    # Human-readable age
    if age_minutes < 1:
        age_text = "Just now"
    elif age_minutes < 60:
        age_text = f"{age_minutes}m ago"
    elif age_minutes < 1440:  # Less than 24 hours
        hours = age_minutes // 60
        age_text = f"{hours}h ago"
    else:
        days = age_minutes // 1440
        age_text = f"{days}d ago"

    return {
        "age_minutes": age_minutes,
        "age_text": age_text,
        "is_stale": is_stale
    }


def expand_tickers_with_duplicates(tickers: set) -> set:
    """Expand a set of tickers to include all related duplicates"""
    expanded = set(tickers)
    for ticker in list(expanded):
        for group in DUPLICATE_TICKERS:
            if ticker in group:
                expanded.update(group)
    return expanded


def filter_duplicate_stocks(stocks, limit: int):
    """Filter out duplicate tickers, keeping highest scorer from each group"""
    seen_groups = set()
    filtered = []
    for stock in stocks:
        # Check if this ticker belongs to a duplicate group
        ticker_group = None
        for group in DUPLICATE_TICKERS:
            if stock.ticker in group:
                ticker_group = frozenset(group)
                break

        # Skip if we've already seen this group
        if ticker_group and ticker_group in seen_groups:
            continue

        if ticker_group:
            seen_groups.add(ticker_group)

        filtered.append(stock)
        if len(filtered) >= limit:
            break
    return filtered


def get_score_change_from_history(db: Session, stock_id: int) -> Optional[float]:
    """Calculate score change from the last 2 historical StockScore entries"""
    scores = db.query(StockScore).filter(
        StockScore.stock_id == stock_id
    ).order_by(desc(StockScore.date)).limit(2).all()

    if len(scores) >= 2 and scores[0].total_score is not None and scores[1].total_score is not None:
        return round(scores[0].total_score - scores[1].total_score, 1)
    return None


def get_score_trend(db: Session, stock_id: int, days: int = 7) -> dict:
    """
    Analyze score trend over recent history.
    Returns trend direction, magnitude, and consistency.
    """
    from datetime import date, timedelta

    cutoff_date = date.today() - timedelta(days=days)
    scores = db.query(StockScore).filter(
        StockScore.stock_id == stock_id,
        StockScore.date >= cutoff_date
    ).order_by(StockScore.date).all()

    if len(scores) < 2:
        return {"trend": None, "change": None, "consistency": None, "data_points": len(scores)}

    # Get oldest and newest scores
    oldest_score = scores[0].total_score
    newest_score = scores[-1].total_score

    if oldest_score is None or newest_score is None:
        return {"trend": None, "change": None, "consistency": None, "data_points": len(scores)}

    # Calculate total change
    total_change = newest_score - oldest_score

    # Calculate consistency (what % of day-over-day changes match the overall direction)
    if len(scores) >= 3:
        daily_changes = []
        for i in range(1, len(scores)):
            if scores[i].total_score is not None and scores[i-1].total_score is not None:
                daily_changes.append(scores[i].total_score - scores[i-1].total_score)

        if daily_changes and total_change != 0:
            matching_direction = sum(1 for c in daily_changes if (c > 0) == (total_change > 0))
            consistency = matching_direction / len(daily_changes)
        else:
            consistency = 0.5
    else:
        consistency = 0.5

    # Determine trend
    if abs(total_change) < 3:
        trend = "stable"
    elif total_change > 0:
        trend = "improving"
    else:
        trend = "deteriorating"

    return {
        "trend": trend,
        "change": round(total_change, 1),
        "consistency": round(consistency, 2),
        "data_points": len(scores)
    }


def get_score_trends_batch(db: Session, stock_ids: List[int], days: int = 7) -> dict:
    """
    Batch fetch score trends for multiple stocks efficiently.
    Returns dict mapping stock_id -> trend data
    """
    from datetime import date, timedelta

    if not stock_ids:
        return {}

    cutoff_date = date.today() - timedelta(days=days)

    # Fetch all scores for all stocks in one query
    all_scores = db.query(StockScore).filter(
        StockScore.stock_id.in_(stock_ids),
        StockScore.date >= cutoff_date
    ).order_by(StockScore.stock_id, StockScore.date).all()

    # Group by stock_id
    scores_by_stock = {}
    for score in all_scores:
        if score.stock_id not in scores_by_stock:
            scores_by_stock[score.stock_id] = []
        scores_by_stock[score.stock_id].append(score)

    # Calculate trends for each stock
    results = {}
    for stock_id in stock_ids:
        scores = scores_by_stock.get(stock_id, [])

        if len(scores) < 2:
            results[stock_id] = {"trend": None, "change": None, "consistency": None, "data_points": len(scores)}
            continue

        oldest_score = scores[0].total_score
        newest_score = scores[-1].total_score

        if oldest_score is None or newest_score is None:
            results[stock_id] = {"trend": None, "change": None, "consistency": None, "data_points": len(scores)}
            continue

        total_change = newest_score - oldest_score

        # Calculate consistency
        if len(scores) >= 3:
            daily_changes = []
            for i in range(1, len(scores)):
                if scores[i].total_score is not None and scores[i-1].total_score is not None:
                    daily_changes.append(scores[i].total_score - scores[i-1].total_score)

            if daily_changes and total_change != 0:
                matching_direction = sum(1 for c in daily_changes if (c > 0) == (total_change > 0))
                consistency = matching_direction / len(daily_changes)
            else:
                consistency = 0.5
        else:
            consistency = 0.5

        # Determine trend
        if abs(total_change) < 3:
            trend = "stable"
        elif total_change > 0:
            trend = "improving"
        else:
            trend = "deteriorating"

        results[stock_id] = {
            "trend": trend,
            "change": round(total_change, 1),
            "consistency": round(consistency, 2),
            "data_points": len(scores)
        }

    return results


def update_market_snapshot(db: Session, force_refresh: bool = False):
    """
    Update market direction data using multi-index approach (SPY, QQQ, DIA).
    Uses the cached market direction from data_fetcher.
    """
    try:
        # Get multi-index market direction (uses internal cache)
        market_data = get_cached_market_direction(force_refresh=force_refresh)

        if not market_data.get("success"):
            logger.error(f"Failed to fetch market direction: {market_data.get('error')}")
            return

        # Save to database
        today = date.today()
        snapshot = db.query(MarketSnapshot).filter(MarketSnapshot.date == today).first()

        if not snapshot:
            snapshot = MarketSnapshot(date=today)
            db.add(snapshot)

        # Update timestamp
        snapshot.timestamp = datetime.now(timezone.utc)

        # Extract index data
        indexes = market_data.get("indexes", {})

        # SPY data
        spy = indexes.get("SPY", {})
        snapshot.spy_price = spy.get("price", 0)
        snapshot.spy_50_ma = spy.get("ma_50", 0)
        snapshot.spy_200_ma = spy.get("ma_200", 0)
        snapshot.spy_signal = spy.get("signal", 0)

        # QQQ data
        qqq = indexes.get("QQQ", {})
        snapshot.qqq_price = qqq.get("price", 0)
        snapshot.qqq_50_ma = qqq.get("ma_50", 0)
        snapshot.qqq_200_ma = qqq.get("ma_200", 0)
        snapshot.qqq_signal = qqq.get("signal", 0)

        # DIA data
        dia = indexes.get("DIA", {})
        snapshot.dia_price = dia.get("price", 0)
        snapshot.dia_50_ma = dia.get("ma_50", 0)
        snapshot.dia_200_ma = dia.get("ma_200", 0)
        snapshot.dia_signal = dia.get("signal", 0)

        # Combined metrics
        snapshot.weighted_signal = market_data.get("weighted_signal", 0)
        snapshot.market_score = market_data.get("market_score", 7.5)
        snapshot.market_trend = market_data.get("market_trend", "neutral")

        db.commit()
        logger.info(f"Market snapshot updated: trend={snapshot.market_trend}, "
                   f"score={snapshot.market_score}, weighted_signal={snapshot.weighted_signal:.2f}")

    except Exception as e:
        logger.error(f"Error updating market snapshot: {e}")


@app.get("/api/market-direction")
async def get_market_direction(
    refresh: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get current market direction based on SPY, QQQ, and DIA.
    Pass refresh=true to force a fresh fetch from Yahoo.
    """
    # Update database snapshot if requested
    if refresh:
        update_market_snapshot(db, force_refresh=True)

    # Get cached market direction
    market_data = get_cached_market_direction(force_refresh=refresh)

    # Get latest snapshot from database
    latest_snapshot = db.query(MarketSnapshot).order_by(
        desc(MarketSnapshot.date)
    ).first()

    return {
        "success": market_data.get("success", False),
        "market_score": market_data.get("market_score", 7.5),
        "market_trend": market_data.get("market_trend", "neutral"),
        "weighted_signal": market_data.get("weighted_signal", 0),
        "indexes": market_data.get("indexes", {}),
        "last_updated": (latest_snapshot.timestamp.isoformat() + "Z") if latest_snapshot and latest_snapshot.timestamp else None,
        "error": market_data.get("error"),
    }


@app.post("/api/market-direction/refresh")
async def refresh_market_direction(db: Session = Depends(get_db)):
    """Force refresh market direction data from Yahoo Finance."""
    update_market_snapshot(db, force_refresh=True)

    market_data = get_cached_market_direction()

    return {
        "success": market_data.get("success", False),
        "message": "Market direction refreshed",
        "market_score": market_data.get("market_score", 7.5),
        "market_trend": market_data.get("market_trend", "neutral"),
        "indexes": market_data.get("indexes", {}),
    }


def analyze_stock(ticker: str) -> dict:
    """Analyze a single stock and return full results"""
    try:
        # Get stock data (returns StockData object)
        stock_data = data_fetcher.get_stock_data(ticker)
        if not stock_data or not stock_data.is_valid:
            return None

        # Get CANSLIM score (pass StockData object, not ticker string)
        score_obj = canslim_scorer.score_stock(stock_data)
        if not score_obj:
            return None

        # Convert score object to dict with enhanced details (matching scheduler format)
        volume_ratio = getattr(stock_data, 'volume_ratio', 1.0) or 1.0
        inst_pct = getattr(stock_data, 'institutional_holders_pct', 0) or 0

        score_result = {
            "total_score": score_obj.total_score,
            "c": {
                "score": score_obj.c_score,
                "summary": score_obj.c_detail,
                "quarterly_eps": stock_data.quarterly_earnings[:4] if stock_data.quarterly_earnings else [],
                "earnings_surprise_pct": getattr(stock_data, 'earnings_surprise_pct', 0),
            },
            "a": {
                "score": score_obj.a_score,
                "summary": score_obj.a_detail,
                "annual_eps": stock_data.annual_earnings[:3] if stock_data.annual_earnings else [],
                "roe": getattr(stock_data, 'roe', 0) or 0,
            },
            "n": {
                "score": score_obj.n_score,
                "summary": score_obj.n_detail,
                "current_price": stock_data.current_price,
                "week_52_high": getattr(stock_data, 'high_52w', 0),
            },
            "s": {
                "score": score_obj.s_score,
                "summary": score_obj.s_detail,
                "volume_ratio": volume_ratio,
                "avg_volume": getattr(stock_data, 'avg_volume_50d', 0),
            },
            "l": {
                "score": score_obj.l_score,
                "summary": score_obj.l_detail,
            },
            "i": {
                "score": score_obj.i_score,
                "summary": score_obj.i_detail,
                "institutional_pct": inst_pct,
            },
            "m": {
                "score": score_obj.m_score,
                "summary": score_obj.m_detail,
            },
        }

        # Get growth projection (pass StockData and CANSLIMScore objects)
        growth_obj = growth_projector.project_growth(stock_data, score_obj)

        return {
            "ticker": ticker,
            "name": getattr(stock_data, 'name', ticker),
            "sector": getattr(stock_data, 'sector', "Unknown"),
            "industry": getattr(stock_data, 'industry', "") or getattr(stock_data, 'sector', "Unknown"),
            "current_price": getattr(stock_data, 'current_price', 0),
            "market_cap": getattr(stock_data, 'market_cap', 0) or (getattr(stock_data, 'shares_outstanding', 0) * getattr(stock_data, 'current_price', 0)),
            "week_52_high": getattr(stock_data, 'high_52w', 0),
            "week_52_low": getattr(stock_data, 'low_52w', 0),

            # CANSLIM scores (use lowercase keys to match score_result structure)
            "canslim_score": score_result.get("total_score", 0),
            "c_score": score_result.get("c", {}).get("score", 0),
            "a_score": score_result.get("a", {}).get("score", 0),
            "n_score": score_result.get("n", {}).get("score", 0),
            "s_score": score_result.get("s", {}).get("score", 0),
            "l_score": score_result.get("l", {}).get("score", 0),
            "i_score": score_result.get("i", {}).get("score", 0),
            "m_score": score_result.get("m", {}).get("score", 0),

            # Score details for display (using lowercase keys to match scheduler format)
            "score_details": {
                "c": score_result.get("c", {}),
                "a": score_result.get("a", {}),
                "n": score_result.get("n", {}),
                "s": score_result.get("s", {}),
                "l": score_result.get("l", {}),
                "i": score_result.get("i", {}),
                "m": score_result.get("m", {}),
            },

            # Growth projection
            "projected_growth": getattr(growth_obj, 'projected_growth_pct', 0) if growth_obj else 0,
            "growth_confidence": getattr(growth_obj, 'confidence', "low") if growth_obj else "low",
            "growth_details": {
                "momentum": getattr(growth_obj, 'momentum_projection', 0),
                "earnings": getattr(growth_obj, 'earnings_projection', 0),
                "analyst": getattr(growth_obj, 'analyst_projection', 0),
                "analyst_target": getattr(growth_obj, 'analyst_target', 0),
                "analyst_upside": getattr(growth_obj, 'analyst_upside', 0),
            } if growth_obj else {},

            "analyzed_at": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        return None


def save_stock_to_db(db: Session, analysis: dict):
    """Save or update stock analysis in database"""
    stock = db.query(Stock).filter(Stock.ticker == analysis["ticker"]).first()

    if not stock:
        stock = Stock(ticker=analysis["ticker"])
        db.add(stock)

    # Update stock data
    stock.name = analysis["name"]
    stock.sector = analysis["sector"]
    stock.industry = analysis["industry"]
    stock.market_cap = analysis["market_cap"]
    stock.current_price = analysis["current_price"]
    stock.week_52_high = analysis["week_52_high"]
    stock.week_52_low = analysis["week_52_low"]

    stock.canslim_score = analysis["canslim_score"]
    stock.c_score = analysis["c_score"]
    stock.a_score = analysis["a_score"]
    stock.n_score = analysis["n_score"]
    stock.s_score = analysis["s_score"]
    stock.l_score = analysis["l_score"]
    stock.i_score = analysis["i_score"]
    stock.m_score = analysis["m_score"]

    stock.projected_growth = analysis["projected_growth"]
    stock.growth_confidence = analysis["growth_confidence"]
    stock.score_details = analysis.get("score_details")  # Save detailed breakdown with annual_eps, roe, etc.
    stock.last_updated = datetime.now(timezone.utc)

    db.flush()

    # Save score history (one per scan for granular backtesting)
    score_history = StockScore(
        stock_id=stock.id,
        timestamp=datetime.now(timezone.utc),
        date=date.today(),
        total_score=analysis["canslim_score"],
        c_score=analysis["c_score"],
        a_score=analysis["a_score"],
        n_score=analysis["n_score"],
        s_score=analysis["s_score"],
        l_score=analysis["l_score"],
        i_score=analysis["i_score"],
        m_score=analysis["m_score"],
        projected_growth=analysis["projected_growth"],
        current_price=analysis["current_price"],
        week_52_high=analysis.get("week_52_high")
    )
    db.add(score_history)

    db.commit()
    return stock


# ============== Health Check ==============

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    stock_count = db.query(func.count(Stock.id)).scalar() or 0
    portfolio_count = db.query(func.count(PortfolioPosition.id)).scalar() or 0

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "stocks_cached": stock_count,
        "portfolio_positions": portfolio_count,
        "version": settings.VERSION
    }


# ============== Dashboard ==============

@app.get("/api/dashboard")
async def get_dashboard(db: Session = Depends(get_db)):
    """Get dashboard overview data"""
    # Update market snapshot if stale (older than 5 minutes during market hours)
    latest_market = db.query(MarketSnapshot).order_by(
        desc(MarketSnapshot.date)
    ).first()

    # Check if we need to refresh SPY data
    should_refresh = False
    if not latest_market:
        should_refresh = True
    elif latest_market.date < date.today():
        should_refresh = True
    elif latest_market.created_at:
        # Refresh if data is older than 5 minutes
        created_at = latest_market.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
        if age_seconds > 300:  # 5 minutes
            should_refresh = True

    if should_refresh:
        update_market_snapshot(db)
        latest_market = db.query(MarketSnapshot).order_by(
            desc(MarketSnapshot.date)
        ).first()

    # Get current market M score to adjust stock scores dynamically
    current_m_score = latest_market.market_score if latest_market else 0

    # Get top stocks - prioritize those with positive growth projections
    # First: stocks with high scores AND positive projected growth (quality data)
    top_quality_raw = db.query(Stock).filter(
        Stock.canslim_score != None,
        Stock.canslim_score >= 65,
        Stock.projected_growth != None,
        Stock.projected_growth > 0
    ).order_by(desc(Stock.canslim_score)).limit(12).all()

    # Fallback: remaining high-score stocks (may have missing data)
    top_quality_tickers = {s.ticker for s in top_quality_raw}
    top_fallback_raw = db.query(Stock).filter(
        Stock.canslim_score != None,
        Stock.canslim_score >= 65,
        ~Stock.ticker.in_(top_quality_tickers) if top_quality_tickers else True
    ).order_by(desc(Stock.canslim_score)).limit(8).all()

    top_stocks_raw = top_quality_raw + top_fallback_raw
    top_stocks = filter_duplicate_stocks(top_stocks_raw, 10)

    # Get top stocks under $25 - same prioritization
    top_u25_quality_raw = db.query(Stock).filter(
        Stock.canslim_score != None,
        Stock.current_price != None,
        Stock.current_price > 0,
        Stock.current_price <= 25,
        Stock.projected_growth != None,
        Stock.projected_growth > 0
    ).order_by(desc(Stock.canslim_score)).limit(12).all()

    top_u25_quality_tickers = {s.ticker for s in top_u25_quality_raw}
    top_u25_fallback_raw = db.query(Stock).filter(
        Stock.canslim_score != None,
        Stock.current_price != None,
        Stock.current_price > 0,
        Stock.current_price <= 25,
        ~Stock.ticker.in_(top_u25_quality_tickers) if top_u25_quality_tickers else True
    ).order_by(desc(Stock.canslim_score)).limit(8).all()

    top_stocks_under_25_raw = top_u25_quality_raw + top_u25_fallback_raw
    top_stocks_under_25 = filter_duplicate_stocks(top_stocks_under_25_raw, 10)

    # Get score trends for all displayed stocks (batch query for efficiency)
    all_display_stocks = top_stocks + top_stocks_under_25
    stock_ids = [s.id for s in all_display_stocks]
    score_trends = get_score_trends_batch(db, stock_ids, days=7)

    # Get portfolio summary
    positions = db.query(PortfolioPosition).all()
    total_value = sum(p.current_value or 0 for p in positions)
    total_gain = sum(p.gain_loss or 0 for p in positions)

    # Count by recommendation
    buy_count = len([p for p in positions if p.recommendation == "buy"])
    hold_count = len([p for p in positions if p.recommendation == "hold"])
    sell_count = len([p for p in positions if p.recommendation == "sell"])

    # Get stats in a single query (avoids N+1)
    stock_stats_row = db.query(
        func.count(Stock.id).filter(Stock.canslim_score != None).label('total'),
        func.count(Stock.id).filter(Stock.canslim_score >= 80).label('high_score')
    ).first()
    stock_stats = {
        "total_stocks": stock_stats_row.total if stock_stats_row else 0,
        "high_score_count": stock_stats_row.high_score if stock_stats_row else 0
    }
    watchlist_count = db.query(func.count(Watchlist.id)).scalar() or 0

    def get_data_quality(stock):
        """Assess data quality based on available projection data"""
        if stock.growth_confidence == 'high' and stock.projected_growth and stock.projected_growth > 0:
            return 'high'
        elif stock.growth_confidence in ('medium', 'high') and stock.projected_growth is not None:
            return 'medium'
        else:
            return 'low'

    return {
        "top_stocks": [{
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "canslim_score": adjust_score_for_market(s, current_m_score),
            "score_change": s.score_change,  # Updates every scan
            "score_trend": score_trends.get(s.id, {}).get("trend"),  # improving/stable/deteriorating
            "trend_change": score_trends.get(s.id, {}).get("change"),  # 7-day change
            "projected_growth": s.projected_growth,
            "current_price": s.current_price,
            "growth_confidence": s.growth_confidence,
            "data_quality": get_data_quality(s),
            "m_score": current_m_score  # Use current market M score
        } for s in top_stocks],

        "top_stocks_under_25": [{
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "canslim_score": adjust_score_for_market(s, current_m_score),
            "score_change": s.score_change,  # Updates every scan
            "score_trend": score_trends.get(s.id, {}).get("trend"),  # improving/stable/deteriorating
            "trend_change": score_trends.get(s.id, {}).get("change"),  # 7-day change
            "projected_growth": s.projected_growth,
            "current_price": s.current_price,
            "growth_confidence": s.growth_confidence,
            "data_quality": get_data_quality(s),
            "m_score": current_m_score  # Use current market M score
        } for s in top_stocks_under_25],

        "market": {
            "trend": latest_market.market_trend if latest_market else "unknown",
            "score": latest_market.market_score if latest_market else 0,
            "weighted_signal": latest_market.weighted_signal if latest_market else 0,
            # Legacy SPY fields
            "spy_price": latest_market.spy_price if latest_market else 0,
            "spy_50_ma": latest_market.spy_50_ma if latest_market else 0,
            "spy_200_ma": latest_market.spy_200_ma if latest_market else 0,
            # Multi-index data
            "indexes": {
                "SPY": {
                    "price": latest_market.spy_price if latest_market else 0,
                    "ma_50": latest_market.spy_50_ma if latest_market else 0,
                    "ma_200": latest_market.spy_200_ma if latest_market else 0,
                    "signal": latest_market.spy_signal if latest_market else 0,
                },
                "QQQ": {
                    "price": latest_market.qqq_price if latest_market else 0,
                    "ma_50": latest_market.qqq_50_ma if latest_market else 0,
                    "ma_200": latest_market.qqq_200_ma if latest_market else 0,
                    "signal": latest_market.qqq_signal if latest_market else 0,
                },
                "DIA": {
                    "price": latest_market.dia_price if latest_market else 0,
                    "ma_50": latest_market.dia_50_ma if latest_market else 0,
                    "ma_200": latest_market.dia_200_ma if latest_market else 0,
                    "signal": latest_market.dia_signal if latest_market else 0,
                },
            },
            "date": latest_market.date.isoformat() if latest_market else None
        },

        "stats": {
            **stock_stats,  # total_stocks, high_score_count (computed above)
            "portfolio_count": len(positions),
            "watchlist_count": watchlist_count
        },

        "portfolio": {
            "total_value": total_value,
            "total_gain": total_gain,
            "total_gain_pct": (total_gain / (total_value - total_gain) * 100) if total_value > total_gain else 0,
            "positions_count": len(positions),
            "buy_signals": buy_count,
            "hold_signals": hold_count,
            "sell_signals": sell_count
        },

        "last_scan": (lambda dt: dt.isoformat() + "Z" if dt else None)(db.query(func.max(Stock.last_updated)).scalar()),

        # Data freshness summary
        "data_freshness": get_data_freshness(db.query(func.max(Stock.last_updated)).scalar())
    }


# ============== Market Data ==============

@app.get("/api/market")
async def get_market_data(db: Session = Depends(get_db)):
    """Get current market direction data (SPY price, MAs, trend)"""
    latest = get_latest_market_snapshot(db)

    if not latest:
        return {"error": "No market data available", "spy_price": None}

    return {
        "spy_price": latest.spy_price,
        "spy_50_ma": latest.spy_50_ma,
        "spy_200_ma": latest.spy_200_ma,
        "market_score": latest.market_score,
        "market_trend": latest.market_trend,
        "date": latest.date.isoformat(),
        "last_updated": (latest.created_at.isoformat() + "Z") if latest.created_at else None
    }


@app.post("/api/market/refresh")
async def refresh_market_data(db: Session = Depends(get_db)):
    """Force refresh SPY price and moving averages - call this independently of scans"""
    try:
        update_market_snapshot(db)
        latest = get_latest_market_snapshot(db)

        if latest:
            return {
                "message": "Market data refreshed",
                "spy_price": latest.spy_price,
                "spy_50_ma": latest.spy_50_ma,
                "spy_200_ma": latest.spy_200_ma,
                "market_trend": latest.market_trend,
                "market_score": latest.market_score
            }
        return {"message": "Refresh completed but no data available"}
    except Exception as e:
        logger.error(f"Market refresh error: {e}")
        return {"error": str(e)}


@app.get("/api/rate-limit-stats")
async def get_rate_limit_stats():
    """Get FMP API rate limit statistics (429 errors tracked)"""
    from data_fetcher import get_rate_limit_stats
    stats = get_rate_limit_stats()
    return {
        "errors_429": stats["errors_429"],
        "total_requests": stats["total_requests"],
        "error_rate": f"{(stats['errors_429'] / stats['total_requests'] * 100):.1f}%" if stats["total_requests"] > 0 else "0%",
        "last_reset": stats["last_reset"].isoformat()
    }


@app.post("/api/rate-limit-stats/reset")
async def reset_rate_limit_stats():
    """Reset FMP API rate limit statistics"""
    from data_fetcher import reset_rate_limit_stats
    reset_rate_limit_stats()
    return {"message": "Rate limit stats reset"}


# ============== Stock Screener ==============

@app.get("/api/stocks")
async def get_stocks(
    db: Session = Depends(get_db),
    sort_by: str = Query("canslim_score", enum=["canslim_score", "projected_growth", "current_price", "name"]),
    sort_dir: str = Query("desc", enum=["asc", "desc"]),
    sector: Optional[str] = None,
    min_score: Optional[float] = None,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    limit: int = Query(50, le=200),
    offset: int = Query(0)
):
    """Get filtered and sorted stock list"""
    # Get current market M score for dynamic adjustment
    latest_market = get_latest_market_snapshot(db)
    current_m_score = latest_market.market_score if latest_market else 0

    query = db.query(Stock).filter(Stock.canslim_score != None)

    # Apply filters
    if sector:
        query = query.filter(Stock.sector == sector)
    if min_score:
        query = query.filter(Stock.canslim_score >= min_score)
    if max_price:
        query = query.filter(Stock.current_price <= max_price)
    if min_price:
        query = query.filter(Stock.current_price >= min_price)

    # Apply sorting
    sort_column = getattr(Stock, sort_by, Stock.canslim_score)
    if sort_dir == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column)

    # Get total count
    total = query.count()

    # Apply pagination - fetch extra to allow for duplicate filtering
    stocks_raw = query.offset(offset).limit(limit + 10).all()

    # Filter duplicate tickers (GOOG/GOOGL) - keep highest scorer
    stocks = filter_duplicate_stocks(stocks_raw, limit)

    return {
        "stocks": [{
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "industry": s.industry,
            "canslim_score": adjust_score_for_market(s, current_m_score),
            "projected_growth": s.projected_growth,
            "growth_confidence": s.growth_confidence,
            "current_price": s.current_price,
            "market_cap": s.market_cap,
            "week_52_high": s.week_52_high,
            "week_52_low": s.week_52_low,
            "c_score": s.c_score,
            "a_score": s.a_score,
            "n_score": s.n_score,
            "s_score": s.s_score,
            "l_score": s.l_score,
            "i_score": s.i_score,
            "m_score": current_m_score,  # Use current market M score
            "last_updated": (s.last_updated.isoformat() + "Z") if s.last_updated else None
        } for s in stocks],
        "total": total,
        "limit": limit,
        "offset": offset,
        "current_m_score": current_m_score
    }


@app.get("/api/stocks/search")
async def search_stocks(
    q: str = Query(..., min_length=1, max_length=10),
    limit: int = Query(8, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """Quick search stocks by ticker or name prefix."""
    q = q.upper().strip()
    results = db.query(
        Stock.ticker, Stock.name, Stock.canslim_score, Stock.sector
    ).filter(
        (Stock.ticker.ilike(f"{q}%")) | (Stock.name.ilike(f"%{q}%"))
    ).order_by(
        # Exact ticker match first, then by score
        case((Stock.ticker == q, 0), else_=1),
        desc(Stock.canslim_score)
    ).limit(limit).all()

    return [
        {"ticker": r.ticker, "name": r.name, "score": r.canslim_score, "sector": r.sector}
        for r in results
    ]


@app.get("/api/stocks/sectors")
async def get_sectors(db: Session = Depends(get_db)):
    """Get list of available sectors"""
    sectors = db.query(Stock.sector).distinct().filter(Stock.sector != None).all()
    return [s[0] for s in sectors if s[0]]


@app.get("/api/top-growth-stocks")
async def get_top_growth_stocks(
    db: Session = Depends(get_db),
    limit: int = Query(10, le=50)
):
    """
    Get top stocks by Growth Mode score.
    Growth Mode scoring is designed for pre-revenue/high-growth companies.
    """
    # Get current market M score
    latest_market = get_latest_market_snapshot(db)
    current_m_score = latest_market.market_score if latest_market else 0

    # Query stocks with growth_mode_score (fetch extra to allow for duplicate filtering)
    stocks_raw = db.query(Stock).filter(
        Stock.growth_mode_score.isnot(None),
        Stock.growth_mode_score >= 50,  # Minimum threshold
        Stock.current_price > 0
    ).order_by(
        desc(Stock.growth_mode_score)
    ).limit(limit * 2).all()

    # Filter duplicates (e.g., GOOG/GOOGL) - keep highest scorer
    stocks = filter_duplicate_stocks(stocks_raw, limit)

    return {
        "stocks": [{
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "growth_mode_score": s.growth_mode_score,
            "growth_mode_details": s.growth_mode_details,
            "canslim_score": s.canslim_score,
            "is_growth_stock": s.is_growth_stock,
            "projected_growth": s.projected_growth,
            "growth_confidence": s.growth_confidence,
            "current_price": s.current_price,
            "market_cap": s.market_cap,
            "week_52_high": s.week_52_high,
            "revenue_growth_pct": s.revenue_growth_pct,
            "is_breaking_out": s.is_breaking_out,
            "volume_ratio": s.volume_ratio if s.volume_ratio is not None else 1.0,
            "base_type": s.base_type,
            "weeks_in_base": s.weeks_in_base,
            "days_to_earnings": getattr(s, 'days_to_earnings', None),
            "earnings_beat_streak": getattr(s, 'earnings_beat_streak', None),
            "institutional_holders_pct": (s.score_details or {}).get('i', {}).get('institutional_pct') if s.score_details else None,
            "c_score": s.c_score,
            "l_score": s.l_score,
            "last_updated": (s.last_updated.isoformat() + "Z") if s.last_updated else None
        } for s in stocks],
        "total": len(stocks),
        "current_m_score": current_m_score
    }


@app.get("/api/stocks/breaking-out")
async def get_breaking_out_stocks(
    db: Session = Depends(get_db),
    limit: int = Query(10, le=50)
):
    """
    Get stocks that are currently breaking out or near breakout.
    These are high-probability buy points in CANSLIM methodology.

    Includes:
    1. Stocks flagged as is_breaking_out (price near pivot with volume)
    2. Fallback: Stocks within 5% of 52-week high with decent volume/score
    """
    # Get current market M score for consistent scoring
    latest_market = get_latest_market_snapshot(db)
    current_m_score = latest_market.market_score if latest_market else 0

    # First: Get stocks with is_breaking_out flag
    # Minimal filters - trust the is_breaking_out flag from the scanner
    breakout_stocks = db.query(Stock).filter(
        Stock.is_breaking_out == True,
        Stock.canslim_score >= 50,  # Lowered threshold
        Stock.current_price > 0
    ).order_by(
        # Prioritize stocks WITH base patterns first, then by score
        desc(Stock.base_type != 'none'),
        desc(Stock.base_type != None),
        desc(Stock.canslim_score)
    ).limit(limit * 2).all()

    # Filter duplicates and return only stocks with is_breaking_out=True
    # No fallback - the is_breaking_out flag is the source of truth
    # If there aren't many breakouts, that's accurate (don't pad with extended stocks)
    stocks = filter_duplicate_stocks(breakout_stocks, limit)

    return {
        "stocks": [{
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "canslim_score": adjust_score_for_market(s, current_m_score),  # Use market-adjusted score
            "growth_mode_score": s.growth_mode_score,
            "current_price": s.current_price,
            "week_52_high": s.week_52_high,
            "base_type": s.base_type,
            "weeks_in_base": s.weeks_in_base,
            "breakout_volume_ratio": s.breakout_volume_ratio or s.volume_ratio or 1.0,
            "volume_ratio": s.volume_ratio if s.volume_ratio is not None else 1.0,  # Default to 1.0 for NULL
            "projected_growth": s.projected_growth,
            "is_breaking_out": s.is_breaking_out or False
        } for s in stocks],
        "total": len(stocks)
    }


# ============== Insider Sentiment ==============

@app.get("/api/insider-sentiment")
async def get_insider_sentiment(
    sentiment: str = Query("bullish", regex="^(bullish|bearish|all)$"),
    min_score: float = Query(40),
    sort_by: str = Query("insider_net_value", regex="^(insider_net_value|insider_buy_count|canslim_score|insider_buy_value)$"),
    sort_dir: str = Query("desc", regex="^(asc|desc)$"),
    sector: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Get stocks ranked by insider trading activity with CANSLIM score filter"""
    from sqlalchemy import nullslast

    latest_market = db.query(MarketSnapshot).order_by(
        desc(MarketSnapshot.date)
    ).first()
    current_m_score = latest_market.market_score if latest_market else 0

    # Base query: must have insider data and meet minimum score
    base_filter = [
        Stock.canslim_score >= min_score,
        Stock.insider_updated_at != None,
    ]

    # Sentiment filter
    if sentiment != "all":
        base_filter.append(Stock.insider_sentiment == sentiment)

    # Sector filter
    if sector:
        base_filter.append(Stock.sector == sector)

    # Summary stats (across all matching stocks, ignoring pagination)
    summary_q = db.query(
        func.count(Stock.id).filter(Stock.insider_sentiment == "bullish").label("total_bullish"),
        func.count(Stock.id).filter(Stock.insider_sentiment == "bearish").label("total_bearish"),
        func.count(Stock.id).label("total_with_data"),
        func.sum(Stock.insider_net_value).label("net_insider_value"),
        func.avg(
            case(
                (Stock.insider_sentiment == "bullish", Stock.canslim_score),
                else_=None,
            )
        ).label("avg_score_bullish"),
    ).filter(
        Stock.canslim_score >= min_score,
        Stock.insider_updated_at != None,
    ).first()

    # Get distinct sectors that have insider data
    sectors = [
        r[0] for r in db.query(Stock.sector).filter(
            Stock.canslim_score >= min_score,
            Stock.insider_updated_at != None,
            Stock.sector != None,
        ).distinct().order_by(Stock.sector).all()
    ]

    # Sort column mapping
    sort_col_map = {
        "insider_net_value": Stock.insider_net_value,
        "insider_buy_count": Stock.insider_buy_count,
        "canslim_score": Stock.canslim_score,
        "insider_buy_value": Stock.insider_buy_value,
    }
    sort_col = sort_col_map[sort_by]
    order = desc(sort_col) if sort_dir == "desc" else sort_col.asc()

    # Main query with pagination (fetch extra for dedup)
    stocks_raw = db.query(Stock).filter(
        *base_filter
    ).order_by(
        nullslast(order)
    ).limit((limit + offset) * 2).all()

    stocks = filter_duplicate_stocks(stocks_raw, limit + offset)
    paginated = stocks[offset:offset + limit]

    # Count total for pagination
    total = db.query(func.count(Stock.id)).filter(*base_filter).scalar() or 0

    summary = {
        "total_bullish": summary_q.total_bullish or 0,
        "total_bearish": summary_q.total_bearish or 0,
        "total_with_data": summary_q.total_with_data or 0,
        "net_insider_value": float(summary_q.net_insider_value or 0),
        "avg_score_bullish": round(float(summary_q.avg_score_bullish or 0), 1),
        "sectors": sectors,
    }

    return {
        "summary": summary,
        "stocks": [{
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "canslim_score": adjust_score_for_market(s, current_m_score),
            "current_price": s.current_price,
            "market_cap": s.market_cap,
            "projected_growth": s.projected_growth,
            "insider_sentiment": s.insider_sentiment,
            "insider_buy_count": s.insider_buy_count,
            "insider_sell_count": s.insider_sell_count,
            "insider_net_value": s.insider_net_value,
            "insider_buy_value": s.insider_buy_value,
            "insider_sell_value": s.insider_sell_value,
            "insider_largest_buy": s.insider_largest_buy,
            "insider_largest_buyer_title": s.insider_largest_buyer_title,
            "insider_updated_at": (s.insider_updated_at.isoformat() + "Z") if s.insider_updated_at else None,
            "short_interest_pct": s.short_interest_pct,
            "base_type": s.base_type,
            "is_breaking_out": s.is_breaking_out or False,
        } for s in paginated],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ============== Single Stock Analysis ==============

@app.get("/api/stocks/{ticker}")
async def get_stock(ticker: str, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """Get detailed stock analysis with background refresh for stale data"""
    from backend.database import SessionLocal

    ticker = validate_ticker_param(ticker)

    # Check cache first
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()

    # Determine if cache is stale
    def is_cache_stale(last_updated):
        if not last_updated:
            return False
        lu = last_updated if last_updated.tzinfo else last_updated.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - lu).total_seconds() > settings.SCORE_CACHE_HOURS * 3600

    cache_stale = not stock or is_cache_stale(stock.last_updated)

    if cache_stale:
        if stock and background_tasks:
            # STALE: Return cached data immediately, refresh in background
            logger.info(f"Cache stale for {ticker}, triggering background refresh")

            def refresh_stock_background():
                refresh_db = SessionLocal()
                try:
                    analysis = analyze_stock(ticker)
                    if analysis:
                        save_stock_to_db(refresh_db, analysis)
                        refresh_db.commit()
                        logger.info(f"Background refresh completed for {ticker}")
                except Exception as e:
                    logger.error(f"Background refresh failed for {ticker}: {e}")
                finally:
                    refresh_db.close()

            background_tasks.add_task(refresh_stock_background)
            # Continue to return stale data below
        else:
            # MISSING: Must block and fetch (no cached data to return)
            analysis = analyze_stock(ticker)
            if not analysis:
                raise HTTPException(status_code=404, detail=f"Could not analyze stock {ticker}")
            stock = save_stock_to_db(db, analysis)

    # Get score history
    history = db.query(StockScore).filter(
        StockScore.stock_id == stock.id
    ).order_by(StockScore.date.desc()).limit(30).all()

    return {
        "ticker": stock.ticker,
        "name": stock.name,
        "sector": stock.sector,
        "industry": stock.industry,
        "current_price": stock.current_price,
        "market_cap": stock.market_cap,
        "week_52_high": stock.week_52_high,
        "week_52_low": stock.week_52_low,

        "canslim_score": stock.canslim_score,
        "c_score": stock.c_score,
        "a_score": stock.a_score,
        "n_score": stock.n_score,
        "s_score": stock.s_score,
        "l_score": stock.l_score,
        "i_score": stock.i_score,
        "m_score": stock.m_score,
        "scores": {
            "C": {"score": stock.c_score, "max": 15, "label": "Current Earnings"},
            "A": {"score": stock.a_score, "max": 15, "label": "Annual Earnings"},
            "N": {"score": stock.n_score, "max": 15, "label": "New Highs"},
            "S": {"score": stock.s_score, "max": 15, "label": "Supply/Demand"},
            "L": {"score": stock.l_score, "max": 15, "label": "Leader/Laggard"},
            "I": {"score": stock.i_score, "max": 10, "label": "Institutional"},
            "M": {"score": stock.m_score, "max": 15, "label": "Market Direction"},
        },
        "score_details": stock.score_details,  # Detailed breakdown for each component

        "projected_growth": stock.projected_growth,
        "growth_confidence": stock.growth_confidence,

        # Growth Mode scoring
        "is_growth_stock": stock.is_growth_stock,
        "growth_mode_score": stock.growth_mode_score,
        "growth_mode_details": stock.growth_mode_details,

        # Enhanced earnings
        "eps_acceleration": stock.eps_acceleration,
        "earnings_surprise_pct": stock.earnings_surprise_pct,
        "revenue_growth_pct": stock.revenue_growth_pct,
        "quarterly_earnings": stock.quarterly_earnings if stock.quarterly_earnings else [],
        "annual_earnings": stock.annual_earnings if stock.annual_earnings else [],
        "quarterly_revenue": stock.quarterly_revenue if stock.quarterly_revenue else [],

        # Technical analysis
        "volume_ratio": stock.volume_ratio if stock.volume_ratio is not None else 1.0,
        "weeks_in_base": stock.weeks_in_base,
        "base_type": stock.base_type,
        "is_breaking_out": stock.is_breaking_out,
        "breakout_volume_ratio": stock.breakout_volume_ratio or stock.volume_ratio or 1.0,

        # Earnings catalyst / Coiled Spring data
        "days_to_earnings": getattr(stock, 'days_to_earnings', None),
        "earnings_beat_streak": getattr(stock, 'earnings_beat_streak', None),
        "institutional_holders_pct": (stock.score_details or {}).get('i', {}).get('institutional_pct') if stock.score_details else None,

        # Insider trading signals
        "insider_buy_count": stock.insider_buy_count,
        "insider_sell_count": stock.insider_sell_count,
        "insider_net_shares": stock.insider_net_shares,
        "insider_sentiment": stock.insider_sentiment,
        "insider_updated_at": stock.insider_updated_at.isoformat() if stock.insider_updated_at else None,

        # Short interest
        "short_interest_pct": stock.short_interest_pct,
        "short_ratio": stock.short_ratio,
        "short_updated_at": stock.short_updated_at.isoformat() if stock.short_updated_at else None,

        "score_history": [{
            "date": h.date.isoformat(),
            "total_score": h.total_score,
            "price": h.current_price,
            "projected_growth": h.projected_growth
        } for h in reversed(history)],

        "last_updated": (stock.last_updated.isoformat() + "Z") if stock.last_updated else None,

        # Data freshness indicators
        "data_freshness": get_data_freshness(stock.last_updated)
    }


@app.post("/api/stocks/{ticker}/refresh")
async def refresh_stock(ticker: str, db: Session = Depends(get_db)):
    """Force refresh a stock's analysis"""
    ticker = validate_ticker_param(ticker)

    analysis = analyze_stock(ticker)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Could not analyze stock {ticker}")

    stock = save_stock_to_db(db, analysis)

    return {"message": f"Refreshed {ticker}", "canslim_score": stock.canslim_score}


# ============== Batch Analysis ==============

@app.post("/api/analyze/scan")
async def start_scan(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    data: Optional[ScanRequest] = None,
    source: str = Query("sp500", enum=["sp500", "top50", "russell", "all"]),
    max_price: Optional[float] = None
):
    """Start a background scan of stocks"""
    # If specific tickers provided in body, use those
    if data and data.tickers:
        tickers = [t.upper() for t in data.tickers]
    # Otherwise get ticker list based on source
    elif source == "sp500":
        tickers = get_sp500_tickers()
    elif source == "top50":
        tickers = get_sp500_tickers()[:50]  # First 50 S&P 500
    elif source == "russell":
        tickers = get_russell2000_tickers()
    elif source == "all":
        tickers = get_all_tickers()
    else:
        tickers = get_sp500_tickers()

    # Create job record
    job = AnalysisJob(
        job_type=f"scan_{source}",
        status="running",
        tickers_total=len(tickers),
        started_at=datetime.now(timezone.utc)
    )
    db.add(job)
    db.commit()

    # Run analysis in background with parallel processing
    def run_scan():
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        scan_db = SessionLocal()
        processed = 0
        successful = 0
        lock = threading.Lock()

        def process_single_stock(ticker):
            """Process a single stock - runs in thread pool"""
            import time
            import random
            nonlocal processed, successful
            # Delay to stay under FMP's 300 calls/min limit (4 calls/stock)
            # Using 2.5-4.0s with 4 workers = ~60-90 stocks/min, well under limit
            time.sleep(random.uniform(2.5, 4.0))
            thread_db = SessionLocal()
            try:
                analysis = analyze_stock(ticker)
                if analysis:
                    # Apply price filter
                    if max_price and analysis["current_price"] > max_price:
                        return ticker, False
                    save_stock_to_db(thread_db, analysis)
                    logger.info(f"Scanned {ticker}: score={analysis.get('canslim_score', 0):.1f}")
                    return ticker, True
                return ticker, False
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                return ticker, False
            finally:
                thread_db.close()

        try:
            # Process stocks in parallel with 4 workers (reduces 429 errors)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_single_stock, t): t for t in tickers}

                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        _, was_successful = future.result()
                        with lock:
                            processed += 1
                            if was_successful:
                                successful += 1

                            # Update progress every few stocks
                            if processed % 4 == 0 or processed == len(tickers):
                                scan_job = scan_db.query(AnalysisJob).filter(AnalysisJob.id == job.id).first()
                                scan_job.tickers_processed = processed
                                scan_db.commit()
                    except Exception as e:
                        logger.error(f"Future error for {ticker}: {e}")
                        with lock:
                            processed += 1

            # Mark complete
            scan_job = scan_db.query(AnalysisJob).filter(AnalysisJob.id == job.id).first()
            scan_job.status = "completed"
            scan_job.completed_at = datetime.now(timezone.utc)
            scan_db.commit()

        except Exception as e:
            scan_job = scan_db.query(AnalysisJob).filter(AnalysisJob.id == job.id).first()
            scan_job.status = "failed"
            scan_job.error_message = str(e)
            scan_db.commit()
        finally:
            scan_db.close()

    # Need to import SessionLocal for background task
    from backend.database import SessionLocal
    background_tasks.add_task(run_scan)

    return {
        "job_id": job.id,
        "status": "started",
        "tickers_total": len(tickers),
        "message": f"Scanning {len(tickers)} stocks from {source}"
    }


@app.get("/api/analyze/jobs/{job_id}")
async def get_job_status(job_id: int, db: Session = Depends(get_db)):
    """Get status of an analysis job"""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.id,
        "type": job.job_type,
        "status": job.status,
        "tickers_processed": job.tickers_processed or 0,
        "tickers_total": job.tickers_total or 0,
        "percent": round(job.tickers_processed / job.tickers_total * 100, 1) if job.tickers_total else 0,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error_message
    }


# ============== Continuous Scanning ==============

from backend.scheduler import (
    get_scan_status, start_continuous_scanning,
    stop_continuous_scanning, update_scan_config
)

@app.get("/api/scanner/status")
async def get_scanner_status():
    """Get continuous scanner status"""
    return get_scan_status()


@app.post("/api/scanner/start")
async def start_scanner(
    source: str = Query("sp500", enum=["sp500", "top50", "russell", "all"]),
    interval: int = Query(15, ge=5, le=120, description="Scan interval in minutes")
):
    """Start continuous scanning"""
    return start_continuous_scanning(source=source, interval_minutes=interval)


@app.post("/api/scanner/stop")
async def stop_scanner():
    """Stop continuous scanning"""
    return stop_continuous_scanning()


@app.patch("/api/scanner/config")
async def update_scanner_config(
    source: str = Query(None, enum=["sp500", "top50", "russell", "all"]),
    interval: int = Query(None, ge=5, le=120)
):
    """Update scanner configuration"""
    return update_scan_config(source=source, interval_minutes=interval)


# ============== Portfolio ==============

@app.get("/api/portfolio")
async def get_portfolio(db: Session = Depends(get_db)):
    """Get all portfolio positions with trend data"""
    positions = db.query(PortfolioPosition).all()

    total_value = sum(p.current_value or 0 for p in positions)
    total_cost = sum((p.cost_basis or 0) * (p.shares or 0) for p in positions)
    total_gain = total_value - total_cost

    # Get stock IDs for positions to fetch trend data
    tickers = [p.ticker for p in positions]
    stocks_by_ticker = {}
    if tickers:
        stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all()
        stocks_by_ticker = {s.ticker: s for s in stocks}
        stock_ids = [s.id for s in stocks]
        score_trends = get_score_trends_batch(db, stock_ids, days=7) if stock_ids else {}
    else:
        score_trends = {}

    def get_data_quality(stock):
        """Determine data quality based on growth confidence"""
        if not stock:
            return "low"
        if stock.growth_confidence == 'high' and stock.projected_growth and stock.projected_growth > 0:
            return "high"
        elif stock.growth_confidence in ('medium', 'high') and stock.projected_growth is not None:
            return "medium"
        return "low"

    position_data = []
    for p in positions:
        stock = stocks_by_ticker.get(p.ticker)
        trend_info = score_trends.get(stock.id, {}) if stock else {}
        position_data.append({
            "id": p.id,
            "ticker": p.ticker,
            "shares": p.shares,
            "cost_basis": p.cost_basis,
            "current_price": p.current_price,
            "current_value": p.current_value,
            "gain_loss": p.gain_loss,
            "gain_loss_pct": p.gain_loss_pct,
            "recommendation": p.recommendation,
            # CANSLIM score
            "canslim_score": p.canslim_score,
            "score_change": p.score_change,
            "score_trend": trend_info.get("trend"),  # improving/stable/deteriorating
            "trend_change": trend_info.get("change"),  # 7-day change
            # Growth Mode scores (from Stock table)
            "is_growth_stock": stock.is_growth_stock if stock else False,
            "growth_mode_score": stock.growth_mode_score if stock else None,
            "data_quality": get_data_quality(stock),
            "notes": p.notes
        })

    return {
        "positions": position_data,
        "summary": {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_gain": total_gain,
            "total_gain_pct": (total_gain / total_cost * 100) if total_cost > 0 else 0,
            "positions_count": len(positions)
        }
    }


@app.post("/api/portfolio")
async def add_position(data: PositionCreate, db: Session = Depends(get_db)):
    """Add a new portfolio position"""
    # Pydantic validators handle input validation (ticker format, shares > 0, cost_basis limits)
    ticker = data.ticker  # Already uppercase from validator
    shares = data.shares
    cost_basis = data.cost_basis

    # Check if position already exists
    existing = db.query(PortfolioPosition).filter(
        PortfolioPosition.ticker == ticker
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail=f"Position in {ticker} already exists")

    # Get current stock data
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    current_price = stock.current_price if stock else None

    position = PortfolioPosition(
        ticker=ticker,
        shares=shares,
        cost_basis=cost_basis,
        current_price=current_price,
        current_value=current_price * shares if current_price else None,
        canslim_score=stock.canslim_score if stock else None
    )

    if current_price and cost_basis:
        position.gain_loss = (current_price - cost_basis) * shares
        position.gain_loss_pct = (current_price - cost_basis) / cost_basis * 100

    db.add(position)
    db.commit()

    return {"message": f"Added {shares} shares of {ticker}", "id": position.id}


@app.delete("/api/portfolio/{position_id}")
async def remove_position(position_id: int, db: Session = Depends(get_db)):
    """Remove a portfolio position"""
    position = db.query(PortfolioPosition).filter(
        PortfolioPosition.id == position_id
    ).first()

    if not position:
        raise HTTPException(status_code=404, detail="Position not found")

    db.delete(position)
    db.commit()

    return {"message": f"Removed position in {position.ticker}"}


@app.put("/api/portfolio/{position_id}")
async def update_position(position_id: int, data: PositionUpdate, db: Session = Depends(get_db)):
    """Update a portfolio position (shares, cost basis, notes)"""
    position = db.query(PortfolioPosition).filter(
        PortfolioPosition.id == position_id
    ).first()

    if not position:
        raise HTTPException(status_code=404, detail="Position not found")

    # Update fields if provided
    if data.shares is not None:
        if data.shares <= 0:
            raise HTTPException(status_code=400, detail="Shares must be greater than 0")
        position.shares = data.shares

    if data.cost_basis is not None:
        if data.cost_basis < 0:
            raise HTTPException(status_code=400, detail="Cost basis cannot be negative")
        position.cost_basis = data.cost_basis

    if data.notes is not None:
        position.notes = data.notes

    # Recalculate current value and gain/loss if we have a current price
    if position.current_price and position.current_price > 0:
        position.current_value = position.shares * position.current_price
        if position.cost_basis and position.cost_basis > 0:
            position.gain_loss = position.current_value - (position.shares * position.cost_basis)
            position.gain_loss_pct = ((position.current_price / position.cost_basis) - 1) * 100

    db.commit()

    return {
        "message": f"Updated position in {position.ticker}",
        "position": {
            "id": position.id,
            "ticker": position.ticker,
            "shares": position.shares,
            "cost_basis": position.cost_basis,
            "notes": position.notes,
            "current_value": position.current_value,
            "gain_loss": position.gain_loss,
            "gain_loss_pct": position.gain_loss_pct
        }
    }


def fetch_price_yahoo_chart(ticker: str) -> float | None:
    """Fetch current price using Yahoo Finance chart API (less rate-limited)"""
    import requests
    import time

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1d", "range": "5d"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if result:
                meta = result[0].get("meta", {})
                price = meta.get("regularMarketPrice") or meta.get("previousClose")
                return float(price) if price else None
        else:
            logger.warning(f"{ticker}: Yahoo chart API returned {resp.status_code}")
    except Exception as e:
        logger.error(f"{ticker}: chart API error: {e}")

    return None


@app.post("/api/portfolio/refresh")
async def refresh_portfolio(db: Session = Depends(get_db)):
    """Refresh all portfolio positions with current prices and auto-scan missing stocks"""
    import time

    positions = db.query(PortfolioPosition).all()
    logger.info(f"Refreshing {len(positions)} positions")

    updated = 0
    scanned = 0
    errors = []

    # First pass: identify and scan stocks without data
    for position in positions:
        stock = db.query(Stock).filter(Stock.ticker == position.ticker).first()
        if not stock or stock.canslim_score is None:
            logger.info(f"Auto-scanning {position.ticker} (no existing data)")
            try:
                analysis = analyze_stock(position.ticker)
                if analysis:
                    save_stock_to_db(db, analysis)
                    scanned += 1
                    logger.info(f"Scanned {position.ticker}: score={analysis.get('canslim_score', 0):.1f}")
                else:
                    logger.warning(f"Could not analyze {position.ticker}")
                # Rate limit delay
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error scanning {position.ticker}: {e}")
                errors.append(f"{position.ticker}: scan failed - {str(e)}")

    # Calculate total portfolio value for position weight calculations
    total_value = sum(p.current_value or 0 for p in positions)
    if total_value == 0:
        total_value = 10000  # Default for calculations

    # Second pass: update all positions with current prices and scores
    for position in positions:
        try:
            # Get stock data for CANSLIM score (may have just been scanned)
            stock = db.query(Stock).filter(Stock.ticker == position.ticker).first()

            current_price = fetch_price_yahoo_chart(position.ticker)

            if current_price:
                position.current_price = current_price
                position.current_value = current_price * position.shares

                if position.cost_basis:
                    position.gain_loss = (current_price - position.cost_basis) * position.shares
                    position.gain_loss_pct = (current_price - position.cost_basis) / position.cost_basis * 100

                # Update CANSLIM score from stock data
                if stock:
                    position.canslim_score = stock.canslim_score
                    # Calculate score_change from historical StockScore entries (matches Dashboard)
                    position.score_change = stock.score_change  # Updates every scan

                    # Smart recommendation matching gameplan logic
                    score = stock.canslim_score or 0
                    projected = stock.projected_growth or 0
                    gain_pct = position.gain_loss_pct or 0
                    position_weight = (position.current_value or 0) / total_value * 100

                    # SELL: Weak fundamentals with losses
                    if score < 35 and gain_pct < -10:
                        position.recommendation = "sell"
                    elif score < 40 and projected < -5:
                        position.recommendation = "sell"
                    elif score < 50 and gain_pct < -15:
                        position.recommendation = "sell"
                    # TRIM: Big winners - take profits
                    elif gain_pct >= 100:
                        position.recommendation = "trim"
                    elif gain_pct >= 50 and position_weight >= 15:
                        position.recommendation = "trim"
                    # ADD: Strong stock on pullback with room to grow
                    elif score >= 65 and projected >= 10 and -15 <= gain_pct <= 5 and position_weight < 12:
                        position.recommendation = "add"
                    # BUY: Strong fundamentals in profit
                    elif score >= 70 and gain_pct > -5:
                        position.recommendation = "buy"
                    else:
                        position.recommendation = "hold"

                updated += 1
                logger.info(f"Updated {position.ticker}: ${current_price:.2f}, score={position.canslim_score}")
            else:
                errors.append(f"{position.ticker}: no price found")

            # Small delay between requests to avoid rate limiting
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"Error processing {position.ticker}: {e}")
            errors.append(f"{position.ticker}: {str(e)}")

    db.commit()

    return {
        "message": f"Refreshed {updated} positions, scanned {scanned} new stocks",
        "updated": updated,
        "scanned": scanned,
        "errors": errors
    }


# ============== Portfolio Gameplan ==============

@app.get("/api/portfolio/gameplan")
async def get_portfolio_gameplan(db: Session = Depends(get_db)):
    """Generate actionable gameplan based on portfolio analysis"""
    positions = db.query(PortfolioPosition).all()

    # Calculate portfolio totals
    total_value = sum(p.current_value or 0 for p in positions)
    if total_value == 0:
        total_value = 10000  # Default for empty portfolio

    # Get top CANSLIM stocks for potential buys
    top_canslim_stocks = db.query(Stock).filter(
        Stock.canslim_score != None,
        Stock.canslim_score >= 65
    ).order_by(desc(Stock.canslim_score)).limit(15).all()

    # Get top Growth Mode stocks for potential buys
    top_growth_stocks = db.query(Stock).filter(
        Stock.growth_mode_score != None,
        Stock.growth_mode_score >= 65,
        Stock.is_growth_stock == True
    ).order_by(desc(Stock.growth_mode_score)).limit(10).all()

    # Combine and dedupe, tracking which are growth stocks
    seen_tickers = set()
    top_stocks = []
    for stock in top_canslim_stocks:
        if stock.ticker not in seen_tickers:
            seen_tickers.add(stock.ticker)
            top_stocks.append(stock)
    for stock in top_growth_stocks:
        if stock.ticker not in seen_tickers:
            seen_tickers.add(stock.ticker)
            top_stocks.append(stock)

    # Get tickers we already own (including duplicates like GOOG/GOOGL)
    owned_tickers = expand_tickers_with_duplicates({p.ticker for p in positions})

    # Batch fetch all stocks for positions (avoid N+1 queries)
    position_tickers = [p.ticker for p in positions]
    stocks_by_ticker = {}
    if position_tickers:
        position_stocks = db.query(Stock).filter(Stock.ticker.in_(position_tickers)).all()
        stocks_by_ticker = {s.ticker: s for s in position_stocks}

    actions = []

    # === SELL ACTIONS ===
    for p in positions:
        stock = stocks_by_ticker.get(p.ticker)

        # Use effective score based on stock type
        is_growth = stock.is_growth_stock if stock else False
        if is_growth and stock and stock.growth_mode_score:
            score = stock.growth_mode_score
            score_type = "Growth"
        else:
            score = stock.canslim_score if stock else (p.canslim_score or 0)
            score_type = "CANSLIM"
        projected = stock.projected_growth if stock else 0

        # Strong sell signals
        if score < 35 and (p.gain_loss_pct or 0) < -10:
            actions.append({
                "action": "SELL",
                "priority": 1,
                "ticker": p.ticker,
                "shares_action": p.shares,  # Sell all
                "shares_current": p.shares,
                "current_price": p.current_price,
                "estimated_value": p.current_value,
                "is_growth_stock": is_growth,
                "reason": f"Weak fundamentals ({score_type} {score:.0f}) with loss of {p.gain_loss_pct:.1f}%",
                "details": [
                    f"{score_type} Score: {score:.0f}/100 (below 35 threshold)",
                    f"Current loss: {p.gain_loss_pct:.1f}%",
                    f"Projected growth: {projected:.1f}%" if projected else "No growth projection",
                    "Cut losses early - O'Neil recommends selling at -7% to -8%"
                ]
            })
        elif score < 40 and projected < -5:
            actions.append({
                "action": "SELL",
                "priority": 2,
                "ticker": p.ticker,
                "shares_action": p.shares,
                "shares_current": p.shares,
                "current_price": p.current_price,
                "estimated_value": p.current_value,
                "is_growth_stock": is_growth,
                "reason": f"Deteriorating outlook ({score_type} {score:.0f}, {projected:.1f}% projected)",
                "details": [
                    f"{score_type} Score: {score:.0f}/100",
                    f"Negative growth projection: {projected:.1f}%",
                    "Fundamentals weakening - consider exiting before further decline"
                ]
            })

    # === TRIM / TAKE PROFITS ===
    for p in positions:
        gain_pct = p.gain_loss_pct or 0
        position_weight = (p.current_value or 0) / total_value * 100

        # Big winner - consider taking some profits
        if gain_pct >= 50 and position_weight >= 15:
            trim_shares = int(p.shares * 0.3)  # Trim 30%
            trim_value = trim_shares * (p.current_price or 0)
            actions.append({
                "action": "TRIM",
                "priority": 3,
                "ticker": p.ticker,
                "shares_action": trim_shares,
                "shares_current": p.shares,
                "current_price": p.current_price,
                "estimated_value": trim_value,
                "reason": f"Lock in gains - up {gain_pct:.0f}%, {position_weight:.0f}% of portfolio",
                "details": [
                    f"Position up {gain_pct:.1f}% (${p.gain_loss:,.0f} profit)",
                    f"Position is {position_weight:.1f}% of portfolio (overweight)",
                    f"Trim {trim_shares} shares (~30%) to lock in ~${trim_value:,.0f}",
                    "Reduce concentration risk while letting winner run"
                ]
            })
        elif gain_pct >= 100:
            trim_shares = int(p.shares * 0.5)  # Trim 50% on 100%+ gains
            trim_value = trim_shares * (p.current_price or 0)
            actions.append({
                "action": "TRIM",
                "priority": 2,
                "ticker": p.ticker,
                "shares_action": trim_shares,
                "shares_current": p.shares,
                "current_price": p.current_price,
                "estimated_value": trim_value,
                "reason": f"Exceptional gain of {gain_pct:.0f}% - secure profits",
                "details": [
                    f"Position doubled! Up {gain_pct:.1f}%",
                    f"Trim {trim_shares} shares (50%) to recover original investment",
                    f"Let remaining shares ride as 'house money'",
                    "Classic O'Neil strategy: sell half at 100% gain"
                ]
            })

    # === BUY NEW POSITIONS ===
    max_position_value = total_value * 0.10  # Max 10% per position

    # Track which duplicate groups we've already recommended a BUY for
    recommended_groups = set()

    for stock in top_stocks:
        if stock.ticker in owned_tickers:
            continue

        # Check if this ticker belongs to a duplicate group we've already recommended
        skip_duplicate = False
        ticker_group = None
        for group in DUPLICATE_TICKERS:
            if stock.ticker in group:
                ticker_group = frozenset(group)
                if ticker_group in recommended_groups:
                    # Already recommended a higher-scoring stock from this group
                    skip_duplicate = True
                break

        if skip_duplicate:
            continue

        # Use effective score based on stock type
        is_growth = stock.is_growth_stock or False
        if is_growth and stock.growth_mode_score:
            effective_score = stock.growth_mode_score
            score_type = "Growth"
        else:
            effective_score = stock.canslim_score or 0
            score_type = "CANSLIM"

        if effective_score >= 75 and (stock.projected_growth or 0) >= 15:
            shares_to_buy = int(max_position_value / stock.current_price) if stock.current_price else 0
            buy_value = shares_to_buy * stock.current_price if shares_to_buy else max_position_value

            # Check breakout status and base pattern
            is_breaking_out = getattr(stock, 'is_breaking_out', False)
            base_type = getattr(stock, 'base_type', 'none') or 'none'
            weeks_in_base = getattr(stock, 'weeks_in_base', 0) or 0
            has_base = base_type not in ('none', '', None)

            # Build entry signal description
            entry_signal = ""
            if is_breaking_out and has_base:
                entry_signal = f"🚀 Breaking out of {base_type} base ({weeks_in_base}w)"
            elif is_breaking_out:
                entry_signal = "🚀 Breaking out with volume"
            elif has_base:
                if stock.week_52_high and stock.current_price:
                    pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100
                    if pct_from_high <= 15:
                        entry_signal = f"📈 Pre-breakout: {base_type} base ({weeks_in_base}w), {pct_from_high:.0f}% from pivot"
                    else:
                        entry_signal = f"Base forming: {base_type} ({weeks_in_base}w)"
            elif stock.week_52_high and stock.current_price:
                pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100
                if pct_from_high <= 10:
                    entry_signal = f"Within {pct_from_high:.0f}% of 52-week high (no base detected)"

            # Adjust priority based on entry quality
            priority = 2
            if is_breaking_out and has_base:
                priority = 1  # Highest priority - confirmed breakout from base
            elif has_base and stock.week_52_high and stock.current_price:
                pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100
                if pct_from_high <= 15:
                    priority = 1  # High priority - pre-breakout with base

            actions.append({
                "action": "BUY",
                "priority": priority,
                "ticker": stock.ticker,
                "shares_action": shares_to_buy,
                "shares_current": 0,
                "current_price": stock.current_price,
                "estimated_value": buy_value,
                "is_growth_stock": is_growth,
                "reason": f"Strong candidate - {score_type} {effective_score:.0f}, +{stock.projected_growth:.0f}% projected",
                "details": [
                    f"{score_type} Score: {effective_score:.0f}/100",
                    f"Projected 6-month growth: +{stock.projected_growth:.0f}%",
                    entry_signal if entry_signal else f"Current price: ${stock.current_price:.2f}",
                    f"Sector: {stock.sector}",
                    f"Suggested position: {shares_to_buy} shares (~${buy_value:,.0f}, 10% of portfolio)"
                ]
            })

            # Mark this duplicate group as recommended (if applicable)
            if ticker_group:
                recommended_groups.add(ticker_group)

            if len([a for a in actions if a["action"] == "BUY"]) >= 3:
                break  # Max 3 buy recommendations

    # === ADD TO WINNERS ===
    for p in positions:
        stock = db.query(Stock).filter(Stock.ticker == p.ticker).first()
        if not stock:
            continue

        # Use effective score based on stock type
        is_growth = stock.is_growth_stock or False
        if is_growth and stock.growth_mode_score:
            score = stock.growth_mode_score
            score_type = "Growth"
        else:
            score = stock.canslim_score or 0
            score_type = "CANSLIM"
        projected = stock.projected_growth or 0
        gain_pct = p.gain_loss_pct or 0
        position_weight = (p.current_value or 0) / total_value * 100

        # Good stock that's pulled back - add on dip
        if score >= 65 and projected >= 10 and -15 <= gain_pct <= 5 and position_weight < 12:
            # Calculate how much to add (up to 10% total position)
            target_value = total_value * 0.10
            current_value = p.current_value or 0
            add_value = min(target_value - current_value, total_value * 0.05)  # Add up to 5% more

            if add_value > 500 and stock.current_price:  # Min $500 to add
                add_shares = int(add_value / stock.current_price)
                actions.append({
                    "action": "ADD",
                    "priority": 3,
                    "ticker": p.ticker,
                    "shares_action": add_shares,
                    "shares_current": p.shares,
                    "current_price": p.current_price,
                    "estimated_value": add_value,
                    "is_growth_stock": is_growth,
                    "reason": f"Strong stock on pullback - {score_type} {score:.0f}, currently {gain_pct:+.1f}%",
                    "details": [
                        f"{score_type} Score: {score:.0f}/100 (strong)",
                        f"Projected growth: +{projected:.0f}%",
                        f"Current position: {gain_pct:+.1f}% (buying the dip)",
                        f"Position weight: {position_weight:.1f}% (room to add)",
                        f"Add {add_shares} shares (~${add_value:,.0f})"
                    ]
                })

    # === WATCH LIST CANDIDATES ===
    watch_actions = []
    watched_groups = set()  # Track duplicate groups already on watch list
    for stock in top_stocks:
        if stock.ticker in owned_tickers:
            continue
        if stock.ticker in [a["ticker"] for a in actions if a["action"] == "BUY"]:
            continue

        # Check for duplicate groups (skip if already watching one from same group)
        skip_watch = False
        watch_ticker_group = None
        for group in DUPLICATE_TICKERS:
            if stock.ticker in group:
                watch_ticker_group = frozenset(group)
                if watch_ticker_group in watched_groups or watch_ticker_group in recommended_groups:
                    skip_watch = True
                break

        if skip_watch:
            continue

        # Use effective score based on stock type
        is_growth = stock.is_growth_stock or False
        if is_growth and stock.growth_mode_score:
            effective_score = stock.growth_mode_score
            score_type = "Growth"
        else:
            effective_score = stock.canslim_score or 0
            score_type = "CANSLIM"

        if effective_score >= 70 and stock.week_52_high and stock.current_price:
            pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100

            # Get base pattern info
            base_type = getattr(stock, 'base_type', 'none') or 'none'
            weeks_in_base = getattr(stock, 'weeks_in_base', 0) or 0
            has_base = base_type not in ('none', '', None)

            # Approaching breakout (within 5-15% of high)
            # Prioritize stocks WITH base patterns
            if 5 <= pct_from_high <= 15:
                # Set priority based on base pattern quality
                if has_base and weeks_in_base >= 5:
                    watch_priority = 3  # Higher priority - proper base forming
                    watch_reason = f"📈 Pre-breakout setup: {base_type} base ({weeks_in_base}w), {pct_from_high:.0f}% from pivot"
                elif has_base:
                    watch_priority = 4
                    watch_reason = f"Base forming ({base_type}) - {pct_from_high:.0f}% from high"
                else:
                    watch_priority = 5  # Lower priority - no base pattern
                    watch_reason = f"Approaching 52-week high - {pct_from_high:.0f}% away (no base detected)"

                # Build details
                details = [f"{score_type} Score: {effective_score:.0f}/100"]
                if has_base:
                    details.append(f"Base pattern: {base_type} ({weeks_in_base} weeks)")
                details.extend([
                    f"52-week high: ${stock.week_52_high:.2f}",
                    f"Current: ${stock.current_price:.2f} ({pct_from_high:.1f}% below high)",
                ])
                if has_base:
                    details.append("Watch for breakout above pivot on 1.5x+ volume")
                else:
                    details.append("⚠️ No base pattern - wait for consolidation")
                if stock.projected_growth:
                    details.append(f"Projected growth: +{stock.projected_growth:.0f}%")

                watch_actions.append({
                    "action": "WATCH",
                    "priority": watch_priority,
                    "ticker": stock.ticker,
                    "shares_action": 0,
                    "shares_current": 0,
                    "current_price": stock.current_price,
                    "estimated_value": 0,
                    "is_growth_stock": is_growth,
                    "reason": watch_reason,
                    "details": [d for d in details if d]  # Filter empty strings
                })

                # Mark this duplicate group as watched (if applicable)
                if watch_ticker_group:
                    watched_groups.add(watch_ticker_group)

                if len(watch_actions) >= 3:
                    break

    actions.extend(watch_actions)

    # Sort by priority
    actions.sort(key=lambda x: (x["priority"], -x.get("estimated_value", 0)))

    return {
        "gameplan": actions,
        "summary": {
            "total_actions": len(actions),
            "sell_count": len([a for a in actions if a["action"] == "SELL"]),
            "trim_count": len([a for a in actions if a["action"] == "TRIM"]),
            "buy_count": len([a for a in actions if a["action"] == "BUY"]),
            "add_count": len([a for a in actions if a["action"] == "ADD"]),
            "watch_count": len([a for a in actions if a["action"] == "WATCH"]),
            "portfolio_value": total_value
        }
    }


# ============== AI Portfolio ==============

from backend.ai_trader import (
    get_or_create_config, get_portfolio_value, update_position_prices,
    run_ai_trading_cycle, take_portfolio_snapshot, initialize_ai_portfolio,
    refresh_ai_portfolio
)


# ============== Coiled Spring Alerts ==============

@app.get("/api/coiled-spring/alerts")
async def get_coiled_spring_alerts(
    db: Session = Depends(get_db),
    days: int = Query(7, ge=1, le=30, description="Number of days to look back")
):
    """
    Get Coiled Spring alerts - high-conviction earnings catalyst plays.

    These are stocks meeting ALL criteria:
    - 15+ weeks in base (long consolidation)
    - 3+ consecutive earnings beats
    - C score >= 12, L score >= 8, Total >= 65
    - Institutional ownership <= 40%
    - 1-14 days to earnings
    """
    cutoff_date = date.today() - timedelta(days=days)

    alerts = db.query(CoiledSpringAlert).filter(
        CoiledSpringAlert.alert_date >= cutoff_date
    ).order_by(desc(CoiledSpringAlert.alert_date), desc(CoiledSpringAlert.cs_bonus)).all()

    # Get current stock data for each alert
    alert_tickers = [a.ticker for a in alerts]
    stocks_by_ticker = {}
    if alert_tickers:
        stocks = db.query(Stock).filter(Stock.ticker.in_(alert_tickers)).all()
        stocks_by_ticker = {s.ticker: s for s in stocks}

    return {
        "alerts": [{
            "id": a.id,
            "ticker": a.ticker,
            "alert_date": a.alert_date.isoformat(),
            "days_to_earnings": a.days_to_earnings,
            "weeks_in_base": a.weeks_in_base,
            "beat_streak": a.beat_streak,
            "c_score": a.c_score,
            "total_score": a.total_score,
            "cs_bonus": a.cs_bonus,
            "price_at_alert": a.price_at_alert,
            "current_price": stocks_by_ticker.get(a.ticker, {}).current_price if stocks_by_ticker.get(a.ticker) else None,
            "base_type": stocks_by_ticker.get(a.ticker, {}).base_type if stocks_by_ticker.get(a.ticker) else None,
            "outcome": a.outcome,
            "price_change_pct": a.price_change_pct,
            "email_sent": a.email_sent
        } for a in alerts],
        "total": len(alerts),
        "today_count": sum(1 for a in alerts if a.alert_date == date.today())
    }


@app.get("/api/coiled-spring/candidates")
async def get_coiled_spring_candidates(db: Session = Depends(get_db), limit: int = 10, pre_breakout_only: bool = False):
    """
    Get current stocks that qualify as Coiled Spring candidates.
    These are stocks that meet CS criteria based on current data,
    ranked by quality (best candidates first).

    Args:
        limit: Max candidates to return (default 10)
        pre_breakout_only: If True, only return stocks NOT already breaking out
    """
    from config_loader import config

    cs_config = config.get('coiled_spring', {})
    thresholds = cs_config.get('thresholds', {})
    pre_breakout_thresholds = cs_config.get('pre_breakout_thresholds', {})
    earnings_window = cs_config.get('earnings_window', {})
    ranking_weights = cs_config.get('ranking_weights', {})

    # Base filters for all candidates
    base_filters = [
        Stock.weeks_in_base >= thresholds.get('min_weeks_in_base', 15),
        Stock.l_score >= thresholds.get('min_l_score', 6),
        Stock.days_to_earnings != None,
        Stock.days_to_earnings > earnings_window.get('block_days', 1),
        Stock.days_to_earnings <= earnings_window.get('alert_days', 14)
    ]

    candidates = []

    if not pre_breakout_only:
        # Query breakout candidates with standard thresholds
        breakout_query = db.query(Stock).filter(
            *base_filters,
            Stock.canslim_score >= thresholds.get('min_total_score', 55),
            Stock.c_score >= thresholds.get('min_c_score', 10),
            Stock.is_breaking_out == True
        )
        candidates.extend(breakout_query.all())

    # Query PRE-BREAKOUT candidates with RELAXED thresholds
    # (they haven't had their catalyst yet, so C score may be lower)
    pre_breakout_min_score = pre_breakout_thresholds.get('min_total_score', thresholds.get('min_total_score', 55))
    pre_breakout_min_c = pre_breakout_thresholds.get('min_c_score', thresholds.get('min_c_score', 10))
    pre_breakout_min_beats = pre_breakout_thresholds.get('min_beat_streak', thresholds.get('min_beat_streak', 3))

    pre_breakout_query = db.query(Stock).filter(
        *base_filters,
        Stock.canslim_score >= pre_breakout_min_score,
        Stock.c_score >= pre_breakout_min_c,
        Stock.is_breaking_out == False,
        Stock.earnings_beat_streak >= pre_breakout_min_beats
    )
    candidates.extend(pre_breakout_query.all())

    # Deduplicate (shouldn't happen but just in case)
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c.ticker not in seen:
            seen.add(c.ticker)
            unique_candidates.append(c)
    candidates = unique_candidates

    # Filter by institutional ownership and beat streak
    qualified = []
    for stock in candidates:
        # Extract institutional_pct from score_details JSON
        inst_pct = (stock.score_details or {}).get('i', {}).get('institutional_pct', 0) or 0
        beat_streak = getattr(stock, 'earnings_beat_streak', 0) or 0

        if (inst_pct <= thresholds.get('max_institutional_pct', 75) and
            beat_streak >= thresholds.get('min_beat_streak', 3)):

            # Calculate CS bonus
            scoring = cs_config.get('scoring', {})
            cs_bonus = scoring.get('base_bonus', 20)
            if stock.weeks_in_base and stock.weeks_in_base >= 20:
                cs_bonus += scoring.get('long_base_bonus', 10)
            if beat_streak >= 5:
                cs_bonus += scoring.get('strong_beat_bonus', 5)
            cs_bonus = min(cs_bonus, scoring.get('max_bonus', 35))

            # Calculate % from 52-week high
            high_52w = stock.week_52_high or 0
            price = stock.current_price or 0
            pct_from_high = ((price - high_52w) / high_52w * 100) if high_52w > 0 else 0
            volume_ratio = stock.volume_ratio or 1.0

            # Calculate quality ranking score (higher = better candidate)
            w_base = ranking_weights.get('weeks_in_base', 1.5)
            w_beats = ranking_weights.get('beat_streak', 3.0)
            w_l = ranking_weights.get('l_score', 2.0)
            w_total = ranking_weights.get('total_score', 0.5)
            low_inst_bonus = ranking_weights.get('low_inst_bonus', 10)
            pre_breakout_bonus = ranking_weights.get('pre_breakout_bonus', 15)
            extended_penalty = ranking_weights.get('extended_penalty', -20)

            quality_rank = (
                (stock.weeks_in_base or 0) * w_base +
                beat_streak * w_beats +
                (stock.l_score or 0) * w_l +
                (stock.canslim_score or 0) * w_total
            )

            # Bonus for truly low institutional (< 30%)
            if inst_pct < 30:
                quality_rank += low_inst_bonus

            # Bonus for PRE-BREAKOUT (hasn't broken out yet - ideal entry)
            if not stock.is_breaking_out:
                quality_rank += pre_breakout_bonus
                entry_status = "PRE-BREAKOUT"
            # Penalty for EXTENDED (already at high with volume surge)
            elif pct_from_high > -2 and volume_ratio > 2.0:
                quality_rank += extended_penalty
                entry_status = "EXTENDED"
            else:
                entry_status = "BREAKING_OUT"

            qualified.append({
                "ticker": stock.ticker,
                "name": stock.name,
                "canslim_score": stock.canslim_score,
                "c_score": stock.c_score,
                "l_score": stock.l_score,
                "weeks_in_base": stock.weeks_in_base,
                "base_type": stock.base_type,
                "earnings_beat_streak": beat_streak,
                "days_to_earnings": stock.days_to_earnings,
                "institutional_holders_pct": inst_pct,
                "cs_bonus": cs_bonus,
                "quality_rank": round(quality_rank, 1),
                "current_price": stock.current_price,
                "pct_from_high": round(pct_from_high, 1),
                "volume_ratio": round(volume_ratio, 1),
                "entry_status": entry_status,
                "is_breaking_out": stock.is_breaking_out
            })

    # Sort by quality_rank (best candidates first)
    qualified.sort(key=lambda x: x['quality_rank'], reverse=True)

    # Limit results
    qualified = qualified[:limit]

    return {
        "candidates": qualified,
        "total": len(qualified),
        "thresholds": thresholds,
        "ranking_weights": ranking_weights
    }


def _get_deduped_cs_alert_ids(db):
    """Return IDs of CS alerts deduplicated: one per ticker per earnings cycle.

    For each ticker, alerts within 21 days of each other are considered the same
    earnings event. We keep the first alert (lowest ID) per cycle, preferring
    the one with an outcome if available.
    """
    from backend.database import CoiledSpringAlert
    from datetime import timedelta

    all_alerts = db.query(CoiledSpringAlert).order_by(
        CoiledSpringAlert.ticker, CoiledSpringAlert.alert_date
    ).all()

    keep_ids = []
    # Group alerts by ticker, then by earnings cycle (21-day window)
    ticker_alerts = {}
    for a in all_alerts:
        ticker_alerts.setdefault(a.ticker, []).append(a)

    for ticker, alerts in ticker_alerts.items():
        # Walk through alerts chronologically, grouping by 21-day windows
        cycle_start = None
        cycle_best = None
        for a in alerts:
            if cycle_start is None or (a.alert_date - cycle_start) > timedelta(days=21):
                # New cycle — save the best from previous cycle
                if cycle_best is not None:
                    keep_ids.append(cycle_best.id)
                cycle_start = a.alert_date
                cycle_best = a
            else:
                # Same cycle — prefer the one with an outcome, else keep first
                if not cycle_best.outcome and a.outcome:
                    cycle_best = a
        # Don't forget the last cycle
        if cycle_best is not None:
            keep_ids.append(cycle_best.id)

    return keep_ids


def _get_deduped_cs_alerts(db):
    """Return deduped CS alert query."""
    from backend.database import CoiledSpringAlert
    keep_ids = _get_deduped_cs_alert_ids(db)
    return db.query(CoiledSpringAlert).filter(CoiledSpringAlert.id.in_(keep_ids))


def _cs_stats(alerts_list):
    """Compute CS stats from a list of alert objects."""
    with_outcome = [a for a in alerts_list if a.outcome]
    wins = sum(1 for a in with_outcome if a.outcome in ('win', 'big_win'))
    big_wins = sum(1 for a in with_outcome if a.outcome == 'big_win')
    losses = sum(1 for a in with_outcome if a.outcome == 'loss')
    flat = sum(1 for a in with_outcome if a.outcome == 'flat')
    pending = sum(1 for a in alerts_list if not a.outcome)
    win_rate = round(wins / len(with_outcome) * 100, 1) if with_outcome else 0
    big_win_rate = round(big_wins / len(with_outcome) * 100, 1) if with_outcome else 0
    return {
        "total": len(alerts_list),
        "with_outcome": len(with_outcome),
        "wins": wins,
        "big_wins": big_wins,
        "losses": losses,
        "flat": flat,
        "pending": pending,
        "win_rate": win_rate,
        "big_win_rate": big_win_rate,
    }


@app.get("/api/coiled-spring/history")
async def get_coiled_spring_history(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    """
    Get historical Coiled Spring alerts with outcomes for success rate tracking.
    Deduplicates alerts (one per ticker per date). Supports pagination.
    """
    from backend.database import CoiledSpringAlert

    # Deduped query base
    deduped_q = _get_deduped_cs_alerts(db)
    total = deduped_q.count()

    # Paginated alerts
    alerts = deduped_q.order_by(
        CoiledSpringAlert.alert_date.desc()
    ).offset((page - 1) * page_size).limit(page_size).all()

    # Page stats
    page_stats = _cs_stats(alerts)

    # Cumulative stats (all deduped alerts)
    all_deduped = deduped_q.all()
    cumulative = _cs_stats(all_deduped)

    # Group by base_type
    by_base_type = {}
    for a in all_deduped:
        base = a.base_type or 'unknown'
        if base not in by_base_type:
            by_base_type[base] = {'total': 0, 'with_outcome': 0, 'wins': 0, 'big_wins': 0}
        by_base_type[base]['total'] += 1
        if a.outcome:
            by_base_type[base]['with_outcome'] += 1
            if a.outcome in ('win', 'big_win'):
                by_base_type[base]['wins'] += 1
            if a.outcome == 'big_win':
                by_base_type[base]['big_wins'] += 1
    for base, stats in by_base_type.items():
        stats['win_rate'] = round(stats['wins'] / stats['with_outcome'] * 100, 1) if stats['with_outcome'] > 0 else 0

    return {
        "alerts": [{
            "id": a.id,
            "ticker": a.ticker,
            "alert_date": a.alert_date.isoformat() if a.alert_date else None,
            "days_to_earnings": a.days_to_earnings,
            "weeks_in_base": a.weeks_in_base,
            "beat_streak": a.beat_streak,
            "price_at_alert": a.price_at_alert,
            "price_after_earnings": a.price_after_earnings,
            "price_change_pct": a.price_change_pct,
            "outcome": a.outcome,
            "base_type": a.base_type,
            "institutional_pct": a.institutional_pct,
            "cs_bonus": a.cs_bonus,
            "total_score": a.total_score,
        } for a in alerts],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": (total + page_size - 1) // page_size
        },
        "page_stats": {
            "total_alerts": page_stats["total"],
            "with_outcome": page_stats["with_outcome"],
            "win_rate": page_stats["win_rate"],
            "big_win_rate": page_stats["big_win_rate"],
        },
        "cumulative_stats": {
            "total_alerts_all_time": cumulative["total"],
            "with_outcome": cumulative["with_outcome"],
            "wins": cumulative["wins"],
            "big_wins": cumulative["big_wins"],
            "losses": cumulative["losses"],
            "flat": cumulative["flat"],
            "overall_win_rate": cumulative["win_rate"],
            "overall_big_win_rate": cumulative["big_win_rate"],
            "by_base_type": by_base_type
        }
    }


@app.post("/api/coiled-spring/record")
async def record_coiled_spring_alert(ticker: str, db: Session = Depends(get_db)):
    """
    Record a Coiled Spring alert for tracking.
    Called when a CS candidate is identified for the watchlist.
    """
    from backend.database import CoiledSpringAlert
    from datetime import date

    # Get current stock data
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")

    # Check if already recorded today
    today = date.today()
    existing = db.query(CoiledSpringAlert).filter(
        CoiledSpringAlert.ticker == ticker,
        CoiledSpringAlert.alert_date == today
    ).first()
    if existing:
        return {"status": "already_recorded", "id": existing.id}

    # Extract data
    inst_pct = (stock.score_details or {}).get('i', {}).get('institutional_pct', 0) or 0

    alert = CoiledSpringAlert(
        ticker=ticker,
        alert_date=today,
        days_to_earnings=stock.days_to_earnings,
        weeks_in_base=stock.weeks_in_base,
        beat_streak=stock.earnings_beat_streak,
        c_score=stock.c_score,
        total_score=stock.canslim_score,
        price_at_alert=stock.current_price,
        base_type=stock.base_type,
        institutional_pct=inst_pct,
        l_score=stock.l_score
    )
    db.add(alert)
    db.commit()

    return {"status": "recorded", "id": alert.id}


@app.post("/api/coiled-spring/cleanup-duplicates")
async def cleanup_cs_duplicates(db: Session = Depends(get_db), dry_run: bool = Query(True)):
    """Delete duplicate CS alerts, keeping one per ticker per earnings cycle (21 days).
    Prefers alerts with outcomes. Use dry_run=false to actually delete."""
    from backend.database import CoiledSpringAlert

    keep_ids = set(_get_deduped_cs_alert_ids(db))
    all_alerts = db.query(CoiledSpringAlert).all()

    dupes = [a for a in all_alerts if a.id not in keep_ids]
    deleted_info = [{"id": d.id, "ticker": d.ticker, "date": str(d.alert_date), "outcome": d.outcome} for d in dupes]

    if not dry_run:
        for d in dupes:
            db.delete(d)
        db.commit()

    return {
        "dry_run": dry_run,
        "would_delete" if dry_run else "deleted": len(deleted_info),
        "keeping": len(keep_ids),
        "details": deleted_info
    }


@app.post("/api/coiled-spring/update-outcomes")
async def update_coiled_spring_outcomes(db: Session = Depends(get_db)):
    """
    Update outcomes for past CS alerts where earnings have occurred.
    Compares price_at_alert to current price.
    """
    from backend.database import CoiledSpringAlert
    from datetime import date, datetime

    # Find alerts without outcomes where earnings should have passed
    alerts = db.query(CoiledSpringAlert).filter(
        CoiledSpringAlert.outcome == None,
        CoiledSpringAlert.price_at_alert != None
    ).all()

    updated = 0
    for alert in alerts:
        # Get current stock price
        stock = db.query(Stock).filter(Stock.ticker == alert.ticker).first()
        if not stock or not stock.current_price:
            continue

        # Check if enough time has passed (alert_date + days_to_earnings + 1 day buffer)
        days_since_alert = (date.today() - alert.alert_date).days
        if alert.days_to_earnings and days_since_alert < (alert.days_to_earnings + 1):
            continue  # Earnings haven't happened yet

        # Calculate outcome
        price_change_pct = ((stock.current_price - alert.price_at_alert) / alert.price_at_alert) * 100

        if price_change_pct >= 15:
            outcome = "big_win"
        elif price_change_pct >= 5:
            outcome = "win"
        elif price_change_pct >= -5:
            outcome = "flat"
        else:
            outcome = "loss"

        alert.price_after_earnings = stock.current_price
        alert.price_change_pct = round(price_change_pct, 2)
        alert.outcome = outcome
        alert.outcome_updated_at = datetime.now(timezone.utc)
        updated += 1

    db.commit()

    return {"status": "success", "updated": updated}


@app.get("/api/ai-portfolio")
async def get_ai_portfolio(db: Session = Depends(get_db)):
    """Get AI Portfolio overview"""
    config = get_or_create_config(db)
    portfolio = get_portfolio_value(db)
    positions = db.query(AIPortfolioPosition).all()

    # Batch fetch all stocks for positions (avoid N+1 queries)
    position_tickers = [p.ticker for p in positions]
    stocks_by_ticker = {}
    if position_tickers:
        position_stocks = db.query(Stock).filter(Stock.ticker.in_(position_tickers)).all()
        stocks_by_ticker = {s.ticker: s for s in position_stocks}

    # Build positions with stock data for insider/short signals
    positions_data = []
    for p in positions:
        stock = stocks_by_ticker.get(p.ticker)

        # Calculate trailing stop info
        trailing_stop_info = None
        if p.peak_price and p.cost_basis and p.current_price:
            peak_gain_pct = ((p.peak_price / p.cost_basis) - 1) * 100 if p.cost_basis > 0 else 0
            drop_from_peak = ((p.peak_price - p.current_price) / p.peak_price) * 100 if p.peak_price > 0 else 0

            # Determine threshold
            if peak_gain_pct >= 50:
                threshold = 15
            elif peak_gain_pct >= 30:
                threshold = 12
            elif peak_gain_pct >= 20:
                threshold = 10
            elif peak_gain_pct >= 10:
                threshold = 8
            else:
                threshold = None

            trailing_stop_info = {
                "peak_price": p.peak_price,
                "peak_date": p.peak_date.isoformat() if p.peak_date else None,
                "drop_from_peak_pct": round(drop_from_peak, 1),
                "threshold_pct": threshold,
                "near_stop": threshold and drop_from_peak >= threshold * 0.7  # Within 70% of threshold
            }

        position_data = {
            "id": p.id,
            "ticker": p.ticker,
            "shares": p.shares,
            "cost_basis": p.cost_basis,
            "current_price": p.current_price,
            "current_value": p.current_value,
            "gain_loss": p.gain_loss,
            "gain_loss_pct": p.gain_loss_pct,
            # CANSLIM scores
            "purchase_score": p.purchase_score,
            "current_score": p.current_score,
            # Growth Mode scores
            "is_growth_stock": p.is_growth_stock or False,
            "purchase_growth_score": p.purchase_growth_score,
            "current_growth_score": p.current_growth_score,
            "purchase_date": p.purchase_date.isoformat() if p.purchase_date else None,
            # Trailing stop tracking
            "trailing_stop": trailing_stop_info,
            # Insider/Short signals from Stock table
            "insider_sentiment": stock.insider_sentiment if stock else None,
            "insider_buy_count": stock.insider_buy_count if stock else None,
            "short_interest_pct": stock.short_interest_pct if stock else None,
        }
        positions_data.append(position_data)

    return {
        "config": {
            "starting_cash": config.starting_cash,
            "max_positions": config.max_positions,
            "max_position_pct": config.max_position_pct,
            "min_score_to_buy": config.min_score_to_buy,
            "sell_score_threshold": config.sell_score_threshold,
            "take_profit_pct": config.take_profit_pct,
            "stop_loss_pct": config.stop_loss_pct,
            "is_active": config.is_active,
            "paper_mode": getattr(config, 'paper_mode', False) or False,
            "strategy": getattr(config, 'strategy', None) or "balanced",
        },
        "summary": portfolio,
        "positions": positions_data
    }


@app.get("/api/ai-portfolio/history")
async def get_ai_portfolio_history(
    days: int = Query(30, le=365),
    db: Session = Depends(get_db)
):
    """Get AI Portfolio performance history for charts - includes all snapshots from scans"""
    from datetime import timedelta, datetime as dt
    from sqlalchemy import or_

    start_date = dt.utcnow() - timedelta(days=days)
    start_date_only = date.today() - timedelta(days=days)

    # Get all snapshots - include both timestamp-based (new) and date-based (old/migrated)
    snapshots = db.query(AIPortfolioSnapshot).filter(
        or_(
            AIPortfolioSnapshot.timestamp >= start_date,
            AIPortfolioSnapshot.date >= start_date_only
        )
    ).order_by(
        AIPortfolioSnapshot.timestamp.asc().nullsfirst(),
        AIPortfolioSnapshot.date.asc()
    ).all()

    # Sort properly: use timestamp if available, otherwise use date
    def sort_key(s):
        if s.timestamp:
            return s.timestamp
        elif s.date:
            return dt.combine(s.date, dt.min.time())
        return dt.min

    snapshots = sorted(snapshots, key=sort_key)

    return [{
        "timestamp": s.timestamp.replace(tzinfo=ZoneInfo("America/Chicago")).isoformat() if s.timestamp else None,
        "date": s.date.isoformat() if s.date else (s.timestamp.date().isoformat() if s.timestamp else None),
        "total_value": s.total_value,
        "cash": s.cash,
        "positions_value": s.positions_value,
        "positions_count": s.positions_count,
        "total_return": s.total_return,
        "total_return_pct": s.total_return_pct,
        "value_change": getattr(s, 'value_change', None) or getattr(s, 'day_change', None),
        "value_change_pct": getattr(s, 'value_change_pct', None) or getattr(s, 'day_change_pct', None)
    } for s in snapshots]


@app.post("/api/ai-portfolio/refresh")
async def refresh_ai_portfolio_endpoint(background_tasks: BackgroundTasks, check_stops: bool = True):
    """Refresh position prices and check stop losses (runs in background)"""
    from backend.database import SessionLocal
    from backend.ai_trader import take_portfolio_snapshot, check_and_execute_stop_losses

    def refresh_background():
        db = SessionLocal()
        try:
            logger.info("Refreshing AI portfolio prices in background...")
            if check_stops:
                # Check stop losses and execute sells if triggered
                result = check_and_execute_stop_losses(db)
                sells = result.get("sells_executed", [])
                if sells:
                    logger.info(f"STOP LOSS ALERT: {len(sells)} positions sold: {[s['ticker'] for s in sells]}")
            else:
                # Just refresh prices without checking stops
                result = refresh_ai_portfolio(db)
            logger.info(f"AI portfolio refresh complete: {result.get('message')}")
        except Exception as e:
            logger.error(f"AI portfolio refresh error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            db.close()

    background_tasks.add_task(refresh_background)
    return {"status": "started", "message": "Price refresh + stop loss check started. Refresh page in 10-15 seconds."}


@app.get("/api/ai-portfolio/trades")
async def get_ai_portfolio_trades(
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    """Get AI Portfolio trade history"""
    trades = db.query(AIPortfolioTrade).order_by(
        desc(AIPortfolioTrade.executed_at)
    ).limit(limit).all()

    return [{
        "id": t.id,
        "ticker": t.ticker,
        "action": t.action,
        "shares": t.shares,
        "price": t.price,
        "total_value": t.total_value,
        "reason": t.reason,
        "canslim_score": t.canslim_score,
        "growth_mode_score": t.growth_mode_score,
        "is_growth_stock": t.is_growth_stock or False,
        "cost_basis": t.cost_basis,
        "realized_gain": t.realized_gain,
        "executed_at": t.executed_at.isoformat() + 'Z' if t.executed_at else None
    } for t in trades]


@app.post("/api/ai-portfolio/initialize")
async def initialize_ai_portfolio_endpoint(
    starting_cash: float = Query(25000.0, ge=1000, le=1000000),
    strategy: str = Query("balanced"),
    db: Session = Depends(get_db)
):
    """Initialize or reset the AI Portfolio"""
    validate_strategy_name(strategy)
    result = initialize_ai_portfolio(db, starting_cash, strategy=strategy)
    return result


@app.post("/api/ai-portfolio/run-cycle")
async def run_ai_trading_cycle_endpoint(background_tasks: BackgroundTasks, force: bool = Query(False)):
    """Manually trigger an AI trading cycle (runs in background)"""
    from backend.database import SessionLocal
    from backend.ai_trader import take_portfolio_snapshot, _trading_cycle_lock, _trading_cycle_started, is_market_open
    from datetime import datetime

    # Check if market is open (skip with force=true)
    if not force and not is_market_open():
        return {
            "status": "market_closed",
            "message": "Market is closed. Trading only runs during market hours (Mon-Fri 9:30 AM - 4:00 PM Eastern)."
        }

    # Check if a cycle is already running BEFORE launching background task
    if _trading_cycle_lock and _trading_cycle_started:
        elapsed = (datetime.now() - _trading_cycle_started).total_seconds()
        if elapsed < 300:  # 5 minute timeout
            return {
                "status": "busy",
                "message": f"Trading cycle already running ({int(elapsed)}s elapsed). Please wait."
            }

    def run_cycle_background():
        db = SessionLocal()
        try:
            logger.info("Starting AI trading cycle in background...")
            result = run_ai_trading_cycle(db)
            if result.get("status") == "busy":
                logger.warning(f"Trading cycle was busy: {result.get('message')}")
            else:
                logger.info(f"AI trading cycle complete: {len(result.get('buys_executed', []))} buys, {len(result.get('sells_executed', []))} sells")
            # Note: run_ai_trading_cycle already calls take_portfolio_snapshot internally
        except Exception as e:
            logger.error(f"AI trading cycle error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            db.close()

    background_tasks.add_task(run_cycle_background)
    return {"status": "started", "message": "Trading cycle started in background. Refresh page in 15-20 seconds."}


# ============== Watchlist ==============

@app.get("/api/watchlist")
async def get_watchlist(db: Session = Depends(get_db)):
    """Get watchlist with current stock data"""
    watchlist = db.query(Watchlist).all()

    # Batch-fetch all stocks to avoid N+1 queries
    tickers = [w.ticker for w in watchlist]
    stocks_by_ticker = {}
    if tickers:
        stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all()
        stocks_by_ticker = {s.ticker: s for s in stocks}

    items = []
    for w in watchlist:
        stock = stocks_by_ticker.get(w.ticker)
        items.append({
            "id": w.id,
            "ticker": w.ticker,
            "added_at": w.added_at.isoformat() if w.added_at else None,
            "notes": w.notes,
            "target_price": w.target_price,
            "alert_score": w.alert_score,
            "current_price": stock.current_price if stock else None,
            "canslim_score": stock.canslim_score if stock else None,
            "projected_growth": stock.projected_growth if stock else None
        })

    return {"items": items}


@app.post("/api/watchlist")
async def add_to_watchlist(data: WatchlistCreate, db: Session = Depends(get_db)):
    """Add stock to watchlist"""
    ticker = data.ticker.upper()

    existing = db.query(Watchlist).filter(Watchlist.ticker == ticker).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"{ticker} already in watchlist")

    item = Watchlist(
        ticker=ticker,
        notes=data.notes,
        target_price=data.target_price,
        alert_score=data.alert_score
    )
    db.add(item)
    db.commit()

    return {"message": f"Added {ticker} to watchlist", "id": item.id}


@app.delete("/api/watchlist/{item_id}")
async def remove_from_watchlist(item_id: int, db: Session = Depends(get_db)):
    """Remove stock from watchlist"""
    item = db.query(Watchlist).filter(Watchlist.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")

    db.delete(item)
    db.commit()

    return {"message": f"Removed {item.ticker} from watchlist"}


class WatchlistBulkImport(BaseModel):
    """Request model for bulk watchlist import"""
    tickers: str  # Comma or whitespace separated list of tickers


@app.post("/api/watchlist/bulk")
async def bulk_add_to_watchlist(data: WatchlistBulkImport, db: Session = Depends(get_db)):
    """
    Bulk import tickers to watchlist.

    Accepts a string of tickers separated by commas, spaces, or newlines.
    Skips tickers that already exist in the watchlist.

    Returns:
        added: List of tickers successfully added
        skipped: List of tickers that were already in watchlist
        invalid: List of tickers that failed validation
    """
    import re

    # Split by commas, spaces, or newlines
    ticker_list = re.split(r'[,\s\n]+', data.tickers.upper().strip())
    # Remove empty strings and duplicates while preserving order
    ticker_list = list(dict.fromkeys(t.strip() for t in ticker_list if t.strip()))

    added = []
    skipped = []
    invalid = []

    # Get existing watchlist tickers
    existing_tickers = set(
        item.ticker for item in db.query(Watchlist.ticker).all()
    )

    for ticker in ticker_list:
        # Basic ticker validation (1-5 uppercase letters)
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            invalid.append(ticker)
            continue

        if ticker in existing_tickers:
            skipped.append(ticker)
            continue

        # Add to watchlist
        item = Watchlist(ticker=ticker)
        db.add(item)
        added.append(ticker)
        existing_tickers.add(ticker)  # Track for duplicate detection in same batch

    if added:
        db.commit()

    return {
        "message": f"Added {len(added)} tickers to watchlist",
        "added": added,
        "skipped": skipped,
        "invalid": invalid,
        "total_processed": len(ticker_list)
    }


# ============== Backtesting API ==============

class BacktestCreate(BaseModel):
    """Request model for creating a backtest"""
    name: Optional[str] = None
    start_date: date
    end_date: date
    starting_cash: float = 25000.0
    stock_universe: str = "all"  # sp500, all, custom
    custom_tickers: Optional[List[str]] = None
    max_positions: Optional[int] = None
    min_score_to_buy: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    strategy: str = "balanced"  # balanced, growth
    force_refresh: bool = False  # Force fresh FMP earnings fetch (ignore cache)


@app.post("/api/backtests")
async def create_backtest(
    config: BacktestCreate,
    db: Session = Depends(get_db)
):
    """
    Create and enqueue a new backtest.
    Returns immediately with backtest ID; simulation runs via queue worker.
    """
    # Validate dates
    if config.end_date <= config.start_date:
        raise HTTPException(400, "end_date must be after start_date")

    if (config.end_date - config.start_date).days > 1600:
        raise HTTPException(400, "Maximum backtest period is ~4.5 years (1600 days)")

    if config.end_date > date.today():
        raise HTTPException(400, "end_date cannot be in the future")

    # Read defaults from yaml config when not provided via API
    from config_loader import config as yaml_config
    default_min_score = yaml_config.get('ai_trader.allocation.min_score_to_buy', 72)
    default_stop_loss = yaml_config.get('ai_trader.stops.normal_stop_loss_pct', 10.0)

    # Create backtest run record
    strategy_label = f" [{config.strategy.upper()}]" if config.strategy != "balanced" else ""
    backtest = BacktestRun(
        name=config.name or f"{config.stock_universe.upper()} | {config.start_date} to {config.end_date} | ${config.starting_cash:,.0f}{strategy_label}",
        start_date=config.start_date,
        end_date=config.end_date,
        starting_cash=config.starting_cash,
        stock_universe=config.stock_universe,
        strategy=config.strategy,
        custom_tickers=config.custom_tickers,
        max_positions=config.max_positions or 20,
        min_score_to_buy=config.min_score_to_buy or default_min_score,
        stop_loss_pct=config.stop_loss_pct or default_stop_loss,
        force_refresh=config.force_refresh,
        status="pending"
    )
    db.add(backtest)
    db.commit()
    db.refresh(backtest)

    # Enqueue for sequential execution
    from backend.backtest_queue import backtest_queue
    backtest_queue.enqueue(backtest.id)
    queue_pos = backtest_queue.get_queue_position(backtest.id)

    return {
        "id": backtest.id,
        "status": "pending",
        "message": "Backtest queued",
        "queue_position": queue_pos
    }


@app.get("/api/backtests")
async def list_backtests(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """List all backtests, most recent first"""
    from backend.backtest_queue import backtest_queue
    backtests = db.query(BacktestRun).order_by(desc(BacktestRun.created_at)).limit(limit).all()

    return [
        {
            "id": b.id,
            "name": b.name,
            "status": b.status,
            "start_date": b.start_date.isoformat() if b.start_date else None,
            "end_date": b.end_date.isoformat() if b.end_date else None,
            "starting_cash": b.starting_cash,
            "final_value": b.final_value,
            "total_return_pct": b.total_return_pct,
            "spy_return_pct": b.spy_return_pct,
            "max_drawdown_pct": b.max_drawdown_pct,
            "total_trades": b.total_trades,
            "win_rate": b.win_rate,
            "sharpe_ratio": b.sharpe_ratio,
            "strategy": getattr(b, 'strategy', None) or "balanced",
            "progress_pct": b.progress_pct,
            "queue_position": backtest_queue.get_queue_position(b.id) if b.status == "pending" else None,
            "created_at": b.created_at.isoformat() + "Z" if b.created_at else None,
            "completed_at": b.completed_at.isoformat() + "Z" if b.completed_at else None
        }
        for b in backtests
    ]


BACKTEST_PRESETS = [
    {"name": "2022 Bear Market", "start": "2022-01-01", "end": "2022-12-31", "desc": "S&P 500 -19.4%"},
    {"name": "2020 COVID Crash", "start": "2020-02-01", "end": "2020-12-31", "desc": "V-shaped recovery"},
    {"name": "2023-24 Recovery", "start": "2023-01-01", "end": "2024-01-01", "desc": "AI-led rally"},
    {"name": "2024-25 Bull", "start": "2024-01-01", "end": "2025-01-01", "desc": "Broadening market"},
    {"name": "Full Year 2025", "start": "2025-01-01", "end": "2026-01-01", "desc": "Recent performance"},
]


@app.get("/api/backtests/compare")
async def compare_backtests(
    ids: str = Query(..., description="Comma-separated backtest IDs"),
    db: Session = Depends(get_db)
):
    """Compare 2+ backtests side-by-side with overlaid equity curves"""
    try:
        bt_ids = [int(x.strip()) for x in ids.split(",")]
    except ValueError:
        raise HTTPException(400, "ids must be comma-separated integers")

    if len(bt_ids) < 2:
        raise HTTPException(400, "Need at least 2 backtest IDs to compare")

    backtests_data = []
    all_dates = set()

    for bt_id in bt_ids:
        bt = db.query(BacktestRun).get(bt_id)
        if not bt or bt.status != "completed":
            continue

        snapshots = db.query(BacktestSnapshot).filter(
            BacktestSnapshot.backtest_id == bt_id
        ).order_by(BacktestSnapshot.date).all()

        bt_info = {
            "id": bt.id, "name": bt.name,
            "total_return_pct": bt.total_return_pct,
            "spy_return_pct": bt.spy_return_pct,
            "max_drawdown_pct": bt.max_drawdown_pct,
            "sharpe_ratio": bt.sharpe_ratio,
            "win_rate": bt.win_rate,
            "total_trades": bt.total_trades,
            "start_date": bt.start_date.isoformat() if bt.start_date else None,
            "end_date": bt.end_date.isoformat() if bt.end_date else None,
        }
        bt_info["snapshots"] = {s.date.isoformat(): s.cumulative_return_pct for s in snapshots}
        backtests_data.append(bt_info)

        for s in snapshots:
            all_dates.add(s.date.isoformat())

    if len(backtests_data) < 2:
        raise HTTPException(400, "Need at least 2 completed backtests")

    # Build chart data
    sorted_dates = sorted(all_dates)
    chart_data = []
    for d in sorted_dates:
        point = {"date": d}
        for bt in backtests_data:
            point[f"bt_{bt['id']}_return"] = bt["snapshots"].get(d)
        # Use SPY from first backtest
        if backtests_data[0]["snapshots"].get(d) is not None:
            first_bt_id = backtests_data[0]["id"]
            first_snaps = db.query(BacktestSnapshot).filter(
                BacktestSnapshot.backtest_id == first_bt_id,
                BacktestSnapshot.date == d
            ).first()
            point["spy_return"] = first_snaps.spy_return_pct if first_snaps else None
        chart_data.append(point)

    stat_keys = ["total_return_pct", "spy_return_pct", "max_drawdown_pct", "sharpe_ratio", "win_rate", "total_trades"]
    stats_table = {}
    for key in stat_keys:
        stats_table[key] = [bt.get(key) for bt in backtests_data]

    for bt in backtests_data:
        del bt["snapshots"]

    return {
        "backtests": backtests_data,
        "chart_data": chart_data,
        "stats_table": stats_table
    }


@app.get("/api/backtests/presets")
async def get_backtest_presets():
    """Get preset backtest periods for multi-period testing"""
    return BACKTEST_PRESETS


@app.post("/api/backtests/multi")
async def create_multi_backtest(
    starting_cash: float = Query(25000.0, ge=1000, le=1000000),
    stock_universe: str = Query("all"),
    strategy: str = Query("balanced"),
    force_refresh: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Launch backtests for all preset periods sequentially via queue"""
    from backend.backtest_queue import backtest_queue

    strategy_label = f" [{strategy.upper()}]" if strategy != "balanced" else ""
    created_ids = []
    for preset in BACKTEST_PRESETS:
        bt = BacktestRun(
            name=f"{preset['name']}{strategy_label}",
            status="pending",
            start_date=datetime.strptime(preset["start"], "%Y-%m-%d").date(),
            end_date=datetime.strptime(preset["end"], "%Y-%m-%d").date(),
            starting_cash=starting_cash,
            stock_universe=stock_universe,
            strategy=strategy,
            force_refresh=force_refresh,
            created_at=datetime.now(timezone.utc)
        )
        db.add(bt)
        db.flush()
        created_ids.append(bt.id)

    db.commit()

    # Enqueue all sequentially — cache warming benefits each subsequent run
    for bt_id in created_ids:
        backtest_queue.enqueue(bt_id)

    return {"message": f"Queued {len(created_ids)} backtests", "backtest_ids": created_ids}


class BatchBacktestCreate(BaseModel):
    """Request model for creating a batch of backtests across strategies"""
    strategies: List[str]
    start_date: date
    end_date: date
    starting_cash: float = 25000.0
    stock_universe: str = "all"
    force_refresh: bool = False


@app.post("/api/backtests/batch")
async def create_batch_backtest(
    config: BatchBacktestCreate,
    db: Session = Depends(get_db)
):
    """Create one backtest per strategy for the same date range, enqueued sequentially for cache reuse."""
    from backend.backtest_queue import backtest_queue

    if config.end_date <= config.start_date:
        raise HTTPException(400, "end_date must be after start_date")
    if (config.end_date - config.start_date).days > 1600:
        raise HTTPException(400, "Maximum backtest period is ~4.5 years (1600 days)")
    if config.end_date > date.today():
        raise HTTPException(400, "end_date cannot be in the future")

    from config_loader import config as yaml_config
    default_min_score = yaml_config.get('ai_trader.allocation.min_score_to_buy', 72)
    default_stop_loss = yaml_config.get('ai_trader.stops.normal_stop_loss_pct', 10.0)

    created_ids = []
    for strategy in config.strategies:
        strategy_label = f" [{strategy.upper()}]" if strategy != "balanced" else ""
        bt = BacktestRun(
            name=f"{config.stock_universe.upper()} | {config.start_date} to {config.end_date} | ${config.starting_cash:,.0f}{strategy_label}",
            start_date=config.start_date,
            end_date=config.end_date,
            starting_cash=config.starting_cash,
            stock_universe=config.stock_universe,
            strategy=strategy,
            min_score_to_buy=default_min_score,
            stop_loss_pct=default_stop_loss,
            force_refresh=config.force_refresh,
            status="pending",
            created_at=datetime.now(timezone.utc)
        )
        db.add(bt)
        db.flush()
        created_ids.append(bt.id)

    db.commit()

    for bt_id in created_ids:
        backtest_queue.enqueue(bt_id)

    return {
        "message": f"Queued {len(created_ids)} backtests for {len(config.strategies)} strategies",
        "backtest_ids": created_ids
    }


@app.get("/api/backtests/queue")
async def get_backtest_queue():
    """Return current queue state: running backtest + queued IDs."""
    from backend.backtest_queue import backtest_queue
    return backtest_queue.get_queue_snapshot()


@app.get("/api/backtests/{backtest_id}")
async def get_backtest(backtest_id: int, db: Session = Depends(get_db)):
    """Get detailed backtest results including performance chart data"""
    backtest = db.query(BacktestRun).get(backtest_id)
    if not backtest:
        raise HTTPException(404, "Backtest not found")

    # Get daily snapshots for chart
    snapshots = db.query(BacktestSnapshot).filter(
        BacktestSnapshot.backtest_id == backtest_id
    ).order_by(BacktestSnapshot.date).all()

    # Get trades
    trades = db.query(BacktestTrade).filter(
        BacktestTrade.backtest_id == backtest_id
    ).order_by(BacktestTrade.date).all()

    return {
        "backtest": {
            "id": backtest.id,
            "name": backtest.name,
            "status": backtest.status,
            "start_date": backtest.start_date.isoformat() if backtest.start_date else None,
            "end_date": backtest.end_date.isoformat() if backtest.end_date else None,
            "starting_cash": backtest.starting_cash,
            "final_value": backtest.final_value,
            "total_return_pct": backtest.total_return_pct,
            "spy_return_pct": backtest.spy_return_pct,
            "max_drawdown_pct": backtest.max_drawdown_pct,
            "total_trades": backtest.total_trades,
            "win_rate": backtest.win_rate,
            "sharpe_ratio": backtest.sharpe_ratio,
            "strategy": getattr(backtest, 'strategy', None) or "balanced",
            "progress_pct": backtest.progress_pct,
            "error_message": backtest.error_message,
            "created_at": backtest.created_at.isoformat() + "Z" if backtest.created_at else None,
            "completed_at": backtest.completed_at.isoformat() + "Z" if backtest.completed_at else None
        },
        "performance_chart": [
            {
                "date": s.date.isoformat(),
                "value": s.total_value,
                "return_pct": s.cumulative_return_pct,
                "spy_value": s.spy_value,
                "spy_return_pct": s.spy_return_pct
            } for s in snapshots
        ],
        "trades": [
            {
                "date": t.date.isoformat(),
                "ticker": t.ticker,
                "action": t.action,
                "shares": t.shares,
                "price": t.price,
                "value": t.total_value,
                "reason": t.reason,
                "score": t.canslim_score,
                "gain_pct": t.realized_gain_pct,
                "holding_days": t.holding_days,
                "realized_gain": t.realized_gain,
                "signal_factors": t.signal_factors
            } for t in trades
        ],
        "statistics": {
            "total_buys": len([t for t in trades if t.action == "BUY"]),
            "total_sells": len([t for t in trades if t.action == "SELL"]),
            "total_pyramids": len([t for t in trades if t.action == "PYRAMID"]),
            "avg_holding_days": sum(t.holding_days or 0 for t in trades if t.action == "SELL") / max(1, len([t for t in trades if t.action == "SELL"])),
            "best_trade": max((t.realized_gain_pct or 0 for t in trades if t.action == "SELL"), default=0),
            "worst_trade": min((t.realized_gain_pct or 0 for t in trades if t.action == "SELL"), default=0)
        }
    }


@app.get("/api/backtests/{backtest_id}/status")
async def get_backtest_status(backtest_id: int, db: Session = Depends(get_db)):
    """Get backtest progress (for polling during run)"""
    from backend.backtest_queue import backtest_queue
    backtest = db.query(BacktestRun).get(backtest_id)
    if not backtest:
        raise HTTPException(404, "Backtest not found")

    return {
        "id": backtest.id,
        "status": backtest.status,
        "progress_pct": backtest.progress_pct,
        "queue_position": backtest_queue.get_queue_position(backtest.id) if backtest.status == "pending" else None,
        "error": backtest.error_message,
        "completed_at": backtest.completed_at.isoformat() + "Z" if backtest.completed_at else None
    }


@app.delete("/api/backtests/{backtest_id}")
async def delete_backtest(backtest_id: int, db: Session = Depends(get_db)):
    """Delete a backtest and all associated data"""
    backtest = db.query(BacktestRun).get(backtest_id)
    if not backtest:
        raise HTTPException(404, "Backtest not found")

    db.delete(backtest)  # Cascade deletes snapshots, trades, positions
    db.commit()

    return {"message": "Backtest deleted"}


@app.post("/api/backtests/{backtest_id}/cancel")
async def cancel_backtest(backtest_id: int, db: Session = Depends(get_db)):
    """Cancel a running backtest - handles stuck backtests that may have lost their process"""
    backtest = db.query(BacktestRun).get(backtest_id)
    if not backtest:
        raise HTTPException(404, "Backtest not found")

    if backtest.status not in ("pending", "running"):
        raise HTTPException(400, f"Cannot cancel backtest with status: {backtest.status}")

    # Set the flag so the backtester loop picks it up if still running
    backtest.cancel_requested = True

    # Also force-cancel immediately — the background thread may be stuck in data loading
    # or the process may be dead. The backtester's own cancel check will be a no-op
    # if we've already set the status here.
    backtest.status = "cancelled"
    backtest.completed_at = datetime.now(timezone.utc)
    backtest.error_message = "Cancelled by user"
    db.commit()
    logger.info(f"Backtest {backtest_id} cancelled by user (was at {backtest.progress_pct or 0:.0f}%)")
    return {
        "message": "Backtest cancelled",
        "id": backtest.id,
        "status": "cancelled"
    }


# ============== Trade Analytics ==============

@app.get("/api/analytics/trades")
async def get_trade_analytics(db: Session = Depends(get_db)):
    """Analyze historical trade performance with breakdowns"""
    trades = db.query(AIPortfolioTrade).all()

    if not trades:
        return {"summary": {}, "by_sector": [], "monthly_pnl": [], "by_entry_type": []}

    sells = [t for t in trades if t.action == "SELL"]
    buys = [t for t in trades if t.action == "BUY"]

    # Summary stats
    wins = [t for t in sells if (t.realized_gain or 0) > 0]
    losses = [t for t in sells if (t.realized_gain or 0) < 0]
    total_gains = sum(t.realized_gain or 0 for t in wins)
    total_losses = abs(sum(t.realized_gain or 0 for t in losses))

    summary = {
        "total_trades": len(trades),
        "total_buys": len(buys),
        "total_sells": len(sells),
        "win_rate": (len(wins) / len(sells) * 100) if sells else 0,
        "avg_gain_pct": sum(((t.price / t.cost_basis - 1) * 100) if t.cost_basis and t.cost_basis > 0 else 0 for t in wins) / max(1, len(wins)),
        "avg_loss_pct": sum(((t.price / t.cost_basis - 1) * 100) if t.cost_basis and t.cost_basis > 0 else 0 for t in losses) / max(1, len(losses)),
        "profit_factor": (total_gains / total_losses) if total_losses > 0 else float('inf') if total_gains > 0 else 0,
        "total_realized": sum(t.realized_gain or 0 for t in sells),
    }

    # Batch-fetch all stocks for sector lookup (avoids N+1 queries)
    sell_tickers = list(set(t.ticker for t in sells))
    stocks_by_ticker = {}
    if sell_tickers:
        _stocks = db.query(Stock).filter(Stock.ticker.in_(sell_tickers)).all()
        stocks_by_ticker = {s.ticker: s for s in _stocks}

    # By sector
    sector_data = {}
    for t in sells:
        stock = stocks_by_ticker.get(t.ticker)
        sector = getattr(stock, 'sector', 'Unknown') or 'Unknown' if stock else 'Unknown'
        if sector not in sector_data:
            sector_data[sector] = {"trades": 0, "wins": 0, "pnl": 0}
        sector_data[sector]["trades"] += 1
        sector_data[sector]["pnl"] += t.realized_gain or 0
        if (t.realized_gain or 0) > 0:
            sector_data[sector]["wins"] += 1

    by_sector = [
        {"sector": s, "trades": d["trades"],
         "win_rate": (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else 0,
         "pnl": d["pnl"]}
        for s, d in sorted(sector_data.items(), key=lambda x: x[1]["pnl"], reverse=True)
    ]

    # Monthly P&L
    monthly = {}
    for t in sells:
        if t.executed_at:
            month_key = t.executed_at.strftime("%Y-%m")
            if month_key not in monthly:
                monthly[month_key] = {"month": month_key, "pnl": 0, "trades": 0}
            monthly[month_key]["pnl"] += t.realized_gain or 0
            monthly[month_key]["trades"] += 1

    monthly_pnl = sorted(monthly.values(), key=lambda x: x["month"])

    # By entry type (from signal_factors)
    entry_types = {}
    for t in sells:
        factors = t.signal_factors if hasattr(t, 'signal_factors') and t.signal_factors else {}
        entry_type = factors.get("entry_type", "unknown") if isinstance(factors, dict) else "unknown"
        if entry_type not in entry_types:
            entry_types[entry_type] = {"trades": 0, "wins": 0, "pnl": 0}
        entry_types[entry_type]["trades"] += 1
        entry_types[entry_type]["pnl"] += t.realized_gain or 0
        if (t.realized_gain or 0) > 0:
            entry_types[entry_type]["wins"] += 1

    by_entry_type = [
        {"entry_type": et, "trades": d["trades"],
         "win_rate": (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else 0,
         "pnl": d["pnl"]}
        for et, d in sorted(entry_types.items(), key=lambda x: x[1]["pnl"], reverse=True)
    ]

    # By sell reason (from signal_factors.sell_reason)
    sell_reasons = {}
    for t in sells:
        factors = t.signal_factors if hasattr(t, 'signal_factors') and t.signal_factors else {}
        reason = factors.get("sell_reason", "unknown") if isinstance(factors, dict) else "unknown"
        if reason not in sell_reasons:
            sell_reasons[reason] = {"trades": 0, "wins": 0, "pnl": 0}
        sell_reasons[reason]["trades"] += 1
        sell_reasons[reason]["pnl"] += t.realized_gain or 0
        if (t.realized_gain or 0) > 0:
            sell_reasons[reason]["wins"] += 1

    by_sell_reason = [
        {"sell_reason": sr, "trades": d["trades"],
         "win_rate": (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else 0,
         "pnl": round(d["pnl"], 2)}
        for sr, d in sorted(sell_reasons.items(), key=lambda x: x[1]["pnl"], reverse=True)
    ]

    # By hold duration (bucket sells by days held)
    duration_buckets = {"0-7d": {"trades": 0, "wins": 0, "pnl": 0},
                        "7-30d": {"trades": 0, "wins": 0, "pnl": 0},
                        "30-90d": {"trades": 0, "wins": 0, "pnl": 0},
                        "90d+": {"trades": 0, "wins": 0, "pnl": 0}}
    # Build a map of most recent BUY per ticker for hold duration calculation
    buy_dates = {}
    for t in sorted(buys, key=lambda x: x.executed_at or datetime.min):
        buy_dates[t.ticker] = t.executed_at

    for t in sells:
        buy_date = buy_dates.get(t.ticker)
        if buy_date and t.executed_at:
            days_held = (t.executed_at - buy_date).days
            if days_held < 7:
                bucket = "0-7d"
            elif days_held < 30:
                bucket = "7-30d"
            elif days_held < 90:
                bucket = "30-90d"
            else:
                bucket = "90d+"
        else:
            bucket = "0-7d"  # Default if dates missing

        duration_buckets[bucket]["trades"] += 1
        duration_buckets[bucket]["pnl"] += t.realized_gain or 0
        if (t.realized_gain or 0) > 0:
            duration_buckets[bucket]["wins"] += 1

    by_hold_duration = [
        {"duration": d, "trades": b["trades"],
         "win_rate": (b["wins"] / b["trades"] * 100) if b["trades"] > 0 else 0,
         "pnl": round(b["pnl"], 2)}
        for d, b in duration_buckets.items() if b["trades"] > 0
    ]

    return {
        "summary": summary,
        "by_sector": by_sector,
        "monthly_pnl": monthly_pnl,
        "by_entry_type": by_entry_type,
        "by_sell_reason": by_sell_reason,
        "by_hold_duration": by_hold_duration
    }


# ============== Earnings Calendar ==============

@app.get("/api/ai-portfolio/earnings-calendar")
async def get_earnings_calendar(db: Session = Depends(get_db)):
    """Get upcoming earnings dates for AI portfolio positions"""
    from backend.ai_trader import get_or_create_config
    config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()

    earnings_data = []
    counts = {"high": 0, "medium": 0, "low": 0}

    for pos in positions:
        stock = db.query(Stock).filter(Stock.ticker == pos.ticker).first()
        if not stock:
            continue

        days = stock.days_to_earnings
        if days is None:
            continue

        if days < 7:
            risk = "high"
        elif days <= 14:
            risk = "medium"
        else:
            risk = "low"

        counts[risk] += 1
        gain_pct = ((pos.current_price / pos.cost_basis) - 1) * 100 if pos.cost_basis and pos.cost_basis > 0 else 0

        earnings_data.append({
            "ticker": pos.ticker,
            "days_to_earnings": days,
            "next_earnings_date": stock.next_earnings_date.isoformat() if stock.next_earnings_date else None,
            "beat_streak": stock.earnings_beat_streak or 0,
            "risk_level": risk,
            "gain_pct": round(gain_pct, 1),
            "current_price": pos.current_price,
            "shares": pos.shares,
        })

    # Sort by days to earnings (soonest first)
    earnings_data.sort(key=lambda x: x["days_to_earnings"])

    return {
        "positions": earnings_data,
        "upcoming_count": counts
    }


# ============== Portfolio Risk Monitor ==============

@app.get("/api/ai-portfolio/risk")
async def get_portfolio_risk(db: Session = Depends(get_db)):
    """Get current portfolio risk metrics"""
    from backend.ai_trader import get_or_create_config, get_portfolio_value
    from config_loader import config as yaml_config

    config = get_or_create_config(db)
    portfolio = get_portfolio_value(db)
    positions = db.query(AIPortfolioPosition).all()
    pv = portfolio["total_value"]

    # Calculate portfolio heat
    stop_cfg = yaml_config.get('ai_trader.stops', {})
    base_stop = stop_cfg.get('normal_stop_loss_pct', 8.0)
    total_heat = 0.0
    stop_distances = []

    for pos in positions:
        if pos.current_price and pos.current_price > 0 and pos.cost_basis and pos.cost_basis > 0 and pv > 0:
            pos_pct = ((pos.current_value or 0) / pv) * 100
            g_pct = ((pos.current_price - pos.cost_basis) / pos.cost_basis) * 100
            dist = base_stop + g_pct
            total_heat += pos_pct * (dist / 100)
            stop_distances.append({
                "ticker": pos.ticker,
                "distance_pct": round(dist, 1),
                "gain_pct": round(g_pct, 1),
                "position_pct": round(pos_pct, 1),
            })

    heat_status = "normal" if total_heat < 10 else ("warning" if total_heat < 15 else "danger")

    # Sector concentration
    sector_data = {}
    for pos in positions:
        stock = db.query(Stock).filter(Stock.ticker == pos.ticker).first()
        sector = getattr(stock, 'sector', 'Unknown') or 'Unknown' if stock else 'Unknown'
        if sector not in sector_data:
            sector_data[sector] = {"count": 0, "value": 0}
        sector_data[sector]["count"] += 1
        sector_data[sector]["value"] += pos.current_value or 0

    sector_concentration = [
        {"sector": s, "count": d["count"], "pct": round((d["value"] / pv * 100) if pv > 0 else 0, 1)}
        for s, d in sorted(sector_data.items(), key=lambda x: x[1]["value"], reverse=True)
    ]

    # Position size alerts
    max_pos_pct = config.max_position_pct or 25.0
    position_alerts = []
    for pos in positions:
        if pv > 0:
            pos_pct = ((pos.current_value or 0) / pv) * 100
            if pos_pct > max_pos_pct * 0.9:
                position_alerts.append({
                    "ticker": pos.ticker,
                    "pct": round(pos_pct, 1),
                    "near_limit": True
                })

    return {
        "portfolio_heat": round(total_heat, 1),
        "heat_status": heat_status,
        "sector_concentration": sector_concentration,
        "position_alerts": position_alerts,
        "stop_distances": sorted(stop_distances, key=lambda x: x["distance_pct"]),
    }


# ============== Paper Trading Mode ==============

@app.patch("/api/ai-portfolio/config")
async def update_ai_portfolio_config_v2(
    is_active: bool = Query(None),
    min_score_to_buy: int = Query(None, ge=50, le=100),
    sell_score_threshold: int = Query(None, ge=20, le=80),
    take_profit_pct: float = Query(None, ge=10, le=100),
    stop_loss_pct: float = Query(None, ge=5, le=50),
    paper_mode: bool = Query(None),
    strategy: str = Query(None),
    db: Session = Depends(get_db)
):
    """Update AI Portfolio configuration including paper mode and strategy"""
    from backend.ai_trader import get_or_create_config
    config = get_or_create_config(db)

    if is_active is not None:
        config.is_active = is_active
    if min_score_to_buy is not None:
        config.min_score_to_buy = min_score_to_buy
    if sell_score_threshold is not None:
        config.sell_score_threshold = sell_score_threshold
    if take_profit_pct is not None:
        config.take_profit_pct = take_profit_pct
    if stop_loss_pct is not None:
        config.stop_loss_pct = stop_loss_pct
    if paper_mode is not None:
        config.paper_mode = paper_mode
    if strategy is not None:
        validate_strategy_name(strategy)
        config.strategy = strategy

    db.commit()

    return {
        "message": "Config updated",
        "config": {
            "is_active": config.is_active,
            "min_score_to_buy": config.min_score_to_buy,
            "sell_score_threshold": config.sell_score_threshold,
            "take_profit_pct": config.take_profit_pct,
            "stop_loss_pct": config.stop_loss_pct,
            "paper_mode": getattr(config, 'paper_mode', False) or False,
            "strategy": getattr(config, 'strategy', None) or "balanced",
        }
    }


# ============== Strategy Profiles ==============

@app.get("/api/strategies")
async def list_strategies():
    """List all available strategy profiles with their descriptions."""
    from config_loader import config as yaml_config
    profiles = yaml_config.get('strategy_profiles', {})
    result = []
    for name, profile in profiles.items():
        result.append({
            "name": name,
            "label": profile.get("label", name.replace("_", " ").title()),
            "description": profile.get("description", ""),
            "min_score": profile.get("min_score", 72),
            "max_positions": profile.get("max_positions", 8),
            "stop_loss_pct": profile.get("stop_loss_pct", 8.0),
            "take_profit_pct": profile.get("take_profit_pct", 75.0),
            "market_state_enabled": profile.get("market_state", {}).get("enabled", True) if isinstance(profile.get("market_state"), dict) else True,
            "seed_count": profile.get("seed_count", 3),
        })
    return result


# ============== Earnings Audit ==============

@app.get("/api/earnings-audit")
async def get_earnings_audits(
    limit: int = Query(30, ge=1, le=100),
    min_confidence: float = Query(0, ge=0, le=100),
    db: Session = Depends(get_db)
):
    """Get recent earnings audit results, sorted by confidence score."""
    from backend.database import EarningsAudit
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    query = db.query(EarningsAudit).filter(
        EarningsAudit.audited_at >= cutoff,
    )
    if min_confidence > 0:
        query = query.filter(EarningsAudit.fundamental_confidence >= min_confidence)

    audits = query.order_by(EarningsAudit.fundamental_confidence.desc()).limit(limit).all()

    return {
        "audits": [
            {
                "ticker": a.ticker,
                "fundamental_confidence": a.fundamental_confidence,
                "confidence_breakdown": a.confidence_breakdown,
                "analyst_upside_pct": a.analyst_upside_pct,
                "analyst_avg_target": a.analyst_avg_target,
                "analyst_num": a.analyst_num,
                "beat_streak": a.beat_streak,
                "avg_beat_magnitude": round(a.avg_beat_magnitude, 1) if a.avg_beat_magnitude else None,
                "roe": round(a.roe * 100, 1) if a.roe else None,
                "debt_to_equity": round(a.debt_to_equity, 2) if a.debt_to_equity else None,
                "free_cash_flow_per_share": round(a.free_cash_flow_per_share, 2) if a.free_cash_flow_per_share else None,
                "insider_net_value": a.insider_net_value,
                "insider_cluster_buys": a.insider_cluster_buys,
                "eps_revision_pct": a.eps_revision_pct,
                "price_at_audit": a.price_at_audit,
                "audited_at": a.audited_at.isoformat() + "Z" if a.audited_at else None,
            }
            for a in audits
        ],
        "count": len(audits),
    }


@app.get("/api/earnings-audit/{ticker}")
async def get_earnings_audit_ticker(ticker: str, db: Session = Depends(get_db)):
    """Get the latest earnings audit for a specific ticker."""
    from backend.database import EarningsAudit

    ticker = validate_ticker_param(ticker)
    audit = db.query(EarningsAudit).filter(
        EarningsAudit.ticker == ticker,
    ).order_by(EarningsAudit.audited_at.desc()).first()

    if not audit:
        raise HTTPException(status_code=404, detail=f"No audit found for {ticker}")

    return {
        "ticker": audit.ticker,
        "fundamental_confidence": audit.fundamental_confidence,
        "confidence_breakdown": audit.confidence_breakdown,
        "analyst_upside_pct": audit.analyst_upside_pct,
        "analyst_avg_target": audit.analyst_avg_target,
        "analyst_high_target": audit.analyst_high_target,
        "analyst_low_target": audit.analyst_low_target,
        "analyst_num": audit.analyst_num,
        "beat_streak": audit.beat_streak,
        "avg_beat_magnitude": round(audit.avg_beat_magnitude, 1) if audit.avg_beat_magnitude else None,
        "last_beat_pct": round(audit.last_beat_pct, 1) if audit.last_beat_pct else None,
        "roe": round(audit.roe * 100, 1) if audit.roe else None,
        "debt_to_equity": round(audit.debt_to_equity, 2) if audit.debt_to_equity else None,
        "free_cash_flow_per_share": round(audit.free_cash_flow_per_share, 2) if audit.free_cash_flow_per_share else None,
        "current_ratio": round(audit.current_ratio, 2) if audit.current_ratio else None,
        "insider_net_value": audit.insider_net_value,
        "insider_cluster_buys": audit.insider_cluster_buys,
        "eps_revision_pct": audit.eps_revision_pct,
        "revenue_revision_pct": audit.revenue_revision_pct,
        "price_at_audit": audit.price_at_audit,
        "audited_at": audit.audited_at.isoformat() + "Z" if audit.audited_at else None,
    }


# ============== Command Center ==============

@app.get("/api/command-center")
async def get_command_center(db: Session = Depends(get_db)):
    """
    Consolidated endpoint for the Command Center dashboard.
    Returns market regime, portfolio summary, positions, risk alerts,
    earnings calendar, top candidates, recent trades, and scanner status
    in a single API call.
    """
    from backend.ai_trader import get_or_create_config, get_portfolio_value
    from config_loader import config as yaml_config
    from backend.scheduler import get_scan_status

    config = get_or_create_config(db)
    portfolio = get_portfolio_value(db)
    positions = db.query(AIPortfolioPosition).all()
    pv = portfolio["total_value"]

    # --- 1. Market Regime ---
    latest_market = db.query(MarketSnapshot).order_by(
        desc(MarketSnapshot.date)
    ).first()

    market_data = {}
    if latest_market:
        spy_above_50 = latest_market.spy_price > latest_market.spy_50_ma if latest_market.spy_50_ma else None
        spy_above_200 = latest_market.spy_price > latest_market.spy_200_ma if latest_market.spy_200_ma else None
        regime = "bullish" if spy_above_50 and spy_above_200 else ("bearish" if not spy_above_50 else "neutral")

        # Approximate market state from signals (simple heuristic)
        ws = latest_market.weighted_signal or 0
        if ws >= 1.0:
            approx_state = "TRENDING"
        elif ws >= 0.3:
            approx_state = "CONFIRMED"
        elif ws >= -0.3:
            approx_state = "PRESSURE"
        else:
            approx_state = "CORRECTION"

        market_data = {
            "regime": regime,
            "market_state": approx_state,
            "spy": {
                "price": latest_market.spy_price,
                "ma50": latest_market.spy_50_ma,
                "ma200": latest_market.spy_200_ma,
                "signal": latest_market.spy_signal,
            },
            "qqq": {
                "price": latest_market.qqq_price,
                "ma50": latest_market.qqq_50_ma,
                "signal": getattr(latest_market, 'qqq_signal', None),
            },
            "dia": {
                "price": latest_market.dia_price,
                "ma50": latest_market.dia_50_ma,
                "signal": getattr(latest_market, 'dia_signal', None),
            },
            "weighted_signal": latest_market.weighted_signal,
            "updated_at": latest_market.created_at.isoformat() + "Z" if latest_market.created_at else None,
        }
    else:
        market_data = {"regime": "unknown"}

    # --- 2. Portfolio Summary ---
    portfolio_summary = {
        "total_value": portfolio["total_value"],
        "cash": portfolio["cash"],
        "invested": portfolio["positions_value"],
        "positions_value": portfolio["positions_value"],
        "total_return": portfolio.get("total_return", 0),
        "total_return_pct": portfolio.get("total_return_pct", 0),
        "positions_count": len(positions),
        "max_positions": config.max_positions,
        "strategy": getattr(config, 'strategy', None) or "balanced",
        "paper_mode": getattr(config, 'paper_mode', False) or False,
        "is_active": config.is_active,
    }

    # --- 3. Performance sparkline (last 30 days) ---
    from datetime import timedelta as td
    sparkline_cutoff = datetime.now(timezone.utc) - td(days=30)
    snapshots = db.query(AIPortfolioSnapshot).filter(
        AIPortfolioSnapshot.timestamp >= sparkline_cutoff
    ).order_by(AIPortfolioSnapshot.timestamp).all()

    # Deduplicate to 1 per day for sparkline
    daily_values = {}
    for snap in snapshots:
        day_key = snap.timestamp.date() if snap.timestamp else None
        if day_key:
            daily_values[day_key] = snap.total_value
    sparkline = [{"date": d.isoformat(), "value": round(v, 2)} for d, v in sorted(daily_values.items())]

    # --- 4. Active Positions (dense) ---
    position_tickers = [p.ticker for p in positions]
    stocks_map = {}
    if position_tickers:
        pos_stocks = db.query(Stock).filter(Stock.ticker.in_(position_tickers)).all()
        stocks_map = {s.ticker: s for s in pos_stocks}

    positions_data = []
    stop_cfg = yaml_config.get('ai_trader.stops', {})
    base_stop = stop_cfg.get('normal_stop_loss_pct', 8.0)
    total_heat = 0.0

    for p in positions:
        stock = stocks_map.get(p.ticker)
        g_pct = ((p.current_price - p.cost_basis) / p.cost_basis * 100) if p.cost_basis and p.cost_basis > 0 else 0
        pos_pct = ((p.current_value or 0) / pv * 100) if pv > 0 else 0
        stop_dist = base_stop + g_pct

        total_heat += pos_pct * (stop_dist / 100) if pv > 0 else 0

        # Trailing stop distance
        trail_pct = None
        if p.peak_price and p.current_price and p.peak_price > 0:
            trail_pct = round(((p.peak_price - p.current_price) / p.peak_price) * 100, 1)

        positions_data.append({
            "ticker": p.ticker,
            "shares": p.shares,
            "cost_basis": p.cost_basis,
            "current_price": p.current_price,
            "gain_pct": round(g_pct, 1),
            "position_pct": round(pos_pct, 1),
            "score": stock.canslim_score if stock else None,
            "stop_distance": round(stop_dist, 1),
            "trail_from_peak": trail_pct,
            "days_held": (datetime.now(timezone.utc) - p.purchase_date.replace(tzinfo=timezone.utc)).days if p.purchase_date else None,
        })

    positions_data.sort(key=lambda x: x["gain_pct"], reverse=True)

    # --- 5. Top Buy Candidates (with audit data) ---
    top_candidates = db.query(Stock).filter(
        Stock.canslim_score >= 65,
        Stock.current_price > 0,
        ~Stock.ticker.in_(position_tickers) if position_tickers else True,
    ).order_by(desc(Stock.canslim_score)).limit(8).all()

    # Get audit data for candidates
    audit_cutoff = datetime.now(timezone.utc) - td(hours=48)
    candidate_tickers = [s.ticker for s in top_candidates]
    recent_audits = {}
    if candidate_tickers:
        audits = db.query(EarningsAudit).filter(
            EarningsAudit.ticker.in_(candidate_tickers),
            EarningsAudit.audited_at >= audit_cutoff,
        ).all()
        for a in audits:
            if a.ticker not in recent_audits or (a.audited_at and a.audited_at > recent_audits[a.ticker].audited_at):
                recent_audits[a.ticker] = a

    candidates_data = []
    for s in filter_duplicate_stocks(top_candidates, 8):
        audit = recent_audits.get(s.ticker)
        candidates_data.append({
            "ticker": s.ticker,
            "name": s.name,
            "score": s.canslim_score,
            "price": s.current_price,
            "projected_growth": s.projected_growth,
            "audit_confidence": audit.fundamental_confidence if audit else None,
            "sector": s.sector,
        })

    # --- 6. Risk Alerts ---
    heat_status = "normal" if total_heat < 10 else ("warning" if total_heat < 15 else "danger")

    # Sector concentration
    sector_data = {}
    for p in positions:
        stock = stocks_map.get(p.ticker)
        sector = (stock.sector if stock else None) or "Unknown"
        if sector not in sector_data:
            sector_data[sector] = {"count": 0, "value": 0}
        sector_data[sector]["count"] += 1
        sector_data[sector]["value"] += p.current_value or 0

    sector_concentration = [
        {"sector": s, "count": d["count"], "pct": round((d["value"] / pv * 100) if pv > 0 else 0, 1)}
        for s, d in sorted(sector_data.items(), key=lambda x: x[1]["value"], reverse=True)
    ][:5]  # Top 5 sectors

    risk_data = {
        "portfolio_heat": round(total_heat, 1),
        "heat_status": heat_status,
        "top_sectors": sector_concentration,
    }

    # --- 7. Earnings Calendar (compact) ---
    earnings_data = []
    for p in positions:
        stock = stocks_map.get(p.ticker)
        if stock and stock.days_to_earnings is not None and stock.days_to_earnings <= 21:
            earnings_data.append({
                "ticker": p.ticker,
                "days": stock.days_to_earnings,
                "date": stock.next_earnings_date.isoformat() if stock.next_earnings_date else None,
                "beat_streak": stock.earnings_beat_streak or 0,
            })
    earnings_data.sort(key=lambda x: x["days"])

    # --- 8. Recent Trades ---
    recent_trades = db.query(AIPortfolioTrade).order_by(
        desc(AIPortfolioTrade.executed_at)
    ).limit(10).all()

    trades_data = [
        {
            "ticker": t.ticker,
            "action": t.action,
            "shares": t.shares,
            "price": t.price,
            "value": round(t.shares * t.price, 2) if t.shares and t.price else None,
            "realized_gain": round(t.realized_gain, 2) if t.realized_gain else None,
            "reason": t.reason[:80] if t.reason else None,
            "executed_at": t.executed_at.isoformat() + "Z" if t.executed_at else None,
        }
        for t in recent_trades
    ]

    # --- 9. Scanner Status ---
    scanner_status = get_scan_status()

    # --- 10. Coiled Spring ---
    cs_data = None
    try:
        # Current candidates: stocks approaching earnings with long bases + beat streaks
        cs_candidates = db.query(Stock).filter(
            Stock.days_to_earnings <= 14,
            Stock.days_to_earnings > 0,
            Stock.weeks_in_base >= 15,
            Stock.earnings_beat_streak >= 3,
            Stock.canslim_score >= 48
        ).order_by(desc(Stock.canslim_score)).limit(5).all()

        cs_candidates_data = [{
            "ticker": s.ticker,
            "score": s.canslim_score,
            "days_to_earnings": s.days_to_earnings,
            "weeks_in_base": s.weeks_in_base,
            "beat_streak": s.earnings_beat_streak,
            "base_type": s.base_type,
        } for s in cs_candidates]

        # Aggregate stats from deduped alerts
        all_deduped = _get_deduped_cs_alerts(db).all()
        cs_stats = _cs_stats(all_deduped)

        # Recent resolved alerts (last 5 with outcomes, deduped)
        deduped_resolved = _get_deduped_cs_alerts(db).filter(
            CoiledSpringAlert.outcome.isnot(None)
        ).order_by(desc(CoiledSpringAlert.outcome_updated_at)).limit(5).all()

        cs_recent = [{
            "ticker": a.ticker,
            "outcome": a.outcome,
            "price_change_pct": a.price_change_pct,
            "alert_date": a.alert_date.isoformat() if a.alert_date else None,
        } for a in deduped_resolved]

        cs_data = {
            "candidates": cs_candidates_data,
            "stats": cs_stats,
            "recent_results": cs_recent,
        }
    except Exception as e:
        logger.warning(f"Failed to load CS data for command center: {e}")

    return {
        "market": market_data,
        "portfolio": portfolio_summary,
        "sparkline": sparkline,
        "positions": positions_data,
        "candidates": candidates_data,
        "risk": risk_data,
        "earnings": earnings_data,
        "trades": trades_data,
        "scanner": scanner_status,
        "coiled_spring": cs_data,
    }


# ============== Serve Frontend ==============

frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/assets", StaticFiles(directory=frontend_path / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend for all non-API routes"""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        file_path = frontend_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_path / "index.html")

"""
CANSLIM Analyzer Web API

FastAPI backend wrapping the existing CANSLIM analysis modules.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text
from datetime import datetime, date, timedelta
from typing import Optional, List
import logging

# Add parent directory to path for importing existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import (
    init_db, get_db, Stock, StockScore, PortfolioPosition,
    Watchlist, AnalysisJob, MarketSnapshot
)
from backend.config import settings
from pydantic import BaseModel

# Import existing CANSLIM modules
from canslim_scorer import CANSLIMScorer
from data_fetcher import DataFetcher
from growth_projector import GrowthProjector
from sp500_tickers import get_sp500_tickers, get_russell2000_tickers, get_all_tickers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Request Models ==============

class PositionCreate(BaseModel):
    ticker: str
    shares: float
    cost_basis: Optional[float] = None
    notes: Optional[str] = None

class WatchlistCreate(BaseModel):
    ticker: str
    notes: Optional[str] = None
    target_price: Optional[float] = None
    alert_score: Optional[float] = None

class ScanRequest(BaseModel):
    tickers: Optional[List[str]] = None


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown"""
    logger.info("Starting CANSLIM Analyzer API...")
    init_db()
    yield
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


def update_market_snapshot(db: Session):
    """Update market direction data (SPY price, MAs, trend)"""
    import requests

    try:
        # Fetch SPY data from Yahoo chart API
        url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
        params = {"interval": "1d", "range": "1y"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.error(f"Failed to fetch SPY data: {resp.status_code}")
            return

        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            logger.error("No SPY chart data returned")
            return

        meta = result[0].get("meta", {})
        close_prices = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])

        # Filter out None values
        close_prices = [p for p in close_prices if p is not None]

        if len(close_prices) < 50:
            logger.error("Insufficient SPY price history")
            return

        current_price = meta.get("regularMarketPrice") or close_prices[-1]

        # Calculate moving averages
        ma_50 = sum(close_prices[-50:]) / 50
        ma_200 = sum(close_prices[-200:]) / 200 if len(close_prices) >= 200 else sum(close_prices) / len(close_prices)

        # Determine trend
        if current_price > ma_50 and current_price > ma_200:
            trend = "bullish"
            score = 15.0
        elif current_price > ma_200:
            trend = "neutral"
            score = 10.5
        elif current_price > ma_50:
            trend = "neutral"
            score = 7.5
        else:
            trend = "bearish"
            score = 3.0

        # Save to database
        today = date.today()
        snapshot = db.query(MarketSnapshot).filter(MarketSnapshot.date == today).first()

        if not snapshot:
            snapshot = MarketSnapshot(date=today)
            db.add(snapshot)

        snapshot.spy_price = current_price
        snapshot.spy_50_ma = ma_50
        snapshot.spy_200_ma = ma_200
        snapshot.market_score = score
        snapshot.market_trend = trend

        db.commit()
        logger.info(f"Market snapshot updated: SPY=${current_price:.2f}, trend={trend}, score={score}")

    except Exception as e:
        logger.error(f"Error updating market snapshot: {e}")


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

        # Convert score object to dict for easier handling
        score_result = {
            "total_score": score_obj.total_score,
            "C": {"score": score_obj.c_score, "detail": score_obj.c_detail},
            "A": {"score": score_obj.a_score, "detail": score_obj.a_detail},
            "N": {"score": score_obj.n_score, "detail": score_obj.n_detail},
            "S": {"score": score_obj.s_score, "detail": score_obj.s_detail},
            "L": {"score": score_obj.l_score, "detail": score_obj.l_detail},
            "I": {"score": score_obj.i_score, "detail": score_obj.i_detail},
            "M": {"score": score_obj.m_score, "detail": score_obj.m_detail},
        }

        # Get growth projection (pass StockData and CANSLIMScore objects)
        growth_obj = growth_projector.project_growth(stock_data, score_obj)

        return {
            "ticker": ticker,
            "name": getattr(stock_data, 'name', ticker),
            "sector": getattr(stock_data, 'sector', "Unknown"),
            "industry": getattr(stock_data, 'sector', "Unknown"),  # StockData doesn't have industry
            "current_price": getattr(stock_data, 'current_price', 0),
            "market_cap": getattr(stock_data, 'shares_outstanding', 0) * getattr(stock_data, 'current_price', 0),
            "week_52_high": getattr(stock_data, 'high_52w', 0),
            "week_52_low": 0,  # StockData doesn't have 52w low

            # CANSLIM scores
            "canslim_score": score_result.get("total_score", 0),
            "c_score": score_result.get("C", {}).get("score", 0),
            "a_score": score_result.get("A", {}).get("score", 0),
            "n_score": score_result.get("N", {}).get("score", 0),
            "s_score": score_result.get("S", {}).get("score", 0),
            "l_score": score_result.get("L", {}).get("score", 0),
            "i_score": score_result.get("I", {}).get("score", 0),
            "m_score": score_result.get("M", {}).get("score", 0),

            # Score details for display
            "score_details": {
                "C": score_result.get("C", {}),
                "A": score_result.get("A", {}),
                "N": score_result.get("N", {}),
                "S": score_result.get("S", {}),
                "L": score_result.get("L", {}),
                "I": score_result.get("I", {}),
                "M": score_result.get("M", {}),
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

            "analyzed_at": datetime.utcnow().isoformat()
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
    stock.last_updated = datetime.utcnow()

    db.flush()

    # Save score history (one per day)
    today = date.today()
    existing_score = db.query(StockScore).filter(
        StockScore.stock_id == stock.id,
        StockScore.date == today
    ).first()

    if not existing_score:
        score_history = StockScore(
            stock_id=stock.id,
            date=today,
            total_score=analysis["canslim_score"],
            c_score=analysis["c_score"],
            a_score=analysis["a_score"],
            n_score=analysis["n_score"],
            s_score=analysis["s_score"],
            l_score=analysis["l_score"],
            i_score=analysis["i_score"],
            m_score=analysis["m_score"],
            projected_growth=analysis["projected_growth"],
            current_price=analysis["current_price"]
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

    stock_count = db.query(Stock).count()
    portfolio_count = db.query(PortfolioPosition).count()

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
    # Update market snapshot if stale (older than 1 hour)
    latest_market = db.query(MarketSnapshot).order_by(
        desc(MarketSnapshot.date)
    ).first()

    if not latest_market or latest_market.date < date.today():
        update_market_snapshot(db)
        latest_market = db.query(MarketSnapshot).order_by(
            desc(MarketSnapshot.date)
        ).first()

    # Get top stocks by CANSLIM score
    top_stocks = db.query(Stock).filter(
        Stock.canslim_score != None
    ).order_by(desc(Stock.canslim_score)).limit(10).all()

    # Get portfolio summary
    positions = db.query(PortfolioPosition).all()
    total_value = sum(p.current_value or 0 for p in positions)
    total_gain = sum(p.gain_loss or 0 for p in positions)

    # Count by recommendation
    buy_count = len([p for p in positions if p.recommendation == "buy"])
    hold_count = len([p for p in positions if p.recommendation == "hold"])
    sell_count = len([p for p in positions if p.recommendation == "sell"])

    return {
        "top_stocks": [{
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "canslim_score": s.canslim_score,
            "projected_growth": s.projected_growth,
            "current_price": s.current_price,
            "growth_confidence": s.growth_confidence
        } for s in top_stocks],

        "market": {
            "trend": latest_market.market_trend if latest_market else "unknown",
            "score": latest_market.market_score if latest_market else 0,
            "spy_price": latest_market.spy_price if latest_market else 0,
            "spy_50_ma": latest_market.spy_50_ma if latest_market else 0,
            "spy_200_ma": latest_market.spy_200_ma if latest_market else 0,
            "date": latest_market.date.isoformat() if latest_market else None
        },

        "stats": {
            "total_stocks": db.query(Stock).filter(Stock.canslim_score != None).count(),
            "high_score_count": db.query(Stock).filter(Stock.canslim_score >= 80).count(),
            "portfolio_count": len(positions),
            "watchlist_count": db.query(Watchlist).count()
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

        "last_scan": db.query(func.max(Stock.last_updated)).scalar()
    }


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

    # Apply pagination
    stocks = query.offset(offset).limit(limit).all()

    return {
        "stocks": [{
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "industry": s.industry,
            "canslim_score": s.canslim_score,
            "projected_growth": s.projected_growth,
            "growth_confidence": s.growth_confidence,
            "current_price": s.current_price,
            "market_cap": s.market_cap,
            "c_score": s.c_score,
            "a_score": s.a_score,
            "n_score": s.n_score,
            "s_score": s.s_score,
            "l_score": s.l_score,
            "i_score": s.i_score,
            "m_score": s.m_score,
            "last_updated": s.last_updated.isoformat() if s.last_updated else None
        } for s in stocks],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/stocks/sectors")
async def get_sectors(db: Session = Depends(get_db)):
    """Get list of available sectors"""
    sectors = db.query(Stock.sector).distinct().filter(Stock.sector != None).all()
    return [s[0] for s in sectors if s[0]]


# ============== Single Stock Analysis ==============

@app.get("/api/stocks/{ticker}")
async def get_stock(ticker: str, db: Session = Depends(get_db)):
    """Get detailed stock analysis"""
    ticker = ticker.upper()

    # Check cache first
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()

    # If not cached or stale, analyze fresh
    cache_stale = not stock or (
        stock.last_updated and
        (datetime.utcnow() - stock.last_updated).total_seconds() > settings.SCORE_CACHE_HOURS * 3600
    )

    if cache_stale:
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

        "projected_growth": stock.projected_growth,
        "growth_confidence": stock.growth_confidence,

        "score_history": [{
            "date": h.date.isoformat(),
            "total_score": h.total_score,
            "price": h.current_price,
            "projected_growth": h.projected_growth
        } for h in reversed(history)],

        "last_updated": stock.last_updated.isoformat() if stock.last_updated else None
    }


@app.post("/api/stocks/{ticker}/refresh")
async def refresh_stock(ticker: str, db: Session = Depends(get_db)):
    """Force refresh a stock's analysis"""
    ticker = ticker.upper()

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
        started_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()

    # Run analysis in background
    def run_scan():
        import time
        scan_db = SessionLocal()
        try:
            processed = 0
            successful = 0
            for ticker in tickers:
                try:
                    analysis = analyze_stock(ticker)
                    if analysis:
                        # Apply price filter
                        if max_price and analysis["current_price"] > max_price:
                            processed += 1
                            continue
                        save_stock_to_db(scan_db, analysis)
                        successful += 1
                        logger.info(f"Scanned {ticker}: score={analysis.get('canslim_score', 0):.1f}")
                    processed += 1

                    # Update progress
                    scan_job = scan_db.query(AnalysisJob).filter(AnalysisJob.id == job.id).first()
                    scan_job.tickers_processed = processed
                    scan_db.commit()

                    # Delay to avoid rate limiting (2 seconds between requests)
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Error scanning {ticker}: {e}")
                    processed += 1

            # Mark complete
            scan_job = scan_db.query(AnalysisJob).filter(AnalysisJob.id == job.id).first()
            scan_job.status = "completed"
            scan_job.completed_at = datetime.utcnow()
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


# ============== Portfolio ==============

@app.get("/api/portfolio")
async def get_portfolio(db: Session = Depends(get_db)):
    """Get all portfolio positions"""
    positions = db.query(PortfolioPosition).all()

    total_value = sum(p.current_value or 0 for p in positions)
    total_cost = sum((p.cost_basis or 0) * (p.shares or 0) for p in positions)
    total_gain = total_value - total_cost

    return {
        "positions": [{
            "id": p.id,
            "ticker": p.ticker,
            "shares": p.shares,
            "cost_basis": p.cost_basis,
            "current_price": p.current_price,
            "current_value": p.current_value,
            "gain_loss": p.gain_loss,
            "gain_loss_pct": p.gain_loss_pct,
            "recommendation": p.recommendation,
            "canslim_score": p.canslim_score,
            "score_change": p.score_change,
            "notes": p.notes
        } for p in positions],
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
    ticker = data.ticker.upper()
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
    """Refresh all portfolio positions with current prices"""
    import time

    positions = db.query(PortfolioPosition).all()
    logger.info(f"Fetching prices for {len(positions)} positions")

    updated = 0
    errors = []

    for position in positions:
        try:
            current_price = fetch_price_yahoo_chart(position.ticker)

            if current_price:
                position.current_price = current_price
                position.current_value = current_price * position.shares

                if position.cost_basis:
                    position.gain_loss = (current_price - position.cost_basis) * position.shares
                    position.gain_loss_pct = (current_price - position.cost_basis) / position.cost_basis * 100

                    # Simple recommendation based on performance
                    if position.gain_loss_pct >= 20:
                        position.recommendation = "hold"
                    elif position.gain_loss_pct <= -15:
                        position.recommendation = "sell"
                    else:
                        position.recommendation = "hold"

                updated += 1
                logger.info(f"Updated {position.ticker}: ${current_price:.2f}")
            else:
                errors.append(f"{position.ticker}: no price found")

            # Small delay between requests to avoid rate limiting
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"Error processing {position.ticker}: {e}")
            errors.append(f"{position.ticker}: {str(e)}")

    db.commit()

    return {
        "message": f"Refreshed {updated} positions",
        "updated": updated,
        "errors": errors
    }


# ============== Watchlist ==============

@app.get("/api/watchlist")
async def get_watchlist(db: Session = Depends(get_db)):
    """Get watchlist with current stock data"""
    watchlist = db.query(Watchlist).all()

    items = []
    for w in watchlist:
        stock = db.query(Stock).filter(Stock.ticker == w.ticker).first()
        items.append({
            "id": w.id,
            "ticker": w.ticker,
            "added_at": w.added_at.isoformat() if w.added_at else None,
            "notes": w.notes,
            "target_price": w.target_price,
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

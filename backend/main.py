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
    Watchlist, AnalysisJob, MarketSnapshot,
    AIPortfolioConfig, AIPortfolioPosition, AIPortfolioTrade, AIPortfolioSnapshot
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
            "market_cap": getattr(stock_data, 'market_cap', 0) or (getattr(stock_data, 'shares_outstanding', 0) * getattr(stock_data, 'current_price', 0)),
            "week_52_high": getattr(stock_data, 'high_52w', 0),
            "week_52_low": getattr(stock_data, 'low_52w', 0),

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

    # Get top stocks under $25 by CANSLIM score
    top_stocks_under_25 = db.query(Stock).filter(
        Stock.canslim_score != None,
        Stock.current_price != None,
        Stock.current_price > 0,
        Stock.current_price <= 25
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

        "top_stocks_under_25": [{
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "canslim_score": s.canslim_score,
            "projected_growth": s.projected_growth,
            "current_price": s.current_price,
            "growth_confidence": s.growth_confidence
        } for s in top_stocks_under_25],

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
            "week_52_high": s.week_52_high,
            "week_52_low": s.week_52_low,
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
            time.sleep(random.uniform(1.5, 2.5))
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
            # Process stocks in parallel with 6 workers
            with ThreadPoolExecutor(max_workers=6) as executor:
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
                    old_score = position.canslim_score
                    position.canslim_score = stock.canslim_score
                    if old_score and stock.canslim_score:
                        position.score_change = stock.canslim_score - old_score

                    # Smarter recommendation based on score + performance
                    score = stock.canslim_score or 0
                    gain_pct = position.gain_loss_pct or 0

                    if score < 35 and gain_pct < -10:
                        position.recommendation = "sell"
                    elif score >= 70 and gain_pct > -5:
                        position.recommendation = "buy"
                    elif score < 50 and gain_pct < -15:
                        position.recommendation = "sell"
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

    # Define duplicate ticker groups (same company, different share classes)
    DUPLICATE_TICKERS = [
        {'GOOGL', 'GOOG'},  # Alphabet Class A vs Class C
        # Add more pairs here if needed (e.g., BRK.A/BRK.B)
    ]

    # Get top stocks for potential buys
    top_stocks = db.query(Stock).filter(
        Stock.canslim_score != None,
        Stock.canslim_score >= 65
    ).order_by(desc(Stock.canslim_score)).limit(20).all()

    # Get tickers we already own (including duplicates)
    owned_tickers = {p.ticker for p in positions}

    # Expand owned_tickers to include duplicates
    for ticker in list(owned_tickers):
        for group in DUPLICATE_TICKERS:
            if ticker in group:
                owned_tickers.update(group)

    actions = []

    # === SELL ACTIONS ===
    for p in positions:
        stock = db.query(Stock).filter(Stock.ticker == p.ticker).first()
        score = stock.canslim_score if stock else (p.canslim_score or 0)
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
                "reason": f"Weak fundamentals (score {score:.0f}) with loss of {p.gain_loss_pct:.1f}%",
                "details": [
                    f"CANSLIM Score: {score:.0f}/100 (below 35 threshold)",
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
                "reason": f"Deteriorating outlook (score {score:.0f}, {projected:.1f}% projected)",
                "details": [
                    f"CANSLIM Score: {score:.0f}/100",
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

        if stock.canslim_score >= 75 and (stock.projected_growth or 0) >= 15:
            shares_to_buy = int(max_position_value / stock.current_price) if stock.current_price else 0
            buy_value = shares_to_buy * stock.current_price if shares_to_buy else max_position_value

            # Check if near 52-week high (momentum)
            near_high = ""
            if stock.week_52_high and stock.current_price:
                pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100
                if pct_from_high <= 10:
                    near_high = f"Within {pct_from_high:.0f}% of 52-week high"

            actions.append({
                "action": "BUY",
                "priority": 2,
                "ticker": stock.ticker,
                "shares_action": shares_to_buy,
                "shares_current": 0,
                "current_price": stock.current_price,
                "estimated_value": buy_value,
                "reason": f"Strong candidate - Score {stock.canslim_score:.0f}, +{stock.projected_growth:.0f}% projected",
                "details": [
                    f"CANSLIM Score: {stock.canslim_score:.0f}/100",
                    f"Projected 6-month growth: +{stock.projected_growth:.0f}%",
                    f"Sector: {stock.sector}",
                    near_high if near_high else f"Current price: ${stock.current_price:.2f}",
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

        score = stock.canslim_score or 0
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
                    "reason": f"Strong stock on pullback - Score {score:.0f}, currently {gain_pct:+.1f}%",
                    "details": [
                        f"CANSLIM Score: {score:.0f}/100 (strong)",
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

        if stock.canslim_score >= 70 and stock.week_52_high and stock.current_price:
            pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100

            # Approaching breakout (within 5-15% of high)
            if 5 <= pct_from_high <= 15:
                watch_actions.append({
                    "action": "WATCH",
                    "priority": 4,
                    "ticker": stock.ticker,
                    "shares_action": 0,
                    "shares_current": 0,
                    "current_price": stock.current_price,
                    "estimated_value": 0,
                    "reason": f"Approaching breakout - {pct_from_high:.0f}% from 52-week high",
                    "details": [
                        f"CANSLIM Score: {stock.canslim_score:.0f}/100",
                        f"52-week high: ${stock.week_52_high:.2f}",
                        f"Current: ${stock.current_price:.2f} ({pct_from_high:.1f}% below high)",
                        "Watch for breakout above prior high on volume",
                        f"Projected growth: +{stock.projected_growth:.0f}%" if stock.projected_growth else ""
                    ]
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

@app.get("/api/ai-portfolio")
async def get_ai_portfolio(db: Session = Depends(get_db)):
    """Get AI Portfolio overview"""
    config = get_or_create_config(db)
    portfolio = get_portfolio_value(db)
    positions = db.query(AIPortfolioPosition).all()

    return {
        "config": {
            "starting_cash": config.starting_cash,
            "max_positions": config.max_positions,
            "max_position_pct": config.max_position_pct,
            "min_score_to_buy": config.min_score_to_buy,
            "sell_score_threshold": config.sell_score_threshold,
            "take_profit_pct": config.take_profit_pct,
            "stop_loss_pct": config.stop_loss_pct,
            "is_active": config.is_active
        },
        "summary": portfolio,
        "positions": [{
            "id": p.id,
            "ticker": p.ticker,
            "shares": p.shares,
            "cost_basis": p.cost_basis,
            "current_price": p.current_price,
            "current_value": p.current_value,
            "gain_loss": p.gain_loss,
            "gain_loss_pct": p.gain_loss_pct,
            "purchase_score": p.purchase_score,
            "current_score": p.current_score,
            "purchase_date": p.purchase_date.isoformat() if p.purchase_date else None
        } for p in positions]
    }


@app.get("/api/ai-portfolio/history")
async def get_ai_portfolio_history(
    days: int = Query(30, le=365),
    db: Session = Depends(get_db)
):
    """Get AI Portfolio performance history for charts"""
    from datetime import timedelta
    start_date = date.today() - timedelta(days=days)

    snapshots = db.query(AIPortfolioSnapshot).filter(
        AIPortfolioSnapshot.date >= start_date
    ).order_by(AIPortfolioSnapshot.date).all()

    return [{
        "date": s.date.isoformat(),
        "total_value": s.total_value,
        "cash": s.cash,
        "positions_value": s.positions_value,
        "positions_count": s.positions_count,
        "total_return": s.total_return,
        "total_return_pct": s.total_return_pct,
        "day_change": s.day_change,
        "day_change_pct": s.day_change_pct
    } for s in snapshots]


@app.post("/api/ai-portfolio/refresh")
async def refresh_ai_portfolio_endpoint(db: Session = Depends(get_db)):
    """Refresh position prices without executing trades"""
    result = refresh_ai_portfolio(db)
    return result


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
        "realized_gain": t.realized_gain,
        "executed_at": t.executed_at.isoformat() if t.executed_at else None
    } for t in trades]


@app.post("/api/ai-portfolio/initialize")
async def initialize_ai_portfolio_endpoint(
    starting_cash: float = Query(25000.0, ge=1000, le=1000000),
    db: Session = Depends(get_db)
):
    """Initialize or reset the AI Portfolio"""
    result = initialize_ai_portfolio(db, starting_cash)
    return result


@app.post("/api/ai-portfolio/run-cycle")
async def run_ai_trading_cycle_endpoint(db: Session = Depends(get_db)):
    """Manually trigger an AI trading cycle"""
    result = run_ai_trading_cycle(db)
    return result


@app.patch("/api/ai-portfolio/config")
async def update_ai_portfolio_config(
    is_active: bool = Query(None),
    min_score_to_buy: int = Query(None, ge=50, le=100),
    sell_score_threshold: int = Query(None, ge=20, le=80),
    take_profit_pct: float = Query(None, ge=10, le=100),
    stop_loss_pct: float = Query(None, ge=5, le=50),
    db: Session = Depends(get_db)
):
    """Update AI Portfolio configuration"""
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

    db.commit()

    return {
        "message": "Config updated",
        "config": {
            "is_active": config.is_active,
            "min_score_to_buy": config.min_score_to_buy,
            "sell_score_threshold": config.sell_score_threshold,
            "take_profit_pct": config.take_profit_pct,
            "stop_loss_pct": config.stop_loss_pct
        }
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

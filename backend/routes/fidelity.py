"""Fidelity portfolio sync routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import desc, text
from datetime import date, datetime, timedelta, timezone

from backend.database import (
    get_db, Stock, FidelitySnapshot, FidelityPosition, FidelityTrade,
    AIPortfolioConfig, AIPortfolioPosition, StockScore, User
)
from backend.auth import get_current_active_user

router = APIRouter(prefix="/api/fidelity", tags=["fidelity"])

@router.post("/upload-positions")
async def upload_fidelity_positions(file: UploadFile = File(...), current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """
    Upload a Fidelity Positions CSV export.
    Parses positions for account the configured Fidelity account and stores a snapshot.
    """
    from backend.fidelity_sync import parse_positions_csv

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    content = await file.read()
    try:
        csv_text = content.decode('utf-8')
    except UnicodeDecodeError:
        csv_text = content.decode('latin-1')

    result = parse_positions_csv(csv_text)

    if not result["positions"]:
        raise HTTPException(status_code=400, detail="No positions found for account the configured Fidelity account")

    # Create snapshot
    snapshot = FidelitySnapshot(
        user_id=current_user.id,
        snapshot_date=date.fromisoformat(result["snapshot_date"]),
        cash_balance=result["cash_balance"],
        total_value=result["total_value"],
        positions_count=len(result["positions"]),
    )
    db.add(snapshot)
    db.flush()  # Get snapshot.id

    # Create position records
    for pos in result["positions"]:
        db.add(FidelityPosition(
            snapshot_id=snapshot.id,
            symbol=pos["symbol"],
            description=pos["description"],
            quantity=pos["quantity"],
            last_price=pos["last_price"],
            current_value=pos["current_value"],
            total_gain_loss=pos["total_gain_loss"],
            total_gain_loss_pct=pos["total_gain_loss_pct"],
            cost_basis_total=pos["cost_basis_total"],
            average_cost_basis=pos["average_cost_basis"],
            percent_of_account=pos["percent_of_account"],
            position_type=pos["type"],
        ))

    db.commit()

    return {
        "status": "success",
        "snapshot_id": snapshot.id,
        "snapshot_date": result["snapshot_date"],
        "positions_count": len(result["positions"]),
        "cash_balance": result["cash_balance"],
        "total_value": result["total_value"],
        "parse_errors": result["parse_errors"],
    }


@router.post("/upload-activity")
async def upload_fidelity_activity(file: UploadFile = File(...), current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """
    Upload a Fidelity Activity/History CSV export.
    Parses trades for account the configured Fidelity account and stores them.
    Deduplicates against existing trades by (date, symbol, action).
    """

    from backend.fidelity_sync import parse_activity_csv

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    content = await file.read()
    try:
        csv_text = content.decode('utf-8')
    except UnicodeDecodeError:
        csv_text = content.decode('latin-1')

    result = parse_activity_csv(csv_text)

    # Deduplicate: skip trades that already exist
    new_count = 0
    skipped = 0
    for t in result["trades"]:
        trade_date = date.fromisoformat(t["run_date"])
        existing = db.query(FidelityTrade).filter(
            FidelityTrade.user_id == current_user.id,
            FidelityTrade.run_date == trade_date,
            FidelityTrade.symbol == t["symbol"],
            FidelityTrade.action == t["action"],
        ).first()

        if existing:
            skipped += 1
            continue

        db.add(FidelityTrade(
            user_id=current_user.id,
            run_date=trade_date,
            action=t["action"],
            symbol=t["symbol"],
            description=t["description"],
            price=t["price"],
            quantity=t["quantity"],
            amount=t["amount"],
            commission=t["commission"],
            fees=t["fees"],
            settlement_date=date.fromisoformat(t["settlement_date"]) if t.get("settlement_date") else None,
            raw_action=t["raw_action"],
        ))
        new_count += 1

    db.commit()

    return {
        "status": "success",
        "new_trades": new_count,
        "skipped_duplicates": skipped,
        "total_in_file": len(result["trades"]),
        "dividends_found": len(result["dividends"]),
        "parse_errors": result["parse_errors"],
    }


@router.get("/snapshots")
async def get_fidelity_snapshots(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List Fidelity position snapshots, newest first."""


    snapshots = db.query(FidelitySnapshot).filter(
        FidelitySnapshot.user_id == current_user.id
    ).order_by(
        desc(FidelitySnapshot.snapshot_date)
    ).limit(limit).all()

    return {
        "snapshots": [
            {
                "id": s.id,
                "snapshot_date": s.snapshot_date.isoformat() if s.snapshot_date else None,
                "uploaded_at": s.uploaded_at.isoformat() + "Z" if s.uploaded_at else None,
                "cash_balance": s.cash_balance,
                "total_value": s.total_value,
                "positions_count": s.positions_count,
            }
            for s in snapshots
        ]
    }


@router.get("/latest")
async def get_fidelity_latest(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Get the most recent Fidelity snapshot with full position details."""


    snapshot = db.query(FidelitySnapshot).filter(
        FidelitySnapshot.user_id == current_user.id
    ).order_by(
        desc(FidelitySnapshot.snapshot_date)
    ).first()

    if not snapshot:
        return {"snapshot": None, "positions": []}

    positions = db.query(FidelityPosition).filter(
        FidelityPosition.snapshot_id == snapshot.id
    ).order_by(desc(FidelityPosition.current_value)).all()

    # Batch fetch CANSLIM scores for all Fidelity symbols
    fid_symbols = [p.symbol for p in positions]
    stocks_by_ticker = {}
    if fid_symbols:
        fid_stocks = db.query(Stock).filter(Stock.ticker.in_(fid_symbols)).all()
        stocks_by_ticker = {s.ticker: s for s in fid_stocks}

    enriched_positions = []
    for p in positions:
        stock = stocks_by_ticker.get(p.symbol)
        pos_data = {
            "symbol": p.symbol,
            "description": p.description,
            "quantity": p.quantity,
            "last_price": p.last_price,
            "current_value": p.current_value,
            "total_gain_loss": p.total_gain_loss,
            "total_gain_loss_pct": p.total_gain_loss_pct,
            "cost_basis_total": p.cost_basis_total,
            "average_cost_basis": p.average_cost_basis,
            "percent_of_account": p.percent_of_account,
            "type": p.position_type,
            # Enriched CANSLIM data
            "canslim_score": stock.canslim_score if stock else None,
            "growth_mode_score": stock.growth_mode_score if stock else None,
            "is_growth_stock": stock.is_growth_stock if stock else False,
            "projected_growth": stock.projected_growth if stock else None,
            "sector": stock.sector if stock else None,
        }
        enriched_positions.append(pos_data)

    return {
        "snapshot": {
            "id": snapshot.id,
            "snapshot_date": snapshot.snapshot_date.isoformat() if snapshot.snapshot_date else None,
            "uploaded_at": snapshot.uploaded_at.isoformat() + "Z" if snapshot.uploaded_at else None,
            "cash_balance": snapshot.cash_balance,
            "total_value": snapshot.total_value,
            "positions_count": snapshot.positions_count,
        },
        "positions": enriched_positions,
    }


@router.get("/trades")
async def get_fidelity_trades(
    limit: int = Query(50, ge=1, le=500),
    symbol: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get parsed Fidelity trades, newest first."""


    query = db.query(FidelityTrade).filter(FidelityTrade.user_id == current_user.id)
    if symbol:
        query = query.filter(FidelityTrade.symbol == symbol.upper())
    trades = query.order_by(desc(FidelityTrade.run_date)).limit(limit).all()

    return {
        "trades": [
            {
                "id": t.id,
                "run_date": t.run_date.isoformat() if t.run_date else None,
                "action": t.action,
                "symbol": t.symbol,
                "description": t.description,
                "price": t.price,
                "quantity": t.quantity,
                "amount": t.amount,
                "commission": t.commission,
                "fees": t.fees,
                "settlement_date": t.settlement_date.isoformat() if t.settlement_date else None,
            }
            for t in trades
        ],
        "count": len(trades),
    }


@router.get("/reconciliation")
async def get_fidelity_reconciliation(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """
    Compare the latest Fidelity snapshot against AI portfolio positions.
    Shows matches, discrepancies, and positions unique to each.
    """

    from backend.fidelity_sync import reconcile_portfolios

    # Get latest Fidelity snapshot
    snapshot = db.query(FidelitySnapshot).filter(
        FidelitySnapshot.user_id == current_user.id
    ).order_by(
        desc(FidelitySnapshot.snapshot_date)
    ).first()

    if not snapshot:
        raise HTTPException(status_code=404, detail="No Fidelity snapshot uploaded yet")

    fid_positions = db.query(FidelityPosition).filter(
        FidelityPosition.snapshot_id == snapshot.id
    ).all()

    # Get AI portfolio positions
    ai_positions = db.query(AIPortfolioPosition).filter(
        AIPortfolioPosition.user_id == current_user.id
    ).all()

    # Build comparable dicts
    fid_list = [
        {
            "symbol": p.symbol,
            "quantity": p.quantity,
            "current_value": p.current_value,
            "total_gain_loss_pct": p.total_gain_loss_pct,
            "average_cost_basis": p.average_cost_basis,
        }
        for p in fid_positions
    ]

    ai_list = [
        {
            "ticker": p.ticker,
            "shares": p.shares,
            "current_value": p.current_value,
            "gain_loss_pct": p.gain_loss_pct,
            "current_score": p.current_score,
        }
        for p in ai_positions
    ]

    result = reconcile_portfolios(fid_list, ai_list)
    result["snapshot_date"] = snapshot.snapshot_date.isoformat() if snapshot.snapshot_date else None
    result["snapshot_id"] = snapshot.id

    return result


@router.get("/gameplan")
async def get_fidelity_gameplan(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """
    Generate actionable trading recommendations for Fidelity positions.
    Mirrors the AI trader's actual decision logic (evaluate_sells + evaluate_buys)
    using the nostate_optimized strategy profile thresholds.
    """
    from config_loader import config as yaml_config
    from backend.ai_trader import get_strategy_profile

    snapshot = db.query(FidelitySnapshot).filter(
        FidelitySnapshot.user_id == current_user.id
    ).order_by(
        desc(FidelitySnapshot.snapshot_date)
    ).first()

    if not snapshot:
        return {"gameplan": [], "summary": {"total_actions": 0}}

    fid_positions = db.query(FidelityPosition).filter(
        FidelityPosition.snapshot_id == snapshot.id
    ).all()

    if not fid_positions:
        return {"gameplan": [], "summary": {"total_actions": 0}}

    # Load strategy profile (same one the AI trader uses)
    profile = get_strategy_profile("nostate_optimized")
    stop_loss_pct = profile.get('stop_loss_pct', 7.0)
    bearish_stop_loss_pct = profile.get('bearish_stop_loss_pct', 6.0)
    min_score_to_buy = profile.get('min_score', 72)
    max_positions = profile.get('max_positions', 8)
    max_single_position_pct = profile.get('max_single_position_pct', 25)
    score_crash_drop = profile.get('score_crash_drop_required', 25)
    score_crash_ignore_profit = profile.get('score_crash_ignore_if_profitable', 20)
    trailing_stops = profile.get('trailing_stops', {})
    quality_filters = profile.get('quality_filters', {})
    min_c_score = quality_filters.get('min_c_score', 10)
    min_l_score = quality_filters.get('min_l_score', 8)

    # Partial profit config (from YAML, same as AI trader)
    pp_config = yaml_config.get('ai_trader.partial_profits', {})
    pp_25_gain = pp_config.get('threshold_25pct', {}).get('gain_pct', 25)
    pp_25_sell = pp_config.get('threshold_25pct', {}).get('sell_pct', 25)
    pp_25_min_score = pp_config.get('threshold_25pct', {}).get('min_score', 60)
    pp_40_gain = pp_config.get('threshold_40pct', {}).get('gain_pct', 40)
    pp_40_sell = pp_config.get('threshold_40pct', {}).get('sell_pct', 50)
    pp_40_min_score = pp_config.get('threshold_40pct', {}).get('min_score', 60)
    pp_50_gain = pp_config.get('threshold_50pct', {}).get('gain_pct', 50)
    pp_50_sell = pp_config.get('threshold_50pct', {}).get('sell_pct', 75)
    pp_50_min_score = pp_config.get('threshold_50pct', {}).get('min_score', 55)

    # Check market direction (SPY gate — same as AI trader binary gate)
    spy_below_50ma = False
    is_bearish = False
    spy_status = "unknown"
    try:
        from data_fetcher import get_cached_market_direction
        mkt = get_cached_market_direction()
        spy_info = mkt.get('spy', {}) if mkt else {}
        spy_px = spy_info.get('price', 0)
        spy_50 = spy_info.get('ma_50', 0)
        if spy_px and spy_50:
            spy_below_50ma = spy_px < spy_50
            is_bearish = spy_below_50ma
            spy_status = f"SPY ${spy_px:.0f} {'below' if spy_below_50ma else 'above'} 50MA ${spy_50:.0f}"
    except Exception:
        pass

    effective_stop_loss = bearish_stop_loss_pct if is_bearish else stop_loss_pct

    # Calculate portfolio totals
    total_value = snapshot.total_value or sum(p.current_value or 0 for p in fid_positions)
    if total_value == 0:
        total_value = 10000

    # Batch fetch CANSLIM data for all Fidelity symbols
    fid_symbols = [p.symbol for p in fid_positions]
    stocks_by_ticker = {}
    if fid_symbols:
        fid_stocks = db.query(Stock).filter(Stock.ticker.in_(fid_symbols)).all()
        stocks_by_ticker = {s.ticker: s for s in fid_stocks}

    owned_tickers = expand_tickers_with_duplicates(set(fid_symbols))
    actions = []
    sell_tickers = set()  # Track what we're recommending to sell

    # Helper: get effective score for a stock (same logic as AI trader)
    def _effective_score(stock):
        if not stock:
            return 0, "N/A"
        is_growth = stock.is_growth_stock or False
        if is_growth and stock.growth_mode_score:
            return stock.growth_mode_score, "Growth"
        return stock.canslim_score or 0, "CANSLIM"

    # ==================== SELL LOGIC (mirrors ai_trader evaluate_sells) ====================

    for p in fid_positions:
        stock = stocks_by_ticker.get(p.symbol)
        score, score_type = _effective_score(stock)
        gain_pct = p.total_gain_loss_pct or 0
        is_growth = (stock.is_growth_stock or False) if stock else False

        # --- 1. Hard stop loss (AI trader: gain_pct <= -stop_loss_pct) ---
        if gain_pct <= -effective_stop_loss:
            market_note = " (bearish market, tighter stop)" if is_bearish else ""
            actions.append({
                "action": "SELL",
                "priority": 1,
                "ticker": p.symbol,
                "shares_action": p.quantity,
                "shares_current": p.quantity,
                "current_price": p.last_price,
                "estimated_value": p.current_value,
                "is_growth_stock": is_growth,
                "reason": f"STOP LOSS: Down {gain_pct:.1f}% (limit: -{effective_stop_loss:.0f}%){market_note}",
                "details": [
                    f"{score_type} Score: {score:.0f}/100",
                    f"Loss: {gain_pct:.1f}% exceeds {effective_stop_loss:.0f}% stop loss",
                    f"Cost basis: ${p.average_cost_basis:.2f}" if p.average_cost_basis else None,
                    f"Strategy profile stop: -{stop_loss_pct}% normal / -{bearish_stop_loss_pct}% bearish",
                    "AI trader would execute this sell immediately"
                ]
            })
            sell_tickers.add(p.symbol)
            continue

        # --- 2. Score crash detection (AI trader: 25pt drop, score < 50) ---
        # We can't track consecutive scans per-position, but we can check current vs recent scores
        if stock and score < 50:
            # Check if score dropped significantly (approximate score crash)
            purchase_score = stock.previous_score or 0
            if purchase_score > 0 and (purchase_score - score) >= score_crash_drop:
                # AI trader exception: ignore if position is >= 20% profitable
                if gain_pct < score_crash_ignore_profit:
                    c_val = getattr(stock, 'c_score', 0) or 0
                    a_val = getattr(stock, 'a_score', 0) or 0
                    n_val = getattr(stock, 'n_score', 0) or 0
                    actions.append({
                        "action": "SELL",
                        "priority": 2,
                        "ticker": p.symbol,
                        "shares_action": p.quantity,
                        "shares_current": p.quantity,
                        "current_price": p.last_price,
                        "estimated_value": p.current_value,
                        "is_growth_stock": is_growth,
                        "reason": f"SCORE CRASH: {score_type} dropped {purchase_score - score:.0f}pts to {score:.0f}",
                        "details": [
                            f"Score fell from {purchase_score:.0f} to {score:.0f} (>{score_crash_drop}pt drop required)",
                            f"C={c_val:.0f} A={a_val:.0f} N={n_val:.0f} — fundamentals deteriorating",
                            f"Current P&L: {gain_pct:+.1f}% (below {score_crash_ignore_profit}% profitable exception)",
                            "AI trader would sell after confirming over multiple scans"
                        ]
                    })
                    sell_tickers.add(p.symbol)
                    continue

    # ==================== PARTIAL PROFITS / TRIM (mirrors ai_trader 3-tier system) ====================

    for p in fid_positions:
        if p.symbol in sell_tickers:
            continue  # Already recommending full sell

        stock = stocks_by_ticker.get(p.symbol)
        score, score_type = _effective_score(stock)
        gain_pct = p.total_gain_loss_pct or 0
        is_growth = (stock.is_growth_stock or False) if stock else False

        # 3-tier partial profit system (same thresholds as AI trader)
        # Note: We can't track partial_profit_taken for Fidelity positions,
        # so we recommend based on current gain as if no partials taken yet.
        if gain_pct >= pp_50_gain and score >= pp_50_min_score:
            trim_pct = pp_50_sell  # 75%
            trim_shares = int(p.quantity * trim_pct / 100)
            trim_value = trim_shares * (p.last_price or 0)
            if trim_shares > 0:
                actions.append({
                    "action": "TRIM",
                    "priority": 2,
                    "ticker": p.symbol,
                    "shares_action": trim_shares,
                    "shares_current": p.quantity,
                    "current_price": p.last_price,
                    "estimated_value": trim_value,
                    "is_growth_stock": is_growth,
                    "reason": f"PARTIAL PROFIT {trim_pct}%: Up {gain_pct:.0f}%, {score_type} {score:.0f} still strong",
                    "details": [
                        f"Gain: +{gain_pct:.1f}% (above {pp_50_gain}% tier-3 threshold)",
                        f"{score_type} Score: {score:.0f}/100 (above {pp_50_min_score} minimum)",
                        f"Sell {trim_shares} of {p.quantity} shares (~{trim_pct}%) to protect gains",
                        f"Lock in ~${trim_value:,.0f} profit, let remaining {p.quantity - trim_shares} shares run",
                        "AI trader takes 75% off the table at +50% gain"
                    ]
                })
        elif gain_pct >= pp_40_gain and score >= pp_40_min_score:
            trim_pct = pp_40_sell  # 50%
            trim_shares = int(p.quantity * trim_pct / 100)
            trim_value = trim_shares * (p.last_price or 0)
            if trim_shares > 0:
                actions.append({
                    "action": "TRIM",
                    "priority": 3,
                    "ticker": p.symbol,
                    "shares_action": trim_shares,
                    "shares_current": p.quantity,
                    "current_price": p.last_price,
                    "estimated_value": trim_value,
                    "is_growth_stock": is_growth,
                    "reason": f"PARTIAL PROFIT {trim_pct}%: Up {gain_pct:.0f}%, {score_type} {score:.0f} still strong",
                    "details": [
                        f"Gain: +{gain_pct:.1f}% (above {pp_40_gain}% tier-2 threshold)",
                        f"{score_type} Score: {score:.0f}/100 (above {pp_40_min_score} minimum)",
                        f"Sell {trim_shares} of {p.quantity} shares (~{trim_pct}%)",
                        "AI trader takes 50% off at +40% gain"
                    ]
                })
        elif gain_pct >= pp_25_gain and score >= pp_25_min_score:
            trim_pct = pp_25_sell  # 25%
            trim_shares = int(p.quantity * trim_pct / 100)
            trim_value = trim_shares * (p.last_price or 0)
            if trim_shares > 0:
                actions.append({
                    "action": "TRIM",
                    "priority": 3,
                    "ticker": p.symbol,
                    "shares_action": trim_shares,
                    "shares_current": p.quantity,
                    "current_price": p.last_price,
                    "estimated_value": trim_value,
                    "is_growth_stock": is_growth,
                    "reason": f"PARTIAL PROFIT {trim_pct}%: Up {gain_pct:.0f}%, {score_type} {score:.0f} still strong",
                    "details": [
                        f"Gain: +{gain_pct:.1f}% (above {pp_25_gain}% tier-1 threshold)",
                        f"{score_type} Score: {score:.0f}/100 (above {pp_25_min_score} minimum)",
                        f"Sell {trim_shares} of {p.quantity} shares (~{trim_pct}%)",
                        "AI trader takes first 25% profit at +25% gain"
                    ]
                })

    # ==================== BUY LOGIC (mirrors ai_trader evaluate_buys composite scoring) ====================

    # SPY gate: if SPY below 50MA, no buy recommendations (same as AI trader)
    buy_blocked_reason = None
    if spy_below_50ma:
        buy_blocked_reason = f"BUY BLOCKED: {spy_status} — AI trader skips all buys when SPY < 50MA"

    if not buy_blocked_reason:
        # Check if at max positions
        if len(fid_positions) >= max_positions:
            buy_blocked_reason = f"At max positions ({len(fid_positions)}/{max_positions})"

    if not buy_blocked_reason:
        # Get candidates (same query as AI trader — wider pool, then filter/rank)
        top_canslim_stocks = db.query(Stock).filter(
            Stock.canslim_score != None,
            Stock.canslim_score >= min_score_to_buy,
            Stock.current_price > 0,
            Stock.projected_growth != None,
        ).order_by(desc(Stock.canslim_score)).limit(50).all()

        top_growth_stocks = db.query(Stock).filter(
            Stock.growth_mode_score != None,
            Stock.growth_mode_score >= min_score_to_buy,
            Stock.is_growth_stock == True,
            Stock.current_price > 0,
        ).order_by(desc(Stock.growth_mode_score)).limit(20).all()

        # Combine and dedupe
        seen_tickers = set()
        candidate_stocks = []
        for stock in top_canslim_stocks:
            if stock.ticker not in seen_tickers:
                seen_tickers.add(stock.ticker)
                candidate_stocks.append(stock)
        for stock in top_growth_stocks:
            if stock.ticker not in seen_tickers:
                seen_tickers.add(stock.ticker)
                candidate_stocks.append(stock)

        # Volume gate config (same as AI trader)
        vol_gate_config = yaml_config.get('volume_gate', {})
        vol_gate_enabled = vol_gate_config.get('enabled', True)

        # Earnings avoidance config (same as AI trader)
        earnings_config = yaml_config.get('ai_trader.earnings', {})
        avoidance_days = earnings_config.get('avoidance_days', 5)

        # Scoring weights (from strategy profile, same as AI trader)
        scoring_weights = profile.get('scoring_weights', {})
        w_growth = scoring_weights.get('growth_projection', 0.25)
        w_score = scoring_weights.get('canslim_score', 0.25)
        w_momentum = scoring_weights.get('momentum', 0.20)
        w_breakout = scoring_weights.get('breakout', 0.20)
        w_base = scoring_weights.get('base_quality', 0.10)

        recommended_groups = set()
        scored_candidates = []

        for stock in candidate_stocks:
            if stock.ticker in owned_tickers:
                continue

            # Duplicate group check
            skip_dup = False
            ticker_group = None
            for group in DUPLICATE_TICKERS:
                if stock.ticker in group:
                    ticker_group = frozenset(group)
                    if ticker_group in recommended_groups:
                        skip_dup = True
                    break
            if skip_dup:
                continue

            effective_score, score_type = _effective_score(stock)

            # Quality filters (same as AI trader)
            is_growth = stock.is_growth_stock or False
            c_score = getattr(stock, 'c_score', 0) or 0
            l_score = getattr(stock, 'l_score', 0) or 0

            if not is_growth:
                if c_score < min_c_score:
                    continue
                if l_score < min_l_score:
                    continue

            if not effective_score or effective_score < min_score_to_buy:
                continue

            # Base pattern data (same as AI trader)
            is_breaking_out = getattr(stock, 'is_breaking_out', False)
            base_type = getattr(stock, 'base_type', 'none') or 'none'
            weeks_in_base = getattr(stock, 'weeks_in_base', 0) or 0
            pivot_price = getattr(stock, 'pivot_price', 0) or 0
            has_base = base_type not in ('none', '', None)
            volume_ratio = getattr(stock, 'volume_ratio', 1.0) or 1.0
            breakout_volume_ratio = getattr(stock, 'breakout_volume_ratio', 1.0) or 1.0

            # VOLUME GATE (same context-aware thresholds as AI trader)
            if vol_gate_enabled:
                if is_breaking_out:
                    vol_threshold = vol_gate_config.get('breakout_min_volume_ratio', 1.5)
                elif has_base and pivot_price > 0:
                    pct_check = ((pivot_price - stock.current_price) / pivot_price) * 100 if pivot_price > 0 else 0
                    if 0 <= pct_check <= 15:
                        vol_threshold = vol_gate_config.get('pre_breakout_min_volume_ratio', 0.8)
                    else:
                        vol_threshold = vol_gate_config.get('min_volume_ratio', 1.0)
                else:
                    vol_threshold = vol_gate_config.get('min_volume_ratio', 1.0)
                if volume_ratio < vol_threshold:
                    continue

            # EARNINGS PROXIMITY (same as AI trader — block non-CS stocks within avoidance window)
            days_to_earnings = getattr(stock, 'days_to_earnings', None)
            if days_to_earnings is not None and 0 < days_to_earnings <= avoidance_days:
                continue

            # COMPOSITE SCORING (exact same logic as AI trader)
            momentum_score = 0
            breakout_bonus = 0
            pre_breakout_bonus = 0
            extended_penalty = 0
            base_quality_bonus = 0

            # Base pattern quality bonus (up to 15 points, same as AI trader)
            if has_base:
                base_quality_map = {"cup_with_handle": 10, "cup": 8, "double_bottom": 7, "flat": 6}
                base_quality_bonus = base_quality_map.get(base_type, 4)
                if weeks_in_base >= 8:
                    base_quality_bonus += 5
                elif weeks_in_base >= 6:
                    base_quality_bonus += 3
                elif weeks_in_base >= 5:
                    base_quality_bonus += 1

            entry_category = "NONE"
            pct_from_high = 0
            pct_from_pivot = 0

            if stock.week_52_high and stock.week_52_high > 0 and stock.current_price:
                pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100

                if pivot_price > 0:
                    pct_from_pivot = ((pivot_price - stock.current_price) / pivot_price) * 100

                # PRE-BREAKOUT: 5-15% below pivot (BEST entry — same as AI trader +40 bonus)
                if has_base and pivot_price > 0 and 5 <= pct_from_pivot <= 15:
                    pre_breakout_bonus = 40
                    momentum_score = 35
                    if volume_ratio >= 1.3:
                        pre_breakout_bonus += 5
                    if weeks_in_base >= 10:
                        pre_breakout_bonus += 5
                    entry_category = "PRE_BREAKOUT"

                # AT PIVOT: 0-5% below pivot (ready to break out)
                elif has_base and pivot_price > 0 and 0 <= pct_from_pivot < 5:
                    pre_breakout_bonus = 35
                    momentum_score = 30
                    if volume_ratio >= 1.5:
                        momentum_score += 5
                    entry_category = "AT_PIVOT"

                # BREAKOUT: Already past pivot (less ideal — AI trader only gives +10)
                elif is_breaking_out:
                    breakout_bonus = 10
                    if breakout_volume_ratio >= 2.0:
                        breakout_bonus += 5
                    momentum_score = 15
                    entry_category = "BREAKOUT"

                # EXTENDED: >5% above pivot (the easy money is gone — AI trader penalizes)
                elif has_base and pivot_price > 0 and pct_from_pivot < -5:
                    if pct_from_pivot < -10:
                        extended_penalty = -20
                        momentum_score = 5
                    else:
                        extended_penalty = -10
                        momentum_score = 10
                    entry_category = "EXTENDED"

                # NO BASE: At 52-week high = chasing (same penalty as AI trader)
                elif not has_base:
                    if pct_from_high <= 2:
                        if effective_score < 85:
                            extended_penalty = -15
                            momentum_score = 5
                            entry_category = "CHASING"
                        else:
                            momentum_score = 12
                            entry_category = "NEW_HIGH"
                    elif pct_from_high <= 10:
                        momentum_score = 15
                        if volume_ratio >= 1.5:
                            momentum_score += 3
                        entry_category = "NEAR_HIGH"
                    elif pct_from_high <= 25:
                        momentum_score = 8
                        entry_category = "PULLBACK"
                    else:
                        momentum_score = -5
                        entry_category = "DEEP_PULLBACK"

            # MOMENTUM CONFIRMATION (same as AI trader — 15% penalty if fading)
            rs_12m = getattr(stock, 'rs_12m', 1.0) or 1.0
            rs_3m = getattr(stock, 'rs_3m', 1.0) or 1.0
            momentum_penalty = 0
            if rs_12m > 0 and rs_3m < rs_12m * 0.95:
                momentum_penalty = -0.15

            # Compute composite score (same weights and formula as AI trader)
            growth_projection = min(stock.projected_growth or (effective_score * 0.3), 50)
            composite_score = (
                (growth_projection * w_growth) +
                (effective_score * w_score) +
                (momentum_score * w_momentum) +
                ((breakout_bonus + pre_breakout_bonus) * w_breakout) +
                (base_quality_bonus * w_base) +
                extended_penalty
            )
            if momentum_penalty < 0:
                composite_score *= (1 + momentum_penalty)

            # Build entry signal description
            if entry_category == "PRE_BREAKOUT":
                entry_signal = f"PRE-BREAKOUT: {base_type} ({weeks_in_base}w), {pct_from_pivot:.0f}% below pivot"
                entry_tag = "IDEAL ENTRY"
            elif entry_category == "AT_PIVOT":
                entry_signal = f"AT PIVOT: {base_type} ({weeks_in_base}w), {pct_from_pivot:.0f}% from pivot"
                entry_tag = "GOOD ENTRY"
            elif entry_category == "BREAKOUT":
                entry_signal = f"Breaking out of {base_type} ({weeks_in_base}w), vol {breakout_volume_ratio:.1f}x"
                entry_tag = "ACTIVE BREAKOUT"
            elif entry_category == "EXTENDED":
                entry_signal = f"EXTENDED: {abs(pct_from_pivot):.0f}% above pivot — easy money is gone"
                entry_tag = "LATE ENTRY"
            elif entry_category == "CHASING":
                entry_signal = f"At 52-week high without base pattern — chasing"
                entry_tag = "CHASING"
            elif entry_category in ("NEW_HIGH", "NEAR_HIGH"):
                entry_signal = f"{pct_from_high:.1f}% from 52-week high"
                entry_tag = "NEAR HIGH"
            elif entry_category == "PULLBACK":
                entry_signal = f"{pct_from_high:.0f}% pullback from high"
                entry_tag = "PULLBACK"
            else:
                entry_signal = f"Current price: ${stock.current_price:.2f}" if stock.current_price else ""
                entry_tag = ""

            # Position sizing: 10% default, adjusted for pre-breakout (same as AI trader)
            position_pct = 10.0
            pre_breakout_mult = profile.get('pre_breakout_multiplier', 1.40)
            if entry_category == "PRE_BREAKOUT":
                position_pct *= pre_breakout_mult
            elif entry_category == "AT_PIVOT":
                position_pct *= (pre_breakout_mult * 0.93)
            position_value = total_value * position_pct / 100
            shares_to_buy = int(position_value / stock.current_price) if stock.current_price else 0
            buy_value = shares_to_buy * stock.current_price if shares_to_buy else position_value

            scored_candidates.append({
                "stock": stock,
                "composite_score": composite_score,
                "effective_score": effective_score,
                "score_type": score_type,
                "is_growth": is_growth,
                "c_score": c_score,
                "l_score": l_score,
                "entry_category": entry_category,
                "entry_signal": entry_signal,
                "entry_tag": entry_tag,
                "shares_to_buy": shares_to_buy,
                "buy_value": buy_value,
                "ticker_group": ticker_group,
                "momentum_penalty": momentum_penalty,
                "extended_penalty": extended_penalty,
                "volume_ratio": volume_ratio,
            })

        # Sort by composite score (same ranking as AI trader) and take top 3
        scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        buy_count = 0
        for cand in scored_candidates:
            if buy_count >= 3:
                break
            stock = cand["stock"]
            tg = cand["ticker_group"]
            if tg and frozenset(tg) in recommended_groups:
                continue

            warnings = []
            if cand["entry_category"] in ("EXTENDED", "CHASING"):
                warnings.append(f"AI trader penalizes this entry ({cand['extended_penalty']:+d} composite pts)")
            if cand["momentum_penalty"] < 0:
                warnings.append("Momentum fading (3mo RS < 12mo RS) — 15% composite penalty")

            details = [
                f"{cand['score_type']} Score: {cand['effective_score']:.0f}/100 (min: {min_score_to_buy})",
                f"C={cand['c_score']:.0f} L={cand['l_score']:.0f} (quality gates: C>={min_c_score}, L>={min_l_score})",
                f"Composite rank score: {cand['composite_score']:.1f}",
                cand["entry_signal"],
            ]
            if stock.projected_growth:
                details.append(f"Projected growth: +{stock.projected_growth:.0f}%")
            if cand["entry_tag"]:
                details.append(f"Entry quality: {cand['entry_tag']}")
            details.extend(warnings)
            if stock.sector:
                details.append(f"Sector: {stock.sector}")
            if cand["shares_to_buy"]:
                details.append(f"Suggested: {cand['shares_to_buy']} shares (~${cand['buy_value']:,.0f})")

            # Priority: pre-breakout/at-pivot = 1, breakout = 2, everything else = 3
            if cand["entry_category"] in ("PRE_BREAKOUT", "AT_PIVOT"):
                priority = 1
            elif cand["entry_category"] == "BREAKOUT":
                priority = 2
            else:
                priority = 3

            actions.append({
                "action": "BUY",
                "priority": priority,
                "ticker": stock.ticker,
                "shares_action": cand["shares_to_buy"],
                "shares_current": 0,
                "current_price": stock.current_price,
                "estimated_value": cand["buy_value"],
                "is_growth_stock": cand["is_growth"],
                "composite_score": round(cand["composite_score"], 1),
                "reason": f"{cand['entry_tag'] + ': ' if cand['entry_tag'] else ''}{cand['score_type']} {cand['effective_score']:.0f}, composite {cand['composite_score']:.0f}",
                "details": [d for d in details if d],
            })

            if tg:
                recommended_groups.add(frozenset(tg))
            buy_count += 1

    # ==================== ADD TO WINNERS ====================
    # (Supplementary recommendation — AI trader does this via pyramiding)

    for p in fid_positions:
        if p.symbol in sell_tickers:
            continue
        stock = stocks_by_ticker.get(p.symbol)
        if not stock:
            continue

        score, score_type = _effective_score(stock)
        gain_pct = p.total_gain_loss_pct or 0
        position_weight = (p.current_value or 0) / total_value * 100
        is_growth = (stock.is_growth_stock or False)
        projected = stock.projected_growth or 0

        # Strong stock on pullback with room to add
        if score >= 65 and projected >= 10 and -15 <= gain_pct <= 5 and position_weight < 12:
            target_value = total_value * 0.10
            current_value = p.current_value or 0
            add_value = min(target_value - current_value, total_value * 0.05)

            if add_value > 500 and p.last_price and p.last_price > 0:
                add_shares = int(add_value / p.last_price)
                actions.append({
                    "action": "ADD",
                    "priority": 3,
                    "ticker": p.symbol,
                    "shares_action": add_shares,
                    "shares_current": p.quantity,
                    "current_price": p.last_price,
                    "estimated_value": add_value,
                    "is_growth_stock": is_growth,
                    "reason": f"Strong stock on pullback - {score_type} {score:.0f}, currently {gain_pct:+.1f}%",
                    "details": [
                        f"{score_type} Score: {score:.0f}/100",
                        f"Projected growth: +{projected:.0f}%",
                        f"Current P&L: {gain_pct:+.1f}% (buying the dip)",
                        f"Position weight: {position_weight:.1f}% (room to add to {max_single_position_pct}%)",
                        f"Add {add_shares} shares (~${add_value:,.0f})"
                    ]
                })

    # ==================== WATCH LIST ====================

    if not buy_blocked_reason:
        watch_actions = []
        watched_groups = set()

        # Reuse top_stocks from BUY section if available
        if 'top_stocks' not in dir():
            top_stocks = []

        for stock in top_stocks:
            if stock.ticker in owned_tickers:
                continue
            if stock.ticker in [a["ticker"] for a in actions if a["action"] == "BUY"]:
                continue

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

            effective_score, score_type = _effective_score(stock)
            is_growth = stock.is_growth_stock or False

            if effective_score >= 70 and stock.week_52_high and stock.week_52_high > 0 and stock.current_price:
                pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100
                base_type = getattr(stock, 'base_type', 'none') or 'none'
                weeks_in_base = getattr(stock, 'weeks_in_base', 0) or 0
                has_base = base_type not in ('none', '', None)

                if 5 <= pct_from_high <= 15:
                    if has_base and weeks_in_base >= 5:
                        watch_priority = 3
                        watch_reason = f"Pre-breakout: {base_type} base ({weeks_in_base}w), {pct_from_high:.0f}% from pivot"
                    elif has_base:
                        watch_priority = 4
                        watch_reason = f"Base forming ({base_type}) - {pct_from_high:.0f}% from high"
                    else:
                        watch_priority = 5
                        watch_reason = f"Approaching 52-week high - {pct_from_high:.0f}% away"

                    details = [f"{score_type} Score: {effective_score:.0f}/100"]
                    if has_base:
                        details.append(f"Base pattern: {base_type} ({weeks_in_base} weeks)")
                    details.extend([
                        f"52-week high: ${stock.week_52_high:.2f}",
                        f"Current: ${stock.current_price:.2f} ({pct_from_high:.1f}% below high)",
                    ])
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
                        "details": [d for d in details if d]
                    })

                    if watch_ticker_group:
                        watched_groups.add(watch_ticker_group)
                    if len(watch_actions) >= 3:
                        break

        actions.extend(watch_actions)

    # Filter None from details, sort by priority
    for a in actions:
        a["details"] = [d for d in a.get("details", []) if d]

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
            "portfolio_value": total_value,
            "cash_balance": snapshot.cash_balance,
            "snapshot_date": snapshot.snapshot_date.isoformat() if snapshot.snapshot_date else None,
            "market_status": spy_status,
            "buy_blocked": buy_blocked_reason,
            "strategy": "nostate_optimized",
        }
    }


@router.post("/sync-to-portfolio")
async def sync_fidelity_to_portfolio(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """
    Sync the latest Fidelity snapshot positions into the manual Portfolio.
    Creates/updates PortfolioPosition records to match Fidelity.
    """


    snapshot = db.query(FidelitySnapshot).filter(
        FidelitySnapshot.user_id == current_user.id
    ).order_by(
        desc(FidelitySnapshot.snapshot_date)
    ).first()

    if not snapshot:
        raise HTTPException(status_code=404, detail="No Fidelity snapshot uploaded yet")

    fid_positions = db.query(FidelityPosition).filter(
        FidelityPosition.snapshot_id == snapshot.id
    ).all()

    added = 0
    updated = 0

    for fp in fid_positions:
        existing = db.query(PortfolioPosition).filter(
            PortfolioPosition.ticker == fp.symbol
        ).first()

        if existing:
            existing.shares = fp.quantity
            existing.cost_basis = fp.average_cost_basis
            existing.current_price = fp.last_price
            existing.current_value = fp.current_value
            existing.gain_loss = fp.total_gain_loss
            existing.gain_loss_pct = fp.total_gain_loss_pct
            updated += 1
        else:
            db.add(PortfolioPosition(
                ticker=fp.symbol,
                shares=fp.quantity,
                cost_basis=fp.average_cost_basis,
                current_price=fp.last_price,
                current_value=fp.current_value,
                gain_loss=fp.total_gain_loss,
                gain_loss_pct=fp.total_gain_loss_pct,
            ))
            added += 1

    # Remove portfolio positions not in Fidelity snapshot
    fid_symbols = {fp.symbol for fp in fid_positions}
    stale_positions = db.query(PortfolioPosition).filter(
        ~PortfolioPosition.ticker.in_(fid_symbols) if fid_symbols else True
    ).all()
    removed = 0
    for sp in stale_positions:
        db.delete(sp)
        removed += 1

    db.commit()

    return {
        "status": "success",
        "added": added,
        "updated": updated,
        "removed": removed,
        "snapshot_date": snapshot.snapshot_date.isoformat() if snapshot.snapshot_date else None,
    }


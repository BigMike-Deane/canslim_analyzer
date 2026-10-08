"""Lab tab API (backend/lab.py engine). Read-only for any signed-in user."""
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import LabDecision, LabEquityMark, LabOrder, LabStrategy, get_db

router = APIRouter(prefix="/api/lab", tags=["lab"])


def _strategy(db: Session, name: str) -> LabStrategy:
    s = db.query(LabStrategy).filter(LabStrategy.name == name).first()
    if s is None:
        raise HTTPException(status_code=404, detail=f"unknown lab strategy {name}")
    return s


def _marks(db, s):
    return db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id).order_by(LabEquityMark.date).all()


def _spy_rebased(marks):
    """SPY total return (dividend-adjusted close; price close as fallback) rebased to the
    first mark's equity: 'what the starting money would be worth in SPY'."""
    base = next(((m.spy_adj_close or m.spy_close) for m in marks if (m.spy_adj_close or m.spy_close)), None)
    if not marks or not base:
        return [None] * len(marks)
    start = marks[0].equity
    return [start * (m.spy_adj_close or m.spy_close) / base if (m.spy_adj_close or m.spy_close) else None
            for m in marks]


def _realized(db, s) -> list:
    """FIFO realized gain per filled sell (for win rate / trade list)."""
    lots, out = defaultdict(deque), []
    orders = (db.query(LabOrder).filter(LabOrder.strategy_id == s.id, LabOrder.status == "filled")
              .order_by(LabOrder.filled_at, LabOrder.id).all())
    for o in orders:
        q, px = o.filled_qty or 0.0, o.filled_avg_price or 0.0
        if o.side == "buy":
            lots[o.symbol].append([q, px])
            continue
        gain, left = 0.0, q
        while left > 1e-9 and lots[o.symbol]:
            lot = lots[o.symbol][0]
            take = min(left, lot[0])
            gain += take * (px - lot[1])
            lot[0] -= take
            left -= take
            if lot[0] <= 1e-9:
                lots[o.symbol].popleft()
        out.append((o.id, gain))
    return out


def _latest_decision(db, s):
    return (db.query(LabDecision).filter(LabDecision.strategy_id == s.id)
            .order_by(LabDecision.date.desc()).first())


def _summary(db, s) -> dict:
    from backend.lab import get_client
    marks = _marks(db, s)
    last = marks[-1] if marks else None
    spy = _spy_rebased(marks)
    dec = _latest_decision(db, s)
    ret = (last.equity / marks[0].equity - 1) * 100 if marks else None
    spy_ret = (spy[-1] / marks[0].equity - 1) * 100 if marks and spy[-1] else None
    return {
        "name": s.name, "label": s.label, "kind": s.kind, "description": s.description,
        "is_active": s.is_active, "starting_value": s.starting_value,
        "activated_at": s.activated_at.isoformat() if s.activated_at else None,
        "broker_connected": get_client(s.name) is not None,
        "equity": last.equity if last else None, "cash": last.cash if last else None,
        "positions": last.positions if last else [],
        "as_of": last.date.isoformat() if last else None,
        "days": len(marks),
        "total_return_pct": round(ret, 2) if ret is not None else None,
        "spy_return_pct": round(spy_ret, 2) if spy_ret is not None else None,
        "excess_return_pct": round(ret - spy_ret, 2) if ret is not None and spy_ret is not None else None,
        "latest_decision": {"date": dec.date.isoformat(), "inputs": dec.inputs, "target": dec.target,
                            "status": dec.status, "note": dec.note} if dec else None,
    }


@router.get("/strategies")
def list_strategies(db: Session = Depends(get_db), user=Depends(get_current_user)):
    from backend.lab import sync_strategies
    sync_strategies(db)
    return [_summary(db, s) for s in db.query(LabStrategy).order_by(LabStrategy.id).all()]


@router.get("/strategies/{name}")
def strategy_detail(name: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return _summary(db, _strategy(db, name))


@router.get("/strategies/{name}/history")
def strategy_history(name: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    s = _strategy(db, name)
    marks = _marks(db, s)
    spy = _spy_rebased(marks)
    return [{"date": m.date.isoformat(), "equity": round(m.equity, 2), "cash": m.cash,
             "spy_value": round(v, 2) if v else None} for m, v in zip(marks, spy)]


@router.get("/strategies/{name}/trades")
def strategy_trades(name: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    s = _strategy(db, name)
    gains = dict(_realized(db, s))
    rows = (db.query(LabOrder).filter(LabOrder.strategy_id == s.id)
            .order_by(LabOrder.submitted_at.desc(), LabOrder.id.desc()).limit(200).all())
    return [{"id": o.id, "date": o.date.isoformat(), "symbol": o.symbol, "side": o.side, "qty": o.qty,
             "status": o.status, "filled_qty": o.filled_qty, "filled_avg_price": o.filled_avg_price,
             "value": round((o.filled_qty or 0) * (o.filled_avg_price or 0), 2) if o.filled_avg_price else None,
             "realized_gain": round(gains[o.id], 2) if o.id in gains else None,
             "reason": o.reason, "error": o.error} for o in rows]


@router.get("/strategies/{name}/decisions")
def strategy_decisions(name: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    s = _strategy(db, name)
    rows = (db.query(LabDecision).filter(LabDecision.strategy_id == s.id)
            .order_by(LabDecision.date.desc()).limit(120).all())
    return [{"date": d.date.isoformat(), "inputs": d.inputs, "target": d.target, "status": d.status,
             "note": d.note} for d in rows]


@router.get("/strategies/{name}/checks")
def strategy_checks(name: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Pre-registered stop rules (docs/exposure-plan.md), as evaluated after the latest close."""
    from backend.lab_checks import worst
    s = _strategy(db, name)
    m = (db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.checks.isnot(None))
         .order_by(LabEquityMark.date.desc()).first())
    if m is None:
        return {"as_of": None, "level": "pending", "checks": []}
    return {"as_of": m.date.isoformat(), "level": worst(m.checks), "checks": m.checks}


@router.get("/strategies/{name}/edge")
def strategy_edge(name: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Same scorecard as /api/ai-portfolio/edge (backend.edge_metrics), on the Lab account."""
    from backend.edge_metrics import compute_edge_metrics
    s = _strategy(db, name)
    marks = _marks(db, s)
    return compute_edge_metrics([m.equity for m in marks], _spy_rebased(marks), [g for _, g in _realized(db, s)])

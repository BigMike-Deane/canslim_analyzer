"""CANSLIM 2.0 API (backend/canslim2.py). Read-only for any signed-in user."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.canslim2 import EVIDENCE, EXPLAIN, FEATURES, LETTERS, REBALANCE_SESSIONS, TILT_TOP
from backend.database import Canslim2Score, Stock, get_db

router = APIRouter(prefix="/api/canslim2", tags=["canslim2"])

LETTER_INFO = {
    "C": {"name": "Current earnings", "signals": ["beat_streak", "surprise_pct"],
          "summary": "Beats analysts' estimates, quarter after quarter, by a wide margin."},
    "A": {"name": "Annual profitability", "signals": ["roe"],
          "summary": "Earns a high return on shareholders' capital (growth rates alone showed nothing)."},
    "S": {"name": "Supply & demand", "signals": ["s3", "dtc"],
          "summary": "Shrinking share count (buybacks) and little short-seller pressure."},
    "I": {"name": "Institutional sponsorship", "signals": ["n_brokers"],
          "summary": "Followed by many brokers. (A rising institutional share count predicted WORSE returns.)"},
}
NOT_SCORED = {
    "N": "New highs, breakouts and pivot points showed no signal in any form tested.",
    "L": "Relative strength / momentum (1, 3, 6 and 12-1 months, industry strength) showed no signal.",
    "M": "Market direction lives in the Lab's A1/A5 exposure rules (passed 1928-2026).",
}


def _latest_date(db):
    return db.query(func.max(Canslim2Score.date)).scalar()


def _row(r, name=None, sector=None):
    return {"ticker": r.ticker, "name": name, "sector": sector, "score_pct": r.score_pct, "rank": r.rank,
            "letters": {"C": r.c_pct, "A": r.a_pct, "S": r.s_pct, "I": r.i_pct},
            "market_cap": r.market_cap, "in_tilt": r.in_tilt, "tilt_mult": r.tilt_mult, "inputs": r.inputs}


@router.get("/meta")
def meta(db: Session = Depends(get_db), user=Depends(get_current_user)):
    d = _latest_date(db)
    n = db.query(func.count(Canslim2Score.id)).filter(Canslim2Score.date == d).scalar() if d else 0
    return {
        "as_of": d.isoformat() if d else None, "universe": n,
        "universe_rule": "US-listed, price > $5, market cap >= $1B, 20-day dollar volume >= $5M",
        "letters": {L: {**LETTER_INFO[L]} for L in LETTERS},
        "not_scored": NOT_SCORED,
        "signals": [{"key": f, "letter": L, "direction": "higher is better" if sg > 0 else "lower is better",
                     "explain": EXPLAIN[f], "evidence_t": EVIDENCE[f]} for f, sg, L in FEATURES],
        "evidence": ("Point-in-time test 2016-2026: tilting the 500 largest stocks toward high scores beat SPY "
                     "total return by about +0.7%/yr out of sample (6 of 8 years), consistent on mid-caps it "
                     "never traded, but below a strict luck bar after 10 trials. Best-evidence ranking, not a "
                     "proven edge."),
        "model_portfolio": {"lab_strategy": "canslim2_tilt", "holds": TILT_TOP,
                            "rebalance_sessions": REBALANCE_SESSIONS},
    }


@router.get("/top")
def top(limit: int = Query(50, ge=1, le=500), tilt_only: bool = False, bottom: bool = False,
        db: Session = Depends(get_db), user=Depends(get_current_user)):
    d = _latest_date(db)
    if d is None:
        return {"as_of": None, "stocks": []}
    q = (db.query(Canslim2Score, Stock.name, Stock.sector).outerjoin(Stock, Stock.ticker == Canslim2Score.ticker)
         .filter(Canslim2Score.date == d))
    if tilt_only:
        q = q.filter(Canslim2Score.in_tilt.is_(True))
    q = q.order_by(Canslim2Score.score.asc() if bottom else Canslim2Score.score.desc()).limit(limit)
    return {"as_of": d.isoformat(), "stocks": [_row(r, n, s) for r, n, s in q.all()]}


@router.get("/stock/{ticker}")
def stock(ticker: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    d = _latest_date(db)
    t = ticker.upper()
    r = db.query(Canslim2Score).filter(Canslim2Score.date == d, Canslim2Score.ticker == t).first() if d else None
    if r is None:
        raise HTTPException(status_code=404, detail=f"{t} is not in the CANSLIM 2.0 universe "
                                                    "(price > $5, market cap >= $1B, 20-day $ volume >= $5M)")
    n = db.query(func.count(Canslim2Score.id)).filter(Canslim2Score.date == d).scalar()
    return {"as_of": d.isoformat(), "universe": n, **_row(r)}

"""Buy-candidate funnel ledger (2026-09-02).

`evaluate_buys` runs every candidate through a chain of gates (score floor,
soft zone, quality filters, volume gate, earnings window, correction-zone
CS-only, sector cap, ML veto, chop entry bar, share-class dedupe) and, until
now, recorded nothing about the names it rejected. Reconstructing "why
didn't arm 8 buy PDEX on Aug-25" meant reading trade tapes against config by
hand. This module is the audit trail: one row per candidate per cycle per
strategy, naming the FIRST gate that stopped it (or its rank if it reached
the decision list, or `bought` if the caller executed it).

Design constraints:
- Leaf module: imports only backend.database. ai_trader and shadow_trader
  import it; it must never import them.
- The collector is in-memory and passed INTO evaluate_buys. Shadow arms run
  evaluate_buys against a sandboxed ShadowSession whose writes are rolled
  back, so the evaluator cannot persist rows itself — the caller persists
  with its real session (live: the cycle db; shadow: the `persist` session).
- Bounded: rows per (cycle, strategy) are capped, late-stage rejections and
  ranked names win over early-stage ones, and rows older than RETENTION_DAYS
  are purged opportunistically at persist time.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.database import BuyFunnelRow

logger = logging.getLogger(__name__)

# Display/priority order of the funnel, earliest gate first. `to_rows` keeps
# the LATEST stages when the cap bites — a name that died at the ML veto is
# more informative than one that missed the score floor by 20 points.
STAGE_ORDER = [
    "dead_data",
    "cz_prefilter",
    "score_floor",
    "bear_exception_pool",
    "soft_zone_det",
    "no_score",
    "quality_c",
    "quality_l",
    "quality_growth",
    "volume_gate",
    "earnings_window",
    "cz_cs_only",
    "sector_cap",
    "min_position_value",
    "bad_price",
    "ml_veto",
    "chop_entry_bar",
    "duplicate_class",
    "ranked",
    "exec_skipped",
    "bought",
]
_STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}

# Cycle-level notes use this pseudo-ticker: the evaluator never ran (book
# full, cash reserve, circuit breaker, market gate) so no per-name row exists.
CYCLE_TICKER = "*"
# One note row per cycle carrying the UNCAPPED stage histogram as JSON, so
# the per-name cap (which keeps late stages) never hides how many names died
# at the early gates. First live cycle: 37 ml_veto + 3 ranked filled the cap
# and every earlier stage read as zero.
HISTOGRAM_STAGE = "histogram"

DEFAULT_CAP = 40
RETENTION_DAYS = 21
_PURGE_EVERY = timedelta(hours=6)
_last_purge_at: Optional[datetime] = None


def _clean(v):
    """None for NaN / non-numeric so a bad score never poisons a row."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


class FunnelCollector:
    """Per-cycle, per-strategy record of what happened to each candidate.

    First write wins per ticker EXCEPT `bought`, which always overrides
    `ranked` — the evaluator ranks, the caller executes, and the executed
    outcome is the one the owner wants to see."""

    def __init__(self):
        self._rows: dict = {}
        self.notes: list = []

    # ── writers ────────────────────────────────────────────────────────
    def reject(self, ticker, stage, detail=None, score=None):
        if not ticker or ticker in self._rows:
            return
        self._rows[ticker] = {
            "ticker": ticker, "stage": stage,
            "detail": (str(detail)[:200] if detail else None),
            "score": _clean(score), "composite": None, "rank": None,
        }

    def ranked(self, ticker, rank, composite=None, score=None):
        if not ticker or ticker in self._rows:
            return
        self._rows[ticker] = {
            "ticker": ticker, "stage": "ranked", "detail": None,
            "score": _clean(score), "composite": _clean(composite),
            "rank": int(rank),
        }

    def bought(self, ticker, detail=None):
        """Caller executed this name. Upgrades a `ranked` row in place."""
        if not ticker:
            return
        row = self._rows.get(ticker)
        if row is None:
            self._rows[ticker] = {
                "ticker": ticker, "stage": "bought", "detail": detail,
                "score": None, "composite": None, "rank": None,
            }
            return
        row["stage"] = "bought"
        if detail:
            row["detail"] = str(detail)[:200]

    def exec_skip(self, ticker, detail=None):
        """Ranked, reached the execution loop, but the caller could not fill
        it (no live price, too small, cash). Only downgrades a `ranked` row."""
        row = self._rows.get(ticker) if ticker else None
        if row is None or row["stage"] != "ranked":
            return
        row["stage"] = "exec_skipped"
        if detail:
            row["detail"] = str(detail)[:200]

    def note(self, stage, detail=None):
        """Cycle-level outcome with no per-name rows (evaluator skipped)."""
        self.notes.append({
            "ticker": CYCLE_TICKER, "stage": stage,
            "detail": (str(detail)[:200] if detail else None),
            "score": None, "composite": None, "rank": None,
        })

    # ── readers ────────────────────────────────────────────────────────
    def __len__(self):
        return len(self._rows) + len(self.notes)

    def stage_counts(self) -> dict:
        counts: dict = {}
        for r in self._rows.values():
            counts[r["stage"]] = counts.get(r["stage"], 0) + 1
        return counts

    def to_rows(self, cap: int = DEFAULT_CAP) -> list:
        """Rows to persist: all notes + an uncapped histogram note + the top
        `cap` candidate rows, keeping the latest stages first, then higher
        scores. Ranked/bought rows sort by rank so the decision list stays
        intact under the cap."""
        import json

        def _key(r):
            stage_rank = _STAGE_RANK.get(r["stage"], -1)
            if r["stage"] in ("ranked", "exec_skipped", "bought"):
                return (-stage_rank, r["rank"] or 10**6, 0.0)
            return (-stage_rank, 0, -(r["score"] or 0.0))
        rows = sorted(self._rows.values(), key=_key)
        out = list(self.notes)
        if rows:
            out.append({
                "ticker": CYCLE_TICKER, "stage": HISTOGRAM_STAGE,
                "detail": json.dumps(self.stage_counts(), sort_keys=True),
                "score": None, "composite": None, "rank": None,
            })
        return out + rows[:max(0, int(cap))]


def persist_funnel(db, collector: FunnelCollector, *, strategy_name: str,
                   user_id: Optional[int] = None,
                   shadow_strategy_id: Optional[int] = None,
                   cycle_at: Optional[datetime] = None,
                   cap: int = DEFAULT_CAP) -> int:
    """Write one cycle's funnel. Never raises — a ledger failure must not
    cost the cycle its trades. Returns rows written."""
    if collector is None or len(collector) == 0:
        return 0
    cycle_at = cycle_at or datetime.now(timezone.utc)
    try:
        rows = collector.to_rows(cap=cap)
        for r in rows:
            db.add(BuyFunnelRow(
                cycle_at=cycle_at, strategy_name=strategy_name,
                user_id=user_id, shadow_strategy_id=shadow_strategy_id,
                ticker=r["ticker"], stage=r["stage"], detail=r.get("detail"),
                score=r.get("score"), composite=r.get("composite"),
                rank=r.get("rank"),
            ))
        db.commit()
        _maybe_purge(db)
        return len(rows)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"buy funnel persist failed for {strategy_name}: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        return 0


def _maybe_purge(db, now: Optional[datetime] = None) -> int:
    """Drop rows past retention, at most once per _PURGE_EVERY per process."""
    global _last_purge_at
    now = now or datetime.now(timezone.utc)
    if _last_purge_at is not None and now - _last_purge_at < _PURGE_EVERY:
        return 0
    _last_purge_at = now
    cutoff = (now - timedelta(days=RETENTION_DAYS)).replace(tzinfo=None)
    try:
        n = db.query(BuyFunnelRow).filter(BuyFunnelRow.cycle_at < cutoff).delete(
            synchronize_session=False)
        db.commit()
        if n:
            logger.info(f"buy funnel: purged {n} row(s) older than {RETENTION_DAYS}d")
        return n
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"buy funnel purge failed: {e}")
        db.rollback()
        return 0


def purge_now(db) -> int:
    """Force a purge regardless of the rate limit (tests / ops)."""
    global _last_purge_at
    _last_purge_at = None
    return _maybe_purge(db)


# ── read side (admin API) ──────────────────────────────────────────────
def _row_out(r: BuyFunnelRow) -> dict:
    return {
        "id": r.id,
        "cycle_at": r.cycle_at.isoformat() if r.cycle_at else None,
        "strategy": r.strategy_name,
        "user_id": r.user_id,
        "shadow_strategy_id": r.shadow_strategy_id,
        "ticker": r.ticker,
        "stage": r.stage,
        "detail": r.detail,
        "score": r.score,
        "composite": r.composite,
        "rank": r.rank,
    }


def list_strategies(db, days: int = 7) -> list:
    """Every (strategy, user/shadow) the funnel has seen recently, newest
    cycle first. Live books are keyed by user_id, arms by shadow id."""
    from sqlalchemy import func
    since = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)
    q = (db.query(BuyFunnelRow.strategy_name, BuyFunnelRow.user_id,
                  BuyFunnelRow.shadow_strategy_id,
                  func.max(BuyFunnelRow.cycle_at).label("last_cycle_at"))
         .filter(BuyFunnelRow.cycle_at >= since)
         .group_by(BuyFunnelRow.strategy_name, BuyFunnelRow.user_id,
                   BuyFunnelRow.shadow_strategy_id)
         .order_by(func.max(BuyFunnelRow.cycle_at).desc()))
    return [{
        "strategy": name, "user_id": uid, "shadow_strategy_id": sid,
        "key": _strategy_key(name, uid, sid),
        "last_cycle_at": last.isoformat() if last else None,
    } for name, uid, sid, last in q.all()]


def _strategy_key(name, uid, sid) -> str:
    if sid is not None:
        return f"shadow:{sid}"
    if uid is not None:
        return f"user:{uid}"
    return f"name:{name}"


def _apply_key(q, key: Optional[str]):
    if not key:
        return q
    kind, _, val = key.partition(":")
    if kind == "shadow" and val.isdigit():
        return q.filter(BuyFunnelRow.shadow_strategy_id == int(val))
    if kind == "user" and val.isdigit():
        return q.filter(BuyFunnelRow.user_id == int(val))
    if kind == "name":
        return q.filter(BuyFunnelRow.strategy_name == val)
    return q.filter(BuyFunnelRow.strategy_name == key)


def latest_cycle(db, key: Optional[str] = None, limit: int = 400) -> dict:
    """The most recent cycle for one strategy key (default: the newest cycle
    overall): stage histogram + rows in funnel order."""
    from sqlalchemy import func
    q = _apply_key(db.query(func.max(BuyFunnelRow.cycle_at)), key)
    cycle_at = q.scalar()
    if cycle_at is None:
        return {"key": key, "cycle_at": None, "stage_counts": {}, "rows": []}
    rows_q = _apply_key(db.query(BuyFunnelRow), key).filter(
        BuyFunnelRow.cycle_at == cycle_at)
    rows = rows_q.all()
    rows.sort(key=lambda r: (
        -_STAGE_RANK.get(r.stage, -1), r.rank or 10**6, -(r.score or 0.0)))
    import json
    counts: dict = {}
    histogram = None
    for r in rows:
        if r.ticker == CYCLE_TICKER and r.stage == HISTOGRAM_STAGE and r.detail:
            try:
                histogram = json.loads(r.detail)
            except ValueError:
                histogram = None
        elif r.ticker != CYCLE_TICKER:
            counts[r.stage] = counts.get(r.stage, 0) + 1
    # The histogram note is the truth; row counts are capped.
    stage_counts = histogram if histogram is not None else counts
    return {
        "key": key,
        "strategy": rows[0].strategy_name if rows else None,
        "cycle_at": cycle_at.isoformat(),
        "stage_counts": stage_counts,
        "n_candidates": sum(stage_counts.values()),
        "rows_capped": histogram is not None and sum(histogram.values()) > len(counts) and sum(counts.values()) < sum(histogram.values()),
        "notes": [_row_out(r) for r in rows
                  if r.ticker == CYCLE_TICKER and r.stage != HISTOGRAM_STAGE],
        "rows": [_row_out(r) for r in rows if r.ticker != CYCLE_TICKER][:limit],
    }


def ticker_history(db, ticker: str, days: int = 7, limit: int = 400) -> list:
    """'Why not X?' — every funnel row for one ticker across all strategies,
    newest first."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)
    rows = (db.query(BuyFunnelRow)
            .filter(BuyFunnelRow.ticker == ticker.upper(),
                    BuyFunnelRow.cycle_at >= since)
            .order_by(BuyFunnelRow.cycle_at.desc(), BuyFunnelRow.strategy_name)
            .limit(limit).all())
    return [_row_out(r) for r in rows]

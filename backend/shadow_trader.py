"""Shadow paper-trading harness.

Step 2 of the design in docs/shadow-paper-trading-design.md. Runs registered
ShadowStrategy stacks alongside the live AI trading cycle, emitting virtual
BUY/SELL decisions to ShadowTrade. The /api/admin/strategy-ab-eval endpoint
will read from ShadowTrade in Step 3 to compare candidate scoring stacks
against the live baseline.

Architecture (Option C — synthetic-session wrapper):

  scheduler.run_continuous_scan
        │
        ├── Phase 3 — live ai_trader.run_ai_trading_cycle (writes AIPortfolioTrade)
        │
        └── run_shadow_strategies(db, analysis_results)
              │
              └── for each active ShadowStrategy:
                    ├── open SANDBOX SessionLocal()
                    ├── ShadowSession wraps the sandbox, intercepting:
                    │     - query(AIPortfolioPosition) → synthetic FIFO-derived
                    │     - query(AIPortfolioConfig)   → synthetic config
                    │     - query(AIPortfolioTrade)    → ShadowTrade view
                    │     - add(AIPortfolioTrade)      → pending ShadowTrade
                    │     - everything else            → pass through
                    ├── call live evaluate_sells(shadow_session) → SELL decisions
                    ├── call live evaluate_buys(shadow_session)  → BUY decisions
                    ├── translate decisions to ShadowTrade rows
                    ├── ROLLBACK sandbox (drops any incidental side effects)
                    └── PERSIST pending ShadowTrade rows via a clean session

Side-effect containment:
  - SPY-gate SystemState writes (ai_trader:2007-2030) hit the sandbox and roll back.
  - MLPrediction inserts (ai_trader:842) hit the sandbox and roll back.
  - In-place ORM mutations on AIPortfolioConfig/Position go to synthetic state.
  - Notification webhooks are not in evaluate_buys/sells (those live in execute_trade,
    which we do not call — shadow_trader writes ShadowTrade directly).

Limitations (v1):
  - peak_price: the FIFO rebuild alone collapses peak to max(cost_basis,
    current_price), under which trailing stops can NEVER fire (above water
    drop-from-peak ~0; underwater peak_gain 0 → below the lowest tier).
    FIXED 2026-07-25 via ShadowPositionPeak: peaks are ratcheted forward
    across ticks and initialized from daily highs on first sight — see
    _apply_persisted_peaks / _persist_peaks.
  - Pyramid (action="PYRAMID") is not yet emitted by shadow — pyramids are evaluated
    in run_ai_trading_cycle's outer loop, not in evaluate_buys. Step 3+ will mirror
    pyramid evaluation.
  - Drawdown circuit breaker, FTD penalty, portfolio-heat penalty are run_ai_trading_cycle
    concerns. Shadow runs evaluate_buys with both penalty flags = False; not
    perfectly faithful to live behavior in regimes where those penalties activate.
    Step 3+ will lift these into shadow_trader.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional

from sqlalchemy.orm import Session

from backend.database import (
    AIPortfolioConfig,
    AIPortfolioPosition,
    AIPortfolioSnapshot,
    AIPortfolioTrade,
    SessionLocal,
    ShadowPositionPeak,
    ShadowStrategy,
    ShadowTrade,
)

logger = logging.getLogger(__name__)

# Serializes shadow runs: the scan-tick run_shadow_strategies and the 15-min
# intraday stop pass both derive positions from the ShadowTrade log and write
# back to it. Concurrent runs could each derive the same open position and
# emit duplicate SELLs (over-selling the FIFO book). Scan path acquires
# blocking (it must never skip); the intraday pass acquires non-blocking and
# skips the tick if a scan run is mid-flight.
_shadow_run_lock = threading.Lock()


def _as_utc_naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize aware/naive datetimes for comparison. ShadowTrade.executed_at
    is written tz-aware but reads back naive from a `timestamp` column, so
    generation matching must not trust tzinfo equality."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _same_generation(a: Optional[datetime], b: Optional[datetime],
                     tolerance_seconds: float = 5.0) -> bool:
    """True when two opened_at stamps identify the same position generation.
    Tolerance absorbs micro drift between in-memory and round-tripped stamps."""
    a = _as_utc_naive(a)
    b = _as_utc_naive(b)
    if a is None or b is None:
        return False
    return abs((a - b).total_seconds()) <= tolerance_seconds


# Sentinel user_id passed through to evaluate_buys/sells. Synthetic state is
# already scoped to the active ShadowStrategy; user_id-based filters in the
# AI-portfolio query proxy are ignored. Keep negative so a future real user_id
# collision is impossible (real ids are autoincrement positive integers).
SHADOW_USER_ID = -100


# Models for which the wrapper substitutes synthetic in-memory state. Queries
# and writes against any other model fall through to the sandbox session.
_AI_MODELS = (AIPortfolioPosition, AIPortfolioConfig, AIPortfolioTrade, AIPortfolioSnapshot)

# Target pct embedded in partial-profit sell reasons by
# trading_engine.get_partial_profit_action ("PARTIAL PROFIT 25%: Up ...").
# After a partial executes, live ai_trader's accumulator equals that TARGET
# (take_pct = target - already_taken, and execute adds sell_pct=take_pct),
# so replaying the most recent match reconstructs partial_profit_taken
# exactly.
_PARTIAL_TARGET_RE = re.compile(r"PARTIAL PROFIT (\d+(?:\.\d+)?)%")

# Partial trailing stops ADD their configured pct to the live accumulator
# each time they fire (peak resets, so the tier can fire again on a fresh
# run-up) — additive, unlike the set-to-target profit tiers. Live emits TWO
# formats: the intraday stop pass embeds the pct directly
# ("PARTIAL TRAILING STOP (50%): Peak ..."), while evaluate_sells embeds the
# complement ("PARTIAL TRAILING STOP: Peak ..., keeping 50%" → pct = 100−keep).
# The keep-regex must anchor on "keeping" — a lazy .*(\d+)% would grab the
# drop-from-peak pct instead.
_PARTIAL_TRAILING_RE = re.compile(r"PARTIAL TRAILING STOP \((\d+(?:\.\d+)?)%\)")
_PARTIAL_TRAILING_KEEP_RE = re.compile(
    r"PARTIAL TRAILING STOP:.*keeping (\d+(?:\.\d+)?)%")

# Pre-earnings partials ("PRE-EARNINGS PARTIAL: 3d to earnings, up 12.1% ...")
# embed no pct, but ai_trader's rule is take_pct = 25 − already_taken, gated
# on already_taken < 25 — the accumulator is exactly 25 after any fire, so
# max-to-25 reconstructs it (and self-heals stacks that logged duplicates
# while this pattern was missing: live 2026-07-29, GEF/SLDE re-fired every
# 90-min scan, selling 25% of the remainder each cycle).
_PRE_EARNINGS_PARTIAL_RE = re.compile(r"PRE-EARNINGS PARTIAL")


# ── Synthetic ORM-shaped objects ──────────────────────────────────────────────
# Plain Python classes with the same attribute surface as the SQLAlchemy
# models that ai_trader reads/writes. Mutations in-place stay in-memory.
# Using a class (not SimpleNamespace) so isinstance(...) checks in the wrapper
# distinguish real ORM objects (which the caller may pass through db.add) from
# synthetic ones we created.


class _SyntheticPosition:
    """Mirror of AIPortfolioPosition columns ai_trader reads/writes.

    Defaults are chosen so missing fields don't surprise the live evaluator
    (e.g. partial_profit_taken=0 not None — the live code does arithmetic
    on it).
    """

    __slots__ = (
        "ticker", "shares", "cost_basis", "purchase_date", "purchase_score",
        "current_price", "current_value", "gain_loss", "gain_loss_pct",
        "current_score", "is_growth_stock", "purchase_growth_score",
        "current_growth_score", "peak_price", "peak_date", "pyramid_count",
        "partial_profit_taken", "user_id", "is_coiled_spring",
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))
        if self.pyramid_count is None:
            self.pyramid_count = 0
        if self.partial_profit_taken is None:
            self.partial_profit_taken = 0
        if self.user_id is None:
            self.user_id = SHADOW_USER_ID
        if self.is_coiled_spring is None:
            self.is_coiled_spring = False


class _SyntheticConfig:
    """Mirror of AIPortfolioConfig columns ai_trader reads/writes."""

    __slots__ = (
        "id", "user_id", "starting_cash", "current_cash", "max_positions",
        "max_position_pct", "min_score_to_buy", "sell_score_threshold",
        "take_profit_pct", "stop_loss_pct", "is_active", "strategy",
        "peak_portfolio_value", "spy_sweep_shares", "paper_mode",
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))


# ── Query proxy ───────────────────────────────────────────────────────────────


class _ShadowQuery:
    """Minimal Query-shaped proxy over an in-memory list.

    ai_trader's filter chains use filter(), filter_by(), order_by(), limit(),
    first(), all(), count(), delete(). We accept all of these but ignore the
    filter predicates because the synthetic state is already scoped to one
    ShadowStrategy. AIPortfolioTrade lookups (cooldown) iterate the result
    in Python anyway and apply their own predicates, so returning the full
    list is functionally equivalent.
    """

    def __init__(self, items: list, on_delete=None):
        self._items = list(items)
        self._on_delete = on_delete

    def filter(self, *args, **kwargs):
        return self

    def filter_by(self, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, n):
        new = _ShadowQuery(self._items[:n], on_delete=self._on_delete)
        return new

    def offset(self, n):
        new = _ShadowQuery(self._items[n:], on_delete=self._on_delete)
        return new

    def group_by(self, *args, **kwargs):
        return self

    def options(self, *args, **kwargs):
        return self

    def join(self, *args, **kwargs):
        return self

    def outerjoin(self, *args, **kwargs):
        return self

    def with_entities(self, *args, **kwargs):
        return self

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None

    def one_or_none(self):
        return self.first()

    def count(self):
        return len(self._items)

    def scalar(self):
        return self.first()

    def delete(self, synchronize_session=False):
        n = len(self._items)
        if self._on_delete:
            for item in list(self._items):
                self._on_delete(item)
        self._items.clear()
        return n

    def __iter__(self):
        return iter(self._items)


# ── ShadowSession ─────────────────────────────────────────────────────────────


class ShadowSession:
    """Wraps a sandbox DB session; intercepts AI-portfolio reads/writes.

    Lifecycle:
      1. Init: derive synthetic positions from ShadowTrade FIFO log + build
         synthetic config from ShadowStrategy.starting_value.
      2. Live evaluators (evaluate_buys/evaluate_sells) call query/add/delete
         on this object. AI-portfolio operations route to synthetic state;
         everything else passes through to the sandbox session.
      3. Caller drains pending shadow trades and persists them via a separate
         clean session. Sandbox is rolled back to discard incidental writes.
    """

    def __init__(self, sandbox_db: Session, shadow_strategy: ShadowStrategy,
                 analysis_results: List[dict]):
        self._db = sandbox_db
        self.shadow_strategy = shadow_strategy
        self._analysis_by_ticker = {
            (a.get("ticker") if isinstance(a, dict) else getattr(a, "ticker", None)): a
            for a in (analysis_results or [])
        }
        self._analysis_by_ticker.pop(None, None)
        self._synthetic_positions: List[_SyntheticPosition] = []
        self._synthetic_config: _SyntheticConfig = self._build_synthetic_config()
        self._shadow_trade_view: List[ShadowTrade] = []
        self._pending_shadow_trades: List[ShadowTrade] = []
        self._derive_synthetic_positions()
        self._apply_persisted_peaks()

    # ── Initialisation helpers ───────────────────────────────────────────────

    def _build_synthetic_config(self) -> _SyntheticConfig:
        ss = self.shadow_strategy
        return _SyntheticConfig(
            id=ss.id,
            user_id=SHADOW_USER_ID,
            starting_cash=ss.starting_value or 25000.0,
            current_cash=ss.starting_value or 25000.0,
            max_positions=8,
            max_position_pct=12.0,
            min_score_to_buy=72,
            sell_score_threshold=45,
            take_profit_pct=75.0,
            stop_loss_pct=7.0,
            is_active=True,
            strategy=ss.parent_strategy,
            peak_portfolio_value=ss.starting_value or 25000.0,
            spy_sweep_shares=0.0,
            paper_mode=False,
        )

    def _derive_synthetic_positions(self):
        """FIFO-pair BUYs with SELLs from the strategy's ShadowTrade log.

        The remaining unmatched BUYs (consolidated by ticker) become open
        synthetic positions. Cash is decremented per BUY and incremented per
        SELL; this keeps the synthetic config's current_cash convergent with
        the live AIPortfolioConfig as long as decisions match.

        Same FIFO logic shape as ml/feature_extractor.extract_live_trade_data:392.
        """
        trades = list(
            self._db.query(ShadowTrade)
            .filter(ShadowTrade.shadow_strategy_id == self.shadow_strategy.id)
            .order_by(ShadowTrade.executed_at)
            .all()
        )
        self._shadow_trade_view = trades  # capture for AIPortfolioTrade query intercept

        # Per-ticker FIFO queue of {shares_remaining, cost_basis, executed_at,
        # canslim_score, is_growth_stock, signal_factors}
        from collections import defaultdict, deque
        open_buys = defaultdict(deque)
        partial_taken = defaultdict(float)
        cash = self._synthetic_config.current_cash

        for t in trades:
            entry = {
                "shares": t.shares,
                "price": t.price,
                "executed_at": t.executed_at,
                "canslim_score": t.canslim_score,
                "signal_factors": t.signal_factors or {},
            }
            if t.action in ("BUY", "PYRAMID"):
                open_buys[t.ticker].append(entry)
                cash -= (t.shares or 0) * (t.price or 0)
            elif t.action == "SELL":
                shares_to_sell = t.shares or 0
                cash += shares_to_sell * (t.price or 0)
                queue = open_buys[t.ticker]
                while shares_to_sell > 1e-9 and queue:
                    head = queue[0]
                    if head["shares"] <= shares_to_sell + 1e-9:
                        shares_to_sell -= head["shares"]
                        queue.popleft()
                    else:
                        head["shares"] -= shares_to_sell
                        shares_to_sell = 0
                # Rebuild partial_profit_taken from the sell log. Without
                # this the rebuilt position always claimed 0% taken and the
                # 25% tier re-fired EVERY cycle, selling 25% of the remainder
                # forever (live 2026-07-13: PANW emitted 1,381 geometric
                # partial SELLs; 4,282 SELLs vs 90 BUYs per stack).
                m = _PARTIAL_TARGET_RE.search(t.reason or "")
                if m:
                    partial_taken[t.ticker] = max(
                        partial_taken[t.ticker], float(m.group(1)))
                m = _PARTIAL_TRAILING_RE.search(t.reason or "")
                if m:
                    partial_taken[t.ticker] += float(m.group(1))
                m = _PARTIAL_TRAILING_KEEP_RE.search(t.reason or "")
                if m:
                    partial_taken[t.ticker] += 100.0 - float(m.group(1))
                if _PRE_EARNINGS_PARTIAL_RE.search(t.reason or ""):
                    partial_taken[t.ticker] = max(partial_taken[t.ticker], 25.0)
                if not queue:
                    # Full close — the next BUY starts a fresh position.
                    partial_taken[t.ticker] = 0.0

        self._synthetic_config.current_cash = cash

        # Consolidate same-ticker open buys into one synthetic position
        for ticker, queue in open_buys.items():
            entries = [e for e in queue if e["shares"] > 1e-9]
            if not entries:
                continue
            total_shares = sum(e["shares"] for e in entries)
            if total_shares <= 0:
                continue
            weighted_cost = sum(e["shares"] * e["price"] for e in entries) / total_shares
            first_entry = min(entries, key=lambda e: e["executed_at"])

            analysis = self._analysis_by_ticker.get(ticker, {})
            if isinstance(analysis, dict):
                current_price = analysis.get("current_price") or analysis.get("price") or weighted_cost
                current_score = analysis.get("total_score") or analysis.get("canslim_score") or 0
                growth_score = analysis.get("growth_mode_score") or 0
                is_growth_stock = bool(analysis.get("is_growth_stock", False))
            else:
                current_price = getattr(analysis, "current_price", None) or weighted_cost
                current_score = getattr(analysis, "total_score", None) or 0
                growth_score = getattr(analysis, "growth_mode_score", None) or 0
                is_growth_stock = bool(getattr(analysis, "is_growth_stock", False))

            current_value = total_shares * current_price
            gain_loss = (current_price - weighted_cost) * total_shares
            gain_loss_pct = ((current_price / weighted_cost) - 1) * 100 if weighted_cost > 0 else 0

            position = _SyntheticPosition(
                ticker=ticker,
                shares=total_shares,
                cost_basis=weighted_cost,
                purchase_date=first_entry["executed_at"],
                purchase_score=first_entry["canslim_score"],
                current_price=current_price,
                current_value=current_value,
                gain_loss=gain_loss,
                gain_loss_pct=gain_loss_pct,
                current_score=current_score,
                is_growth_stock=is_growth_stock,
                purchase_growth_score=first_entry["signal_factors"].get("growth_mode_score"),
                current_growth_score=growth_score,
                # peak_price approximation — see module docstring's Limitations.
                peak_price=max(weighted_cost, current_price),
                peak_date=first_entry["executed_at"],
                pyramid_count=max(0, len(entries) - 1),
                partial_profit_taken=partial_taken.get(ticker, 0.0),
                user_id=SHADOW_USER_ID,
            )
            self._synthetic_positions.append(position)

    def _apply_persisted_peaks(self):
        """Overlay ratcheted peaks (ShadowPositionPeak) onto FIFO-derived
        positions. The trade log alone collapses peak to max(cost, current) —
        drop-from-peak ~0 above water, peak_gain 0 underwater — so trailing
        stops could never fire without this. First sight of a position (no
        row, or a stale row from a prior generation of the same ticker)
        initializes the peak from daily highs since entry, mirroring live
        update_position_prices' peak init.
        """
        try:
            rows = self._db.query(ShadowPositionPeak).filter(
                ShadowPositionPeak.shadow_strategy_id == self.shadow_strategy.id
            ).all()
        except Exception as e:
            logger.warning(
                f"shadow[{self.shadow_strategy.name}]: peak overlay query failed: {e}")
            rows = []
        by_ticker = {r.ticker: r for r in rows}
        for pos in self._synthetic_positions:
            row = by_ticker.get(pos.ticker)
            if row is not None and _same_generation(row.opened_at, pos.purchase_date):
                if (row.peak_price or 0) > (pos.peak_price or 0):
                    pos.peak_price = row.peak_price
                    pos.peak_date = row.peak_date or pos.peak_date
            else:
                self._init_peak_from_history(pos)

    def _init_peak_from_history(self, pos: "_SyntheticPosition"):
        """Backfill a first-seen position's peak from daily highs since entry.
        Runs once per position lifetime — the tick's peak persistence writes a
        row, so subsequent derivations take the overlay path. Best-effort: on
        any failure the max(cost, current) approximation stands and the ratchet
        still starts forward from this tick.
        """
        try:
            pd_date = pos.purchase_date.date() if hasattr(pos.purchase_date, "date") else None
            if pd_date is not None and pd_date >= date.today():
                return  # opened today — no prior days to backfill
        except Exception:
            pass
        try:
            from backend.historical_data import HistoricalDataProvider
        except Exception:
            return
        try:
            start = pos.purchase_date.date() if hasattr(pos.purchase_date, "date") \
                else date.today() - timedelta(days=30)
            end = date.today()
            provider = HistoricalDataProvider([pos.ticker])
            provider.preload_data(start, end)
            best = max(pos.peak_price or 0, pos.cost_basis or 0)
            d = start
            while d <= end:
                p = provider.get_price(pos.ticker, d)
                if p and (p.get("high") or 0) > best:
                    best = p["high"]
                d += timedelta(days=1)
            if best > (pos.peak_price or 0):
                logger.info(
                    f"shadow[{self.shadow_strategy.name}]: {pos.ticker} peak "
                    f"initialized to ${best:.2f} from daily highs since entry")
                pos.peak_price = best
        except Exception as e:
            logger.debug(f"shadow peak history init failed for {pos.ticker}: {e}")

    # ── Session API surface ──────────────────────────────────────────────────

    def query(self, *models):
        if len(models) == 1:
            model = models[0]
            if model is AIPortfolioPosition:
                return _ShadowQuery(
                    self._synthetic_positions,
                    on_delete=self._remove_synthetic_position,
                )
            if model is AIPortfolioConfig:
                return _ShadowQuery([self._synthetic_config])
            if model is AIPortfolioTrade:
                # ai_trader queries AIPortfolioTrade for cooldown lookback
                # (recent SELLs) and for last-action-per-ticker. Surface the
                # ShadowTrade rows shaped like AIPortfolioTrade — they share
                # the columns the consumer reads (action, executed_at, ticker,
                # reason). Synthetic objects so isinstance() doesn't fire.
                return _ShadowQuery([self._trade_as_shadow(t) for t in self._shadow_trade_view])
            if model is AIPortfolioSnapshot:
                return _ShadowQuery([])
        # Pass-through (multi-model queries, all other models)
        return self._db.query(*models)

    def add(self, obj):
        if isinstance(obj, AIPortfolioTrade):
            self._capture_trade_as_shadow(obj)
            return
        if isinstance(obj, AIPortfolioPosition):
            self._synthetic_positions.append(self._wrap_real_position(obj))
            return
        if isinstance(obj, (AIPortfolioConfig, AIPortfolioSnapshot)):
            return  # synthetic state already exists / snapshots not modeled
        # Anything else: write to sandbox so reads-after-write within the same
        # cycle see consistent state. Sandbox rollback discards them at end.
        self._db.add(obj)

    def add_all(self, objs):
        for o in objs:
            self.add(o)

    def delete(self, obj):
        if isinstance(obj, _SyntheticPosition) or isinstance(obj, AIPortfolioPosition):
            self._remove_synthetic_position(obj)
            return
        if isinstance(obj, _AI_MODELS):
            return  # synthetic-state; nothing to delete on the real db
        self._db.delete(obj)

    def commit(self):
        # In-shadow commit is a no-op for AI-portfolio mutations (those live
        # in synthetic state). Pass-through writes to the sandbox flush; the
        # final orchestrator-level rollback discards them. We do NOT flush
        # pending shadow trades here — those are persisted by the caller via
        # drain_pending_shadow_trades on a separate session.
        try:
            self._db.flush()
        except Exception:
            self._db.rollback()

    def rollback(self):
        self._db.rollback()
        self._pending_shadow_trades.clear()

    def flush(self, objects=None):
        try:
            self._db.flush(objects=objects) if objects is not None else self._db.flush()
        except Exception:
            self._db.rollback()

    def refresh(self, obj, attribute_names=None):
        if isinstance(obj, (_SyntheticPosition, _SyntheticConfig)):
            return  # synthetic state needs no refresh
        if isinstance(obj, _AI_MODELS):
            return
        self._db.refresh(obj, attribute_names=attribute_names) \
            if attribute_names is not None else self._db.refresh(obj)

    def merge(self, obj):
        if isinstance(obj, _AI_MODELS):
            return obj  # treat as already merged
        return self._db.merge(obj)

    def begin_nested(self):
        return self._db.begin_nested()

    def expire(self, obj, attribute_names=None):
        return  # no-op — synthetic state is fresh

    def expire_all(self):
        return

    def close(self):
        # Caller (run_shadow_strategies) closes the sandbox — don't double-close.
        pass

    def __getattr__(self, name):
        # Last-resort pass-through for anything we forgot. Logged at DEBUG so
        # we can spot live-trader code that hits a new session API.
        attr = getattr(self._db, name)
        logger.debug(f"ShadowSession passthrough: {name}")
        return attr

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _remove_synthetic_position(self, obj):
        # obj may be _SyntheticPosition (our own) or an AIPortfolioPosition
        # (caller-built and never added via .add — unlikely but defensive).
        try:
            self._synthetic_positions.remove(obj)
            return
        except ValueError:
            pass
        ticker = getattr(obj, "ticker", None)
        if ticker:
            self._synthetic_positions[:] = [
                p for p in self._synthetic_positions if p.ticker != ticker
            ]

    def _wrap_real_position(self, p: AIPortfolioPosition) -> _SyntheticPosition:
        return _SyntheticPosition(
            ticker=p.ticker,
            shares=p.shares,
            cost_basis=p.cost_basis,
            purchase_date=p.purchase_date or datetime.now(timezone.utc),
            purchase_score=p.purchase_score,
            current_price=p.current_price,
            current_value=p.current_value,
            gain_loss=p.gain_loss,
            gain_loss_pct=p.gain_loss_pct,
            current_score=p.current_score,
            is_growth_stock=p.is_growth_stock or False,
            purchase_growth_score=p.purchase_growth_score,
            current_growth_score=p.current_growth_score,
            peak_price=p.peak_price or p.cost_basis,
            peak_date=p.peak_date or p.purchase_date,
            pyramid_count=p.pyramid_count or 0,
            partial_profit_taken=p.partial_profit_taken or 0,
            user_id=SHADOW_USER_ID,
        )

    def _capture_trade_as_shadow(self, trade: AIPortfolioTrade):
        """Translate an AIPortfolioTrade-shaped object into a ShadowTrade row.

        Called when the live evaluator's downstream code does
        ``db.add(AIPortfolioTrade(...))``. evaluate_buys / evaluate_sells
        do not call db.add directly — they return decisions that the
        orchestrator translates. This path is here for safety and future-
        proofing if a live-trader code path changes.
        """
        shadow_trade = ShadowTrade(
            shadow_strategy_id=self.shadow_strategy.id,
            ticker=trade.ticker,
            action=trade.action,
            shares=trade.shares,
            price=trade.price,
            total_value=trade.total_value or (trade.shares or 0) * (trade.price or 0),
            reason=trade.reason,
            canslim_score=trade.canslim_score,
            cost_basis=trade.cost_basis,
            realized_gain=trade.realized_gain,
            holding_days=trade.holding_days,
            signal_factors=trade.signal_factors,
            executed_at=trade.executed_at or datetime.now(timezone.utc),
        )
        self._pending_shadow_trades.append(shadow_trade)

    def _trade_as_shadow(self, t: ShadowTrade):
        """Wrap a ShadowTrade in a SimpleNamespace shaped like AIPortfolioTrade
        so the consumer's attribute reads work transparently. Using
        SimpleNamespace (not the SQLAlchemy class) keeps it detached from any
        identity map.
        """
        return SimpleNamespace(
            ticker=t.ticker,
            action=t.action,
            shares=t.shares,
            price=t.price,
            total_value=t.total_value,
            reason=t.reason,
            canslim_score=t.canslim_score,
            cost_basis=t.cost_basis,
            realized_gain=t.realized_gain,
            holding_days=t.holding_days,
            signal_factors=t.signal_factors,
            executed_at=t.executed_at,
            user_id=SHADOW_USER_ID,
            growth_mode_score=None,
            is_growth_stock=False,
            is_paper=False,
        )

    # ── Public exec helpers (called by orchestrator) ─────────────────────────

    def emit_shadow_buy(self, *, ticker: str, shares: float, price: float,
                        reason: str, canslim_score: Optional[float] = None,
                        signal_factors: Optional[dict] = None,
                        is_growth_stock: bool = False) -> ShadowTrade:
        """Construct + queue a BUY ShadowTrade and update synthetic positions.

        Mirrors the relevant fields of execute_trade + the position-create
        block from run_ai_trading_cycle, but writes ShadowTrade and updates
        in-memory state only. No webhooks, no live price fetches.
        """
        total_value = shares * price
        # One shared stamp: purchase_date is the position-generation key for
        # peak persistence and must equal the BUY's executed_at exactly, or
        # the next FIFO rebuild (which reads executed_at) orphans the peak row.
        now = datetime.now(timezone.utc)
        st = ShadowTrade(
            shadow_strategy_id=self.shadow_strategy.id,
            ticker=ticker,
            action="BUY",
            shares=shares,
            price=price,
            total_value=total_value,
            reason=reason,
            canslim_score=canslim_score,
            cost_basis=None,
            realized_gain=None,
            holding_days=None,
            signal_factors=signal_factors or {},
            executed_at=now,
        )
        self._pending_shadow_trades.append(st)
        # Update synthetic state
        self._synthetic_config.current_cash -= total_value
        new_pos = _SyntheticPosition(
            ticker=ticker,
            shares=shares,
            cost_basis=price,
            purchase_date=now,
            purchase_score=canslim_score,
            current_price=price,
            current_value=total_value,
            gain_loss=0.0,
            gain_loss_pct=0.0,
            current_score=canslim_score,
            is_growth_stock=is_growth_stock,
            peak_price=price,
            peak_date=datetime.now(timezone.utc),
            pyramid_count=0,
            partial_profit_taken=0,
            user_id=SHADOW_USER_ID,
        )
        self._synthetic_positions.append(new_pos)
        return st

    def emit_shadow_sell(self, *, position: _SyntheticPosition, shares: float,
                         price: float, reason: str,
                         is_partial: bool = False) -> ShadowTrade:
        """Queue a SELL ShadowTrade and reduce synthetic position shares."""
        cost_basis = position.cost_basis or 0.0
        realized_gain = (price - cost_basis) * shares if cost_basis else 0.0
        holding_days = None
        if position.purchase_date:
            try:
                pd_aware = position.purchase_date
                if pd_aware.tzinfo is None:
                    pd_aware = pd_aware.replace(tzinfo=timezone.utc)
                holding_days = (datetime.now(timezone.utc) - pd_aware).days
            except Exception:
                holding_days = None

        st = ShadowTrade(
            shadow_strategy_id=self.shadow_strategy.id,
            ticker=position.ticker,
            action="SELL",
            shares=shares,
            price=price,
            total_value=shares * price,
            reason=reason,
            canslim_score=position.current_score,
            cost_basis=cost_basis,
            realized_gain=realized_gain,
            holding_days=holding_days,
            signal_factors={"sell_reason": reason},
            executed_at=datetime.now(timezone.utc),
        )
        self._pending_shadow_trades.append(st)
        self._synthetic_config.current_cash += shares * price
        if is_partial and shares < (position.shares or 0):
            position.shares -= shares
            if (position.partial_profit_taken or 0) < 100:
                position.partial_profit_taken = (position.partial_profit_taken or 0) + 100 * (
                    shares / (position.shares + shares) if (position.shares + shares) > 0 else 1
                )
        else:
            self._remove_synthetic_position(position)
        return st

    def drain_pending_shadow_trades(self) -> List[ShadowTrade]:
        out = list(self._pending_shadow_trades)
        self._pending_shadow_trades.clear()
        return out


# ── Scorer overrides (Step 7) ─────────────────────────────────────────────────
# When a ShadowStrategy carries `scorer_overrides`, we mutate both the
# analysis_results (which feed _derive_synthetic_positions → current_score on
# synthetic positions → evaluate_sells thresholds) AND the sandbox Stock ORM
# rows (which evaluate_buys reads for the buy-side score). The sandbox rolls
# back at the end of _run_one_strategy, so live data is untouched.
#
# Currently supported override:
#   disable_excellence_cap (bool) — substitute c_score_uncapped for c_score
#     and adjust canslim_score by the cap delta. Built for the Approach 2
#     parallel-regime A/B (June 18 verdict). Rows that pre-date the
#     c_score_uncapped column write (NULL value) pass through unchanged.


def _apply_disable_excellence_cap(analysis_results: List[dict]) -> List[dict]:
    """Return analysis_results with c_score → c_score_uncapped and
    canslim_score / total_score adjusted by the cap delta."""
    out: List[dict] = []
    for a in (analysis_results or []):
        if not isinstance(a, dict):
            out.append(a)
            continue
        uncapped = a.get("c_score_uncapped")
        capped = a.get("c_score")
        if uncapped is None or capped is None or uncapped == capped:
            out.append(a)
            continue
        delta = uncapped - capped
        new_a = dict(a)
        new_a["c_score"] = uncapped
        if new_a.get("canslim_score") is not None:
            new_a["canslim_score"] = new_a["canslim_score"] + delta
        if new_a.get("total_score") is not None:
            new_a["total_score"] = new_a["total_score"] + delta
        out.append(new_a)
    return out


def _patch_sandbox_stocks_disable_cap(sandbox: Session, analysis_results: List[dict]) -> int:
    """Mutate Stock ORM rows in the sandbox session so evaluate_buys reads
    pre-cap scores. The caller's `finally` rolls the sandbox back."""
    from backend.database import Stock

    deltas: dict = {}
    for a in (analysis_results or []):
        ticker = a.get("ticker") if isinstance(a, dict) else getattr(a, "ticker", None)
        uncapped = a.get("c_score_uncapped") if isinstance(a, dict) else getattr(a, "c_score_uncapped", None)
        capped = a.get("c_score") if isinstance(a, dict) else getattr(a, "c_score", None)
        if ticker and uncapped is not None and capped is not None and uncapped != capped:
            deltas[ticker] = (uncapped, uncapped - capped)
    if not deltas:
        return 0
    rows = sandbox.query(Stock).filter(Stock.ticker.in_(list(deltas.keys()))).all()
    n = 0
    for stock in rows:
        entry = deltas.get(stock.ticker)
        if not entry:
            continue
        new_c, delta = entry
        stock.c_score = new_c
        if stock.canslim_score is not None:
            stock.canslim_score = stock.canslim_score + delta
        n += 1
    return n


# ── Orchestration ─────────────────────────────────────────────────────────────


def _emit_sell_decisions(shadow_session: "ShadowSession", sells: list) -> int:
    """Translate evaluate_sells decision dicts into ShadowTrade SELLs.

    Shared by the scan-tick run and the intraday stop pass. Honors the
    decision's reset_peak flag (partial trailing stop) by snapping the
    in-memory peak to the current price, exactly like live execution — the
    tick's peak persistence then stores the reset so the tier can re-arm on
    a fresh run-up instead of re-firing forever off the old peak.
    """
    n = 0
    for sell in sells:
        position = sell.get("position") if isinstance(sell, dict) else None
        if position is None or position.current_price is None:
            continue
        is_partial = bool(sell.get("is_partial", False)) if isinstance(sell, dict) else False
        sell_pct = float(sell.get("sell_pct", 100)) if isinstance(sell, dict) else 100.0
        shares_to_sell = (position.shares or 0)
        if is_partial:
            shares_to_sell = max(0, (position.shares or 0) * (sell_pct / 100.0))
            if shares_to_sell <= 0:
                continue
        shadow_session.emit_shadow_sell(
            position=position,
            shares=shares_to_sell,
            price=position.current_price,
            reason=sell.get("reason", "shadow sell") if isinstance(sell, dict) else "shadow sell",
            is_partial=is_partial,
        )
        if is_partial and isinstance(sell, dict) and sell.get("reset_peak"):
            position.peak_price = position.current_price
            position.peak_date = datetime.now(timezone.utc)
        n += 1
    return n


def _persist_peaks(persist: Session, strategy_id: int,
                   shadow_session: "ShadowSession") -> None:
    """Upsert ratcheted peaks for the strategy's open positions and drop rows
    whose positions closed. Must run EVERY tick (not just trade-emitting
    ones) — the ratchet is what carries highs across ticks. Writes the
    in-memory value unconditionally: it already reflects max(stored, live)
    from the overlay, and a lower value is an intentional partial-trailing
    reset. Commits the persist session (flushing any pending ShadowTrades
    the caller added first).
    """
    open_positions = list(shadow_session._synthetic_positions or [])
    rows = persist.query(ShadowPositionPeak).filter(
        ShadowPositionPeak.shadow_strategy_id == strategy_id).all()
    by_ticker = {r.ticker: r for r in rows}
    seen = set()
    for pos in open_positions:
        if not pos.ticker or pos.peak_price is None:
            continue
        seen.add(pos.ticker)
        row = by_ticker.get(pos.ticker)
        if row is None:
            persist.add(ShadowPositionPeak(
                shadow_strategy_id=strategy_id,
                ticker=pos.ticker,
                opened_at=pos.purchase_date,
                peak_price=pos.peak_price,
                peak_date=pos.peak_date,
            ))
        else:
            row.opened_at = pos.purchase_date  # re-keys on reopen generations
            row.peak_price = pos.peak_price
            row.peak_date = pos.peak_date
    for r in rows:
        if r.ticker not in seen:
            persist.delete(r)
    persist.commit()


def run_shadow_strategies(real_db: Session, analysis_results: List[dict]) -> dict:
    """Run all active shadow strategies for one scan tick.

    Called from scheduler.run_continuous_scan AFTER live AI trading committed.
    Catches per-strategy exceptions so a shadow failure cannot break the live
    save path. Returns a small summary dict for logging/metrics.
    """
    summary = {"strategies_run": 0, "total_shadow_trades": 0, "errors": 0}
    try:
        active = real_db.query(ShadowStrategy).filter(
            ShadowStrategy.archived_at.is_(None)
        ).all()
    except Exception as e:
        logger.error(f"Shadow strategy query failed: {e}", exc_info=True)
        return summary

    if not active:
        return summary

    logger.info(f"Shadow paper-trading: running {len(active)} active strategy(ies)")

    # Blocking acquire: the scan-tick run must never skip. The intraday stop
    # pass yields (non-blocking acquire) so waits here are ≤ one pass.
    with _shadow_run_lock:
        for strategy in active:
            try:
                n_trades = _run_one_strategy(strategy.id, analysis_results)
                summary["strategies_run"] += 1
                summary["total_shadow_trades"] += n_trades
            except Exception as e:
                summary["errors"] += 1
                logger.error(
                    f"Shadow strategy {strategy.name} (id={strategy.id}) failed: {e}",
                    exc_info=True,
                )

    if summary["total_shadow_trades"]:
        logger.info(
            f"Shadow paper-trading: {summary['strategies_run']} stack(s), "
            f"{summary['total_shadow_trades']} virtual trade(s) emitted"
        )
    return summary


def _run_one_strategy(strategy_id: int, analysis_results: List[dict]) -> int:
    """Run one shadow strategy in a sandboxed session.

    Returns the number of shadow trades persisted.

    Sandbox rollback discards any side effects from live evaluators (SystemState
    writes, MLPrediction inserts, in-place ORM mutations on AIPortfolioConfig).
    Pending shadow trades are persisted via a separate clean session so they
    aren't part of the rolled-back transaction.
    """
    sandbox = SessionLocal()
    persist = SessionLocal()
    try:
        strategy = sandbox.query(ShadowStrategy).filter(ShadowStrategy.id == strategy_id).first()
        if not strategy:
            logger.warning(f"Shadow strategy id={strategy_id} not found")
            return 0
        if strategy.archived_at is not None:
            return 0

        # Apply scorer_overrides to BOTH halves of the read path:
        # analysis_results (feeds synthetic positions' current_score for the
        # SELL evaluator) and sandbox Stock rows (evaluate_buys reads these).
        ar_for_session = analysis_results or []
        overrides = strategy.scorer_overrides or {}
        if overrides.get("disable_excellence_cap"):
            ar_for_session = _apply_disable_excellence_cap(ar_for_session)
            n_patched = _patch_sandbox_stocks_disable_cap(sandbox, ar_for_session)
            logger.info(
                f"shadow[{strategy.name}]: disable_excellence_cap active "
                f"(patched {n_patched} sandbox Stock row(s))"
            )

        shadow_session = ShadowSession(sandbox, strategy, ar_for_session)

        # Lazy import to dodge circular dependency at module load time.
        from backend.ai_trader import evaluate_buys, evaluate_sells

        # ── SELL phase ───────────────────────────────────────────────────────
        try:
            sells = evaluate_sells(shadow_session, user_id=SHADOW_USER_ID)
        except Exception as e:
            logger.error(f"shadow evaluate_sells failed for {strategy.name}: {e}", exc_info=True)
            sells = []

        _emit_sell_decisions(shadow_session, sells)

        # ── BUY phase ────────────────────────────────────────────────────────
        # ftd_penalty_active / heat_penalty_active default False — see module
        # docstring's Limitations. Step 3+ will lift these into shadow_trader.
        try:
            buys = evaluate_buys(
                shadow_session,
                ftd_penalty_active=False,
                heat_penalty_active=False,
                user_id=SHADOW_USER_ID,
            )
        except Exception as e:
            logger.error(f"shadow evaluate_buys failed for {strategy.name}: {e}", exc_info=True)
            buys = []

        # Execution guard (2026-07-22): evaluate_buys returns a ranked DECISION
        # list; the LIVE cycle executes top-N until slots and cash run out,
        # but this loop used to persist EVERY decision. Harmless while stacks
        # held full books (0-1 slots open); a FRESH stack with a wide
        # candidate pool exploded — shadow_chop_damper's first cycle bought
        # 76 names / $127.7k on a $25k stack. Mirror live execution: consume
        # slots and cash in rank order, stop when either runs out.
        cfg = shadow_session._synthetic_config
        open_positions = len(shadow_session._synthetic_positions or [])
        remaining_slots = max(0, int(getattr(cfg, "max_positions", 8) or 8) - open_positions)
        remaining_cash = float(getattr(cfg, "current_cash", 0) or 0)
        for buy in buys:
            if remaining_slots <= 0 or remaining_cash <= 0:
                break
            stock = buy.get("stock") if isinstance(buy, dict) else None
            if stock is None:
                continue
            ticker = getattr(stock, "ticker", None)
            value = float(buy.get("value", 0) or 0) if isinstance(buy, dict) else 0
            price = getattr(stock, "current_price", None) or 0
            if not ticker or value <= 0 or price <= 0:
                continue
            if value > remaining_cash:
                value = remaining_cash  # partial last fill, matches live cash exhaustion
                if value / price <= 0:
                    continue
            remaining_slots -= 1
            remaining_cash -= value
            shares = value / price
            shadow_session.emit_shadow_buy(
                ticker=ticker,
                shares=shares,
                price=price,
                reason=buy.get("reason", "shadow buy") if isinstance(buy, dict) else "shadow buy",
                canslim_score=getattr(stock, "canslim_score", None),
                signal_factors=buy.get("signal_factors") if isinstance(buy, dict) else None,
                is_growth_stock=bool(buy.get("is_growth_stock", False)) if isinstance(buy, dict) else False,
            )

        # ── Persist ──────────────────────────────────────────────────────────
        pending = shadow_session.drain_pending_shadow_trades()
        n_persisted = 0
        if pending:
            for st in pending:
                persist.add(st)
            # Mark first activation
            strategy_row = persist.query(ShadowStrategy).filter(
                ShadowStrategy.id == strategy_id
            ).first()
            if strategy_row and strategy_row.activated_at is None:
                strategy_row.activated_at = datetime.now(timezone.utc)
            persist.commit()
            n_persisted = len(pending)
            logger.info(
                f"shadow[{strategy_row.name if strategy_row else strategy_id}]: "
                f"persisted {n_persisted} virtual trade(s)"
            )

        # Peak ratchet persists EVERY tick, trade or no trade — carrying the
        # session high forward is the whole point (see ShadowPositionPeak).
        try:
            _persist_peaks(persist, strategy_id, shadow_session)
        except Exception as e:
            logger.error(f"shadow peak persistence failed (id={strategy_id}): {e}", exc_info=True)

        return n_persisted
    finally:
        try:
            sandbox.rollback()
        except Exception:
            pass
        sandbox.close()
        persist.close()


# ── Intraday stop pass (2026-07-25) ──────────────────────────────────────────
# The 15-min intraday stop job (scheduler, 2b88bfc) gave the LEAD book intraday
# hard stops and reliable close-window trailing, while shadow stacks still
# exited only when a scan tick happened to run. Verdicts are shadow-vs-shadow,
# so no A/B was biased — but shadow stops slipped past their triggers the way
# lead stops did pre-2b88bfc, and trailing only fired when a scan tick landed
# inside the 15:00-16:00 ET window. This pass runs the live evaluate_sells on
# each stack and emits ONLY stop-family decisions, matching the lead intraday
# checker's scope: score / take-profit / pre-earnings sells stay on scan
# cadence, exactly as they do for the lead.

_STOP_FAMILY_PREFIXES = ("STOP LOSS", "TRAILING STOP", "PARTIAL TRAILING STOP")


def is_stop_family_reason(reason: Optional[str]) -> bool:
    """True for the sell reasons the lead intraday checker also evaluates.
    PRE-EARNINGS STOP is deliberately excluded — the lead checker does not
    evaluate earnings tightening intraday either."""
    return bool(reason) and str(reason).startswith(_STOP_FAMILY_PREFIXES)


def run_shadow_stop_checks() -> dict:
    """Intraday stop-cadence parity for shadow stacks.

    Called from the scheduler's 15-min intraday stop job after the lead
    users' check. Skips the tick entirely (non-blocking lock) if a scan-tick
    shadow run is mid-flight — the scan run will evaluate the same stops.
    """
    summary = {"strategies_run": 0, "stop_sells": 0, "errors": 0, "skipped": False}
    if not _shadow_run_lock.acquire(blocking=False):
        summary["skipped"] = True
        logger.info("Shadow intraday stop check skipped — scan-tick shadow run in progress")
        return summary
    try:
        listing = SessionLocal()
        try:
            active_ids = [
                s.id for s in listing.query(ShadowStrategy)
                .filter(ShadowStrategy.archived_at.is_(None)).all()
            ]
        finally:
            listing.close()

        for sid in active_ids:
            try:
                summary["stop_sells"] += _run_one_stop_check(sid)
                summary["strategies_run"] += 1
            except Exception as e:
                summary["errors"] += 1
                logger.error(
                    f"Shadow intraday stop check failed for strategy id={sid}: {e}",
                    exc_info=True,
                )
    finally:
        _shadow_run_lock.release()

    if summary["stop_sells"]:
        logger.info(f"Shadow intraday stop check: {summary['stop_sells']} stop sell(s) emitted")
    return summary


def _run_one_stop_check(strategy_id: int) -> int:
    """Stops-only intraday evaluation for one shadow stack.

    Same sandbox/persist lifecycle as _run_one_strategy, but: empty
    analysis_results (prices come from a live refresh instead), sells only,
    and only stop-family decisions execute.
    """
    sandbox = SessionLocal()
    persist = SessionLocal()
    try:
        strategy = sandbox.query(ShadowStrategy).filter(
            ShadowStrategy.id == strategy_id).first()
        if not strategy or strategy.archived_at is not None:
            return 0

        shadow_session = ShadowSession(sandbox, strategy, [])
        if not shadow_session._synthetic_positions:
            _persist_peaks(persist, strategy_id, shadow_session)  # prune closed rows
            return 0

        # Lazy import (module-load circularity, same as _run_one_strategy).
        from backend.ai_trader import evaluate_sells, update_position_prices

        # Live-quote refresh against synthetic positions (query intercept):
        # sets current_price/gain_loss_pct, ratchets peaks on new highs, and
        # fills current_score from the Stock table for the partial gate.
        update_position_prices(shadow_session, use_live_prices=True, user_id=SHADOW_USER_ID)

        try:
            sells = evaluate_sells(shadow_session, user_id=SHADOW_USER_ID)
        except Exception as e:
            logger.error(
                f"shadow intraday evaluate_sells failed for {strategy.name}: {e}",
                exc_info=True,
            )
            sells = []

        stop_sells = [
            s for s in sells
            if isinstance(s, dict) and is_stop_family_reason(s.get("reason"))
        ]
        n = _emit_sell_decisions(shadow_session, stop_sells)

        pending = shadow_session.drain_pending_shadow_trades()
        for st in pending:
            persist.add(st)
        _persist_peaks(persist, strategy_id, shadow_session)  # commits trades too

        if n:
            logger.info(f"shadow[{strategy.name}]: intraday pass emitted {n} stop sell(s)")
        return n
    finally:
        try:
            sandbox.rollback()
        except Exception:
            pass
        sandbox.close()
        persist.close()

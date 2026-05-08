# Shadow Paper-Trading Design

**Status**: Design — no code written. Awaiting CTO review before implementation.
**Author**: Claude (session 2026-05-07)
**Context**: Approach 2 live A/B (`ec73f83`, eval window 2026-05-07 → 2026-06-18) demonstrated the value of forward-only scoring evaluation. We need a way to evaluate *the next* scoring change without spending six weeks of live capital.

## Why we are not rebuilding the backtester

`canslim_scorer-refactor-may6.md` documented the structural blocker: `BacktestStaticSnapshot` freezes today's point-in-time scalars (institutional ownership, analyst counts, sentiment, sector rank). Replaying 2022-era trades against 2026-era fundamentals scored frozen=+136.84% vs fresh-compute=-14.99% — the May 6 Bundle 2 revert (`c4a792a`, -79pp/-64pp split-test) confirmed that fresh-compute against a frozen snapshot is semantically broken.

A "true point-in-time" backfill (FMP historical fundamentals for 2,000+ tickers × 4 years) is a 2-3 month engineering project that still leaves unrecoverable gaps in soft signals (institutional ownership history, historical analyst counts, sentiment archives). It would also re-create exactly the data-tampering failure mode the snapshot was supposed to fix.

The `/admin/strategy-ab-eval` framework shipped in `c3a6356` is the correct tool for scoring evaluation: forward-only, real point-in-time data, 4-6 week verdicts. Shadow paper-trading extends that framework so we can run *several* candidate scoring stacks in parallel with the live one, without spending capital on each.

## Goal

A second (or third) scoring stack runs alongside the live scanner. It emits virtual buy/sell decisions to a shadow log. The existing `/admin/strategy-ab-eval` endpoint reads from the shadow log when invoked with `?source=shadow`, comparing alternative scoring rules against the live baseline using the same window-resolution + summarization + decision logic that's already test-locked.

The end state: any scoring change can sit in shadow for 4-6 weeks while live trades continue under the production scorer. If the shadow stack outperforms by the existing decision criteria, it graduates to live.

## Data model

Two new tables. Both live in `backend/database.py`.

### `ShadowStrategy`

Identifies a candidate scoring stack and pins its config snapshot at registration time.

```python
class ShadowStrategy(Base):
    __tablename__ = "shadow_strategies"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)        # e.g. "approach_3_sector_aliases"
    parent_strategy = Column(String, nullable=False)           # e.g. "nostate_cs_bear" — comparison baseline
    config_snapshot = Column(JSON, nullable=False)             # frozen YAML profile at registration
    scorer_overrides = Column(JSON, nullable=False)            # {"c_score_excellence_cap": 18, ...}
    description = Column(String)
    starting_value = Column(Float, nullable=False, default=25000.0)  # virtual cash, mirrors AIPortfolioConfig
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    activated_at = Column(DateTime)                            # null until first shadow scan writes a trade
    archived_at = Column(DateTime)                             # null while running; set to stop emitting trades
```

`config_snapshot` matters because YAML profiles get edited. Without freezing it at registration time, a 4-week shadow run could be reading a config that drifted under it. Same defensive freeze pattern as `BacktestRun.profile_overrides`.

### `ShadowTrade`

Mirrors `AIPortfolioTrade` columns one-to-one so the existing `_summarize_window` aggregation works without per-source code paths.

```python
class ShadowTrade(Base):
    __tablename__ = "shadow_trades"
    id = Column(Integer, primary_key=True)
    shadow_strategy_id = Column(Integer, ForeignKey("shadow_strategies.id"), nullable=False, index=True)
    ticker = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False)            # BUY, SELL — no PYRAMID for v1
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    reason = Column(String)
    canslim_score = Column(Float)                       # shadow-stack score at time of trade
    cost_basis = Column(Float)                          # SELL only
    realized_gain = Column(Float)                       # SELL only
    holding_days = Column(Integer)                      # SELL only
    signal_factors = Column(JSON)                       # same shape as live: entry_type, market_regime, etc.
    executed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    __table_args__ = (
        Index('ix_shadow_trades_strategy_executed', 'shadow_strategy_id', 'executed_at'),
    )
```

We deliberately mirror `AIPortfolioTrade` instead of inheriting or aliasing because:

1. `user_id` is required on the live table and conceptually wrong for shadow (no user owns a virtual trade).
2. Foreign keys differ (`shadow_strategy_id` vs `users.id`).
3. SQLAlchemy single-table inheritance would complicate the existing `AIPortfolioTrade` index `ix_trades_user_executed` and force every existing query to filter `discriminator IS NULL`.

Two tables, one parallel summarizer call. Cleaner than one polymorphic table.

### `ShadowPosition` (optional v1)

Whether v1 needs persisted virtual positions depends on which path we take below. If we re-derive positions from the trade log on every scan tick (FIFO BUY/SELL pairing), we can skip this table. If we go stateful, we need it. **Recommendation: re-derive for v1** — same FIFO logic that `ml/feature_extractor.extract_live_trade_data` already uses to pair BUYs with SELLs. State is implicit in the trade log; positions become a SELECT view, not a persisted entity. We pay for that on every scan tick (2,000 tickers × N shadow stacks × pairing pass), but the math is in-memory python and trivially cheap relative to the network calls already in the scan.

If profiling shows the re-derivation is too expensive at 4+ shadow stacks, add `ShadowPosition` as a denormalized cache.

## Execution path

The smallest hook is in `backend/scheduler.run_continuous_scan` between **scan completion** (line 1338, `analysis_results = run_async_scan(...)`) and **save begin** (line 1359, `for i, analysis in enumerate(analysis_results):`). At that point the live scan has already paid for the network round-trips; `analysis_results` is a list of dicts containing every field a scorer needs (c_score, a_score, n_score, …, next_earnings_date, soft_zone state, sector_rs_rank).

The shadow path is roughly:

```python
# After live save loop completes (~line 1450, after batched commits flush)
from backend.shadow_trader import run_shadow_strategies
try:
    run_shadow_strategies(db, analysis_results)
except Exception as e:
    logger.error(f"Shadow paper-trading run failed: {e}", exc_info=True)
    # Never let shadow failures break the live scan path
```

`run_shadow_strategies` iterates over `ShadowStrategy.archived_at IS NULL` rows. For each, it:

1. Re-scores every ticker in `analysis_results` using the strategy's `scorer_overrides` (e.g. excellence-tier cap value, sector alias map).
2. Replays `ai_trader.evaluate_buys` and `ai_trader.evaluate_sells` on the shadow strategy's virtual position book (re-derived from the `ShadowTrade` log).
3. Persists any virtual BUY/SELL decisions to `ShadowTrade`.

Two important constraints:

- **No fetch path divergence.** The shadow scorer must consume `analysis_results[i]` as-is. If a candidate stack needs a field that the live `analysis` dict doesn't carry, we add the field to the live dict (not a parallel fetch). This is the only way to keep cost bounded as shadow stacks accumulate.
- **No live-state mutation.** Shadow scoring must not write to `Stock`, `StockScore`, or any table other than `ShadowTrade` / `ShadowPosition`. A bug in a shadow stack must not be able to corrupt the live scoring layer.

### Where the scoring math swaps

`canslim_scorer.CANSLIMScorer.score_stock` (line 96) is the live entry. For v1 the shadow path re-uses the same class, but with an alternative config dict:

```python
shadow_scorer = CANSLIMScorer(config=shadow_strategy.config_snapshot)
shadow_score = shadow_scorer.score_stock(stock_data, industry_group_rank)
```

`CANSLIMScorer` already accepts a config arg — same surface the existing test fixtures use. The shadow stack swaps that arg out and gets a different score for the same input data. No core scorer changes needed.

For more invasive scoring experiments (a fundamentally different scoring formula, not just parameter tweaks), the shadow stack can register a different scorer class via a `scorer_class_name` field on `ShadowStrategy` — load by import path, instantiate, score. Out of scope for v1; called out as v2 escape hatch.

## Selection / config

Shadow stacks are configured in `config/default.yaml` under a new `shadow_strategy_profiles:` section. Schema mirrors `strategy_profiles:` with two added keys:

```yaml
shadow_strategy_profiles:
  approach_3_sector_aliases:
    label: "Approach 3 (sector aliases experiment)"
    description: "Reintroduce sector alias map reverted in c4a792a + Turnaround ROE"
    shadow_of: nostate_cs_bear        # parent_strategy in the data model
    starting_value: 25000.0
    # — everything below is a strategy_profiles override —
    min_score: 72
    max_positions: 8
    # ...
    scorer_overrides:
      sector_alias_map_enabled: true
      turnaround_roe_signal: true
```

On scheduler startup, `_sync_shadow_strategies_from_yaml(db)` reads the section and:
- Creates a `ShadowStrategy` row for any name not already in the table.
- Updates `archived_at = now()` for any DB row whose YAML entry was removed.

Crucially: **does not** mutate `config_snapshot` for already-active rows. YAML edits to a running shadow stack would invalidate the comparison; force the user to register a new name (e.g. `approach_3_sector_aliases_v2`) for any meaningful change. Same discipline as `BacktestRun.profile_overrides`.

## Eval surface

Extend `/admin/strategy-ab-eval` and `/admin/strategy-ab-eval-trades` with a `source` query param:

```
GET /api/admin/strategy-ab-eval?strategy=nostate_cs_bear&cutoff_date=2026-05-07
GET /api/admin/strategy-ab-eval?strategy=approach_3_sector_aliases&cutoff_date=2026-05-07&source=shadow
```

`source=live` (default) reads `AIPortfolioTrade` — the existing 32+28 lockfile tests stay untouched.
`source=shadow` reads `ShadowTrade` filtered by the named `ShadowStrategy`.

`_resolve_ab_window` already does the heavy lifting: window math, user/trade resolution, exclude-pyramids filter, starting-value sum. The only change is the table source. Concretely, refactor `_resolve_ab_window` so the trade-fetch query is built by a small helper:

```python
def _build_trade_query(db, source, strategy_or_shadow_name, user_ids_or_shadow_id, ...):
    if source == "shadow":
        return db.query(ShadowTrade).filter(ShadowTrade.shadow_strategy_id == ...)
    return db.query(AIPortfolioTrade).filter(AIPortfolioTrade.user_id.in_(...))
```

Everything downstream (`_summarize_window`, `_decide`) takes a list of trade-like objects with `.action`, `.realized_gain`, `.executed_at`, `.cost_basis` — both tables already have those columns by design (see Data Model section). Zero changes to summarizer or decision logic.

### Frontend (`frontend/src/pages/ABEval.jsx`)

Add a top-bar source toggle: `Live` / `Shadow: <strategy_name>`. When shadow is selected, the strategy dropdown populates from `GET /api/admin/shadow-strategies` (a new tiny route that lists `ShadowStrategy` rows with `archived_at IS NULL`). Everything else on the page stays identical — same decision banner, same delta table, same warnings, same per-trade chart.

This is intentionally minimal UI change. The page is already battle-tested on live data; we want shadow to look identical so verdicts are comparable at a glance.

### Weekly snapshot email

`backend/scheduler._run_weekly_ab_eval_email` becomes parameterized over `(strategy, source)`. Schedule one weekly job per shadow stack with `archived_at IS NULL`, in addition to the live job. Subject lines disambiguate: `"A/B Eval: nostate_cs_bear (LIVE)"` vs `"A/B Eval: approach_3_sector_aliases (SHADOW)"`. Email body unchanged.

## Bootstrap: how do we get enough trades to compare?

This is the hard question. A shadow stack registered today has zero trade history. The decision logic in `_decide` enforces `min_post_sells=5` before declaring keep/revert/marginal — without that floor, decisions are noise.

Two options:

### Option A — Forward-only accumulation (recommended for v1)

Register the shadow stack. Wait. After 4-6 weeks (same window the live A/B uses), `_summarize_window` has enough SELLs in `ShadowTrade` to render a verdict. The `cutoff_date` for shadow comparison is the registration date.

**Why recommended**: zero new code paths, no historical replay risk, perfectly mirrors what live A/B already does. The eval framework was designed for forward-only data and is test-locked against it.

**Cost**: 4-6 week wait per experiment. Mitigated by running multiple candidate stacks in parallel — five shadow stacks registered Monday all reach verdict-readiness in the same six weeks, so we can evaluate five experiments in the time it took to evaluate one (Approach 2).

### Option B — Replay live `StockScore` history through the shadow scorer (rejected for v1)

For each ticker in `StockScore` over the past M months, re-score with the shadow scorer using the *frozen-at-the-time* fundamentals. Synthesize virtual BUY/SELL decisions from the score deltas.

**Why rejected for v1**: This is the backtester problem we just rejected. `StockScore` rows don't carry the full feature input the shadow scorer needs (institutional ownership at scan time, sector rank at scan time, base-pattern state, soft-zone width). Reconstructing those is the snapshot-scalar problem — same data integrity failure mode as Approach 2 Bundle 2.

We can revisit Option B in v2 *only* for shadow stacks whose `scorer_overrides` are pure transformations of the existing `StockScore` row (e.g. an arithmetic re-weighting of c/a/n/s/l/i/m). For overrides that need feature inputs not on `StockScore`, Option A is the only honest path.

## Cost estimate

A scan tick currently runs ~35 minutes wall-clock on the VPS at batch_size=100 over ~2,000 tickers. The dominant cost is network: FMP + Yahoo Finance fetches in `run_async_scan`.

Per-shadow-stack overhead:

- **Re-scoring** ~2,000 tickers from in-memory `analysis_results`: ~5-15 seconds at full python speed (no I/O, all dict lookups + arithmetic). `CANSLIMScorer.score_stock` is lightweight; the 2,000-ticker loop is the bound. Profile in the spike branch to confirm.
- **Re-deriving positions** from `ShadowTrade` FIFO pairing: O(trades) for a single SELECT + Python pass. At ~10 trades/week per stack, that's a no-op even at year scale.
- **Evaluating buys/sells** via `ai_trader.evaluate_buys/evaluate_sells`: same code path live trading uses. Already sub-second per scan tick on the live side.
- **Persisting trades**: 0-3 INSERTs per scan tick per stack, batched commit.

Worst-case projection at **3 shadow stacks**: scan tick goes from ~35 minutes to ~36 minutes. Still well inside the 35-min cron interval — ample headroom.

If we ever push past **8 shadow stacks**, the re-scoring loop starts dominating. At that point we either:
1. Cap shadow registrations (probably the right answer — we're not running a research lab).
2. Parallelize the re-score loop across stacks (multiprocessing pool, 1 stack per worker; still on the in-memory `analysis_results`).

V1 does neither. Cap at 5 active shadow stacks via a YAML validation check; revisit if we ever feel the cap.

## Test plan

- `tests/test_shadow_trader.py`: scoring swap correctness (same input → different output for different `scorer_overrides`), FIFO position re-derivation, evaluate_buys parity with live, persist-only-on-decision, archive flag respected.
- `tests/test_strategy_ab_eval.py`: extend with `source=shadow` parametrization. Existing 32 live-source tests stay green by virtue of `source` defaulting to `live`.
- `tests/test_strategy_ab_eval_trades.py`: same — extend per-trade endpoint with `source=shadow`. Existing 28 stay green.
- `tests/test_ab_eval_email.py`: weekly job parametrization tests. Existing 15 stay green.
- New table migrations need `tests/test_database_migrations.py` row-counts pinned.

Suite goal: 2163 + ~40 new shadow tests passing / 16 skipped / 1 pre-existing fail. No regressions in lockfile suites.

## Eval-safety boundary (during 2026-05-07 → 2026-06-18 window)

Shadow paper-trading is **eval-safe to ship** during the Approach 2 window because:
- It does not touch `canslim_scorer.py`, `growth_projector.py`, or live `ai_trader.py` trading logic.
- It does not modify `/api/admin/strategy-ab-eval` or `/strategy-ab-eval-trades` *response shapes* — only adds an opt-in `source` parameter that defaults to existing behavior.
- It does not touch `email_utils.py` signatures.
- The scan-loop hook is purely additive and wrapped in a try/except that cannot break the live save path.

That said, the *implementation* cost is non-trivial (database migration, two new tables, scorer-config plumbing, frontend toggle, ~40 new tests). Recommendation: **start the spike-branch implementation now, but do not merge until 2026-06-18 verdict closes** — gives us time to dogfood the shadow path on a no-op test stack (e.g. `shadow_baseline` with zero overrides should produce trades nearly identical to live; any divergence reveals an implementation bug). At verdict time we have a battle-tested shadow harness ready to register the *next* candidate stack.

## Open questions

1. **Does shadow need its own ML model state?** Live ML signal layer (v12 active, see `canslim-ml-graduation-apr29.md`) gates trades via `ml_signal.min_confidence`. If a shadow stack experiments with a different `ml_signal.weight` or `ml_signal.log_only`, do we re-run the live model on the same input, or train a parallel model? **Probable answer**: re-run the live model. Training a parallel model is out of scope for shadow paper-trading; we only experiment with how the existing model's output is *used*. Validate with the user before scoping.

2. **Sector / sizing constraints in shadow.** `max_sector_pct=50` is enforced at trade time on the live book. Shadow re-derives positions from its trade log; does it also re-derive sector exposure? Yes — the `Stock.sector` field is point-in-time-stable enough that a SELECT-and-aggregate on the shadow book gives the right answer. But if a sector reclassification happens mid-run, shadow and live diverge slightly. Negligible for v1.

3. **Shadow stack interactions with breakout monitor / coiled spring alerts.** These are signal-generation layers downstream of scoring. Should shadow stacks see them? Probably not for v1 — keeps the shadow surface a pure scoring + entry-rule experiment. Signal-layer experiments are a separate framework, deferred.

4. **Per-user shadow stacks.** Currently shadow is global (one virtual book per stack). If we ever want per-user shadow A/B (e.g. user A on Approach 3, user B on Approach 4), the `ShadowStrategy` table needs a `user_id` column. v1 does not need this. Revisit if it becomes a real ask.

5. **Archival semantics.** When we archive a shadow stack (`archived_at = now`), do we keep its trades around for historical comparison or hard-delete? Keep them (cheap on disk; valuable for retrospectives). Add a `GET /api/admin/shadow-strategies?archived=true` listing for the UI.

## Implementation phasing

If the user approves this design, suggested order:

1. **DB migration + models** (1 day): `ShadowStrategy` + `ShadowTrade` tables, alembic migration, `tests/test_database_migrations.py` pin.
2. **Scorer config swap + scan-loop hook** (1 day): `backend/shadow_trader.py` module, scheduler wiring inside the existing try/except, no-op shadow stack ("shadow_baseline" mirroring nostate_cs_bear) to validate parity.
3. **Eval endpoint extension** (0.5 day): `source=shadow` param on `/strategy-ab-eval` and `/strategy-ab-eval-trades`, refactor `_build_trade_query` helper.
4. **YAML config + sync** (0.5 day): `shadow_strategy_profiles:` section, `_sync_shadow_strategies_from_yaml` on scheduler startup.
5. **Frontend toggle** (0.5 day): `ABEval.jsx` source dropdown, shadow-strategies listing route.
6. **Weekly email parametrization** (0.5 day): one job per active shadow stack, subject-line disambiguation.
7. **Dogfood window** (1-2 weeks idle): no-op `shadow_baseline` runs; verify SELL counts and decision verdicts converge with the live `nostate_cs_bear` numbers.
8. **First real candidate stack** (post-2026-06-18): register the next scoring experiment as a shadow stack, publish 4-6 week verdict timeline.

Total active-engineering: ~4 days. Total wait time before first real verdict: ~6-8 weeks from go.

## Provisioning shadow stacks (Step 4 — shipped)

Operator workflow once Step 4 is deployed:

1. Edit `config/default.yaml` — add or remove an entry under
   `shadow_strategy_profiles:`:
   ```yaml
   shadow_strategy_profiles:
     shadow_baseline:
       parent_strategy: nostate_cs_bear
       starting_value: 25000
       description: "Forward-only parity check against live cs_bear baseline"
       scorer_overrides: {}
   ```
2. Push + deploy. On scheduler boot, `backend.shadow_strategy_sync.
   sync_shadow_strategies_from_yaml(db)` reconciles the YAML against the
   `shadow_strategies` table. The boot log shows
   `shadow_strategy_profiles sync complete: inserted=[...] updated=[...]
   archived=[...] skipped=[...]`.
3. The stack lights up in `/admin/shadow-strategies` and starts emitting
   virtual trades on the next scan tick.

Reconcile semantics:

- **YAML name not in DB** → INSERT (config_snapshot frozen from current YAML
  parent profile).
- **YAML name in active DB row** → UPDATE mutable fields only
  (`description`, `scorer_overrides`, `starting_value`).
  `config_snapshot` and `parent_strategy` are immutable: changing them
  mid-stream would silently invalidate forward-only telemetry. YAML drift
  is logged as a WARNING but not applied.
- **YAML name in archived DB row** → REACTIVATE in place (`archived_at` =
  None, `activated_at` = now, mutable fields refreshed). Existing
  `shadow_trades` rows stay attached via FK to id. Reactivate-in-place
  rather than insert-fresh because the schema enforces UNIQUE(name) and
  admin readers do name-based lookups. Operators wanting a clean
  forward-only window after a long archive should rename the stack.
- **DB name not in YAML and currently active** → SOFT-ARCHIVE
  (`archived_at` = now). `shadow_trades` rows are preserved.

Validation: `parent_strategy` must resolve in the running
`strategy_profiles:` config. Bad parents raise `ValueError` and the entire
reconcile transaction rolls back — partial inserts are never persisted.
Boot continues unimpaired (logged at ERROR; the live scanner doesn't depend
on shadow stacks).

Limitation: `scorer_overrides` is recorded but NOT applied during shadow
scan. That lands in Step 7 (post-eval, requires CANSLIMScorer config
refactor). Until then, every registered shadow stack effectively scores
identically to its `parent_strategy`, which is fine for the parity-check
use case (`shadow_baseline`) but means non-baseline candidates can't yet
be meaningfully evaluated.

## Step 5 — UI selector (shipped)

Wires the AB-eval dashboard to the shadow registry so operators can read
shadow verdicts without hand-editing query strings.

- The `Strategy` dropdown on `/admin/ab-eval` is replaced by a `Source`
  dropdown. First option is fixed: `Live: nostate_cs_bear (Approach 2)`.
  Subsequent options are pulled from
  `GET /api/admin/shadow-strategies?archived=false` and rendered as
  `Shadow: <name> — <description>`.
- Selecting a shadow option swaps `source=shadow&strategy=<name>` on
  both `/api/admin/strategy-ab-eval` and
  `/api/admin/strategy-ab-eval-trades` calls in parallel. Window math,
  pyramid filter, and pre/post params are unchanged.
- Selection is persisted to `localStorage["abeval.selected"]` so a page
  reload mid-watch doesn't bounce the operator back to live.
- The "Send test email" button is hidden when shadow is selected. The
  weekly cron (`backend.scheduler._run_weekly_ab_eval_email`) is still
  hard-coded to `nostate_cs_bear` + the Approach 2 cutoff, so a
  test-email on shadow params would deliver a misleading body. Email
  parametrization for shadow stacks is Step 6.
- A caption renders under the controls when shadow is active:
  *"scorer_overrides not yet applied — shadow stack scores identically
  to {parent_strategy} until Step 7."* Sets correct operator
  expectations during the dogfood window where the only registered
  stack is the parity-check `shadow_baseline`.

Why ship this before Step 7's real candidate: by the time the first real
candidate stack lands post-2026-06-18, we want to read its verdict on a
trusted dashboard, not debug the harness under decision pressure. The
`shadow_baseline` parity readout against live `nostate_cs_bear` is the
acceptance test — they should converge to identical SELL counts and
decision verdicts.

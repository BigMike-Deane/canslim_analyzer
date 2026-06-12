# Road Session Log — June 2026

Cross-session continuity log (see `docs/ROAD-HANDOFF.md`). Append a dated
entry at the end of any session that ships, decides, or discovers something.

---

## 2026-06-11 — handoff verification (cloud session)

Ran the ROAD-HANDOFF check list. Findings:

- **Deps**: core imports clean (`fastapi, sqlalchemy, yfinance, pandas,
  numpy, aiohttp, httpx, yaml, redis`); session-start pip hook ran OK.
- **Live API: BLOCKED — two toggles missing.** (1) `CANSLIM_API_TOKEN`
  env var is absent from this session. (2) The egress proxy returns
  `403 "Host not in allowlist"` for `canslim.duckdns.org`, so even with a
  token the host is unreachable. Could **not** read the Edge Scorecard
  (alpha) or run `/api/admin/strategy-ab-eval` for `nostate_cs_bear`
  (cutoff 2026-05-07, pre_window_days=14). Owner: add the env var **and**
  allowlist `canslim.duckdns.org` in the claude.ai environment config to
  unblock.
- **Tests**: `pytest tests/test_spy_overlay.py -q` → **11 passed**, 1 warning.
- No code touched (freeze on `ai_trader.py`/`canslim_scorer.py` respected;
  this was docs-only).

## 2026-06-11 — handoff verification, re-run (cloud session, live API now unblocked)

Re-ran the ROAD-HANDOFF checklist. The earlier "BLOCKED" entry above is now
superseded — both toggles are in place this session.

- **Deps**: core imports clean (`fastapi, sqlalchemy, yfinance, pandas,
  numpy, aiohttp, httpx, yaml, redis`); session-start pip hook ran OK.
- **Live API: UNBLOCKED.** `CANSLIM_API_TOKEN` present (len 140) and
  `canslim.duckdns.org` reachable (HTTP 200, no egress 403).
- **Edge Scorecard** (`/api/ai-portfolio/edge`, all history since inception
  2026-04-07, as-of 2026-06-11): **alpha +7.05%** (beta-adj). Total return
  +24.56% vs SPY +11.91% (excess +12.65pp), Sharpe 3.94, beta 1.47, max DD
  −5.72%, win rate 63.0% over 27 closed trades, 47 trading days.
- **A/B eval** `nostate_cs_bear`, cutoff 2026-05-07, pre_window_days=14
  (post auto = 35d): **decision = KEEP.** Return delta −2.68pp (≥ −5pp floor),
  Sharpe delta +0.027 (≥ 0). Post win rate 76.7% vs pre 68.4%, 30 post sells.
  Warnings: 14d pre window noisy + pre/post length mismatch >50%.
- **Tests**: `pytest tests/test_spy_overlay.py -q` → **11 passed**, 1 warning.
- No code touched (freeze respected; docs-only).

## 2026-06-11 — D1 behavioral parity harness + shadow-arm cross-check (cloud session)

Acting on the CTO/CSO read that D1 is the highest-value next item (live money
leak + it's contaminating the June-18 C-score-cap verdict). Freeze-safe prep
done now so the June-19 port lands with proof.

- **NEW: `tests/test_evaluate_sells_d1_parity.py`** — behavioral D1 regression
  (3 tests, freeze-safe, adds a test only). Runs an identical fast-falling new
  position through the live `evaluate_sells` path:
  - core test = `xfail(strict=True)`: a 5-day, −12% position under an 18% ATR
    stop is NOT cut today (guard missing) → XFAIL now; XPASSes when the June-19
    `new_position_guard` port clamps to 8%, and strict turns XPASS into a hard
    failure that forces removal of the marker (and deletion of the
    `stop_guard_monitor` sentinel + its test, per their own instructions).
  - 2 un-xfailed controls bracket it: −20% new position IS cut today (stop
    branch wired); seasoned (>21d) −12% is NOT cut in either era (isolates the
    xfail to the guard window). Verified the trip-wire's positive direction by
    simulating an 8% clamp → position cut, reason "STOP LOSS: Down 12.0%
    (ATR-adjusted 8.0%)". Complements the *source-string* sentinel in
    `test_stop_guard_monitor.py` — catches a wrong-direction/threshold port the
    string check would miss.
  - Adjacent suites green: `test_evaluate_sells_d1_parity` +
    `test_stop_guard_monitor` + `test_ai_trader_sync` +
    `test_backtester_trading_parity` → **88 passed, 1 xfailed**.
- **Shadow-arm cross-check (June-18 evidence): UNUSABLE as a clean control.**
  `/api/admin/strategy-ab-eval?strategy=shadow_no_excellence_cap&source=shadow`
  (cutoff 2026-05-07, pre=14) → **decision=insufficient_data**: pre window has
  **0 SELLs**, post has **678** (648 PARTIAL_PROFIT churn, return delta −8.09pp,
  no pre baseline → no Sharpe delta). The shadow arm trades a wildly different
  cadence than live, so it can't corroborate the live `nostate_cs_bear` KEEP.
  Implication for June 18: the live A/B stands on its own (and is itself dragged
  ~1.9pp by the very FSLR D1 stop-out this harness targets) — flagging that the
  KEEP verdict rests on contaminated, un-cross-checked data.
- No frozen file touched (`ai_trader.py` / `canslim_scorer.py` byte-identical).

## 2026-06-12 — UI declutter, deploy/version infra, + ML ceiling investigation (cloud session)

Long session. Three threads, all freeze-safe; live model + trading untouched.

### Cloud (iPad) deploy pipeline + version visibility
- Established the vacation deploy workflow (runbook in `ROAD-HANDOFF.md`):
  cloud session ships freeze-safe change → `main → deploy` PR merge → VPS poller
  rebuilds ~10min. **Auto-deploy freeze-safe changes** is the standing policy.
- Added `backend/build_info.py` deploy stamp surfaced at `/health` `build` field
  + in the UI (Sidebar footer + System Health "App Build" card). Stamp is
  **Central time** (`America/Chicago`, auto CST/CDT) per project convention.
  `.git` is excluded from the Docker context + poller passes no SHA, so the
  committed stamp (bumped each deploy) is the verification mechanism.

### AI Portfolio declutter (frontend)
- Holdings rows + Coiled Spring candidate rows were `flex-wrap` badge-stacks
  (sector full-name, insider, short-interest, growth, dual score / 6-metric
  strip). Collapsed to single-line at-a-glance rows matching Command Center's
  column discipline; demoted signals moved to the tap-through detail modal /
  stock page. `AIPortfolio.jsx`.

### ML investigation — CONCLUSION: low generalizable ceiling, stop ML rankers
Live model is v29 (XGBoost classifier, entry P(win), AUC **0.5676**, f1 0.03,
779 samples). Investigated whether more data/signals/label/exit-pivot help:
- **Feature data is half-empty:** many training rows carry `_nan_safe` defaults
  (legacy backtests predate full `signal_factors` capture). 150/230 backtests
  excluded as ML-contaminated → only 46 clean → data-starved.
- **Label fix (magnitude/regression):** v31 experimental → Spearman 0.059. No help.
- **Clean-data refresh** (4 ML-off backtests 827-830) → v32 Spearman 0.035. No help.
- **Exit/hold pivot** (NEW, default-off `hold_snapshot_capture` lever +
  `backtest_hold_snapshots` table + `POST /api/ml/train-exit`, all freeze-safe):
  predict held position's forward-15d return. Naive in-window Spearman **0.44**
  — but that was **overlap autocorrelation inflation** (consecutive-day snapshots,
  overlapping fwd windows). Corrected (weekly subsample): **−0.03**;
  **out-of-window** (train 2018-22 / test 2022-24): **−0.13**; (hold out 2018-20):
  **−0.06**. Does NOT generalize.
- **Verdict:** neither entry nor exit outcome is predictable from these features
  beyond noise once evaluated honestly (overlap-corrected, out-of-regime). Edge
  is in the deterministic CANSLIM score + rule-based exits, not an ML residual.
  ML's only real value is the existing crude veto. **Do not invest further in ML
  rankers; redirect to the June-19 deterministic exit-parity work (D1, cadence).**
- Infra left in place for future use: the exit-capture lever + train-exit endpoint
  are default-off / measure-only, harmless. v29 never touched; all retrains were
  `auto_activate=false`.

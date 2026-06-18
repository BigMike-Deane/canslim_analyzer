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

## 2026-06-12 — June-19 exit-parity queue fully STAGED (freeze-safe, cloud session)

Same session as above. Staged every June-19 exit-parity item so the post-freeze
landing (after 2026-06-18 17:00 UTC) is mechanical. ai_trader.py/canslim_scorer.py
stayed byte-identical throughout; each frozen edit is a documented one-liner with
its freeze-safe scaffolding + tests already shipped (default-inert).

- **D1 stop-loss clamp** — verified playbook `docs/d1-stoploss-clamp-playbook.md`.
  The guard block is byte-identical to the proven `_check_and_execute_stop_losses_impl`
  block; deps in scope; backtester already has the guard (no change). Regression nets:
  `tests/test_evaluate_sells_d1_parity.py` + the parity harness D1 case (both
  strict-xfail, flip on landing). ⚠️ existing `fix/stoploss-new-position-guard`
  branch is STALE — do NOT merge; apply fresh.
- **Item 2 trailing cadence** — shipped `trading_engine.trailing_stops_allowed_now()`
  + `ai_trader.trailing_cadence` config (default OFF) + 7 tests. June-19: one-line
  gate at `ai_trader.py:1651` + flip lever. Backtester already daily (control arm).
  `docs/trailing-cadence-playbook.md`.
- **Item 3 parity harness** — `tests/test_evaluate_sells_engine_parity.py`: one
  fixture drives BOTH evaluate_sells + backtester._evaluate_sells, asserts same
  decision. 3 scenarios in parity (stop-loss/hold/trailing) + D1 strict-xfail.
  (Gotcha pinned: nostate_optimized overrides default.yaml trailing bands → 18% not 12%.)
- **N1 partial-stop notification** — `send_stop_loss_webhook(is_partial=, shares_kept=)`
  → "PARTIAL … STILL OPEN" instead of looking like a full exit. Default unchanged.
- **D2 score-crash dedup** — `trading_engine.dedup_scores_to_daily()`. LOW priority
  (measured sign-mixed; even helped in 2021-23 bear). Don't ship mid-bear.
- **D3 ATR-stop cache** — `trading_engine.cache_atr_stop` + `atr_stop_fallback`:
  reuse last-known good ATR stop (≤5d) on a failed Yahoo fetch instead of snapping
  to base (random premature stop-outs). Default-inert.
- N1/D2/D3 frozen one-liners all in `docs/june19-minor-items-playbook.md`.
- Also shipped this session (deployed): Central-time `/health` build stamp + UI
  version display; AI Portfolio + Coiled Spring + ML-tag declutter; iPad
  auto-deploy pipeline. All freeze-safe; live model v29 + trading untouched.

## 2026-06-18 — OWNER BACK: freeze lifted early, full exit-parity queue LANDED + deployed

Owner returned from the road. Decided to **lift the freeze ~3.5h early** (the
17:00 UTC cutoff was meant to be ~08:00 CST; owner was awake and ready). Before
touching any frozen file, captured the **authoritative clean-surface C-score-cap
eval** (nostate_cs_bear, cutoff 2026-05-07, pre=14): **decision = KEEP**, return
delta **−1.38pp** (≥ −5pp floor; narrowed from −2.68pp on Jun 11), Sharpe delta
**+0.0591**, post win rate 79.4% over 34 sells. Post-window data even shows the
symptoms we then fixed (15 TRAILING_STOP churn exits; a −22.34% FSLR D1 stop-out).
The 17:00 UTC routine still fires and will email the same KEEP — independent
confirmation. Approach 2 stands; revert branch + `.draft-revert-pr-body.md` now
moot (file gitignored).

**Finding: `main` shipped RED.** The Jun-12 staging added 4 `trading_engine`
helpers but never wired them, so `test_trading_engine_imports_complete` flagged
them as dead code (prod ran them harmlessly as unused fns). Resolved by landing
the whole queue.

**Landed the full June-19 exit-parity queue** (main `51bb80a`→`b66cbe2`, suite
**3212 passed / 0 failed**, deploy stamp `2026-06-18T08:48 CDT`). Every change
restores live↔backtester parity — they make live match the already-validated
champion, not new behavior:
- **D1** (`59f8bb9`): `new_position_guard` 8% clamp ported into `evaluate_sells`
  (the live money leak). Deleted `stop_guard_monitor.py` sentinel + its test +
  scheduler job. D1 regression nets flipped green.
- **Cadence / item 2** (`564c248`): trailing stops gated to daily close-window
  via `trailing_stops_allowed_now`. ENABLED for live in `config/production.yaml`;
  `default.yaml` stays OFF so dev/test trailing-trigger tests stay time-independent.
- **D3** (`79d775b`): ATR-stop last-known cache + `atr_stop_fallback` on Yahoo
  fetch failure (was snapping to base → premature stop-outs).
- **D2** (`570aca6`): score-crash clock dedups scans→trading days. Landed now
  (uptrend, not mid-bear) as parity hygiene.
- **N1** (`9cb7282`): partial trailing stops now say STILL OPEN, not full exit.
- Test adaptations (`b66cbe2`): un-xfailed both D1 parity nets; D2 score test
  re-spaced to daily.

**VPS cleanup (owner back):** retired the deploy poller (removed its cron line,
**kept** the finance-tracker line), cleared the detached HEAD (`git checkout main`),
deleted `.env.bak-jwt-rotation`. Deployed via direct SSH (not the poller).

## 2026-06-18 PM — CTO/CSO read + winner-protection backtest study → KILLED (leave bands)

Pulled current live data (user 1): book = **8 positions, ALL 2x-pyramided**, +28.95% vs SPY +12.4%, **alpha +13.84%, beta 1.22, Sharpe 4.28** (52d since 4-07). Alpha estimate has swung +4%→+7%→+13.8% over 3wks on the same window = small-sample/regime-inflated, don't trust the magnitude. Key structural read: **the dominant risk is giving back WINNERS, not new-buy losses** — $3,411 (11.9% of book) sits between current prices and trailing triggers, concentrated in MU (29% give-back band), DELL (22%), STRL (15%). Corollary: **the just-shipped D1 is a no-op on this book** (everything pyramided → guard skips; only 2 new positions, both green); the CADENCE change is the one acting on it.

**Study (runs 834-845):** does tightening the 50+ trailing band (where MU/DELL/STRL sit) protect give-back? nostate_optimized, "all", $25k, 4 regime windows × {ctrl=25, T1=20, T2=18}. Result = **NO ROBUST EDGE → leave bands as-is:**
- W1 2018-06→20-06: ctrl=T1=T2 IDENTICAL (band never bound).
- W2 2020-06→22-06: ctrl=T1 identical; **T2 HURT** (-4.62 vs -2.08 ret, maxDD 8.3→10.4).
- W3 2022-06→24-06: ctrl=T1=T2 IDENTICAL (59.0% ret).
- W4 2024-01→26-02: T1/T2 marginally better (+0.9pp ret, -0.8pp maxDD, +0.02 sharpe).
- Net: T1 neutral (0/0/0/+slight), T2 net-negative. Classic sign-mixed/one-window pattern that the project's killed experiments (score-floor, rotation, extension-guard) all showed. **Fails the robust-multi-window gate.**
- Pyramid-widening sub-question is BRACKETED by these arms (widening was active; T1=24% / T2=22% effective for 2x-pyramided vs current 29%), so a "disable widening" lever (→25% flat) would land neutral too — NOT worth building.
- **Conclusion:** the wide 50+ band (+ pyramid-widening) is correct; the 11.9% give-back exposure is the deliberate, validated price of the let-winners-run edge that produces the alpha. Do NOT tighten. Side-obs: there IS a weak strategy window (W2 2020-06→22-06: -2% vs SPY +38%) — known choppy-regime lag, not this study's question.
- No code shipped. Sweep runs 834-845 kept in VPS Postgres as the record.

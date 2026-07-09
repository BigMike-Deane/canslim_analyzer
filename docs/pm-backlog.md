# PM Work-Loop Backlog

Owner directive (2026-07-06): standing improvement loop — process, bugs, UI,
big or small — to build the best possible app for stock-market returns.
One item shipped per iteration, tests green, deploy on green.

Constraints (from auto-memory, do not relitigate):
- Don't reopen killed strategy levers (score floor, extension guard, winner
  trailing bands, rotation feature, cadence/D2 tuning).
- Don't re-enable ML without a LIVE A/B; don't trust ML backtests.
- Don't reformulate exits — divergence was parity debt, now fixed.
- Backtests cannot validate scorer changes (frozen-score replay).

## Done
- [x] 2026-07-06 `0c3857c` — Freshness gate on buy candidates.
      evaluate_buys excluded 134 scanner-abandoned rows (some frozen since
      March, several scoring 70-80) from the live buy pool.
      Config: `ai_trader.buy_candidates.max_staleness_hours: 48`.
- [x] 2026-07-06 `bd760e7` — FMP company-screener universe supplement.
      Small-cap sources had silently degraded (IWM rate-limited + its >500
      gate can never pass; Finviz 403) → scanner blind to ~1,367 active
      names (ONON, MNDY, FROG, IOT). Universe 2,081 → 3,474, live-verified.
      Config: `scanner.universe.fmp_screener`. Root-cause findings: most
      "missing large caps" were LEGIT index exits (BK→BNY rename, TEAM off
      N100, HOLX/MASI off S&P500) — iter-1 freshness gate handles their rows.
      ▶ WATCH: scan-cycle freshness for a few days (stocks_stale_1h) — if it
      degrades, tighten screener filters via config (no redeploy).

- [x] 2026-07-06 (iter 3, verification-only) — First expanded scan cycle
      VERDICT: 3471/3471 in 78.7 min (cold cache), 0 FMP 429s over ~9.5k
      calls, API responsive throughout, no interval change needed (90-min
      holds with ~11 min worst-case headroom; warm cycle expected faster).
      Payoff: 44 new names >= 72, 79 >= 67, 113 >= 64 (ELMD 87.6, ITIC 87.2,
      OPY 86.3, TSM 82.6). Leak found: HQH/HQL closed-end funds passed FMP
      isFund=false misclassification, scored ~78. Contained: manual
      DelistedTicker block + stocks rows score-zeroed; nothing was bought.

- [x] 2026-07-06 `e00b66c` (iter 4) — Security-type guard SHIPPED+DEPLOYED.
      Key discoveries: (1) BOTH providers' type flags call CEFs equities
      (Yahoo quoteType=EQUITY for HQH!); working signal = desc mentions
      closed-end AND zero employees. (2) clear_delisted_ticker's
      self-healing DELETED the manual HQH/HQL blocks on next good fetch —
      added PROTECTED_BLOCK_SOURCES + block_ticker_permanently().
      (3) Manual DelistedTicker rows with count<3/no recheck_after never
      actually gated (get_delisted_tickers needs count>=3 AND recheck
      future). Retro-sweep flagged 7 CEFs total (HQH, HQL, ZTR 63.2,
      RMT 59.9, NXP, EIC, FSCO) — all blocked + zeroed. Suite 3403.

- [x] 2026-07-06 `061ea02` (iter 5) — Stale rows hidden from ranked lists.
      /api/stocks + /api/stocks/breaking-out exclude rows >
      api.stale_row_max_hours (72h); include_stale=true opts in; search
      untouched. Live-verified: 68 zombies hidden (FLXS 80, BK 71.6 top).
      DEPLOYED. Note: FLXS is legitimately outside the screener (volume
      < 50k floor) — not a coverage bug.
- [x] 2026-07-06 `8ee3b28` (iter 6) — Dead Yahoo IWM path removed.
      **COMMITTED, NOT YET DEPLOYED** — rides with next deploy to avoid
      another mid-cycle scan restart. Suite 3404.

- [x] 2026-07-06 (iter 7, verification-only) — Exit Plan card VERIFIED
      accurate: imports the trader's own get_trailing_stop_pct /
      apply_pyramid_widening (no drift by construction); live MU output
      exact (trail $881.16 = peak × 0.71; TP $1194.28; SL $634.67).
      Frontend renders ExitPlanChip (card) + ExitPlanSection (modal).
      Sector caps VERIFIED on live data: per-user FS counts 3/4/2 (cap 4),
      user-2 FS value ≈39% (cap 50%). The "7 FS rows" alarm was
      cross-user summing — caps are per-user and correct.

- [x] 2026-07-06 — Warm-cycle VERDICT + OWNER DECISION: 78.9 min for
      3,718/3,724 (same as cold — floor is structural: per-stock live
      price + always-on Yahoo adjusted-EPS at ~1.3s/stock; cache saves
      FMP calls, not wall-clock). 90-min interval holds w/ ~11 min
      headroom. **Owner chose: KEEP FULL COVERAGE, no filter tightening,
      no interval change.** `8ee3b28` (IWM cleanup) deployed in the
      between-cycles window. Guard live catches this cycle: AINV, NCDL,
      OXLC, DXYZ, TYG, MFIC, PSUS (7).

- [x] 2026-07-06 (verify-loop iter 1) — END-TO-END VERIFICATION of the
      day's ships + universe-shrink alarm (`14913e1`, deployed).
      Verified live: freshness gate fires ("excluded 8 stale candidates"),
      buy pool 642 unique (487 CANSLIM + 184 growth), HTTP /api/stocks
      serves 3,812 fresh rows with zombies absent and new names (ELMD,
      TBLA, ITIC, OPY) on top. New: scan start compares universe size to
      SystemSetting baseline; >10% drop → warning log + high-priority
      webhook. Suite 3410.

- [x] 2026-07-08 (iter: candidate-flow payoff + freshness watch) — BOTH
      VERIFIED. Universe payoff CONFIRMED: Jul-6 user-3 initial fill bought
      3 new-universe names (ELMD 87.6, ITIC 87.2, OPY 86.3) out of 8 buys —
      new names outcompeted the incumbent pool. Buys sane (pre-breakout cup,
      0-2% below pivot, ITIC vol 3.3x). Users 1/2 didn't participate: fully
      invested (cash $803/$1,684 < position size), not a bug. Freshness
      HEALTHY: 3,915/4,000 rows <24h, cycle completed on schedule. WATCH
      item closed.

- [x] 2026-07-08 (iter: mass C-score wipe) — **CRITICAL BUG found via the
      payoff check itself: 424 stocks lost C-scores Jul-6→8** (ELMD 87.6→70.6,
      CBL — both LIVE positions understated ~13 pts; phantom score-crash sell
      risk). Root cause chain: universe 2,081→3,724 exceeded
      MAX_CACHED_TICKERS=2500 → set_cached_data evicted data entries but NOT
      freshness stamps → async fetcher saw is_data_fresh=True +
      get_cached_data=None (77,106 warnings/30h in VPS logs), skipped the
      refetch, scored with [] → save_stock_to_db persisted the wipe (blip
      guard can't catch ≤15-pt single-component wipes; C max is 15 < the
      25-pt threshold). DB cache layer (stock_data_cache) still had GOOD
      data throughout — only the memory layer misread. 3-layer fix:
      (1) cap 2500→6000 + eviction now clears freshness stamps too,
      (2) new _apply_cached_financials: fresh-but-missing/empty cache falls
      through to real FMP fetch, (3) save_stock_to_db keeps existing
      earnings/revenue lists when scan produced none. Self-healing on
      deploy: restart rehydrates memory from intact DB cache → next scan
      restores scores. Sync fetch_with_cache was never affected (already
      handled None).

- [x] 2026-07-08 (iter: C-score wipe alarm) — Score-integrity telemetry
      SHIPPED, mirroring the universe-shrink alarm: `_check_component_wipe`
      runs at scan end (Phase 5.5), tracks the share of freshly-scanned
      stocks with c_score=0 cycle-over-cycle (baseline in SystemSetting
      `scan_zero_c_pct`), fires a high-priority webhook when it jumps more
      than `scanner.integrity.c_wipe_alert_pct` (default 5pp). C is the
      canary: it collapses to exactly 0 when earnings data goes missing, and
      a single-component wipe stays under the save-path blip guard — the
      Jul-8 wipe ran silent for 2 days. Skips partial cycles (<100 rows).

- [x] 2026-07-09 (iter: verify C-score recovery) — **VERIFIED HEALTHY.**
      ELMD c=13.0 (87.8) and CBL c=13.0 (77.9) restored; desync warning
      ("is_data_fresh=True but get_cached_data returned None") 0 hits/24h
      (was 77k/30h); new refetch-fallback path also 0 hits (cap 6000 means
      no eviction at all — fallback is an unused safety net); wipe alarm
      silent. Of 875 fresh rows at c=0, 787 have earnings data (legit
      zeros); 88 empty-earnings are preferreds/ADRs/warrants/fresh-IPOs
      with no FMP coverage. RESIDUE (accepted, no action): 38 rows wiped
      during the incident then dropped from the scan universe (e.g. FRAF,
      under the screener volume floor) stay frozen with understated C —
      contained by the 48h buy-freshness gate + 72h stale-row filter, and
      self-repair if they ever re-enter the universe.

- [x] 2026-07-09 (iter: circular-import papercut) — main↔fidelity import
      cycle BROKEN, not papered over: fidelity.py's module-level
      `from backend.main import ...` only survived when backend.main was
      already fully loaded (broke cold under top-level `main`, i.e.
      test_bug_regressions.py standalone). DUPLICATE_TICKERS /
      expand_tickers_with_duplicates / filter_duplicate_stocks moved to new
      leaf module `backend/ticker_utils.py` (imports nothing → can't cycle);
      main.py re-exports for existing importers. Zero behavior change.
      Suite 3426. **COMMITTED, NOT DEPLOYED — pure refactor, rides with
      next deploy (iter-6 precedent: don't restart containers mid-cycle
      for no-behavior changes).**

- [x] 2026-07-09 (iter: SPAC/ETF universe leak) — Security-type guard
      EXTENDED to SPAC shells + ETFs. Found via live log sweep (DATA GAPS
      roster read like a SPAC prospectus): the FMP screener universe admits
      blank-check shells — trust interest gives them small positive EPS
      (huge growth %), the $10 band mimics a flat base, and **CEPV scored
      69.8, three points under the buy threshold**; 50 shells + 5 ETFs
      (NANC, BETZ, DIVI...) live in the stocks table. New
      `non_equity_reason(profile)` classifier in data_fetcher.py: CEF
      conjunction (unchanged reason) → FMP industry 'Shell Companies'
      (structural, reliable — unlike CEF flags) → isEtf/isFund flags (now
      mapped from FMP profile) → case-sensitive \bETF\b name token (legacy-
      source ETFs carry no flags; 'Netflix' must not trip). Permanent block
      is safe for shells: a completed deSPAC lists under a NEW ticker,
      which arrives first-seen and gets scanned normally. Retro-sweep
      blocked + score-zeroed the 55 existing rows (guard only fires on
      first-seen profile fetch).
      **⚠️ HOTFIX `e8fce41` 20 min later: isFund must NOT block — FMP sets
      isFund=true for REIT trust structures** (live: FRT, RLJ, KREF, RPT,
      ILPT, CPT wrongly perma-blocked as the scan ran; raw payloads
      confirmed isFund=true/isEtf=false for all six, SPY/NANC isEtf=true).
      Dropped the isFund check + mapping, deleted the 6 wrong blocks,
      redeployed. **Provider-flag lie ledger: isEtf false-negative for
      CEFs (iter 4), isFund false-positive for REITs (this) — only isEtf
      + industry taxonomy + desc/employees conjunction are trustworthy.**

## Next up (ranked by expected returns impact)
1. **(idea pool)** From live results: entry-rate re-check late July (ML
   demotion), H-fix A/B mid-Aug, exit-reconciliation ~Sept.

## Monitoring clocks (no action until due)
- ~mid-Aug: H-fix strategy-ab-eval re-check (cutoff 2026-07-02).
- ~late Jul: ML-demotion entry-rate re-check (baseline 0.367/day,
  cutoff 2026-07-04; expect entry_rate to RISE if veto was filtering).
- ~Sept: exit-reconciliation poller fires at ≥10 post-fix exits.

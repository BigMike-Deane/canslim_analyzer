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

- [x] 2026-07-09 (iter: universe time-of-day oscillation) — Screener
      volume floor REMOVED; universe now stable. The post-deploy cycle
      scanned 2,879 vs 3,620 (-20%) which led to the discovery: **FMP
      volumeMoreThan filters on TODAY'S cumulative intraday volume**, so
      universe size was a function of cycle start time (live-probed:
      1,786 names at 9:57 ET → 2,292 by 11 ET → ~3,500 evening; vol>10M
      returns 18 rows mid-morning) — a silent daily oscillation since the
      Jul-6 supplement shipped (its "3,474 verified" was an evening
      measurement). averageVolumeMoreThan is silently IGNORED by
      /stable/company-screener (4,101 rows even at a 10M threshold).
      Fix: omit the param unless config sets volume_more_than > 0;
      default.yaml documents the semantics. **DEPLOYED + LIVE-VERIFIED:
      screener fetches 4,103, scan universe 4,317, stable at any start
      time** (~+480 names ≈ +10 min warm cycle — consistent with owner's
      keep-full-coverage call; FRAF's wiped frozen row self-heals by
      re-entering). Note: the intermediate container's logs (incl. the
      shrink alarm that likely fired) were destroyed by docker-compose
      down — alarms that only log+webhook leave no trace across deploys.

- [x] 2026-07-13 (iter: live anomaly sweep) — Two fixes from the 96h
      log sweep.
      (1) **Shadow A/B eval email crash**: the 2026-07-13 Monday 9:00 UTC
      run sent the live verdict fine but crashed for BOTH shadow stacks —
      ab_eval_email's prior-BUY map keyed on t.user_id, which ShadowTrade
      rows don't have (they scope by shadow_strategy_id; the dashboard
      endpoint already used a _scope_id helper, the email path never got
      it). Hoisted _trade_scope_id to module level in routes/admin.py,
      shared by both paths. Tests missed it because the shadow fixture
      seeded only SELLs — BUYs added + explicit regression test.
      (2) **Security-type guard round 3**: the recurring DATA-GAPS 90 in
      the logs decomposed into (a) SPAC shells FMP does NOT label 'Shell
      Companies' — they arrive as 'Financial - Conglomerates' / 'Asset
      Management' (OHAC "Blank check SPAC", GUAC, HCACU "special purpose
      acquisition company"); caught by description-phrase + <50-employee
      conjunction (sponsor-safe; DKNG/LCID descriptions verified clean of
      SPAC history), (b) preferred/baby-bond series with the coupon in the
      listing name ("Saratoga Investment Corp 7.50%", "Ellington Credit
      Co. 8.5% 30-MAR-2031"); caught by a rate-token regex on the name
      ('Capstone Energy+' must not trip), (c) legit data-poor micro caps
      (AIBZ) — correctly left alone. BOT-class (self-described CEF with
      25 staff) deliberately NOT blocked: relaxing the CEF zero-employee
      conjunction risks FPs, and it scores 17. Retro-sweep blocked+zeroed
      existing rows (guard fires first-seen only); follow-ups same session:
      Units?/Warrants?$ name-suffix signal (XCBEU-class, 17 more blocked
      incl. NOVTU tangible-equity units) and block_ticker_permanently now
      zeroes the stocks row (32 zombie scores found). Guard blocks ~328.
      Sweep recipe: MUST pace FMP calls (0.4s) + retry unresolved — an
      unpaced sweep trips the 429 circuit and silently classifies the rest
      of the list against empty profiles.
      (3) **Shadow-trader geometric partial-sell loop** (found while
      verifying fix 1: 4,282 SELLs vs 90 BUYs per stack): shadow positions
      are rebuilt each cycle by FIFO-replaying the ShadowTrade log, and the
      rebuild hard-coded partial_profit_taken=0 — so the 25% tier re-fired
      EVERY cycle on 25% of the remainder (PANW alone: 1,381 geometric
      partial SELLs). Fix: the replay now reconstructs the accumulator from
      partial-SELL reason strings (the "PARTIAL PROFIT {target}%" target
      equals the live accumulator after the sell since take_pct = target −
      already_taken), resetting on full close. ⚠️ Both stacks' trade
      histories before 2026-07-13 are polluted by the loop — shadow A/B
      summary stats (post_sell_count ~4,246) are garbage until stacks are
      reset or windows exclude the polluted span. RESET DONE (user-approved): 8,744 rows wiped, both stacks
      restarted clean at $25k on 2026-07-13.
      Also verified in the same sweep: 0 desync/96h, 4,211/4,382 rows
      fresh <24h, 0 stale high-scores, universe stable.

- [x] 2026-07-13 (iter: REIT-trust supplement) — isFund-REIT decision made
      with live data. Measured: isFund=true at our filters = 3,624 rows of
      which 98.7% are actual funds/CEFs — letting them all in would flood
      the guard, so NOT dropping the isFund param. Instead a second NARROW
      screener pull: isFund=true + sector='Real Estate' (27 rows live,
      ~22 real REITs). Coverage math: 20/21 of those REITs were ALREADY
      scanned via index lists (WSR fresh 66.1) — the real hole is the
      non-index tail (CLDT today; future Russell drops + REIT IPOs), so
      this is a small structural fix, not a payoff play. Contaminant
      handling: mutual-fund share classes (TCREX/VRSGX/JERNX, 5-letter
      X-suffix NASDAQ convention) dropped at source; MITN/RVI fund-trusts
      blocked by the guard at first profile fetch. Hard cap 200 rows: if
      FMP ever silently ignores the sector filter (the volumeMoreThan
      lesson), an uncapped merge would pour 3,624 funds into the universe —
      oversized results are discarded uncached. Config:
      `scanner.universe.fmp_screener.reit_trust_supplement`.

- [x] 2026-07-13 (iter: frontend race guards) — Closed the LAST deferred
      item from the Jul-03 audit: out-of-order response guards on the four
      flagged spots (`ce9d69c`). EdgeScorecard window switch + Screener
      filter/slider + Notifications filter/paging get a monotonic fetchSeq
      ref (a slower earlier response is dropped, its finally doesn't clear
      the newer fetch's spinner); PositionDetailModal gets the effect-scoped
      stale flag (tap A → tap B: A's late resolve must not render under B's
      header; also covers close-then-reopen). Same family as QuickSearch's
      existing abort-token idiom. Frontend build green; backend untouched.

- [x] 2026-07-13 (iter: live-results reads + ML cohort endpoint) — Early
      reads on both monitoring clocks, one methodology fix shipped.
      **Exit-recon early read (owner acct, since=2026-06-18): 6 post-fix
      exits, HEALTHY shape** — TRAILING 62.7d hold / 66.7% WR / +19.3% avg
      (modeled 81d/76%; pre-fix pathology was 9.2d/50%), PARTIAL PROFIT
      +40% @ 84d. Accumulating ~1 exit/4d → the ≥10-exit poller likely
      fires ~Aug-1, earlier than the ~Sept estimate. Today's LQDA stop
      loss: 8.0% trigger, −9.2% realized = 1.2pp cadence slippage
      (documented class, fine).
      **ML entry-rate clock is DOUBLY CONFOUNDED** — all 3 users pinned at
      8/8 max_positions with cash < one position size (entry rate is
      mechanically capped), and the pre-period had the veto active. Direct
      read instead: 9 of 10 post-demotion BUYs carried ml_confidence under
      the old 0.30 threshold (cluster 0.16–0.25) — the veto would have
      blocked user-3's ENTIRE Jul-6 initial fill; demotion un-froze the
      buy pipeline. SHIPPED `/api/admin/ml-demotion-cohort`: splits
      post-demotion BUYs into would_veto vs passed cohorts, outcomes
      blended open (mark-to-market) + closed (realized). Verdict rule:
      demotion vindicated if would_veto performs no worse once closed_n
      accumulates; if it clearly underperforms, model had edge →
      re-graduation via LIVE A/B only.

- [x] 2026-07-13 (iter: scan cadence honesty) — Scanner interval was
      STILL 35 min (persisted May-13, sized for the pre-expansion
      ~28.5-min cycle). Against the post-Jul-6 ~79-min structural cycle,
      the 35-min tick just skip-fired (overlap guard held — no bug) into
      an accidental 105-min cadence with a misleading "35" UI label; the
      Jul-06 "90-min holds w/ ~11-min headroom" verdict was written
      believing the interval WAS 90. Set 90 everywhere: persisted DB row
      (UPDATE applied), boot default 35→90 in main.py/scheduler.py.
      Effective cadence 105→90 min (~+2 cycles/day; Jul-06 verified 0
      FMP 429s at this load). Tail case: a cycle >90 min skips one firing
      → 180-min gap, rare and trivial vs the 48h freshness gate.

- [x] 2026-07-13 (iter: Screener max-price filter) — Owner's documented
      primary use case ("stocks under $25 that fit CANSLIM") had no product
      surface: /api/stocks supported max_price since forever (live-verified
      200) but the Screener UI never exposed it. Added a Max Price preset
      select (Any/$10/$25/$50/$100) wired through api.js and URL
      persistence (shareable/bookmarkable like the other filters).
      Non-bug note: FilterBar's "Market Cap" sort looked like a 422 (not in
      the API's Query enum=) but FastAPI's enum= kwarg is documentation-
      only, not validation — probed live, 200, sorts fine. Don't re-flag.

- [x] 2026-07-15 (iter: universe stickiness) — High-score coverage-loss hole
      CLOSED. The 48h sweep was clean (0 tracebacks, alarms silent, shadow
      stacks 9 BUYs/0 SELLs post-reset, C-zero share stable 21.5%, interval=90
      live, retention pruning healthy) EXCEPT: TBRG 70.6 and HURC 66.5 frozen
      stale — both silently dropped from the scan universe. Two mechanisms:
      (a) HURC market cap $149.76M flapping 0.16% under the screener's $150M
      floor; (b) **FMP marks TBRG isActivelyTrading=false while it trades
      300k shares/day** (stuck since the 2024 CPSI→TruBridge rename —
      THIRD provider-flag lie: isEtf/CEF, isFund/REIT, now this). Fix:
      score-anchored universe hysteresis — get_sticky_high_score_tickers()
      retains recently-scanned names with CANSLIM or growth score >= 65
      (scanned <= 21d), appended BEFORE the delisted/blocked filter so blocks
      still apply; self-limiting (retention keeps the row fresh only while
      the score holds the bar; decayed names age out; dead tickers exit via
      the delisted counter). Logged at scan time ("Universe stickiness
      retained N...") so log sweeps see it working. BK zombie correctly
      excluded (54d stale). Config: `scanner.universe.stickiness`.
      Also confirmed no-action items: ml-demotion-cohort early read
      (would_veto n=9 avg −1.77% vs passed n=1 +1.05%, closed_n=1 — still
      parked to late July), CS-alert "limits reached" log line is normal cap
      behavior, stock_scores retention healthy (30d full / 90d sparse tiers).
      **VERIFIED end-to-end same day**: first post-deploy cycle rescanned
      both (TBRG 70.6→68.6, HURC 66.5→56.1); next boot retained ONLY TBRG —
      HURC's fresh score fell under the 65 bar and it self-released after
      one corrective rescan. The frozen 66.5 was stale-high; stickiness
      repriced it and let go. Don't re-flag HURC leaving the universe.

- [x] 2026-07-15 (iter: alarm persistence) — Fired system alarms now survive
      redeploys. The Jul-9 lesson ("the intermediate container's logs — incl.
      the shrink alarm that likely fired — were destroyed by docker-compose
      down") closed: new `_fire_system_alarm(title, msg, priority, tags)` in
      scheduler.py delivers webhook + a persisted owner Notification row
      (user_id=1, kind='system_alarm' — surfaces in the bell/Notifications
      page). Wired into _check_universe_shrink, _check_component_wipe, and
      _record_failure (consecutive scan/trade-cycle failure alerts). Anomaly
      sweeps can now query `notifications WHERE kind='system_alarm'` for
      alarm history regardless of container lifecycle.

## Next up (ranked by expected returns impact)
1. **(idea pool)** From live results: entry-rate re-check late July (ML
   demotion), H-fix A/B mid-Aug, exit-reconciliation ~Sept.

## Monitoring clocks (no action until due)
- ~mid-Aug: H-fix strategy-ab-eval re-check (cutoff 2026-07-02).
- ~late Jul: ML-demotion re-check — USE /api/admin/ml-demotion-cohort
  (entry-rate metric is confounded: portfolios pinned at max_positions +
  pre-period had veto active). Verdict: would_veto cohort no worse than
  passed = demotion vindicated.
- ~Sept: exit-reconciliation poller fires at ≥10 post-fix exits.

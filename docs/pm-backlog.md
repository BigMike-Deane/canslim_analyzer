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

- [x] 2026-07-15 (loop 2 iter: position-state invariant audit) —
      VERIFICATION-ONLY, all clean. The shadow Zeno bug (Jul-13) was a
      position-state-drift class; the live portfolio got the same audit:
      all 24 positions' shares reconcile against FIFO trade-net to 4
      decimals; stored gain% matches price math; every partial_profit_taken
      accumulator consistent with trade-reason history under the correct
      semantics (PARTIAL PROFIT sets-to-target: CARE 50, DELL 75, u2-MU
      trailing-50→profit-75; PARTIAL TRAILING adds: u1-MU 50+50+50=150);
      MU-u1 sitting 26.6% off peak is NOT a missed 25% trail — pyramid
      widening ×2 puts the stop at peak×0.71=$881.16, matching the Jun-04
      verified exit plan exactly. All 3 users at 8/8 positions.

- [x] 2026-07-15 (loop 2 iter: no-financials negative cache) — The ~40-name
      data-poor roster (BOT/BXDC/CEPL class) reburned 2 FMP income-statement
      calls EVERY cycle (~1,300 calls/day) on an answer that never changes.
      fetch_fmp_financials_async now negative-caches tickers where BOTH
      endpoints answered authoritatively empty (HTTP 200 + `[]`) for 3 days.
      Safety: fetch_json_async returns None for ALL transient failures
      (circuit open / 429 / 5xx / timeout) and None is never cached — a
      tripped circuit breaker cannot poison the cache (empty-cache-forever
      lesson respected). Memory-only (restart = one refetch cycle); tracker
      still records gaps so the DATA GAPS report stays accurate; TTL expiry
      re-probes so maturing IPOs get picked up.

- [x] 2026-07-15 (iter: performance-chart "All" range, owner report) —
      Owner: "graph shows a different starting date/position — should start
      at 25k". Root cause: AIPortfolio.jsx hardcoded a 90-day history fetch,
      and the chart's range buttons (incl. "All") only filter WITHIN the
      fetched window — so "All" silently became "last 90 days" once the
      portfolio outlived it (owner inception 2026-03-09 @ exactly $25,000;
      chart start had slid to ~Apr-15 @ ~$24.8k). Data was always intact
      (6,930 snapshots). Fix: endpoint days cap 365→3650 + new
      `resolution=auto` (intraday for trailing 7d, last-snapshot-per-day
      older — a year stays ~a few hundred rows, not ~16/day); page fetches
      (3650, 'auto'). The deliberate leading-flat-day trim is kept: "All"
      starts at the first BUY (≈$25k) with the Start reference line at
      exactly 25k. Query(pattern=) used for validation (enum= is doc-only).

- [x] 2026-07-20 (iter: gap-week sweep + deploy backlog cleared) — Tailscale
      restored; deployed the two pending commits (`2b35e2a` perf-chart All
      + `84f26ff` negative cache). Perf chart LIVE-VERIFIED: 253 downsampled
      points, first point 2026-03-09 @ exactly $25,000; portfolio $30,608
      (+22.4%) vs SPY +10.0%. 5-day-gap sweep ALL CLEAN: 0 persisted system
      alarms (first real use of Jul-15 alarm persistence), shadow stacks
      25 BUY/3 SELL each (no Zeno), C-zero 21.4% vs 21.5% baseline, trades
      sane. GS trailing at -4.9% is NOT a bug — undocumented-in-CLAUDE.md
      gain_5_to_10 tier trails at 4%. MU exits -31.7% vs 29% widened trail
      = 2.7pp cadence slippage (documented class). SHIPPED `c3d791d`:
      ml-demotion-cohort age stats (age_days/avg_age_days) — Jul-20 read
      (would_veto -3.74% vs passed -0.41%) is VINTAGE-CONFOUNDED (14d vs 4d
      cohorts); age stats make the late-Jul verdict read honest.
      **COMMITTED, NOT DEPLOYED — rides next deploy/between-cycles window.**

- [x] 2026-07-20 (iter 2-4: deploys + owner reports) — `c3d791d` cohort age
      stats DEPLOYED + live-verified (avg_age_days 13.2 vs 5.2 — vintage gap
      quantified). Negative cache END-TO-END VERIFIED: cycle 1 populated 49
      tickers, cycle 2 re-fetched ZERO (count held at 49, DATA GAPS report
      intact, cycle 45 min vs 55). OWNER REPORTS both fixed (`4210ab8`,
      deployed): (1) CS alerts "Conf 8900%" — frontend scaled 0-100
      confidence as 0-1 (×100); color tiers had the same wrong-scale bug
      (everything rendered "strong"); now Math.round(conf)% with 70/50
      tiers, modal shows N/100. (2) Live vs Backtest Exits blended
      pre-parity-fix exits into its verdict — now defaults
      since=2026-06-18 (Post-fix/All-time toggle, ref-mirrored for the
      polling closure + out-of-order guard), footer explains young-
      portfolio structural gaps. Post-fix window: 9 live exits, remaining
      flags are presence/hold-days youth artifacts (footer contextualizes;
      real verdict = recon poller ~Aug-1). Also `b6c6cb2` docs: champion
      config lists the 5-10%: 4% trail tier (GS false-alarm prevention).

- [x] 2026-07-21 (impact loop: max_positions capacity sweep — VERDICT: KEEP 8)
      Motivation: 36 fresh >=72 unbought, all users pinned 8/8. Full grid
      (nostate_optimized, all-universe, $25k, W1-W4), baselines 897-908 +
      variants 909-916 via profile_overrides:
      | W       | mp8          | mp10         | mp12         |
      | W1 chop | +7.4 /11.1  | +7.4 /11.1  | +7.4 /11.0  |
      | W2 chop | -13.7/15.5  | -13.4/15.1  | -14.2/15.9  |
      | W3 trend| +42.5/ 8.0  | +36.2/11.4  | +37.7/ 9.1  |
      | W4 trend| +110.0/11.1 | +96.6/13.1  | +93.8/12.6  |
      GATE FAILS BOTH STRONG WINDOWS (limit 3pp): W3 -6.3/-4.8pp, W4
      -13.4/-16.2pp, DD worse everywhere it binds; chop untouched (cap never
      binds there — peak 5 positions). CONCENTRATION IS THE ALPHA — wider
      books add churn (W4: 217/244 trades vs 171, WR down) and dilute
      winners. max_positions stays 8; the unbought high-scorers are the
      price of concentration, not free money. DON'T RE-SWEEP position count.
      BUG FOUND EN ROUTE, FIXED+DEPLOYED `7dfed59`: BacktestCreate's
      explicit max_positions/min_score_to_buy/stop_loss_pct were silently
      ignored for profile-defined strategies (backtester resolves
      profile-first; creation pads row with defaults). Sweep 1 (897-908) ran
      12x mp8. Creation now folds explicit scalars into profile_overrides
      (setdefault; power path wins; untouched requests stay NULL).
      Op-note: cold pass ~2h/window after universe expansion (one-time
      historical fetch for ~2k new tickers); warm ~25 min.

- [x] 2026-07-21 (owner pick: DR + web push audit — "exists but might have
      broken") — `0c3d5a1` DEPLOYED. Backups: daily 2AM pg_dump job HEALTHY
      (7d+4w rotation, never missed) but never restore-tested and all
      on-box. Shipped: (1) verify_latest_backup() + Sun 3:30 UTC job —
      scratch-DB restore + row-count floors, system alarm on failure;
      manual proof: Jul-21 dump → 4,401 stocks / 2.43M scores clean.
      (2) scripts/pull_backup.sh off-box leg (Tailscale→Windows, keep 4);
      first copy pulled to C:\Users\bayer\canslim_backups.
      Web push: whole stack healthy (VAPID in .env+container, endpoints
      200, sw.js/manifest fine, pywebpush installed) — push_subscriptions
      EMPTY because iOS only exposes PushManager inside an installed PWA;
      Settings dead-ended with "not supported". Shipped: iOS
      add-to-home-screen onboarding steps + real 180x180
      apple-touch-icon.png (iOS ignores SVG). ntfy leg confirmed live
      (BUY: FTI push delivered same day). ▶ OWNER ACTION: install PWA
      from Safari (Share → Add to Home Screen) then Enable Push in
      Settings; run scripts/pull_backup.sh weekly for the off-box leg.

- [x] 2026-07-21 (owner ask: "something original") — **Improving Radar
      SHIPPED `96b83c1` + DEPLOYED.** First innovation mined from our own
      2.4M-row point-in-time StockScore history. Event study (113k events,
      level-controlled, May-Jul, no lookahead by construction): rising
      scores at LOW levels lead static peers ~+1.1pp/14d (sub-55: +2.6% vs
      +1.5% avg); **fast risers at 75+ went NEGATIVE (−1.1%)** — rapid
      score rise at the top = the price run-up feeding momentum components
      (extension, not emergence). Product: /api/stocks/improving-radar +
      CommandCenter panel — radar list (40-65 band, +8/14d, sparklines) +
      75+ chase-risk caution block. Live first read: AMC +23.1, RPC +20,
      HOG +19.2, MMM +18.2 on radar; DSGR/UTZ +29s flagged as chase risk.
      Caveats recorded: single 2.5-mo trending regime, overlapping event
      windows, not SPY-adjusted. Config api.improving_radar.*.
      ▶ WATCH (~Sept): re-run the event study with 2 more months of
      history incl. the radar's own picks — does the sub-65 velocity edge
      hold out-of-sample?

- [x] 2026-07-22 (owner directive: full signal-vs-noise audit, 4 parallel
      agents) — VERDICTS (single-regime May-Jul caveat applies):
      SIGNAL: A (only monotone-positive letter, ~0.6-0.75pp tercile gap
      both bands); C hump-shaped (mid best, top stalls = EPS-pop class);
      score-velocity at low levels (radar, validated Jul-21).
      INVERTED at 14d: L/S/N — high price-momentum components within a
      score band mean-revert (top-tercile L NEGATIVE both bands). NOT a
      defect: entry discipline (pre-breakout ≤ pivot) structurally avoids
      buying extended moments — components + rules are coherent.
      NOISE: I (5-bucket coarse, non-monotone, 40% tied at 4.0).
      REGIME DIAL: M (per-day constant, zero cross-sectional info; makes
      absolute score thresholds implicitly market-adaptive — by design).
      COHERENCE: no letter-pair |r|>0.38 (no redundancy); breakout vs
      coiled-spring surfaces complementary (2/45 overlap in 7d).
      HYGIENE: app already pruned AD-bonus/insider/sector-rotation/ML/
      market-state from trading (audit-flagged inert: get_audit_bonus
      dead code). NO scoring changes without live A/B (standing rule) —
      findings are observability. ▶ Re-test L/S/N inversion on a weak
      tape before ever acting on it.
      A/B TAB BUG FIXED+DEPLOYED `e3dcb50`: live-source cohort resolved
      from CURRENT config → strategy switches retroactively reshuffled
      history (June baseline vanished; UI compared vs phantom 0.0%).
      Fix: strategy stamped on trades at execution + stamp-first query
      (legacy NULL falls back) + empty-window return=None + explicit
      membership-drift warning.

- [x] 2026-07-22 (owner directive: "prove we beat SPY before real money")
      — REGIME ATTRIBUTION (73 live trading days): ALL outperformance from
      strong-trend days (+20.4pp vs SPY, 52d); chop days (SPY 0-1.5% over
      50MA) cost −7.9pp in 20d; live confirmation of the W1/W2 backtest
      signature. Owner's caution quantified. Two ships (`93ce40d`+
      `70f5e82`, DEPLOYED):
      (1) chop_damper lever (trading_engine.chop_damper_multiplier —
      half-size buys in the 0-1.5%-over-MA band; profile-gated DEFAULT
      OFF; ai_trader + backtester mirrored) + hidden profile
      nostate_chop_damper + **ShadowStrategy 'shadow_chop_damper'
      registered on live (id=3)** — the honest live-A/B evaluation June's
      backtest levers never got. Champion untouched.
      (2) Verdict clock promoted to Edge card headline: "≈N trading days
      (~X mo) until edge vs SPY statistically provable" (existing power
      analysis, was a buried footnote); green banner at significance.
      ⚠️ Ops lesson: scripted whole-file rewrites on /mnt/c clobber CRLF
      endings (8.7k-line phantom diff) — patched byte-exactly in
      `8308618`; use binary-mode edits for import insertions.
      ▶ shadow_chop_damper verdict via ABEval (shadow source) once weeks
      of mixed-regime data accumulate — DON'T promote to a real user
      before then.

- [x] 2026-07-22 (owner: "proceed with all, audit, then patience") —
      three ships + audit-caught fixes, ALL DEPLOYED:
      (1) REGIME RESEARCH: tested 6 chop classifiers on 72 live days —
      winner is the app's own weighted_signal: ws<2.0 days bleed
      −73 bps/d (11d) vs +31 on ws=2.0 days; ONLY classifier robust
      across sample halves (spread 131/102). RECORDED as chop-damper v2
      cut — NOT acted on (72d sample; patience directive).
      (2) STOP-SLIPPAGE `2b88bfc`: check_and_execute_stop_losses was
      built for intraday use but NEVER SCHEDULED (page-load only) —
      May-era slippage up to 5.6pp (HRTG). Now every 15 min in market
      hours, all users, per-user error isolation.
      (3) AUDIT CATCH `e8b380b`: shadow_chop_damper's first cycle bought
      76 names/$127.7k on $25k — evaluate_buys returns a ranked DECISION
      list, live executes top-N, shadow harness persisted ALL (invisible
      while stacks held full books; fresh stack exposed it). Fixed:
      slot+cash guard in translation loop (partial last fill). ALSO
      direct-DB shadow registration gets soft-archived by the YAML sync —
      registered properly in shadow_strategy_profiles; 76 bogus rows
      wiped; stack restarted clean at $25k. 3 shadows active.
      **▶ NOW IN PATIENCE MODE (owner directive): no new experiments
      unless evidence demands. Watch: exit poller (7/10), chop shadow,
      trend-day clock (~136 trend days), Gate-1 checkboxes.**

- [x] 2026-07-22 (owner: UI audit/overhaul, full approval) — `18a6520` +
      `38793da` DEPLOYED. Retired signal-dead surfaces (evidence = same-day
      signal audit): InsiderSentiment, SectorRotation, Breadth (A/D + NH/NL
      salvaged onto CommandCenter market card), DecisionLog (duplicate
      endpoint), Backtest ML-Matrix tab, CommandCenter MarketStateBadge.
      Consolidated: TradeJournal → Analytics 'Journal' tab (embedded prop;
      /trade-journal redirects); /portfolio + /portfolio-summary →
      /ai-portfolio; Admin ML section gets unmissable DEMOTED banner
      (SystemHealth = the single ML status readout). Follow-through:
      scanner.fetch_insider_short gated OFF (per-cycle fetch fed only the
      retired page). Sidebar 18→13, bundle 516→490kB, −1,083 LOC.
      KEPT deliberately: BearBase + CorrelationMatrix (both still drive
      trading), Breakouts, CS History, ABEval.

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

# Road Handoff — 2026-06-11 → ~2026-06-18

Context document for **cloud Claude Code sessions** (claude.ai/code / iPad
app) while the owner travels. Local-machine memory does not transfer; this
doc + `CLAUDE.md` + `docs/parity-audit-evaluate-sells.md` are the working
context. Written by the local session on 2026-06-11.

## Hard rules (read first)

1. **FREEZE until 2026-06-18 17:00 UTC:** `backend/ai_trader.py` and
   `canslim_scorer.py` must stay byte-identical. The June-18 C-score-cap
   eval (below) depends on it. The deploy poller enforces this too, but do
   not rely on the guard — just don't touch those files before the freeze
   lifts.
2. **Trader/backtester sync:** any post-freeze change to `ai_trader.py`
   trading logic MUST be mirrored in `backend/backtester.py` (see CLAUDE.md).
3. **Tests before any merge:** `python3 -m pytest tests/ -q` — expect
   **3213 passed / 16 skipped / 0 failed** as of HEAD `296491a`.
4. **Never force-push.** Normal pushes only.

## How deploys work this week

Cloud sessions cannot SSH to the VPS. A poller on the VPS watches the
**`deploy` branch** every 10 minutes and rebuilds when it moves:

- Work on feature branches → PR → owner reviews in the app → merge to
  `main`, then fast-forward `deploy` to `main` to ship it.
- Pre-June-19, the poller **refuses** any deploy that changes
  `ai_trader.py`/`canslim_scorer.py` (freeze guard).
- Kill switch (owner, from any session with VPS access):
  `touch /opt/canslim_analyzer/.deploy-poller-disabled`.
- Deployed state on departure: `main` = `deploy` = `296491a`, all 4
  containers healthy, suite green.

### Deploying from a cloud session (iPad) — vacation workflow (set 2026-06-12)

The `deploy`-branch poller **is** the cloud deploy path — no SSH needed. The
owner deploys from the iPad exactly the way the PC terminal did today.

- **Standing policy: auto-deploy freeze-safe changes.** Once the owner
  requests/approves a change in a session, ship it immediately — do **not**
  wait for a separate "deploy" instruction. Procedure:
  1. Implement the change. **Bump the deploy stamp:** set `BUILD_VERSION` in
     `backend/build_info.py` to the current UTC minute (e.g. `2026-06-12T14:05Z`)
     in the same change — this is how the deploy is verified in step 3. Push.
     (In this environment the git proxy routes the designated-branch push onto a
     branch on GitHub but does NOT reliably advance `main`; use a GitHub PR to
     land on `main`/`deploy`, not raw `git push`.)
  2. Ensure a `main → deploy` PR exists (create one if not) and **merge it** via
     the GitHub API/app. Merging moves `deploy`; the VPS poller rebuilds within
     ~10 min.
  3. Verify: poll `GET https://canslim.duckdns.org/health` until `build` equals
     the stamp you set in step 1 (that confirms the rebuild picked up your
     commit — not just that the site is up). For UI work also eyeball the app.
- **Freeze guard is the safety net.** The poller refuses any deploy touching
  `ai_trader.py`/`canslim_scorer.py` before June 19. A change touching those is
  NOT freeze-safe — never auto-deploy it; flag it to the owner instead.
- **Freeze-safe = deployable now:** `frontend/`, backend endpoints outside
  ai_trader/scorer, docs, tests, non-trading config (mirror of "Good
  road-session tasks" below).

## Live data access — NO tokens in this repo

> **History note:** an earlier revision of this file embedded an owner JWT.
> That key was **rotated on 2026-06-11** (`JWT_SECRET_KEY` changed on the
> VPS) — any token found in git history is dead. Do not add credentials to
> this repo.

Live API access comes from the **cloud environment's settings**, never
from this repo:
- If the env var `CANSLIM_API_TOKEN` is present in your session, use it:
  `curl -H "Authorization: Bearer $CANSLIM_API_TOKEN" https://canslim.duckdns.org/api/...`
  It is an owner/admin JWT the owner placed in the claude.ai environment
  config (expires ~2026-06-21). This unlocks the full workflow: portfolio
  reads, `/api/admin/strategy-ab-eval`, and **backtest sweeps** via
  `POST /api/backtests` (ctrl/treatment pairs with `profile_overrides`,
  poll `GET /api/backtests` for completion — see the parity-lever sweeps
  pattern, `tests/test_parity_fidelity_levers.py` + audit doc D2/D4/D5).
- If the env var is missing or the network blocks the domain, fall back
  to code-only work and tell the owner which toggle is missing
  (environment env var / network allowlist for `canslim.duckdns.org`).
- **Owner:** the web app at `https://canslim.duckdns.org` works from any
  device (mobile-first) — portfolio, Edge Scorecard, charts.
- The June-18 eval routine carries its own short-lived token (managed in
  the routine config on claude.ai, not in git) and emails the owner its
  readout. The June-19 kickoff routine is repo-only and needs no token.

## Session log — REQUIRED for continuity

The owner's local Claude memory does not exist in cloud sessions, and
cloud sessions don't share context with each other. The replacement:
**append a dated entry to `docs/road-log-2026-06.md`** (create it if
missing) at the end of any session that ships, decides, or discovers
something — 3-8 lines: what was done, commits/PRs, verdicts, open ends.
Read that file at session start, right after this one. When the owner
returns, the local session syncs it back into long-term memory.

## State of play (2026-06-11)

### June-18 C-score-cap eval ("Approach 2") — the open strategic thread
- Live A/B of commit `ec73f83` (C-score excellence-tier cap), cutoff
  2026-05-07. Cloud routine `trig_011K2Vzgq8KDZ4aJTk2bJHAM` fires
  **2026-06-18 17:00 UTC** and renders keep/revert via
  `/api/admin/strategy-ab-eval` (criteria: return delta ≥ −5pp AND Sharpe
  delta ≥ 0, min 5 post sells).
- **June-11 dry run rendered KEEP**: return −2.57pp (within floor), Sharpe
  +0.087, post win rate 76.7% vs 56.5%. ~1.9pp of the return drag is one
  FSLR −22.3% stop-out caused by the D1 bug (below), not the cap.
- Pre-staged revert: branch `prepare-approach-2-revert` +
  `.draft-revert-pr-body.md` (local only). If verdict = keep: close/ignore.
  If revert: rebase that branch, test, merge, ship via `deploy`.

### June-19 exit-parity work — queued, quantified
Routine `trig_01U9dF3HiUnTJB927uMD4A7F` fires **2026-06-19 14:00 UTC** to
kick this off. Full inventory in `docs/parity-audit-evaluate-sells.md`.
Priority order (all measured):
1. **D1** — port the `new_position_guard` 8% clamp into
   `ai_trader.evaluate_sells()` (~1623). Missing clamp already cost a live
   −$967 FSLR stop-out at −19.7% on June 9. Mirror of backtester
   2602-2615 and `_check_and_execute_stop_losses_impl` 1304-1314. Delete
   the `stop_guard_monitor.py` sentinel in the same commit.
2. **Trailing-stop cadence redesign** — evaluate live trailing stops once
   daily near the close (hard stops stay intraday). Measured drag of
   intraday cadence: −1..−17pp per 2yr window, 8/8 arms; live-validated
   (trailing exits 9.2d hold / 50% WR / −0.1% avg vs 81d / 76% / +3.6%).
3. **Parity harness** — shared-fixture test asserting `evaluate_sells` ≡
   backtester `_evaluate_sells` decisions.
4. **D2** — per-day dedup of live `check_score_stability` (parity hygiene
   only; June-11 sweep runs 817-826 showed sign-mixed, low impact).
5. **D3** — ATR caching/last-known fallback. 6. **N1** — partial-stop
   notification wording.

### Recent ships (all deployed, `296491a`)
- D2 scan-clock lever (`5fbab9b`) + verdict (`3678c0d`) — see audit doc.
- Intraday SPY benchmark (`296491a`): new `spy_intraday_prices` table,
  ticks every market refresh (5-min throttle), `/ai-portfolio/history`
  prefers tick at-or-before each snapshot. Movement accrues from June 11;
  older days are flat-stepped forever (history wasn't stored).

### Known live issues (do not "fix" before June 18)
- **D1 stop-guard bug**: user 3 holds FSLR ~−19% and AMD ~−11% with
  unclamped stops. A daily breach-alert notification covers it. Fix is
  item 1 above, frozen until June 18.

## Good road-session tasks (freeze-safe)
- Frontend/UX work (anything in `frontend/`), backend endpoints outside
  ai_trader/scorer, test coverage, `backend/backtester.py` research levers
  (default-off, follow `tests/test_parity_fidelity_levers.py` pattern).
- Deferred PM ideas: Morning Briefing email leading with Edge Scorecard
  numbers (`backend/scheduler.py`); AI-vs-You cross-user leaderboard.
- Shadow-arm health check: verify ShadowTrade rows are still accruing for
  `shadow_no_excellence_cap` (db id 2) and pull
  `/api/admin/strategy-ab-eval?source=shadow` as concurrent evidence for
  the June-18 verdict.

## Cleanup when the owner returns
- Decide whether to keep the deploy poller (remove the cron line if not).
- `git checkout main` in `/opt/canslim_analyzer` if the poller left a
  detached HEAD.
- Delete `/opt/canslim_analyzer/.env.bak-jwt-rotation` on the VPS (backup
  of the pre-rotation env, made 2026-06-11).
- Delete or archive this file.

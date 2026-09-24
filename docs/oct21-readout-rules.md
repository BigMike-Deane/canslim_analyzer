# Oct-21 Readout — Decision Rules (pre-registered)

**Status: BINDING — approved by the owner 2026-09-23.** Any change needs a new dated amendment below.
Written before any arm's gate has filled, so the read cannot be fitted to
the result. Sections marked **OWNER DECISION** need a pick; the rest follows
from rules already pre-registered on each arm's YAML profile.

Sign-off: `[x] approved by owner on 2026-09-23 (edits: none)`

## Why this file exists

Go-live criterion 1 (blended edge vs SPY) cannot pass by waiting — see
`real-money-gates.md` § "not a waiting game". As served 2026-09-23: trend
+18.4 bps/day, chop −26.8, breakeven trend share 59.3% vs the market's recent
46.7%. At today's trend mean, parity needs the chop mean at about **−16**
bps/day (−22 if the trend mean recovers to its Sep-12 +25.3): a **20–40% cut in
chop bleed**. Four chop arms, one small-cap arm and three exit/sizing arms read
out around Oct-21. Every arm's own gate says "beat the comparator", but none
says *by how much*, and none says what the program does if nothing wins.
This file fills both gaps.

## When

- Each arm is read on the first trading day its **own pre-registered
  accrual gate** is met (chop days now count only finished NYSE sessions,
  `81bc23c`). No early calls.
- **The program decision** (sections C–E) waits until all four chop arms have
  been read, or **2026-10-30**, whichever comes first. If some chop gates are
  still short on Oct-30, they are read as-is and marked *under-accrued*.
- Accrual pace: 9 real chop days in 24 sessions (~0.375/day) → 15 chop days
  around Oct-15 for the Aug-19 arms and ~Oct-20 for the Aug-25 arms.

## A. What "beat" means (applies to every arm)

The unit is **σ = `program_clocks.vintage_spread.stdev_pp`**, read on the
readout day. It is the measured path noise between books running the same
strategy from the same start (4.4pp on 2026-09-23). It is measured on a longer
horizon than any arm's window, so it overstates arm-window noise, which is the
conservative direction.

Δ = arm alpha − comparator alpha, over the common window (the later of the
two activations → readout), from closing marks.

| Result | Rule |
|---|---|
| **Clear win** | Δ ≥ +1σ, the lever's sufficiency gate met, and the mechanism check passes |
| **No detectable effect** | −1σ < Δ < +1σ |
| **Clear loss** | Δ ≤ −1σ |

One-sided 1σ ≈ 84%, the same convention as the go-live gate (85% one-sided).

**Comparator** = the arm's YAML `comparator:` if set (stop_band →
vintage_sep16, small_cap_gate → vintage_sep23), otherwise shadow_baseline.
Arms without a same-day control (chop_entry_bar and chop_trim, started Aug-25;
ml_veto_off, started Aug-20) carry launch mismatch. For those, a clear win must
ALSO hold against shadow_vintage_sep02 over its window, as a second, differently
dated control.

**Mechanism check.** The arm must win *where its lever acts*. For chop arms:
the arm's mean excess on chop days beats the comparator's on chop days, and its
trend-day mean is no worse than −1σ-equivalent. A chop arm that "wins" on
trend days is a vintage accident, not a fix.

### OWNER DECISION A1 — multiple-comparisons bar for the chop family
**DECIDED 2026-09-23: A1-a (+1.5σ plus the mechanism check).**
The four chop arms test one hypothesis four ways. With four null arms,
P(at least one clears +1σ by luck) ≈ **50%**.
- **(recommended) A1-a:** the best chop arm must clear **+1.5σ** plus the
  mechanism check. P(any null clears) ≈ 24%, and the mechanism check cuts it
  further.
- A1-b: +1σ plus mechanism (accepts a ~50% family false-win rate before the
  mechanism check).
- A1-c: +2σ (≈ 9%; will likely call real ~25% bleed cuts "no effect").

## B. Per-arm outcomes (from their own pre-registrations)

| Arm | Clear win → | Clear loss → | No effect → |
|---|---|---|---|
| chop_spy | promote candidate (must also beat chop_damper, per its pre-reg) | archive | archive at Oct-30 |
| chop_damper | promote candidate | archive | archive at Oct-30 |
| chop_entry_bar | promote candidate | archive | archive at Oct-30 |
| chop_trim | promote candidate | archive | archive at Oct-30 |
| small_cap_gate | promote candidate | archive | keep running (only ~4 weeks old at readout) |
| stop_band | promote candidate | archive | keep until Nov/Dec fleet review (its pre-reg) |
| ml_veto_off | RETIRE the veto (its pre-reg) | BLESS the veto | keep veto, archive arm |
| wide_trail / cap50 / sector_relief | promote candidate | archive | archive at Oct-30 |

**Dormant-lever rule** (same as the Sep-9 fleet review): an arm whose lever
fired fewer times than its dormant gate by Oct-30 (cap50: cap-tier fires;
chop_trim: trims; sector_relief: exempt pyramids) is archived as *structurally
dormant*, not *lost*, and its YAML comment records why.

## C. Promotion discipline

- **One lever per readout.** If several arms win, promote the one with the
  largest Δ in σ units. The runner-up is re-launched as a new arm **on top of
  the new champion** with a same-day vintage control. Stacking two untested-
  together levers is how we lose the ability to attribute anything.
- Owner policy (Aug-21): on promotion, **all** user portfolios and the default
  flip together.
- A new vintage copy launches the same day as the promotion, so the next
  readout has a same-day control from day one.

## D. What the program does with the result

Project each winner's chop-day mean into criterion 1 (breakeven share vs the
market's recent trend share):

- **D1 — a chop arm wins AND its projected breakeven ≤ the recent market
  trend share.** Promote it. Criterion 1 is expected to pass, so go to OWNER
  DECISION D-clock.
- **D2 — an arm wins but the projection still falls short.** Promote it (it is
  better), and criterion 1 stays blocked. Continue to E.
- **D3 — nothing wins.** Continue to E.

### OWNER DECISION D-clock — what history does go-live read after a promotion?
**DECIDED 2026-09-23: keep the full history.** The post-promotion window is
reported beside it for visibility but does not gate.
- **(recommended) D-clock-a:** criterion 1 keeps the full history, and the
  gate also reports the post-promotion window separately. Go-live needs both
  the full-history pass AND a post-promotion blended edge ≥ 0 over ≥ 40
  sessions including ≥ 12 chop days.
- D-clock-b: reset all five criteria to post-promotion data only (cleanest,
  but it costs months: 50 closed trades ≈ 3–4 months).

## E. If criterion 1 is still blocked after the readout

### OWNER DECISION E — pick the next move now, not after the result
**DECIDED 2026-09-23 (rule set BEFORE the check ran):** E-a goes ahead only if
a retro check on live u1/u2 history shows that the buys/pyramids an industry
≤30% cap would have blocked were, in aggregate, **net losers**. If they were
net winners, it is the Aug-12 lesson again ("adds to leaders earn the money")
and E-a is dropped in favour of E-c.

**Retro check RESULT (2026-09-23, u1+u2, 157 buys/pyramids since March;
FIFO lot outcomes, open lots marked to the latest close):** at ≤30% the cap
blocks **6 adds (4%)**: **−$551, mean −4.2%, 1 of 6 winners**, against +5.7% mean
and a 51% win rate for the 151 it allows. At 25% it blocks 7 (−$645) and at 40% it
blocks 4 (−$231). Blocked pyramids: 2, both DSX, both losers. **Per the rule
above, E-a goes ahead.** Caveats: n=6, all Aug-6 or later, 4 of 6 still open,
first-order only (it ignores what the freed cash would have bought), u1 only.
That is why this is an arm to test and not a switch to flip. Unlike the Aug-12
regime gate (which in W4 blocked 20 trades into leaders), this cap binds
rarely and, in this sample, has blocked no winning pyramid. Script:
session scratchpad `indcap.py`; re-run it at the readout.
- **(recommended) E-a: concentration arm.** Sep-21 (−356 bps, the worst trend
  day) was 50% of the book in two Marine Shipping names, both pyramided twice
  at +3–4% within days, plus 11% in oil E&P. The sector cap is **50% of
  equity per sector by design** (Feb-7 "concentrated portfolio", O'Neil:
  follow the best sectors), so this was the strategy working as built, not a
  broken guard (verified live 2026-09-23: Industrials 46%, cap 50%). Candidate
  lever: cap any single
  industry at ≤ 30% of equity, enforced on new buys and pyramids only (never
  force-sells, so it is not a winner cap). Launch it with a same-day vintage
  control, one lever, standard gates. ⚑ This touches pyramids, which the
  Aug-12 bearish-gate kill covered. That verdict was about a *regime* gate on
  pyramids, not an exposure cap, but the owner should confirm the distinction.
  **BUILT 2026-09-23:** `shadow_industry_cap` launches 2026-09-30 alongside its
  same-day control `shadow_vintage_sep30` (gate pre-registered on
  `nostate_industry_cap`), so it reads ~5 weeks earlier than a post-readout launch.
- E-b: trend-regime-only real-money path. `real-money-gates.md` Gate 1
  already allows a ≤5% starter that mirrors buys only on trend days. That
  framework (Jul-22) and the Sep-9 go-live gate currently **both** claim to
  govern real money. Pick one as binding (see F).
- E-c: keep paper-trading unchanged and re-read at the Sep-30 vintage's
  30-day mark. (Waiting does not move criterion 1, so this is a decision to
  do nothing.)

## F. Housekeeping the owner should resolve before Oct-21
- **DECIDED 2026-09-23: the go-live gate governs real money.** Gate 1's
  trend-days-only starter survives only as fallback option E-b.
- **Two frameworks (resolved above).** `real-money-gates.md` (Gates 0–3, Jul-22) and the
  go-live gate (5 criteria, Sep-9) overlap and differ: Gate 1 needs trend-day
  t ≥ 1.5 and ignores the chop mix; go-live needs the blended edge. State
  which governs, or how they compose.
- **Prerequisite build (engineering, before the readout):** a daily
  closing-equity series per shadow stack. Today shadow equity is marked only
  "now", so the mechanism check (chop-day vs trend-day excess per arm) cannot
  be computed. The series can be rebuilt from `shadow_trades` + daily closes,
  so no history is lost by building it later, but it must exist and be tested
  before Oct-15.
  **BUILT 2026-09-23:** `shadow_equity_marks` table + `backend/shadow_equity.py`
  (backfills on boot, marks each session after the close).
  `GET /api/admin/shadow-equity/{name}` returns the marks and the arm-vs-
  comparator trend/chop split used by the mechanism check.

## Known execution haircut (measured 2026-09-23)
The app's books record every trade at the app's own quote: no spread or
slippage. The Alpaca paper mirror measures what fills actually cost: 10 fills,
**19 bps average** (buys 24.8, sells 13.2). At the owner book's real turnover
(~$1,950/day on ~$30k equity since March) that is **~1.2–1.3 bps/day** of
friction the gate never charges. Criterion 1's blended edge is therefore about
1.3 bps/day **better** than real money would see (−5.7 as served ≈ −7.0 net).
The pre-registered thresholds are unchanged; the readout reports the net figure
beside the served one. Re-measure from `broker_mirror.summary` at the readout
(paper fills are optimistic for thin small caps, so treat this as a floor).

## Data provenance for the readout
- Gate day counts: finished NYSE sessions only (`81bc23c`).
- Index closes: Yahoo with Alpaca gap-fill (`8a73347`); check
  `missing_dates` is empty over the window.
- Book closes: post-close closing mark from `736f826` onward. Earlier days
  use the last intraday tick (≈14:50 CT), a ±10-minute mismatch vs SPY
  (noise, not bias). Sep-22's close is the 14:15 CT tick (restart).
- Weekly emails use each arm's same-day comparator (`ae67e9b`).

## Amendment 2026-09-24 — shadow execution parity (owner-approved)
The Sep-24 audit found two ways every shadow arm traded unlike the live book.
Both are fixed for ALL arms at once, effective the **2026-09-25** session
(deployed after the Sep-24 close), so arm-vs-comparator deltas stay like for
like. No arm clock is reset and no shadow history is edited.

1. **Off-hours fills.** Live trades only while `is_market_open()`; the shadow
   phase had no gate and ran after every overnight/weekend scan. 175 of ~470
   shadow fills Aug-18..Sep-24 were off-hours, mostly at the day's close after
   full-day volume had confirmed the signal (a price live cannot get).
   Against the next session's open (the earliest price live could get), per-arm
   advantage was uneven: chop_entry_bar **+$237**, chop_trim **+$246**,
   baseline/cap50/chop_damper/sector_relief +$97, wide_trail +$91,
   chop_spy +$40, cs_exempt −$61, cs_window14 −$77, ml_veto_off −$62
   (pooled +2 bps, sd 112: noise overall, not per arm).
2. **Runt buys.** Live skips a buy below `min_position_value` (max $100,
   1.5% of book ≈ $375) after the cash clamp; the shadow loop did not, and
   bought $0.34–$274 positions that held one of the 8 slots for weeks (on
   Sep-24 every arm except chop_entry_bar, ml_veto_off and wide_trail had one
   open; stop_band and its comparator vintage_sep16 hold the same $0.34 FMAO).
   Existing runts exit through normal sell rules.

**Readout rule added:** each arm's Δ vs its comparator is reported (a) as
measured and (b) minus the arm's off-hours advantage over its comparator from
item 1 (re-measured at the readout with the same next-open method). An arm
"beats" only if it clears its bar under **both**. The post-Sep-25 window is
reported beside the full window for information; it is too short to govern.

## Amendment 2026-09-24 (b) — circuit breaker, fill quotes, corporate actions (owner-approved)
Same principle as (a): arms execute exactly as the live book would. Effective
the **2026-09-25** session for all arms at once; no clock resets.

1. **Drawdown circuit breaker.** Live halts new buys and pyramids at 15%
   below its equity peak and liquidates at 25% (`ai_trader.drawdown_protection`);
   arms had no breaker. Each arm now keeps a peak (`shadow_strategies.peak_equity`,
   seeded from its best daily mark) and applies live's rule after its sells.
   On Sep-24 shadow_chop_spy was 10.4% below its peak.
2. **Fill quotes.** Live fetches a fresh quote before every buy and pyramid;
   arms filled at the scan price. Measured against the 1-minute tape at the
   fill minute: live buys median 4 bps off, arm buys 66 bps (p90 197), arm
   pyramids 43; signed means ≈ −7 bps (noise, not bias). Arms now use the same
   fetch (`fetch_live_price`), one quote per name per tick shared by all arms.
3. **Corporate actions.** Cash buyouts were never booked: positions froze at
   the last trade and held a slot for good. A daily sweep now closes them at
   the deal cash on the effective date (a CVR counts as $0) for live books and
   arms alike. Retroactive corrections applied at the Sep-24 deploy:
   shadow_ml_veto_off ATAI (eff Sep-11, $6.75) and FBRX (eff Aug-27, $77),
   its equity marks rebuilt from those dates; live u3 ATAI likewise, with its
   snapshots from Sep-11 restated (not a gate book). shadow_ml_veto_off's
   trajectory after Aug-20 was distorted by up to two dead slots and should be
   read with that caveat.

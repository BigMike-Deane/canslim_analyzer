# Exposure strategies (A-family) — pre-registration (2026-10-07, before any number)

**Owner goal:** a data-driven way to beat SPY; "pivot into indexes as the market
churns, then lean in as it trends." Stock selection gives at best ~+1 pp/yr (Score
v5b, a tilt). This track asks the other question: **how much** market exposure to
hold, and when.

**Data (free):** S&P 500 index daily closes from 1927-12-30 (FMP `^GSPC`);
dividends from Shiller's monthly series (trailing 12-month D / P, accrued daily)
until SPY exists, then SPY dividend-adjusted closes (1993-02 →). Cash rate: NBER
short-term Treasury yields (FRED M1329AUSM193NNBR) 1928–33, 3-month T-bill
monthly (TB3MS) 1934–53, daily (DTB3) 1954 →.

**Periods:** **P1 = 1928-10-01 → 1993-12-31 (never examined in this program)**;
P2 = 1994-01-03 → 2026-10-05 (the 50-day unleveraged gate was tested here and
failed — T1). Benchmark: S&P total return, buy and hold.

**Execution and costs (all rules):** signal from the close of day t−1, position
taken at the close of day t, earning day t+1's return (one full day of delay).
Leverage above 1× pays the cash rate + 0.5%/yr on the borrowed amount plus a
0.9%/yr fund fee on the whole leveraged position (SSO-like); 1× pays 0.09%/yr.
Each change in exposure costs 5 bps of the amount traded.

**Rules (parameters from the published literature, not tuned here):**
- **A1 — 200-day trend, 2× (Gayed & Bilello 2016 "Leverage for the Long Run"):**
  2× S&P when the index is above its 200-day simple average, else cash.
- **A1r — same, 1× (reference only):** the unleveraged version.
- **A2 — volatility targeting (Moreira & Muir 2017 style):** exposure =
  min(2, 16% / realized volatility of the last 21 sessions, annualized), rest in
  cash; 16% ≈ the S&P's long-run volatility.
- **A3 — trend + volatility:** A2's exposure when above the 200-day average,
  cash otherwise.

**Pass rules (each of A1, A2, A3):**
1. CAGR > S&P TR in **P1 and in P2**.
2. Beats S&P TR in **≥ 60% of rolling 10-year windows** (monthly steps, full span).
3. Max drawdown ≤ S&P TR's max drawdown **in each period**.

**PASS → candidate overlay** for the stock model (v5b) and forward paper. Reported:
Sharpe, % time invested, switches/yr, worst year, the 2-day-delay and 0-day-delay
sensitivities, and the same rules with 1.5× and 3× (reporting only, not chosen).

**Prior (stated before running):** published work finds leveraged 200-day trend
beat buy-and-hold over 1928–2015 mainly by sidestepping 1929–32 and 2008; fast
V-shaped crashes (1987, 2020) and choppy decades hurt. P1 likely favourable, P2
uncertain; volatility targeting improves Sharpe but its CAGR edge depends on
leverage costs.

## Results (2026-10-07 ~10:45 AM CT, single run) — `research/pit/a_exposure.py`

Data check first (no strategy numbers): S&P TR 1928–93 9.85%/yr, max DD 83.6%,
worst year −43% (1931); 1994–2026 10.9%/yr, DD 55% — matching the record.

| Rule | P1 1928–93 CAGR (vs 9.61%) | P1 max DD (vs 83.6%) | P2 1994–26 CAGR (vs 10.90%) | P2 max DD (vs 55.2%) | Sharpe P1 / P2 (vs 0.38 / 0.51) | 10-yr windows beating | Verdict |
|---|---|---|---|---|---|---|---|
| **A1 200d 2×** | **14.55%** | 79.9% | **12.33%** | 45.7% | 0.52 / 0.51 | **86%** | **PASS** |
| A1r 200d 1× (ref) | 10.67% | 52.0% | 8.93% | 22.6% | 0.57 / 0.57 | 41% | ref |
| A2 vol target | 9.58% | 60.3% | 9.95% | 48.3% | 0.40 / 0.49 | 43% | FAIL |
| A3 trend + vol | 11.25% | 39.3% | 8.78% | 29.0% | 0.58 / 0.47 | 49% | FAIL |

A1: invested 64% / 75% of days, 5.6 / 6.9 switches a year. Sensitivity
(reporting): 1-day delay 14.93% / 11.11%, 3-day delay 13.31% / 11.07% — **P2
edge shrinks to +0.2 pp/yr with slower execution**; 1.5× 12.52% / 10.48%, 3×
17.30% / 14.87% (DD 93% / 65%).

**A1 PASSES** — the published "leverage for the long run" result replicates on
65 untouched years and on 1994–2026. **Caveats, in plain terms:**
1. The 1994–26 gain is leverage, not better risk-adjusted performance (Sharpe
   0.51 = 0.51); P1 is where the trend filter adds real Sharpe (0.52 vs 0.38).
2. The P2 margin (+1.4 pp/yr) is thin and sensitive to execution speed.
3. It still loses ~46–80% in the worst crashes; worst year −35.6% (P2).
4. ~6 switches a year → short-term gains in a taxable account (fine in an IRA).
5. Implementation = a 2× S&P fund (SSO) or margin, daily-rebalanced; the
   simulation models daily leverage costs and decay.

**Next:** combine A1 (when to lever) with v5b (what to hold) and run both as
forward paper arms.

## C1 pre-registration — A1 exposure × v5b holdings (2026-10-07 ~12:20 PM CT, before the corrected-data v5b numbers exist)

**Question:** does holding the v5b tilted book instead of the index improve A1?

**Rule C1:** each session, exposure from A1 (2× when the S&P closed above its
200-day average on the prior session, cash otherwise, same 1-day delay); the
invested sleeve holds **the v5b tilted portfolio** (re-weighted every 20
sessions exactly as v5b) instead of the S&P. Leverage, financing (cash rate +
0.5%), 0.9%/yr leverage fee and 5 bps per exposure change as A1; v5b's 19 bps
turnover cost on rebalances.

**Data / period:** v5b's out-of-sample years only, **2019-01 → 2026-09**, on the
**corrected ("v4") data** — the v5b re-run in `run_v4data.sh`. Daily returns
of the v5b book: each rebalance's weights × daily stock total returns until the
next rebalance (weights drift).

**Pass (all):** (1) C1 CAGR > A1 CAGR over the same dates; (2) C1 CAGR > SPY
TR; (3) C1 beats A1 in ≥ 5 of 8 calendar years; (4) C1 max DD ≤ A1 max DD +
5 pp. Reported: Sharpe, tracking vs A1, and C1 with 1× (unlevered timing) for
reference. **If v5b fails on the corrected data, C1 is not run** (no holdings
edge to combine). PASS → third Lab strategy (needs a 4th Alpaca paper account).

## Exposure family 2 pre-registration (2026-10-07 ~2:25 PM CT, before any number)

Owner: "What combination will create a model that competes and can potentially do
better than the SPY long term?" A1 passed, but its 1994–2026 edge fell to +0.2 pp/yr
with a 1–3 day execution delay (≈ 7 switches a year). Two published refinements
target exactly that; parameters from the papers, not tuned:

- **A4 — Faber (2007) monthly rule, 2×:** at each month-end, 2× S&P if the index
  closes above its 10-month simple average of month-end closes, else cash; held
  for the whole next month (traded at the first close of the month).
- **A5 — graded dual momentum (Antonacci 2014 absolute momentum + 200-day trend):**
  each session, signal 1 = index above its 200-day average; signal 2 = index
  12-month total return > the cash rate's 12-month return. Exposure 2× if both,
  1× if exactly one, cash if neither.

Same data, costs, delay, periods and gates as A1–A3 (CAGR > S&P TR in P1 and P2;
≥ 60% of rolling 10-year windows; max DD ≤ S&P's in each period). Exposure ledger:
A1–A3 tried → A4, A5 are trials 4–5 on the same 1928–2026 data, so a pass is
reported with the count. Sensitivity (reporting): 1- and 3-day delays.

## Exposure family 2 + combination results (2026-10-07 ~2:35 PM CT)

| Rule | P1 1928–93 (S&P 9.61%) | P2 1994–26 (S&P 10.90%) | P2 max DD (S&P 55.2%) | Sharpe P1 / P2 | Switches/yr | 10-yr windows | P2 at 1 / 3 / 5-day delay | Verdict |
|---|---|---|---|---|---|---|---|---|
| A4 Faber monthly 2× | 11.99% | **14.04%** | 48.6% | 0.42 / 0.55 | 1.4 | 73% | 14.69 / 14.79 / 14.11% | **PASS** |
| **A5 graded dual momentum** | **13.30%** | **13.67%** | **37.3%** | 0.49 / 0.55 | 4.1 | 80% | 12.10 / 13.21 / 14.40% | **PASS** |
| A1 (ref) | 14.55% | 12.33% | 45.7% | 0.52 / 0.51 | 6.9 | 86% | 11.11 / 11.07 / 12.85% | PASS |

Exposure ledger: 5 rules tried on 1928–2026, **3 pass** — the trend/absolute-momentum
+ leverage family works broadly, not one lucky rule. A5 has the best risk profile
(DD 37% vs 55%, worst year −30.5% vs −36.8%) and is delay-robust.

**Combinations (exploratory — C1's pre-registration said skip if v5b failed; run at
the owner's request), 2019-01 → 2026-09:** SPY TR 17.25%; A1 17.53% / on v5b book
17.56%; A4 13.89% / 13.31%; **A5 18.23% / 18.22%**. Holding the v5b book instead of
the S&P adds nothing under leverage: its weak years (2021 −6 pp, 2022 −3 pp vs SPY)
are doubled. All C-variants FAIL their gates. In this bull-dominated window only A5 is
clearly ahead of SPY (+1.0 pp/yr; A4's monthly checks were too slow for 2020).

**Candidate for the Lab: A5** (SSO when both signals, SPY when one, SGOV when none).

## International replication pre-registration (2026-10-07 ~2:30 PM CT, before any number)

**Why:** the strongest independent check of the exposure family — markets this
programme has never touched, including Japan's 1990–2012 collapse.

**Data (free):** FMP price indexes — Japan Nikkei 225 (1950→), UK FTSE 100 (1984→),
Hong Kong Hang Seng (1986-12→), Euro Stoxx 50 (1986-12→). Local cash: Japan discount
rate (FRED INTDSRJPM193N) to 1985-06 then call money (IRSTCI01JPM156N); UK 3-month
interbank (IR3TIB01GBM156N); Euro Stoxx: German 3-month (IR3TIB01DEM156N) to 1993,
then euro area (IR3TIB01EZM156N); Hong Kong: US 3-month T-bill (HKD pegged to USD
since 1983). **Dividends:** the indexes are price-only, so a constant yield is accrued
daily on both the strategy's invested leg and buy-and-hold: Japan 1.5%, UK 3.5%,
Hong Kong 3.0%, Euro Stoxx 3.0% (long-run averages); sensitivity ±1 pp reported.

**Rules:** A1, A4, A5 exactly as defined above (200-day / 10-month / 12-month vs
local cash), same costs (local cash + 0.5% financing, 0.9% leverage fee, 5 bps
per change), same one-day delay. Each market from its first date with a 252-session
warm-up and a cash rate.

**Pass (per rule):** beats local buy-and-hold CAGR in **≥ 3 of 4 markets** AND
has a shallower max drawdown in **≥ 3 of 4**. Reported: per-market CAGR, DD,
Sharpe, worst year, and Japan 1990–2012 separately.

## International replication results (2026-10-07 ~2:38 PM CT, single run) — **FAIL**

| Market (dividend assumed) | Buy & hold CAGR / DD | A1 excess | A4 excess | A5 excess | A5 DD |
|---|---|---|---|---|---|
| Japan 1954–2026 (1.5%) | 9.15% / 76.2% | **+2.46** | **+2.54** | **+2.79** | 69.0% |
| — Japan 1990–2012 | −4.18% / 76.1% | A1 −0.96% | A4 −3.68% | A5 −1.60% | |
| UK FTSE 1985–2026 (3.5%) | 9.09% / 47.2% | **−6.53** | **−2.47** | **−4.57** | 61.5% |
| Hong Kong 1988–2026 (3.0%) | 9.24% / 64.2% | +0.82 | −2.38 | +0.48 | 68.1% |
| Euro Stoxx 1988–2026 (3.0%) | 9.28% / 63.0% | +0.60 | +1.85 | +0.10 | 61.4% |

Excess in pp/yr. Signs unchanged across ±1 pp dividend assumptions.
**Tally: A1 beat 3/4, shallower DD 2/4; A4 2/4, 2/4; A5 3/4, 2/4 → all FAIL.**

**What it means:** the exposure family's US result (98 years, both eras) does not
generalize cleanly. It works where long trends dominate (Japan, including its
1990–2012 bust) and fails badly in a range-bound, high-rate market (UK 1985–2026:
leveraged trend paid 10–15% financing in the late 1980s and whipsawed through decades
of sideways prices). **A1/A5 are downgraded from "validated" to "US-validated
candidates"**: the Lab forward test is the deciding evidence, not a formality.
Caveat: price indexes + constant assumed dividends; local leverage costs modelled
with the same spread/fee as the US.

## Lab stop rules — pre-registration (2026-10-08 ~9:00 AM CT, before the first Lab order or fill)

**Why:** the Lab (A1 and A5 on their own Alpaca paper accounts) cannot prove an edge
quickly: the strategies switch ~4–7 times a year, and even in 1994–2026, where they
work, A1 trailed SPY by more than 44% over a year in 1% of windows. What the Lab
CAN test is whether live trading matches the backtest's assumptions. These rules say,
in advance, what counts as a bug, what weakens the case, and what ends a strategy.
Committed before any Lab fill; **no rule may be loosened after this commit.**
Thresholds may be tightened only with a written reason, and never in response to a
breach.

### Calibration (history the Lab never sees) — `research/pit/lab_stop_calibration.py`

- **Real SSO vs the backtest's cost model** (2× S&P TR − (T-bill + 0.5%) − 2 × 0.9% fee):
  SSO did **better** than modelled, +1.41%/yr since 2009 and +0.65%/yr in 2021–26.
  Trailing 126-session gap: 5th pct +0.37%/yr, worst −4.10%/yr. SPY vs the 1× leg
  +0.07%/yr; SGOV / BIL vs the 3-month T-bill −0.10 / −0.11%/yr. The backtest's
  leverage costs were conservative, so the leveraged edge is understated if anything:
  charging the 2021–26 gap, the 1994–2026 edge would be A1 +1.97% (was +1.43%) and A5 +3.29%
  (was +2.77%).
- **Break-even costs (1994–2026 edge → 0):** A1 trades 13.8× its equity a year → edge gone at
  **14 bps** per unit traded (the model assumes 5), or at 1.71%/yr extra drag on the 2×
  leg. A5 (11.4×/yr): **27 bps**, or 3.47%/yr. **A1's edge is fragile to execution cost.**
- **Live code = research code:** on 894 sampled sessions 2009–2026, `backend/lab.py`
  reproduces the research rule on 100.0% (A1) / 99.3% (A5) of sessions; all 6 A5
  differences fall within 0.2 pp of the momentum tie (SPY + BIL proxies vs S&P TR +
  T-bill rate).
- **Normal bad stretches** (trailing excess vs S&P TR, compounded):

| Sessions | A1 1994–26 1st pct | A1 1994–26 worst | A5 1994–26 1st pct | A5 1994–26 worst |
|---|---|---|---|---|
| 63 | −21.4% | −40.8% (2009-06) | −16.0% | −40.0% (2009-06) |
| 126 | −29.2% | −45.0% (2009-09) | −20.7% | −47.7% (2009-09) |
| 252 | −44.0% | −46.0% (2000-09) | −28.3% | −46.3% (2010-03) |
| 756 | −58.1% | −72.5% (2012-03) | −35.7% | −72.2% (2012-03) |

  1928–2026 worsts (1932–33 rebound) exceed −100%, so they would never fire; the modern
  era is the yardstick.

### Rules (per strategy, from its first broker-backed decision, 2026-10-08)

**M: mechanics.** Checked automatically every session. A breach is a **bug**: pause
and fix, then note it here. It is not evidence about the edge.
- **M1** A decision is recorded every NYSE session before the close.
- **M2** Every order placed is `filled` by the 17:35 ET mark (not canceled, expired,
  rejected or errored).
- **M3** After the close, ≥ 90% of equity is in the decision's target fund and no
  other fund is held.
- **M4** (monthly, research side) Each live decision is reproduced by the research
  rule; A5 differences are allowed only within 0.25 pp of the momentum tie.

**C: costs.** Checked automatically. These test the backtest's assumptions.
- **C1 Fill cost:** the notional-weighted adverse gap between fill price and the fund's
  official close, over all fills, once ≥ 6 fills exist. **REVIEW** above half the
  break-even (A1 > 7 bps, A5 > 13 bps). **STOP** at or above break-even (A1 ≥ 14 bps, A5 ≥ 27 bps).
  Caveat: Alpaca paper *simulates* closing-auction fills. Passing here is necessary,
  not sufficient, for real money.
- **C2 Fund tracking:** SSO's trailing 126-session return minus the modelled 2× leg
  (2× SPY TR − (BIL return + 0.5%) − 1.8%/yr), annualized. **REVIEW** below −1.71%/yr
  (A1's whole edge). **STOP** for a strategy when the trailing 252-session gap is below
  its break-even (A1 −1.71%, A5 −3.47%/yr) **and** the 1994–2026 backtest re-run with
  that gap shows edge ≤ 0.

**P: performance vs SPY total return.** Checked automatically from the daily marks.
- **P1** Trailing excess (Lab equity return − SPY TR, compounded) at 63 / 126 / 252 /
  756 sessions, once that many marks exist. **REVIEW** below the 1994–2026 1st
  percentile. **STOP** below the 1994–2026 worst (both from the table above).
- **P2** Lab drawdown deeper than the rule's 1994–2026 backtest maximum (A1 45.7%,
  A5 37.3%): **REVIEW**.

### Breach log

- **2026-10-08 (first order day): M2 + M3 breached on both strategies. Mechanics, not edge.**
  - Both submitted 344-share SSO market-on-close orders at 2:45 PM CT (on time).
  - Alpaca paper gave each **one partial fill at 15:59:58 ET** at the then-quote, then expired
    the rest at 16:02 ET:
    - A1: 288 shares @ $71.28, 82% invested
    - A5: 259 shares @ $71.30, 74% invested
  - The two accounts got different prices in the same second, so the paper engine fills cls
    orders against quotes rather than simulating the closing auction. A real closing-auction
    order would fill fully at the official close.
  - **Cause (Alpaca paper docs):** "When orders are eligible to be filled, they will receive
    partial fills for a random size 10% of the time", and quantity "is not checked against the
    NBBO quantities". A cls order gets no second fill before the close, so the rest expires.
    This is random simulator behaviour, not liquidity and not the strategy.
  - Trading code unchanged: on Oct-9 `plan_orders` tops up automatically (gap >5% of target),
    so M3 clears once that fills.
  - **Check-code fixes (same day):**
    - C1 skipped fills whose order ended `expired`. Now every execution counts, as the rule
      says ("over all fills").
    - M2 now lists breaches recorded here under `stop_rules.noted_breaches`, so they no
      longer show as open bugs. Any new breach still fires.
    - Rule thresholds are unchanged (no loosening).

**What each level means.**
- **REVIEW:** push to the owner. Within a week, a note here says whether it is a known
  failure mode: a V-shaped rebound while in T-bills, a whipsaw, or a crash at 2×. The
  strategy keeps running and no rule changes.
- **STOP:** the strategy is disabled (`enabled: false`) and its account moves to
  SGOV. Any revived or re-tuned version is a new pre-registered trial, counted in the
  ledger.

**Real-money readiness** (the owner's decision; this is the bar for recommending it):
- ≥ 63 live sessions
- ≥ 2 executed signal changes (or ≥ 126 sessions if fewer happen)
- no open M breach, C1 below REVIEW, no STOP

**This is deliberately not a performance test.** One year of live returns cannot
separate the edge from luck. The evidence for the edge is the 1928–2026 backtest, with
the international caveat above. The Lab's job is to show that live execution matches
the model, and that the owner can hold a 2× position through a drawdown.

**Implementation (2026-10-08, same day, no threshold changes):**
- Code: `backend/lab_checks.py`, thresholds in `config/default.yaml` `lab_strategies.<name>.stop_rules`.
- When: runs after each Lab close mark (16:35 + 17:35 ET).
- Where results go: stored on `lab_equity_marks.checks`, served at
  `/api/lab/strategies/<name>/checks` and shown as the "Stop rules" card on the Lab page.
- Alerts: pushes `lab_stop_rule` to the owner only when a rule reaches review or worse,
  and only when that is worse than the previous evaluation.
- Not automated: M4 (monthly, research side) and the C2 STOP re-run.

## Diagnostic: timing or just leverage? (2026-10-08, reporting only) — `research/pit/a_constant_leverage.py`

A1's 1994–2026 Sharpe equals the S&P's (0.51 = 0.51), so is the edge simply "hold more
market"? Each rule vs **constant leverage at the same volatility**:

| Period | S&P TR | A1 | A5 | Constant leverage, same vol |
|---|---|---|---|---|
| 1928–93 | 9.61% (DD 84%) | **14.55%** (DD 80%) | **13.30%** (DD 66%) | ~9.1% (1.24–1.31×, DD 91%) |
| 1994–2026 | 10.90% (DD 55%) | **12.33%** (DD 46%) | **13.67%** (DD 37%) | ~11.2% (1.27–1.30×, DD 67–68%) |
| 2009–2026 | 14.91% (DD 34%) | 14.76% (DD 41%) | 15.33% (DD 36%) | **~17.5%** (1.33–1.37×, DD 43–44%) |

**The timing is real, but all of it comes from big slow bear markets** (1929–32, 2000–02,
2008). In those the rules beat equal-risk leverage by 1–5 pp/yr and cut drawdowns
from ~70–90% to 37–80%. In the 2009–2026 bull market, including the V-shaped 2020 crash,
timing **cost** ~2.5 pp/yr against plain 1.35× leverage, and A1 trailed SPY itself.

These are crash insurance whose premium is paid in bull markets. They beat SPY over full
cycles (86% / 80% of 10-year windows), **not year by year**. If the next decade looks like
2009–2026, expect them to trail.

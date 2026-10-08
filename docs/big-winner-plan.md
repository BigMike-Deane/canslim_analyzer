# Big-winner study — pre-registration (2026-10-08 ~9:05 AM CT, before any number)

**Owner reframing (Oct-8):** the goal is not to beat SPY every year. It is to **find the big
winners** (DELL +110%, CARE, ECO) and accept uneven years, e.g. −5% vs SPY one year and
+40% another, as long as it is "grounded in something" with data. The owner also asked
whether to **take gains earlier** (e.g. at +30%, as with CARE) and park the money in the
index until the next pick.

**What we already know (from history, not this study):**
- The 2016–2026 replay of the app's rules (H7 diagnostic, `docs/phase2-pit-backtest-plan.md`):
  - All of its net realized profit came from the 4.75% of exits at ≥ +50% ($118k); the
    other 95% of exits lost money net.
  - Yearly excess vs SPY ranged from −36 to +16 pp, beating SPY in 3 of 11 years.
- Live (owner book, Apr-7 → Oct-7): DELL is ~$3.9k of the ~$5.6k gain (69%).
- Live DELL: the app sold slices at +25 / +41 / +51% and the last slice at +110%. That
  last slice made $2.6k of DELL's $3.9k.

## Part 1 — Can big winners be identified in advance? (`research/pit/b1_bigwinners.py`)

- **Panel:** the daily PIT score panel (`meta/daily`) sampled every 10th session,
  2016-01 → 2026-03 (the last date with a full 126-session future).
- **Universe:** price > $3 and market cap ≥ $300M (the M3 universe), all stocks
  including later-delisted ones.
- **Outcomes (next 126 sessions ≈ 6 months, from that day's close; delisted names
  valued at their last price; split-adjusted price return):**
  - **big winner (BW)** = return ≥ +50%
  - **big loser (BL)** = return ≤ −25%
- **Candidates (fixed list of 6, no others):**
  1. the app's composite score (`total`)
  2. earnings beat streak
  3. earnings surprise %
  4. earnings momentum = mean percentile rank of (beat streak, surprise %, EPS growth)
  5. 6-month price momentum, skipping the last month (close[t−21] / close[t−126])
  6. nearness to the 52-week high (close / 252-session high)
- **Reported, not candidates:** 60-session volatility and the largest daily gain in the
  last 21 sessions ("lottery" controls).
- **Top group:** percentile rank ≥ 0.90 within each date.
- **Gate (per candidate, in BOTH halves: entries 2016-01 → 2020-12 and 2021-01 → 2026-03):**
  1. **win lift** = P(BW | top) / P(BW | all) ≥ **1.5**
  2. **skew lift** = [P(BW | top) / P(BL | top)] ÷ [P(BW | all) / P(BL | all)] ≥ **1.25**
     (it must find winners faster than it finds losers)
  3. mean 126-session return of the top group > that of all eligible stocks
- **PASS** = all three in both halves. A pass makes the signal a candidate for a
  separately pre-registered big-winner portfolio. It is not a trading rule yet.
  6 candidates are tested, so one marginal pass is reported with that count.

## Part 2 — Take gains earlier? Park in the index? (`research/pit/p7_backtest.py --exit`)

The app's actual rules, run on point-in-time data 2016-01 → 2026-10 (the H7 engine,
current code incl. the Oct-7 breaker fix). The only change between variants is the exit
policy:

| Variant | Exit policy |
|---|---|
| **E0** | current rules (partial profits at +25/40/50%, trailing stops, take-profit) |
| **E1** | sell the whole position at the first close ≥ **+20%** |
| **E2** | sell the whole position at the first close ≥ **+30%** |
| **E3** | sell the whole position at the first close ≥ **+40%** |
| **E4** | E2 + park idle cash in SPY (the engine's `spy_sweep`: idle cash > 10% bought into SPY while SPY > its 50-day average, sold for the next buy or when SPY < 50-day) |
| **E5** | E0 + the same SPY parking |

E1–E4: the cap is checked on each day's close against the position's average cost, and
replaces any other sell that day. Every other rule is unchanged.

**Vintages:** start offsets 0, 20, 40 sessions (3 per variant; 18 runs).

**Score:** calendar-year excess vs SPY total return, 2016–2025 (10 full years; 2026 YTD
reported only). Take the median across the 3 vintages for each year.

**Gate (owner's terms):**
- **(a)** the average yearly excess is > 0
- **(b)** ≥ 3 of 10 years have excess ≥ **+20 pp** (the big years exist)
- **(c)** no year worse than **−15 pp** (looser than the owner's "−5%" example)
- **PASS** = (a) and (b) and (c).
- **"Does capping help?"** (reported): E1–E3 vs E0 and E4 vs E5, full-period CAGR. It
  "helps" only if better in all 3 vintages, "hurts" if worse in all 3, else "mixed".

**Prior (stated before running):** E0 likely fails. Its 2016–26 years already ran −36
to +16 pp. Caps likely lower returns: earlier live caps below +40% failed 5 times, and
the replay's profit is all in ≥ +50% exits. SPY parking likely helps a little, because
it puts idle cash to work.

**Caveats, known before running:**
- The replay's score approximates the live one: the C component uses GAAP EPS and
  there is no analyst-revision term. Over the same dates it did not buy DELL.
- Residual survivorship: later-delisted names are covered at 71–79%.
- Closing-price exits only.

**What happens next:**
- Part 1 PASS → a pre-registered big-winner portfolio test.
- Part 2 PASS → that exit policy becomes a shadow arm candidate.
- Everything FAILS → the live outperformance is most likely luck. The owner decides
  between the index and the A5 track.

## Part 1 results (2026-10-08 ~9:20 AM CT, single run) — **FAIL (0 of 6)**

Panel: 708,604 stock-dates, 5,171 companies, 258 dates; 126-session outcome known for
99.7%. Base rates: big winner (≥ +50%) 7.8% / 6.0% and big loser (≤ −25%) 11.9% / 16.6%
in 2016–20 / 2021–26.

| Signal (top 10%) | Win lift 2016–20 / 2021–26 (need ≥ 1.5) | Skew lift (need ≥ 1.25) | Mean 6-mo return, top vs all | Verdict |
|---|---|---|---|---|
| App composite score | 0.73 / 0.87 | 1.02 / 1.36 | +7.5 vs +15.0% / +4.8 vs +2.7% | fail |
| Beat streak | 0.78 / 0.77 | 1.21 / 1.05 | +9.2 vs +15.0 / +3.2 vs +2.7 | fail |
| Earnings surprise | 1.42 / 1.43 | 1.29 / 1.13 | +9.9 vs +15.0 / +2.3 vs +2.7 | fail |
| Earnings momentum (combo) | 0.88 / 1.19 | 1.00 / 1.42 | +8.0 vs +15.0 / +5.2 vs +2.7 | fail |
| **6-mo price momentum** | **1.59 / 1.98** | **1.12** / 1.36 | **+24.1 vs +15.1 / +6.7 vs +2.9** | fail (skew, 2016–20) |
| Near 52-week high | 0.66 / 0.70 | 0.92 / 1.08 | +6.4 vs +15.2 / +3.1 vs +3.3 | fail |
| *Control: 60-day volatility* | 2.19 / 2.29 | 0.94 / 0.91 | (means outlier-driven) | lottery |
| *Control: max 1-day gain (21d)* | 1.91 / 2.01 | 0.99 / 0.94 | | lottery |

**What it says:**
- **The app's score does not find big winners.** Its top decile holds *fewer* +50%
  stocks than the average stock, and fewer big losers: it selects steadier names.
- **6-month price momentum is the only signal that finds big winners and earns more on
  average** in both halves. But in 2016–20 it found big losers almost as fast (skew
  1.12 < 1.25), so it fails the pre-registered bar. It is the near-miss worth noting,
  not a pass.
- Volatility-type signals find big winners and big losers equally (skew ≈ 0.9–1.0):
  a lottery, as in the v3 scoreboard.
- Means here are equal-weighted and include small caps, so a few extreme outliers move
  them. Medians are in `meta/bigwin/b1_results.json`.

## Part 3 pre-registration — momentum entries + the app's exits (2026-10-08, before any number) — `research/pit/b3_momentum.py`

**Why:** Part 1 found the app's score picks *fewer* big winners than average, while
6-month momentum roughly doubles the big-winner rate (12% vs 6–8%) but also raises the
big-loser rate. The app's exits (cut at −7%, trail, trim, let the last slice run) are
built for big winners. **Question:** do the app's exits cut momentum's losers early
while its winners run?

**Simulation (standalone, daily closes, 2016-01 → 2026-10):**
- **Universe** (refreshed each 10th session): the M3 universe (price > $3, market cap ≥
  $300M, delisted names included) + 20-day average dollar volume ≥ $5M.
- **Candidates:** top 10% by 6-month momentum (close[t−21] / close[t−126] − 1), highest
  first.
- **Buys:** only when SPY closes above its 50-day average (the app's gate). Buy the
  highest-ranked candidate not held and not exited in the last 10 sessions, at the next
  session's close, sized at 1/8 of current equity (or the remaining cash). Max 8
  positions.
- **Exits (the app's rules, on each day's close):**
  - stop at −7% from cost
  - trailing stop from the peak close, by peak gain: ≥ 50% → 25%; 30–50% → 18%;
    20–30% → 12%; 10–20% → 6%; 5–10% → 4%
  - partial profits: sell 25% of the original shares at +25%, to 50% at +40%, and to
    75% at +50% (the app's tiers; its score conditions are dropped because there is no
    score here)
  - a ticker change or delisting exits at the last price
- **Costs and cash:** 0.095% per side; idle cash earns nothing.

**Variants:**

| Variant | Entries | Exits |
|---|---|---|
| **M1** (the hypothesis) | momentum | the app's |
| M2 (do the exits help?) | momentum | hold 126 sessions, no stops or partials |
| R (is momentum the reason?) | random draw from the eligible universe, 20 seeds | the app's |

M1 and M2 use start offsets 0 / 20 / 40 sessions; R uses offset 0.

**Score:** as Part 2. Calendar-year excess vs SPY TR, median across vintages, 2016–2025.

**PASS (M1) requires all of:**
- **(a)** the average yearly excess is > 0
- **(b)** ≥ 3 years with excess ≥ +20 pp
- **(c)** no year worse than −15 pp
- **(d)** M1's full-period CAGR (offset 0) is above the 90th percentile of the 20
  random-entry runs

**Reported, not gated:**
- M1 vs M2: do the stops help?
- Big-winner capture: exits ≥ +50% and ≥ +100%, and their share of profit
- Max drawdown
- Sensitivities: 20 slots, a 10% stop, idle cash in SPY

**Prior (stated before running):** momentum entries should beat random entries (Part 1
and the published literature). The open question is the app's tight early trailing
stops (4–6% below the peak), which may shake momentum winners out before they run. I
lean towards them hurting, i.e. M2 ≥ M1.

**PASS →** a shadow-arm or Lab candidate, with its own forward test.
**FAIL →** the big-winner framing has no tested strategy on free data yet.

## Part 3 results (2026-10-08 ~9:50 AM CT, single run) — **M1 FAIL**; the controls are the finding

SPY TR 2016-01 → 2026-10: 15.2–15.9%/yr depending on the start offset. Small caps (IWM,
price only) 9.2%/yr.

| Version | CAGR (offset 0 / 20 / 40) | Max DD | Exits ≥ +50% / ≥ +100% | Profit share from ≥ +50% | Median yearly excess: mean / years ≥ +20 / worst |
|---|---|---|---|---|---|
| **M1 momentum + app exits** | 11.3 / 11.4 / 9.1% | 53–57% | 29 / 7 | 32% | −0.9 / 2 / **−43.2** (2019) |
| M2 momentum, hold 6 months | **15.4 / 15.6 / 14.0%** | 56% | 34 / **12** | **85%** | **+7.0 / 3** / −54.9 (2023) |
| R random picks + app exits (20 seeds) | median 5.0%, 90th pct 8.5%, best 9.8% | | | | |

Sensitivities (offset 0, reporting only): 20 slots 6.8%; 10% stop 5.0%; **M1 with
idle cash parked in SPY 15.8%** (DD 57%).

**Gate:** M1 (a) fail, (b) fail, (c) fail, (d) pass (11.3% > random 90th pct 8.5%)
→ **FAIL**.

**What it says:**
1. **Momentum entries beat random entries by a wide margin** (11.3% vs the random median
   5.0%; better than all 20 random runs). The data does push big-winner picking past
   chance.
2. **The app's exits destroy most of that edge.** The same momentum picks held for 6
   months earned 15.4%/yr against 11.3% with the app's stops and trails. Holding caught
   12 doublers against 7; 85% of its profit came from ≥ +50% winners, against 32%. The
   tight early trailing stops (4–6% below the peak) shake winners out before they run.
   This matches the prior and Part 2's E0 profile. A wider 10% hard stop did not fix it.
3. **Even the best version only ties SPY over 2016–26.** It is far ahead of its own
   small/mid-cap universe (random 5%, IWM ~10% with dividends), but SPY's mega-cap decade
   (15%) set a high bar. Its years are exactly the "uneven" shape: +82, +46, +25 pp in
   2024 / 2025 / 2022. But the bad years run to −55 (2023) and −26 (2017), not −5.
4. **Idle cash is a big lever:** parking M1's idle cash in SPY added ~4.5 pp/yr (single
   run, sensitivity only). Part 2's E4/E5 test this on the app's real rules.

**Status:** M2 and "M1 + SPY parking" came out of this run as controls and
sensitivities. They are **new hypotheses, not passes**. Promoting either needs its own
pre-registration and a test it hasn't seen (forward paper or a holdout).

## Part 4 pre-registration — exit rules for momentum entries, company-split holdout (2026-10-08, before any number) — `research/pit/b4_exits.py`

**Why:** Part 3 showed the app's exits cut momentum's edge (11.3% vs 15.4%/yr held).
Searching exits on the same stocks and years would overfit, so **the companies are split
before anything is run**:
- **Half A** = sha256(CIK) mod 2 == 0. Every choice is made here.
- **Half B** = the rest. One confirmation run of the single chosen rule.

**Entries:** exactly Part 3, with the universe and the top-10% momentum ranking
restricted to the half being run:
- next-session close execution
- the SPY > 50-day gate
- 8 slots
- a 10-session cooldown
- 0.095% per side

**Exit family (5 rules from the momentum and trend-following literature, no tuning):**

| Rule | Exit |
|---|---|
| X0 | the app's exits (Part 3 M1), reference only, not selectable |
| X1 | hold 126 sessions (the standard 6-month momentum hold) |
| X2 | X1 + a −25% disaster stop from cost |
| X3 | 25% trailing stop below the peak close, no time limit |
| X4 | hold while still a leader: sell at the first 10th-session ranking where the stock is no longer in the top 30% by momentum |

X1–X4 are each run with idle cash as cash and with idle cash parked in SPY (total
return, while uninvested). That makes 8 selectable variants.

**Selection (half A only):** highest median full-period CAGR across start offsets
0 / 20 / 40. All 8 are reported.

**Confirmation (half B, the chosen variant, run once; same 3 offsets).**
**PASS requires all of:**
- **(a)** the average of the median-vintage yearly excess vs SPY TR, 2016–2025, is > 0
- **(b)** ≥ 3 years ≥ +20 pp
- **(c)** no year worse than −15 pp
- **(d)** CAGR (offset 0) is above the 90th percentile of 20 random-entry runs on half B
  with the same exit
- **(e)** the median-vintage CAGR exceeds SPY TR's CAGR over the same dates

Reported label if only (c) fails: **"passes except the drawdown tolerance."** Momentum
books are known to have deep bad years, so the owner decides whether that tolerance
moves. The label is set now, not after the fact.

**Prior (stated before running):** X1/X3 beat X0 on half A. Parking in SPY helps. On
half B the chosen rule beats random picks (d), but (c) fails: Part 3's best version had
a −55 pp year. Passing (e) is roughly a coin flip.

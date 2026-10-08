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

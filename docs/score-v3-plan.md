# Score v3 — a data-driven stock score, built from scratch (2026-10-07)

**Owner goal (Oct-7):** drop the CANSLIM score if needed; build "an extremely
educated and data driven estimate for the potential of a stock. Combining
several of these highly scored stocks should drive success over the long term.
It's ok if a few miss as long as we can narrow down on those big wins." Free
data only. Diversified, actively traded (not an index).

Everything below is **pre-registered before any v3 feature, model or portfolio
number exists.** The PIT data, gates and lessons come from
`docs/phase2-pit-backtest-plan.md` (every earlier pre-registered test failed).

## Design

**Universe (per panel date):** v3 PIT panel dates (every 10th session,
2016-01 → 2026-09), companies with actual close > $5, market cap ≥ $1B as known
that day (glitch rows dropped as in `m3_signal_tests.load()`), and 20-session
average dollar volume ≥ $5M.

**Target:** forward 60-session return minus SPY's 60-session total return,
winsorized per date at the 1st/99th percentile (keeps the right skew the owner
wants to capture; removes data glitches).

**Features (frozen list; all known strictly before D; per-date percentile
rank, centered; missing → 0):**

| Group | Features |
|---|---|
| CANSLIM inputs | C, A, N, S, L, I letter scores; EPS growth; annual EPS CAGR; earnings surprise %; beat streak; institutional % |
| Earnings / fundamentals | PEAD (S1); gross profitability (S2); net issuance (S3); earnings yield; book-to-market; ROE; −accruals |
| Price / volume | return 1m (reversal); momentum 3m, 6m, 12-1; distance from 52-week high; realized vol 60d; max daily return 21d; beta 252d vs SPY; log market cap; log dollar volume; volume trend (20d / 120d) |
| Setups | % from pivot; breaking out (live `detect_base_pattern`, as H6) |
| Analyst | net rating changes 90d (AM, as H8); # brokers 365d |
| Insider (new) | # distinct insiders with open-market purchases (Form 4 code P), prior 90d; net open-market $ (P − S) / market cap, prior 90d |
| Short interest (new, from mid-2018) | short interest / shares; days to cover; 3-month change in short interest |
| Market context (same for every stock on D; usable by the tree model only) | SPY vs 200-day MA; SPY 3-month return |

**Models (both fit; settings fixed now):**
- **M1 (primary): ridge regression** on the ranked features. Penalty chosen
  inside each training window from {1, 10, 100, 1000} by using the last
  training year as validation. Weights are printed every year (readable).
- **M2: gradient-boosted trees** (`sklearn` HistGradientBoostingRegressor,
  max_depth 3, learning_rate 0.05, max_iter 300, l2 1.0, min_samples_leaf
  200) — can learn interactions (e.g. "momentum only in bull markets").

**Walk-forward:** for each test year Y = 2019 … 2026, train on panel dates
whose 60-session target window ends before Jan 1 of Y (embargo — no
overlap), score every panel date in Y. Nothing from Y or later touches the
model that scores Y.

**Portfolio simulation (out-of-sample 2019-01 → 2026-09):** every 20
sessions, rank by score; hold the **top 25 equal-weight**, at most 5 per
sector; a holding is kept while it stays in the top 50 (cuts turnover).
19 bps round-trip cost on all turnover. Delisted names exit at their last
price. Compared against SPY total return over identical dates.

## Pass rules (all required, for the model that is reported as the candidate)

1. **Signal:** out-of-sample per-date Spearman IC (score vs 60-session excess)
   averaged over 2019–2026, **Newey–West t ≥ 3.0** (6 lags), and mean IC > 0
   in both 2019–22 and 2023–26.
2. **Portfolio:** net CAGR **> SPY total-return CAGR** over 2019-01 → 2026-09.
3. **Consistency:** the portfolio beats SPY in **≥ 5 of the 8** calendar
   years (2026 year-to-date counts).
4. **Risk:** max drawdown ≤ SPY's max drawdown + 10 pp over the same period.

The candidate is whichever of M1/M2 has the higher mean OOS IC; both are
reported in full. (Two models tried → the t ≥ 3 bar already covers the
multiplicity.)

**PASS → forward paper trading** as a shadow arm (same rules, live data) for
**3 months** before any real money or app change. **FAIL → stop scoring
research on free data;** the honest recommendation becomes index funds (or a
paid deeper dataset for a new, fully out-of-sample test).

## Reported, not gating

- Big winners: share of portfolio picks gaining ≥ +50% within 120 sessions vs
  the universe base rate; profit share of the best 1 / 5 positions.
- Annual weights (M1) and feature importances (M2, permutation).
- Each year's IC, hit rate, turnover, sector mix.
- Same portfolio on the H7 universe (≥ $300M) as a robustness view.

## Known limitations (stated before running)

- Several features (PEAD, issuance, CANSLIM letters, momentum, AM) were
  examined one by one on 2016–2026 in earlier tests, so the researcher has
  seen their full-sample behaviour. The walk-forward fit is still genuinely
  out-of-sample for the **weights**, but feature selection is not fully
  blind. A pass is therefore confirmed only by forward paper trading.
- 10 years of data, 8 test years; one market era (mega-cap led).
- C uses GAAP EPS; no estimate-revision history; short interest from 2018.

## Amendment 1 — before the real run (2026-10-07, after the placebo only)

The placebo run (target shuffled within date; no real score result exists yet)
and two baselines showed the yardstick needs fixing:
- **Dividends.** Stock returns are split-adjusted *price* returns while SPY is
  total return. Over the test dates the cap-weighted universe returned 14.7%/yr
  price-only vs SPY TR 17.2%; ~1.5 pp/yr of that is missing dividends — enough
  to decide gate 2. **Fix: add each stock's cash dividends (FMP `/stable/dividends`,
  ex-dates inside the window) to r20/r60**, so both sides are total return.
- **Luck band.** Three random top-25 portfolios returned 13.6%, 9.7% and 4.0%/yr
  over the same rebalances. Portfolio CAGR alone is a noisy judge. **Added to the
  report (not a gate): the model portfolio's percentile among 200 random-score
  portfolios** built by the identical rules.
- **Placebo calibration.** M1 placebo IC +0.0003 (t 0.19). M2 placebo IC +0.0036
  (t 2.06) on one seed → run 4 more placebo seeds and report M2's null spread
  beside the real t.
Gates, features, models and settings are unchanged.

## Results (2026-10-07 ~9:35 AM CT, single run, rules as pre-registered + amendment 1)

Dividends added (median 60-session yield 0.12%, 53% of rows pay). Walk-forward
2019–2026, 8 test years, 344k out-of-sample stock-dates.

| | M1 ridge | M2 boosted trees | SPY TR |
|---|---|---|---|
| Mean OOS IC (NW t) | −0.0055 (t −0.30) | −0.0030 (t −0.13) | |
| IC 2019–22 / 2023–26 | −0.037 / +0.031 | −0.037 / +0.036 | |
| Portfolio CAGR | 7.1% | 5.3% | 17.2% |
| Max drawdown | 62.3% | 69.5% | 31.0% |
| Years beating SPY | 3/8 | 1/8 | |
| Percentile among 200 random portfolios | 16th | 6th | (median 10.0%) |

M1 by year: 2019 +27.7 vs +30.6 · 2020 +20.3 vs +14.0 · 2021 −17.2 vs +26.8 ·
2022 −28.0 vs −12.4 · 2023 +11.9 vs +20.8 · 2024 +12.5 vs +28.4 · 2025 +20.7
vs +16.3 · 2026 YTD +20.8 vs +11.5.

**Score v3 FAIL — all four gates.** Both models did *worse than random picks*.

**What the model learned (ridge weights, stable across years):** high beta
(+2 to +4), smaller companies (log cap −1 to −2), expensive / low earnings yield
(E/P −1 to −2), far below the 52-week high (−1 to −5), short-term losers (1-month
return −1.3 to −2.6), plus N and A from CANSLIM. That is a **high-beta,
small-cap "junk rebound" tilt** — it is what 2016–2020 training data rewarded,
it paid hugely in the 2020 rebound (M2 +119%) and then crashed in 2021–22
(M1 −17%/−28%, M2 −22%/−44%). The sign of the signal flipped with the regime:
IC −0.037 in 2019–22, +0.03 in 2023–26. A score whose sign depends on an era it
cannot see in advance is not a score.

**Note, not evidence:** the 2023–26 half is positive for both models (IC
+0.03, and M1 beat SPY in 2025 and 2026 YTD), after training on 7+ years that
include a crash and recovery. One half-period after a failure is a hypothesis,
not a result; the pre-registered rule is FAIL.

**Pre-registered consequence: stop scoring research on free data.** Across
this program, every pre-registered test of stock selection on 2016–2026 has
failed — single signals, CANSLIM letters, reweightings, the live rules, and now a
learned multi-signal score with 38 inputs. What remains open needs either more
history (a paid point-in-time dataset to test on years never touched) or forward
paper trading.

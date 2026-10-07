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

## Score v4 pre-registration — stable signals, equal weights, large caps (2026-10-07, before any v4 number)

**Why v4 (owner: "find a better way of picking stocks"):** an exploratory
per-signal scoreboard on the v3 table (`meta/v3_signal_scoreboard.csv`, full
2016–2026, not a test) showed a handful of signals with consistent sign in both
halves (beat streak t 3.9, low days-to-cover 3.6, low issuance 3.0, surprise
2.9, ROE 2.8), while ridge in v3 put its weight on loud, regime-flipping
factors (beta, size, value). Equal-weighting only *stable* signals is the
standard remedy. **Honest label:** this is the second model family tried on
the 2019–2026 test years, and its design was chosen after seeing the
scoreboard. The selection itself is mechanical and walk-forward (below), but
**a pass is confirmed only by forward paper trading.**

**Universe (primary):** each date, the **500 largest** companies of the v3
universe by market cap as known that day. (Reporting only: the full v3
universe.) Rationale fixed in advance: the equal-weight mid-cap universe
trailed SPY ~5 pp/yr over 2019–26 before any selection.

**Signal selection (per test year Y, training dates only, 60-session embargo
as v3):** candidate pool = all 38 v3 features (no hand-picking). For each
feature compute per-date Spearman IC vs the winsorized 60-session excess
return **within the 500-largest universe** over the training dates. **Keep it
if NW t ≥ 2.0 (6 lags) and the mean IC has the same sign in the first and
second halves of the training dates.** Direction = sign of its training IC.

**Score:** equal-weight mean of the kept features' signed, centered per-date
percentile ranks (missing → 0). If no feature is kept, hold the cap-weighted
universe that year.

**Portfolio (out-of-sample 2019-01 → 2026-09):** every 20 sessions take the
top **30** by score; weights ∝ √(market cap), capped at 8% per name, ≤ 6 names
per sector; a holding is kept while it ranks in the top 60. 19 bps round trip
on turnover (weight-based). Returns are total return (dividends, amendment 1).
Benchmark SPY TR over identical dates.

**Gates (identical to v3):** (1) out-of-sample score IC in the 500-largest
universe, NW t ≥ 3.0 and mean > 0 in both 2019–22 and 2023–26; (2) net CAGR >
SPY TR; (3) beats SPY in ≥ 5 of 8 calendar years; (4) max DD ≤ SPY + 10 pp.
Reported: luck-band percentile (200 random-score portfolios, same rules and
universe), features kept each year, IC/portfolio on the full v3 universe.

**Either way → forward paper trading** (owner, Oct-7): v4 (and v3 ridge for
comparison) run as shadow arms on live data; a v4 PASS here is a candidate,
not a proof.

## Score v4 results (2026-10-07 ~10:30 AM CT, single run)

Placebo first (3 seeds): OOS IC t −0.32 / −1.09 / −0.45 — clean; the t ≥ 2 rule
still admits 1–5 noise features per year from 38 candidates.

Features kept (500 largest, training years only): 2019 −AM; 2020–22 mixes of
ROE, institutional %, analyst coverage, beat streak, −EPS growth, GP/A;
2023–26 essentially **ROE alone** (± GP/A).

| | v4 | SPY TR |
|---|---|---|
| OOS IC, 500 largest (NW t) | −0.0081 (t −0.61); 2019–22 +0.008, 2023–26 −0.027 | |
| Portfolio CAGR | 12.1% | 17.2% |
| Max drawdown | 33.3% | 31.0% |
| Years beating SPY | 2/8 (2019, 2022) | |
| Luck band (200 random 30-stock portfolios, same rules) | 56th percentile (5th 8.3%, median 11.8%, 95th 15.6%) | |
| Full v3 universe IC (reporting) | +0.0084 (t 0.74) | |

**Score v4 FAIL (gates 1–3; gate 4 passes).**

**The decisive finding is the luck band, not the score:** the *median random*
30-stock portfolio drawn from the 500 largest, √cap-weighted, returned 11.8%/yr
vs SPY's 17.2% — a **~5.4 pp/yr structural gap** before any selection. 2019–26
returns were concentrated in a handful of mega-caps held at index weight; a
diversified 30-name book, random or smart, rarely holds them at that weight.
The stable signals measured on the full universe (IC ≈ 0.02–0.04) are worth
perhaps 1–2 pp/yr — not enough to close 5. **To beat SPY by stock selection in
this era, the portfolio has to be built relative to SPY's own weights
(overweight / underweight), not as a stand-alone list of picks.**

## Trial ledger + Score v5 pre-registration — SPY-relative tilt (2026-10-07, before any v5 number)

Owner (Oct-7): "keep iterating and learning until we have a model that starts
to have an edge on the SPY." Iterating on the same 2019–2026 test years raises
the chance of a lucky winner, so every variant is logged here and the bar rises
with the count.

**Trial ledger (stock selection on 2019–2026 OOS):** 1 v3-M1 ridge, 2 v3-M2
trees, 3 v4 stable EW (all FAIL). v5 adds 4 v5a, 5 v5b, 6 v5c.

**Benchmark validated before any v5 number:** cap-weighted 500 largest of the v3
universe, total return, 2019–26 = **17.2%/yr = SPY TR 17.2%**, tracking error
1.7%/yr. v5 tilts are built on it.

**Portfolio (all v5 variants):** every 20 sessions, weight w_i ∝ cap_i ×
m_i with m_i = 1 + (2u_i − 1), u_i = score percentile in the 500 largest
(bottom-scored → 0×, median → 1×, top → 2×; missing score → 1×). Long-only,
fully invested. 19 bps per unit of one-way weight turnover. Total return.

**Scores:**
- **v5a:** v4's walk-forward stable-signal score (selection inside the 500 largest).
- **v5b:** same selection rule, but IC measured on the **full v3 universe** (more
  stocks → more power), score applied to the 500 largest.
- **v5c:** fixed composite of the six signals with consistent sign in the
  exploratory scoreboard (+beat streak, +surprise %, +PEAD, +low issuance (S3),
  +ROE, −days to cover), equal weights. **In-sample by construction** (chosen
  from a full-2016–26 look) — reported, but it cannot pass on its own.

**Gates (v5a / v5b):** (1) CAGR > SPY TR, 2019-01 → 2026-09; (2) beats SPY in
≥ 5 of 8 calendar years; (3) **luck-adjusted:** active return (CAGR − SPY TR)
above the 95th percentile of the *best of 6* random-score tilts (6 = trials in
the ledger so far; deflates for multiple tries); (4) max DD ≤ SPY + 10 pp.
Reported: OOS IC in the 500 largest, tracking error, information ratio,
turnover. A pass → forward paper as a candidate (not proof).

## Score v5 results (2026-10-07 ~11:15 AM CT, single run + robustness)

Placebo (2 seeds): v5a/v5b OOS IC t −0.32…+1.20, noise-selected tilts −0.1 to
−3.5 %/yr active (a bad tilt is not free).

| | Active vs SPY TR /yr | Years > SPY | OOS IC, 500 largest (t) | TE | IR | Gates |
|---|---|---|---|---|---|---|
| v5a | +0.11% | 3/8 | −0.008 (−0.61) | 2.4% | +0.11 | **FAIL** |
| **v5b** | **+0.92%** | **5/8** | **+0.029 (+2.38)** | 2.3% | +0.42 | **PASS (all 4)** |
| v5c (in-sample) | +1.68% | 6/8 | +0.040 (+3.20) | 2.1% | +0.73 | n/a |
| luck bar (best of 6 random tilts, 95th pct) | +0.91% | | | | | |

v5b CAGR 18.15% vs SPY TR 17.23%; max DD 31.4% vs 31.0%; turnover 7% per
rebalance. By year: 2019 −0.2 · 2020 +4.2 · 2021 −4.0 · 2022 −2.3 · 2023 +2.4 ·
2024 +5.7 · 2025 +1.8 · 2026 YTD +0.6 pp vs SPY.

**Features v5b kept (walk-forward, full-universe IC):** 2019 none; 2020–22 beat
streak, analyst coverage, EPS growth, surprise, AM; **2023–26 converge on beat
streak, surprise, analyst coverage, low days-to-cover, ROE, low issuance (+
inst %, A, SI)** — the same family as the hand-picked v5c, found without
looking ahead.

**Robustness (reporting; nothing re-chosen):** v5b active +1.09% (6/8) on the
alternate rebalance calendar, +0.71% with costs ×2, +0.46% / +1.29% at tilt
0.5× / 1.5× (monotone in tilt — what a real signal does). Luck with 2,000 random
tilts: median −1.01%, best-of-6 95th +0.86%; **P(best of 6 random ≥ +0.92%) =
3.8%.**

**Reading:** a **modest, real-looking edge of about +1 pp/yr over SPY** at
~2.3% tracking error (IR ≈ 0.4), from earnings-execution + quality + low
short-pressure signals tilting an S&P-like book. The pass is at the margin of a
deliberately strict bar and comes after 6 trials on the same years. **Status:
candidate → forward paper ledger** (weights committed to git before the
returns happen).

## v5b diagnostics → data repair (2026-10-07 ~12 PM CT)

`v5_diagnostics.py` (nothing chosen; v5b's yearly selections reused):
1. **Factor attribution** (active vs cap-weighted 500 largest, +1.10%/yr raw):
   alpha after factors +0.25%/yr (t 0.57), R² 0.56. Main loadings: **beat-streak
   factor +0.099 (t 6.3; the factor returned +3.3%/yr long-short in large caps)**
   and **anti-low-volatility −0.042 (t −4.8; low-vol returned −10%/yr in 2019–26
   — partly an era exposure)**. Size/value/momentum/quality loadings ~0.
2. **Losing years:** active vs its own benchmark was positive every year except
   2022 (−1.2 pp); mostly sector allocation (+0.9 to +1.7 pp/yr). The large
   "losses" vs SPY in 2021 came from the **benchmark proxy** (−5.6 pp vs SPY in
   2021, +3.9 pp in 2020), not the tilt.
3. **Generalization to ranks 501–1000** (never traded): IC +0.029 (t 2.02),
   tilt +0.66%/yr vs that universe's cap-weighted benchmark, 90th percentile of
   300 random tilts. Supportive, not decisive.

**Data defect found via (2):** market caps built from SEC share counts were off
by > 2× for ~6% of companies (170 of 2,791 checked against FMP): per-class or
scaled counts (AVGO 0.08×, KLAC 0.11×, MA 0.13×, CRWD 0.18×, V 0.24×), ADR
ratios (ONC/BeiGene 11.6×), and **Alphabet absent 2016 → mid-2024** (no combined
SEC share count before 2024). The full-period proxy matched SPY by offsetting
errors. **Fix (`4f5b16e`): FMP's daily historical market cap is primary
(`m2_adapter.mcap_asof`), SEC shares the fallback.** Every stage is rebuilt
("v4 data", `run_v4data.sh`) and v3, v4, v5 and the diagnostics re-run with
rules unchanged; v3-data results are kept (`*_v3_oct07`). Earlier programme
verdicts (H1–H8) used the same caps; their equal-weighted tests are only mildly
affected (universe membership), and will be noted, not re-litigated.

## Re-runs on corrected ("v4") data (2026-10-07 1:35 PM CT, rules unchanged)

Panel 727,669 rows / 5,209 CIKs (Alphabet present every year; AVGO $1.76T,
KLAC $283B, MA $474B, ONC $34B as of 2026-07). v3 table 485,322 rows, median
1,857 names/date. Placebo IC t +1.54 / +1.25 (null). Benchmark (cap-weighted
500 largest, TR) 17.19%/yr vs SPY TR 17.23%.

| Model | v3-data result | **Corrected result** | Verdict |
|---|---|---|---|
| v3 M1 / M2 | IC t −0.30 / −0.13 | IC t −0.35 / +0.07; 8.8% / 10.6% vs 17.2%; 35th / 57th pct of random | FAIL (unchanged) |
| v4 | IC t −0.61, 12.1% | IC t +0.41, 8.3%, 4th pct of random | FAIL (unchanged) |
| v5a | +0.11%/yr | −0.40%/yr, 5/8 yrs | FAIL |
| **v5b** | +0.92%/yr, PASS at the margin | **+0.67%/yr, 6/8 yrs, IC t +1.59, TE 2.6%, IR 0.30; luck bar +0.71%** | **FAIL (gate 3 by 0.04 pp)** |
| v5c (in-sample) | +1.68%/yr | +1.45%/yr, IC t 3.76 | n/a |

v5b by year vs SPY: 2019 +1.9 · 2020 +4.8 · 2021 −6.0 · 2022 −3.0 (vs SPY; vs
its own benchmark −0.3 / −1.1) · 2023 +2.4 · 2024 +3.7 · 2025 +2.3 · 2026 +0.4.

**Diagnostics (corrected):** active vs benchmark +0.88%/yr; factor-adjusted
alpha +0.25%/yr (t 0.66), R² 0.64; **beat-streak factor loading t +8.2** (that
factor: +2.9%/yr long-short in large caps), anti-value t −2.8, anti-low-vol
t −3.0. Ranks 501–1000 (never traded): +0.80%/yr vs own benchmark, **98th
percentile** of random tilts (IC t 1.0).

**Reading:** the stock-selection edge is real-looking but small — about +0.7
pp/yr over SPY, consistent across years and in a universe it never traded,
yet not distinguishable from the best of 6 tries on 2019–2026 alone. It is
essentially the earnings-beat-streak effect. **Per the C1 pre-registration, C1
is not run.** Next: forward paper (free, the only clean evidence) and a
pre-registered v6 that targets the beat-streak mechanism directly.

## Score v6 pre-registration — earnings-reaction signals (2026-10-07 ~1:50 PM CT, before any v6 number)

Owner (Oct-7): "keep iterating until we get a better model and then put that in
the Lab." v5b's corrected-data edge is essentially the earnings-beat-streak
effect, so v6 targets the earnings mechanism with a **feature never examined in
this program**:

- **EAR (earnings-announcement return; Brandt, Kishore, Santa-Clara & Venkatachalam
  2008):** the stock's return minus SPY's over the 3 sessions around its latest
  report (close of session E−2 → close of E+1, E = FMP report date; timing
  before/after the bell unknown, so the window spans both). Known on D only if
  E+1 < D. Missing if the latest report is > 120 days old.
- **days_since_report** (for recency weighting).
- Revenue surprise was considered and **dropped before use**: FMP revenue
  estimates are unreliable (20% of 2016+ rows off by > 30%; AAPL Oct-2021
  estimate $118B vs actual $83B).

**Variants (both = trials 7 and 8 in the ledger):**
- **v6a:** v5b exactly, with EAR added to the candidate pool (the walk-forward
  selection rule decides whether it is kept and its sign).
- **v6b:** v6a, plus **recency weighting** of the earnings-event features
  (beat streak, surprise %, PEAD, EAR): their centered rank × exp(−days since
  report / 60); plus **sector-neutral ranks** (all features ranked within sector
  on each date).
Portfolio, costs, benchmark, universe (500 largest) and walk-forward exactly as
v5. Corrected ("v4") data.

**Gates (each variant):** (1) CAGR > SPY TR; (2) beats SPY ≥ 5/8 years; (3)
active return > 95th percentile of the **best of 8** random-score tilts; (4) DD
≤ SPY + 10 pp. **Then one confirmation on ranks 501–1000** (never used for
selection): the same tilt vs that universe's cap-weighted benchmark must be
above the 90th percentile of 300 random tilts there. Pass both → Lab candidate.

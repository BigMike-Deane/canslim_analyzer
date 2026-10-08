# Decision memo — can the app beat SPY? (testing Oct 5–8, 2026)

**The question (owner, Oct-5):** "get to a point where I trust it to beat the SPY consistently
enough that I can copy its buys and sells… if I can't beat the SPY then what's the point?
I'll just put all of my money into an index."

**Answer:**
- **Stock picking: no.** More than 20 pre-registered tests, all on point-in-time data from
  2016 to 2026, and none beat SPY.
- **Market exposure (how much to hold in stocks):** two rules passed US history from 1928.
  They are paper-trading now, on day 1.

> **DECISION (owner, 2026-10-08 ~4:10 PM CT): Option A.**
> - Real money stays in index funds.
> - The app becomes the A1/A5 forward-test platform; AI Portfolio keeps paper trading.
> - Stock-picking research stops.
> - Shadow strategies get archived after the binding Oct-21 readout (owner approves the archive).

## What was tested

| Test | Result |
|---|---|
| Live score, Jul–Oct 2026: do higher scores do better? | No (t −0.92) |
| CANSLIM score rebuilt point-in-time, 2016–26, ~4,800 companies | FAIL (t 0.40) |
| Market timing: 50-day SPY gate, M composite, 1994–2026 | FAIL |
| Earnings drift, value, low volatility, quality, Score v2 | FAIL. Earnings drift is the strongest (t 2.7) but doesn't beat SPY after costs |
| Do high scores or chart setups find the big winners? | FAIL. They find *fewer* |
| The app's exact live rules, replayed 2016–26 | **1.9%/yr** (5.8% with the breaker trap removed) vs SPY 15.2% |
| Analyst rating momentum | FAIL |
| Score v3–v7, built from scratch (9 versions) | Best: +0.67%/yr over SPY. Fails the luck test. |
| Big-winner study: momentum entries + 8 exit rules | FAIL. 33%/yr on half the companies, **0.3%/yr on the other half** |
| **Exposure rules A1 / A5** (2× S&P in uptrends, T-bills otherwise) | **PASS** 1928–93 and 1994–2026 (A5: 13.7% vs 10.9%/yr, max drawdown 37% vs 55%) |

## Why stock picking keeps failing

1. **The decade belonged to mega-caps.** A random 30-stock large-cap portfolio made 11.8%/yr
   against SPY's 17.2%. Any stock list starts ~5 pp/yr behind before it picks anything. The
   app's small/mid-cap universe is even further back.
2. **Big winners are luck of the draw.** In the last test, 5 stocks made 113% of one half's
   profit. The other half drew none and lost money. The live book has the same shape: DELL
   is ~15 pp of the +20.7% from April to Oct-6, and every other closed trade together is about −$190.
3. **Tight exits cut the little edge there was.** Trailing stops 4–6% below the peak sell
   the winners early.

## What A1 / A5 are, and are not

- They decide **how much** stock market to hold, not which stocks.
- Their edge is **crash insurance**: they sidestep slow bear markets.
- They **trail in long bull runs**: 2009–26, they lost ~2.5 pp/yr to simply holding 1.35×.
- Abroad: they **work in Japan and fail in the UK.**
- So: real, but **not "beats SPY every year."** Expect stretches of trailing.
- Forward test started Oct-8, on two separate Alpaca paper accounts. Stop rules are
  pre-registered (`docs/exposure-plan.md`) and checked automatically after every close.

## Options

| | What | Cost / risk |
|---|---|---|
| **A (recommended)** | Real money stays in index funds (already true: VTI/VXUS). The app becomes the A1/A5 forward-test platform. Stock-picking research stops. AI Portfolio keeps paper trading. | Nothing new. Honest about the evidence. |
| B | A + a small "fun money" sleeve that copies the stock picks, sized so a DELL-type win matters and a wipeout doesn't | Expect to trail SPY on that sleeve |
| C | Buy longer data (Sharadar $39–69/mo, 1998→) and re-run on 2000–15 | Weeks of work. After 20+ fails, low odds it flips the answer |

**Suggested checkpoints (owner to set):**
- A1/A5 first review ~Apr-2027 (6 months).
- No real money on A1/A5 before a full year forward. They hold 2× leverage, and one year
  can't show the crash protection they are for.

## Record

Detailed results:
- `docs/phase2-pit-backtest-plan.md`
- `docs/score-v3-plan.md`
- `docs/big-winner-plan.md`
- `docs/exposure-plan.md`

Big-winner Part 2 (exit caps on the app's own entries) was stopped at 7 of 18 runs, unscored.
It re-tested a question already answered (caps below +40% destroyed value in Jul and earlier),
it used the app's entries, which Part 1 showed have no big-winner edge, and it had no holdout.
It can resume from where it stopped.

import { useState } from 'react'

function ScoreCard({ letter, title, maxPoints, description, factors, color }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="card mb-3">
      <div
        className="flex justify-between items-center cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg ${color} flex items-center justify-center font-bold text-lg`}>
            {letter}
          </div>
          <div>
            <div className="font-semibold">{title}</div>
            <div className="text-dark-400 text-sm">{maxPoints} points max</div>
          </div>
        </div>
        <span className="text-dark-400">{expanded ? '▲' : '▼'}</span>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-dark-700">
          <p className="text-dark-300 text-sm mb-3">{description}</p>
          <div className="space-y-2">
            {factors.map((factor, i) => (
              <div key={i} className="flex justify-between text-sm">
                <span className="text-dark-400">{factor.condition}</span>
                <span className="text-dark-200">{factor.points}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function WeightSlider({ label, value, onChange, description }) {
  return (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-sm text-primary-500">{value}%</span>
      </div>
      <input
        type="range"
        min="0"
        max="40"
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider"
      />
      <p className="text-xs text-dark-400 mt-1">{description}</p>
    </div>
  )
}

function GrowthProjectionSection() {
  const [weights, setWeights] = useState({
    momentum: 20,
    earnings: 15,
    analyst: 25,
    valuation: 15,
    canslim: 15,
    sector: 10,
  })

  const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0)
  const isValid = totalWeight === 100

  const updateWeight = (key, value) => {
    setWeights(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div className="card mb-4">
      <h2 className="text-lg font-semibold mb-4">Growth Projection Weights</h2>
      <p className="text-dark-400 text-sm mb-4">
        The 6-month growth projection combines multiple factors. Adjust weights below to see how different
        emphasis affects the model. (Note: Changes here are for visualization only - actual scoring uses default weights.)
      </p>

      <WeightSlider
        label="Momentum"
        value={weights.momentum}
        onChange={(v) => updateWeight('momentum', v)}
        description="Price trend extrapolation using 6-month linear regression"
      />
      <WeightSlider
        label="Earnings"
        value={weights.earnings}
        onChange={(v) => updateWeight('earnings', v)}
        description="Earnings trajectory from forward estimates or historical growth"
      />
      <WeightSlider
        label="Analyst Targets"
        value={weights.analyst}
        onChange={(v) => updateWeight('analyst', v)}
        description="Consensus price target upside, discounted for 6-month horizon"
      />
      <WeightSlider
        label="Valuation"
        value={weights.valuation}
        onChange={(v) => updateWeight('valuation', v)}
        description="PEG ratio analysis - rewards low P/E relative to growth"
      />
      <WeightSlider
        label="CANSLIM Score"
        value={weights.canslim}
        onChange={(v) => updateWeight('canslim', v)}
        description="Higher CANSLIM scores correlate with better performance"
      />
      <WeightSlider
        label="Sector Momentum"
        value={weights.sector}
        onChange={(v) => updateWeight('sector', v)}
        description="Stocks in leading sectors get a boost"
      />

      <div className={`mt-4 pt-4 border-t border-dark-700 flex justify-between items-center ${isValid ? 'text-green-400' : 'text-red-400'}`}>
        <span className="font-medium">Total Weight</span>
        <span className="text-lg font-bold">{totalWeight}%</span>
      </div>
      {!isValid && (
        <p className="text-red-400 text-xs mt-1">Weights should sum to 100%</p>
      )}
    </div>
  )
}

export default function Documentation() {
  const canslimScores = [
    {
      letter: 'C',
      title: 'Current Quarterly Earnings',
      maxPoints: 15,
      color: 'bg-blue-500/20 text-blue-400',
      description: "Measures current earnings momentum using TTM (Trailing Twelve Months) comparison plus acceleration bonus. O'Neil recommends 25%+ quarterly EPS growth. Thresholds are sector-adjusted.",
      factors: [
        { condition: 'TTM EPS growth >= excellent (sector-adjusted)', points: '10 pts' },
        { condition: 'TTM EPS growth >= good (sector-adjusted)', points: 'Scaled 5-10 pts' },
        { condition: 'EPS accelerating (current Q > prior Q growth)', points: '+5 pts bonus' },
        { condition: 'Positive earnings surprise', points: '+2 pts bonus' },
      ]
    },
    {
      letter: 'A',
      title: 'Annual Earnings Growth',
      maxPoints: 15,
      color: 'bg-green-500/20 text-green-400',
      description: "Long-term earnings growth measured by 3-year CAGR plus ROE quality check. O'Neil recommends 25%+ annual growth and 17%+ ROE.",
      factors: [
        { condition: '3-year EPS CAGR >= 25%', points: '12 pts' },
        { condition: '3-year EPS CAGR 0-25%', points: 'Scaled 0-12 pts' },
        { condition: 'ROE >= 25%', points: '+3 pts bonus' },
        { condition: 'ROE >= 17% (O\'Neil threshold)', points: '+2.1 pts bonus' },
        { condition: 'ROE >= 10%', points: '+0.9 pts bonus' },
      ]
    },
    {
      letter: 'N',
      title: 'New Highs',
      maxPoints: 15,
      color: 'bg-yellow-500/20 text-yellow-400',
      description: "Stocks making new highs tend to continue higher. Measures proximity to 52-week high with volume confirmation for breakouts.",
      factors: [
        { condition: 'Within 5% of 52-week high', points: '12 pts' },
        { condition: 'Within 10% of high', points: '9 pts' },
        { condition: 'Within 15% of high', points: '6 pts' },
        { condition: 'Volume >= 1.5x average (near high)', points: '+3 pts bonus' },
        { condition: 'Volume >= 1.2x average (near high)', points: '+1.8 pts bonus' },
      ]
    },
    {
      letter: 'S',
      title: 'Supply & Demand',
      maxPoints: 15,
      color: 'bg-orange-500/20 text-orange-400',
      description: "Measures buying pressure through volume analysis and price trend. High volume with rising prices indicates institutional accumulation.",
      factors: [
        { condition: 'Volume >= 2x 50-day average', points: '8 pts' },
        { condition: 'Volume >= 1.5x average', points: '6 pts' },
        { condition: 'Volume >= 1.2x average', points: '4 pts' },
        { condition: 'Rising price trend + high volume', points: '+7 pts' },
        { condition: 'Rising price trend', points: '+4 pts' },
      ]
    },
    {
      letter: 'L',
      title: 'Leader vs Laggard',
      maxPoints: 15,
      color: 'bg-purple-500/20 text-purple-400',
      description: "Relative strength vs S&P 500. Uses multi-timeframe analysis (60% 12-month + 40% 3-month) to identify consistent outperformers with recent momentum.",
      factors: [
        { condition: 'Weighted RS >= 1.5 (50%+ outperformance)', points: '15 pts' },
        { condition: 'Weighted RS >= 1.3', points: '13.5 pts' },
        { condition: 'Weighted RS >= 1.15', points: '11.25 pts' },
        { condition: 'Weighted RS >= 1.0 (market performer)', points: '8.25 pts' },
        { condition: 'RS trend improving (3mo > 12mo)', points: '+1 pt bonus' },
      ]
    },
    {
      letter: 'I',
      title: 'Institutional Ownership',
      maxPoints: 10,
      color: 'bg-pink-500/20 text-pink-400',
      description: "Institutional sponsorship validates the stock but too much ownership means the \"smart money\" is already in. Sweet spot is 25-75%.",
      factors: [
        { condition: 'Ownership 25-75% (ideal)', points: '10 pts' },
        { condition: 'Ownership 15-25% or 75-85%', points: '7 pts' },
        { condition: 'Ownership < 15% (too little interest)', points: '3 pts' },
        { condition: 'Ownership > 85% (too crowded)', points: '4 pts' },
        { condition: 'No data available', points: '5 pts (neutral)' },
      ]
    },
    {
      letter: 'M',
      title: 'Market Direction',
      maxPoints: 15,
      color: 'bg-cyan-500/20 text-cyan-400',
      description: "The market determines 75% of a stock's move. Uses a weighted analysis of 3 major indexes (SPY 50%, QQQ 30%, DIA 20%) to gauge overall market health.",
      factors: [
        { condition: 'Weighted signal ≥ 1.5 (strong bullish)', points: '15 pts' },
        { condition: 'Weighted signal ≥ 0.5 (bullish)', points: '12 pts' },
        { condition: 'Weighted signal ≥ 0 (neutral-bullish)', points: '9 pts' },
        { condition: 'Weighted signal ≥ -0.5 (neutral-bearish)', points: '5 pts' },
        { condition: 'Weighted signal < -0.5 (bearish)', points: '2 pts' },
      ]
    },
  ]

  return (
    <div className="p-4 pb-24">
      <h1 className="text-xl font-bold mb-2">Documentation</h1>
      <p className="text-dark-400 text-sm mb-6">
        Understanding the CANSLIM scoring methodology and growth projections
      </p>

      <div className="card mb-4 bg-primary-500/10 border border-primary-500/30">
        <h2 className="font-semibold text-primary-400 mb-2">What is CANSLIM?</h2>
        <p className="text-dark-300 text-sm">
          CANSLIM is an investment strategy developed by William O'Neil, founder of Investor's Business Daily.
          It identifies growth stocks with strong fundamentals and technical characteristics before major price moves.
          The acronym represents 7 key criteria that historically correlate with winning stocks.
        </p>
      </div>

      <div className="card mb-4 bg-cyan-500/10 border border-cyan-500/30">
        <h2 className="font-semibold text-cyan-400 mb-2">Market Direction (Dashboard)</h2>
        <p className="text-dark-300 text-sm mb-3">
          The Market Direction panel shows overall market health using a weighted analysis of three major indexes.
          O'Neil emphasized that 75% of stocks follow the general market direction, making this crucial for timing.
        </p>

        <h3 className="font-medium text-dark-200 text-sm mt-4 mb-2">Three-Index Weighted Analysis</h3>
        <div className="grid grid-cols-3 gap-2 mb-4 text-sm">
          <div className="bg-dark-800 rounded p-2 text-center">
            <div className="font-bold text-blue-400">SPY</div>
            <div className="text-dark-400 text-xs">S&P 500</div>
            <div className="text-dark-300 font-medium">50%</div>
          </div>
          <div className="bg-dark-800 rounded p-2 text-center">
            <div className="font-bold text-purple-400">QQQ</div>
            <div className="text-dark-400 text-xs">NASDAQ 100</div>
            <div className="text-dark-300 font-medium">30%</div>
          </div>
          <div className="bg-dark-800 rounded p-2 text-center">
            <div className="font-bold text-green-400">DIA</div>
            <div className="text-dark-400 text-xs">Dow Jones</div>
            <div className="text-dark-300 font-medium">20%</div>
          </div>
        </div>

        <h3 className="font-medium text-dark-200 text-sm mt-4 mb-2">Index Signal Calculation</h3>
        <p className="text-dark-400 text-sm mb-2">Each index receives a signal based on its position relative to moving averages:</p>
        <div className="space-y-1 text-sm mb-4">
          <div className="flex items-center gap-2">
            <span className="text-green-400 font-bold w-8">▲▲</span>
            <span className="text-dark-300">+2</span>
            <span className="text-dark-400">Above both 50 & 200 MA (strong bullish)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-green-400 font-bold w-8">▲</span>
            <span className="text-dark-300">+1</span>
            <span className="text-dark-400">Above 200 MA, below 50 MA (bullish)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-yellow-400 font-bold w-8">►</span>
            <span className="text-dark-300">0</span>
            <span className="text-dark-400">Below 200 MA, above 50 MA (recovery)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-red-400 font-bold w-8">▼</span>
            <span className="text-dark-300">-1</span>
            <span className="text-dark-400">Below both MAs (bearish)</span>
          </div>
        </div>

        <h3 className="font-medium text-dark-200 text-sm mt-4 mb-2">Weighted Signal Formula</h3>
        <div className="bg-dark-800 rounded p-3 text-sm font-mono text-dark-300 mb-3">
          Weighted Signal = (SPY × 0.50) + (QQQ × 0.30) + (DIA × 0.20)
        </div>
        <p className="text-dark-400 text-sm">
          The weighted signal ranges from -1 (all bearish) to +2 (all strong bullish).
          This combined score is then converted to the M Score (0-15 points).
        </p>

        <h3 className="font-medium text-dark-200 text-sm mt-4 mb-2">Why Three Indexes?</h3>
        <ul className="text-dark-400 text-sm space-y-1 list-disc list-inside">
          <li><span className="text-blue-400">SPY</span> - Broad market exposure (500 largest companies)</li>
          <li><span className="text-purple-400">QQQ</span> - Tech/growth indicator (NASDAQ 100)</li>
          <li><span className="text-green-400">DIA</span> - Blue chip stability (Dow Jones 30)</li>
        </ul>
        <p className="text-dark-400 text-sm mt-2">
          This approach provides a more comprehensive view than SPY alone, capturing both growth momentum (QQQ) and value stability (DIA).
        </p>
      </div>

      <h2 className="text-lg font-semibold mb-3">Scoring Breakdown</h2>
      <p className="text-dark-400 text-sm mb-4">
        Total score is out of 100 points. Tap each criterion to see detailed scoring logic.
      </p>

      {canslimScores.map(score => (
        <ScoreCard key={score.letter} {...score} />
      ))}

      <div className="my-6 border-t border-dark-700" />

      <div className="card mb-4 bg-purple-500/10 border border-purple-500/30">
        <h2 className="font-semibold text-purple-400 mb-2">Growth Mode Scoring</h2>
        <p className="text-dark-300 text-sm mb-3">
          Pre-revenue and high-growth stocks (50%+ revenue growth, &lt;1% profit margin) use an alternative scoring system
          since traditional CANSLIM earnings metrics don't apply. Look for the purple "Growth" badge.
        </p>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-dark-800 rounded p-2">
            <div className="text-purple-400 font-semibold">R - Revenue (20 pts)</div>
            <div className="text-dark-400 text-xs">Year-over-year revenue growth rate</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-purple-400 font-semibold">F - Funding (15 pts)</div>
            <div className="text-dark-400 text-xs">Cash runway, institutional backing</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-purple-400 font-semibold">N - New Highs (15 pts)</div>
            <div className="text-dark-400 text-xs">Same as CANSLIM</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-purple-400 font-semibold">S - Supply (15 pts)</div>
            <div className="text-dark-400 text-xs">Volume surge patterns</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-purple-400 font-semibold">L - Leader (15 pts)</div>
            <div className="text-dark-400 text-xs">Relative strength vs market</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-purple-400 font-semibold">I - Institutional (10 pts)</div>
            <div className="text-dark-400 text-xs">Same as CANSLIM</div>
          </div>
          <div className="bg-dark-800 rounded p-2 col-span-2">
            <div className="text-purple-400 font-semibold">M - Market (10 pts)</div>
            <div className="text-dark-400 text-xs">Same as CANSLIM (reduced weight for growth stocks)</div>
          </div>
        </div>
      </div>

      <div className="my-6 border-t border-dark-700" />

      <div className="card mb-4 bg-purple-500/10 border border-purple-500/30">
        <h2 className="font-semibold text-purple-400 mb-2">Coiled Spring Alerts</h2>
        <p className="text-dark-300 text-sm mb-3">
          "Coiled Spring" stocks have stored energy ready to release on an earnings catalyst. These are stocks with
          long consolidation patterns, consistent earnings beats, and low institutional ownership approaching earnings.
          Think of a spring being compressed - the longer the base, the more explosive the potential move.
        </p>

        <h3 className="font-medium text-dark-200 text-sm mt-4 mb-2">Qualification Criteria (ALL must pass)</h3>
        <div className="space-y-2 text-sm mb-4">
          <div className="flex justify-between items-center bg-dark-800 rounded p-2">
            <span className="text-dark-300">Weeks in base</span>
            <span className="text-purple-400 font-medium">&ge; 15 weeks</span>
          </div>
          <div className="flex justify-between items-center bg-dark-800 rounded p-2">
            <span className="text-dark-300">Earnings beat streak</span>
            <span className="text-purple-400 font-medium">&ge; 3 consecutive</span>
          </div>
          <div className="flex justify-between items-center bg-dark-800 rounded p-2">
            <span className="text-dark-300">C Score (current earnings)</span>
            <span className="text-purple-400 font-medium">&ge; 12 / 15</span>
          </div>
          <div className="flex justify-between items-center bg-dark-800 rounded p-2">
            <span className="text-dark-300">Total CANSLIM score</span>
            <span className="text-purple-400 font-medium">&ge; 55 / 100</span>
          </div>
          <div className="flex justify-between items-center bg-dark-800 rounded p-2">
            <span className="text-dark-300">L Score (relative strength)</span>
            <span className="text-purple-400 font-medium">&ge; 8 / 15</span>
          </div>
          <div className="flex justify-between items-center bg-dark-800 rounded p-2">
            <span className="text-dark-300">Institutional ownership</span>
            <span className="text-purple-400 font-medium">&le; 50%</span>
          </div>
          <div className="flex justify-between items-center bg-dark-800 rounded p-2">
            <span className="text-dark-300">Days to earnings</span>
            <span className="text-purple-400 font-medium">1 - 14 days</span>
          </div>
        </div>

        <h3 className="font-medium text-dark-200 text-sm mt-4 mb-2">Bonus Scoring</h3>
        <p className="text-dark-400 text-sm mb-2">
          Stocks that qualify receive bonus points added to their composite score:
        </p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">Base bonus (qualifies)</span>
            <span className="text-green-400">+20 points</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Long base bonus (20+ weeks)</span>
            <span className="text-green-400">+10 points</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Strong beat bonus (5+ consecutive)</span>
            <span className="text-green-400">+5 points</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Maximum CS bonus</span>
            <span className="text-dark-300">35 points (capped)</span>
          </div>
        </div>

        <h3 className="font-medium text-dark-200 text-sm mt-4 mb-2">Why This Works</h3>
        <ul className="text-dark-400 text-sm space-y-1 list-disc list-inside">
          <li><span className="text-purple-400">Long base</span> = institutions quietly accumulating, price coiled</li>
          <li><span className="text-purple-400">Beat streak</span> = management executing, likely to beat again</li>
          <li><span className="text-purple-400">Low institutional</span> = room for big buyers to pile in post-earnings</li>
          <li><span className="text-purple-400">Strong RS</span> = already outperforming, momentum on its side</li>
        </ul>
        <p className="text-dark-500 text-xs mt-3">
          The CS alerts card appears on the Dashboard and AI Portfolio when qualifying stocks are found.
          These are pre-earnings plays - the goal is to identify explosive setups BEFORE they move.
        </p>
      </div>

      <div className="my-6 border-t border-dark-700" />

      <h2 className="text-lg font-semibold mb-3">Technical Analysis</h2>
      <p className="text-dark-400 text-sm mb-4">
        O'Neil emphasized buying stocks breaking out of proper base patterns. The app detects these patterns automatically.
      </p>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Base Patterns</h3>
        <p className="text-dark-400 text-sm mb-3">
          A "base" is a period of consolidation (5-15 weeks) where a stock digests gains before its next move up.
          Proper bases show institutional accumulation (tight price action, declining volume on pullbacks).
        </p>
        <div className="space-y-3 text-sm">
          <div className="bg-dark-800 rounded p-3">
            <div className="flex justify-between items-center mb-1">
              <span className="font-semibold text-yellow-400">Flat Base</span>
              <span className="text-dark-400">5+ weeks, &lt;15% range</span>
            </div>
            <p className="text-dark-400 text-xs">Tight consolidation pattern showing strong support. Most common pattern.</p>
          </div>
          <div className="bg-dark-800 rounded p-3">
            <div className="flex justify-between items-center mb-1">
              <span className="font-semibold text-blue-400">Cup with Handle</span>
              <span className="text-dark-400">7-65 weeks</span>
            </div>
            <p className="text-dark-400 text-xs">U-shaped recovery with small pullback (handle) before breakout. Classic O'Neil pattern.</p>
          </div>
          <div className="bg-dark-800 rounded p-3">
            <div className="flex justify-between items-center mb-1">
              <span className="font-semibold text-green-400">Double Bottom</span>
              <span className="text-dark-400">7+ weeks</span>
            </div>
            <p className="text-dark-400 text-xs">W-shaped pattern with two lows within 3% of each other. Shows support validation.</p>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Pivot Points & Entry Timing</h3>
        <p className="text-dark-400 text-sm mb-3">
          The "pivot point" is the optimal buy price - the point where a stock breaks out of its base pattern.
          Entry timing relative to the pivot dramatically affects returns.
        </p>
        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-3 bg-green-500/10 rounded p-2">
            <span className="text-green-400 font-bold text-lg">+30</span>
            <div>
              <div className="text-green-400 font-medium">Pre-Breakout (5-15% below pivot)</div>
              <div className="text-dark-400 text-xs">Best entry - before the crowd notices. 30% larger position size.</div>
            </div>
          </div>
          <div className="flex items-center gap-3 bg-blue-500/10 rounded p-2">
            <span className="text-blue-400 font-bold text-lg">+25</span>
            <div>
              <div className="text-blue-400 font-medium">At Pivot (0-5% below)</div>
              <div className="text-dark-400 text-xs">Optimal timing - coiled for breakout. 20% larger position size.</div>
            </div>
          </div>
          <div className="flex items-center gap-3 bg-yellow-500/10 rounded p-2">
            <span className="text-yellow-400 font-bold text-lg">+20</span>
            <div>
              <div className="text-yellow-400 font-medium">Breakout (0-5% above pivot)</div>
              <div className="text-dark-400 text-xs">Confirmed move but slightly extended. 15% larger position size.</div>
            </div>
          </div>
          <div className="flex items-center gap-3 bg-red-500/10 rounded p-2">
            <span className="text-red-400 font-bold text-lg">-20</span>
            <div>
              <div className="text-red-400 font-medium">Extended (&gt;10% above pivot)</div>
              <div className="text-dark-400 text-xs">Chasing - higher risk of pullback. Avoid buying.</div>
            </div>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Volume Confirmation</h3>
        <p className="text-dark-400 text-sm mb-3">
          Volume validates price moves. Breakouts on high volume (1.5x+ average) are more likely to succeed.
        </p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">Volume ratio &ge; 2.5x</span>
            <span className="text-green-400">Strong confirmation (+15 bonus)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Volume ratio &ge; 2.0x</span>
            <span className="text-green-400">Good confirmation (+10 bonus)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Volume ratio &ge; 1.5x</span>
            <span className="text-yellow-400">Moderate confirmation (+5 bonus)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Volume ratio &lt; 1.5x</span>
            <span className="text-dark-500">Weak - proceed with caution</span>
          </div>
        </div>
      </div>

      <div className="my-6 border-t border-dark-700" />

      <h2 className="text-lg font-semibold mb-3">AI Trader Logic</h2>
      <p className="text-dark-400 text-sm mb-4">
        The AI Portfolio automatically buys and sells based on CANSLIM principles. Here's how it makes decisions.
      </p>

      <div className="card mb-4 bg-green-500/10 border border-green-500/30">
        <h3 className="font-semibold text-green-400 mb-3">Buy Signals</h3>
        <p className="text-dark-300 text-sm mb-3">
          The AI calculates a composite score (0-100) for each stock, weighing multiple factors:
        </p>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">Projected Growth (6-month)</span>
            <span className="text-dark-300">25% weight</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">CANSLIM/Growth Mode Score</span>
            <span className="text-dark-300">25% weight</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Momentum (RS ratio)</span>
            <span className="text-dark-300">20% weight</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Entry Timing (pivot proximity)</span>
            <span className="text-dark-300">20% weight</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Base Pattern Quality</span>
            <span className="text-dark-300">10% weight</span>
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-dark-700">
          <div className="text-dark-400 text-xs">
            <strong>Minimum requirements:</strong> Score &ge; 65, Composite &ge; 55, &lt; 20 positions, &lt; 20% per sector
          </div>
        </div>
      </div>

      <div className="card mb-4 bg-red-500/10 border border-red-500/30">
        <h3 className="font-semibold text-red-400 mb-3">Sell Signals</h3>
        <div className="space-y-3 text-sm">
          <div className="bg-dark-800 rounded p-2">
            <div className="text-red-400 font-medium">Stop Loss (-8%)</div>
            <div className="text-dark-400 text-xs">O'Neil's cardinal rule - cut losses before they grow. Automatic exit.</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-red-400 font-medium">Trailing Stop (8-15%)</div>
            <div className="text-dark-400 text-xs">Locks in gains: 15% stop at 50%+ gain, 12% at 30%+, 10% at 20%+, 8% at 10%+</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-red-400 font-medium">Score Crash (2 consecutive scans)</div>
            <div className="text-dark-400 text-xs">Score drops below 50 AND drops 20+ points for 2+ scans. Avoids single-blip sells.</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-orange-400 font-medium">Partial Profit Taking</div>
            <div className="text-dark-400 text-xs">Sells 25% at +25% gain, 50% at +40% gain (if score &ge; 60). Lets winners run.</div>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Additional Signals</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between items-center">
            <div>
              <span className="text-dark-300">Insider Buying (CEO/CFO)</span>
              <div className="text-dark-500 text-xs">Bullish when insiders are buying</div>
            </div>
            <span className="text-green-400">+5 bonus</span>
          </div>
          <div className="flex justify-between items-center">
            <div>
              <span className="text-dark-300">Insider Selling</span>
              <div className="text-dark-500 text-xs">Sells exceed buys by 1.5x+</div>
            </div>
            <span className="text-red-400">-3 penalty</span>
          </div>
          <div className="flex justify-between items-center">
            <div>
              <span className="text-dark-300">High Short Interest (&gt;20%)</span>
              <div className="text-dark-500 text-xs">Elevated risk of volatility</div>
            </div>
            <span className="text-red-400">-5 penalty</span>
          </div>
          <div className="flex justify-between items-center">
            <div>
              <span className="text-dark-300">Momentum Fading</span>
              <div className="text-dark-500 text-xs">3-month RS &lt; 95% of 12-month RS</div>
            </div>
            <span className="text-red-400">-15% to composite</span>
          </div>
        </div>
      </div>

      <div className="my-6 border-t border-dark-700" />

      <h2 className="text-lg font-semibold mb-3">Sector-Adjusted Scoring</h2>
      <p className="text-dark-400 text-sm mb-4">
        Not all sectors grow at the same rate. A 20% EPS growth is excellent for Industrials but mediocre for Technology.
        The app adjusts expectations by sector.
      </p>

      <div className="card mb-4">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-dark-800 rounded p-2">
            <div className="text-blue-400 font-semibold">Technology</div>
            <div className="text-dark-400 text-xs">Excellent: 30% | Good: 20%</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-pink-400 font-semibold">Healthcare</div>
            <div className="text-dark-400 text-xs">Excellent: 25% | Good: 15%</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-yellow-400 font-semibold">Industrials</div>
            <div className="text-dark-400 text-xs">Excellent: 20% | Good: 12%</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-green-400 font-semibold">Utilities</div>
            <div className="text-dark-400 text-xs">Excellent: 12% | Good: 8%</div>
          </div>
        </div>
        <p className="text-dark-500 text-xs mt-2">Other sectors use default thresholds: Excellent 25%, Good 15%</p>
      </div>

      <div className="my-6 border-t border-dark-700" />

      <h2 className="text-lg font-semibold mb-3">Growth Projection Model</h2>
      <p className="text-dark-400 text-sm mb-4">
        The 6-month growth projection uses a weighted combination of factors to estimate potential upside.
        Higher confidence projections have more data points supporting them.
      </p>

      <GrowthProjectionSection />

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Valuation Factor (NEW)</h3>
        <p className="text-dark-400 text-sm mb-3">
          The valuation factor uses PEG-style analysis to identify undervalued growth stocks:
        </p>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">PEG &lt; 0.5 (deeply undervalued)</span>
            <span className="text-green-400">+30%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">PEG &lt; 1.0 (classic buy signal)</span>
            <span className="text-green-400">+20%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">PEG 1.0-1.5 (fairly valued)</span>
            <span className="text-yellow-400">+5%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">PEG &gt; 2.0 (overvalued)</span>
            <span className="text-red-400">-15%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Earnings yield &gt; 5%</span>
            <span className="text-green-400">+5% bonus</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">FCF yield &gt; 5%</span>
            <span className="text-green-400">+5% bonus</span>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Confidence Levels</h3>
        <p className="text-dark-400 text-sm mb-3">
          Projection confidence is based on data quality and signal consistency:
        </p>
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <span className="px-2 py-1 rounded bg-green-500/20 text-green-400 text-xs font-medium">HIGH</span>
            <span className="text-sm text-dark-300">Strong analyst coverage, full historical data, high CANSLIM score</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="px-2 py-1 rounded bg-yellow-500/20 text-yellow-400 text-xs font-medium">MEDIUM</span>
            <span className="text-sm text-dark-300">Adequate data but some factors missing or mixed signals</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="px-2 py-1 rounded bg-red-500/20 text-red-400 text-xs font-medium">LOW</span>
            <span className="text-sm text-dark-300">Limited data, few analysts, or conflicting indicators</span>
          </div>
        </div>
      </div>

      <div className="my-6 border-t border-dark-700" />

      <h2 className="text-lg font-semibold mb-3">Portfolio Recommendations</h2>
      <p className="text-dark-400 text-sm mb-4">
        The BUY/HOLD/SELL recommendations for portfolio positions use a weighted signal system analyzing 5 factors.
      </p>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Factor 1: CANSLIM Score</h3>
        <p className="text-dark-400 text-sm mb-2">Strong influence on the recommendation:</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">Score &ge; 70</span>
            <span className="text-green-400">+2 buy signals</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Score 50-69</span>
            <span className="text-yellow-400">+1 hold signal</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Score 35-49</span>
            <span className="text-yellow-400">+1 hold signal</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Score &lt; 35</span>
            <span className="text-red-400">+2 sell signals</span>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Factor 2: Projected Growth</h3>
        <p className="text-dark-400 text-sm mb-2">6-month growth projection impact:</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">&ge; 20% projected</span>
            <span className="text-green-400">+2 buy signals</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">10-20% projected</span>
            <span className="text-green-400">+1 buy signal</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">0-10% projected</span>
            <span className="text-yellow-400">+1 hold signal</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">-10% to 0%</span>
            <span className="text-red-400">+1 sell signal</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">&lt; -10% projected</span>
            <span className="text-red-400">+2 sell signals</span>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Factor 3: Current Gain/Loss</h3>
        <p className="text-dark-400 text-sm mb-2">Your position's performance:</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">Large loss (&lt; -20%) + weak score</span>
            <span className="text-red-400">+2 sell signals</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Moderate loss (-10% to -20%) + weak outlook</span>
            <span className="text-red-400">+1 sell signal</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Most other gain/loss scenarios</span>
            <span className="text-yellow-400">+1 hold signal</span>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Factor 4: Analyst Sentiment</h3>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">Analyst upside &ge; 30%</span>
            <span className="text-green-400">+1 buy signal</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Analyst downside &le; -10%</span>
            <span className="text-red-400">+1 sell signal</span>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Factor 5: 52-Week High Proximity</h3>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">Within 5% of 52-week high</span>
            <span className="text-green-400">+1 buy signal (momentum)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">40%+ below high + good score</span>
            <span className="text-green-400">+1 buy signal (value)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">40%+ below high + weak score</span>
            <span className="text-red-400">+1 sell signal</span>
          </div>
        </div>
      </div>

      <div className="card mb-4 bg-dark-700/50">
        <h3 className="font-semibold mb-3">Final Decision Logic</h3>
        <div className="space-y-2 text-sm">
          <div className="flex items-start gap-2">
            <span className="px-2 py-1 rounded bg-red-500/20 text-red-400 text-xs font-medium">SELL</span>
            <span className="text-dark-300">4+ sell signals, OR 3+ sells with 0 buys, OR sells &gt; buys + 1</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="px-2 py-1 rounded bg-green-500/20 text-green-400 text-xs font-medium">BUY MORE</span>
            <span className="text-dark-300">4+ buy signals, OR 3+ buys with 0 sells, OR buys &gt; sells + 1</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="px-2 py-1 rounded bg-yellow-500/20 text-yellow-400 text-xs font-medium">HOLD</span>
            <span className="text-dark-300">Everything else - mixed signals or balanced outlook</span>
          </div>
        </div>
      </div>

      <div className="my-6 border-t border-dark-700" />

      <div className="my-6 border-t border-dark-700" />

      <h2 className="text-lg font-semibold mb-3">Using the App</h2>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Stock Universe</h3>
        <p className="text-dark-400 text-sm mb-3">
          The app scans ~2,000+ stocks from major indexes. Your portfolio stocks are always scanned first.
        </p>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-dark-800 rounded p-2">
            <div className="text-blue-400 font-semibold">S&P 500</div>
            <div className="text-dark-400 text-xs">~500 large-cap stocks</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-green-400 font-semibold">S&P MidCap 400</div>
            <div className="text-dark-400 text-xs">~400 mid-cap stocks</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-yellow-400 font-semibold">S&P SmallCap 600</div>
            <div className="text-dark-400 text-xs">~600 small-cap stocks</div>
          </div>
          <div className="bg-dark-800 rounded p-2">
            <div className="text-purple-400 font-semibold">Russell 2000</div>
            <div className="text-dark-400 text-xs">~1,200 curated small-caps</div>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Breaking Out Page</h3>
        <p className="text-dark-400 text-sm mb-3">
          Shows stocks with active base patterns that are near their pivot point. These are the best opportunities for timely entries.
        </p>
        <div className="text-sm space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-green-400">&#x2022;</span>
            <span className="text-dark-300">Pre-breakout: 0-5% below pivot (building for breakout)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-blue-400">&#x2022;</span>
            <span className="text-dark-300">Breaking out: 0-5% above pivot (confirmed, still buyable)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-dark-500">&#x2022;</span>
            <span className="text-dark-400">Stocks &gt;5% above pivot are excluded (too extended)</span>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Backtesting</h3>
        <p className="text-dark-400 text-sm mb-3">
          Test the strategy against historical data to validate performance. Access via AI Portfolio → "Run Historical Backtest".
        </p>
        <div className="text-sm space-y-1">
          <div className="flex justify-between">
            <span className="text-dark-400">Timeframe</span>
            <span className="text-dark-300">1 year historical simulation</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Benchmark</span>
            <span className="text-dark-300">SPY (S&P 500 ETF)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Metrics</span>
            <span className="text-dark-300">Total return, max drawdown, Sharpe ratio, win rate</span>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <h3 className="font-semibold mb-3">Key O'Neil Principles</h3>
        <ul className="text-dark-400 text-sm space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-green-400 mt-1">&#x2713;</span>
            <span><strong className="text-dark-200">Buy leaders, not laggards.</strong> Focus on stocks with Relative Strength &ge; 80 (top 20%).</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-400 mt-1">&#x2713;</span>
            <span><strong className="text-dark-200">Cut losses at 7-8%.</strong> Small losses are recoverable; big losses require outsized gains.</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-400 mt-1">&#x2713;</span>
            <span><strong className="text-dark-200">Let winners run.</strong> Use trailing stops to protect gains while staying in strong trends.</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-400 mt-1">&#x2713;</span>
            <span><strong className="text-dark-200">Buy on proper bases.</strong> Wait for consolidation patterns before buying - don't chase extended stocks.</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-400 mt-1">&#x2713;</span>
            <span><strong className="text-dark-200">Follow the market.</strong> 3 out of 4 stocks follow the general market direction.</span>
          </li>
        </ul>
      </div>

      <div className="card bg-dark-700/50">
        <h3 className="font-semibold mb-2">Data Sources</h3>
        <ul className="text-dark-400 text-sm space-y-1">
          <li>• <strong className="text-dark-200">Financial Modeling Prep:</strong> Earnings, ROE, key metrics, analyst targets, insider trading</li>
          <li>• <strong className="text-dark-200">Yahoo Finance:</strong> Price history, volume data, short interest</li>
          <li>• <strong className="text-dark-200">Finviz:</strong> Institutional ownership percentage</li>
        </ul>
        <p className="text-dark-500 text-xs mt-3">
          Data is refreshed every 90 minutes during market hours. Earnings data cached for 24 hours. Institutional data cached for 7 days.
        </p>
      </div>

      <div className="h-8" />
    </div>
  )
}

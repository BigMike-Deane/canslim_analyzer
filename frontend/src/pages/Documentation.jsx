import { useState } from 'react'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { TagBadge } from '../components/Badge'
import { StatRow } from '../components/StatGrid'
import PageHeader from '../components/PageHeader'

/* ─── Collapsible Score Card ──────────────────────────────────────── */

function ScoreCard({ letter, title, maxPoints, description, factors, color }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <Card variant="glass" className="mb-3" onClick={() => setExpanded(!expanded)}>
      <div className="flex justify-between items-center cursor-pointer">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl ${color} flex items-center justify-center font-bold text-lg`}>
            {letter}
          </div>
          <div>
            <div className="font-semibold text-dark-50">{title}</div>
            <div className="text-dark-400 text-xs font-data">{maxPoints} points max</div>
          </div>
        </div>
        <svg
          width="16" height="16" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round"
          className={`text-dark-400 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-dark-700/50">
          <p className="text-dark-300 text-sm mb-3">{description}</p>
          <div className="space-y-1.5">
            {factors.map((factor, i) => (
              <StatRow key={i} label={factor.condition} value={factor.points} />
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}

/* ─── Weight Slider ───────────────────────────────────────────────── */

function WeightSlider({ label, value, onChange, description }) {
  return (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-medium text-dark-200">{label}</span>
        <span className="text-sm font-data text-primary-400">{value}%</span>
      </div>
      <input
        type="range"
        min="0"
        max="40"
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider"
      />
      <p className="text-[10px] text-dark-400 mt-1">{description}</p>
    </div>
  )
}

/* ─── Growth Projection Section ───────────────────────────────────── */

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
    <Card variant="glass" className="mb-4">
      <CardHeader title="Growth Projection Weights" />
      <p className="text-dark-400 text-xs mb-4">
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

      <div className={`mt-4 pt-4 border-t border-dark-700/50 flex justify-between items-center ${isValid ? 'text-emerald-400' : 'text-red-400'}`}>
        <span className="font-medium text-sm">Total Weight</span>
        <span className="text-lg font-bold font-data">{totalWeight}%</span>
      </div>
      {!isValid && (
        <p className="text-red-400 text-[10px] mt-1">Weights should sum to 100%</p>
      )}
    </Card>
  )
}

/* ─── Main Documentation Page ─────────────────────────────────────── */

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
      color: 'bg-emerald-500/20 text-emerald-400',
      description: "Long-term earnings growth measured by 3-year CAGR plus ROE quality check. O'Neil recommends 25%+ annual growth and 17%+ ROE.",
      factors: [
        { condition: '3-year EPS CAGR >= 25%', points: '12 pts' },
        { condition: '3-year EPS CAGR 0-25%', points: 'Scaled 0-12 pts' },
        { condition: 'ROE >= 25%', points: '+3 pts bonus' },
        { condition: "ROE >= 17% (O'Neil threshold)", points: '+2.1 pts bonus' },
        { condition: 'ROE >= 10%', points: '+0.9 pts bonus' },
      ]
    },
    {
      letter: 'N',
      title: 'New Highs',
      maxPoints: 15,
      color: 'bg-amber-500/20 text-amber-400',
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
      color: 'bg-primary-500/20 text-primary-400',
      description: "The market determines 75% of a stock's move. Uses a weighted analysis of 3 major indexes (SPY 50%, QQQ 30%, DIA 20%) to gauge overall market health.",
      factors: [
        { condition: 'Weighted signal >= 1.5 (strong bullish)', points: '15 pts' },
        { condition: 'Weighted signal >= 0.5 (bullish)', points: '12 pts' },
        { condition: 'Weighted signal >= 0 (neutral-bullish)', points: '9 pts' },
        { condition: 'Weighted signal >= -0.5 (neutral-bearish)', points: '5 pts' },
        { condition: 'Weighted signal < -0.5 (bearish)', points: '2 pts' },
      ]
    },
  ]

  return (
    <div className="p-4 md:p-6 pb-24">
      <PageHeader
        title="Documentation"
        subtitle="Understanding the CANSLIM scoring methodology and growth projections"
      />

      {/* What is CANSLIM */}
      <Card variant="accent" accent="cyan" className="mb-4">
        <h2 className="font-semibold text-primary-400 text-sm mb-2">What is CANSLIM?</h2>
        <p className="text-dark-300 text-sm">
          CANSLIM is an investment strategy developed by William O'Neil, founder of Investor's Business Daily.
          It identifies growth stocks with strong fundamentals and technical characteristics before major price moves.
          The acronym represents 7 key criteria that historically correlate with winning stocks.
        </p>
      </Card>

      {/* Market Direction */}
      <Card variant="accent" accent="cyan" className="mb-4">
        <h2 className="font-semibold text-primary-400 text-sm mb-2">Market Direction (Dashboard)</h2>
        <p className="text-dark-300 text-sm mb-3">
          The Market Direction panel shows overall market health using a weighted analysis of three major indexes.
          O'Neil emphasized that 75% of stocks follow the general market direction, making this crucial for timing.
        </p>

        <SectionLabel>Three-Index Weighted Analysis</SectionLabel>
        <div className="grid grid-cols-3 gap-2 mb-4">
          <Card variant="stat" padding="p-2" rounded="rounded-lg">
            <div className="text-center">
              <div className="font-bold font-data text-blue-400">SPY</div>
              <div className="text-dark-400 text-[10px]">S&P 500</div>
              <div className="text-dark-300 font-data font-medium text-sm">50%</div>
            </div>
          </Card>
          <Card variant="stat" padding="p-2" rounded="rounded-lg">
            <div className="text-center">
              <div className="font-bold font-data text-purple-400">QQQ</div>
              <div className="text-dark-400 text-[10px]">NASDAQ 100</div>
              <div className="text-dark-300 font-data font-medium text-sm">30%</div>
            </div>
          </Card>
          <Card variant="stat" padding="p-2" rounded="rounded-lg">
            <div className="text-center">
              <div className="font-bold font-data text-emerald-400">DIA</div>
              <div className="text-dark-400 text-[10px]">Dow Jones</div>
              <div className="text-dark-300 font-data font-medium text-sm">20%</div>
            </div>
          </Card>
        </div>

        <SectionLabel>Index Signal Calculation</SectionLabel>
        <p className="text-dark-400 text-xs mb-2">Each index receives a signal based on its position relative to moving averages:</p>
        <div className="space-y-1.5 text-sm mb-4">
          <div className="flex items-center gap-2">
            <span className="text-emerald-400 font-bold font-data w-8">+2</span>
            <span className="text-dark-300">Above both 50 & 200 MA (strong bullish)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-emerald-400 font-bold font-data w-8">+1</span>
            <span className="text-dark-300">Above 200 MA, below 50 MA (bullish)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-amber-400 font-bold font-data w-8">0</span>
            <span className="text-dark-300">Below 200 MA, above 50 MA (recovery)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-red-400 font-bold font-data w-8">-1</span>
            <span className="text-dark-300">Below both MAs (bearish)</span>
          </div>
        </div>

        <SectionLabel>Weighted Signal Formula</SectionLabel>
        <Card variant="stat" padding="p-3" rounded="rounded-lg" className="mb-3">
          <code className="text-sm font-data text-dark-300">
            Weighted Signal = (SPY x 0.50) + (QQQ x 0.30) + (DIA x 0.20)
          </code>
        </Card>
        <p className="text-dark-400 text-xs">
          The weighted signal ranges from -1 (all bearish) to +2 (all strong bullish).
          This combined score is then converted to the M Score (0-15 points).
        </p>

        <SectionLabel className="mt-4">Why Three Indexes?</SectionLabel>
        <ul className="text-dark-400 text-sm space-y-1 list-disc list-inside">
          <li><span className="text-blue-400">SPY</span> - Broad market exposure (500 largest companies)</li>
          <li><span className="text-purple-400">QQQ</span> - Tech/growth indicator (NASDAQ 100)</li>
          <li><span className="text-emerald-400">DIA</span> - Blue chip stability (Dow Jones 30)</li>
        </ul>
        <p className="text-dark-400 text-xs mt-2">
          This approach provides a more comprehensive view than SPY alone, capturing both growth momentum (QQQ) and value stability (DIA).
        </p>
      </Card>

      {/* Scoring Breakdown */}
      <SectionLabel className="mt-6">Scoring Breakdown</SectionLabel>
      <p className="text-dark-400 text-xs mb-4">
        Total score is out of 100 points. Tap each criterion to see detailed scoring logic.
      </p>

      {canslimScores.map(score => (
        <ScoreCard key={score.letter} {...score} />
      ))}

      <div className="my-6 h-px bg-dark-700/50" />

      {/* Growth Mode Scoring */}
      <Card variant="accent" accent="purple" className="mb-4">
        <h2 className="font-semibold text-purple-400 text-sm mb-2">Growth Mode Scoring</h2>
        <p className="text-dark-300 text-sm mb-3">
          Pre-revenue and high-growth stocks (50%+ revenue growth, &lt;1% profit margin) use an alternative scoring system
          since traditional CANSLIM earnings metrics don't apply. Look for the purple "Growth" badge.
        </p>
        <div className="grid grid-cols-2 gap-2 text-sm">
          {[
            { label: 'R - Revenue', pts: '20 pts', desc: 'Year-over-year revenue growth rate' },
            { label: 'F - Funding', pts: '15 pts', desc: 'Cash runway, institutional backing' },
            { label: 'N - New Highs', pts: '15 pts', desc: 'Same as CANSLIM' },
            { label: 'S - Supply', pts: '15 pts', desc: 'Volume surge patterns' },
            { label: 'L - Leader', pts: '15 pts', desc: 'Relative strength vs market' },
            { label: 'I - Institutional', pts: '10 pts', desc: 'Same as CANSLIM' },
          ].map(item => (
            <Card key={item.label} variant="stat" padding="p-2" rounded="rounded-lg">
              <div className="text-purple-400 font-semibold text-xs">{item.label} ({item.pts})</div>
              <div className="text-dark-400 text-[10px]">{item.desc}</div>
            </Card>
          ))}
          <Card variant="stat" padding="p-2" rounded="rounded-lg" className="col-span-2">
            <div className="text-purple-400 font-semibold text-xs">M - Market (10 pts)</div>
            <div className="text-dark-400 text-[10px]">Same as CANSLIM (reduced weight for growth stocks)</div>
          </Card>
        </div>
      </Card>

      <div className="my-6 h-px bg-dark-700/50" />

      {/* Coiled Spring Alerts */}
      <Card variant="accent" accent="purple" className="mb-4">
        <h2 className="font-semibold text-purple-400 text-sm mb-2">Coiled Spring Alerts</h2>
        <p className="text-dark-300 text-sm mb-3">
          "Coiled Spring" stocks have stored energy ready to release on an earnings catalyst. These are stocks with
          long consolidation patterns, consistent earnings beats, and low institutional ownership approaching earnings.
          Think of a spring being compressed - the longer the base, the more explosive the potential move.
        </p>

        <SectionLabel>Qualification Criteria (ALL must pass)</SectionLabel>
        <div className="space-y-1.5 mb-4">
          {[
            { label: 'Weeks in base', value: '>= 15 weeks' },
            { label: 'Earnings beat streak', value: '>= 3 consecutive' },
            { label: 'C Score (current earnings)', value: '>= 12 / 15' },
            { label: 'Total CANSLIM score', value: '>= 55 / 100' },
            { label: 'L Score (relative strength)', value: '>= 8 / 15' },
            { label: 'Institutional ownership', value: '<= 50%' },
            { label: 'Days to earnings', value: '1 - 14 days' },
          ].map(item => (
            <StatRow key={item.label} label={item.label} value={<span className="text-purple-400 font-data">{item.value}</span>} />
          ))}
        </div>

        <SectionLabel>Bonus Scoring</SectionLabel>
        <p className="text-dark-400 text-xs mb-2">
          Stocks that qualify receive bonus points added to their composite score:
        </p>
        <div className="space-y-1.5">
          <StatRow label="Base bonus (qualifies)" value={<span className="text-emerald-400 font-data">+20 points</span>} />
          <StatRow label="Long base bonus (20+ weeks)" value={<span className="text-emerald-400 font-data">+10 points</span>} />
          <StatRow label="Strong beat bonus (5+ consecutive)" value={<span className="text-emerald-400 font-data">+5 points</span>} />
          <StatRow label="Maximum CS bonus" value={<span className="text-dark-300 font-data">35 points (capped)</span>} />
        </div>

        <SectionLabel className="mt-4">Why This Works</SectionLabel>
        <ul className="text-dark-400 text-sm space-y-1 list-disc list-inside">
          <li><span className="text-purple-400">Long base</span> = institutions quietly accumulating, price coiled</li>
          <li><span className="text-purple-400">Beat streak</span> = management executing, likely to beat again</li>
          <li><span className="text-purple-400">Low institutional</span> = room for big buyers to pile in post-earnings</li>
          <li><span className="text-purple-400">Strong RS</span> = already outperforming, momentum on its side</li>
        </ul>
        <p className="text-dark-500 text-[10px] mt-3">
          The CS alerts card appears on the Dashboard and AI Portfolio when qualifying stocks are found.
          These are pre-earnings plays - the goal is to identify explosive setups BEFORE they move.
        </p>
      </Card>

      <div className="my-6 h-px bg-dark-700/50" />

      {/* Technical Analysis */}
      <SectionLabel>Technical Analysis</SectionLabel>
      <p className="text-dark-400 text-xs mb-4">
        O'Neil emphasized buying stocks breaking out of proper base patterns. The app detects these patterns automatically.
      </p>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Base Patterns" />
        <p className="text-dark-300 text-sm mb-3">
          A "base" is a period of consolidation (5-15 weeks) where a stock digests gains before its next move up.
          Proper bases show institutional accumulation (tight price action, declining volume on pullbacks).
        </p>
        <div className="space-y-2">
          {[
            { name: 'Flat Base', color: 'text-amber-400', range: '5+ weeks, <15% range', desc: 'Tight consolidation pattern showing strong support. Most common pattern.' },
            { name: 'Cup with Handle', color: 'text-blue-400', range: '7-65 weeks', desc: "U-shaped recovery with small pullback (handle) before breakout. Classic O'Neil pattern." },
            { name: 'Double Bottom', color: 'text-emerald-400', range: '7+ weeks', desc: 'W-shaped pattern with two lows within 3% of each other. Shows support validation.' },
          ].map(p => (
            <Card key={p.name} variant="stat" padding="p-3" rounded="rounded-lg">
              <div className="flex justify-between items-center mb-1">
                <span className={`font-semibold text-sm ${p.color}`}>{p.name}</span>
                <span className="text-dark-400 text-[10px]">{p.range}</span>
              </div>
              <p className="text-dark-400 text-[10px]">{p.desc}</p>
            </Card>
          ))}
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Pivot Points & Entry Timing" />
        <p className="text-dark-300 text-sm mb-3">
          The "pivot point" is the optimal buy price - the point where a stock breaks out of its base pattern.
          Entry timing relative to the pivot dramatically affects returns.
        </p>
        <div className="space-y-2">
          {[
            { bonus: '+30', label: 'Pre-Breakout (5-15% below pivot)', desc: 'Best entry - before the crowd notices. 30% larger position size.', color: 'emerald' },
            { bonus: '+25', label: 'At Pivot (0-5% below)', desc: 'Optimal timing - coiled for breakout. 20% larger position size.', color: 'blue' },
            { bonus: '+20', label: 'Breakout (0-5% above pivot)', desc: 'Confirmed move but slightly extended. 15% larger position size.', color: 'amber' },
            { bonus: '-20', label: 'Extended (>10% above pivot)', desc: 'Chasing - higher risk of pullback. Avoid buying.', color: 'red' },
          ].map(entry => (
            <Card key={entry.label} variant="stat" padding="p-2.5" rounded="rounded-lg"
              className={`border-l-2 border-l-${entry.color}-500/40`}
            >
              <div className="flex items-center gap-3">
                <span className={`text-${entry.color}-400 font-bold font-data text-lg w-8 shrink-0`}>{entry.bonus}</span>
                <div>
                  <div className={`text-${entry.color}-400 font-medium text-sm`}>{entry.label}</div>
                  <div className="text-dark-400 text-[10px]">{entry.desc}</div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Volume Confirmation" />
        <p className="text-dark-300 text-sm mb-3">
          Volume validates price moves. Breakouts on high volume (1.5x+ average) are more likely to succeed.
        </p>
        <div className="space-y-1.5">
          <StatRow label="Volume ratio >= 2.5x" value={<span className="text-emerald-400 font-data text-xs">Strong confirmation (+15 bonus)</span>} />
          <StatRow label="Volume ratio >= 2.0x" value={<span className="text-emerald-400 font-data text-xs">Good confirmation (+10 bonus)</span>} />
          <StatRow label="Volume ratio >= 1.5x" value={<span className="text-amber-400 font-data text-xs">Moderate confirmation (+5 bonus)</span>} />
          <StatRow label="Volume ratio < 1.5x" value={<span className="text-dark-500 font-data text-xs">Weak - proceed with caution</span>} />
        </div>
      </Card>

      <div className="my-6 h-px bg-dark-700/50" />

      {/* AI Trader Logic */}
      <SectionLabel>AI Trader Logic</SectionLabel>
      <p className="text-dark-400 text-xs mb-4">
        The AI Portfolio automatically buys and sells based on CANSLIM principles. Here's how it makes decisions.
      </p>

      <Card variant="accent" accent="green" className="mb-4">
        <h3 className="font-semibold text-emerald-400 text-sm mb-3">Buy Signals</h3>
        <p className="text-dark-300 text-sm mb-3">
          The AI calculates a composite score (0-100) for each stock, weighing multiple factors:
        </p>
        <div className="space-y-1.5">
          <StatRow label="Projected Growth (6-month)" value={<span className="font-data text-dark-200 text-xs">25% weight</span>} />
          <StatRow label="CANSLIM/Growth Mode Score" value={<span className="font-data text-dark-200 text-xs">25% weight</span>} />
          <StatRow label="Momentum (RS ratio)" value={<span className="font-data text-dark-200 text-xs">20% weight</span>} />
          <StatRow label="Entry Timing (pivot proximity)" value={<span className="font-data text-dark-200 text-xs">20% weight</span>} />
          <StatRow label="Base Pattern Quality" value={<span className="font-data text-dark-200 text-xs">10% weight</span>} />
        </div>
        <div className="mt-3 pt-3 border-t border-dark-700/50">
          <p className="text-dark-400 text-[10px]">
            <strong className="text-dark-200">Minimum requirements:</strong> Score &ge; 65, Composite &ge; 55, &lt; 20 positions, &lt; 20% per sector
          </p>
        </div>
      </Card>

      <Card variant="accent" accent="red" className="mb-4">
        <h3 className="font-semibold text-red-400 text-sm mb-3">Sell Signals</h3>
        <div className="space-y-2">
          {[
            { title: 'Stop Loss (-8%)', desc: "O'Neil's cardinal rule - cut losses before they grow. Automatic exit.", color: 'red' },
            { title: 'Trailing Stop (8-15%)', desc: 'Locks in gains: 15% stop at 50%+ gain, 12% at 30%+, 10% at 20%+, 8% at 10%+', color: 'red' },
            { title: 'Score Crash (2 consecutive scans)', desc: 'Score drops below 50 AND drops 20+ points for 2+ scans. Avoids single-blip sells.', color: 'red' },
            { title: 'Partial Profit Taking', desc: 'Sells 25% at +25% gain, 50% at +40% gain (if score >= 60). Lets winners run.', color: 'orange' },
          ].map(sell => (
            <Card key={sell.title} variant="stat" padding="p-2.5" rounded="rounded-lg">
              <div className={`text-${sell.color}-400 font-medium text-sm`}>{sell.title}</div>
              <div className="text-dark-400 text-[10px]">{sell.desc}</div>
            </Card>
          ))}
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Additional Signals" />
        <div className="space-y-2">
          {[
            { label: 'Insider Buying (CEO/CFO)', sublabel: 'Bullish when insiders are buying', value: '+5 bonus', color: 'text-emerald-400' },
            { label: 'Insider Selling', sublabel: 'Sells exceed buys by 1.5x+', value: '-3 penalty', color: 'text-red-400' },
            { label: 'High Short Interest (>20%)', sublabel: 'Elevated risk of volatility', value: '-5 penalty', color: 'text-red-400' },
            { label: 'Momentum Fading', sublabel: '3-month RS < 95% of 12-month RS', value: '-15% to composite', color: 'text-red-400' },
          ].map(sig => (
            <div key={sig.label} className="flex items-center justify-between py-1.5">
              <div>
                <div className="text-dark-300 text-sm">{sig.label}</div>
                <div className="text-dark-500 text-[10px]">{sig.sublabel}</div>
              </div>
              <span className={`font-data text-xs font-semibold ${sig.color}`}>{sig.value}</span>
            </div>
          ))}
        </div>
      </Card>

      <div className="my-6 h-px bg-dark-700/50" />

      {/* Sector-Adjusted Scoring */}
      <SectionLabel>Sector-Adjusted Scoring</SectionLabel>
      <p className="text-dark-400 text-xs mb-4">
        Not all sectors grow at the same rate. A 20% EPS growth is excellent for Industrials but mediocre for Technology.
        The app adjusts expectations by sector.
      </p>

      <Card variant="glass" className="mb-4">
        <div className="grid grid-cols-2 gap-2">
          {[
            { name: 'Technology', color: 'text-blue-400', thresholds: 'Excellent: 30% | Good: 20%' },
            { name: 'Healthcare', color: 'text-pink-400', thresholds: 'Excellent: 25% | Good: 15%' },
            { name: 'Industrials', color: 'text-amber-400', thresholds: 'Excellent: 20% | Good: 12%' },
            { name: 'Utilities', color: 'text-emerald-400', thresholds: 'Excellent: 12% | Good: 8%' },
          ].map(sector => (
            <Card key={sector.name} variant="stat" padding="p-2" rounded="rounded-lg">
              <div className={`${sector.color} font-semibold text-xs`}>{sector.name}</div>
              <div className="text-dark-400 text-[10px]">{sector.thresholds}</div>
            </Card>
          ))}
        </div>
        <p className="text-dark-500 text-[10px] mt-2">Other sectors use default thresholds: Excellent 25%, Good 15%</p>
      </Card>

      <div className="my-6 h-px bg-dark-700/50" />

      {/* Growth Projection Model */}
      <SectionLabel>Growth Projection Model</SectionLabel>
      <p className="text-dark-400 text-xs mb-4">
        The 6-month growth projection uses a weighted combination of factors to estimate potential upside.
        Higher confidence projections have more data points supporting them.
      </p>

      <GrowthProjectionSection />

      <Card variant="glass" className="mb-4">
        <CardHeader title="Valuation Factor (NEW)" />
        <p className="text-dark-300 text-sm mb-3">
          The valuation factor uses PEG-style analysis to identify undervalued growth stocks:
        </p>
        <div className="space-y-1.5">
          <StatRow label="PEG < 0.5 (deeply undervalued)" value={<span className="text-emerald-400 font-data text-xs">+30%</span>} />
          <StatRow label="PEG < 1.0 (classic buy signal)" value={<span className="text-emerald-400 font-data text-xs">+20%</span>} />
          <StatRow label="PEG 1.0-1.5 (fairly valued)" value={<span className="text-amber-400 font-data text-xs">+5%</span>} />
          <StatRow label="PEG > 2.0 (overvalued)" value={<span className="text-red-400 font-data text-xs">-15%</span>} />
          <StatRow label="Earnings yield > 5%" value={<span className="text-emerald-400 font-data text-xs">+5% bonus</span>} />
          <StatRow label="FCF yield > 5%" value={<span className="text-emerald-400 font-data text-xs">+5% bonus</span>} />
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Confidence Levels" />
        <p className="text-dark-300 text-sm mb-3">
          Projection confidence is based on data quality and signal consistency:
        </p>
        <div className="space-y-2.5">
          <div className="flex items-center gap-3">
            <TagBadge color="green">HIGH</TagBadge>
            <span className="text-sm text-dark-300">Strong analyst coverage, full historical data, high CANSLIM score</span>
          </div>
          <div className="flex items-center gap-3">
            <TagBadge color="amber">MEDIUM</TagBadge>
            <span className="text-sm text-dark-300">Adequate data but some factors missing or mixed signals</span>
          </div>
          <div className="flex items-center gap-3">
            <TagBadge color="red">LOW</TagBadge>
            <span className="text-sm text-dark-300">Limited data, few analysts, or conflicting indicators</span>
          </div>
        </div>
      </Card>

      <div className="my-6 h-px bg-dark-700/50" />

      {/* Portfolio Recommendations */}
      <SectionLabel>Portfolio Recommendations</SectionLabel>
      <p className="text-dark-400 text-xs mb-4">
        The BUY/HOLD/SELL recommendations for portfolio positions use a weighted signal system analyzing 5 factors.
      </p>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Factor 1: CANSLIM Score" />
        <p className="text-dark-400 text-xs mb-2">Strong influence on the recommendation:</p>
        <div className="space-y-1.5">
          <StatRow label="Score >= 70" value={<span className="text-emerald-400 font-data text-xs">+2 buy signals</span>} />
          <StatRow label="Score 50-69" value={<span className="text-amber-400 font-data text-xs">+1 hold signal</span>} />
          <StatRow label="Score 35-49" value={<span className="text-amber-400 font-data text-xs">+1 hold signal</span>} />
          <StatRow label="Score < 35" value={<span className="text-red-400 font-data text-xs">+2 sell signals</span>} />
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Factor 2: Projected Growth" />
        <p className="text-dark-400 text-xs mb-2">6-month growth projection impact:</p>
        <div className="space-y-1.5">
          <StatRow label=">= 20% projected" value={<span className="text-emerald-400 font-data text-xs">+2 buy signals</span>} />
          <StatRow label="10-20% projected" value={<span className="text-emerald-400 font-data text-xs">+1 buy signal</span>} />
          <StatRow label="0-10% projected" value={<span className="text-amber-400 font-data text-xs">+1 hold signal</span>} />
          <StatRow label="-10% to 0%" value={<span className="text-red-400 font-data text-xs">+1 sell signal</span>} />
          <StatRow label="< -10% projected" value={<span className="text-red-400 font-data text-xs">+2 sell signals</span>} />
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Factor 3: Current Gain/Loss" />
        <p className="text-dark-400 text-xs mb-2">Your position's performance:</p>
        <div className="space-y-1.5">
          <StatRow label="Large loss (< -20%) + weak score" value={<span className="text-red-400 font-data text-xs">+2 sell signals</span>} />
          <StatRow label="Moderate loss (-10% to -20%) + weak outlook" value={<span className="text-red-400 font-data text-xs">+1 sell signal</span>} />
          <StatRow label="Most other gain/loss scenarios" value={<span className="text-amber-400 font-data text-xs">+1 hold signal</span>} />
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Factor 4: Analyst Sentiment" />
        <div className="space-y-1.5">
          <StatRow label="Analyst upside >= 30%" value={<span className="text-emerald-400 font-data text-xs">+1 buy signal</span>} />
          <StatRow label="Analyst downside <= -10%" value={<span className="text-red-400 font-data text-xs">+1 sell signal</span>} />
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Factor 5: 52-Week High Proximity" />
        <div className="space-y-1.5">
          <StatRow label="Within 5% of 52-week high" value={<span className="text-emerald-400 font-data text-xs">+1 buy signal (momentum)</span>} />
          <StatRow label="40%+ below high + good score" value={<span className="text-emerald-400 font-data text-xs">+1 buy signal (value)</span>} />
          <StatRow label="40%+ below high + weak score" value={<span className="text-red-400 font-data text-xs">+1 sell signal</span>} />
        </div>
      </Card>

      <Card variant="stat" className="mb-4">
        <CardHeader title="Final Decision Logic" />
        <div className="space-y-2">
          <div className="flex items-start gap-2">
            <TagBadge color="red">SELL</TagBadge>
            <span className="text-dark-300 text-sm">4+ sell signals, OR 3+ sells with 0 buys, OR sells &gt; buys + 1</span>
          </div>
          <div className="flex items-start gap-2">
            <TagBadge color="green">BUY MORE</TagBadge>
            <span className="text-dark-300 text-sm">4+ buy signals, OR 3+ buys with 0 sells, OR buys &gt; sells + 1</span>
          </div>
          <div className="flex items-start gap-2">
            <TagBadge color="amber">HOLD</TagBadge>
            <span className="text-dark-300 text-sm">Everything else - mixed signals or balanced outlook</span>
          </div>
        </div>
      </Card>

      <div className="my-6 h-px bg-dark-700/50" />

      {/* Using the App */}
      <SectionLabel>Using the App</SectionLabel>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Stock Universe" />
        <p className="text-dark-300 text-sm mb-3">
          The app scans ~2,000+ stocks from major indexes. Your portfolio stocks are always scanned first.
        </p>
        <div className="grid grid-cols-2 gap-2">
          {[
            { name: 'S&P 500', color: 'text-blue-400', desc: '~500 large-cap stocks' },
            { name: 'S&P MidCap 400', color: 'text-emerald-400', desc: '~400 mid-cap stocks' },
            { name: 'S&P SmallCap 600', color: 'text-amber-400', desc: '~600 small-cap stocks' },
            { name: 'Russell 2000', color: 'text-purple-400', desc: '~1,200 curated small-caps' },
          ].map(u => (
            <Card key={u.name} variant="stat" padding="p-2" rounded="rounded-lg">
              <div className={`${u.color} font-semibold text-xs`}>{u.name}</div>
              <div className="text-dark-400 text-[10px]">{u.desc}</div>
            </Card>
          ))}
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Breaking Out Page" />
        <p className="text-dark-300 text-sm mb-3">
          Shows stocks with active base patterns that are near their pivot point. These are the best opportunities for timely entries.
        </p>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span className="text-dark-300 text-sm">Pre-breakout: 0-5% below pivot (building for breakout)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            <span className="text-dark-300 text-sm">Breaking out: 0-5% above pivot (confirmed, still buyable)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-dark-500" />
            <span className="text-dark-400 text-sm">Stocks &gt;5% above pivot are excluded (too extended)</span>
          </div>
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Backtesting" />
        <p className="text-dark-300 text-sm mb-3">
          Test the strategy against historical data to validate performance. Access via AI Portfolio &rarr; "Run Historical Backtest".
        </p>
        <div className="space-y-1.5">
          <StatRow label="Timeframe" value="1 year historical simulation" />
          <StatRow label="Benchmark" value="SPY (S&P 500 ETF)" />
          <StatRow label="Metrics" value="Total return, max drawdown, Sharpe ratio, win rate" />
        </div>
      </Card>

      <Card variant="glass" className="mb-4">
        <CardHeader title="Key O'Neil Principles" />
        <ul className="text-dark-400 text-sm space-y-2.5">
          {[
            { text: 'Buy leaders, not laggards.', detail: 'Focus on stocks with Relative Strength >= 80 (top 20%).' },
            { text: 'Cut losses at 7-8%.', detail: 'Small losses are recoverable; big losses require outsized gains.' },
            { text: 'Let winners run.', detail: 'Use trailing stops to protect gains while staying in strong trends.' },
            { text: 'Buy on proper bases.', detail: "Wait for consolidation patterns before buying - don't chase extended stocks." },
            { text: 'Follow the market.', detail: '3 out of 4 stocks follow the general market direction.' },
          ].map((p, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="text-emerald-400 mt-0.5 shrink-0">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M20 6L9 17l-5-5" />
                </svg>
              </span>
              <span><strong className="text-dark-50">{p.text}</strong> {p.detail}</span>
            </li>
          ))}
        </ul>
      </Card>

      <Card variant="stat">
        <CardHeader title="Data Sources" />
        <ul className="text-dark-400 text-sm space-y-1.5">
          <li><strong className="text-dark-200">Financial Modeling Prep:</strong> Earnings, ROE, key metrics, analyst targets, insider trading</li>
          <li><strong className="text-dark-200">Yahoo Finance:</strong> Price history, volume data, short interest</li>
          <li><strong className="text-dark-200">Finviz:</strong> Institutional ownership percentage</li>
        </ul>
        <p className="text-dark-500 text-[10px] mt-3">
          Data is refreshed every 90 minutes during market hours. Earnings data cached for 24 hours. Institutional data cached for 7 days.
        </p>
      </Card>

      <div className="h-8" />
    </div>
  )
}

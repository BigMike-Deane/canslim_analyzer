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
      description: "Measures current earnings momentum using TTM (Trailing Twelve Months) comparison plus acceleration bonus. O'Neil recommends 25%+ quarterly EPS growth.",
      factors: [
        { condition: 'TTM EPS growth >= 25%', points: '12 pts' },
        { condition: 'TTM EPS growth 0-25%', points: 'Scaled 0-12 pts' },
        { condition: 'EPS accelerating (current Q > prior Q growth)', points: '+3 pts bonus' },
        { condition: 'Maintaining strong growth', points: '+1.5 pts bonus' },
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
      description: "Institutional sponsorship validates the stock but too much ownership means the \"smart money\" is already in. Sweet spot is 20-60%.",
      factors: [
        { condition: 'Ownership 20-60% (ideal)', points: '10 pts' },
        { condition: 'Ownership 10-20% or 60-80%', points: '7 pts' },
        { condition: 'Ownership < 10% (too little interest)', points: '3 pts' },
        { condition: 'Ownership > 80% (too crowded)', points: '4 pts' },
        { condition: 'No data available', points: '5 pts (neutral)' },
      ]
    },
    {
      letter: 'M',
      title: 'Market Direction',
      maxPoints: 15,
      color: 'bg-cyan-500/20 text-cyan-400',
      description: "The market determines 75% of a stock's move. Uses S&P 500 position vs 50-day and 200-day moving averages to gauge market health.",
      factors: [
        { condition: 'SPY above both 50 & 200 MA (bullish)', points: '15 pts' },
        { condition: 'SPY above 200 MA, below 50 MA', points: '10.5 pts' },
        { condition: 'SPY below 200 MA, above 50 MA (recovery)', points: '7.5 pts' },
        { condition: 'SPY below both MAs (bearish)', points: '3 pts' },
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
          The Market Direction box on the Dashboard shows the overall market health using the S&P 500 (SPY) as a benchmark.
          O'Neil emphasized that 75% of stocks follow the general market direction, making this crucial for timing.
        </p>
        <div className="space-y-2 text-sm">
          <div className="flex items-start gap-2">
            <span className="text-green-400 font-bold">BULLISH</span>
            <span className="text-dark-400">SPY is above both 50-day and 200-day moving averages - strong uptrend</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-yellow-400 font-bold">NEUTRAL</span>
            <span className="text-dark-400">SPY is between the moving averages - mixed signals, proceed with caution</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-red-400 font-bold">BEARISH</span>
            <span className="text-dark-400">SPY is below both moving averages - downtrend, consider staying in cash</span>
          </div>
        </div>
        <p className="text-dark-400 text-sm mt-3">
          The M Score (15 pts max) reflects this analysis and is applied to every stock's CANSLIM score.
          Even great stocks struggle in bear markets.
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

      <div className="card bg-dark-700/50">
        <h3 className="font-semibold mb-2">Data Sources</h3>
        <ul className="text-dark-400 text-sm space-y-1">
          <li>• <strong className="text-dark-200">Financial Modeling Prep:</strong> Earnings, ROE, key metrics, analyst targets</li>
          <li>• <strong className="text-dark-200">Yahoo Finance:</strong> Price history, volume data</li>
          <li>• <strong className="text-dark-200">Finviz:</strong> Institutional ownership percentage</li>
        </ul>
      </div>

      <div className="h-8" />
    </div>
  )
}

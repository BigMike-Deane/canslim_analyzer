import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts'
import { api, formatScore, getScoreClass, getScoreLabel, formatCurrency, formatPercent, formatMarketCap, formatDateTime } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { ScoreBadge, TagBadge, PnlText } from '../components/Badge'
import StatGrid, { StatRow } from '../components/StatGrid'
import Modal from '../components/Modal'

/* ─── Score Gauge (SVG ring) ──────────────────────────────────────── */

function ScoreGauge({ score, label }) {
  const radius = 44
  const circumference = 2 * Math.PI * radius
  const progress = (score || 0) / 100
  const strokeDashoffset = circumference * (1 - progress)

  const getColor = (s) =>
    s >= 80 ? '#34d399' : s >= 65 ? '#34d399' : s >= 50 ? '#fbbf24' : s >= 35 ? '#fb923c' : '#f87171'

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50" cy="50" r={radius}
            stroke="#1e1e2e"
            strokeWidth="3"
            fill="none"
          />
          <circle
            cx="50" cy="50" r={radius}
            stroke={getColor(score)}
            strokeWidth="3"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold font-data text-dark-50">{formatScore(score)}</span>
        </div>
      </div>
      <span className="text-dark-400 text-[10px] mt-1 uppercase tracking-wide">{label}</span>
    </div>
  )
}

/* ─── Score Detail Modal (CANSLIM letter drill-down) ──────────────── */

function ScoreDetailContent({ scoreKey, scoreData, details, stock }) {
  const detailData = details && typeof details === 'object' ? details : null
  const summaryText = detailData?.summary || (typeof details === 'string' ? details : '')

  const formatPrice = (val) => val != null ? `$${val.toFixed(2)}` : '-'
  const formatPct = (val) => val != null ? `${val.toFixed(1)}%` : '-'
  const formatEps = (val) => val != null ? `$${val.toFixed(2)}` : '-'

  const normalizedColor =
    scoreData.normalized >= 80 ? 'text-emerald-400' :
    scoreData.normalized >= 65 ? 'text-emerald-400' :
    scoreData.normalized >= 50 ? 'text-amber-400' :
    scoreData.normalized >= 35 ? 'text-orange-400' : 'text-red-400'

  const barColor =
    scoreData.normalized >= 80 ? 'bg-emerald-500' :
    scoreData.normalized >= 65 ? 'bg-emerald-500' :
    scoreData.normalized >= 50 ? 'bg-amber-500' :
    scoreData.normalized >= 35 ? 'bg-orange-500' : 'bg-red-500'

  const renderDataSection = () => {
    switch (scoreKey) {
      case 'C': {
        const quarterlyEps = detailData?.quarterly_eps || []
        return (
          <div className="space-y-3">
            <SectionLabel>Quarterly EPS (Most Recent First)</SectionLabel>
            {quarterlyEps.length > 0 ? (
              <div className="grid grid-cols-4 gap-2">
                {quarterlyEps.slice(0, 4).map((eps, i) => (
                  <Card key={i} variant="stat" padding="p-2" rounded="rounded-lg">
                    <div className="text-center">
                      <div className="text-dark-400 text-[10px]">Q{i === 0 ? ' (Latest)' : `-${i}`}</div>
                      <div className={`font-data font-semibold text-sm ${eps >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {formatEps(eps)}
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-dark-500 text-sm">No quarterly data available</div>
            )}
            {detailData?.earnings_surprise_pct != null && (
              <StatRow
                label="Latest Earnings Surprise"
                value={
                  <PnlText
                    value={detailData.earnings_surprise_pct}
                    className="text-sm"
                    prefix={detailData.earnings_surprise_pct >= 0 ? '+' : ''}
                  />
                }
              />
            )}
          </div>
        )
      }

      case 'A': {
        const annualEps = detailData?.annual_eps || []
        return (
          <div className="space-y-3">
            <SectionLabel>Annual EPS (Most Recent First)</SectionLabel>
            {annualEps.length > 0 ? (
              <div className="grid grid-cols-3 gap-2">
                {annualEps.slice(0, 3).map((eps, i) => (
                  <Card key={i} variant="stat" padding="p-2" rounded="rounded-lg">
                    <div className="text-center">
                      <div className="text-dark-400 text-[10px]">{i === 0 ? 'Latest' : `${i}Y Ago`}</div>
                      <div className={`font-data font-semibold text-sm ${eps >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {formatEps(eps)}
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-dark-500 text-sm">No annual data available</div>
            )}
            {detailData?.roe != null && (
              <StatRow
                label="Return on Equity (ROE)"
                value={
                  <span className={`font-data text-sm ${(detailData.roe * 100) >= 17 ? 'text-emerald-400' : (detailData.roe * 100) >= 10 ? 'text-amber-400' : 'text-red-400'}`}>
                    {(detailData.roe * 100).toFixed(1)}%
                  </span>
                }
              />
            )}
          </div>
        )
      }

      case 'N':
        return (
          <div className="space-y-2">
            <SectionLabel>Price Position</SectionLabel>
            <Card variant="stat" padding="p-3" rounded="rounded-lg">
              <div className="space-y-2">
                <StatRow
                  label="Current Price"
                  value={formatPrice(detailData?.current_price || stock?.current_price)}
                />
                <StatRow
                  label="52-Week High"
                  value={formatPrice(detailData?.week_52_high || stock?.week_52_high)}
                />
                <StatRow
                  label="Distance from High"
                  value={
                    <span className={`font-data text-sm ${(detailData?.pct_from_high || 0) <= 10 ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {formatPct(detailData?.pct_from_high)} below
                    </span>
                  }
                />
              </div>
            </Card>
          </div>
        )

      case 'S':
        return (
          <div className="space-y-2">
            <SectionLabel>Volume &amp; Supply</SectionLabel>
            <Card variant="stat" padding="p-3" rounded="rounded-lg">
              <div className="space-y-2">
                <StatRow
                  label="Volume Ratio"
                  value={
                    <span className={`font-data text-sm ${(detailData?.volume_ratio || 0) >= 1.5 ? 'text-emerald-400' : 'text-dark-300'}`}>
                      {detailData?.volume_ratio?.toFixed(2) || '-'}x average
                    </span>
                  }
                />
                {detailData?.avg_volume && (
                  <StatRow
                    label="Avg Daily Volume"
                    value={`${(detailData.avg_volume / 1e6).toFixed(2)}M`}
                  />
                )}
                {detailData?.shares_outstanding && (
                  <StatRow
                    label="Shares Outstanding"
                    value={`${(detailData.shares_outstanding / 1e9).toFixed(2)}B`}
                  />
                )}
              </div>
            </Card>
          </div>
        )

      case 'L':
        return (
          <div className="space-y-2">
            <SectionLabel>Relative Strength</SectionLabel>
            <Card variant="stat" padding="p-3" rounded="rounded-lg">
              <p className="text-dark-300 text-sm">
                {summaryText || 'Measures how well this stock performs relative to the overall market.'}
              </p>
            </Card>
          </div>
        )

      case 'I':
        return (
          <div className="space-y-2">
            <SectionLabel>Institutional Ownership</SectionLabel>
            <Card variant="stat" padding="p-3" rounded="rounded-lg">
              <StatRow
                label="Institutional Ownership"
                value={
                  <span className={`font-data text-sm ${(detailData?.institutional_pct || 0) >= 50 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {formatPct(detailData?.institutional_pct || stock?.institutional_ownership)}
                  </span>
                }
              />
            </Card>
          </div>
        )

      case 'M':
        return (
          <div className="space-y-2">
            <SectionLabel>Market Direction</SectionLabel>
            <Card variant="stat" padding="p-3" rounded="rounded-lg">
              <p className="text-dark-300 text-sm">
                {summaryText || 'Overall market trend based on SPY, QQQ, and DIA vs their moving averages.'}
              </p>
            </Card>
          </div>
        )

      default:
        return null
    }
  }

  const titles = {
    C: 'Current Quarterly Earnings',
    A: 'Annual Earnings Growth',
    N: 'New Highs',
    S: 'Supply and Demand',
    L: 'Leader or Laggard',
    I: 'Institutional Sponsorship',
    M: 'Market Direction',
  }

  return (
    <div className="space-y-4">
      {/* Letter + Title */}
      <div className="flex items-center gap-3">
        <div className={`w-10 h-10 rounded-xl font-bold text-xl flex items-center justify-center ${getScoreClass(scoreData.normalized)}`}>
          {scoreKey}
        </div>
        <div>
          <div className="font-semibold text-dark-50">{titles[scoreKey] || scoreKey}</div>
          <div className="text-dark-400 text-xs font-data">
            {scoreData.value != null ? `${scoreData.value.toFixed(1)}/${scoreData.max} points` : 'No data'}
          </div>
        </div>
      </div>

      {/* Score Bar */}
      <div>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-dark-400">Score</span>
          <span className={`font-data font-semibold ${normalizedColor}`}>{scoreData.normalized.toFixed(0)}%</span>
        </div>
        <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${barColor}`}
            style={{ width: `${scoreData.normalized}%` }}
          />
        </div>
      </div>

      {/* Summary */}
      {summaryText && (
        <Card variant="accent" accent="cyan" padding="p-3" rounded="rounded-lg">
          <p className="text-primary-400 text-sm">{summaryText}</p>
        </Card>
      )}

      {/* Data Section */}
      {renderDataSection()}
    </div>
  )
}

/* ─── CANSLIM Breakdown ───────────────────────────────────────────── */

function CANSLIMDetail({ stock }) {
  const [selectedScore, setSelectedScore] = useState(null)

  const scores = [
    { key: 'C', label: 'Current Earnings', value: stock.c_score, max: 15, desc: 'Quarterly earnings growth' },
    { key: 'A', label: 'Annual Earnings', value: stock.a_score, max: 15, desc: 'Annual earnings growth' },
    { key: 'N', label: 'New Highs', value: stock.n_score, max: 15, desc: 'New products, management, price highs' },
    { key: 'S', label: 'Supply/Demand', value: stock.s_score, max: 15, desc: 'Shares outstanding and volume' },
    { key: 'L', label: 'Leader/Laggard', value: stock.l_score, max: 15, desc: 'Relative strength vs market' },
    { key: 'I', label: 'Institutional', value: stock.i_score, max: 10, desc: 'Institutional sponsorship' },
    { key: 'M', label: 'Market Direction', value: stock.m_score, max: 15, desc: 'Overall market trend' },
  ]

  const normalizeScore = (value, max) => {
    if (value == null || max === 0) return 0
    return (value / max) * 100
  }

  const getDetail = (key) => {
    if (!stock.score_details) return null
    return stock.score_details[key.toLowerCase()] || stock.score_details[key] || null
  }

  const selectedScoreObj = selectedScore ? scores.find(s => s.key === selectedScore) : null

  return (
    <>
      <Card variant="glass" className="mb-4">
        <CardHeader title="CANSLIM Breakdown" />
        <div className="space-y-3">
          {scores.map(s => {
            const normalized = normalizeScore(s.value, s.max)
            const barColor =
              normalized >= 80 ? 'bg-emerald-500' :
              normalized >= 65 ? 'bg-emerald-500' :
              normalized >= 50 ? 'bg-amber-500' :
              normalized >= 35 ? 'bg-orange-500' : 'bg-red-500'

            return (
              <div key={s.key} className="flex items-center gap-3">
                <button
                  onClick={() => setSelectedScore(s.key)}
                  className={`w-10 h-10 rounded-xl font-bold text-lg flex items-center justify-center shrink-0 ${getScoreClass(normalized)} hover:scale-110 active:scale-95 transition-all cursor-pointer`}
                  title={`Click for ${s.label} details`}
                >
                  {s.key}
                </button>
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-sm text-dark-200">{s.label}</span>
                    <span className={`text-xs font-data font-semibold ${getScoreClass(normalized)}`}>
                      {s.value != null ? `${s.value.toFixed(1)}/${s.max}` : '-'}
                    </span>
                  </div>
                  <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden mt-1">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                      style={{ width: `${normalized}%` }}
                    />
                  </div>
                  <div className="text-dark-500 text-[10px] mt-0.5">{s.desc}</div>
                </div>
              </div>
            )
          })}
        </div>
      </Card>

      {/* Score Detail Modal */}
      <Modal
        open={!!selectedScore}
        onClose={() => setSelectedScore(null)}
        title={selectedScoreObj ? `${selectedScoreObj.key} - ${selectedScoreObj.label}` : ''}
        size="sm"
      >
        {selectedScore && selectedScoreObj && (
          <ScoreDetailContent
            scoreKey={selectedScore}
            scoreData={{
              value: selectedScoreObj.value,
              max: selectedScoreObj.max,
              normalized: normalizeScore(selectedScoreObj.value, selectedScoreObj.max),
            }}
            details={getDetail(selectedScore)}
            stock={stock}
          />
        )}
      </Modal>
    </>
  )
}

/* ─── Price Information ────────────────────────────────────────────── */

function PriceInfo({ stock }) {
  const fromHigh = stock.week_52_high
    ? ((stock.current_price / stock.week_52_high - 1) * 100)
    : null

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Price Information" />
      <StatGrid
        columns={2}
        stats={[
          {
            label: 'Current Price',
            value: <span className="text-xl">{formatCurrency(stock.current_price)}</span>,
          },
          {
            label: 'Market Cap',
            value: formatMarketCap(stock.market_cap),
          },
          {
            label: '52 Week High',
            value: formatCurrency(stock.week_52_high),
          },
          {
            label: '52 Week Low',
            value: formatCurrency(stock.week_52_low),
          },
          {
            label: 'From 52W High',
            value: fromHigh != null ? formatPercent(fromHigh, true) : '-',
            color: fromHigh < 0 ? 'text-red-400' : 'text-emerald-400',
          },
          {
            label: 'Projected Growth',
            value: stock.projected_growth != null ? `+${stock.projected_growth.toFixed(0)}%` : '-',
            color: 'text-emerald-400',
          },
        ]}
      />
    </Card>
  )
}

/* ─── Score History (Line Chart) ───────────────────────────────────── */

function ScoreHistory({ history }) {
  if (!history || history.length < 2) {
    return (
      <Card variant="glass" className="mb-4 text-center py-6">
        <div className="text-dark-500 text-xs">Not enough score history yet</div>
      </Card>
    )
  }

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Score History" />
      <div className="h-40 -mx-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <Line
              type="monotone"
              dataKey="total_score"
              stroke="#00e5ff"
              strokeWidth={2}
              dot={false}
            />
            <Tooltip
              contentStyle={{
                background: '#14141f',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: '8px',
                boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
              }}
              labelStyle={{ color: '#6b7280', fontSize: '11px' }}
              formatter={(value) => [formatScore(value), 'Score']}
            />
            <XAxis dataKey="date" hide />
            <YAxis hide domain={[0, 100]} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

/* ─── Market Signals (Insider + Short Interest) ────────────────────── */

function InsiderShortSection({ stock }) {
  const hasInsider = stock.insider_sentiment || stock.insider_buy_count > 0 || stock.insider_sell_count > 0
  const hasShort = stock.short_interest_pct != null

  if (!hasInsider && !hasShort) return null

  const getSentimentColor = (sentiment) => {
    if (sentiment === 'bullish') return 'green'
    if (sentiment === 'bearish') return 'red'
    return 'default'
  }

  const getShortColor = (pct) => {
    if (pct >= 20) return 'text-red-400'
    if (pct >= 10) return 'text-orange-400'
    return 'text-emerald-400'
  }

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Market Signals" />

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
        {hasInsider && (
          <>
            <div>
              <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Insider Sentiment</div>
              <TagBadge color={getSentimentColor(stock.insider_sentiment)}>
                {stock.insider_sentiment === 'bullish' && 'Bullish'}
                {stock.insider_sentiment === 'bearish' && 'Bearish'}
                {stock.insider_sentiment !== 'bullish' && stock.insider_sentiment !== 'bearish' && (stock.insider_sentiment || 'Unknown')}
              </TagBadge>
            </div>
            <div>
              <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Insider Activity (3mo)</div>
              <div className="flex items-center gap-3">
                <span className="text-emerald-400 font-data text-sm font-semibold">
                  {stock.insider_buy_count || 0} buys
                </span>
                <span className="text-red-400 font-data text-sm font-semibold">
                  {stock.insider_sell_count || 0} sells
                </span>
              </div>
            </div>
          </>
        )}

        {hasShort && (
          <>
            <div>
              <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Short Interest</div>
              <span className={`font-data text-sm font-semibold ${getShortColor(stock.short_interest_pct)}`}>
                {stock.short_interest_pct?.toFixed(1)}% of float
              </span>
            </div>
            <div>
              <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Days to Cover</div>
              <span className="font-data text-sm font-semibold text-dark-200">
                {stock.short_ratio?.toFixed(1) || '-'} days
              </span>
            </div>
          </>
        )}
      </div>

      {/* Warning for high short interest */}
      {stock.short_interest_pct >= 20 && (
        <div className="mt-3 pt-3 border-t border-dark-700/50 text-sm text-orange-400">
          High short interest - stock may be volatile
        </div>
      )}

      {/* Positive signal for bullish insiders */}
      {stock.insider_sentiment === 'bullish' && stock.insider_buy_count >= 3 && (
        <div className="mt-3 pt-3 border-t border-dark-700/50 text-sm text-emerald-400">
          Strong insider buying - management is confident
        </div>
      )}
    </Card>
  )
}

/* ─── Growth Mode Section ──────────────────────────────────────────── */

function GrowthModeSection({ stock }) {
  if (!stock.is_growth_stock && !stock.growth_mode_score) return null

  const details = stock.growth_mode_details || {}

  const growthScores = [
    { key: 'R', label: 'Revenue Growth', value: details.r, color: 'text-emerald-400' },
    { key: 'F', label: 'Funding Health', value: details.f, color: 'text-blue-400' },
  ]

  return (
    <Card variant="accent" accent="green" className="mb-4">
      <CardHeader
        title="Growth Mode Score"
        action={
          <div className="flex items-center gap-2">
            <TagBadge color="green">
              {stock.is_growth_stock ? 'Growth Stock' : 'Hybrid'}
            </TagBadge>
            <span className="text-xl font-bold font-data text-emerald-400">
              {stock.growth_mode_score?.toFixed(1) || '-'}
            </span>
          </div>
        }
      />

      <p className="text-dark-400 text-xs mb-3">
        Alternative scoring for pre-revenue and high-growth companies. Uses revenue momentum instead of earnings.
      </p>

      <div className="space-y-2">
        {growthScores.map(s => (
          <div key={s.key} className="flex items-center justify-between py-1.5 border-b border-dark-700/50 last:border-0">
            <div className="flex items-center gap-2">
              <span className={`font-bold font-data ${s.color}`}>{s.key}</span>
              <span className="text-sm text-dark-200">{s.label}</span>
            </div>
            <span className="text-dark-400 text-sm font-data">{s.value || '-'}</span>
          </div>
        ))}
      </div>

      {stock.revenue_growth_pct != null && (
        <div className="mt-3 pt-3 border-t border-dark-700/50 flex justify-between items-center">
          <span className="text-dark-400 text-sm">Revenue Growth (YoY)</span>
          <PnlText value={stock.revenue_growth_pct} className="text-sm font-semibold" />
        </div>
      )}
    </Card>
  )
}

/* ─── Technical Analysis ───────────────────────────────────────────── */

function TechnicalAnalysis({ stock }) {
  return (
    <Card variant="glass" className="mb-4">
      <CardHeader
        title="Technical Analysis"
        action={
          stock.is_breaking_out && (
            <TagBadge color="amber">Breaking Out</TagBadge>
          )
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
        <div>
          <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Base Pattern</div>
          <div className="font-semibold text-sm capitalize">
            {stock.base_type && stock.base_type !== 'none' ? (
              <span className="text-blue-400">{stock.base_type} base</span>
            ) : (
              <span className="text-dark-500">No base</span>
            )}
          </div>
        </div>

        <div>
          <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Weeks in Base</div>
          <div className="font-data text-sm font-semibold">
            {stock.weeks_in_base > 0 ? (
              <span className="text-dark-200">{stock.weeks_in_base} weeks</span>
            ) : (
              <span className="text-dark-500">-</span>
            )}
          </div>
        </div>

        <div>
          <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Volume Ratio</div>
          <div className={`font-data text-sm font-semibold ${stock.volume_ratio >= 1.5 ? 'text-emerald-400' : stock.volume_ratio >= 1.0 ? 'text-amber-400' : 'text-dark-400'}`}>
            {stock.volume_ratio ? `${stock.volume_ratio.toFixed(1)}x avg` : '-'}
          </div>
        </div>

        <div>
          <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Breakout Volume</div>
          <div className="font-data text-sm font-semibold">
            {stock.breakout_volume_ratio ? (
              <span className="text-amber-400">{stock.breakout_volume_ratio.toFixed(1)}x</span>
            ) : (
              <span className="text-dark-500">-</span>
            )}
          </div>
        </div>
      </div>

      {stock.eps_acceleration && (
        <div className="mt-3 pt-3 border-t border-dark-700/50 flex items-center gap-2 flex-wrap">
          <TagBadge color="green">EPS Accelerating</TagBadge>
          {stock.earnings_surprise_pct > 0 && (
            <TagBadge color="cyan">
              Beat estimates +{stock.earnings_surprise_pct.toFixed(0)}%
            </TagBadge>
          )}
        </div>
      )}

      <p className="text-dark-500 text-xs mt-3">
        {stock.is_breaking_out
          ? 'Stock is breaking out of a consolidation pattern with strong volume - potential buy zone.'
          : stock.base_type && stock.base_type !== 'none'
          ? 'Stock is building a base pattern. Watch for breakout with volume.'
          : 'No clear base pattern detected.'}
      </p>
    </Card>
  )
}

/* ─── Main Page Component ──────────────────────────────────────────── */

export default function StockDetail() {
  const { ticker } = useParams()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [stock, setStock] = useState(null)
  const [refreshing, setRefreshing] = useState(false)

  const fetchStock = async () => {
    try {
      setLoading(true)
      const data = await api.getStock(ticker)
      setStock(data)
    } catch (err) {
      console.error('Failed to fetch stock:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStock()
  }, [ticker])

  const handleRefresh = async () => {
    try {
      setRefreshing(true)
      await api.refreshStock(ticker)
      await fetchStock()
    } catch (err) {
      console.error('Failed to refresh:', err)
    } finally {
      setRefreshing(false)
    }
  }

  const handleAddToWatchlist = async () => {
    try {
      await api.addToWatchlist({ ticker })
      alert('Added to watchlist!')
    } catch (err) {
      console.error('Failed to add to watchlist:', err)
    }
  }

  const handleAddToPortfolio = async () => {
    const shares = prompt('Number of shares:')
    const costBasis = prompt('Cost per share:')
    if (shares && costBasis) {
      try {
        await api.addPosition({
          ticker,
          shares: parseFloat(shares),
          cost_basis: parseFloat(costBasis)
        })
        alert('Added to portfolio!')
      } catch (err) {
        console.error('Failed to add to portfolio:', err)
      }
    }
  }

  if (loading) {
    return (
      <div className="p-4 md:p-6">
        <div className="skeleton h-8 w-32 mb-4" />
        <div className="skeleton h-32 rounded-xl mb-4" />
        <div className="skeleton h-48 rounded-xl mb-4" />
        <div className="skeleton h-32 rounded-xl" />
      </div>
    )
  }

  if (!stock) {
    return (
      <div className="p-4 md:p-6">
        <Card variant="glass" className="text-center py-8">
          <div className="text-4xl mb-3">?</div>
          <div className="font-semibold text-dark-50 mb-2">Stock Not Found</div>
          <p className="text-dark-400 text-sm mb-4">
            {ticker} has not been analyzed yet.
          </p>
          <button onClick={() => navigate(-1)} className="btn-primary">
            Go Back
          </button>
        </Card>
      </div>
    )
  }

  return (
    <div className="p-4 md:p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between mb-5 gap-3">
        <div className="min-w-0">
          <button
            onClick={() => navigate(-1)}
            className="inline-flex items-center gap-1 text-xs text-dark-400 hover:text-dark-200 transition-colors mb-2"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M15 18l-6-6 6-6" />
            </svg>
            Back
          </button>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-dark-50">{stock.ticker}</h1>
            <ScoreBadge score={stock.canslim_score} size="md" />
          </div>
          <div className="text-dark-300 text-sm truncate">{stock.name}</div>
          <div className="text-xs text-dark-400 mt-0.5">{stock.sector} / {stock.industry}</div>
        </div>
        <div className="flex-shrink-0 self-center sm:self-start">
          <ScoreGauge score={stock.canslim_score} label={getScoreLabel(stock.canslim_score)} />
        </div>
      </div>

      <PriceInfo stock={stock} />

      <GrowthModeSection stock={stock} />

      <TechnicalAnalysis stock={stock} />

      <InsiderShortSection stock={stock} />

      <CANSLIMDetail stock={stock} />

      <ScoreHistory history={stock.score_history} />

      {/* Actions */}
      <SectionLabel>Actions</SectionLabel>
      <div className="grid grid-cols-2 gap-2 sm:gap-3 mb-4">
        <button onClick={handleAddToWatchlist} className="btn-secondary">
          + Watchlist
        </button>
        <button onClick={handleAddToPortfolio} className="btn-primary">
          + Portfolio
        </button>
      </div>

      <button
        onClick={handleRefresh}
        disabled={refreshing}
        className="w-full btn-secondary flex items-center justify-center gap-2"
      >
        {refreshing ? (
          <span>Refreshing...</span>
        ) : (
          <span>Refresh Analysis</span>
        )}
      </button>

      <div className="text-dark-500 text-[10px] text-center mt-3">
        Last updated: {stock.last_updated ? formatDateTime(stock.last_updated) : 'Never'}
      </div>

      <div className="h-4" />
    </div>
  )
}

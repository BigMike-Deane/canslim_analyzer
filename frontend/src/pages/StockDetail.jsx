import { useState, useEffect, useMemo } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { getAdjacentTickers } from '../stockListContext'
import { ComposedChart, LineChart, Line, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend, ReferenceLine } from 'recharts'
import { api, formatScore, getScoreClass, getScoreLabel, getScoreHex, formatCurrency, formatPercent, formatMarketCap, formatDateTime } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { chartAxis, chartColors } from '../components/chartTheme'
import { TagBadge, PnlText } from '../components/Badge'
import StatGrid, { StatRow } from '../components/StatGrid'
import Spinner from '../components/Spinner'
import Modal from '../components/Modal'
import CollapsibleSection from '../components/CollapsibleSection'
import { useToast } from '../components/Toast'
import PositionSizingCard from '../components/PositionSizingCard'
import { computePositionSizing } from '../positionSizing'

/* ─── Held-position hero (ui-revamp) ──────────────────────────────────
   The page's job is "buy / hold / sell — and why?". When the AI already
   holds this ticker, the buy-oriented sizing card self-suppresses and,
   before this card existed, NOTHING replaced it — you could arrive from
   a needs-attention chip or defense-day row and see no P&L, no off-peak,
   no exit plan. This is the hold story: the position's state plus the
   server-computed nearest exit trigger (backend/exit_plan.py — same data
   the AI Portfolio modal renders; presentation-only, no re-derivation). */
function HeldPositionCard({ position }) {
  if (!position) return null
  const p = position
  const offPeak = p.trailing_stop?.drop_from_peak_pct
  const plan = p.exit_plan
  const nearest = plan?.triggers?.find(t => t.kind === plan.nearest_kind)
  const nearTone = nearest?.direction === 'up'
    ? 'text-emerald-400'
    : nearest?.distance_pct == null ? 'text-dark-400'
      : nearest.distance_pct <= 5 ? 'text-red-400'
        : nearest.distance_pct <= 12 ? 'text-amber-400'
          : 'text-dark-300'
  return (
    <Card variant="glass" className="mb-4 border-t-2 border-t-primary-500">
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <span className="text-[10px] font-semibold tracking-widest uppercase text-primary-400">
          AI portfolio holds this position
        </span>
        <Link to="/ai-portfolio" className="text-[10px] text-primary-400 hover:text-primary-300 transition-colors">
          Manage in AI Portfolio &rarr;
        </Link>
      </div>
      <div className="flex items-baseline gap-4 flex-wrap mb-2">
        <div className="text-2xl font-bold font-data text-dark-100">
          {formatCurrency(p.current_value)}
          <span className={`text-base font-semibold ml-2 ${p.gain_loss_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {p.gain_loss_pct >= 0 ? '+' : ''}{p.gain_loss_pct?.toFixed(1)}%
          </span>
        </div>
        <span className="text-xs text-dark-400 font-data">
          {p.shares?.toFixed(2)} sh @ {formatCurrency(p.cost_basis)}
        </span>
        {offPeak != null && (
          <span className="text-xs text-dark-400 font-data">
            {offPeak.toFixed(1)}% off peak {p.trailing_stop?.peak_price ? `(${formatCurrency(p.trailing_stop.peak_price)})` : ''}
          </span>
        )}
      </div>
      {nearest && (
        <div className="flex items-center justify-between bg-dark-850 rounded-md px-2.5 py-1.5 text-xs">
          <span className="text-dark-300">
            Nearest exit: <b className="text-dark-100">{nearest.label}</b>
            {nearest.note && <span className="text-dark-500 ml-1.5 hidden sm:inline">{nearest.note}</span>}
          </span>
          <span className="font-data text-right">
            <span className="text-dark-100">{nearest.price != null ? formatCurrency(nearest.price) : `< ${nearest.threshold}`}</span>
            {nearest.distance_pct != null && (
              <span className={`ml-2 ${nearTone}`}>
                {nearest.direction === 'up' ? `${nearest.distance_pct}% to go` : `${nearest.distance_pct}% away`}
              </span>
            )}
          </span>
        </div>
      )}
    </Card>
  )
}

/* ─── Score Gauge (SVG ring) ──────────────────────────────────────── */

function ScoreGauge({ score, label }) {
  const radius = 44
  const circumference = 2 * Math.PI * radius
  const progress = (score || 0) / 100
  const strokeDashoffset = circumference * (1 - progress)

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50" cy="50" r={radius}
            stroke="#261a1c"
            strokeWidth="3"
            fill="none"
          />
          <circle
            cx="50" cy="50" r={radius}
            stroke={getScoreHex(score)}
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

  // Tiers + hues mirror getScoreClass / getScoreHex so this drill-down agrees
  // with the gauge and badges (and keeps mid-scores off brand amber).
  const normalizedColor =
    scoreData.normalized >= 80 ? 'text-emerald-400' :
    scoreData.normalized >= 65 ? 'text-green-400' :
    scoreData.normalized >= 50 ? 'text-stone-300' :
    scoreData.normalized >= 35 ? 'text-rose-300' : 'text-red-400'

  const barColor =
    scoreData.normalized >= 80 ? 'bg-emerald-500' :
    scoreData.normalized >= 65 ? 'bg-green-500' :
    scoreData.normalized >= 50 ? 'bg-stone-400' :
    scoreData.normalized >= 35 ? 'bg-rose-500' : 'bg-red-500'

  const renderDataSection = () => {
    switch (scoreKey) {
      case 'C': {
        const quarterlyEps = detailData?.quarterly_eps || []
        return (
          <div className="space-y-3">
            <SectionLabel>Quarterly EPS (Most Recent First)</SectionLabel>
            {quarterlyEps.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
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
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
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
  // Which letter's breakdown is expanded inline (null = all collapsed).
  const [expandedScore, setExpandedScore] = useState(null)

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

  return (
    <Card as="section" aria-labelledby="sd-canslim-heading" variant="glass" className="mb-4">
      <CardHeader title="CANSLIM Breakdown" titleId="sd-canslim-heading" />
      <div className="space-y-3">
        {scores.map(s => {
          const normalized = normalizeScore(s.value, s.max)
          const barColor =
            normalized >= 80 ? 'bg-emerald-500' :
            normalized >= 65 ? 'bg-green-500' :
            normalized >= 50 ? 'bg-stone-400' :
            normalized >= 35 ? 'bg-rose-500' : 'bg-red-500'
          const isExpanded = expandedScore === s.key

          return (
            <div key={s.key}>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setExpandedScore(prev => prev === s.key ? null : s.key)}
                  aria-expanded={isExpanded}
                  className={`w-10 h-10 rounded-xl font-bold text-lg flex items-center justify-center shrink-0 ${getScoreClass(normalized)} hover:scale-110 active:scale-95 transition-all cursor-pointer ${isExpanded ? 'ring-2 ring-primary-500/60' : ''}`}
                  title={`${isExpanded ? 'Hide' : 'Show'} ${s.label} details`}
                >
                  {s.key}
                </button>
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-sm text-dark-200 flex items-center gap-1">
                      {s.label}
                      <svg
                        className={`text-dark-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                        width="11" height="11" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2.5"
                      >
                        <polyline points="6 9 12 15 18 9" />
                      </svg>
                    </span>
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

              {/* Inline drill-down (replaces the old modal) — keeps the score
                  bar in view and lets the user expand several letters at once. */}
              {isExpanded && (
                <div className="mt-2 ml-[52px] pl-3 border-l-2 border-primary-500/30">
                  <ScoreDetailContent
                    scoreKey={s.key}
                    scoreData={{ value: s.value, max: s.max, normalized }}
                    details={getDetail(s.key)}
                    stock={stock}
                  />
                </div>
              )}
            </div>
          )
        })}
      </div>
    </Card>
  )
}

/* ─── Price Information ────────────────────────────────────────────── */

function PriceInfo({ stock }) {
  const fromHigh = stock.week_52_high
    ? ((stock.current_price / stock.week_52_high - 1) * 100)
    : null

  return (
    <Card as="section" aria-labelledby="sd-price-heading" variant="glass" className="mb-4">
      <CardHeader title="Price Information" titleId="sd-price-heading" />
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

/* ─── Analyst Consensus ────────────────────────────────────────────── */

function AnalystConsensus({ stock }) {
  const target = stock.analyst_target_price
  // No analyst coverage → render nothing (the backend sends null, not 0).
  if (!target) return null

  const current = stock.current_price
  const low = stock.analyst_target_low
  const high = stock.analyst_target_high
  const upside = stock.analyst_upside_pct
  const count = stock.analyst_count

  const upsideColor =
    upside == null ? 'text-dark-300' : upside >= 0 ? 'text-emerald-400' : 'text-red-400'

  // Range bar maps low→high onto 0→100%. Only drawn when we have a real spread.
  const haveRange = low != null && high != null && high > low
  const pos = (v) =>
    v == null ? null : Math.min(100, Math.max(0, ((v - low) / (high - low)) * 100))

  return (
    <Card as="section" aria-labelledby="sd-analyst-heading" variant="glass" className="mb-4">
      <CardHeader title="Analyst Consensus" titleId="sd-analyst-heading" />
      <StatGrid
        columns={2}
        stats={[
          {
            label: 'Consensus Target',
            value: <span className="text-xl">{formatCurrency(target)}</span>,
          },
          {
            label: 'Upside vs Current',
            value: upside != null ? formatPercent(upside, true) : '-',
            color: upsideColor,
          },
        ]}
      />

      {haveRange && (
        <div className="mt-4">
          <div className="flex justify-between text-[10px] text-dark-500 font-data mb-1.5">
            <span>Low {formatCurrency(low)}</span>
            <span>High {formatCurrency(high)}</span>
          </div>
          <div className="relative h-2 bg-dark-700 rounded-full">
            {current != null && pos(current) != null && (
              <div
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-1 h-4 bg-dark-300 rounded"
                style={{ left: `${pos(current)}%` }}
                title={`Current ${formatCurrency(current)}`}
              />
            )}
            <div
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 rounded-full bg-primary-400 ring-2 ring-dark-900"
              style={{ left: `${pos(target)}%` }}
              title={`Consensus ${formatCurrency(target)}`}
            />
          </div>
          <div className="flex items-center gap-3 mt-2 text-[10px] text-dark-500">
            <span className="flex items-center gap-1">
              <span className="w-1 h-3 bg-dark-300 rounded inline-block" /> Current
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-primary-400 inline-block" /> Consensus
            </span>
            {count ? (
              <span className="ml-auto">{count} analyst{count === 1 ? '' : 's'}</span>
            ) : null}
          </div>
        </div>
      )}
      {!haveRange && count ? (
        <div className="mt-2 text-[10px] text-dark-500">
          {count} analyst{count === 1 ? '' : 's'}
        </div>
      ) : null}
    </Card>
  )
}

/* ─── Score Replay Chart (Dual-Axis: Score + Price) ───────────────── */

const COMPONENT_COLORS = {
  c: '#f87171', a: '#fb923c', n: '#fbbf24',
  s: '#34d399', l: '#60a5fa', i: '#a78bfa', m: '#f472b6',
}

const TOOLTIP_STYLE = {
  background: '#1f1416',  // dark-800 — warm-tinted card surface
  border: '1px solid rgba(255,255,255,0.06)',
  borderRadius: '8px',
  boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
}

function ScoreReplayTooltip({ active, payload, label, showComponents, isPerScan }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  let displayLabel = label
  if (isPerScan && label) {
    const dt = new Date(label)
    if (!isNaN(dt)) {
      displayLabel = dt.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    }
  }
  // "Biggest mover" — which component changed most vs the prior point. Threshold
  // of 0.5 filters rounding noise; non-trivial moves like N -7.4 surface clearly.
  let topMover = null
  if (d._deltas) {
    for (const [k, v] of Object.entries(d._deltas)) {
      if (Math.abs(v) < 0.5) continue
      if (!topMover || Math.abs(v) > Math.abs(topMover.delta)) {
        topMover = { letter: k, delta: v }
      }
    }
  }

  return (
    <div style={TOOLTIP_STYLE} className="px-3 py-2 text-xs">
      <div className="text-dark-500 mb-1">{displayLabel}</div>
      <div className="flex items-center gap-3 mb-1">
        {/* Brand amber for score, pale gold for price — was red/gold pre-rebrand. */}
        <span style={{ color: chartColors.brand }}>Score: <b>{formatScore(d.total_score)}</b></span>
        {d.price != null && (
          <span style={{ color: '#fde68a' }}>
            Price: <b>{formatCurrency(d.price)}</b>
            {/* Matches the right axis, which plots % vs window start. */}
            {d._pricePct != null && (
              <span className="text-dark-500"> ({d._pricePct >= 0 ? '+' : ''}{d._pricePct.toFixed(1)}%)</span>
            )}
          </span>
        )}
      </div>
      {topMover && (
        <div className="text-[11px] mb-1">
          <span className="text-dark-500">Δ vs prior {isPerScan ? 'scan' : 'day'}: </span>
          <span style={{ color: COMPONENT_COLORS[topMover.letter] }}>
            <b>{topMover.letter.toUpperCase()}</b> {topMover.delta > 0 ? '+' : ''}{topMover.delta.toFixed(1)}
          </span>
        </div>
      )}
      {showComponents && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-3 gap-y-0.5 mt-1 pt-1 border-t border-white/5">
          {['c','a','n','s','l','i','m'].map(k => (
            <span key={k} style={{ color: COMPONENT_COLORS[k] }}>
              {k.toUpperCase()}: {d[k] != null ? Math.round(d[k]) : '-'}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function ScoreHistory({ history, resolution = 'daily', onResolutionChange }) {
  const [showComponents, setShowComponents] = useState(false)
  const [period, setPeriod] = useState('30')

  if (!history || history.length < 2) {
    return (
      <Card as="section" aria-label="Score Replay" variant="glass" className="mb-4 text-center py-6">
        <div className="text-dark-500 text-xs">Not enough score history yet</div>
      </Card>
    )
  }

  // At per-scan resolution we plot against the full timestamp so multiple
  // scans on the same day separate visually; daily mode keeps using `date`.
  const isPerScan = resolution === 'all'
  const xKey = isPerScan ? 'timestamp' : 'date'

  // Filter by period (use timestamp at per-scan, date at daily)
  const cutoff = new Date()
  cutoff.setDate(cutoff.getDate() - parseInt(period))
  const refDate = (h) => new Date(isPerScan ? (h.timestamp || h.date) : h.date)
  const filtered = period === 'all' ? history : history.filter(h => refDate(h) >= cutoff)
  const baseData = filtered.length >= 2 ? filtered : history

  // Stamp each point with the per-component deltas vs the prior point so the
  // tooltip can answer "what just moved?" without re-scanning the whole array.
  // First point has no prior, so it gets no _deltas.
  const data = baseData.map((d, i) => {
    if (i === 0) return d
    const prev = baseData[i - 1]
    const _deltas = {}
    for (const k of ['c','a','n','s','l','i','m']) {
      _deltas[k] = (d[k] ?? 0) - (prev[k] ?? 0)
    }
    return { ...d, _deltas }
  })

  // Score axis floor: scores below ~40 are dead space (buy floor is 72,
  // even weak holds sit in the 50s), so a fixed 0 floor spends half the
  // chart's resolution on a region no stock visits. Snap the floor to the
  // largest of {0, 25, 50} sitting ≥5 points below the window's low —
  // snapping (not min-fitting) caps magnification at 2x and keeps two
  // stocks with the same axis directly comparable.
  const scoreVals = data.map(d => d.total_score).filter(v => v != null)
  const scoreMin = scoreVals.length ? Math.min(...scoreVals) : 0
  const scoreFloor = Math.max(0, Math.min(50, Math.floor((scoreMin - 5) / 25) * 25))
  const scoreSpan = 100 - scoreFloor
  const scoreTicks = Array.from({ length: scoreSpan / 25 + 1 }, (_, i) => scoreFloor + i * 25)

  // Price overlay: rendered as % change from the window's first priced
  // point, on an axis whose span in percentage points EQUALS the score
  // axis span — that makes 1pp of price move = 1 score point of height,
  // so the two lines' shapes are directly comparable at every zoom level.
  // (Previously the price axis was min/max-fitted with 5% padding,
  // magnifying price moves ~3x vs the score and making score drops read
  // as minor by comparison.)
  const anchorPrice = data.find(d => d.price > 0)?.price
  const chartData = anchorPrice
    ? data.map(d => (d.price > 0 ? { ...d, _pricePct: ((d.price / anchorPrice) - 1) * 100 } : d))
    : data
  const pricePcts = chartData.map(d => d._pricePct).filter(v => v != null)
  const hasPrice = pricePcts.length > 0
  // Center the price series inside the score-span window. Widen only when
  // the move can't fit, sacrificing 1:1 scale rather than clipping.
  let priceDomain = [-scoreSpan / 2, scoreSpan / 2]
  if (hasPrice) {
    const lo = Math.min(...pricePcts)
    const hi = Math.max(...pricePcts)
    const span = Math.max(scoreSpan, (hi - lo) * 1.1)
    const mid = (lo + hi) / 2
    priceDomain = [mid - span / 2, mid + span / 2]
  }

  // Buy gate for context — champion config min_score (nostate_optimized).
  const BUY_FLOOR = 72

  // Per-component std dev over the visible window. Flat letters (sd < 0.5)
  // are dropped from the chart and legend so the dynamic ones (usually N/S/M)
  // aren't drowned out by 7 overlapping near-flat lines.
  const COMPONENT_KEYS = ['c','a','n','s','l','i','m']
  const activeComponents = (() => {
    if (!data.length) return COMPONENT_KEYS
    const stats = {}
    for (const k of COMPONENT_KEYS) {
      const vals = data.map(d => d[k]).filter(v => v != null)
      if (vals.length < 2) { stats[k] = 0; continue }
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length
      const variance = vals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / vals.length
      stats[k] = Math.sqrt(variance)
    }
    return COMPONENT_KEYS.filter(k => stats[k] >= 0.5)
  })()

  const periods = [
    { value: '14', label: '2W' },
    { value: '30', label: '1M' },
    { value: '90', label: '3M' },
    { value: 'all', label: 'All' },
  ]

  // Peak-context badge: chart geometry can't communicate the drop when the
  // peak sits outside the selected window (default 1M), so state it
  // numerically. Computed from the FULL fetched history, not the visible
  // slice — the badge is the one element that never loses the peak.
  const latestPoint = history[history.length - 1]
  const peakPoint = history.reduce(
    (best, h) => (h.total_score != null && h.total_score > (best?.total_score ?? -Infinity) ? h : best),
    null
  )
  const peakDrop = peakPoint && latestPoint?.total_score != null
    ? peakPoint.total_score - latestPoint.total_score
    : 0
  const peakDateLabel = (() => {
    if (!peakPoint?.date) return ''
    const dt = new Date(peakPoint.date)
    // Date-only strings parse as UTC midnight; format in UTC too, or the
    // label shifts back a day in western timezones.
    return isNaN(dt) ? '' : dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })
  })()

  return (
    <Card as="section" aria-labelledby="sd-replay-heading" variant="glass" className="mb-4">
      <div className="flex items-center justify-between mb-2">
        <CardHeader
          title="Score Replay"
          titleId="sd-replay-heading"
          subtitle={peakDrop >= 5 ? (
            <span className="text-red-400" title="Drop from the highest score in the fetched history — the peak may sit outside the selected window">
              ▼ {peakDrop.toFixed(1)} from {formatScore(peakPoint.total_score)} peak{peakDateLabel ? ` (${peakDateLabel})` : ''}
            </span>
          ) : undefined}
        />
        <div className="flex items-center gap-2">
          {onResolutionChange && (
            <div className="flex bg-dark-900/50 rounded overflow-hidden border border-white/5">
              {[{ v: 'daily', l: 'Daily' }, { v: 'all', l: 'Per scan' }].map(r => (
                <button
                  key={r.v}
                  onClick={() => onResolutionChange(r.v)}
                  title={r.v === 'all' ? 'Show every scan — exposes intraday score swings' : 'One point per day (last scan)'}
                  className={`text-[10px] px-2 py-0.5 transition-colors ${
                    resolution === r.v
                      ? 'bg-white/10 text-white'
                      : 'text-dark-500 hover:text-dark-400'
                  }`}
                >
                  {r.l}
                </button>
              ))}
            </div>
          )}
          <button
            onClick={() => setShowComponents(!showComponents)}
            className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${
              showComponents
                ? 'border-primary-500/40 text-primary-400 bg-primary-500/10'
                : 'border-white/10 text-dark-500 hover:text-dark-400'
            }`}
          >
            CANSLIM
          </button>
          <div className="flex bg-dark-900/50 rounded overflow-hidden border border-white/5">
            {periods.map(p => (
              <button
                key={p.value}
                onClick={() => setPeriod(p.value)}
                className={`text-[10px] px-2 py-0.5 transition-colors ${
                  period === p.value
                    ? 'bg-white/10 text-white'
                    : 'text-dark-500 hover:text-dark-400'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="h-56 -mx-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 5, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                {/* Brand amber gradient for the score area — was red #dc2626. */}
                <stop offset="0%" stopColor={chartColors.brand} stopOpacity={0.18} />
                <stop offset="100%" stopColor={chartColors.brand} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey={xKey}
              tick={{ fontSize: 10, fill: chartAxis.tick }}
              tickFormatter={d => {
                if (!d) return ''
                if (isPerScan) {
                  const dt = new Date(d)
                  if (isNaN(dt)) return ''
                  return `${dt.getMonth() + 1}/${dt.getDate()} ${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}`
                }
                const p = String(d).split('-')
                return p.length >= 3 ? `${p[1]}/${p[2]}` : d
              }}
              interval="preserveStartEnd"
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              yAxisId="score"
              domain={[scoreFloor, 100]}
              ticks={scoreTicks}
              tick={{ fontSize: 10, fill: chartColors.brand }}
              axisLine={false}
              tickLine={false}
              width={28}
            />
            <YAxis
              yAxisId="price"
              orientation="right"
              domain={priceDomain}
              tick={{ fontSize: 10, fill: '#fde68a' }}
              axisLine={false}
              tickLine={false}
              width={45}
              tickFormatter={v => `${v > 0 ? '+' : ''}${Math.round(v)}%`}
            />
            {/* Hidden axis for component lines so they get the full 0-15 vertical
                range instead of being squashed at the bottom of the 0-100 score axis. */}
            <YAxis yAxisId="component" domain={[0, 15]} hide={true} />
            <Tooltip content={<ScoreReplayTooltip showComponents={showComponents} isPerScan={isPerScan} />} />
            {/* Buy gate — above this line the stock is buyable. Gives the
                (possibly cropped) score axis an actionable anchor. */}
            <ReferenceLine
              yAxisId="score"
              y={BUY_FLOOR}
              stroke="rgba(255,255,255,0.18)"
              strokeDasharray="3 3"
              label={{ value: `buy ${BUY_FLOOR}`, position: 'insideBottomLeft', fill: 'rgba(255,255,255,0.35)', fontSize: 9 }}
            />
            {/* Score area */}
            <Area
              yAxisId="score"
              type="monotone"
              dataKey="total_score"
              stroke={chartColors.brand}
              strokeWidth={2}
              fill="url(#scoreGrad)"
              dot={false}
              name="Score"
            />
            {/* Price line — pale gold (was red/gold) for high-contrast pairing
                with the brand-amber score line; dashed for axis disambiguation.
                Plots _pricePct (% vs window start) so its vertical scale
                matches the score axis 1:1 — see priceDomain above. */}
            {hasPrice && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="_pricePct"
                stroke="#fde68a"
                strokeWidth={1.5}
                dot={false}
                strokeDasharray="4 2"
                name="Price"
              />
            )}
            {/* CANSLIM component lines (toggled). Only letters that actually
                moved in the visible window are drawn — flat ones add noise. */}
            {showComponents && activeComponents.map(k => (
              <Line
                key={k}
                yAxisId="component"
                type="monotone"
                dataKey={k}
                stroke={COMPONENT_COLORS[k]}
                strokeWidth={1}
                dot={false}
                opacity={0.7}
                name={k.toUpperCase()}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {hasPrice && (
        <div className="text-[10px] text-dark-500 mt-1 px-1">
          Price axis: % vs window start, scaled so 1% ≈ 1 score point
        </div>
      )}
      {showComponents && (
        <div className="flex flex-wrap gap-3 mt-2 px-1 items-center">
          {activeComponents.map(k => (
            <span key={k} className="flex items-center gap-1 text-[10px]">
              <span className="w-2.5 h-0.5 rounded" style={{ background: COMPONENT_COLORS[k] }} />
              <span style={{ color: COMPONENT_COLORS[k] }}>{k.toUpperCase()}</span>
            </span>
          ))}
          {activeComponents.length < COMPONENT_KEYS.length && (
            <span className="text-[10px] text-dark-500">
              · {COMPONENT_KEYS.filter(k => !activeComponents.includes(k)).map(k => k.toUpperCase()).join('/')} flat
            </span>
          )}
        </div>
      )}
    </Card>
  )
}

/* ─── Market Signals (Insider + Short Interest) ────────────────────── */

function InsiderShortSection({ stock }) {
  const hasInsider = stock.insider_sentiment || stock.insider_buy_count > 0 || stock.insider_sell_count > 0
  const hasShort = stock.short_interest_pct != null
  const hasStrength = stock.rs_3m != null || stock.rs_12m != null
    || stock.eps_estimate_revision_pct != null || stock.industry_group_rank != null

  if (!hasInsider && !hasShort && !hasStrength) return null

  // rs_* are return ratios vs SPY ((1+stock)/(1+spy), capped 3.0): 1.0 = market-neutral
  const rsColor = (rs) => rs >= 1.05 ? 'text-emerald-400' : rs <= 0.95 ? 'text-red-400' : 'text-dark-300'
  const rsLabel = (rs) => rs == null ? '-' : `${rs.toFixed(2)}x SPY`
  const groupRankColor = (r) => r >= 80 ? 'text-emerald-400' : r >= 50 ? 'text-amber-400' : 'text-red-400'

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
    <Card as="section" aria-labelledby="sd-signals-heading" variant="glass" className="mb-4">
      <CardHeader title="Market Signals" titleId="sd-signals-heading" />

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

      {hasStrength && (
        <div className={`grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 ${(hasInsider || hasShort) ? 'mt-3 pt-3 border-t border-dark-700/50' : ''}`}>
          <div>
            <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Rel Strength 3M</div>
            <span className={`font-data text-sm font-semibold ${rsColor(stock.rs_3m)}`}>
              {rsLabel(stock.rs_3m)}
            </span>
          </div>
          <div>
            <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Rel Strength 12M</div>
            <span className={`font-data text-sm font-semibold ${rsColor(stock.rs_12m)}`}>
              {rsLabel(stock.rs_12m)}
            </span>
          </div>
          <div>
            <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Est. Revisions</div>
            <span className={`font-data text-sm font-semibold ${
              stock.eps_estimate_revision_pct > 0 ? 'text-emerald-400'
                : stock.eps_estimate_revision_pct < 0 ? 'text-red-400' : 'text-dark-300'
            }`}>
              {stock.eps_estimate_revision_pct != null
                ? `${stock.eps_estimate_revision_pct > 0 ? '+' : ''}${stock.eps_estimate_revision_pct.toFixed(1)}%`
                : '-'}
            </span>
          </div>
          <div>
            <div className="text-dark-400 text-[10px] uppercase tracking-wide mb-1">Group Rank</div>
            <span className={`font-data text-sm font-semibold ${stock.industry_group_rank != null ? groupRankColor(stock.industry_group_rank) : 'text-dark-400'}`}>
              {stock.industry_group_rank != null ? `${stock.industry_group_rank} / 100` : '-'}
            </span>
          </div>
        </div>
      )}

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
    <Card as="section" aria-labelledby="sd-growth-heading" variant="accent" accent="green" className="mb-4">
      <CardHeader
        title="Growth Mode Score"
        titleId="sd-growth-heading"
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
    <Card as="section" aria-labelledby="sd-technical-heading" variant="glass" className="mb-4">
      <CardHeader
        title="Technical Analysis"
        titleId="sd-technical-heading"
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
              (() => {
                // Detected geometry != live setup: flag bases the price has
                // already left (breakout done) or fallen out of (base failed).
                const rel = stock.pivot_price > 0 && stock.current_price > 0
                  ? ((stock.current_price - stock.pivot_price) / stock.pivot_price) * 100
                  : null
                const status = rel == null ? null : rel > 5 ? 'extended' : rel < -20 ? 'broken' : null
                return (
                  <span className={status ? 'text-dark-400' : 'text-blue-400'}>
                    {stock.base_type} base
                    {status && (
                      <span
                        className="text-dark-500"
                        title={status === 'extended'
                          ? 'Price is >5% above the base pivot — the breakout already happened.'
                          : 'Price is >20% below the base pivot — the base failed.'}
                      > · {status}</span>
                    )}
                  </span>
                )
              })()
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

/* ─── Add-to-Portfolio Form (modal body) ───────────────────────────── */

function AddPositionForm({ ticker, currentPrice, onClose }) {
  const toast = useToast()
  const [shares, setShares] = useState('')
  // Prefill cost basis with the loaded price — most adds are "I just bought
  // at market", and a prefilled field is one less mobile keyboard round-trip.
  const [costBasis, setCostBasis] = useState(
    currentPrice != null ? String(Number(currentPrice.toFixed(2))) : ''
  )
  const [errors, setErrors] = useState({})
  const [submitting, setSubmitting] = useState(false)

  const validate = () => {
    const next = {}
    const sharesNum = parseFloat(shares)
    const costNum = parseFloat(costBasis)
    if (!shares.trim() || !Number.isFinite(sharesNum) || sharesNum <= 0) {
      next.shares = 'Enter a number of shares greater than 0'
    }
    if (!costBasis.trim() || !Number.isFinite(costNum) || costNum <= 0) {
      next.costBasis = 'Enter a cost per share greater than 0'
    }
    setErrors(next)
    return Object.keys(next).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (submitting) return
    if (!validate()) return
    setSubmitting(true)
    try {
      await api.addPosition({
        ticker,
        shares: parseFloat(shares),
        cost_basis: parseFloat(costBasis),
      })
      toast.success(`${ticker} added to portfolio`)
      onClose()
    } catch (err) {
      toast.error(err.message || 'Failed to add to portfolio')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4" noValidate>
      <div>
        <label htmlFor="add-pos-shares" className="text-[10px] uppercase tracking-wider text-dark-400 font-semibold">
          Shares
        </label>
        <input
          id="add-pos-shares"
          type="number"
          inputMode="decimal"
          step="any"
          min="0"
          value={shares}
          onChange={(e) => {
            setShares(e.target.value)
            setErrors(prev => ({ ...prev, shares: null }))
          }}
          placeholder="10"
          className="w-full mt-1"
          autoFocus
        />
        {errors.shares && (
          <p className="text-red-400 text-xs mt-1">{errors.shares}</p>
        )}
      </div>

      <div>
        <label htmlFor="add-pos-cost" className="text-[10px] uppercase tracking-wider text-dark-400 font-semibold">
          Cost Basis (per share)
        </label>
        <input
          id="add-pos-cost"
          type="number"
          inputMode="decimal"
          step="0.01"
          min="0"
          value={costBasis}
          onChange={(e) => {
            setCostBasis(e.target.value)
            setErrors(prev => ({ ...prev, costBasis: null }))
          }}
          placeholder="25.00"
          className="w-full mt-1"
        />
        {errors.costBasis && (
          <p className="text-red-400 text-xs mt-1">{errors.costBasis}</p>
        )}
        {currentPrice != null && (
          <p className="text-dark-500 text-[11px] mt-1">
            Current price: {formatCurrency(currentPrice)}
          </p>
        )}
      </div>

      <button type="submit" disabled={submitting} className="w-full btn-primary disabled:opacity-50">
        {submitting ? (
          <span className="inline-flex items-center gap-2"><Spinner size="xs" inline />Adding…</span>
        ) : (
          `Add ${ticker} to Portfolio`
        )}
      </button>
    </form>
  )
}

/* ─── Fundamental Audit (AI buy-gate view) ─────────────────────────── */

function FundamentalAudit({ audit }) {
  // No audit for this ticker (404 / fetch failure) → render nothing.
  if (!audit) return null

  const pnlColor = (v) =>
    v == null ? 'text-dark-300' : v >= 0 ? 'text-emerald-400' : 'text-red-400'
  const fmtPct = (v) => (v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(1)}%` : '-')

  const confidence = audit.fundamental_confidence
  const confidenceColor =
    confidence == null ? 'text-dark-300' :
    confidence >= 70 ? 'text-emerald-400' :
    confidence >= 50 ? 'text-amber-400' : 'text-red-400'

  // confidence_breakdown is a JSON column — defensively handle dict vs string.
  const breakdown = audit.confidence_breakdown
  const breakdownEntries =
    breakdown && typeof breakdown === 'object' && !Array.isArray(breakdown)
      ? Object.entries(breakdown)
      : null

  return (
    <Card as="section" aria-labelledby="sd-audit-heading" variant="glass" className="mb-4">
      <CollapsibleSection
        title="Fundamental Audit"
        titleId="sd-audit-heading"
        defaultOpen={false}
        badge={
          confidence != null && (
            <span className={`text-xs font-data font-semibold ${confidenceColor}`}>
              {confidence.toFixed(0)} confidence
            </span>
          )
        }
      >
        <p className="text-dark-500 text-xs mb-3">
          What the AI's buy gate sees for {audit.ticker} — fundamentals, analyst targets,
          earnings track record, and estimate revisions.
        </p>

        {/* Balance-sheet / quality stats */}
        <StatGrid
          columns={4}
          className="mb-4"
          stats={[
            {
              label: 'ROE',
              value: audit.roe != null ? `${audit.roe.toFixed(1)}%` : '-',
              color: audit.roe == null ? undefined : audit.roe >= 17 ? 'text-emerald-400' : audit.roe >= 10 ? 'text-amber-400' : 'text-red-400',
            },
            {
              label: 'Debt / Equity',
              value: audit.debt_to_equity != null ? audit.debt_to_equity.toFixed(2) : '-',
            },
            {
              label: 'FCF / Share',
              value: audit.free_cash_flow_per_share != null ? formatCurrency(audit.free_cash_flow_per_share) : '-',
              color: audit.free_cash_flow_per_share == null ? undefined : pnlColor(audit.free_cash_flow_per_share),
            },
            {
              label: 'Current Ratio',
              value: audit.current_ratio != null ? audit.current_ratio.toFixed(2) : '-',
            },
          ]}
        />

        {/* Analyst targets */}
        <SectionLabel>Analyst Targets</SectionLabel>
        <StatGrid
          columns={4}
          className="mb-4"
          stats={[
            { label: 'Low', value: audit.analyst_low_target != null ? formatCurrency(audit.analyst_low_target) : '-' },
            { label: 'Average', value: audit.analyst_avg_target != null ? formatCurrency(audit.analyst_avg_target) : '-' },
            { label: 'High', value: audit.analyst_high_target != null ? formatCurrency(audit.analyst_high_target) : '-' },
            {
              label: 'Upside',
              value: fmtPct(audit.analyst_upside_pct),
              sublabel: audit.analyst_num ? `${audit.analyst_num} analyst${audit.analyst_num === 1 ? '' : 's'}` : undefined,
              color: pnlColor(audit.analyst_upside_pct),
            },
          ]}
        />

        {/* Earnings track record + revisions */}
        <SectionLabel>Earnings &amp; Revisions</SectionLabel>
        <StatGrid
          columns={3}
          className="mb-1"
          stats={[
            {
              label: 'Beat Streak',
              value: audit.beat_streak != null ? `${audit.beat_streak}Q` : '-',
              color: (audit.beat_streak || 0) >= 4 ? 'text-emerald-400' : undefined,
            },
            { label: 'Avg Beat', value: fmtPct(audit.avg_beat_magnitude), color: pnlColor(audit.avg_beat_magnitude) },
            { label: 'Last Beat', value: fmtPct(audit.last_beat_pct), color: pnlColor(audit.last_beat_pct) },
            { label: 'EPS Revision', value: fmtPct(audit.eps_revision_pct), color: pnlColor(audit.eps_revision_pct) },
            { label: 'Rev Revision', value: fmtPct(audit.revenue_revision_pct), color: pnlColor(audit.revenue_revision_pct) },
            {
              label: 'Insider Cluster Buys',
              value: audit.insider_cluster_buys ?? '-',
              color: (audit.insider_cluster_buys || 0) > 0 ? 'text-emerald-400' : undefined,
            },
          ]}
        />

        {/* Confidence breakdown */}
        {(breakdownEntries?.length || typeof breakdown === 'string') && (
          <div className="mt-3 pt-3 border-t border-dark-700/50">
            <SectionLabel>Confidence Breakdown</SectionLabel>
            {breakdownEntries ? (
              <div className="space-y-0.5">
                {breakdownEntries.map(([k, v]) => (
                  <StatRow
                    key={k}
                    label={k.replace(/_/g, ' ')}
                    value={typeof v === 'number' ? v.toFixed(1) : String(v)}
                  />
                ))}
              </div>
            ) : (
              <p className="text-dark-400 text-xs">{breakdown}</p>
            )}
          </div>
        )}

        {audit.audited_at && (
          <div className="text-dark-500 text-[10px] mt-3">
            Audited {formatDateTime(audit.audited_at)}
            {audit.price_at_audit != null ? ` · price ${formatCurrency(audit.price_at_audit)}` : ''}
          </div>
        )}
      </CollapsibleSection>
    </Card>
  )
}

/* ─── Main Page Component ──────────────────────────────────────────── */

export default function StockDetail() {
  const { ticker } = useParams()
  const navigate = useNavigate()
  const toast = useToast()
  const [loading, setLoading] = useState(true)
  const [stock, setStock] = useState(null)
  const [error, setError] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [scoreResolution, setScoreResolution] = useState('daily')
  // AI Portfolio config + summary drives the position-sizing card below.
  // Lightweight piggyback fetch — same endpoint AIPortfolio.jsx uses, so
  // it'll hit the api.js 120s cache on the second view of this page.
  const [aiPortfolio, setAiPortfolio] = useState(null)
  // Add-to-Portfolio modal (replaces the old chained window.prompt() flow).
  const [showAddPositionModal, setShowAddPositionModal] = useState(false)
  // Latest earnings audit for this ticker — best-effort; null (404 or any
  // failure) hides the Fundamental Audit section entirely.
  const [audit, setAudit] = useState(null)

  // Adjacent tickers from the list page the user came from (Screener,
  // Watchlist, Breakouts). null source = no context (direct URL, etc.),
  // in which case the prev/next nav is hidden entirely.
  const adjacents = useMemo(() => getAdjacentTickers(ticker), [ticker])

  const fetchStock = async (resolution = scoreResolution) => {
    try {
      // Reset before fetching so a failure on a new ticker doesn't render
      // the previously-loaded ticker's data with no error indicator.
      setStock(null)
      setError(null)
      setLoading(true)
      const data = await api.getStock(ticker, { resolution })
      setStock(data)
    } catch (err) {
      console.error('Failed to fetch stock:', err)
      setError(err.message || 'Failed to load stock data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStock(scoreResolution)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker, scoreResolution])

  // Fetch AI Portfolio state once on mount — drives PositionSizingCard.
  // Swallow failures: the card is best-effort. If the API errors, the
  // card simply hides and the existing Actions buttons still work.
  useEffect(() => {
    api.getAIPortfolio()
      .then(setAiPortfolio)
      .catch(() => setAiPortfolio(null))
  }, [])

  // Fetch the fundamental audit best-effort. Keyed on ticker (this page is
  // reused across tickers via prev/next nav) and cleared first so a slow
  // response never shows the previous ticker's audit. Many tickers have no
  // audit — the 404 lands in .catch and the section stays hidden.
  useEffect(() => {
    let cancelled = false
    setAudit(null)
    api.getEarningsAudit(ticker)
      .then(data => { if (!cancelled) setAudit(data) })
      .catch(() => null)
    return () => { cancelled = true }
  }, [ticker])

  // Scroll to top when the ticker changes — without this, clicking the
  // prev/next nav from the page header leaves the viewport at the bottom
  // of the previous page (same component, same route pattern, so React
  // Router does not auto-scroll).
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' })
  }, [ticker])

  const handleRefresh = async () => {
    try {
      setRefreshing(true)
      await api.refreshStock(ticker)
      await fetchStock(scoreResolution)
    } catch (err) {
      console.error('Failed to refresh:', err)
      toast.error(err.message || 'Failed to refresh analysis')
    } finally {
      setRefreshing(false)
    }
  }

  const handleAddToWatchlist = async () => {
    try {
      await api.addToWatchlist({ ticker })
      toast.success(`${ticker} added to watchlist`)
    } catch (err) {
      toast.error(err.message || 'Failed to add to watchlist')
    }
  }

  const handleAddToPortfolio = () => setShowAddPositionModal(true)

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

  if (error && !stock) {
    return (
      <div className="p-4 md:p-6">
        <Card variant="glass" className="text-center py-8">
          <div className="text-4xl mb-3">!</div>
          <div className="font-semibold text-dark-50 mb-2">Failed to Load</div>
          <p className="text-dark-400 text-sm mb-4">{error}</p>
          <div className="flex gap-3 justify-center">
            <button onClick={() => fetchStock()} className="btn-primary">Retry</button>
            <button onClick={() => navigate(-1)} className="btn-secondary">Go Back</button>
          </div>
        </Card>
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
      <header className="flex flex-col sm:flex-row sm:items-start sm:justify-between mb-5 gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-3 mb-2">
            <button
              onClick={() => navigate(-1)}
              className="inline-flex items-center gap-1 text-xs text-dark-400 hover:text-dark-200 transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M15 18l-6-6 6-6" />
              </svg>
              Back
            </button>
            {adjacents.source && (adjacents.prev || adjacents.next) && (
              <div className="flex items-center gap-2 text-xs">
                <button
                  onClick={() => adjacents.prev && navigate(`/stock/${adjacents.prev}`)}
                  disabled={!adjacents.prev}
                  className="inline-flex items-center gap-1 text-dark-400 hover:text-dark-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  title={adjacents.prev ? `Previous: ${adjacents.prev}` : 'No previous ticker'}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M15 18l-6-6 6-6" />
                  </svg>
                  {adjacents.prev || '—'}
                </button>
                <span className="text-dark-500 text-[10px] font-data">
                  {adjacents.position
                    ? `${adjacents.position.idx}/${adjacents.position.total} · ${adjacents.source}`
                    : adjacents.source}
                </span>
                <button
                  onClick={() => adjacents.next && navigate(`/stock/${adjacents.next}`)}
                  disabled={!adjacents.next}
                  className="inline-flex items-center gap-1 text-dark-400 hover:text-dark-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  title={adjacents.next ? `Next: ${adjacents.next}` : 'No next ticker'}
                >
                  {adjacents.next || '—'}
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M9 18l6-6-6-6" />
                  </svg>
                </button>
              </div>
            )}
          </div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-dark-50 flex items-center gap-2">
              {stock.ticker}
              <button
                onClick={() => { navigator.clipboard.writeText(stock.ticker); toast.success('Copied!') }}
                className="text-dark-500 hover:text-dark-300 transition-colors"
                title="Copy ticker"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
              </button>
            </h1>
            {/* Score lives in the gauge to the right — the small badge here
                was a duplicate of the same number (one hero per screen). */}
          </div>
          <div className="text-dark-300 text-sm truncate">{stock.name}</div>
          <div className="text-xs text-dark-400 mt-0.5">{stock.sector} / {stock.industry}</div>
        </div>
        <div className="flex-shrink-0 self-center sm:self-start flex flex-col items-center gap-1">
          <ScoreGauge score={stock.canslim_score} label={getScoreLabel(stock.canslim_score)} />
          {stock.data_freshness?.age_text && (
            <span
              className={`text-[10px] font-data px-1.5 py-0.5 rounded border ${
                stock.data_freshness.is_stale
                  ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                  : 'bg-dark-800/70 text-dark-400 border-dark-700'
              }`}
              title={stock.last_updated ? `Last scan: ${formatDateTime(stock.last_updated)}` : 'No scan history'}
            >
              {stock.data_freshness.is_stale ? 'Stale · ' : ''}{stock.data_freshness.age_text}
            </span>
          )}
        </div>
      </header>

      <PriceInfo stock={stock} />

      {/* Position sizing — derived from current AI Portfolio state +
          this stock's pivot/price. Rendered high in the page (right
          after PriceInfo) because it's the most decision-relevant
          info: shares, limit, stop, dollar risk, copyable trade
          ticket. The card self-suppresses when the trade isn't
          actionable (extended, below score gate, no cash, already
          at max-positions). */}
      {(() => {
        if (!aiPortfolio?.summary || !stock?.current_price) return null
        const heldTickers = new Set(
          (aiPortfolio.positions || []).map(p => p.ticker?.toUpperCase())
        )
        const alreadyHeld = heldTickers.has((stock.ticker || ticker).toUpperCase())
        // Held ⇒ the hold story replaces the buy-side sizing card.
        if (alreadyHeld) {
          const pos = (aiPortfolio.positions || []).find(
            p => p.ticker?.toUpperCase() === (stock.ticker || ticker).toUpperCase()
          )
          return <HeldPositionCard position={pos} />
        }
        const sizing = computePositionSizing({
          ticker: stock.ticker || ticker,
          currentPrice: stock.current_price,
          pivotPrice: stock.pivot_price,
          score: stock.canslim_score,
          cash: aiPortfolio.summary.cash,
          totalValue: aiPortfolio.summary.total_value,
          positionsCount: aiPortfolio.positions?.length || 0,
          maxPositions: aiPortfolio.config?.max_positions,
          stopLossPct: aiPortfolio.config?.stop_loss_pct,
          minScore: aiPortfolio.config?.min_score_to_buy,
        })
        return <PositionSizingCard sizing={sizing} alreadyHeld={alreadyHeld} />
      })()}

      {/* Decision inputs first (setup + the score's evidence), then the
          supporting research cluster — same first-glance vs drill-down
          ordering as the portfolio pages. */}
      <TechnicalAnalysis stock={stock} />

      <CANSLIMDetail stock={stock} />

      <AnalystConsensus stock={stock} />

      <GrowthModeSection stock={stock} />

      <InsiderShortSection stock={stock} />

      <FundamentalAudit audit={audit} />

      <ScoreHistory
        history={stock.score_history}
        resolution={scoreResolution}
        onResolutionChange={setScoreResolution}
      />

      {/* Actions */}
      <section aria-label="Actions">
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
            <span className="inline-flex items-center gap-2"><Spinner size="xs" inline />Refreshing…</span>
          ) : (
            <span>Refresh Analysis</span>
          )}
        </button>

        <div className="text-dark-500 text-[10px] text-center mt-3">
          Last updated: {stock.last_updated ? formatDateTime(stock.last_updated) : 'Never'}
        </div>
      </section>

      <Modal
        open={showAddPositionModal}
        onClose={() => setShowAddPositionModal(false)}
        title={`Add ${ticker} to Portfolio`}
        size="sm"
      >
        <AddPositionForm
          ticker={ticker}
          currentPrice={stock.current_price}
          onClose={() => setShowAddPositionModal(false)}
        />
      </Modal>

      <div className="h-4" />
    </div>
  )
}

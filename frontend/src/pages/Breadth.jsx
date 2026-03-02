import { useState, useEffect } from 'react'
import { api } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import StatGrid from '../components/StatGrid'
import PageHeader from '../components/PageHeader'

function SectorRow({ sector }) {
  const adColor = sector.ad_ratio >= 1.5 ? 'text-emerald-400' : sector.ad_ratio >= 1 ? 'text-amber-400' : 'text-red-400'
  const scoreColor = sector.avg_score >= 60 ? 'text-emerald-400' : sector.avg_score >= 45 ? 'text-amber-400' : 'text-red-400'

  return (
    <div className="flex justify-between items-center py-2.5 border-b border-dark-700/30 last:border-0">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm text-dark-100 font-medium truncate">{sector.sector}</span>
          <span className="text-[10px] text-dark-500 font-data">{sector.count} stocks</span>
        </div>
        <div className="flex items-center gap-3 mt-0.5">
          <span className="text-[10px] text-dark-500">
            <span className="text-emerald-500">{sector.advancing}</span>
            {' / '}
            <span className="text-red-500">{sector.declining}</span>
            {' A/D'}
          </span>
          {sector.breakouts > 0 && (
            <span className="text-[10px] text-cyan-400">{sector.breakouts} breakouts</span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-4 shrink-0">
        <div className="text-right">
          <div className={`text-sm font-data font-medium ${scoreColor}`}>{sector.avg_score}</div>
          <div className="text-[10px] text-dark-500">avg score</div>
        </div>
        <div className="text-right">
          <div className={`text-sm font-data font-medium ${adColor}`}>{sector.ad_ratio}</div>
          <div className="text-[10px] text-dark-500">A/D</div>
        </div>
        <div className="text-right w-12">
          <div className="text-sm font-data text-dark-200">{sector.near_high_pct}%</div>
          <div className="text-[10px] text-dark-500">nr high</div>
        </div>
      </div>
    </div>
  )
}

function BreadthGauge({ label, value, max, color = 'emerald' }) {
  const pct = Math.min(100, (value / max) * 100)
  const colors = {
    emerald: 'bg-emerald-500',
    red: 'bg-red-500',
    amber: 'bg-amber-500',
    cyan: 'bg-cyan-500',
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-[11px] text-dark-400">{label}</span>
        <span className="text-sm font-data text-dark-100 font-medium">{value}</span>
      </div>
      <div className="h-2 rounded-full bg-dark-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${colors[color] || colors.emerald}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function Breadth() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await api.getMarketBreadth()
        setData(result)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch market breadth:', err)
        setError(err.message || 'Failed to load')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="p-4 md:p-6 max-w-4xl mx-auto">
        <PageHeader title="Market Breadth" />
        <div className="space-y-3">
          <div className="skeleton h-24 rounded-2xl" />
          <div className="skeleton h-48 rounded-2xl" />
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="p-4 md:p-6 max-w-4xl mx-auto">
        <PageHeader title="Market Breadth" />
        <Card variant="glass" className="text-center py-8">
          <div className="text-red-400 text-sm">{error || 'No data'}</div>
        </Card>
      </div>
    )
  }

  const adHealthy = data.ad_ratio >= 1.5
  const hlHealthy = data.hl_ratio >= 2

  return (
    <div className="p-4 md:p-6 max-w-4xl mx-auto">
      <PageHeader title="Market Breadth" />

      {/* Key metrics */}
      <SectionLabel>Breadth Indicators</SectionLabel>
      <Card variant="glass" className="mb-4">
        <StatGrid
          columns={4}
          stats={[
            {
              label: 'A/D Ratio',
              value: data.ad_ratio,
              sublabel: `${data.advancing} adv / ${data.declining} dec`,
              color: adHealthy ? 'text-emerald-400' : data.ad_ratio >= 1 ? 'text-amber-400' : 'text-red-400',
            },
            {
              label: 'New Highs',
              value: data.new_highs,
              sublabel: `vs ${data.new_lows} new lows`,
              color: hlHealthy ? 'text-emerald-400' : data.hl_ratio >= 1 ? 'text-amber-400' : 'text-red-400',
            },
            {
              label: 'Near 52w High',
              value: `${data.near_high_pct}%`,
              sublabel: `of ${data.total} stocks`,
            },
            {
              label: 'Breakouts',
              value: data.breaking_out,
              color: data.breaking_out >= 20 ? 'text-cyan-400' : 'text-dark-200',
            },
          ]}
        />
      </Card>

      {/* Visual gauges */}
      <Card variant="glass" className="mb-4">
        <CardHeader title="Score Distribution" />
        <div className="space-y-3">
          <BreadthGauge label="Score 70+" value={data.score_distribution.above_70} max={data.total} color="emerald" />
          <BreadthGauge label="Score 50+" value={data.score_distribution.above_50} max={data.total} color="amber" />
          <BreadthGauge label="Score < 30" value={data.score_distribution.below_30} max={data.total} color="red" />
          <div className="flex justify-between items-center pt-1 border-t border-dark-700/30">
            <span className="text-[11px] text-dark-400">Universe Avg Score</span>
            <span className="text-sm font-data text-dark-100 font-medium">{data.score_distribution.avg_score}</span>
          </div>
        </div>
      </Card>

      {/* Sector rotation */}
      <SectionLabel>Sector Rotation</SectionLabel>
      <Card variant="glass" className="mb-4">
        <div className="space-y-0">
          {data.sectors && data.sectors.filter(s => s.sector !== 'Unknown').map((s) => (
            <SectorRow key={s.sector} sector={s} />
          ))}
        </div>
      </Card>

      <div className="h-4" />
    </div>
  )
}

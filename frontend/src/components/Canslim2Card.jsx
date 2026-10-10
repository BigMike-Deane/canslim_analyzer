import { Link } from 'react-router-dom'
import { api } from '../api'
import useApi from '../hooks/useApi'
import Card, { CardHeader } from './Card'

// CANSLIM 2.0 (backend/canslim2.py): the letters rebuilt from what held up in
// point-in-time testing. Every number comes from /api/canslim2/stock/{ticker}.

const LETTERS = [
  { key: 'C', label: 'Current earnings' },
  { key: 'A', label: 'Profitability' },
  { key: 'S', label: 'Supply & demand' },
  { key: 'I', label: 'Sponsorship' },
]

export const pctColor = (p) =>
  p == null ? 'bg-dark-600' : p >= 80 ? 'bg-emerald-500' : p >= 60 ? 'bg-green-500' : p >= 40 ? 'bg-stone-400' : p >= 20 ? 'bg-rose-500' : 'bg-red-500'

const fmt = (v, d = 1) => (v == null ? '—' : Number(v).toFixed(d))

// Plain-language read of each letter's raw inputs.
export function letterFacts(key, i = {}) {
  switch (key) {
    case 'C':
      return `${i.beat_streak ?? 0} straight estimate beat${i.beat_streak === 1 ? '' : 's'} · last surprise ${i.surprise_pct > 0 ? '+' : ''}${fmt(i.surprise_pct)}%`
    case 'A':
      return i.roe == null ? 'Return on equity not available' : `Return on equity ${fmt(i.roe * 100)}%`
    case 'S': {
      const chg = i.s3 == null ? null : (Math.exp(-i.s3) - 1) * 100   // s3 = -log(now / a year ago)
      const sh = chg == null ? 'share count n/a'
        : Math.abs(chg) < 0.1 ? 'share count flat over the year'
        : `share count ${chg < 0 ? '−' : '+'}${fmt(Math.abs(chg))}% in a year${chg < 0 ? ' (buybacks)' : ''}`
      const dtc = i.dtc == null ? 'days to cover n/a' : `${fmt(i.dtc)} days to cover`
      return `${sh} · ${dtc}`
    }
    case 'I':
      return `${i.n_brokers ?? 0} broker${i.n_brokers === 1 ? '' : 's'} with a rating action in the past year`
    default:
      return ''
  }
}

// Compact CANSLIM 2.0 score for list rows (Screener, Watchlist): "2.0 99 ▲12".
export function C2Chip({ c2 }) {
  if (!c2 || c2.score_pct == null) return null
  const p = Math.round(c2.score_pct)
  const tone = p >= 80 ? 'text-emerald-400 border-emerald-500/30' : p >= 40 ? 'text-dark-200 border-dark-600' : 'text-red-400 border-red-500/30'
  const ch = c2.change
  const small = c2.segment === 'small'
  return (
    <span className={`inline-flex items-center gap-1 text-[10px] font-data px-1.5 py-0.5 rounded border ${tone} ${small ? 'border-dashed' : ''}`}
          title={`CANSLIM 2.0: ${p}/100${small ? ' among small caps' : ''}${ch != null ? `, ${ch >= 0 ? '+' : ''}${ch} since the previous day` : ''}${c2.label ? ` — ${c2.label}` : ''}`}>
      <span className="text-dark-400">2.0{small ? 'ˢ' : ''}</span>{p}
      {ch != null && Math.abs(ch) >= 5 && <span className={ch > 0 ? 'text-emerald-400' : 'text-red-400'}>{ch > 0 ? '▲' : '▼'}{Math.abs(Math.round(ch))}</span>}
    </span>
  )
}

export function LetterBar({ letter, label, pct, facts }) {
  return (
    <div>
      <div className="flex items-center gap-3">
        <span className="w-5 text-sm font-bold text-dark-100">{letter}</span>
        <div className="flex-1">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-dark-300">{label}</span>
            <span className="font-data text-dark-200">{pct == null ? '—' : `${Math.round(pct)}`}<span className="text-dark-500">/100</span></span>
          </div>
          <div className="h-1.5 rounded-full bg-dark-700 overflow-hidden">
            <div className={`h-full rounded-full ${pctColor(pct)}`} style={{ width: `${Math.max(2, pct ?? 0)}%` }} />
          </div>
        </div>
      </div>
      {facts && <div className="ml-8 mt-1 text-[11px] text-dark-400">{facts}</div>}
    </div>
  )
}

export default function Canslim2Card({ ticker }) {
  const { data, error, loading } = useApi(() => api.getCanslim2Stock(ticker), [ticker])
  if (loading && !data) return <div className="skeleton h-40 rounded-xl mb-4" />
  if (error) {
    if (error.status !== 404) return null
    return (
      <Card variant="glass" className="mb-4">
        <CardHeader title="CANSLIM 2.0" />
        <div className="text-sm text-dark-400">
          Not scored: CANSLIM 2.0 covers stocks over $5 with a market cap of $100M or more.{' '}
          <Link to="/canslim2" className="text-primary-400">How it works</Link>
        </div>
      </Card>
    )
  }
  if (!data) return null
  const top = 100 - (data.score_pct ?? 0)
  return (
    <Card as="section" aria-labelledby="sd-c2-heading" variant="glass" className="mb-4">
      <CardHeader title="CANSLIM 2.0" titleId="sd-c2-heading" subtitle={`Evidence-based letters · ${data.as_of}`}
                  action={<Link to="/canslim2" className="text-xs text-primary-400">How it works</Link>} />
      <div className="flex items-baseline gap-2 mb-1">
        <span className="text-2xl font-semibold font-data text-dark-50">{Math.round(data.score_pct)}</span>
        <span className="text-sm text-dark-400">/100</span>
        <span className="text-sm text-dark-300 ml-1">
          {top < 1 ? 'Top 1%' : `Top ${Math.ceil(top)}%`} · #{data.rank} of {data.universe?.toLocaleString()}
          {data.segment === 'small' ? ' small caps' : ''}
        </span>
      </div>
      {data.label && (
        <div className="text-xs text-amber-400/90 mb-2 flex gap-1.5">
          <span aria-hidden>◔</span><span>{data.label}</span>
        </div>
      )}
      {data.in_tilt && (
        <div className="text-xs text-dark-400 mb-3">
          Model portfolio holds it at <span className="font-data text-dark-200">{fmt(data.tilt_mult, 2)}×</span> its market weight
        </div>
      )}
      <div className="space-y-3 mt-3">
        {LETTERS.map((L) => (
          <LetterBar key={L.key} letter={L.key} label={L.label} pct={data.letters?.[L.key]} facts={letterFacts(L.key, data.inputs)} />
        ))}
      </div>
      <div className="mt-3 text-[11px] text-dark-500">
        N and L aren't scored (no signal in testing); M lives in the Lab (A1/A5). Best evidence available, not a proven edge — see How it works.
      </div>
    </Card>
  )
}

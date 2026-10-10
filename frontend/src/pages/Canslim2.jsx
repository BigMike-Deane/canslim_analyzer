import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import useApi from '../hooks/useApi'
import Card, { CardHeader } from '../components/Card'
import PageHeader from '../components/PageHeader'
import { pctColor } from '../components/Canslim2Card'

// CANSLIM 2.0 (backend/canslim2.py, docs/score-v3-plan.md): the ranking, what each
// letter measures, and what the evidence does and doesn't show. All copy that states
// a result is templated from /api/canslim2/meta.

const VIEWS = [
  { key: 'top', label: 'Top 50', params: { limit: 50 } },
  { key: 'big', label: 'Top of the 500 largest', params: { limit: 50, tiltOnly: true } },
  { key: 'bottom', label: 'Bottom 50', params: { limit: 50, bottom: true } },
]

function MiniBars({ letters }) {
  return (
    <div className="flex gap-1">
      {['C', 'A', 'S', 'I'].map((k) => (
        <div key={k} className="w-6 text-center" title={`${k}: ${letters?.[k] == null ? '—' : Math.round(letters[k])}/100`}>
          <div className="h-6 w-full bg-dark-700 rounded-sm flex items-end overflow-hidden">
            <div className={`w-full ${pctColor(letters?.[k])}`} style={{ height: `${Math.max(4, letters?.[k] ?? 0)}%` }} />
          </div>
          <div className="text-[9px] text-dark-400 mt-0.5">{k}</div>
        </div>
      ))}
    </div>
  )
}

export default function Canslim2() {
  const navigate = useNavigate()
  const [view, setView] = useState('top')
  const { data: meta } = useApi(() => api.getCanslim2Meta(), [])
  const v = VIEWS.find((x) => x.key === view)
  const { data: list, loading, error } = useApi(() => api.getCanslim2Top(v.params), [view])
  return (
    <div className="p-4 md:p-6 max-w-3xl mx-auto pb-24">
      <PageHeader title="CANSLIM 2.0" subtitle="The CANSLIM letters, rebuilt from what held up in ten years of point-in-time data" />

      {meta && (
        <Card variant="glass" className="mb-4">
          <CardHeader title="What the evidence shows" subtitle={meta.as_of ? `Scores as of ${meta.as_of} · ${meta.universe?.toLocaleString()} stocks` : 'First scores after the first data refresh'} />
          <div className="text-sm text-dark-200">{meta.evidence}</div>
          <div className="mt-2 text-xs text-dark-400">Universe: {meta.universe_rule}.</div>
        </Card>
      )}

      {meta && (
        <Card variant="glass" className="mb-4">
          <CardHeader title="The letters" />
          <div className="space-y-3">
            {Object.entries(meta.letters).map(([k, L]) => (
              <div key={k}>
                <div className="text-sm"><span className="font-bold text-dark-50 mr-2">{k}</span><span className="text-dark-100">{L.name}</span></div>
                <div className="text-xs text-dark-300 ml-6">{L.summary}</div>
                <ul className="ml-6 mt-1 space-y-0.5">
                  {meta.signals.filter((s) => s.letter === k).map((s) => (
                    <li key={s.key} className="text-[11px] text-dark-400">• {s.explain} <span className="text-dark-500">({s.direction}; test t = {s.evidence_t > 0 ? '+' : ''}{s.evidence_t})</span></li>
                  ))}
                </ul>
              </div>
            ))}
            <div className="pt-2 border-t border-dark-700/50 space-y-1">
              {Object.entries(meta.not_scored).map(([k, txt]) => (
                <div key={k} className="text-xs text-dark-400"><span className="font-bold text-dark-300 mr-2">{k}</span>{txt}</div>
              ))}
            </div>
          </div>
          <div className="mt-3 text-xs">
            <Link to="/lab" className="text-primary-400">Model portfolio in the Lab →</Link>
          </div>
        </Card>
      )}

      <div className="flex gap-2 mb-3 overflow-x-auto">
        {VIEWS.map((x) => (
          <button key={x.key} onClick={() => setView(x.key)}
                  className={`px-3 py-1.5 rounded-lg text-sm whitespace-nowrap border ${view === x.key ? 'border-primary-500 text-primary-400 bg-primary-500/10' : 'border-dark-700 text-dark-300'}`}>
            {x.label}
          </button>
        ))}
      </div>

      <Card variant="glass" padding="p-0" className="mb-4 overflow-hidden">
        {loading && !list && <div className="skeleton h-64" />}
        {error && <div className="p-4 text-sm text-red-400">Couldn't load scores: {String(error.message || error)}</div>}
        {list && !list.stocks.length && <div className="p-4 text-sm text-dark-400">No scores yet — the first refresh runs shortly after deploy.</div>}
        <div className="divide-y divide-dark-700/50">
          {(list?.stocks || []).map((s) => (
            <button key={s.ticker} onClick={() => navigate(`/stock/${s.ticker}`)}
                    className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-dark-700/30">
              <div className="w-8 text-xs text-dark-400 font-data">#{s.rank}</div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold text-dark-50">{s.ticker}</div>
                <div className="text-[11px] text-dark-400 truncate">{s.name || '—'}{s.market_cap ? ` · $${(s.market_cap / 1e9).toFixed(0)}B` : ''}</div>
              </div>
              <MiniBars letters={s.letters} />
              <div className="w-10 text-right font-data text-sm text-dark-100">{Math.round(s.score_pct)}</div>
            </button>
          ))}
        </div>
      </Card>
    </div>
  )
}

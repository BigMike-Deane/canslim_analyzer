import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import Card, { CardHeader } from './Card'

const VARIANT_LABELS = ['A baseline', 'B bonus-only', 'C veto-only', 'D both']

const VARIANT_COLORS = {
  'A baseline': 'border-l-dark-500',
  'B bonus-only': 'border-l-blue-500/60',
  'C veto-only': 'border-l-amber-500/60',
  'D both': 'border-l-purple-500/60',
}

const STAT_ROWS = [
  { key: 'return_pct', label: 'Return %', format: v => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`, higherWins: true },
  { key: 'sharpe', label: 'Sharpe', format: v => v.toFixed(2), higherWins: true },
  { key: 'max_drawdown_pct', label: 'Max DD %', format: v => `${v.toFixed(1)}%`, higherWins: false },
  { key: 'total_trades', label: 'Trades', format: v => v.toFixed(0), higherWins: null },
]

function fmtMlConfig(cfg) {
  if (!cfg || typeof cfg !== 'object') return null
  const logOnly = cfg.log_only ? 'T' : 'F'
  const minConf = cfg.min_confidence != null ? cfg.min_confidence : 0
  return `log_only=${logOnly}, min_conf=${minConf}`
}

function bestIndex(values, higherWins) {
  if (higherWins === null) return -1
  const valid = values
    .map((v, i) => ({ v, i }))
    .filter(x => x.v != null && !Number.isNaN(x.v))
  if (valid.length === 0) return -1
  valid.sort((a, b) => higherWins ? b.v - a.v : a.v - b.v)
  return valid[0].i
}

function MatrixCard({ matrix }) {
  const [expanded, setExpanded] = useState(true)
  const variants = VARIANT_LABELS.map(label => matrix.variants[label])
  const created = matrix.created_at ? new Date(matrix.created_at).toLocaleString() : ''
  const window = `${matrix.start_date || '?'} → ${matrix.end_date || '?'}`

  return (
    <Card variant="glass" className="mb-4">
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full flex items-center justify-between text-left"
      >
        <div>
          <div className="text-sm font-semibold text-dark-100">
            {matrix.strategy} <span className="text-dark-400 font-normal">{window}</span>
          </div>
          <div className="text-xs text-dark-500 mt-0.5">Created {created}</div>
        </div>
        <svg
          width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          className={`text-dark-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {expanded && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-dark-700/50">
                <th className="text-left text-dark-400 font-medium py-2 px-3 w-32">Metric</th>
                {VARIANT_LABELS.map((label, i) => {
                  const v = variants[i]
                  const cfg = fmtMlConfig(v?.ml_signal)
                  return (
                    <th key={label} className={`border-l-2 ${VARIANT_COLORS[label]} text-left py-2 px-3 align-top`}>
                      <div className="text-dark-200 font-semibold text-xs">{label}</div>
                      <div className="text-[10px] text-dark-500 font-data mt-0.5 normal-case font-normal">
                        {cfg || '—'}
                      </div>
                      {v?.id && (
                        <Link
                          to={`/backtest?id=${v.id}`}
                          className="text-[10px] text-primary-400 hover:text-primary-300 mt-1 inline-block normal-case font-normal"
                        >
                          Open #{v.id} →
                        </Link>
                      )}
                      {v?.status && v.status !== 'completed' && (
                        <div className="text-[10px] text-amber-400 mt-0.5 normal-case font-normal">
                          {v.status}
                        </div>
                      )}
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {STAT_ROWS.map(({ key, label, format, higherWins }) => {
                const values = variants.map(v => v?.[key] ?? null)
                const winnerIdx = bestIndex(values, higherWins)
                return (
                  <tr key={key} className="border-b border-dark-700/20 last:border-0">
                    <td className="py-2.5 px-3 text-dark-300 font-medium">{label}</td>
                    {values.map((val, i) => {
                      const isWinner = i === winnerIdx
                      const cellCls = val == null
                        ? 'text-dark-500'
                        : isWinner
                          ? 'text-emerald-400 font-semibold'
                          : 'text-dark-200'
                      return (
                        <td
                          key={i}
                          className={`border-l-2 ${VARIANT_COLORS[VARIANT_LABELS[i]]} py-2.5 px-3 text-right font-data ${cellCls}`}
                        >
                          {val == null ? '—' : format(val)}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

export default function MLMatrixView() {
  const [matrices, setMatrices] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    async function fetchData() {
      try {
        setLoading(true)
        const res = await api.getMLMatrices(20)
        const data = res?.data || res
        if (!cancelled) {
          setMatrices(data?.matrices || [])
          setError(null)
        }
      } catch (err) {
        if (!cancelled) setError('Failed to load ML matrices: ' + err.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    fetchData()
    return () => { cancelled = true }
  }, [])

  return (
    <div>
      <Card variant="glass" className="mb-4">
        <CardHeader title="Variants" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
          <div className="text-dark-300"><span className="text-dark-100 font-semibold">A baseline</span> &mdash; ML log-only, no influence on trading</div>
          <div className="text-dark-300"><span className="text-blue-400 font-semibold">B bonus-only</span> &mdash; ML modulates composite_score, no veto</div>
          <div className="text-dark-300"><span className="text-amber-400 font-semibold">C veto-only</span> &mdash; ML can skip entries, no score bonus</div>
          <div className="text-dark-300"><span className="text-purple-400 font-semibold">D both</span> &mdash; full activation: bonus + veto</div>
        </div>
      </Card>

      {error && (
        <Card variant="glass" className="mb-4 border-red-500/30">
          <p className="text-red-400 text-sm">{error}</p>
        </Card>
      )}

      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="w-8 h-8 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
        </div>
      )}

      {!loading && matrices.length === 0 && !error && (
        <Card variant="glass">
          <p className="text-dark-400 text-sm text-center py-8">
            No ML matrices found. Run <span className="font-data text-dark-200">POST /api/ml/compare-matrix</span> to create one.
          </p>
        </Card>
      )}

      {!loading && matrices.map((m, i) => (
        <MatrixCard key={`${m.suffix}-${i}`} matrix={m} />
      ))}
    </div>
  )
}

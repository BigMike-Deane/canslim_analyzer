import { useState, useEffect } from 'react'
import { api } from '../api'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import Card, { CardHeader } from '../components/Card'
import PageHeader from '../components/PageHeader'

const COLORS = {
  btA: '#22c55e',
  btB: '#3b82f6',
  spy: '#94a3b8',
}

const TOOLTIP_STYLE = {
  background: '#14141f',
  border: '1px solid rgba(255,255,255,0.06)',
  borderRadius: '10px',
  fontFamily: 'JetBrains Mono',
  fontSize: 12,
}

const STAT_ROWS = [
  { key: 'total_return_pct', label: 'Return %', format: v => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`, higherWins: true },
  { key: 'spy_return_pct', label: 'SPY %', format: v => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`, higherWins: true },
  { key: 'max_drawdown_pct', label: 'Max Drawdown', format: v => `${v.toFixed(1)}%`, higherWins: false },
  { key: 'sharpe_ratio', label: 'Sharpe Ratio', format: v => v.toFixed(2), higherWins: true },
  { key: 'win_rate', label: 'Win Rate', format: v => `${v.toFixed(1)}%`, higherWins: true },
  { key: 'total_trades', label: 'Total Trades', format: v => v.toFixed(0), higherWins: null },
]

function ComparisonChart({ chartData, backtests }) {
  if (!chartData || chartData.length < 2) {
    return (
      <Card variant="glass" className="mb-4 h-48 flex items-center justify-center text-dark-400">
        No chart data available
      </Card>
    )
  }

  const btAKey = `bt_${backtests[0]?.id}_return`
  const btBKey = `bt_${backtests[1]?.id}_return`

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    const a = payload.find(p => p.dataKey === btAKey)
    const b = payload.find(p => p.dataKey === btBKey)
    const spy = payload.find(p => p.dataKey === 'spy_return')
    return (
      <div className="bg-dark-900 border border-dark-700/60 rounded-xl px-3 py-2 shadow-lg">
        <div className="text-[10px] text-dark-500 mb-1">{label}</div>
        {a && (
          <div className="flex items-center gap-2 text-xs">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS.btA }} />
            <span className="text-dark-300 truncate max-w-[120px]">{backtests[0]?.name || `#${backtests[0]?.id}`}</span>
            <span className={`font-data ml-auto ${a.value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {a.value >= 0 ? '+' : ''}{a.value.toFixed(1)}%
            </span>
          </div>
        )}
        {b && (
          <div className="flex items-center gap-2 text-xs">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS.btB }} />
            <span className="text-dark-300 truncate max-w-[120px]">{backtests[1]?.name || `#${backtests[1]?.id}`}</span>
            <span className={`font-data ml-auto ${b.value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {b.value >= 0 ? '+' : ''}{b.value.toFixed(1)}%
            </span>
          </div>
        )}
        {spy && (
          <div className="flex items-center gap-2 text-xs">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS.spy }} />
            <span className="text-dark-300">SPY</span>
            <span className="font-data ml-auto text-gray-400">
              {spy.value >= 0 ? '+' : ''}{spy.value.toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    )
  }

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Equity Curves Comparison" />
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: '#6b7280', fontFamily: 'JetBrains Mono' }}
              tickFormatter={(d) => {
                const parts = d.split('-')
                return parts.length >= 2 ? `${parts[0]}-${parts[1]}` : d
              }}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#6b7280', fontFamily: 'JetBrains Mono' }}
              tickFormatter={(v) => `${v.toFixed(0)}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11, fontFamily: 'JetBrains Mono' }}
              formatter={(value) => <span className="text-dark-300 text-xs">{value}</span>}
            />
            <Line
              type="monotone"
              dataKey={btAKey}
              name={backtests[0]?.name || `Backtest #${backtests[0]?.id}`}
              stroke={COLORS.btA}
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey={btBKey}
              name={backtests[1]?.name || `Backtest #${backtests[1]?.id}`}
              stroke={COLORS.btB}
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="spy_return"
              name="SPY"
              stroke={COLORS.spy}
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function StatsTable({ statsTable, backtests }) {
  if (!statsTable || !backtests || backtests.length < 2) return null

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Stats Comparison" />
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-dark-700/50">
              <th className="text-left text-dark-400 font-medium py-2 px-3">Metric</th>
              <th className="text-right text-dark-400 font-medium py-2 px-3">
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full" style={{ background: COLORS.btA }} />
                  {backtests[0]?.name || `#${backtests[0]?.id}`}
                </span>
              </th>
              <th className="text-right text-dark-400 font-medium py-2 px-3">
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full" style={{ background: COLORS.btB }} />
                  {backtests[1]?.name || `#${backtests[1]?.id}`}
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {STAT_ROWS.map(({ key, label, format, higherWins }) => {
              const vals = statsTable[key] || [null, null]
              const valA = vals[0]
              const valB = vals[1]
              const aWins = higherWins !== null && valA != null && valB != null
                ? (higherWins ? valA > valB : valA < valB)
                : null
              const bWins = higherWins !== null && valA != null && valB != null
                ? (higherWins ? valB > valA : valB < valA)
                : null

              return (
                <tr key={key} className="border-b border-dark-700/20 last:border-0">
                  <td className="py-2.5 px-3 text-dark-300 font-medium">{label}</td>
                  <td className={`py-2.5 px-3 text-right font-data ${aWins ? 'text-emerald-400 font-semibold' : aWins === false ? 'text-dark-400' : 'text-dark-200'}`}>
                    {valA != null ? format(valA) : '-'}
                  </td>
                  <td className={`py-2.5 px-3 text-right font-data ${bWins ? 'text-emerald-400 font-semibold' : bWins === false ? 'text-dark-400' : 'text-dark-200'}`}>
                    {valB != null ? format(valB) : '-'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

export default function BacktestCompare() {
  const [backtestList, setBacktestList] = useState([])
  const [idA, setIdA] = useState('')
  const [idB, setIdB] = useState('')
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)
  const [listLoading, setListLoading] = useState(true)
  const [error, setError] = useState(null)

  // Fetch completed backtests
  useEffect(() => {
    async function fetchList() {
      try {
        setListLoading(true)
        const res = await api.getBacktests()
        const all = res?.data?.backtests || res?.backtests || []
        const completed = all.filter(bt => bt.status === 'completed')
        setBacktestList(completed)
      } catch (err) {
        setError('Failed to load backtests: ' + err.message)
      } finally {
        setListLoading(false)
      }
    }
    fetchList()
  }, [])

  // Fetch comparison when both selected
  useEffect(() => {
    if (!idA || !idB || idA === idB) {
      setComparison(null)
      return
    }
    async function fetchComparison() {
      try {
        setLoading(true)
        setError(null)
        const res = await api.compareBacktests([Number(idA), Number(idB)])
        setComparison(res?.data || res)
      } catch (err) {
        setError('Failed to load comparison: ' + err.message)
        setComparison(null)
      } finally {
        setLoading(false)
      }
    }
    fetchComparison()
  }, [idA, idB])

  const btA = comparison?.backtests?.[0]
  const btB = comparison?.backtests?.[1]
  const returnA = btA?.total_return_pct ?? null
  const returnB = btB?.total_return_pct ?? null

  let summaryText = null
  if (returnA != null && returnB != null) {
    const diff = Math.abs(returnA - returnB).toFixed(1)
    const nameA = btA?.name || `#${btA?.id}`
    const nameB = btB?.name || `#${btB?.id}`
    if (returnA > returnB) {
      summaryText = { text: `${nameA} outperforms by ${diff}pp`, winner: 'A' }
    } else if (returnB > returnA) {
      summaryText = { text: `${nameB} outperforms by ${diff}pp`, winner: 'B' }
    } else {
      summaryText = { text: 'Both backtests have identical returns', winner: null }
    }
  }

  const formatOption = (bt) => {
    const ret = bt.total_return_pct != null ? `${bt.total_return_pct >= 0 ? '+' : ''}${bt.total_return_pct.toFixed(1)}%` : ''
    const strat = bt.strategy ? ` [${bt.strategy}]` : ''
    return `#${bt.id}${strat} ${ret} — ${bt.start_date?.slice(0, 10) || '?'} to ${bt.end_date?.slice(0, 10) || '?'}`
  }

  return (
    <div className="px-4 pt-6">
      <PageHeader
        title="Backtest Comparison"
        subtitle="Compare two backtests side by side"
        backTo="/backtest"
        backLabel="Backtests"
      />

      {/* Selectors */}
      <Card variant="glass" className="mb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-dark-400 mb-1.5 font-medium">Backtest A</label>
            <select
              value={idA}
              onChange={e => setIdA(e.target.value)}
              disabled={listLoading}
              className="w-full bg-dark-700 text-white border border-dark-600 rounded-lg p-2 text-sm focus:outline-none focus:border-primary-500 transition-colors"
            >
              <option value="">Select a backtest...</option>
              {backtestList.map(bt => (
                <option key={bt.id} value={bt.id} disabled={String(bt.id) === idB}>
                  {formatOption(bt)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-dark-400 mb-1.5 font-medium">Backtest B</label>
            <select
              value={idB}
              onChange={e => setIdB(e.target.value)}
              disabled={listLoading}
              className="w-full bg-dark-700 text-white border border-dark-600 rounded-lg p-2 text-sm focus:outline-none focus:border-primary-500 transition-colors"
            >
              <option value="">Select a backtest...</option>
              {backtestList.map(bt => (
                <option key={bt.id} value={bt.id} disabled={String(bt.id) === idA}>
                  {formatOption(bt)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      {/* Error */}
      {error && (
        <Card variant="glass" className="mb-4 border-red-500/30">
          <p className="text-red-400 text-sm">{error}</p>
        </Card>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="w-8 h-8 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
        </div>
      )}

      {/* Empty state */}
      {!loading && idA && idB && idA === idB && (
        <Card variant="glass" className="mb-4">
          <p className="text-dark-400 text-sm text-center py-8">Please select two different backtests to compare.</p>
        </Card>
      )}

      {!loading && (!idA || !idB) && (
        <Card variant="glass" className="mb-4">
          <p className="text-dark-400 text-sm text-center py-8">Select two backtests above to begin comparison.</p>
        </Card>
      )}

      {/* Results */}
      {!loading && comparison && (
        <>
          {/* Quick Summary */}
          {summaryText && (
            <Card variant="glass" className="mb-4">
              <div className="flex items-center justify-center gap-3 py-2">
                <span className="text-lg font-semibold">
                  {summaryText.winner === 'A' && <span className="text-emerald-400">{summaryText.text}</span>}
                  {summaryText.winner === 'B' && <span className="text-blue-400">{summaryText.text}</span>}
                  {summaryText.winner === null && <span className="text-dark-300">{summaryText.text}</span>}
                </span>
              </div>
            </Card>
          )}

          {/* Equity Curves */}
          <ComparisonChart
            chartData={comparison.chart_data}
            backtests={comparison.backtests}
          />

          {/* Stats Table */}
          <StatsTable
            statsTable={comparison.stats_table}
            backtests={comparison.backtests}
          />
        </>
      )}
    </div>
  )
}

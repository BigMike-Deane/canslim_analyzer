import { useState } from 'react'
import { XAxis, YAxis, ResponsiveContainer, Tooltip, Area, ComposedChart, Legend } from 'recharts'
import { api, formatCurrency, formatPercent } from '../api'
import useApi from '../hooks/useApi'
import Card, { CardHeader } from '../components/Card'
import StatGrid from '../components/StatGrid'
import PageHeader from '../components/PageHeader'
import AlertChip from '../components/AlertChip'
import CollapsedDrawer from '../components/CollapsedDrawer'
import { tooltipStyle, tooltipLabelStyle, chartAxis, chartColors } from '../components/chartTheme'

// Lab: research strategies paper-trading live on their own Alpaca paper accounts
// (backend/lab.py). Every number on this page comes from the API -- nothing is
// authored into the JSX that the data could contradict.

const pnl = (v) => (v == null ? 'text-dark-400' : v >= 0 ? 'text-emerald-400' : 'text-red-400')

const VERDICT = {
  significant_edge: { tone: 'ok', text: 'Statistically significant edge over SPY (95%).' },
  promising_insufficient_sample: { tone: 'warm', text: 'Ahead of SPY, but not yet distinguishable from luck — needs more days.' },
  no_measurable_edge: { tone: 'warm', text: 'No measurable edge over SPY so far.' },
  significant_negative: { tone: 'hot', text: 'Trailing SPY by a statistically significant margin.' },
  inconclusive_small_sample: { tone: 'ok', text: 'Too few days to judge yet.' },
}

const SIMULATED = new Set(['canslim2_tilt'])   // simulated at closing prices (no broker)

function Rebalance({ decision }) {
  const i = decision.inputs || {}
  return (
    <div className="text-sm text-dark-200">
      <span className="text-dark-400">{decision.date}: </span>
      rebalanced to <span className="font-semibold text-dark-50">{i.names}</span> stocks
      <span className="text-dark-400"> — turnover {((i.turnover ?? 0) * 100).toFixed(1)}%, cost {formatCurrency(i.cost)}</span>
      {i.top_weights?.length > 0 && (
        <div className="text-xs text-dark-400 mt-0.5">
          Largest: {i.top_weights.slice(0, 6).map(([t, w]) => `${t} ${(w * 100).toFixed(1)}%`).join(' · ')}
        </div>
      )}
      {decision.note && <div className="text-xs text-dark-400 mt-0.5">{decision.note}</div>}
    </div>
  )
}

function Signal({ decision, simulated }) {
  if (!decision) return <div className="text-sm text-dark-400">{simulated
    ? 'No rebalance yet — the first runs after the next close (5:20 PM ET).'
    : 'No decision recorded yet — the first runs ~15 minutes before the next close.'}</div>
  if (decision.inputs?.kind === 'rebalance') return <Rebalance decision={decision} />
  const i = decision.inputs || {}
  const held = Object.keys(decision.target || {}).join(', ')
  return (
    <div className="text-sm text-dark-200">
      <span className="text-dark-400">{decision.date}: </span>
      {i.index ?? 'Index'} closed at <span className="font-data">{i.index_close?.toLocaleString()}</span> vs its{' '}
      {i.sma_days}-day average <span className="font-data">{i.sma?.toLocaleString()}</span> —{' '}
      <span className={i.above ? 'text-emerald-400' : 'text-red-400'}>{i.above ? 'above' : 'below'}</span>, so the
      target is <span className="font-semibold text-dark-50">{held}</span>
      {i.exposure != null && (
        <span className="text-dark-400">
          {' '}— 12-month return {i.mkt_12m}% vs T-bills {i.cash_12m}% (momentum {i.momentum_positive ? 'positive' : 'negative'}), exposure {i.exposure}×
        </span>
      )}
      {decision.status === 'unchanged' && <span className="text-dark-400"> (already held)</span>}
      {decision.status === 'no_broker' && <span className="text-amber-400"> (not traded: broker not connected)</span>}
      {decision.status === 'error' && <span className="text-red-400"> (error: {decision.note})</span>}
    </div>
  )
}

const RANGES = ['1M', '3M', 'YTD', 'All']
const SERIES_COLORS = [chartColors.brand, chartColors.accent, chartColors.violet, chartColors.muted]

function inRange(rows, range) {
  if (!rows?.length || range === 'All') return rows || []
  const last = new Date(rows[rows.length - 1].date)
  const from = range === 'YTD' ? new Date(Date.UTC(last.getUTCFullYear(), 0, 1))
    : new Date(last.getTime() - (range === '1M' ? 31 : 92) * 86400000)
  return rows.filter((r) => new Date(r.date) >= from)
}

function RangeButtons({ range, setRange }) {
  return (
    <div className="flex gap-1">
      {RANGES.map((r) => (
        <button key={r} onClick={() => setRange(r)}
                className={`px-2 py-0.5 rounded text-[11px] border ${range === r ? 'border-primary-500 text-primary-400' : 'border-dark-700 text-dark-400'}`}>
          {r}
        </button>
      ))}
    </div>
  )
}

// All Lab strategies + SPY as % return from the start of the selected range.
function CompareChart({ strategies, range }) {
  const { data: histories } = useApi(
    () => Promise.all(strategies.map((s) => api.getLabHistory(s.name))),
    [strategies.map((s) => s.name).join(',')], { pollMs: 300000 })
  if (!histories) return <div className="skeleton h-48 rounded-xl" />
  const byDate = {}
  let spySeries = null
  histories.forEach((h, i) => {
    const rows = inRange(h, range)
    if (rows.length < 2) return
    const base = rows[0].equity
    rows.forEach((r) => { (byDate[r.date] ||= { date: r.date })[strategies[i].name] = (r.equity / base - 1) * 100 })
    if (!spySeries || rows.length > spySeries.length) spySeries = rows
  })
  if (spySeries) {
    const b = spySeries.find((r) => r.spy_value)?.spy_value
    if (b) spySeries.forEach((r) => { if (r.spy_value && byDate[r.date]) byDate[r.date].SPY = (r.spy_value / b - 1) * 100 })
  }
  const data = Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date))
  if (data.length < 2) return <div className="h-48 flex items-center justify-center text-dark-400 text-sm">Comparison starts after the second daily close.</div>
  return (
    <div className="h-60">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <XAxis dataKey="date" tick={{ fill: chartAxis.tick, fontSize: 10 }} axisLine={{ stroke: chartAxis.axisLine }} tickLine={false} minTickGap={40} />
          <YAxis tick={{ fill: chartAxis.tick, fontSize: 10 }} axisLine={false} tickLine={false} width={44} tickFormatter={(v) => `${v.toFixed(Math.abs(v) < 5 ? 1 : 0)}%`} />
          <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle}
                   formatter={(v, k) => [`${v >= 0 ? '+' : ''}${v.toFixed(2)}%`, strategies.find((s) => s.name === k)?.label || k]} />
          <Legend formatter={(k) => strategies.find((s) => s.name === k)?.label || k} wrapperStyle={{ fontSize: 11 }} />
          {strategies.map((s, i) => (
            <Area key={s.name} type="monotone" dataKey={s.name} stroke={SERIES_COLORS[i % SERIES_COLORS.length]} fill="none" strokeWidth={2} isAnimationActive={false} connectNulls />
          ))}
          <Area type="monotone" dataKey="SPY" stroke={chartColors.spy} fill="none" strokeWidth={1.5} strokeDasharray="4 3" isAnimationActive={false} connectNulls />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

function EquityChart({ history: all, range }) {
  const history = inRange(all, range)
  if (!history || history.length < 2) {
    return <div className="h-48 flex items-center justify-center text-dark-400 text-sm">Chart starts after the second daily close.</div>
  }
  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={history} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="labEq" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={chartColors.brand} stopOpacity={0.35} />
              <stop offset="100%" stopColor={chartColors.brand} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="date" tick={{ fill: chartAxis.tick, fontSize: 10 }} axisLine={{ stroke: chartAxis.axisLine }} tickLine={false} minTickGap={40} />
          <YAxis tick={{ fill: chartAxis.tick, fontSize: 10 }} axisLine={false} tickLine={false} width={52}
                 domain={['auto', 'auto']} tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`} />
          <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle}
                   formatter={(v, k) => [formatCurrency(v), k === 'equity' ? 'Strategy' : 'SPY (total return)']} />
          <Area type="monotone" dataKey="equity" stroke={chartColors.brand} fill="url(#labEq)" strokeWidth={2} isAnimationActive={false} />
          <Area type="monotone" dataKey="spy_value" stroke={chartColors.spy} fill="none" strokeWidth={1.5} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

function EdgeCard({ edge }) {
  if (!edge) return null
  if (edge.status !== 'ok') {
    return (
      <Card variant="glass" className="mb-4">
        <CardHeader title="Is the edge real?" />
        <div className="text-sm text-dark-400">Needs at least two daily closes ({edge.trading_days ?? 0} so far).</div>
      </Card>
    )
  }
  const sig = edge.alpha_significance
  const v = VERDICT[edge.edge_verdict] || VERDICT.inconclusive_small_sample
  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Is the edge real?" subtitle={`${edge.trading_days} trading days`} />
      <StatGrid columns={3} stats={[
        { label: 'Strategy return', value: formatPercent(edge.total_return_pct, true), color: pnl(edge.total_return_pct) },
        { label: 'SPY return', value: formatPercent(edge.spy_return_pct, true), color: pnl(edge.spy_return_pct) },
        { label: 'Excess', value: formatPercent(edge.excess_return_pct, true), color: pnl(edge.excess_return_pct) },
        { label: 'Beta', value: edge.beta?.toFixed(2) },
        { label: 'Sharpe (SPY)', value: `${edge.sharpe?.toFixed(2) ?? '-'} (${edge.spy_sharpe?.toFixed(2) ?? '-'})` },
        { label: 'Max DD (SPY)', value: `${formatPercent(edge.max_drawdown_pct)} (${formatPercent(edge.spy_max_drawdown_pct)})` },
      ]} />
      {sig && (
        <div className="mt-3 text-xs text-dark-400">
          Alpha (annualized) <span className={pnl(sig.alpha_annualized_pct)}>{formatPercent(sig.alpha_annualized_pct, true)}</span>
          {' · '}95% range {formatPercent(sig.alpha_annualized_ci_low_pct, true)} … {formatPercent(sig.alpha_annualized_ci_high_pct, true)}
          {' · '}p = {sig.p_value}
        </div>
      )}
      <div className={`mt-3 text-sm ${{ hot: 'text-red-400', warm: 'text-amber-400', ok: 'text-dark-200' }[v.tone]}`}>{v.text}</div>
    </Card>
  )
}

function WeightedPositions({ positions }) {
  const shown = positions.slice(0, 15)
  return (
    <div className="divide-y divide-dark-700/50">
      {shown.map((p) => (
        <div key={p.symbol} className="flex items-center justify-between py-2 text-sm">
          <div className="font-semibold text-dark-50">{p.symbol}</div>
          <div className="text-right">
            <div className="font-data text-dark-100">{formatCurrency(p.market_value)}</div>
            <div className="text-[11px] text-dark-400">{(p.weight * 100).toFixed(2)}% of book</div>
          </div>
        </div>
      ))}
      {positions.length > shown.length && (
        <div className="py-2 text-xs text-dark-400">…and {positions.length - shown.length} more</div>
      )}
    </div>
  )
}

function Positions({ positions, equity, simulated }) {
  if (!positions?.length) return <div className="text-sm text-dark-400">{simulated ? 'Not started — positions appear after the first rebalance.' : 'No positions (all cash).'}</div>
  if (positions[0].weight != null) return <WeightedPositions positions={positions} />
  return (
    <div className="divide-y divide-dark-700/50">
      {positions.map((p) => (
        <div key={p.symbol} className="flex items-center justify-between py-2 text-sm">
          <div>
            <div className="font-semibold text-dark-50">{p.symbol}</div>
            <div className="text-[11px] text-dark-400 font-data">{p.qty} sh @ {formatCurrency(p.avg_entry_price)}</div>
          </div>
          <div className="text-right">
            <div className="font-data text-dark-100">{formatCurrency(p.market_value)}</div>
            <div className="text-[11px] text-dark-400">
              {equity ? `${((p.market_value / equity) * 100).toFixed(0)}% of book · ` : ''}
              <span className={pnl(p.unrealized_plpc)}>{formatPercent((p.unrealized_plpc ?? 0) * 100, true)}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function Trades({ trades }) {
  if (!trades?.length) return <div className="text-sm text-dark-400">No orders yet.</div>
  return (
    <div className="divide-y divide-dark-700/50">
      {trades.map((t) => (
        <div key={t.id} className="flex items-center justify-between py-2 text-sm">
          <div>
            <span className={t.side === 'buy' ? 'text-emerald-400' : 'text-red-400'}>{t.side.toUpperCase()}</span>{' '}
            <span className="font-semibold text-dark-50">{t.symbol}</span>{' '}
            <span className="text-dark-400 font-data">{t.filled_qty ?? t.qty} sh</span>
            <div className="text-[11px] text-dark-400">{t.date} · {t.status}{t.error ? ` · ${t.error}` : ''}</div>
          </div>
          <div className="text-right font-data">
            <div className="text-dark-100">{t.filled_avg_price ? formatCurrency(t.filled_avg_price) : '—'}</div>
            {t.realized_gain != null && <div className={`text-[11px] ${pnl(t.realized_gain)}`}>{formatCurrency(t.realized_gain)}</div>}
          </div>
        </div>
      ))}
    </div>
  )
}

// Pre-registered stop rules (docs/exposure-plan.md, backend/lab_checks.py), evaluated after each close.
const RULE_LABELS = {
  M1: 'Decision every session', M2: 'Orders filled', M3: 'Holding the target fund',
  C1: 'Fill cost vs the close', C2: 'SSO tracking vs the model', P1: 'Trailing excess vs SPY', P2: 'Drawdown',
}
const LEVEL = {
  ok: { cls: 'text-emerald-400', text: 'OK' },
  pending: { cls: 'text-dark-400', text: 'Waiting for data' },
  review: { cls: 'text-amber-400', text: 'Review' },
  breach: { cls: 'text-red-400', text: 'Bug — fix' },
  stop: { cls: 'text-red-400', text: 'STOP' },
}

function StopRules({ checks }) {
  if (!checks) return null
  const overall = LEVEL[checks.level] || LEVEL.pending
  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Stop rules" subtitle={checks.as_of ? `Checked after the ${checks.as_of} close` : 'First check runs after the first close'}
                  action={<span className={`text-sm font-semibold ${overall.cls}`}>{overall.text}</span>} />
      {checks.checks?.length > 0 && (
        <div className="divide-y divide-dark-700/50">
          {checks.checks.map((c) => {
            const lv = LEVEL[c.level] || LEVEL.pending
            return (
              <div key={c.rule} className="py-2">
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-dark-200">{RULE_LABELS[c.rule] || c.rule}</span>
                  <span className={`text-xs font-semibold ${lv.cls}`}>{lv.text}</span>
                </div>
                <div className="text-xs text-dark-400 mt-0.5 break-words">{c.detail}</div>
              </div>
            )
          })}
        </div>
      )}
      <div className="mt-2 text-[11px] text-dark-500">Set before the first trade. Review = look into it; Bug = fix the plumbing; STOP = the strategy ends.</div>
    </Card>
  )
}

function StrategyView({ s, range, setRange }) {
  const { data: history } = useApi(() => api.getLabHistory(s.name), [s.name], { pollMs: 300000 })
  const { data: edge } = useApi(() => api.getLabEdge(s.name), [s.name], { pollMs: 300000 })
  const { data: trades } = useApi(() => api.getLabTrades(s.name), [s.name], { pollMs: 300000 })
  const { data: decisions } = useApi(() => api.getLabDecisions(s.name), [s.name])
  const { data: checks } = useApi(() => api.getLabChecks(s.name), [s.name], { pollMs: 300000 })
  const simulated = SIMULATED.has(s.kind)
  return (
    <>
      <Card variant="glass" className="mb-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-[11px] text-dark-400">Equity{s.as_of ? ` · ${s.as_of}` : ''}</div>
            <div className="text-2xl font-semibold font-data text-dark-50">{s.equity != null ? formatCurrency(s.equity) : '—'}</div>
            {s.days > 0 && <div className="text-xs mt-1">
              <span className={pnl(s.total_return_pct)}>{formatPercent(s.total_return_pct, true)}</span>
              <span className="text-dark-400"> vs SPY </span>
              <span className={pnl(s.spy_return_pct)}>{formatPercent(s.spy_return_pct, true)}</span>
              <span className="text-dark-400"> · {s.days} day{s.days === 1 ? '' : 's'}</span>
            </div>}
          </div>
          {simulated
            ? <AlertChip tone="ok" label="Simulated" value="Closing prices, no broker" />
            : !s.broker_connected && <AlertChip tone="ok" label="Broker" value="Waiting for Alpaca keys" />}
        </div>
        <div className="mt-3"><Signal decision={s.latest_decision} simulated={simulated} /></div>
        {s.description && <div className="mt-2 text-xs text-dark-400">{s.description}</div>}
      </Card>
      <Card variant="glass" className="mb-4">
        <CardHeader title="Equity vs SPY" subtitle="SPY = same starting money, dividends reinvested"
                    action={<RangeButtons range={range} setRange={setRange} />} />
        <EquityChart history={history} range={range} />
      </Card>
      <EdgeCard edge={edge} />
      {!simulated && <StopRules checks={checks} />}
      <Card variant="glass" className="mb-4">
        <CardHeader title="Positions" subtitle={simulated && s.positions?.length ? `${s.positions.length} stocks, largest first` : undefined} />
        <Positions positions={s.positions} equity={s.equity} simulated={simulated} />
      </Card>
      {!simulated && (
        <div className="mb-4">
          <CollapsedDrawer title="Orders" badge={trades?.length ? `${trades.length}` : undefined}>
            <Trades trades={trades} />
          </CollapsedDrawer>
        </div>
      )}
      <CollapsedDrawer title={simulated ? 'Rebalance log' : 'Daily decision log'}
                       badge={decisions?.length ? `${decisions.length} ${simulated ? 'rebalances' : 'days'}` : undefined}>
        <div className="divide-y divide-dark-700/50">
          {(decisions || []).map((d) => <div key={d.date} className="py-2"><Signal decision={d} simulated={simulated} /></div>)}
        </div>
      </CollapsedDrawer>
    </>
  )
}

export default function Lab() {
  const { data: strategies, error, loading } = useApi(() => api.getLabStrategies(), [], { pollMs: 300000 })
  const [active, setActive] = useState(null)
  const [range, setRange] = useState('All')
  const list = strategies || []
  const current = list.find((s) => s.name === active) || list[0]
  return (
    <div className="p-4 md:p-6 max-w-3xl mx-auto pb-24">
      <PageHeader title="Lab" subtitle="Research strategies paper-trading live — each decision is recorded before the close it trades at" />
      {loading && !strategies && <div className="skeleton h-40 rounded-xl mb-4" />}
      {error && <Card className="mb-4 text-red-400 text-sm">Couldn't load Lab strategies: {String(error.message || error)}</Card>}
      {list.length > 1 && (
        <Card variant="glass" className="mb-4">
          <CardHeader title="All strategies vs SPY" subtitle="% return from the start of the range"
                      action={<RangeButtons range={range} setRange={setRange} />} />
          <CompareChart strategies={list} range={range} />
        </Card>
      )}
      {list.length > 1 && (
        <div className="flex gap-2 mb-4 overflow-x-auto">
          {list.map((s) => (
            <button key={s.name} onClick={() => setActive(s.name)}
                    className={`px-3 py-1.5 rounded-lg text-sm whitespace-nowrap border ${current?.name === s.name ? 'border-primary-500 text-primary-400 bg-primary-500/10' : 'border-dark-700 text-dark-300'}`}>
              {s.label}
            </button>
          ))}
        </div>
      )}
      {current && (
        <>
          {list.length === 1 && <div className="text-sm font-semibold text-dark-100 mb-2">{current.label}</div>}
          <StrategyView key={current.name} s={current} range={range} setRange={setRange} />
        </>
      )}
      {!loading && strategies && !list.length && <Card className="text-sm text-dark-400">No Lab strategies configured.</Card>}
    </div>
  )
}

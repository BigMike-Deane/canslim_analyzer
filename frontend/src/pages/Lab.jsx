import { useState } from 'react'
import { XAxis, YAxis, ResponsiveContainer, Tooltip, Area, ComposedChart } from 'recharts'
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

function Signal({ decision }) {
  if (!decision) return <div className="text-sm text-dark-400">No decision recorded yet — the first runs ~15 minutes before the next close.</div>
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

function EquityChart({ history }) {
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

function Positions({ positions, equity }) {
  if (!positions?.length) return <div className="text-sm text-dark-400">No positions (all cash).</div>
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

function StrategyView({ s }) {
  const { data: history } = useApi(() => api.getLabHistory(s.name), [s.name], { pollMs: 300000 })
  const { data: edge } = useApi(() => api.getLabEdge(s.name), [s.name], { pollMs: 300000 })
  const { data: trades } = useApi(() => api.getLabTrades(s.name), [s.name], { pollMs: 300000 })
  const { data: decisions } = useApi(() => api.getLabDecisions(s.name), [s.name])
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
          {!s.broker_connected && <AlertChip tone="ok" label="Broker" value="Waiting for Alpaca keys" />}
        </div>
        <div className="mt-3"><Signal decision={s.latest_decision} /></div>
        {s.description && <div className="mt-2 text-xs text-dark-400">{s.description}</div>}
      </Card>
      <Card variant="glass" className="mb-4">
        <CardHeader title="Equity vs SPY" subtitle="SPY = same starting money, dividends reinvested" />
        <EquityChart history={history} />
      </Card>
      <EdgeCard edge={edge} />
      <Card variant="glass" className="mb-4">
        <CardHeader title="Positions" />
        <Positions positions={s.positions} equity={s.equity} />
      </Card>
      <div className="mb-4">
        <CollapsedDrawer title="Orders" badge={trades?.length ? `${trades.length}` : undefined}>
          <Trades trades={trades} />
        </CollapsedDrawer>
      </div>
      <CollapsedDrawer title="Daily decision log" badge={decisions?.length ? `${decisions.length} days` : undefined}>
        <div className="divide-y divide-dark-700/50">
          {(decisions || []).map((d) => <div key={d.date} className="py-2"><Signal decision={d} /></div>)}
        </div>
      </CollapsedDrawer>
    </>
  )
}

export default function Lab() {
  const { data: strategies, error, loading } = useApi(() => api.getLabStrategies(), [], { pollMs: 300000 })
  const [active, setActive] = useState(null)
  const list = strategies || []
  const current = list.find((s) => s.name === active) || list[0]
  return (
    <div className="p-4 md:p-6 max-w-3xl mx-auto pb-24">
      <PageHeader title="Lab" subtitle="Research strategies paper-trading live — each decision is recorded before the close it trades at" />
      {loading && !strategies && <div className="skeleton h-40 rounded-xl mb-4" />}
      {error && <Card className="mb-4 text-red-400 text-sm">Couldn't load Lab strategies: {String(error.message || error)}</Card>}
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
          <StrategyView key={current.name} s={current} />
        </>
      )}
      {!loading && strategies && !list.length && <Card className="text-sm text-dark-400">No Lab strategies configured.</Card>}
    </div>
  )
}

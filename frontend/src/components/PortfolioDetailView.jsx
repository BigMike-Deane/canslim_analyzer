import { useState, useEffect } from 'react'
import { api, formatDateTime, formatPercent } from '../api'
import Card, { CardHeader } from './Card'
import { PnlText } from './Badge'

// ui-revamp dedup: this view now owns ONLY its unique content — signal
// attribution and exit quality. The stat tiles, positions table, and
// last-7-days trades it used to duplicate all have single homes on the
// Overview tab (ledger, SummaryCard, PositionsList, TradeHistory).
export default function PortfolioDetailView() {
  const [attribution, setAttribution] = useState(null)
  const [exitQuality, setExitQuality] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('signals')

  useEffect(() => {
    Promise.all([
      api.getSignalAttribution().catch(() => null),
      api.getExitQuality().catch(() => null),
    ]).then(([attr, exit]) => {
      setAttribution(attr)
      setExitQuality(exit)
    }).finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center min-h-[40vh]">
      <div className="w-6 h-6 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
    </div>
  )

  if (!attribution && !exitQuality) {
    return <div className="text-dark-400 py-8 text-center">No trade-quality data yet — closed trades feed this view.</div>
  }

  return (
    <div className="space-y-5">
      <div className="flex gap-1 border-b border-dark-700">
        {['signals', 'exits'].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              tab === t ? 'border-primary-500 text-primary-400' : 'border-transparent text-dark-400 hover:text-dark-200'
            }`}
          >
            {t === 'signals' ? 'Signal Attribution' : 'Exit Quality'}
          </button>
        ))}
      </div>

      {tab === 'signals' && attribution && (
        <div className="space-y-4">
          <Card>
            <CardHeader title="Performance by Entry Type" />
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-dark-400 text-xs border-b border-dark-700">
                    <th className="text-left py-2 font-medium">Entry Type</th>
                    <th className="text-right py-2 font-medium">Trades</th>
                    <th className="text-right py-2 font-medium">Win Rate</th>
                    <th className="text-right py-2 font-medium">Avg Return</th>
                    <th className="text-right py-2 font-medium">Total P&L</th>
                    <th className="text-right py-2 font-medium">Avg Hold</th>
                  </tr>
                </thead>
                <tbody>
                  {attribution.by_entry_type.map(e => (
                    <tr key={e.entry_type} className="border-b border-dark-700/50">
                      <td className="py-2 text-dark-200 font-medium capitalize">{e.entry_type}</td>
                      <td className="py-2 text-right text-dark-300">{e.trades}</td>
                      <td className={`py-2 text-right font-medium ${e.win_rate >= 60 ? 'text-green-400' : e.win_rate >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {formatPercent(e.win_rate)}
                      </td>
                      <td className={`py-2 text-right ${e.avg_return_pct > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatPercent(e.avg_return_pct, true)}
                      </td>
                      <td className="py-2 text-right"><PnlText value={e.total_pnl} prefix="$" /></td>
                      <td className="py-2 text-right text-dark-400">{e.avg_days_held}d</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card>
            <CardHeader title="Performance by Signal" />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {attribution.by_signal.map(s => (
                <div key={s.signal} className="bg-dark-900 rounded-lg p-3">
                  <div className="text-sm font-medium text-dark-200 mb-2">{s.signal}</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div><span className="text-dark-500">Trades:</span> <span className="text-dark-200">{s.trades}</span></div>
                    <div><span className="text-dark-500">Win Rate:</span> <span className={s.win_rate >= 60 ? 'text-green-400' : 'text-dark-200'}>{formatPercent(s.win_rate)}</span></div>
                    <div><span className="text-dark-500">Avg Return:</span> <span className={s.avg_return > 0 ? 'text-green-400' : 'text-red-400'}>{formatPercent(s.avg_return, true)}</span></div>
                    <div><span className="text-dark-500">Total P&L:</span> <PnlText value={s.total_pnl} prefix="$" /></div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {tab === 'exits' && exitQuality && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Total Sells', value: exitQuality.summary.total_sells },
              { label: 'Avg Peak Gain', value: formatPercent(exitQuality.summary.avg_peak_gain) },
              { label: 'Avg Exit Gain', value: formatPercent(exitQuality.summary.avg_exit_gain) },
              { label: 'Total Realized', value: <PnlText value={exitQuality.summary.total_realized} prefix="$" /> },
            ].map((s, i) => (
              <Card key={i}>
                <div className="text-xs text-dark-400">{s.label}</div>
                <div className="text-lg font-semibold text-dark-100 mt-1">{s.value}</div>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Card>
              <div className="text-xs text-dark-400 mb-1">Stop Losses</div>
              <div className="text-lg font-semibold text-red-400">{exitQuality.summary.stop_loss_count} trades</div>
              <div className="text-sm"><PnlText value={exitQuality.summary.stop_loss_total} prefix="$" /></div>
            </Card>
            <Card>
              <div className="text-xs text-dark-400 mb-1">Trailing Stops</div>
              <div className="text-lg font-semibold text-yellow-400">{exitQuality.summary.trailing_stop_count} trades</div>
              <div className="text-sm"><PnlText value={exitQuality.summary.trailing_stop_total} prefix="$" /></div>
            </Card>
            <Card>
              <div className="text-xs text-dark-400 mb-1">Partial Profits</div>
              <div className="text-lg font-semibold text-green-400">{exitQuality.summary.partial_profit_count} trades</div>
              <div className="text-sm"><PnlText value={exitQuality.summary.partial_profit_total} prefix="$" /></div>
            </Card>
          </div>

          {exitQuality.alt_configs?.length > 0 && exitQuality.alt_configs.some(c => c.trades > 0) && (
            <Card>
              <CardHeader title="Stop-Width What-If (trailing exits)" />
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-dark-400 text-xs border-b border-dark-700">
                      <th className="text-left py-2 font-medium">Config</th>
                      <th className="text-right py-2 font-medium">Est. Trailing P&L</th>
                      <th className="text-right py-2 font-medium">vs Current</th>
                      <th className="text-right py-2 font-medium">Estimate Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {exitQuality.alt_configs.map((c, i) => (
                      <tr key={i} className={`border-b border-dark-700/50 ${c.bias === 'actual' ? 'bg-dark-800/50' : ''}`}>
                        <td className="py-1.5 text-dark-200 font-medium">{c.label}</td>
                        <td className="py-1.5 text-right"><PnlText value={c.est_trailing_pnl} prefix="$" /></td>
                        <td className="py-1.5 text-right">
                          {c.bias === 'actual' ? <span className="text-dark-500">—</span> : <PnlText value={c.delta_vs_current} prefix="$" />}
                        </td>
                        <td className="py-1.5 text-right">
                          <span className={`text-xs px-1.5 py-0.5 rounded ${
                            c.bias === 'actual' ? 'bg-dark-700 text-dark-300'
                              : c.bias === 'upper_bound' ? 'bg-yellow-900/40 text-yellow-400'
                              : 'bg-blue-900/40 text-blue-400'
                          }`}>
                            {c.bias === 'actual' ? 'actual' : c.bias === 'upper_bound' ? 'optimistic ceiling' : 'pessimistic floor'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {exitQuality.alt_configs_note && (
                <p className="text-xs text-dark-500 mt-2">{exitQuality.alt_configs_note}</p>
              )}
            </Card>
          )}

          <Card>
            <CardHeader title="Recent Exits" />
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-dark-400 text-xs border-b border-dark-700">
                    <th className="text-left py-2 font-medium">Ticker</th>
                    <th className="text-left py-2 font-medium">Reason</th>
                    <th className="text-right py-2 font-medium">Exit Gain</th>
                    <th className="text-right py-2 font-medium">Peak Gain</th>
                    <th className="text-right py-2 font-medium">P&L</th>
                    <th className="text-right py-2 font-medium">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {exitQuality.trades.map((t, i) => (
                    <tr key={i} className="border-b border-dark-700/50">
                      <td className="py-1.5 text-dark-200 font-medium">{t.ticker}</td>
                      <td className="py-1.5 text-dark-400 text-xs">{t.sell_reason}</td>
                      <td className={`py-1.5 text-right ${t.gain_pct > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatPercent(t.gain_pct, true)}
                      </td>
                      <td className="py-1.5 text-right text-dark-300">{formatPercent(t.peak_gain_pct)}</td>
                      <td className="py-1.5 text-right"><PnlText value={t.actual_pnl} prefix="$" /></td>
                      <td className="py-1.5 text-right text-dark-500 text-xs">{formatDateTime(t.date)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

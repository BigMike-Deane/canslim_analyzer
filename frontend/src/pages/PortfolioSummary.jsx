import { useState, useEffect } from 'react'
import { api } from '../api'

function Card({ title, children, className = '' }) {
  return (
    <div className={`bg-dark-800 rounded-lg border border-dark-700 p-4 ${className}`}>
      {title && <h3 className="text-sm font-semibold text-dark-200 uppercase tracking-wide mb-3">{title}</h3>}
      {children}
    </div>
  )
}

function PnlText({ value, pct }) {
  if (value == null) return <span className="text-dark-500">-</span>
  const color = value > 0 ? 'text-green-400' : value < 0 ? 'text-red-400' : 'text-dark-400'
  return (
    <span className={color}>
      ${Math.abs(value).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
      {pct != null && <span className="text-xs ml-1">({pct > 0 ? '+' : ''}{pct.toFixed(1)}%)</span>}
    </span>
  )
}

export default function PortfolioSummary() {
  const [data, setData] = useState(null)
  const [attribution, setAttribution] = useState(null)
  const [exitQuality, setExitQuality] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('positions')

  useEffect(() => {
    Promise.all([
      api.getPortfolioSummary(),
      api.getSignalAttribution().catch(() => null),
      api.getExitQuality().catch(() => null),
    ]).then(([summary, attr, exit]) => {
      setData(summary)
      setAttribution(attr)
      setExitQuality(exit)
    }).finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="p-6 flex items-center justify-center min-h-[60vh]">
      <div className="w-6 h-6 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
    </div>
  )

  if (!data) return <div className="p-6 text-dark-400">No portfolio data</div>

  const { portfolio, positions, recent_trades } = data

  return (
    <div className="p-4 md:p-6 space-y-5">
      <h1 className="text-xl font-bold text-dark-100">Portfolio Summary</h1>

      {/* Top Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { label: 'Total Value', value: `$${portfolio.total_value.toLocaleString(undefined, {maximumFractionDigits: 0})}` },
          { label: 'Cash', value: `$${portfolio.cash.toLocaleString(undefined, {maximumFractionDigits: 0})} (${portfolio.cash_pct}%)` },
          { label: 'Unrealized P&L', value: <PnlText value={portfolio.unrealized_pnl} /> },
          { label: 'Positions', value: `${portfolio.positions_count}` },
          { label: 'Strategy', value: portfolio.strategy },
        ].map((s, i) => (
          <Card key={i}>
            <div className="text-xs text-dark-400">{s.label}</div>
            <div className="text-lg font-semibold text-dark-100 mt-1">{s.value}</div>
          </Card>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-dark-700">
        {['positions', 'signals', 'exits'].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              tab === t ? 'border-primary-500 text-primary-400' : 'border-transparent text-dark-400 hover:text-dark-200'
            }`}
          >
            {t === 'positions' ? 'Positions' : t === 'signals' ? 'Signal Attribution' : 'Exit Quality'}
          </button>
        ))}
      </div>

      {/* Positions Tab */}
      {tab === 'positions' && (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-dark-400 text-xs border-b border-dark-700">
                  <th className="text-left py-2 px-2 font-medium">Ticker</th>
                  <th className="text-right py-2 px-2 font-medium">Price</th>
                  <th className="text-right py-2 px-2 font-medium">P&L</th>
                  <th className="text-right py-2 px-2 font-medium">%</th>
                  <th className="text-right py-2 px-2 font-medium">Stop</th>
                  <th className="text-right py-2 px-2 font-medium">To Stop</th>
                  <th className="text-right py-2 px-2 font-medium">Days</th>
                  <th className="text-right py-2 px-2 font-medium">Score</th>
                  <th className="text-right py-2 px-2 font-medium">% Port</th>
                </tr>
              </thead>
              <tbody>
                {positions.map(p => {
                  const stopColor = p.pct_to_stop < 3 ? 'text-red-400' : p.pct_to_stop < 5 ? 'text-yellow-400' : 'text-dark-300'
                  return (
                    <tr key={p.ticker} className="border-b border-dark-700/50 hover:bg-dark-750">
                      <td className="py-2 px-2">
                        <a href={`/stock/${p.ticker}`} className="text-primary-400 hover:underline font-medium">{p.ticker}</a>
                        {p.pyramid_count > 0 && <span className="ml-1 text-xs text-dark-500">x{p.pyramid_count + 1}</span>}
                        {p.is_growth_stock && <span className="ml-1 text-xs text-purple-400">G</span>}
                      </td>
                      <td className="py-2 px-2 text-right text-dark-200">${p.current_price}</td>
                      <td className="py-2 px-2 text-right"><PnlText value={p.gain_loss} /></td>
                      <td className={`py-2 px-2 text-right font-medium ${p.gain_loss_pct > 0 ? 'text-green-400' : p.gain_loss_pct < 0 ? 'text-red-400' : 'text-dark-400'}`}>
                        {p.gain_loss_pct > 0 ? '+' : ''}{p.gain_loss_pct}%
                      </td>
                      <td className="py-2 px-2 text-right text-dark-400">${p.stop_price}</td>
                      <td className={`py-2 px-2 text-right font-medium ${stopColor}`}>{p.pct_to_stop}%</td>
                      <td className="py-2 px-2 text-right text-dark-300">{p.days_held}d</td>
                      <td className="py-2 px-2 text-right text-dark-300">{p.current_score}</td>
                      <td className="py-2 px-2 text-right text-dark-400">{p.pct_of_portfolio}%</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Recent Trades */}
          {recent_trades.length > 0 && (
            <Card title="Last 7 Days">
              <div className="space-y-1">
                {recent_trades.map((t, i) => (
                  <div key={i} className="flex items-center justify-between text-sm py-1">
                    <div className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        t.action === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      }`}>{t.action}</span>
                      <span className="text-dark-200 font-medium">{t.ticker}</span>
                      <span className="text-dark-500">{t.shares} @ ${t.price}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      {t.pnl != null && <PnlText value={t.pnl} />}
                      <span className="text-dark-500 text-xs">{t.date}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      {/* Signal Attribution Tab */}
      {tab === 'signals' && attribution && (
        <div className="space-y-4">
          <Card title="Performance by Entry Type">
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
                      {e.win_rate}%
                    </td>
                    <td className={`py-2 text-right ${e.avg_return_pct > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {e.avg_return_pct > 0 ? '+' : ''}{e.avg_return_pct}%
                    </td>
                    <td className="py-2 text-right"><PnlText value={e.total_pnl} /></td>
                    <td className="py-2 text-right text-dark-400">{e.avg_days_held}d</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Performance by Signal">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {attribution.by_signal.map(s => (
                <div key={s.signal} className="bg-dark-900 rounded-lg p-3">
                  <div className="text-sm font-medium text-dark-200 mb-2">{s.signal}</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div><span className="text-dark-500">Trades:</span> <span className="text-dark-200">{s.trades}</span></div>
                    <div><span className="text-dark-500">Win Rate:</span> <span className={s.win_rate >= 60 ? 'text-green-400' : 'text-dark-200'}>{s.win_rate}%</span></div>
                    <div><span className="text-dark-500">Avg Return:</span> <span className={s.avg_return > 0 ? 'text-green-400' : 'text-red-400'}>{s.avg_return > 0 ? '+' : ''}{s.avg_return}%</span></div>
                    <div><span className="text-dark-500">Total P&L:</span> <PnlText value={s.total_pnl} /></div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* Exit Quality Tab */}
      {tab === 'exits' && exitQuality && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Total Sells', value: exitQuality.summary.total_sells },
              { label: 'Avg Peak Gain', value: `${exitQuality.summary.avg_peak_gain}%` },
              { label: 'Avg Exit Gain', value: `${exitQuality.summary.avg_exit_gain}%` },
              { label: 'Total Realized', value: <PnlText value={exitQuality.summary.total_realized} /> },
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
              <div className="text-sm"><PnlText value={exitQuality.summary.stop_loss_total} /></div>
            </Card>
            <Card>
              <div className="text-xs text-dark-400 mb-1">Trailing Stops</div>
              <div className="text-lg font-semibold text-yellow-400">{exitQuality.summary.trailing_stop_count} trades</div>
              <div className="text-sm"><PnlText value={exitQuality.summary.trailing_stop_total} /></div>
            </Card>
            <Card>
              <div className="text-xs text-dark-400 mb-1">Partial Profits</div>
              <div className="text-lg font-semibold text-green-400">{exitQuality.summary.partial_profit_count} trades</div>
              <div className="text-sm"><PnlText value={exitQuality.summary.partial_profit_total} /></div>
            </Card>
          </div>

          <Card title="Recent Exits">
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
                      {t.gain_pct > 0 ? '+' : ''}{t.gain_pct}%
                    </td>
                    <td className="py-1.5 text-right text-dark-300">{t.peak_gain_pct}%</td>
                    <td className="py-1.5 text-right"><PnlText value={t.actual_pnl} /></td>
                    <td className="py-1.5 text-right text-dark-500 text-xs">{t.date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}

// ── PositionSizingCard ─────────────────────────────────────────────
// Renders a derived trade ticket for a BUY-zone stock: shares, limit,
// stop, dollar risk, % of portfolio risk. Plus a "Copy Ticket" button
// that puts a Fidelity-ready order string on the clipboard so the user
// doesn't retype it from the chart.
//
// `sizing` is the tagged result from computePositionSizing — see that
// file for the kind values. Non-actionable kinds render a small
// explanation chip instead of the full ticket so the user still knows
// what's gating the action.

import { useState } from 'react'
import { formatCurrency } from '../api'
import { formatTradeTicket } from '../positionSizing'
import Card from './Card'
import { useToast } from './Toast'

export default function PositionSizingCard({ sizing, alreadyHeld = false }) {
  const toast = useToast()
  const [copied, setCopied] = useState(false)

  // Already-held trumps everything else — no point sizing a new entry
  // if the position exists. The Actions buttons handle adds separately.
  if (alreadyHeld) {
    return (
      <Card variant="glass" className="mb-3 border-l-2 border-l-primary-500/60 bg-primary-500/[0.03]">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold tracking-wider uppercase text-primary-300">Already Held</span>
          <span className="text-[10px] text-dark-500">No new sizing — manage via AI Portfolio</span>
        </div>
      </Card>
    )
  }

  if (!sizing) return null

  // Non-actionable kinds — show a one-line "why" chip, no ticket.
  if (sizing.kind !== 'actionable') {
    const tone =
      sizing.kind === 'extended' ? 'red' :
      sizing.kind === 'below_threshold' ? 'amber' :
      'dark'
    const colorCls =
      tone === 'red'   ? 'border-l-red-500/60 bg-red-500/[0.03] text-red-300' :
      tone === 'amber' ? 'border-l-amber-500/60 bg-amber-500/[0.03] text-amber-300' :
                         'border-l-dark-500/60 bg-dark-500/[0.03] text-dark-300'
    return (
      <Card variant="glass" className={`mb-3 border-l-2 ${colorCls}`}>
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] font-semibold tracking-wider uppercase">Not Actionable</span>
          <span className="text-[10px] text-dark-300">{sizing.reason}</span>
        </div>
      </Card>
    )
  }

  // ── Actionable: full ticket ────────────────────────────────────
  const ticket = formatTradeTicket(sizing)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(ticket)
      setCopied(true)
      toast.success('Trade ticket copied — paste into Fidelity order entry')
      setTimeout(() => setCopied(false), 2500)
    } catch (err) {
      // Older Safari + some mobile WebViews lack clipboard.writeText. Fall
      // back to a hidden textarea + execCommand so the button still works
      // on whatever device the user reaches for.
      try {
        const ta = document.createElement('textarea')
        ta.value = ticket
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        document.body.removeChild(ta)
        setCopied(true)
        toast.success('Trade ticket copied')
        setTimeout(() => setCopied(false), 2500)
      } catch {
        toast.error('Copy failed — long-press to select the ticket text')
      }
    }
  }

  return (
    <Card variant="glass" className="mb-3 border-l-2 border-l-emerald-500/60 bg-emerald-500/[0.04]">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] font-semibold tracking-wider uppercase text-emerald-300">Suggested Trade</span>
        <span className="text-[10px] text-emerald-200/70">{sizing.entryNote}</span>
      </div>

      {/* Stat row — keeps the eye on shares + limit, with risk below.
          Mobile-first: 4 columns at small viewports, looks fine wider. */}
      <div className="grid grid-cols-4 gap-2 mb-2">
        <div>
          <div className="text-[9px] text-dark-500 uppercase tracking-wider">Shares</div>
          <div className="text-base font-data font-semibold text-dark-50">{sizing.shares}</div>
        </div>
        <div>
          <div className="text-[9px] text-dark-500 uppercase tracking-wider">Limit</div>
          <div className="text-base font-data font-semibold text-emerald-300">${sizing.limitPrice.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-[9px] text-dark-500 uppercase tracking-wider">Stop</div>
          <div className="text-base font-data font-semibold text-red-300">${sizing.stopPrice.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-[9px] text-dark-500 uppercase tracking-wider">Cost</div>
          <div className="text-base font-data font-semibold text-dark-200">{formatCurrency(sizing.costBasis)}</div>
        </div>
      </div>

      {/* Risk row — dollars + portfolio %, so the user sees both the
          absolute downside and the relative bite. */}
      <div className="flex items-center justify-between text-[10px] text-dark-400 border-t border-dark-700/30 pt-2 mb-2.5">
        <span>
          Risk <span className="font-data text-red-300/80">{formatCurrency(sizing.riskDollars)}</span>
          {' '}({sizing.stopLossPct}% stop)
        </span>
        <span>
          <span className="font-data text-dark-200">{sizing.riskPct}%</span> of portfolio
        </span>
      </div>

      <button
        onClick={handleCopy}
        className="w-full text-xs font-medium py-2 rounded-lg bg-emerald-600/15 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-600/25 transition-colors"
      >
        {copied ? '✓ Copied' : 'Copy Trade Ticket'}
      </button>

      <div className="mt-2 text-[10px] text-dark-500 font-data text-center break-all">
        {ticket}
      </div>
    </Card>
  )
}

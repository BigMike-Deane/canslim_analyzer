// Small chip-style render of position-health output. Caller renders it
// inline on a row; tooltip (title attr) carries the full reasons list
// so the user can hover to see WHY a position got flagged.
//
// Variants:
//   compact = true  → 16px round dot, emoji only (use on dense rows)
//   compact = false → pill with emoji + label (use on detail rows)
//
// Pure visual layer over computePositionHealth — no state, safe to
// render in long lists.

import { computePositionHealth } from '../positionHealth'

const TIER_META = {
  healthy: { emoji: '🟢', label: 'Healthy', cls: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' },
  watch:   { emoji: '🟡', label: 'Watch',   cls: 'bg-amber-500/10 text-amber-400 border-amber-500/30' },
  act:     { emoji: '🔴', label: 'Act',     cls: 'bg-red-500/15 text-red-400 border-red-500/40' },
}

export default function PositionHealthChip({ position, opts, compact = false }) {
  const { tier, reasons } = computePositionHealth(position, opts)
  const meta = TIER_META[tier] || TIER_META.healthy
  const title = reasons.join(' · ')

  if (compact) {
    return (
      <span
        title={title}
        className={`inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] border ${meta.cls} shrink-0`}
        aria-label={`Position health: ${meta.label}. ${title}`}
      >
        <span className="-mt-px">{meta.emoji}</span>
      </span>
    )
  }

  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border ${meta.cls}`}
      aria-label={`Position health: ${meta.label}. ${title}`}
    >
      <span>{meta.emoji}</span>
      <span>{meta.label}</span>
    </span>
  )
}

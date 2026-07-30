// AlertChip — the UI-grammar unit for "something needs your attention".
//
// Grammar (ui-revamp, Jul-30):
//   tone 'hot'  = act now (deep drawdown, near stop)     → red stripe
//   tone 'warm' = watch (score fade, approaching cap)    → amber stripe
//   tone 'ok'   = context (cash ready, all clear)        → neutral stripe
//
// An alert that names a problem must carry its action: pass `onClick` and
// the chip renders as a button that takes the user to the decision (e.g.
// the position modal). Without `onClick` it renders as a static div — use
// that only for context chips, never for problems.
export default function AlertChip({ tone = 'ok', label, value, onClick, title }) {
  const toneCls = {
    hot: 'border-l-red-400/80 text-red-400',
    warm: 'border-l-amber-400/80 text-amber-400',
    ok: 'border-l-dark-500 text-dark-200',
  }[tone] || 'border-l-dark-500 text-dark-200'

  const inner = (
    <>
      <div className="text-[11px] text-dark-400">{label}</div>
      <div className="text-sm font-semibold font-data">{value}</div>
    </>
  )

  const base = `border border-dark-600 border-l-[3px] rounded px-3 py-1.5 text-left ${toneCls}`

  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        title={title}
        className={`${base} hover:bg-dark-750/60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary-500 transition-colors cursor-pointer`}
      >
        {inner}
      </button>
    )
  }
  return <div className={base} title={title}>{inner}</div>
}

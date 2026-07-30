const variants = {
  default: 'bg-dark-800 border-dark-700/50',
  glass: 'bg-dark-800/80 border-white/[0.04] hover:border-white/[0.08] hover:bg-dark-800/90',
  stat: 'bg-dark-850 border-dark-700/30',
  accent: 'bg-dark-800/80 border-l-2',
  flat: 'bg-transparent border-transparent',
}

const accentColors = {
  cyan: 'border-l-primary-500/60',
  green: 'border-l-emerald-500/60',
  red: 'border-l-red-500/60',
  purple: 'border-l-purple-500/60',
  amber: 'border-l-amber-500/60',
  teal: 'border-l-teal-500/60',
}

export default function Card({
  variant = 'default',
  accent,
  className = '',
  padding = 'p-4',
  rounded = 'rounded-xl',
  animate = false,
  stagger,
  onClick,
  as: Tag = 'div',
  children,
  ...rest
}) {
  const base = variants[variant] || variants.default
  const accentCls = variant === 'accent' && accent ? accentColors[accent] || '' : ''
  const animCls = animate ? 'opacity-0 animate-fade-in-up' : ''
  const staggerCls = animate && stagger ? `stagger-${stagger}` : ''
  const clickCls = onClick ? 'cursor-pointer' : ''

  // `as` lets a card render as a semantic element (e.g. <section>) and `...rest`
  // forwards ARIA props like aria-labelledby so it can act as a named landmark.
  return (
    <Tag
      className={`border ${base} ${accentCls} ${rounded} ${padding} ${animCls} ${staggerCls} ${clickCls} ${className}`}
      onClick={onClick}
      {...rest}
    >
      {children}
    </Tag>
  )
}

// One title voice for every card (ui-align, Jul-30): the small-caps
// eyebrow the ledger, chart, summary, and drawer headers already use.
// Three competing header styles (bold sentence case here, caps in
// SectionLabel/CollapsibleSection, hand-rolled spans elsewhere) were the
// main source of the page reading as separate boxes instead of a system.
export function CardHeader({ title, subtitle, action, className = '', titleId }) {
  return (
    <div className={`flex items-center justify-between mb-3 ${className}`}>
      <div>
        <h3 id={titleId} className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">{title}</h3>
        {subtitle && <p className="text-xs text-dark-500 mt-0.5 normal-case tracking-normal">{subtitle}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}

export function SectionLabel({ children, className = '', id }) {
  return (
    <div className={`flex items-center gap-2 mb-3 ${className}`}>
      <span id={id} className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">
        {children}
      </span>
      <div className="flex-1 h-px bg-dark-700/50" />
    </div>
  )
}

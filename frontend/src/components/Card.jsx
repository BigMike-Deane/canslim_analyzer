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
  children,
}) {
  const base = variants[variant] || variants.default
  const accentCls = variant === 'accent' && accent ? accentColors[accent] || '' : ''
  const animCls = animate ? 'opacity-0 animate-fade-in-up' : ''
  const staggerCls = animate && stagger ? `stagger-${stagger}` : ''
  const clickCls = onClick ? 'cursor-pointer' : ''

  return (
    <div
      className={`border ${base} ${accentCls} ${rounded} ${padding} ${animCls} ${staggerCls} ${clickCls} ${className}`}
      onClick={onClick}
    >
      {children}
    </div>
  )
}

export function CardHeader({ title, subtitle, action, className = '' }) {
  return (
    <div className={`flex items-center justify-between mb-3 ${className}`}>
      <div>
        <h3 className="text-sm font-semibold text-dark-100">{title}</h3>
        {subtitle && <p className="text-xs text-dark-400 mt-0.5">{subtitle}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}

export function SectionLabel({ children, className = '' }) {
  return (
    <div className={`flex items-center gap-2 mb-3 ${className}`}>
      <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">
        {children}
      </span>
      <div className="flex-1 h-px bg-dark-700/50" />
    </div>
  )
}

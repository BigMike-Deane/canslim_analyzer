const colMap = {
  2: 'grid-cols-2',
  3: 'grid-cols-3',
  4: 'grid-cols-4',
  5: 'grid-cols-5',
  6: 'grid-cols-6',
}

export default function StatGrid({ stats, columns, className = '' }) {
  const colCls = columns
    ? (colMap[columns] || 'grid-cols-3')
    : stats.length <= 3 ? 'grid-cols-3'
    : stats.length === 4 ? 'grid-cols-4'
    : 'grid-cols-5'

  return (
    <div className={`grid ${colCls} gap-3 ${className}`}>
      {stats.map((stat, i) => (
        <StatItem key={stat.label || i} {...stat} />
      ))}
    </div>
  )
}

function StatItem({ label, value, sublabel, color, className = '' }) {
  const colorCls = color || 'text-dark-50'

  return (
    <div className={`text-center ${className}`}>
      <div className={`text-lg font-semibold font-data ${colorCls}`}>
        {value ?? '-'}
      </div>
      <div className="text-[10px] text-dark-400 mt-0.5 leading-tight">{label}</div>
      {sublabel && (
        <div className="text-[10px] text-dark-500 mt-0.5">{sublabel}</div>
      )}
    </div>
  )
}

export function StatRow({ label, value, sublabel, className = '' }) {
  return (
    <div className={`flex items-center justify-between py-1.5 ${className}`}>
      <span className="text-xs text-dark-400">{label}</span>
      <div className="text-right">
        <span className="text-sm font-data text-dark-100">{value ?? '-'}</span>
        {sublabel && <span className="text-[10px] text-dark-500 ml-1.5">{sublabel}</span>}
      </div>
    </div>
  )
}

export function MiniStat({ label, value, color, className = '' }) {
  return (
    <div className={className}>
      <div className={`text-xs font-data font-medium ${color || 'text-dark-100'}`}>
        {value ?? '-'}
      </div>
      <div className="text-[10px] text-dark-500">{label}</div>
    </div>
  )
}

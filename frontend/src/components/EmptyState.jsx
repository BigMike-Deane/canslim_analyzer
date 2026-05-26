import Card from './Card'

// Shared empty-state. Two contexts:
//   <EmptyState message="No backtests yet" hint="Run one above." icon={...} />
//     → standalone block, wraps itself in a glass Card.
//   <EmptyState bare compact message="No active positions" />
//     → drops the Card wrapper + tightens padding, for use INSIDE an existing
//       card (dashboard tiles) where a card-in-card would look wrong.
export default function EmptyState({
  message,
  hint,
  icon,
  action,
  bare = false,
  compact = false,
  className = '',
}) {
  const pad = compact ? 'py-6' : 'py-8'

  const body = (
    <div className={`text-center ${bare ? pad : ''}`}>
      {icon && <div className="flex justify-center mb-3 text-dark-500">{icon}</div>}
      <div className="text-dark-300 text-sm">{message}</div>
      {hint && <div className="text-dark-500 text-xs mt-1">{hint}</div>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  )

  if (bare) return body

  return (
    <Card variant="glass" className={`${pad} ${className}`}>
      {body}
    </Card>
  )
}

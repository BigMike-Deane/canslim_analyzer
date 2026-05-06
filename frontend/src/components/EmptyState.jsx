import Card from './Card'

export default function EmptyState({ message, icon, action, className = '' }) {
  return (
    <Card variant="glass" className={`text-center py-8 ${className}`}>
      {icon && <div className="flex justify-center mb-3 text-dark-500">{icon}</div>}
      <div className="text-dark-300 text-sm">{message}</div>
      {action && <div className="mt-3">{action}</div>}
    </Card>
  )
}

import { Link } from 'react-router-dom'

export default function PageHeader({
  title,
  subtitle,
  backTo,
  backLabel,
  badge,
  actions,
  className = '',
}) {
  return (
    <div className={`mb-5 ${className}`}>
      {backTo && (
        <Link
          to={backTo}
          className="inline-flex items-center gap-1 text-xs text-dark-400 hover:text-dark-200 transition-colors mb-2"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M15 18l-6-6 6-6" />
          </svg>
          {backLabel || 'Back'}
        </Link>
      )}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold text-dark-50">{title}</h1>
          {badge}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      {subtitle && (
        <p className="text-xs text-dark-400 mt-1">{subtitle}</p>
      )}
    </div>
  )
}

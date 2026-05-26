// Canonical loading spinner. Amber ring with a spinning top arc — mirrors the
// app-boot spinner in App.jsx so async feedback reads consistently everywhere.
//
//   <Spinner />                      → standalone, brand amber
//   <Spinner size="xs" inline />     → inside a button, inherits text color
//   <Spinner label="Loading…" />     → centered block with caption (loading states)

const sizes = {
  xs: 'w-3 h-3 border',
  sm: 'w-4 h-4 border-2',
  md: 'w-6 h-6 border-2',
  lg: 'w-8 h-8 border-2',
}

export default function Spinner({ size = 'sm', inline = false, label, className = '' }) {
  // `inline` spinners (e.g. inside buttons) borrow the parent's currentColor so
  // they match whatever text they sit beside; standalone ones use brand amber.
  const colorCls = inline
    ? 'border-current/30 border-t-current'
    : 'border-primary-500/30 border-t-primary-500'

  const ring = (
    <span
      className={`inline-block rounded-full animate-spin ${sizes[size] || sizes.sm} ${colorCls} ${className}`}
      role="status"
      aria-label={label || 'Loading'}
    />
  )

  if (!label) return ring

  return (
    <div className="flex flex-col items-center justify-center gap-2 py-8 text-dark-400">
      {ring}
      <span className="text-xs">{label}</span>
    </div>
  )
}

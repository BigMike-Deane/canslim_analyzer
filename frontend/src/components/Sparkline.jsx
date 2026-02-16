export default function Sparkline({
  data,
  width = 80,
  height = 24,
  color,
  gradient = true,
  strokeWidth = 1.5,
  className = '',
}) {
  if (!data || data.length < 2) return null

  const values = data.map(d => typeof d === 'number' ? d : d?.value ?? d?.close ?? 0)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1

  const isPositive = values[values.length - 1] >= values[0]
  const lineColor = color || (isPositive ? '#10b981' : '#ef4444')
  const gradientId = `spark-${Math.random().toString(36).slice(2, 8)}`

  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width
    const y = height - 2 - ((v - min) / range) * (height - 4)
    return `${x},${y}`
  })

  const pathD = points.reduce((acc, pt, i) => {
    return acc + (i === 0 ? `M${pt}` : `L${pt}`)
  }, '')

  const areaD = pathD + `L${width},${height}L0,${height}Z`

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
    >
      {gradient && (
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={lineColor} stopOpacity="0.25" />
            <stop offset="100%" stopColor={lineColor} stopOpacity="0" />
          </linearGradient>
        </defs>
      )}
      {gradient && (
        <path d={areaD} fill={`url(#${gradientId})`} />
      )}
      <path
        d={pathD}
        fill="none"
        stroke={lineColor}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* End dot */}
      <circle
        cx={(values.length - 1) / (values.length - 1) * width}
        cy={height - 2 - ((values[values.length - 1] - min) / range) * (height - 4)}
        r="2"
        fill={lineColor}
      />
    </svg>
  )
}

export function SparklineBar({ data, width = 80, height = 24, className = '' }) {
  if (!data || data.length === 0) return null

  const values = data.map(d => typeof d === 'number' ? d : d?.value ?? 0)
  const max = Math.max(...values.map(Math.abs)) || 1
  const barWidth = width / values.length - 1

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className={className}>
      {values.map((v, i) => {
        const barH = (Math.abs(v) / max) * (height - 2)
        const x = i * (width / values.length) + 0.5
        const color = v >= 0 ? '#10b981' : '#ef4444'
        return (
          <rect
            key={i}
            x={x}
            y={height - barH - 1}
            width={Math.max(barWidth, 2)}
            height={barH}
            fill={color}
            opacity={0.8}
            rx={1}
          />
        )
      })}
    </svg>
  )
}

// Client-side CSV export helpers.
//
// The app builds CSVs entirely in the browser (no streaming endpoint):
// assemble the text, wrap it in a Blob, and trigger a temporary
// <a download> click. Shared by the Trade Analytics page and the AI
// Portfolio trade history — extracted here once a second consumer appeared.

export function csvEscape(v) {
  if (v === null || v === undefined) return ''
  const s = String(v)
  // RFC 4180: wrap in quotes when the value contains a comma, quote, or
  // newline; embedded quotes are escaped by doubling them.
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`
  return s
}

// Build CSV text from an ordered column list and an array of row objects.
// Each cell is row[column]; missing keys render as empty strings.
export function buildCsv(columns, rows) {
  const header = columns.join(',')
  const lines = rows.map(r => columns.map(c => csvEscape(r[c])).join(','))
  return [header, ...lines].join('\n')
}

// Trigger a browser download of `csv` text under `filename`.
export function downloadCsv(filename, csv) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

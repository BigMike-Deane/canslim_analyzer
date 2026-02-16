import { useEffect, useRef } from 'react'

export default function Modal({ open, onClose, title, children, size = 'md' }) {
  const backdropRef = useRef(null)

  useEffect(() => {
    if (open) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => { document.body.style.overflow = '' }
  }, [open])

  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape') onClose?.()
    }
    if (open) window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [open, onClose])

  if (!open) return null

  const sizes = {
    sm: 'max-w-sm',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
    full: 'max-w-full mx-4',
  }

  return (
    <div
      ref={backdropRef}
      className="fixed inset-0 z-50 flex items-end md:items-center justify-center"
      onClick={(e) => { if (e.target === backdropRef.current) onClose?.() }}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-dark-950/80 backdrop-blur-sm animate-fade-in" />

      {/* Panel — bottom sheet on mobile, centered on desktop */}
      <div className={`relative w-full ${sizes[size]} bg-dark-800 border border-dark-700/50
        rounded-t-2xl md:rounded-xl shadow-xl
        animate-slide-up md:animate-fade-in-up
        max-h-[85vh] flex flex-col`}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-4 pb-3 border-b border-dark-700/50">
          {/* Mobile drag handle */}
          <div className="absolute top-2 left-1/2 -translate-x-1/2 w-8 h-1 bg-dark-600 rounded-full md:hidden" />

          <h2 className="text-sm font-semibold text-dark-100">{title}</h2>
          <button
            onClick={onClose}
            className="text-dark-400 hover:text-dark-200 transition-colors p-1 -mr-1"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto flex-1">
          {children}
        </div>
      </div>
    </div>
  )
}

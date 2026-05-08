import React from 'react'
import Card from './Card'

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    console.error('Render error:', error, info)
  }

  handleReload = () => {
    window.location.reload()
  }

  handleHome = () => {
    window.location.href = '/'
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children
    }

    const message = this.state.error?.message || 'Unknown error'

    return (
      <div className="min-h-screen bg-dark-900 p-4 md:p-6 flex items-start justify-center">
        <div className="w-full max-w-2xl mt-12">
          <Card variant="accent" accent="red" padding="p-5 md:p-6">
            <h1 className="text-base md:text-lg font-semibold text-red-400 mb-1">
              Something went wrong
            </h1>
            <p className="text-xs text-dark-500 mb-4">
              The application hit an unexpected error. Refreshing usually fixes it.
            </p>
            <pre className="font-data text-xs text-dark-300 bg-dark-850 border border-dark-700/50 rounded-lg p-3 overflow-auto whitespace-pre-wrap break-words mb-4">
              {message}
            </pre>
            <div className="flex gap-2">
              <button
                onClick={this.handleReload}
                className="text-xs font-medium px-3 py-2 rounded-lg bg-primary-600/20 text-primary-400 border border-primary-500/30 hover:bg-primary-600/30 transition-colors"
              >
                Reload
              </button>
              <button
                onClick={this.handleHome}
                className="text-xs font-medium px-3 py-2 rounded-lg bg-dark-700/40 text-dark-200 border border-dark-700/60 hover:bg-dark-700/60 transition-colors"
              >
                Go Home
              </button>
            </div>
          </Card>
        </div>
      </div>
    )
  }
}

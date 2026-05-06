/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ─── Warm-tinted near-blacks (red undertone, not blue) ───
        // Replaces the previous cool-dark scale. All bg-dark-*, text-dark-*,
        // border-dark-* classes throughout the app pick this up automatically.
        dark: {
          950: '#0a0606',  // Page background — deepest warm black
          900: '#120909',  // Sidebar / nav background
          850: '#180c0c',  // Stat card background
          800: '#1f1416',  // Card background (was #14141f cool-blue)
          750: '#261a1c',
          700: '#2f2225',  // Borders, dividers
          600: '#473538',  // Subtle borders
          500: '#6b5559',  // Muted text
          400: '#8c7479',  // Secondary text
          300: '#a89499',  // Body text on subtle surfaces
          200: '#cbb9bd',
          100: '#e8dcde',  // Primary text
          50:  '#f5eced',
        },
        // ─── Brand red (deep crimson) ───
        // The "primary" token used for sidebar accents, active states, focus
        // rings, primary CTAs. Distinct from P&L red by hue depth: brand is
        // burgundy/crimson while P&L loss is the brighter Tailwind red-400/500.
        primary: {
          700: '#7f1d1d',  // Burgundy — pressed CTA states
          600: '#991b1b',  // Deep crimson — primary CTA bg
          500: '#b91c1c',  // Brand red — primary accent (active rail, focus)
          400: '#dc2626',  // Lifted brand — hover states, active text
          300: '#ef4444',  // (kept available; rarely needed for brand)
        },
        // ─── Accent gold ───
        // Secondary highlight for things like the search hotkey hint, info
        // badges, ML-modulation marker. Already amber-leaning before the
        // rebrand; tightened the palette to a copper/gold range for cohesion.
        accent: {
          700: '#b45309',  // Copper — pressed
          600: '#d97706',  // Amber-deep
          500: '#f59e0b',  // Gold — primary accent highlight
          400: '#fbbf24',
          300: '#fcd34d',
        },
        // ─── Semantic tokens (new) ───
        // Use these in new code instead of raw color names so future retheming
        // is one-file-edit. Existing code still works via the legacy aliases
        // above.
        brand: {
          DEFAULT: '#b91c1c',
          deep:    '#7f1d1d',
          bright:  '#dc2626',
        },
        surface: {
          DEFAULT:  '#1f1416',  // = dark-800
          elevated: '#261a1c',  // = dark-750 — modals, popovers
          sunken:   '#120909',  // = dark-900 — sidebar, nav
        },
        pnl: {
          up:        '#10b981',  // emerald-500 — gain (kept; do not theme)
          'up-soft': '#34d399',  // emerald-400 — softer chart line
          down:      '#ef4444',  // red-500 — loss (kept; do not theme)
          'down-soft':'#f87171', // red-400 — softer chart line
        },
        // Score tier colors — semantic, used by getScoreClass() consumers.
        canslim: {
          excellent: '#10b981',
          good: '#22c55e',
          average: '#eab308',
          poor: '#f97316',
          bad: '#ef4444',
        },
      },
      fontFamily: {
        sans: ['Sora', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-out forwards',
        'fade-in-up': 'fadeInUp 0.4s ease-out forwards',
        'slide-up': 'slideUp 0.3s ease-out forwards',
        'slide-down': 'slideDown 0.3s ease-out forwards',
        'pulse-dot': 'pulseDot 2s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(100%)' },
          '100%': { transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-8px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        pulseDot: {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.5', transform: 'scale(1.5)' },
        },
        glow: {
          // Re-tinted from cyan to brand red.
          '0%': { boxShadow: '0 0 5px rgba(185, 28, 28, 0.15)' },
          '100%': { boxShadow: '0 0 20px rgba(185, 28, 28, 0.20)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}

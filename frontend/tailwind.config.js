/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ─── Bloomberg Terminal-style warm-near-blacks ───
        // Pure-warm neutrals (slight amber undertone) so amber brand reads
        // crisply against them without competing for attention.
        dark: {
          950: '#0a0908',  // Page background — warm pure-black
          900: '#11100e',  // Sidebar / nav background
          850: '#15130f',  // Stat card background
          800: '#1a1815',  // Card surface
          750: '#232019',
          700: '#2c2820',  // Borders, dividers
          600: '#3d3628',  // Subtle borders
          500: '#5a5240',  // Muted text
          400: '#847a64',  // Secondary text
          300: '#a39875',  // Body text on subtle surfaces
          200: '#c9bfa0',
          100: '#ebe3c8',  // Primary text — warm cream-amber
          50:  '#f5edd5',
        },
        // ─── Brand amber (Bloomberg Terminal) ───
        // The canonical financial-terminal aesthetic: amber-on-black. The
        // primary scale walks from deep amber (pressed) up to canonical
        // Bloomberg amber #f59e0b (focus/accent) up to lighter amber
        // (hover). P&L red is hue-distant; loss-red still pops.
        primary: {
          700: '#b45309',  // Pressed CTA — deepest amber
          600: '#d97706',  // Primary CTA bg — amber-deep
          500: '#f59e0b',  // Brand accent — canonical Bloomberg amber
          400: '#fbbf24',  // Hover lift
          300: '#fcd34d',  // Brightest variant; cream-gold
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
          DEFAULT: '#f59e0b',
          deep:    '#b45309',
          bright:  '#fbbf24',
        },
        surface: {
          DEFAULT:  '#1a1815',  // = dark-800
          elevated: '#232019',  // = dark-750 — modals, popovers
          sunken:   '#11100e',  // = dark-900 — sidebar, nav
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
          // Brand amber rgb(245, 158, 11) — Bloomberg Terminal canonical.
          '0%': { boxShadow: '0 0 5px rgba(245, 158, 11, 0.18)' },
          '100%': { boxShadow: '0 0 20px rgba(245, 158, 11, 0.24)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}

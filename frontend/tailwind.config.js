/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ─── Warm-tinted near-blacks (copper-brown undertone) ───
        // Vintage-modern leather-bound aesthetic. P&L red stays distinct
        // from the brand by hue (copper vs scarlet) and saturation.
        dark: {
          950: '#0c0a08',  // Page background — warmest near-black
          900: '#15110d',  // Sidebar / nav background
          850: '#1a140f',  // Stat card background
          800: '#1c1714',  // Card surface
          750: '#251d18',
          700: '#2e241d',  // Borders, dividers
          600: '#3d2f23',  // Subtle borders, warm brown-tinted
          500: '#5c4a3d',  // Muted text
          400: '#8a7a66',  // Secondary text
          300: '#a89a86',  // Body text on subtle surfaces
          200: '#cbbeac',
          100: '#e8dccc',  // Primary text — warm cream
          50:  '#f5ecdc',
        },
        // ─── Brand copper (burnished) ───
        // The "primary" token used for sidebar accents, active states, focus
        // rings, primary CTAs. Distinct from P&L red by hue: brand is copper/
        // sienna; loss is the cooler Tailwind red-400/500.
        primary: {
          700: '#92400e',  // Pressed CTA — deepest copper
          600: '#b45309',  // Primary CTA bg — burnished copper
          500: '#c2410c',  // Brand accent — active rail, focus
          400: '#d97706',  // Hover lift — kept on the copper side, not pure orange
          300: '#ea580c',  // Brightest variant; rarely needed for brand
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
          DEFAULT: '#c2410c',
          deep:    '#92400e',
          bright:  '#d97706',
        },
        surface: {
          DEFAULT:  '#1c1714',  // = dark-800
          elevated: '#251d18',  // = dark-750 — modals, popovers
          sunken:   '#15110d',  // = dark-900 — sidebar, nav
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
          // Brand copper rgb(194, 65, 12).
          '0%': { boxShadow: '0 0 5px rgba(194, 65, 12, 0.15)' },
          '100%': { boxShadow: '0 0 20px rgba(194, 65, 12, 0.20)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}

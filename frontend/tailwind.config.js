/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#06060c',
          900: '#0a0a12',
          850: '#0e0e18',
          800: '#14141f',
          750: '#1a1a28',
          700: '#222233',
          600: '#333348',
          500: '#4a4a5e',
          400: '#6e6e82',
          300: '#9090a0',
          200: '#b8b8c8',
          100: '#e0e0ec',
          50:  '#f0f0f8',
        },
        primary: {
          700: '#008ba3',
          600: '#00b8d4',
          500: '#00e5ff',
          400: '#4df0ff',
          300: '#80f5ff',
        },
        accent: {
          600: '#d69e2e',
          500: '#fbbf24',
          400: '#fcd34d',
        },
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
          '0%': { boxShadow: '0 0 5px rgba(0, 229, 255, 0.1)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 229, 255, 0.15)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}

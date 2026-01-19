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
          900: '#000000',
          800: '#1c1c1e',
          700: '#2c2c2e',
          600: '#3a3a3c',
          500: '#48484a',
          400: '#8e8e93',
          300: '#aeaeb2',
          200: '#c7c7cc',
          100: '#f2f2f7',
        },
        primary: {
          500: '#007aff',
          600: '#0066d6',
          400: '#4da3ff',
        },
        canslim: {
          excellent: '#34c759',
          good: '#30d158',
          average: '#ffcc00',
          poor: '#ff9500',
          bad: '#ff3b30',
        }
      }
    },
  },
  plugins: [],
}

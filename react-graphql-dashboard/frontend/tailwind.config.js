/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      animation: {
        'pulse': 'pulse 4s ease-in-out infinite',
        'consciousness': 'consciousness 6s ease-in-out infinite',
      },
      keyframes: {
        consciousness: {
          '0%, 100%': { 
            transform: 'scale(1)', 
            opacity: '0.7',
            filter: 'hue-rotate(0deg)'
          },
          '50%': { 
            transform: 'scale(1.1)', 
            opacity: '1',
            filter: 'hue-rotate(90deg)'
          },
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'consciousness-glow': 'radial-gradient(circle, rgba(0,255,136,0.2) 0%, transparent 70%)',
      },
      colors: {
        consciousness: {
          primary: '#00ff88',
          secondary: '#00bbff',
          accent: '#ffff00',
          dark: '#0a0a1a',
          medium: '#1a1a2e',
          light: '#2a2a3e',
        },
        neural: {
          active: '#00ff88',
          inactive: '#4444ff',
          emerging: '#ffff00',
        }
      },
      fontFamily: {
        'consciousness': ['Courier New', 'monospace'],
      },
      boxShadow: {
        'consciousness': '0 0 20px rgba(0, 255, 136, 0.3)',
        'neural': '0 0 15px rgba(68, 68, 255, 0.2)',
        'emergence': '0 0 25px rgba(255, 255, 0, 0.4)',
      },
      blur: {
        'consciousness': '1px',
      }
    },
  },
  plugins: [
    function({ addUtilities }) {
      const newUtilities = {
        '.slider': {
          '-webkit-appearance': 'none',
          'appearance': 'none',
          'background': 'transparent',
          'cursor': 'pointer',
        },
        '.slider::-webkit-slider-track': {
          'background': '#374151',
          'height': '8px',
          'border-radius': '4px',
        },
        '.slider::-webkit-slider-thumb': {
          '-webkit-appearance': 'none',
          'appearance': 'none',
          'height': '20px',
          'width': '20px',
          'border-radius': '50%',
          'background': '#00ff88',
          'cursor': 'pointer',
          'box-shadow': '0 0 10px rgba(0, 255, 136, 0.5)',
        },
        '.slider::-moz-range-track': {
          'background': '#374151',
          'height': '8px',
          'border-radius': '4px',
          'border': 'none',
        },
        '.slider::-moz-range-thumb': {
          'height': '20px',
          'width': '20px',
          'border-radius': '50%',
          'background': '#00ff88',
          'cursor': 'pointer',
          'border': 'none',
          'box-shadow': '0 0 10px rgba(0, 255, 136, 0.5)',
        },
        '.consciousness-panel': {
          'background': 'linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%)',
          'border': '1px solid #2a2a3e',
          'box-shadow': '0 4px 20px rgba(0, 0, 0, 0.5)',
        },
        '.neural-glow': {
          'animation': 'consciousness 6s ease-in-out infinite',
          'filter': 'drop-shadow(0 0 10px rgba(0, 255, 136, 0.3))',
        }
      }
      addUtilities(newUtilities)
    }
  ],
}
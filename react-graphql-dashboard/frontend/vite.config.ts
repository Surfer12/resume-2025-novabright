/// <reference types="vite/client" />

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
  define: {
    // Define process.env for client-side code
    'process.env': {},
    // Define global for crypto compatibility
    global: 'globalThis',
  },
  build: {
    // Optimize bundle for production
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate Apollo Client into its own chunk
          apollo: ['@apollo/client'],
          // Separate React libraries
          react: ['react', 'react-dom'],
          // Separate UI libraries
          ui: ['@headlessui/react', '@heroicons/react', 'framer-motion'],
          // Separate charts library
          charts: ['recharts'],
        },
      },
    },
    // Enable source maps for debugging
    sourcemap: true,
    // Optimize chunk size
    chunkSizeWarningLimit: 1000,
  },
  server: {
    port: 3000,
    proxy: {
      // Proxy GraphQL requests during development
      '/graphql': {
        target: 'http://localhost:4000',
        changeOrigin: true,
      },
    },
  },
  // Enable dev optimizations
  optimizeDeps: {
    include: [
      '@apollo/client',
      'graphql',
      'crypto-hash',
    ],
  },
});
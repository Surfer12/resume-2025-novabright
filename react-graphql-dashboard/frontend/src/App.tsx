import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ApolloProvider } from '@apollo/client';
import { apolloClient } from './lib/apollo-client';
import { motion } from 'framer-motion';

// Lazy load components for better performance
const Dashboard = React.lazy(() => import('./components/Dashboard'));
const ConsciousnessVisualization = React.lazy(() => import('./components/SimpleConsciousnessVisualization'));

// Loading component with consciousness theme
const LoadingComponent: React.FC = () => (
  <div className="min-h-screen bg-gray-900 flex items-center justify-center">
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="text-center"
    >
      <div className="w-16 h-16 mx-auto mb-4 border-4 border-green-400 border-t-transparent rounded-full animate-spin"></div>
      <p className="text-green-400 text-lg font-semibold">Loading Consciousness Interface...</p>
      <p className="text-gray-400 text-sm mt-2">Initializing neural-symbolic integration</p>
    </motion.div>
  </div>
);

// Navigation component
const Navigation: React.FC = () => (
  <nav className="bg-gray-800 border-b border-gray-700 px-6 py-4">
    <div className="flex items-center justify-between max-w-7xl mx-auto">
      <div className="flex items-center space-x-8">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center space-x-3"
        >
          <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-400 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">Ψ</span>
          </div>
          <h1 className="text-xl font-bold text-white">Cognitive Dashboard</h1>
        </motion.div>
        
        <div className="flex space-x-6">
          <a
            href="/dashboard"
            className="text-gray-300 hover:text-green-400 transition-colors px-3 py-2 rounded-md text-sm font-medium"
          >
            Performance Dashboard
          </a>
          <a
            href="/consciousness"
            className="text-gray-300 hover:text-green-400 transition-colors px-3 py-2 rounded-md text-sm font-medium"
          >
            Consciousness Visualization
          </a>
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="bg-green-400/10 text-green-400 px-3 py-1 rounded-full text-xs font-medium">
          30% Latency Improvement Achieved
        </div>
        <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
          <span className="text-gray-300 text-sm">U</span>
        </div>
      </div>
    </div>
  </nav>
);

const App: React.FC = () => {
  return (
    <ApolloProvider client={apolloClient}>
      <Router>
        <div className="min-h-screen bg-gray-900 text-white">
          <Navigation />
          
          <Suspense fallback={<LoadingComponent />}>
            <Routes>
              {/* Redirect root to dashboard */}
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              
              {/* Main dashboard route */}
              <Route 
                path="/dashboard" 
                element={
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                  >
                    <Dashboard />
                  </motion.div>
                } 
              />
              
              {/* Consciousness visualization route */}
              <Route 
                path="/consciousness" 
                element={
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="p-6"
                  >
                    <ConsciousnessVisualization />
                  </motion.div>
                } 
              />
              
              {/* Performance analytics route */}
              <Route 
                path="/analytics" 
                element={
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="p-6"
                  >
                    <div className="max-w-7xl mx-auto">
                      <h1 className="text-3xl font-bold mb-6">Performance Analytics</h1>
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <Dashboard timeRange="week" />
                        <ConsciousnessVisualization />
                      </div>
                    </div>
                  </motion.div>
                } 
              />
              
              {/* Catch-all route */}
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Suspense>
        </div>
      </Router>
    </ApolloProvider>
  );
};

export default App;
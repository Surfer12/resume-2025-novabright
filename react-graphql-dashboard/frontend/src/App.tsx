import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, NavLink } from 'react-router-dom';
import { ApolloProvider } from '@apollo/client';
import { apolloClient } from './lib/apollo-client';
import { motion } from 'framer-motion';

// Lazy load components for better performance
const Dashboard = React.lazy(() => import('./components/Dashboard'));
const ConsciousnessVisualization = React.lazy(() => import('./components/ConsciousnessVisualization'));

// Loading component with consciousness theme
const LoadingComponent: React.FC = () => (
  <div className="min-h-screen bg-gray-900 flex items-center justify-center">
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="text-center"
    >
      <div className="w-16 h-16 mx-auto mb-4 border-4 border-sky-400 border-t-transparent rounded-full animate-spin"></div>
      <p className="text-sky-400 text-lg font-semibold">Initializing Cognitive Interface...</p>
      <p className="text-gray-400 text-sm mt-2">Bridging Minds and Machines</p>
    </motion.div>
  </div>
);

// Navigation component
const Navigation: React.FC = () => {
  const navLinkClasses = "px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300";
  const activeLinkClasses = "bg-gray-700/50 text-white";
  const inactiveLinkClasses = "text-gray-400 hover:bg-gray-800/60 hover:text-white";

  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass-pane">
      <nav className="container mx-auto px-6 py-3">
        <div className="flex items-center justify-between">
          <NavLink to="/dashboard" className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-sky-500 to-violet-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">Ψ</span>
            </div>
            <h1 className="text-xl font-bold tracking-wider brand-gradient-text">CognitiveAI</h1>
          </NavLink>
          
          <div className="hidden md:flex items-center space-x-2">
            <NavLink to="/dashboard" className={({ isActive }) => `${navLinkClasses} ${isActive ? activeLinkClasses : inactiveLinkClasses}`}>
              Dashboard
            </NavLink>
            <NavLink to="/consciousness" className={({ isActive }) => `${navLinkClasses} ${isActive ? activeLinkClasses : inactiveLinkClasses}`}>
              Consciousness
            </NavLink>
            <NavLink to="/analytics" className={({ isActive }) => `${navLinkClasses} ${isActive ? activeLinkClasses : inactiveLinkClasses}`}>
              Analytics
            </NavLink>
          </div>

          <div className="flex items-center space-x-4">
             <div className="bg-green-400/10 text-green-300 px-3 py-1 rounded-full text-xs font-medium border border-green-400/30">
              30% Latency Improvement
            </div>
            <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
              <span className="text-gray-300 text-sm">U</span>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
};

const App: React.FC = () => {
  return (
    <ApolloProvider client={apolloClient}>
      <Router>
        <div className="min-h-screen bg-gray-900 text-white">
          <Navigation />
          <main className="pt-20">
            <Suspense fallback={<LoadingComponent />}>
              <Routes>
                {/* Redirect root to dashboard */}
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                
                {/* Main dashboard route */}
                <Route 
                  path="/dashboard" 
                  element={
                    <motion.div
                      key="dashboard"
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
                      key="consciousness"
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
                      key="analytics"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5 }}
                      className="p-6"
                    >
                      <div className="max-w-7xl mx-auto">
                        <h1 className="text-3xl font-bold mb-6 brand-gradient-text">Performance Analytics</h1>
                        {/* Placeholder for analytics components */}
                        <div className="text-center py-12 glass-pane rounded-lg">
                          <p>Advanced analytics coming soon.</p>
                        </div>
                      </div>
                    </motion.div>
                  } 
                />
                
                {/* Catch-all route */}
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </Suspense>
          </main>
        </div>
      </Router>
    </ApolloProvider>
  );
};

export default App;
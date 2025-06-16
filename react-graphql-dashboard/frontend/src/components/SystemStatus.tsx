import React from 'react';
import { motion } from 'framer-motion';

interface SystemStatusProps {
  status: {
    status: string;
    healthScore: number;
    cpu: number;
    memory: number;
    lastUpdated: string;
  } | null;
  loading: boolean;
}

const SystemStatus: React.FC<SystemStatusProps> = ({ status, loading }) => {
  if (loading || !status) {
    return (
      <div className="bg-gray-800/80 backdrop-blur-sm p-6 rounded-lg shadow-lg h-full animate-pulse border border-gray-700">
        <div className="h-4 bg-gray-700 rounded w-3/4 mb-6"></div>
        <div className="space-y-4">
          <div className="h-3 bg-gray-700 rounded w-full"></div>
          <div className="h-3 bg-gray-700 rounded w-5/6"></div>
          <div className="h-3 bg-gray-700 rounded w-full"></div>
          <div className="h-3 bg-gray-700 rounded w-4/6"></div>
          <div className="h-3 bg-gray-700 rounded w-full"></div>
          <div className="h-3 bg-gray-700 rounded w-3/4"></div>
        </div>
      </div>
    );
  }

  const healthColor =
    status.status === 'excellent' ? 'text-green-400' :
    status.status === 'good' ? 'text-blue-400' :
    status.status === 'warning' ? 'text-yellow-400' : 'text-red-400';

  const healthRingColor =
    status.status === 'excellent' ? 'stroke-green-400' :
    status.status === 'good' ? 'stroke-blue-400' :
    status.status === 'warning' ? 'stroke-yellow-400' : 'stroke-red-400';

  return (
    <motion.div 
      className="bg-gray-800/80 backdrop-blur-sm p-6 rounded-lg shadow-lg h-full border border-gray-700"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
      <div className="flex items-center justify-center mb-6">
        <div className="relative w-32 h-32">
          <svg className="w-full h-full" viewBox="0 0 36 36">
            <path
              className="text-gray-700"
              d="M18 2.0845
                a 15.9155 15.9155 0 0 1 0 31.831
                a 15.9155 15.9155 0 0 1 0 -31.831"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            />
            <motion.path
              className={healthRingColor}
              d="M18 2.0845
                a 15.9155 15.9155 0 0 1 0 31.831
                a 15.9155 15.9155 0 0 1 0 -31.831"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeDasharray={`${status.healthScore}, 100`}
              initial={{ strokeDashoffset: 100 }}
              animate={{ strokeDashoffset: 0 }}
              transition={{ duration: 1, ease: "easeInOut" }}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={`text-2xl font-bold ${healthColor}`}>
              {status.healthScore.toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
      <div className="space-y-3 text-sm">
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Overall Health</span>
          <span className={`font-semibold ${healthColor}`}>{status.status.toUpperCase()}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-400">CPU Load</span>
          <span className="font-mono text-white">{status.cpu.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Memory Usage</span>
          <span className="font-mono text-white">{status.memory.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Network Status</span>
          <span className="font-mono text-green-400">HEALTHY</span>
        </div>
         <div className="flex justify-between items-center">
          <span className="text-gray-400">Last Updated</span>
          <span className="font-mono text-white">{new Date(status.lastUpdated).toLocaleTimeString()}</span>
        </div>
      </div>
    </motion.div>
  );
};

export default SystemStatus; 
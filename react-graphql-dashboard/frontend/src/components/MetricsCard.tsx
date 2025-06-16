import React from 'react';
import { motion } from 'framer-motion';

interface MetricsCardProps {
  metric: {
    id: string;
    type?: string;
    value?: string | number;
    changePercent?: number;
    changeIndicator: 'up' | 'down' | 'stable';
  };
}

const getMetricIcon = (type?: string) => {
  switch (type) {
    case 'CPU':
      return '🧠';
    case 'MEMORY':
      return '💾';
    case 'NETWORK':
      return '🌐';
    case 'STORAGE':
      return '💽';
    case 'REQUESTS':
        return '🚀';
    default:
      return '⚙️';
  }
};

const MetricsCard: React.FC<MetricsCardProps> = ({ metric }) => {
  const type = metric?.type || 'Metric';
  const value = metric?.value !== undefined ? String(metric.value) : 'N/A';
  const changePercent = metric?.changePercent;

  const changeColor =
    metric.changeIndicator === 'up' ? 'text-green-400' :
    metric.changeIndicator === 'down' ? 'text-red-400' : 'text-gray-500';

  const ChangeArrow =
    metric.changeIndicator === 'up' ? '↑' :
    metric.changeIndicator === 'down' ? '↓' : '-';

  return (
    <motion.div
      className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-4 shadow-lg border border-gray-700/80 transition-all duration-300 hover:border-green-400/50 hover:shadow-green-400/10"
      whileHover={{ y: -5 }}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider flex items-center">
          <span className="mr-2 text-lg">{getMetricIcon(type)}</span>
          {type}
        </h3>
        <p className={`text-sm font-semibold flex items-center ${changeColor}`}>
          {ChangeArrow} {changePercent !== undefined ? `${Math.abs(changePercent).toFixed(1)}%` : ''}
        </p>
      </div>
      <p className="text-3xl font-bold text-white tracking-tight">{value}</p>
    </motion.div>
  );
};

export default MetricsCard; 
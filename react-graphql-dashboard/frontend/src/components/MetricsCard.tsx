import React from 'react';

interface MetricsCardProps {
  metric: {
    id: string;
    type?: string;
    value?: string | number;
    changePercent?: number;
    // Add other expected metric properties here
  };
}

const MetricsCard: React.FC<MetricsCardProps> = ({ metric }) => {
  // Provide default values for metric properties to prevent runtime errors
  const type = metric?.type || 'Metric Type';
  const value = metric?.value !== undefined ? String(metric.value) : 'N/A';
  const changePercent = metric?.changePercent;
  const changeColor = changePercent && changePercent > 0 ? 'text-green-500' : changePercent && changePercent < 0 ? 'text-red-500' : 'text-gray-500';

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold text-gray-700">{type}</h3>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      <p className={`text-sm ${changeColor}`}>
        {changePercent !== undefined ? `${changePercent.toFixed(1)}%` : '-'}
      </p>
    </div>
  );
};

export default MetricsCard; 
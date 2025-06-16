import React, { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { motion } from 'framer-motion';

interface PerformanceChartProps {
  data: any[]; // Consider defining a more specific type
  performanceData: {
    queryLatency?: {
      average: number;
    }
  };
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ data, performanceData }) => {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    // Assuming the first metric's trend is representative
    return data[0]?.trend?.map((t: { timestamp: string | number; value: number }) => ({
      timestamp: new Date(t.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      value: t.value,
    })).slice(-20) || []; // Show last 20 points
  }, [data]);

  return (
    <motion.div
      className="glass-pane p-6 rounded-2xl shadow-lg h-full"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-white">Performance Overview</h3>
        <div className="text-xs font-mono text-green-300">
          Avg Latency: {performanceData?.queryLatency?.average.toFixed(0)}ms
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
          <defs>
            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.7}/>
              <stop offset="95%" stopColor="#38bdf8" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
          <XAxis dataKey="timestamp" stroke="#9ca3af" fontSize={12} tickLine={false} axisLine={false} />
          <YAxis stroke="#9ca3af" fontSize={12} tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(17, 24, 39, 0.8)',
              borderColor: 'rgba(255, 255, 255, 0.2)',
              borderRadius: '0.75rem',
            }}
            labelStyle={{ color: '#e5e7eb' }}
          />
          <Legend wrapperStyle={{ fontSize: '14px' }} />
          <Area type="monotone" dataKey="value" stroke="#38bdf8" fillOpacity={1} fill="url(#colorValue)" name="Metric Trend" />
        </AreaChart>
      </ResponsiveContainer>
    </motion.div>
  );
};

export default PerformanceChart; 
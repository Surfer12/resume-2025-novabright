import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';

const chartData = [
  { alpha: 0.1, psi: 0.466 },
  { alpha: 0.5, psi: 0.65 },
  { alpha: 1.0, psi: 0.86 },
];

const ParameterImpactChart: React.FC = () => {
  return (
    <motion.div
      className="bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700 shadow-2xl h-full"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h3 className="text-lg font-semibold text-white mb-4">
        Parameter Impact Analysis: α vs. Ψ(x)
      </h3>
      <p className="text-sm text-gray-400 mb-4">
        Illustrates how increasing the symbolic weight (α) impacts the final output (Ψ), shifting from pure neural performance to symbolic interpretability.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
          <XAxis dataKey="alpha" stroke="#cbd5e0" name="α (Symbolic Weight)" />
          <YAxis stroke="#cbd5e0" name="Ψ(x) Output" />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(30, 41, 59, 0.9)',
              borderColor: '#4a5568',
            }}
            labelStyle={{ color: '#e2e8f0' }}
          />
          <Legend wrapperStyle={{ color: '#e2e8f0' }} />
          <Line 
            type="monotone" 
            dataKey="psi" 
            stroke="#4ade80" 
            strokeWidth={2} 
            dot={{ r: 4 }}
            activeDot={{ r: 8 }}
            name="Ψ(x) Output"
          />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
};

export default ParameterImpactChart;
 
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';

const chartData = [
  { alpha: 0, psi: 0.85 },
  { alpha: 0.2, psi: 0.79 },
  { alpha: 0.4, psi: 0.73 },
  { alpha: 0.6, psi: 0.67 },
  { alpha: 0.8, psi: 0.61 },
  { alpha: 1.0, psi: 0.55 },
];

const ParameterImpactChart = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
 
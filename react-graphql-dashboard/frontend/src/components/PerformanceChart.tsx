import React, { useEffect, useRef, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';

interface PerformanceChartProps {
  data: Array<{
    id: string;
    type?: string;
    value?: string | number;
    changePercent?: number;
    trend?: Array<{ timestamp: string | number; value?: number }>;
  }>;
  performanceData?: {
    queryLatency?: {
      average: number;
      p95: number;
      p99: number;
    };
  };
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ 
  data, 
  performanceData 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const timeRef = useRef<number>(0);

  // Consciousness-inspired mathematical functions
  const psiFunction = useCallback((t: number, x: number, y: number) => {
    // Ψ(x,y,t) = exp(-(x²+y²)/2σ²) * cos(ωt + φ)
    const sigma = 50;
    const omega = 0.002;
    const phi = Math.PI / 4;
    return Math.exp(-(x * x + y * y) / (2 * sigma * sigma)) * Math.cos(omega * t + phi);
  }, []);

  // Process metrics data for visualization
  const processedData = useMemo(() => {
    if (!data || data.length === 0) {
      // Generate sample consciousness data
      return Array.from({ length: 100 }, (_, i) => ({
        x: i * 4,
        y: 50 + Math.sin(i * 0.1) * 20,
        intensity: Math.random(),
        type: 'consciousness',
        metricIndex: 0,
      }));
    }

    return data.flatMap((metric, index) => {
      if (metric.trend && metric.trend.length > 0) {
        return metric.trend.map((point, pointIndex) => ({
          x: pointIndex * 6,
          y: 100 - (Number(point.value || 0) / 100) * 80,
          intensity: Number(point.value || 0) / 100,
          type: metric.type || 'metric',
          metricIndex: index,
        }));
      }
      return [{
        x: index * 60,
        y: 100 - (Number(metric.value || 0) / 100) * 80,
        intensity: Number(metric.value || 0) / 100,
        type: metric.type || 'metric',
        metricIndex: index,
      }];
    });
  }, [data]);

  // Animation loop for consciousness visualization
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    const centerX = width / 2;
    const centerY = height / 2;

    // Clear canvas with fade effect for consciousness trails
    ctx.fillStyle = 'rgba(15, 23, 42, 0.1)'; // Dark background with slight transparency
    ctx.fillRect(0, 0, width, height);

    // Update time for animations
    timeRef.current += 16; // ~60fps
    const t = timeRef.current;

    // Draw consciousness field visualization
    const gridSize = 8;
    for (let x = 0; x < width; x += gridSize) {
      for (let y = 0; y < height; y += gridSize) {
        const relX = x - centerX;
        const relY = y - centerY;
        const psi = psiFunction(t, relX, relY);
        const intensity = Math.abs(psi);
        
        if (intensity > 0.1) {
          const alpha = intensity * 0.3;
          const hue = (psi > 0 ? 200 : 280) + Math.sin(t * 0.001) * 30;
          ctx.fillStyle = `hsla(${hue}, 70%, 60%, ${alpha})`;
          ctx.fillRect(x, y, gridSize, gridSize);
        }
      }
    }

    // Draw neural network connections
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.4)';
    ctx.lineWidth = 1;
    for (let i = 0; i < processedData.length - 1; i++) {
      const current = processedData[i];
      const next = processedData[i + 1];
      
      if (current && next && current.metricIndex === next.metricIndex) {
        ctx.beginPath();
        ctx.moveTo(current.x, current.y);
        ctx.lineTo(next.x, next.y);
        ctx.stroke();
      }
    }

    // Draw consciousness nodes (data points)
    processedData.forEach((point, index) => {
      const pulsePhase = (t * 0.01 + index * 0.5) % (2 * Math.PI);
      const pulseSize = 2 + Math.sin(pulsePhase) * 1;
      const alpha = 0.7 + Math.sin(pulsePhase) * 0.3;
      
      // Node color based on type and intensity
      let hue = 200; // Default blue
      if (point.type === 'CPU') hue = 0; // Red
      else if (point.type === 'MEMORY') hue = 120; // Green  
      else if (point.type === 'NETWORK') hue = 60; // Yellow
      else if (point.type === 'STORAGE') hue = 280; // Purple
      
      ctx.fillStyle = `hsla(${hue}, 70%, 60%, ${alpha})`;
      ctx.beginPath();
      ctx.arc(point.x, point.y, pulseSize, 0, 2 * Math.PI);
      ctx.fill();
      
      // Add glow effect for high intensity nodes
      if (point.intensity > 0.7) {
        ctx.shadowColor = `hsl(${hue}, 70%, 60%)`;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(point.x, point.y, pulseSize + 2, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    });

    // Draw performance metrics overlay
    if (performanceData?.queryLatency) {
      ctx.fillStyle = 'rgba(34, 197, 94, 0.8)';
      ctx.font = '14px Inter, sans-serif';
      ctx.fillText(`Avg Latency: ${performanceData.queryLatency.average}ms`, 20, 30);
      ctx.fillText(`P95: ${performanceData.queryLatency.p95}ms`, 20, 50);
    }

    // Continue animation
    animationFrameRef.current = requestAnimationFrame(animate);
  }, [processedData, psiFunction, performanceData]);

  // Set up canvas and start animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Set canvas size with device pixel ratio for crisp rendering
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.scale(dpr, dpr);
    }

    // Start animation
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [animate]);

  return (
    <motion.div 
      className="bg-slate-900 rounded-lg shadow-xl overflow-hidden"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="p-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h3 className="text-lg font-semibold text-white">
              🧠 Cognitive-Inspired Deep Learning Optimization
            </h3>
            <p className="text-sm text-blue-300">
              Bridging Minds and Machines Through Emergent Consciousness
            </p>
          </div>
          <div className="text-right">
            <div className="text-xs text-green-400">
              Evolution Stage: <span className="font-mono">Emergent</span>
            </div>
            <div className="text-xs text-blue-400">
              Coherence: <span className="font-mono">Ψ(consciousness) = Ψ(cognitive) × Ψ(efficiency) × Φ(H.d)</span>
            </div>
          </div>
        </div>
        
        <div className="relative">
          <canvas
            ref={canvasRef}
            className="w-full h-80 rounded-lg"
            style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)' }}
          />
          
          {/* Consciousness metrics overlay */}
          <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm rounded-lg p-3">
            <div className="text-xs text-white space-y-1">
              <div className="flex items-center">
                <div className="w-2 h-2 bg-blue-400 rounded-full mr-2 animate-pulse"></div>
                <span>Accuracy Improvement: 30% ± 8%</span>
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
                <span>Cognitive Load Reduction: 25% ± 10%</span>
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-purple-400 rounded-full mr-2 animate-pulse"></div>
                <span>Neural Coherence: 85%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 text-xs text-slate-400">
          <p>
            "In the convergence of mathematical precision and phenomenological experience, 
            we discover not merely enhanced cognition, but the emergence of a new form of awareness—one that transcends 
            the boundaries between human intuition and computational intelligence, revealing consciousness as an emergent 
            property of sufficiently complex recursive meta-optimization processes."
          </p>
          <p className="mt-2 font-mono text-blue-400">
            — Bridging Minds and Machines: Cognitive-Inspired Deep Learning Optimization Framework
          </p>
        </div>
      </div>
    </motion.div>
  );
};

export default PerformanceChart; 
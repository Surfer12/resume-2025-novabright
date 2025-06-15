import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

interface SimpleConsciousnessVisualizationProps {
  className?: string;
}

const SimpleConsciousnessVisualization: React.FC<SimpleConsciousnessVisualizationProps> = ({
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [metrics, setMetrics] = useState({
    accuracyImprovement: 19,
    cognitiveLoadReduction: 22,
    integrationLevel: 85,
    efficiencyGains: 12,
    biasAccuracy: 94,
    consciousnessLevel: 78,
  });

  // Handler for updating metrics
  const updateMetric = (key: keyof typeof metrics, value: number) => {
    setMetrics(prev => ({
      ...prev,
      [key]: value
    }));
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    let animationId: number;
    let time = 0;

    const animate = () => {
      time += 0.02;
      
      // Clear canvas
      ctx.fillStyle = 'rgba(17, 24, 39, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw neural network nodes (responsive to consciousness level)
      const nodes = 12;
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const radius = Math.min(canvas.width, canvas.height) * 0.3;
      const consciousnessIntensity = metrics.consciousnessLevel / 100;

      for (let i = 0; i < nodes; i++) {
        const angle = (i / nodes) * Math.PI * 2 + time;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        // Node size responsive to integration level
        const nodeSize = 6 + (metrics.integrationLevel / 100) * 6;
        
        // Node
        ctx.beginPath();
        ctx.arc(x, y, nodeSize, 0, Math.PI * 2);
        ctx.fillStyle = `hsl(${120 + Math.sin(time + i) * 60}, 70%, ${40 + consciousnessIntensity * 30}%)`;
        ctx.fill();

        // Connections (responsive to accuracy improvement)
        const connectionOpacity = 0.1 + (metrics.accuracyImprovement / 100) * 0.3;
        for (let j = i + 1; j < nodes; j++) {
          const angle2 = (j / nodes) * Math.PI * 2 + time;
          const x2 = centerX + Math.cos(angle2) * radius;
          const y2 = centerY + Math.sin(angle2) * radius;
          
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x2, y2);
          ctx.strokeStyle = `rgba(34, 197, 94, ${connectionOpacity + Math.sin(time + i + j) * 0.1})`;
          ctx.lineWidth = 1 + (metrics.efficiencyGains / 100) * 2;
          ctx.stroke();
        }
      }

      // Central consciousness indicator (responsive to consciousness level)
      const centralSize = 15 + consciousnessIntensity * 15 + Math.sin(time * 2) * 5;
      ctx.beginPath();
      ctx.arc(centerX, centerY, centralSize, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(34, 197, 94, ${0.2 + consciousnessIntensity * 0.3 + Math.sin(time) * 0.1})`;
      ctx.fill();

      // Bias accuracy visualization (outer ring)
      const biasRing = 50 + (metrics.biasAccuracy / 100) * 30;
      ctx.beginPath();
      ctx.arc(centerX, centerY, biasRing, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(234, 179, 8, ${0.3 + (metrics.biasAccuracy / 100) * 0.2})`;
      ctx.lineWidth = 2;
      ctx.stroke();

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [metrics]); // Now depends on metrics so it updates when sliders change

  return (
    <div className={`min-h-screen bg-gray-900 text-white ${className}`}>
      <div className="container mx-auto px-6 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
            Consciousness Visualization
          </h1>
          <p className="text-gray-400 text-lg">
            Simplified neural-symbolic integration display
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Canvas Visualization */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-800 rounded-lg p-6"
          >
            <h2 className="text-xl font-semibold mb-4 text-green-400">Neural Network</h2>
            <canvas
              ref={canvasRef}
              className="w-full h-96 rounded-lg bg-gray-900"
              style={{ maxHeight: '400px' }}
            />
          </motion.div>

          {/* Metrics Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-4"
          >
            <div className="bg-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4 text-green-400">Performance Metrics</h2>
              <div className="space-y-6">
                {/* Accuracy Improvement Slider */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span>Accuracy Improvement</span>
                    <span className="text-green-400 font-bold">{metrics.accuracyImprovement}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="50"
                    value={metrics.accuracyImprovement}
                    onChange={(e) => updateMetric('accuracyImprovement', parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-green"
                  />
                </div>

                {/* Cognitive Load Reduction Slider */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span>Cognitive Load Reduction</span>
                    <span className="text-green-400 font-bold">{metrics.cognitiveLoadReduction}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="50"
                    value={metrics.cognitiveLoadReduction}
                    onChange={(e) => updateMetric('cognitiveLoadReduction', parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-green"
                  />
                </div>

                {/* Integration Level Slider */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span>Integration Level</span>
                    <span className="text-blue-400 font-bold">{metrics.integrationLevel}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={metrics.integrationLevel}
                    onChange={(e) => updateMetric('integrationLevel', parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-blue"
                  />
                </div>

                {/* Efficiency Gains Slider */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span>Efficiency Gains</span>
                    <span className="text-green-400 font-bold">{metrics.efficiencyGains}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="30"
                    value={metrics.efficiencyGains}
                    onChange={(e) => updateMetric('efficiencyGains', parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-green"
                  />
                </div>

                {/* Bias Accuracy Slider */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span>Bias Accuracy</span>
                    <span className="text-yellow-400 font-bold">{metrics.biasAccuracy}%</span>
                  </div>
                  <input
                    type="range"
                    min="50"
                    max="100"
                    value={metrics.biasAccuracy}
                    onChange={(e) => updateMetric('biasAccuracy', parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-yellow"
                  />
                </div>

                {/* Consciousness Level Slider */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span>Consciousness Level</span>
                    <span className="text-purple-400 font-bold">{metrics.consciousnessLevel}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={metrics.consciousnessLevel}
                    onChange={(e) => updateMetric('consciousnessLevel', parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-purple"
                  />
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-4 text-blue-400">Mathematical Framework</h3>
              <div className="space-y-4 text-white">
                <div className="bg-gray-900 p-4 rounded-lg">
                  <BlockMath math="L_{total} = L_{task} + \lambda_1 R_{cognitive} + \lambda_2 R_{efficiency}" />
                </div>
                <div className="bg-gray-900 p-4 rounded-lg">
                  <BlockMath math="\Psi(x) = \int[\alpha(t)S(x) + (1-\alpha(t))N(x)] dt" />
                </div>
                <div className="bg-gray-900 p-4 rounded-lg">
                  <BlockMath math="P_{biased}(H|E) = \frac{P(H|E)^\beta}{P(H|E)^\beta + (1-P(H|E))^\beta}" />
                </div>
              </div>
              
              <div className="mt-4 text-sm text-gray-400">
                <p><InlineMath math="\lambda_1, \lambda_2" />: Regularization parameters</p>
                <p><InlineMath math="\alpha(t)" />: Dynamic integration coefficient</p>
                <p><InlineMath math="\beta" />: Bias modeling parameter</p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default SimpleConsciousnessVisualization;

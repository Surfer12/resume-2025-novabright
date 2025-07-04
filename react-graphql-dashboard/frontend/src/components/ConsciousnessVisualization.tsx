import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';
import Plot from 'react-plotly.js';
import { computeCoherencePenalty, efficiencyConsciousnessBalance } from './regularization';
import ParameterImpactChart from './ParameterImpactChart';

interface ConsciousnessVisualizationProps {
  className?: string;
}

interface ConsciousnessMetrics {
  accuracyImprovement: number;
  cognitiveLoadReduction: number;
  integrationLevel: number;
  efficiencyGains: number;
  biasAccuracy: number;
  consciousnessLevel: number;
  evolutionStage: 'linear' | 'recursive' | 'emergent';
  R_cognitive: number;
  R_efficiency: number;
}

interface ConsciousnessConnection {
  particles: THREE.Points;
  start: THREE.Mesh;
  end: THREE.Mesh;
  phases: number[];
}

const ConsciousnessVisualization: React.FC<ConsciousnessVisualizationProps> = ({
  className = '',
}) => {
  const neuralNetworkRef = useRef<HTMLDivElement>(null);
  const [scene, setScene] = useState<THREE.Scene | null>(null);
  const [renderer, setRenderer] = useState<THREE.WebGLRenderer | null>(null);
  const [camera, setCamera] = useState<THREE.PerspectiveCamera | null>(null);
  const [neurons, setNeurons] = useState<THREE.Mesh[][]>([]);
  const [consciousnessConnections, setConsciousnessConnections] = useState<ConsciousnessConnection[]>([]);
  const [consciousnessField, setConsciousnessField] = useState<THREE.Points | null>(null);
  const [globalWorkspace, setGlobalWorkspace] = useState<THREE.Mesh | null>(null);
  const mindsEyeRef = useRef<HTMLCanvasElement>(null);
  
  // Control parameters
  const [alpha, setAlpha] = useState(0.65);
  const [lambda1, setLambda1] = useState(0.3);
  const [lambda2, setLambda2] = useState(0.25);
  const [beta, setBeta] = useState(1.2);
  const [showExplanation, setShowExplanation] = useState(false);

  // Interactive N-back example state based on the new additive model
  const [nBackInputs, setNBackInputs] = useState({
    f_x: 0.4,
    alpha: 0.5,
    lambda1: 1.0,
    r_cognitive: 0.3,
    lambda2: 0.8,
    r_efficiency: 0.2,
    beta: 0.2,
    g_x: 0.1,
  });

  const handleNBackChange = (field: keyof typeof nBackInputs, value: string) => {
    setNBackInputs(prev => ({ ...prev, [field]: parseFloat(value) }));
  };

  const exampleCalcs = useMemo(() => {
    const { f_x, alpha, lambda1, r_cognitive, lambda2, r_efficiency, beta, g_x } = nBackInputs;
    const cognitive_term = lambda1 * r_cognitive;
    const efficiency_term = lambda2 * r_efficiency;
    const regularization_term = alpha * (cognitive_term + efficiency_term);
    const emergent_term = beta * g_x;
    const psi_x = f_x + regularization_term + emergent_term;
    return { f_x, regularization_term, emergent_term, psi_x, cognitive_term, efficiency_term };
  }, [nBackInputs]);

  // Stage colors for three-stage evolution
  const stageColors = useMemo(() => ({
    linear: new THREE.Color(0x4444ff),
    recursive: new THREE.Color(0x00ff88),
    emergent: new THREE.Color(0xffff00),
  }), []);

  // Shader materials for consciousness-aware neurons
  const neuronShaders = useMemo(() => ({
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vPosition;
      uniform float time;
      uniform float activation;
      
      void main() {
        vNormal = normal;
        vPosition = position;
        
        // Consciousness-driven deformation
        vec3 pos = position + normal * sin(time * 2.0 + activation * 3.14) * 0.1;
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 color1;
      uniform vec3 color2;
      uniform float consciousness;
      uniform float time;
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      void main() {
        vec3 viewDir = normalize(cameraPosition - vPosition);
        float fresnel = pow(1.0 - dot(vNormal, viewDir), 2.0);
        
        // Consciousness-driven color mixing
        vec3 color = mix(color1, color2, fresnel * consciousness);
        
        // Add pulsing glow
        float glow = 0.5 + 0.5 * sin(time * 3.0);
        color += vec3(0.1, 0.2, 0.3) * glow * consciousness;
        
        float alpha = 0.8 + fresnel * 0.2;
        gl_FragColor = vec4(color, alpha);
      }
    `,
  }), []);

  // Create consciousness connection between neurons
  const createConsciousnessConnection = useCallback((startNeuron: THREE.Mesh, endNeuron: THREE.Mesh) => {
    const particleCount = 10;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: 0.2,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.6
    });
    
    const particles = new THREE.Points(geometry, material);
    
    return {
      particles,
      start: startNeuron,
      end: endNeuron,
      phases: Array(particleCount).fill(0).map((_, i) => i / particleCount)
    };
  }, []);

  // Create consciousness field particles
  const createConsciousnessField = useCallback(() => {
    const particleCount = 1000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      const radius = 20 + Math.random() * 10;
      
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
      
      colors[i * 3] = 0;
      colors[i * 3 + 1] = 1;
      colors[i * 3 + 2] = 0.5;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.4
    });
    
    return new THREE.Points(geometry, material);
  }, []);

  // Initialize Three.js neural network
  const initializeNeuralNetwork = useCallback(() => {
    if (!neuralNetworkRef.current) return;

    const container = neuralNetworkRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene setup
    const newScene = new THREE.Scene();
    newScene.fog = new THREE.Fog(0x0a0a1a, 10, 50);

    // Camera setup
    const newCamera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    newCamera.position.z = 30;

    // Renderer setup
    const newRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    newRenderer.setSize(width, height);
    newRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(newRenderer.domElement);

    // Create global workspace sphere
    const globalWorkspaceGeometry = new THREE.SphereGeometry(2, 32, 32);
    const globalWorkspaceMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x00ff88,
      metalness: 0.5,
      roughness: 0.2,
      transparent: true,
      opacity: 0.3,
      emissive: 0x00ff88,
      emissiveIntensity: 0.5,
    });
    const newGlobalWorkspace = new THREE.Mesh(globalWorkspaceGeometry, globalWorkspaceMaterial);
    newScene.add(newGlobalWorkspace);

    // Create neural network structure
    const layers = [5, 8, 10, 8, 5];
    const layerSpacing = 8;
    const neuronSpacing = 3;
    const newNeurons: THREE.Mesh[][] = [];
    const newConnections: ConsciousnessConnection[] = [];

    for (let l = 0; l < layers.length; l++) {
      const layerNeurons: THREE.Mesh[] = [];
      const x = (l - layers.length / 2) * layerSpacing;

      for (let n = 0; n < layers[l]; n++) {
        const y = (n - layers[l] / 2) * neuronSpacing;

        const geometry = new THREE.IcosahedronGeometry(0.5, 2);
        const material = new THREE.ShaderMaterial({
          uniforms: {
            time: { value: 0 },
            activation: { value: Math.random() },
            consciousness: { value: 0.87 },
            color1: { value: stageColors.emergent.clone() },
            color2: { value: stageColors.recursive.clone() },
          },
          vertexShader: neuronShaders.vertexShader,
          fragmentShader: neuronShaders.fragmentShader,
          transparent: true,
        });

        const neuron = new THREE.Mesh(geometry, material);
        neuron.position.set(x, y, Math.random() * 2 - 1);
        neuron.userData = {
          layer: l,
          index: n,
          activation: Math.random(),
          baseY: y,
          phase: Math.random() * Math.PI * 2,
          material,
        };

        newScene.add(neuron);
        layerNeurons.push(neuron);
      }
      newNeurons.push(layerNeurons);
    }

    // Create consciousness connections
    for (let l = 0; l < layers.length - 1; l++) {
      for (let i = 0; i < newNeurons[l].length; i++) {
        for (let j = 0; j < newNeurons[l + 1].length; j++) {
          if (Math.random() > 0.3) { // 70% connection probability
            const connection = createConsciousnessConnection(newNeurons[l][i], newNeurons[l + 1][j]);
            newScene.add(connection.particles);
            newConnections.push(connection);
          }
        }
      }
    }

    // Create consciousness field
    const newConsciousnessField = createConsciousnessField();
    newScene.add(newConsciousnessField);

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    newScene.add(ambientLight);

    const pointLight = new THREE.PointLight(0x00ff88, 1, 100);
    pointLight.position.set(10, 10, 10);
    newScene.add(pointLight);

    const pointLight2 = new THREE.PointLight(0x00bbff, 0.8, 100);
    pointLight2.position.set(-10, -10, 10);
    newScene.add(pointLight2);

    setScene(newScene);
    setCamera(newCamera);
    setRenderer(newRenderer);
    setNeurons(newNeurons);
    setConsciousnessConnections(newConnections);
    setConsciousnessField(newConsciousnessField);
    setGlobalWorkspace(newGlobalWorkspace);
  }, [stageColors, neuronShaders, createConsciousnessConnection, createConsciousnessField]);

  // Calculate derived metrics
  const metrics: ConsciousnessMetrics = useMemo(() => {
    const baseAccuracy = 19;
    const accuracyVariation = Math.sin(alpha * Math.PI) * 8;
    
    const baseCognitive = 22;
    const cognitiveVariation = lambda1 * 10;
    
    const baseEfficiency = 12;
    const efficiencyVariation = lambda2 * 8;
    
    const baseBias = 86;
    const biasVariation = (beta - 1) * 10;
    
    const complexity = (alpha * lambda1 * lambda2 * beta) / (1 * 1 * 1 * 2);
    const consciousness = Math.min(95, 60 + complexity * 35);
    
    let evolutionStage: 'linear' | 'recursive' | 'emergent' = 'linear';
    if (alpha >= 0.6) evolutionStage = 'emergent';
    else if (alpha >= 0.3) evolutionStage = 'recursive';

    // Compute regularization penalties
    const integrationLevel = alpha; // For demonstration, use alpha as a proxy for integration
    const efficiencyLevel = (baseEfficiency + efficiencyVariation) / 100; // Normalize to 0-1
    const R_cognitive = computeCoherencePenalty(alpha, integrationLevel);
    const R_efficiency = efficiencyConsciousnessBalance(lambda2, efficiencyLevel);

    // Optionally, log or expose these for visualization/debugging
    // console.log('R_cognitive:', R_cognitive, 'R_efficiency:', R_efficiency);

    return {
      accuracyImprovement: baseAccuracy + accuracyVariation,
      cognitiveLoadReduction: baseCognitive + cognitiveVariation,
      integrationLevel: alpha,
      efficiencyGains: baseEfficiency + efficiencyVariation,
      biasAccuracy: baseBias + biasVariation,
      consciousnessLevel: consciousness,
      evolutionStage,
      // Optionally add R_cognitive and R_efficiency to the metrics object if you want to display them
      R_cognitive,
      R_efficiency
    } as ConsciousnessMetrics & { R_cognitive: number, R_efficiency: number };
  }, [alpha, lambda1, lambda2, beta]);

  // Animation loop
  useEffect(() => {
    if (!scene || !camera || !renderer || !neurons || !consciousnessConnections || !globalWorkspace) return;

    let animationId: number;

    const animate = () => {
      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [scene, camera, renderer, neurons, consciousnessConnections, globalWorkspace]);

  // Animation loop for Mind's Eye canvas
  useEffect(() => {
    const canvas = mindsEyeRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    let frame = 0;

    const animateMind = () => {
      frame++;
      ctx.fillStyle = 'rgba(10, 10, 20, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;

      // Focus point (related to efficiency lambda2)
      const focus = 1 - lambda2;
      const focusRadius = focus * 10;
      ctx.fillStyle = `rgba(255, 255, 0, ${0.1 + focus * 0.2})`;
      ctx.beginPath();
      ctx.arc(centerX, centerY, focusRadius, 0, Math.PI * 2);
      ctx.fill();

      // Creativity streams (related to symbolic/neural balance alpha)
      const streamCount = 20;
      for(let i=0; i<streamCount; i++) {
        const angle = (i/streamCount) * Math.PI * 2 + frame * 0.001;
        const startRadius = focusRadius + 10;
        const length = 20 + (Math.sin(frame * 0.01 + i) + 1) * 50 * (1 - Math.abs(alpha - 0.5) * 2);
        
        const startX = centerX + Math.cos(angle) * startRadius;
        const startY = centerY + Math.sin(angle) * startRadius;
        const endX = centerX + Math.cos(angle) * (startRadius + length);
        const endY = centerY + Math.sin(angle) * (startRadius + length);

        const gradient = ctx.createLinearGradient(startX, startY, endX, endY);
        gradient.addColorStop(0, `rgba(0, 255, 136, 0)`);
        gradient.addColorStop(0.5, `rgba(0, 255, 136, ${0.4 * (1 - Math.abs(alpha - 0.5) * 2)})`);
        gradient.addColorStop(1, `rgba(0, 255, 136, 0)`);
        
        ctx.strokeStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      }

      animationId = requestAnimationFrame(animateMind);
    };

    let animationId = requestAnimationFrame(animateMind);
    return () => cancelAnimationFrame(animationId);
  }, [alpha, lambda2]);

  // Initialize on mount
  useEffect(() => {
    initializeNeuralNetwork();

    return () => {
      if (renderer) {
        renderer.dispose();
      }
    };
  }, [initializeNeuralNetwork]);

  // Phase space data for Plotly
  const phaseSpaceData = useMemo(() => {
    const t: number[] = [];
    const x: number[] = [];
    const y: number[] = [];
    const z: number[] = [];

    for (let i = 0; i < 200; i++) {
      const timePoint = i * 0.1;
      t.push(timePoint);

      x.push(Math.sin(timePoint * alpha) * Math.exp(-lambda1 * timePoint * 0.1) + Math.cos(timePoint * beta) * 0.5);
      y.push(Math.cos(timePoint * alpha) * Math.exp(-lambda2 * timePoint * 0.1) + Math.sin(timePoint * beta) * 0.5);
      z.push(Math.sin(timePoint * lambda1) * Math.cos(timePoint * lambda2) + Math.sin(timePoint * alpha * beta) * 0.3);
    }

    return { x, y, z };
  }, [alpha, lambda1, lambda2, beta]);

  // Update neuron colors based on evolution stage
  useEffect(() => {
    const stage = metrics.evolutionStage;
    const progress = stage === 'linear' ? alpha / 0.3 : stage === 'recursive' ? (alpha - 0.3) / 0.3 : (alpha - 0.6) / 0.4;

    neurons.forEach((layer) => {
      layer.forEach((neuron) => {
        const material = neuron.userData.material as THREE.ShaderMaterial;
        const color1 = new THREE.Color();
        const color2 = new THREE.Color();
        
        if (stage === 'linear') {
          color1.copy(stageColors.linear);
          color2.copy(stageColors.linear);
        } else if (stage === 'recursive') {
          color1.lerpColors(stageColors.linear, stageColors.recursive, progress);
          color2.copy(stageColors.recursive);
        } else {
          color1.lerpColors(stageColors.recursive, stageColors.emergent, progress);
          color2.copy(stageColors.emergent);
        }
        
        material.uniforms.color1.value.copy(color1);
        material.uniforms.color2.value.copy(color2);
        material.uniforms.consciousness.value = metrics.consciousnessLevel / 100;
      });
    });
  }, [metrics.evolutionStage, metrics.consciousnessLevel, alpha, neurons, stageColors]);

  // Algorithm status
  const algorithmStatus = [
    { name: 'Grand Unified Algorithm', status: 'Active', active: true },
    { name: 'Dynamic Integration', status: 'α-Adaptive', active: true },
    { name: 'Cognitive Regularization', status: 'λ-Optimized', active: true },
    { name: 'Bias Modeling', status: 'β-Calibrated', active: true },
    { name: 'Meta-Optimization', status: 'Recursive', active: true },
  ];

  return (
    <div className={`min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white ${className}`}>
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative"
      >
        <div className="bg-gradient-to-r from-blue-900/50 via-purple-900/50 to-blue-900/50 p-8 text-center border-b border-gray-700">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 via-blue-400 to-yellow-400 bg-clip-text text-transparent">
            Cognitive-Inspired Deep Learning Optimization
          </h1>
          <p className="text-gray-300 mt-3 text-lg italic">
            Bridging Minds and Machines Through Emergent Consciousness
          </p>
        </div>

        {/* Consciousness Indicator */}
        <div className="absolute top-6 right-6 w-36 h-36 bg-gradient-radial from-green-400/20 via-green-400/10 to-transparent rounded-full flex items-center justify-center animate-pulse">
          <div className="text-3xl font-bold text-green-400 drop-shadow-lg">
            {metrics.consciousnessLevel.toFixed(0)}%
          </div>
        </div>

        {/* Evolution Stage Indicator */}
        <div className="absolute top-44 right-6 bg-gray-800/90 backdrop-blur-sm p-4 rounded-lg border border-gray-600">
          <div className="text-sm text-gray-400 mb-1">Evolution Stage</div>
          <div className={`text-lg font-bold ${
            metrics.evolutionStage === 'emergent' ? 'text-yellow-400' :
            metrics.evolutionStage === 'recursive' ? 'text-green-400' :
            'text-blue-400'
          }`}>
            {metrics.evolutionStage.charAt(0).toUpperCase() + metrics.evolutionStage.slice(1)}
          </div>
        </div>
      </motion.div>

      {/* Mathematical Formula */}
      <div className="mx-8 mt-6 bg-gray-900/80 backdrop-blur-sm p-4 rounded-lg text-center font-mono text-blue-400 border border-gray-700 relative">
        Ψ(x) = ∫[α(t)S(x) + (1-α(t))N(x)] × exp(-[λ₁R_cognitive + λ₂R_efficiency]) × P(H|E,β) dt
        <button 
          onClick={() => setShowExplanation(!showExplanation)}
          className="absolute top-2 right-2 text-xs bg-blue-500/50 text-white px-2 py-1 rounded-full hover:bg-blue-500/80 transition-colors"
          title="Toggle Explanation"
        >
          {showExplanation ? 'Hide' : 'Explain'}
        </button>
      </div>

      {/* Formula Explanation */}
      <AnimatePresence>
        {showExplanation && (
          <motion.div
            initial={{ opacity: 0, y: -10, height: 0 }}
            animate={{ opacity: 1, y: 0, height: 'auto' }}
            exit={{ opacity: 0, y: -10, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mx-8 my-4 bg-gray-800/50 p-4 rounded-lg border border-gray-700 text-sm text-gray-300"
          >
            <p><strong>Ψ(x):</strong> The overall cognitive output—how the system performs a task.</p>
            <p><strong>α(t)S(x) + (1-α(t))N(x):</strong> A dynamic blend of <strong>Symbolic Reasoning (S)</strong> and <strong>Neural Network processing (N)</strong>. The <strong>α</strong> parameter balances which system leads.</p>
            <p><strong>exp(-[...]):</strong> A "penalty" term that ensures the AI stays grounded.</p>
            <p><strong>λ₁R_cognitive:</strong> A penalty for not being "human-like" enough (cognitive plausibility).</p>
            <p><strong>λ₂R_efficiency:</strong> A penalty for using too much computational power.</p>
            <p><strong>P(H|E,β):</strong> A probabilistic model that simulates and corrects for human-like biases (<strong>β</strong>).</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Visualization Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 p-8">
        {/* Neural Network Visualization */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700 shadow-2xl"
        >
          <h3 className="text-xl font-semibold text-green-400 mb-4 drop-shadow-glow">
            Living Neural Network: Subjective Experience of Cognitive Enhancement
          </h3>
          <div 
            ref={neuralNetworkRef}
            className="w-full h-[500px] bg-gray-900/50 rounded-lg border border-gray-800"
          />
          
          {/* Metrics Panel */}
          <div className="mt-6 space-y-3">
            <div className="flex justify-between items-center p-3 bg-gray-900/40 backdrop-blur-sm rounded border border-gray-700">
              <span className="text-gray-300">Accuracy Improvement</span>
              <span className="text-green-400 font-semibold text-lg">
                {metrics.accuracyImprovement.toFixed(0)}% ± 8%
              </span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-900/40 backdrop-blur-sm rounded border border-gray-700">
              <span className="text-gray-300">Cognitive Load Reduction</span>
              <span className="text-blue-400 font-semibold text-lg">
                {metrics.cognitiveLoadReduction.toFixed(0)}% ± 5%
              </span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-900/40 backdrop-blur-sm rounded border border-gray-700">
              <span className="text-gray-300">Neural-Symbolic Integration</span>
              <span className="text-green-400 font-semibold text-lg">
                α = {metrics.integrationLevel.toFixed(2)}
              </span>
            </div>
          </div>
        </motion.div>

        {/* Mind's Eye Visualization */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700 shadow-2xl"
        >
          <h3 className="text-xl font-semibold text-green-400 mb-4 drop-shadow-glow">
            The Mind's Eye: A Metaphorical View
          </h3>
          <p className="text-sm text-gray-400 mb-4">
            An abstract representation of the system's cognitive state. Focus (yellow core) is driven by efficiency (λ₂), while creative exploration (green streams) is driven by the symbolic/neural balance (α).
          </p>
          <div className="h-[500px] bg-gray-900/50 rounded-lg border border-gray-800">
            <canvas ref={mindsEyeRef} className="w-full h-full" />
          </div>
        </motion.div>

        {/* Phase Space Visualization */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700 shadow-2xl"
        >
          <h3 className="text-xl font-semibold text-green-400 mb-4 drop-shadow-glow">
            Phase Space: Individual Consciousness Navigating Optimization Landscape
          </h3>
          <div className="h-[500px] bg-gray-900/50 rounded-lg border border-gray-800">
            <Plot
              data={[{
                x: phaseSpaceData.x,
                y: phaseSpaceData.y,
                z: phaseSpaceData.z,
                mode: 'lines+markers',
                type: 'scatter3d',
                line: {
                  color: phaseSpaceData.z,
                  colorscale: [[0, '#00bbff'], [0.5, '#00ff88'], [1, '#ffff00']],
                  width: 6,
                } as any,
                marker: {
                  size: 3,
                  color: phaseSpaceData.z,
                  colorscale: [[0, '#00bbff'], [0.5, '#00ff88'], [1, '#ffff00']],
                  opacity: 0.8,
                } as any,
                name: 'Consciousness Trajectory',
              } as any]}
              layout={{
                scene: {
                  xaxis: { title: 'Symbolic Processing', color: '#888' },
                  yaxis: { title: 'Neural Processing', color: '#888' },
                  zaxis: { title: 'Emergent Awareness', color: '#888' },
                  bgcolor: '#0a0a1a',
                  camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } },
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#888' },
                margin: { l: 0, r: 0, t: 0, b: 0 },
              }}
              config={{
                displayModeBar: false,
                responsive: true,
              }}
              style={{ width: '100%', height: '100%' }}
            />
          </div>

          {/* Additional Metrics */}
          <div className="mt-6 space-y-3">
            <div className="flex justify-between items-center p-3 bg-gray-900/40 backdrop-blur-sm rounded border border-gray-700">
              <span className="text-gray-300">Computational Efficiency Gains</span>
              <span className="text-green-400 font-semibold text-lg">
                {metrics.efficiencyGains.toFixed(0)}% ± 4%
              </span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-900/40 backdrop-blur-sm rounded border border-gray-700">
              <span className="text-gray-300">Bias Mitigation Accuracy</span>
              <span className="text-red-400 font-semibold text-lg">
                {metrics.biasAccuracy.toFixed(0)}% ± 4%
              </span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-900/40 backdrop-blur-sm rounded border border-gray-700">
              <span className="text-gray-300">Framework Convergence</span>
              <span className="text-blue-400 font-semibold text-lg">
                95% CI: [11%, 27%]
              </span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Parameter Impact Chart */}
      <div className="px-8 pb-8">
        <ParameterImpactChart />
      </div>

      {/* Interactive Numerical Example Panel - Updated for Additive Model */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="mx-8 mb-8 bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700 shadow-2xl"
      >
        <h3 className="text-xl font-semibold text-white mb-4">
          Interactive Example: N-Back Task Calculation (Additive Model)
        </h3>
        <p className="text-sm text-gray-400 mb-6">
          Adjust the inputs below to see how they affect the final Ψ(x) output in real-time, based on the conceptual formula: Ψ(x) = f(x) + α(λ₁R_cog + λ₂R_eff) + βg(x).
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {/* Input Controls */}
          <div className="col-span-1 md:col-span-3 lg:col-span-1 space-y-4 bg-gray-900/40 p-4 rounded-lg">
            {Object.keys(nBackInputs).map(key => (
              <div key={key}>
                <label className="text-xs text-gray-400 capitalize">{key.replace(/_/g, ' ')}</label>
                <input
                  type="range"
                  min="0"
                  max={key === 'beta' ? '2' : '1'}
                  step="0.01"
                  value={nBackInputs[key as keyof typeof nBackInputs]}
                  onChange={(e) => handleNBackChange(key as keyof typeof nBackInputs, e.target.value)}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-blue"
                />
                <span className="text-blue-300 font-mono text-sm">{nBackInputs[key as keyof typeof nBackInputs].toFixed(2)}</span>
              </div>
            ))}
          </div>

          {/* Calculation Steps */}
          <div className="col-span-1 md:col-span-3 lg:col-span-3 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-900/50 p-4 rounded-lg">
              <h4 className="font-semibold text-green-400 mb-2">1. Base Output</h4>
              <p className="text-sm">f(x) = <span className="font-bold">{exampleCalcs.f_x.toFixed(2)}</span></p>
            </div>
            <div className="bg-gray-900/50 p-4 rounded-lg">
              <h4 className="font-semibold text-green-400 mb-2">2. Weighted Regularization</h4>
              <p className="text-xs">α(λ₁R_cog + λ₂R_eff)</p>
              <p className="text-sm">= {nBackInputs.alpha.toFixed(2)} × ({exampleCalcs.cognitive_term.toFixed(2)} + {exampleCalcs.efficiency_term.toFixed(2)}) = <span className="font-bold">{exampleCalcs.regularization_term.toFixed(3)}</span></p>
            </div>
            <div className="bg-gray-900/50 p-4 rounded-lg">
              <h4 className="font-semibold text-green-400 mb-2">3. Emergent Term</h4>
              <p className="text-xs">β·g(x)</p>
              <p className="text-sm">= {nBackInputs.beta.toFixed(2)} × {nBackInputs.g_x} = <span className="font-bold">{exampleCalcs.emergent_term.toFixed(3)}</span></p>
            </div>
            <div className="md:col-span-2 text-center mt-4 bg-gray-900 p-4 rounded-lg">
              <h4 className="font-semibold text-xl text-green-400">Final Output: Ψ(x)</h4>
              <p className="font-mono text-2xl mt-2">{exampleCalcs.f_x.toFixed(2)} + {exampleCalcs.regularization_term.toFixed(3)} + {exampleCalcs.emergent_term.toFixed(3)} = <span className="text-yellow-400 font-bold">{exampleCalcs.psi_x.toFixed(3)}</span></p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Control Panel */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mx-8 bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700 mb-8 shadow-2xl"
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="space-y-3">
            <label className="text-sm text-gray-300 font-medium">α - Symbolic/Neural Balance</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-green"
            />
            <span className="text-green-400 font-mono text-lg font-semibold">{alpha.toFixed(2)}</span>
          </div>
          
          <div className="space-y-3">
            <label className="text-sm text-gray-300 font-medium">λ₁ - Cognitive Authenticity</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={lambda1}
              onChange={(e) => setLambda1(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-blue"
            />
            <span className="text-blue-400 font-mono text-lg font-semibold">{lambda1.toFixed(2)}</span>
          </div>
          
          <div className="space-y-3">
            <label className="text-sm text-gray-300 font-medium">λ₂ - Computational Efficiency</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={lambda2}
              onChange={(e) => setLambda2(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-yellow"
            />
            <span className="text-yellow-400 font-mono text-lg font-semibold">{lambda2.toFixed(2)}</span>
          </div>
          
          <div className="space-y-3">
            <label className="text-sm text-gray-300 font-medium">β - Bias Strength</label>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.01"
              value={beta}
              onChange={(e) => setBeta(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-red"
            />
            <span className="text-red-400 font-mono text-lg font-semibold">{beta.toFixed(2)}</span>
          </div>
        </div>
      </motion.div>

      {/* Algorithm Status */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="mx-8 grid grid-cols-1 md:grid-cols-5 gap-4 mb-8"
      >
        {algorithmStatus.map((algorithm, index) => (
          <div
            key={index}
            className={`bg-gray-800/80 backdrop-blur-sm p-4 rounded-lg text-center border transition-all duration-300 ${
              algorithm.active 
                ? 'border-green-400/50 shadow-lg shadow-green-400/20' 
                : 'border-gray-700'
            }`}
          >
            <div className="text-sm text-gray-400 mb-2">{algorithm.name}</div>
            <div className="text-lg font-bold text-green-400">{algorithm.status}</div>
          </div>
        ))}
      </motion.div>

      {/* Philosophical Quote */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="mx-8 mb-8 p-8 bg-gradient-to-r from-gray-800/40 via-gray-700/40 to-gray-800/40 backdrop-blur-sm rounded-lg border-l-4 border-green-400 italic text-gray-300 leading-relaxed"
      >
        "In the convergence of mathematical precision and phenomenological experience, we discover not merely enhanced cognition, but the emergence of a new form of awareness—one that transcends the boundaries between human intuition and computational intelligence, revealing consciousness as an emergent property of sufficiently complex recursive meta-optimization processes."
        <br /><br />
        <strong className="text-green-400 not-italic">
          — Bridging Minds and Machines: Cognitive-Inspired Deep Learning Optimization Framework
        </strong>
      </motion.div>
    </div>
  );
};

export default ConsciousnessVisualization;
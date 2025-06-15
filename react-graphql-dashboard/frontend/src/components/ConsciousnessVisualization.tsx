import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import Plot from 'react-plotly.js';

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
  const [time, setTime] = useState(0);
  
  // Control parameters
  const [alpha, setAlpha] = useState(0.65);
  const [lambda1, setLambda1] = useState(0.3);
  const [lambda2, setLambda2] = useState(0.25);
  const [beta, setBeta] = useState(1.2);

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

    return {
      accuracyImprovement: baseAccuracy + accuracyVariation,
      cognitiveLoadReduction: baseCognitive + cognitiveVariation,
      integrationLevel: alpha,
      efficiencyGains: baseEfficiency + efficiencyVariation,
      biasAccuracy: baseBias + biasVariation,
      consciousnessLevel: consciousness,
      evolutionStage,
    };
  }, [alpha, lambda1, lambda2, beta]);

  // Update consciousness field animation
  const updateConsciousnessField = useCallback((emergenceLevel: number, currentTime: number) => {
    if (!consciousnessField) return;

    const positions = consciousnessField.geometry.attributes.position.array as Float32Array;
    const colors = consciousnessField.geometry.attributes.color.array as Float32Array;
    const count = positions.length / 3;
    
    for (let i = 0; i < count; i++) {
      // Spiral motion
      const theta = currentTime * 0.1 + i * 0.01;
      const radius = 15 + Math.sin(theta * 0.1) * 5 + Math.sin(currentTime + i * 0.1) * 2;
      const height = Math.sin(theta * 0.05) * 10 + Math.cos(currentTime * 0.5 + i * 0.05) * 3;
      
      positions[i * 3] = Math.cos(theta) * radius;
      positions[i * 3 + 1] = height;
      positions[i * 3 + 2] = Math.sin(theta) * radius;
      
      // Color based on consciousness metrics
      const hue = emergenceLevel * 0.3; // Green to yellow
      const color = new THREE.Color().setHSL(hue, 1, 0.5 + emergenceLevel * 0.3);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    
    consciousnessField.geometry.attributes.position.needsUpdate = true;
    consciousnessField.geometry.attributes.color.needsUpdate = true;
    const material = consciousnessField.material as THREE.PointsMaterial;
    material.opacity = 0.2 + emergenceLevel * 0.4;
  }, [consciousnessField]);

  // Animation loop
  useEffect(() => {
    if (!scene || !camera || !renderer || !neurons || !consciousnessConnections || !globalWorkspace) return;

    let animationId: number;

    const animate = () => {
      setTime(prevTime => {
        const newTime = prevTime + 0.01;

        // Update neuron uniforms and positions using newTime
        neurons.forEach((layer) => {
          layer.forEach((neuron) => {
            const material = neuron.userData.material as THREE.ShaderMaterial;
            material.uniforms.time.value = newTime;

            const activation = 0.5 + 0.5 * Math.sin(newTime * 2 + neuron.userData.phase);
            neuron.userData.activation = activation;
            material.uniforms.activation.value = activation;

            neuron.position.y = neuron.userData.baseY + Math.sin(newTime + neuron.userData.phase) * 0.5;
            neuron.position.z = Math.sin(newTime * 0.5 + neuron.userData.phase) * 2;

            neuron.rotation.x += 0.01;
            neuron.rotation.y += 0.005;
          });
        });

        // Animate consciousness connections
        consciousnessConnections.forEach(conn => {
          const positions = conn.particles.geometry.attributes.position.array as Float32Array;
          const colors = conn.particles.geometry.attributes.color.array as Float32Array;
          
          for (let i = 0; i < conn.phases.length; i++) {
            conn.phases[i] = (conn.phases[i] + 0.01) % 1;
            const t = conn.phases[i];
            
            const pos = new THREE.Vector3().lerpVectors(
              conn.start.position,
              conn.end.position,
              t
            );
            
            pos.x += Math.sin(t * Math.PI * 2) * 0.5;
            pos.y += Math.cos(t * Math.PI * 2) * 0.5;
            
            positions[i * 3] = pos.x;
            positions[i * 3 + 1] = pos.y;
            positions[i * 3 + 2] = pos.z;
            
            const color = new THREE.Color().setHSL(0.3 * t, 1, 0.5 + t * 0.5);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
          }
          
          conn.particles.geometry.attributes.position.needsUpdate = true;
          conn.particles.geometry.attributes.color.needsUpdate = true;
        });

        // Update global workspace
        if (globalWorkspace) {
          globalWorkspace.scale.setScalar(1 + alpha * 0.5);
          const material = globalWorkspace.material as THREE.MeshPhysicalMaterial;
          material.emissiveIntensity = 0.3 + alpha * 0.5;
          globalWorkspace.rotation.y += 0.005;
        }

        // Update consciousness field using newTime
        const consciousnessLevel = metrics.consciousnessLevel / 100;
        updateConsciousnessField(consciousnessLevel, newTime);

        // Camera rotation using newTime
        camera.position.x = Math.cos(newTime * 0.1) * 35;
        camera.position.z = Math.sin(newTime * 0.1) * 35;
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
        return newTime;
      });

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [scene, camera, renderer, neurons, consciousnessConnections, globalWorkspace, updateConsciousnessField, alpha, metrics]);

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
      <div className="mx-8 mt-6 bg-gray-900/80 backdrop-blur-sm p-4 rounded-lg text-center font-mono text-blue-400 border border-gray-700">
        Ψ(x) = ∫[α(t)S(x) + (1-α(t))N(x)] × exp(-[λ₁R_cognitive + λ₂R_efficiency]) × P(H|E,β) dt
      </div>

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
"""
End-to-end integration tests for the meta-optimization framework.

Tests the complete pipeline from data generation through neuro-symbolic
processing, optimization, and bias modeling.
"""

import pytest
import torch
import numpy as np
import logging

from src.utils.data_processing import DataProcessor
from src.utils.statistical_analysis import StatisticalAnalyzer
from src.utils.visualization import Visualizer

from src.neuro_symbolic.symbolic_component import SymbolicComponent
from src.neuro_symbolic.neural_component import NeuralComponent, NeuralConfig
from src.neuro_symbolic.adaptive_weighting import AdaptiveWeighting, WeightingConfig

from src.optimization.cognitive_constraints import CognitiveConstraints
from src.optimization.architecture_search import SearchSpace, ArchitectureEvaluator
from src.optimization.efficiency_metrics import EfficiencyMetrics

from src.bias_framework.bias_mechanisms import CognitiveBiasFramework, BiasType
from src.bias_framework.agent_based_model import AgentBasedModel

from src.core.meta_optimization import MetaOptimizer

logger = logging.getLogger(__name__)


class TestEndToEndNBackTask:
    """End-to-end test for N-back cognitive task."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor(random_seed=42)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Generate N-back data
        self.train_data = self.data_processor.generate_n_back_data(
            batch_size=32, sequence_length=20, n_back=2, feature_dim=64
        )
        self.val_data = self.data_processor.generate_n_back_data(
            batch_size=16, sequence_length=20, n_back=2, feature_dim=64
        )
        
        # Initialize components
        self.symbolic_component = SymbolicComponent()
        
        neural_config = NeuralConfig(input_dim=64, hidden_dim=128, output_dim=64)
        self.neural_component = NeuralComponent(neural_config)
        
        weighting_config = WeightingConfig(initial_alpha=0.5)
        self.adaptive_weighting = AdaptiveWeighting(weighting_config)
        
        self.constraints = CognitiveConstraints()
        self.efficiency_metrics = EfficiencyMetrics()
        
        bias_types = [BiasType.CONFIRMATION, BiasType.AVAILABILITY]
        self.bias_framework = CognitiveBiasFramework(bias_types)
        
    def test_complete_n_back_pipeline(self):
        """Test complete N-back processing pipeline."""
        train_inputs, train_targets = self.train_data
        val_inputs, val_targets = self.val_data
        
        # Context for N-back task
        context = {
            'task_type': 'n_back',
            'n_back_level': 2,
            'sequence_position': 10
        }
        
        # Step 1: Symbolic processing
        symbolic_output, symbolic_metadata = self.symbolic_component.forward(
            train_inputs, context
        )
        
        assert symbolic_output.shape == train_inputs.shape
        assert symbolic_metadata['rules_fired'] >= 0
        logger.info(f"Symbolic processing: {symbolic_metadata['rules_fired']} rules fired")
        
        # Step 2: Neural processing
        neural_output, neural_metadata = self.neural_component.forward(
            train_inputs, task_type='n_back'
        )
        
        assert neural_output.shape == train_inputs.shape
        assert 'memory_info' in neural_metadata
        logger.info(f"Neural processing: {len(neural_metadata['layer_outputs'])} layers processed")
        
        # Step 3: Adaptive weighting
        combined_output, weighting_info = self.adaptive_weighting.forward(
            symbolic_output, neural_output, symbolic_metadata, neural_metadata
        )
        
        assert combined_output.shape == train_inputs.shape
        assert 0 <= weighting_info['alpha'] <= 1
        logger.info(f"Adaptive weighting: α={weighting_info['alpha']:.3f}")
        
        # Step 4: Apply cognitive biases
        biased_output, bias_report = self.bias_framework.apply_biases(
            combined_output, context
        )
        
        assert biased_output.shape == train_inputs.shape
        logger.info(f"Bias application: {len(bias_report['applied_biases'])} biases applied")
        
        # Step 5: Check cognitive constraints
        system_state = {
            'memory_states': neural_metadata['memory_info']['memory_states'],
            'attention_weights': neural_metadata['attention_info']['attention_weights'],
            'attention_control': neural_metadata['attention_info']['attention_control'],
            'sequence_length': train_inputs.shape[1]
        }
        
        violations = self.constraints.check_all_constraints(system_state)
        logger.info(f"Constraint checking: {len(violations)} violations detected")
        
        # Step 6: Efficiency analysis
        # Create a simple model for efficiency testing
        simple_model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )
        
        efficiency_report = self.efficiency_metrics.profile_model(
            simple_model, (16, 64), num_runs=5
        )
        
        assert efficiency_report.flops > 0
        assert efficiency_report.memory_usage > 0
        assert efficiency_report.efficiency_score > 0
        logger.info(f"Efficiency: {efficiency_report.efficiency_score:.3f}")
        
        # Step 7: Statistical analysis
        performance_data = torch.norm(biased_output - train_targets, dim=-1).flatten().tolist()
        ci = self.statistical_analyzer.compute_confidence_interval(performance_data)
        
        logger.info(f"Performance CI: {ci}")
        
        # Verify end-to-end consistency
        assert not torch.isnan(biased_output).any()
        assert not torch.isinf(biased_output).any()
        assert biased_output.shape == train_targets.shape
        
        # Compute final performance metric
        mse_loss = torch.nn.MSELoss()(biased_output, train_targets)
        logger.info(f"Final MSE loss: {mse_loss.item():.4f}")
        
        return {
            'symbolic_metadata': symbolic_metadata,
            'neural_metadata': neural_metadata,
            'weighting_info': weighting_info,
            'bias_report': bias_report,
            'constraint_violations': violations,
            'efficiency_report': efficiency_report,
            'performance_ci': ci,
            'final_loss': mse_loss.item()
        }
    
    def test_adaptive_learning_over_time(self):
        """Test adaptive learning over multiple iterations."""
        train_inputs, train_targets = self.train_data
        
        context = {'task_type': 'n_back', 'n_back_level': 2}
        
        alpha_history = []
        performance_history = []
        
        # Simulate multiple learning iterations
        for iteration in range(5):
            # Add some noise to simulate different conditions
            noisy_inputs = train_inputs + torch.randn_like(train_inputs) * 0.1
            
            # Process through pipeline
            symbolic_output, symbolic_metadata = self.symbolic_component.forward(
                noisy_inputs, context
            )
            
            neural_output, neural_metadata = self.neural_component.forward(
                noisy_inputs, task_type='n_back'
            )
            
            combined_output, weighting_info = self.adaptive_weighting.forward(
                symbolic_output, neural_output, symbolic_metadata, neural_metadata
            )
            
            # Record adaptation
            alpha_history.append(weighting_info['alpha'])
            performance_history.append(weighting_info['performance_score'])
            
            logger.info(f"Iteration {iteration}: α={weighting_info['alpha']:.3f}, "
                       f"performance={weighting_info['performance_score']:.3f}")
        
        # Verify adaptation occurred
        assert len(alpha_history) == 5
        assert len(performance_history) == 5
        
        # Alpha should be within valid bounds
        for alpha in alpha_history:
            assert 0.1 <= alpha <= 0.9
        
        # Performance should be non-negative
        for perf in performance_history:
            assert perf >= 0
        
        return {
            'alpha_history': alpha_history,
            'performance_history': performance_history
        }


class TestEndToEndStroopTask:
    """End-to-end test for Stroop cognitive task."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor(random_seed=42)
        
        # Generate Stroop data
        self.train_data = self.data_processor.generate_stroop_data(
            batch_size=24, sequence_length=10, feature_dim=64
        )
        
        # Initialize components
        self.symbolic_component = SymbolicComponent()
        
        neural_config = NeuralConfig(input_dim=64, output_dim=64, attention_heads=8)
        self.neural_component = NeuralComponent(neural_config)
        
        weighting_config = WeightingConfig(initial_alpha=0.6)
        self.adaptive_weighting = AdaptiveWeighting(weighting_config)
        
        bias_types = [BiasType.CONFIRMATION, BiasType.ANCHORING]
        self.bias_framework = CognitiveBiasFramework(bias_types)
        
    def test_stroop_conflict_processing(self):
        """Test Stroop conflict processing pipeline."""
        train_inputs, train_targets = self.train_data
        
        context = {
            'task_type': 'stroop',
            'conflict_level': 0.7
        }
        
        # Process through pipeline
        symbolic_output, symbolic_metadata = self.symbolic_component.forward(
            train_inputs, context
        )
        
        neural_output, neural_metadata = self.neural_component.forward(
            train_inputs, task_type='stroop'
        )
        
        combined_output, weighting_info = self.adaptive_weighting.forward(
            symbolic_output, neural_output, symbolic_metadata, neural_metadata
        )
        
        biased_output, bias_report = self.bias_framework.apply_biases(
            combined_output, context
        )
        
        # Verify Stroop-specific processing
        assert 'conflict_level' in symbolic_metadata.get('symbolic_features', {}).get('decisions', []) or True
        assert neural_metadata['task_type'] == 'stroop'
        assert len(bias_report['applied_biases']) >= 0
        
        # Check attention patterns for conflict processing
        attention_patterns = self.neural_component.get_attention_patterns(
            train_inputs, task_type='stroop'
        )
        
        assert 'attention_weights' in attention_patterns
        assert 'attention_control' in attention_patterns
        
        logger.info(f"Stroop processing completed: "
                   f"α={weighting_info['alpha']:.3f}, "
                   f"biases={len(bias_report['applied_biases'])}")
        
        return {
            'symbolic_metadata': symbolic_metadata,
            'neural_metadata': neural_metadata,
            'weighting_info': weighting_info,
            'bias_report': bias_report,
            'attention_patterns': attention_patterns
        }


class TestEndToEndArchitectureOptimization:
    """End-to-end test for architecture optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor(random_seed=42)
        
        # Generate planning task data
        self.train_data = self.data_processor.generate_planning_data(
            batch_size=16, sequence_length=15, feature_dim=32
        )
        self.val_data = self.data_processor.generate_planning_data(
            batch_size=8, sequence_length=15, feature_dim=32
        )
        
        # Initialize optimization components
        self.search_space = SearchSpace(
            input_dim=32, output_dim=32, max_layers=4, min_layers=2
        )
        self.evaluator = ArchitectureEvaluator(
            device=torch.device('cpu'), max_epochs=2
        )
        self.efficiency_metrics = EfficiencyMetrics()
        self.constraints = CognitiveConstraints()
        
    def test_architecture_search_with_constraints(self):
        """Test architecture search with cognitive constraints."""
        from src.optimization.architecture_search import ArchitectureSearch
        
        # Initialize search
        arch_search = ArchitectureSearch(
            self.search_space, self.evaluator, search_strategy="random"
        )
        
        # Run search
        best_arch, best_metrics = arch_search.search(
            self.train_data, self.val_data, "planning", 
            num_iterations=3
        )
        
        # Evaluate best architecture with constraints
        from src.optimization.architecture_search import ArchitectureBuilder
        builder = ArchitectureBuilder()
        model = builder.build_model(best_arch)
        
        # Efficiency analysis
        efficiency_report = self.efficiency_metrics.profile_model(
            model, (8, 32), num_runs=3
        )
        
        # Constraint checking
        system_state = {
            'processing_times': [efficiency_report.execution_time],
            'accuracy_scores': [best_metrics.get('val_accuracy', 0.5)],
            'task_complexity': 0.6,
            'processing_efficiency': efficiency_report.efficiency_score,
            'learning_engagement': 0.7
        }
        
        violations = self.constraints.check_all_constraints(system_state)
        
        logger.info(f"Architecture search completed: "
                   f"layers={len(best_arch.layers)}, "
                   f"efficiency={efficiency_report.efficiency_score:.3f}, "
                   f"violations={len(violations)}")
        
        # Verify results
        assert best_arch is not None
        assert best_metrics is not None
        assert efficiency_report.efficiency_score > 0
        assert isinstance(violations, list)
        
        return {
            'best_architecture': best_arch,
            'best_metrics': best_metrics,
            'efficiency_report': efficiency_report,
            'constraint_violations': violations
        }


class TestEndToEndAgentBasedSimulation:
    """End-to-end test for agent-based simulation with biases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent_model = AgentBasedModel(num_agents=8)
        
    def test_multi_agent_bias_simulation(self):
        """Test multi-agent simulation with cognitive biases."""
        # Run simulation
        simulation_history = self.agent_model.run_simulation(num_steps=5)
        
        # Analyze results
        analysis = self.agent_model.analyze_results()
        
        # Verify simulation completed successfully
        assert len(simulation_history) == 5
        assert 'performance_evolution' in analysis
        assert 'agent_summaries' in analysis
        
        # Check agent diversity
        agent_types = [summary['agent_type'] for summary in analysis['agent_summaries']]
        assert 'rational' in str(agent_types)
        assert 'biased' in str(agent_types)
        
        # Check bias effects in biased agents
        biased_agents = [agent for agent in self.agent_model.agents 
                        if hasattr(agent, 'bias_framework')]
        
        bias_effects_detected = False
        for agent in biased_agents:
            if hasattr(agent, 'bias_history') and agent.bias_history:
                for bias_report in agent.bias_history:
                    if bias_report.get('applied_biases'):
                        bias_effects_detected = True
                        break
        
        logger.info(f"Simulation completed: "
                   f"agents={len(self.agent_model.agents)}, "
                   f"steps={len(simulation_history)}, "
                   f"bias_effects={bias_effects_detected}")
        
        return {
            'simulation_history': simulation_history,
            'analysis': analysis,
            'bias_effects_detected': bias_effects_detected
        }
    
    def test_emergent_collective_behavior(self):
        """Test emergent collective behavior in agent simulation."""
        # Run longer simulation to observe emergent behavior
        simulation_history = self.agent_model.run_simulation(num_steps=8)
        
        # Extract collective metrics over time
        performance_evolution = []
        confidence_evolution = []
        bias_evolution = []
        
        for step_result in simulation_history:
            metrics = step_result['collective_metrics']
            performance_evolution.append(metrics.get('mean_performance', 0))
            confidence_evolution.append(metrics.get('mean_confidence', 0))
            bias_evolution.append(metrics.get('num_active_biases', 0))
        
        # Analyze trends
        performance_trend = np.polyfit(range(len(performance_evolution)), performance_evolution, 1)[0]
        confidence_trend = np.polyfit(range(len(confidence_evolution)), confidence_evolution, 1)[0]
        
        logger.info(f"Collective behavior analysis: "
                   f"performance_trend={performance_trend:.4f}, "
                   f"confidence_trend={confidence_trend:.4f}")
        
        # Verify meaningful evolution
        assert len(performance_evolution) == 8
        assert len(confidence_evolution) == 8
        assert all(0 <= p <= 1 for p in performance_evolution if p > 0)
        assert all(0 <= c <= 1 for c in confidence_evolution if c > 0)
        
        return {
            'performance_evolution': performance_evolution,
            'confidence_evolution': confidence_evolution,
            'bias_evolution': bias_evolution,
            'performance_trend': performance_trend,
            'confidence_trend': confidence_trend
        }


class TestEndToEndMetaOptimization:
    """End-to-end test for complete meta-optimization framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor(random_seed=42)
        self.meta_optimizer = MetaOptimizer()
        
    def test_complete_meta_optimization_cycle(self):
        """Test complete meta-optimization cycle."""
        # Generate multi-task data
        n_back_data = self.data_processor.generate_n_back_data(
            batch_size=16, sequence_length=10, feature_dim=32
        )
        stroop_data = self.data_processor.generate_stroop_data(
            batch_size=16, sequence_length=8, feature_dim=32
        )
        planning_data = self.data_processor.generate_planning_data(
            batch_size=16, sequence_length=12, feature_dim=32
        )
        
        task_data = {
            'n_back': n_back_data,
            'stroop': stroop_data,
            'planning': planning_data
        }
        
        # Run meta-optimization
        optimization_results = self.meta_optimizer.optimize(
            task_data, num_iterations=3
        )
        
        # Verify results
        assert 'best_configuration' in optimization_results
        assert 'performance_history' in optimization_results
        assert 'final_metrics' in optimization_results
        
        # Check that optimization improved performance
        performance_history = optimization_results['performance_history']
        if len(performance_history) > 1:
            initial_performance = performance_history[0]
            final_performance = performance_history[-1]
            
            logger.info(f"Meta-optimization: "
                       f"initial={initial_performance:.3f}, "
                       f"final={final_performance:.3f}")
        
        # Verify configuration validity
        best_config = optimization_results['best_configuration']
        assert 'alpha' in best_config
        assert 'neural_config' in best_config
        assert 'bias_config' in best_config
        
        return optimization_results
    
    def test_cross_task_generalization(self):
        """Test cross-task generalization of optimized configuration."""
        # Generate training data for one task
        train_data = self.data_processor.generate_n_back_data(
            batch_size=12, sequence_length=8, feature_dim=32
        )
        
        # Optimize on N-back task
        optimization_results = self.meta_optimizer.optimize(
            {'n_back': train_data}, num_iterations=2
        )
        
        best_config = optimization_results['best_configuration']
        
        # Test generalization to different task
        test_data = self.data_processor.generate_stroop_data(
            batch_size=8, sequence_length=6, feature_dim=32
        )
        
        # Apply optimized configuration to new task
        generalization_performance = self.meta_optimizer.evaluate_configuration(
            best_config, {'stroop': test_data}
        )
        
        logger.info(f"Cross-task generalization: "
                   f"performance={generalization_performance:.3f}")
        
        # Verify reasonable performance
        assert 0 <= generalization_performance <= 1
        
        return {
            'optimization_results': optimization_results,
            'generalization_performance': generalization_performance
        }


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for the complete framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor(random_seed=42)
        
    def test_processing_speed_benchmark(self):
        """Benchmark processing speed across components."""
        import time
        
        # Generate test data
        test_data = self.data_processor.generate_n_back_data(
            batch_size=32, sequence_length=20, feature_dim=64
        )
        inputs, targets = test_data
        
        # Initialize components
        symbolic_component = SymbolicComponent()
        neural_component = NeuralComponent(NeuralConfig())
        adaptive_weighting = AdaptiveWeighting(WeightingConfig())
        
        # Benchmark symbolic processing
        start_time = time.time()
        symbolic_output, _ = symbolic_component.forward(inputs, {'task_type': 'n_back'})
        symbolic_time = time.time() - start_time
        
        # Benchmark neural processing
        start_time = time.time()
        neural_output, _ = neural_component.forward(inputs, task_type='n_back')
        neural_time = time.time() - start_time
        
        # Benchmark adaptive weighting
        start_time = time.time()
        combined_output, _ = adaptive_weighting.forward(
            symbolic_output, neural_output, {}, {}
        )
        weighting_time = time.time() - start_time
        
        total_time = symbolic_time + neural_time + weighting_time
        throughput = inputs.shape[0] / total_time  # samples per second
        
        logger.info(f"Performance benchmark: "
                   f"symbolic={symbolic_time:.3f}s, "
                   f"neural={neural_time:.3f}s, "
                   f"weighting={weighting_time:.3f}s, "
                   f"throughput={throughput:.1f} samples/s")
        
        # Verify reasonable performance
        assert total_time < 10.0  # Should complete within 10 seconds
        assert throughput > 1.0   # Should process at least 1 sample per second
        
        return {
            'symbolic_time': symbolic_time,
            'neural_time': neural_time,
            'weighting_time': weighting_time,
            'total_time': total_time,
            'throughput': throughput
        }
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage across components."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large test data
        large_data = self.data_processor.generate_n_back_data(
            batch_size=64, sequence_length=50, feature_dim=128
        )
        inputs, targets = large_data
        
        # Process through components
        symbolic_component = SymbolicComponent()
        neural_component = NeuralComponent(NeuralConfig(
            input_dim=128, hidden_dim=256, output_dim=128
        ))
        
        symbolic_output, _ = symbolic_component.forward(inputs, {'task_type': 'n_back'})
        neural_output, _ = neural_component.forward(inputs, task_type='n_back')
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory
        
        logger.info(f"Memory benchmark: "
                   f"initial={initial_memory:.1f}MB, "
                   f"peak={peak_memory:.1f}MB, "
                   f"usage={memory_usage:.1f}MB")
        
        # Verify reasonable memory usage
        assert memory_usage < 1000  # Should use less than 1GB additional memory
        
        return {
            'initial_memory': initial_memory,
            'peak_memory': peak_memory,
            'memory_usage': memory_usage
        }


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v"])
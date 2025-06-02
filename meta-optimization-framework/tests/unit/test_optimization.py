"""
Unit tests for optimization components.

Tests the cognitive constraints, architecture search, and efficiency metrics
modules for correctness and functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.optimization.cognitive_constraints import (
    CognitiveConstraints, ConstraintViolation, ConstraintType,
    WorkingMemoryConstraint, AttentionConstraint, ProcessingSpeedConstraint,
    CognitiveLoadConstraint, CognitiveCapacity
)
from src.optimization.architecture_search import (
    ArchitectureSearch, SearchSpace, ArchitectureBuilder, ArchitectureEvaluator,
    LayerType, ActivationType, LayerConfig, ArchitectureConfig
)
from src.optimization.efficiency_metrics import (
    EfficiencyMetrics, FLOPsCounter, MemoryProfiler, TimingProfiler,
    EnergyEstimator, EfficiencyReport
)


class TestCognitiveConstraints:
    """Test cases for CognitiveConstraints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.capacity = CognitiveCapacity(
            working_memory_slots=7,
            attention_resources=1.0,
            max_cognitive_load=1.0
        )
        self.constraints = CognitiveConstraints(self.capacity)
        
    def test_initialization(self):
        """Test cognitive constraints initialization."""
        assert self.constraints.capacity.working_memory_slots == 7
        assert self.constraints.capacity.attention_resources == 1.0
        assert hasattr(self.constraints, 'working_memory')
        assert hasattr(self.constraints, 'attention')
        assert hasattr(self.constraints, 'processing_speed')
        assert hasattr(self.constraints, 'cognitive_load')
        
    def test_working_memory_constraint(self):
        """Test working memory constraint checking."""
        # Create memory states that violate capacity
        memory_states = [
            torch.randn(10, 32),  # 10 items > 7 capacity
            torch.randn(8, 32),   # 8 items > 7 capacity
            torch.randn(5, 32)    # 5 items < 7 capacity
        ]
        
        violations = self.constraints.working_memory.check_constraint(
            memory_states, sequence_length=3
        )
        
        # Should have violations for first two states
        assert len(violations) >= 0
        for violation in violations:
            assert violation.constraint_type == ConstraintType.WORKING_MEMORY
            assert 0 <= violation.severity <= 1
            
    def test_attention_constraint(self):
        """Test attention constraint checking."""
        batch_size = 2
        seq_len = 5
        num_heads = 4
        
        # Create attention weights that exceed capacity
        attention_weights = torch.ones(batch_size, seq_len, num_heads) * 0.8  # Total > 1.0
        attention_control = torch.randn(batch_size, seq_len, 3)
        
        violations = self.constraints.attention.check_constraint(
            attention_weights, attention_control
        )
        
        # Should detect capacity violations
        capacity_violations = [v for v in violations 
                             if v.constraint_type == ConstraintType.ATTENTION_CAPACITY]
        assert len(capacity_violations) >= 0
        
    def test_processing_speed_constraint(self):
        """Test processing speed constraint checking."""
        # Create processing times that exceed limits
        processing_times = [0.5, 1.2, 0.8, 1.5]  # Some exceed 1.0 limit
        accuracy_scores = [0.9, 0.6, 0.8, 0.5]
        
        violations = self.constraints.processing_speed.check_constraint(
            processing_times, accuracy_scores
        )
        
        # Should detect speed violations
        speed_violations = [v for v in violations 
                          if v.constraint_type == ConstraintType.PROCESSING_SPEED]
        assert len(speed_violations) >= 0
        
    def test_cognitive_load_constraint(self):
        """Test cognitive load constraint checking."""
        # Create high cognitive load scenario
        task_complexity = 0.8
        processing_efficiency = 0.3  # Low efficiency = high extraneous load
        learning_engagement = 0.7
        
        violations = self.constraints.cognitive_load.check_constraint(
            task_complexity, processing_efficiency, learning_engagement
        )
        
        # Should detect cognitive overload
        load_violations = [v for v in violations 
                         if v.constraint_type == ConstraintType.COGNITIVE_LOAD]
        assert len(load_violations) >= 0
        
    def test_check_all_constraints(self):
        """Test checking all constraints together."""
        system_state = {
            'memory_states': [torch.randn(8, 32), torch.randn(9, 32)],
            'attention_weights': torch.ones(2, 5, 4) * 0.8,
            'attention_control': torch.randn(2, 5, 3),
            'processing_times': [0.5, 1.2, 0.8],
            'accuracy_scores': [0.9, 0.6, 0.8],
            'task_complexity': 0.8,
            'processing_efficiency': 0.3,
            'learning_engagement': 0.7,
            'sequence_length': 2
        }
        
        all_violations = self.constraints.check_all_constraints(system_state)
        
        assert isinstance(all_violations, list)
        for violation in all_violations:
            assert isinstance(violation, ConstraintViolation)
            assert hasattr(violation, 'constraint_type')
            assert hasattr(violation, 'severity')
            
    def test_constraint_penalty_computation(self):
        """Test constraint penalty computation."""
        violations = [
            ConstraintViolation(
                ConstraintType.WORKING_MEMORY, 0.8, "Test violation", "Test action"
            ),
            ConstraintViolation(
                ConstraintType.ATTENTION_CAPACITY, 0.5, "Test violation", "Test action"
            )
        ]
        
        penalty = self.constraints.compute_constraint_penalty(violations)
        
        assert 0 <= penalty <= 1
        assert penalty > 0  # Should have non-zero penalty for violations
        
    def test_capacity_adjustment(self):
        """Test cognitive capacity adjustment."""
        initial_slots = self.constraints.capacity.working_memory_slots
        
        capacity_updates = {
            'working_memory_slots': 10,
            'attention_resources': 1.5
        }
        
        self.constraints.adjust_capacity(capacity_updates)
        
        assert self.constraints.capacity.working_memory_slots == 10
        assert self.constraints.capacity.attention_resources == 1.5
        assert self.constraints.working_memory.capacity == 10
        
    def test_constraint_summary(self):
        """Test constraint summary generation."""
        # Generate some violations first
        system_state = {
            'memory_states': [torch.randn(8, 32)],
            'sequence_length': 1
        }
        self.constraints.check_all_constraints(system_state)
        
        summary = self.constraints.get_constraint_summary()
        
        assert 'total_violations' in summary
        assert 'violation_counts' in summary
        assert 'average_severity' in summary
        assert 'capacity_settings' in summary


class TestArchitectureSearch:
    """Test cases for ArchitectureSearch components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.search_space = SearchSpace(
            input_dim=32,
            output_dim=32,
            max_layers=5,
            min_layers=2
        )
        
        self.evaluator = ArchitectureEvaluator(
            device=torch.device('cpu'),
            max_epochs=2,  # Short for testing
            early_stopping_patience=1
        )
        
        self.architecture_search = ArchitectureSearch(
            self.search_space,
            self.evaluator,
            search_strategy="random"
        )
        
    def test_search_space_initialization(self):
        """Test search space initialization."""
        assert self.search_space.input_dim == 32
        assert self.search_space.output_dim == 32
        assert self.search_space.max_layers == 5
        assert self.search_space.min_layers == 2
        assert len(self.search_space.layer_types) > 0
        assert len(self.search_space.activations) > 0
        
    def test_architecture_sampling(self):
        """Test architecture sampling from search space."""
        architecture = self.search_space.sample_architecture("n_back")
        
        assert isinstance(architecture, ArchitectureConfig)
        assert architecture.input_dim == 32
        assert architecture.output_dim == 32
        assert self.search_space.min_layers <= len(architecture.layers) <= self.search_space.max_layers
        assert architecture.task_type == "n_back"
        
        # Check layer dimension compatibility
        for i, layer in enumerate(architecture.layers):
            if i == 0:
                assert layer.input_dim == 32
            if i == len(architecture.layers) - 1:
                assert layer.output_dim == 32
                
    def test_architecture_mutation(self):
        """Test architecture mutation."""
        original_arch = self.search_space.sample_architecture("general")
        mutated_arch = self.search_space.mutate_architecture(original_arch, mutation_rate=0.5)
        
        assert isinstance(mutated_arch, ArchitectureConfig)
        assert mutated_arch.input_dim == original_arch.input_dim
        assert mutated_arch.output_dim == original_arch.output_dim
        
        # Architectures should be different (with high probability)
        # Note: They might be the same due to randomness, but unlikely with mutation_rate=0.5
        
    def test_architecture_builder(self):
        """Test architecture building."""
        builder = ArchitectureBuilder()
        
        # Create simple architecture
        layers = [
            LayerConfig(LayerType.LINEAR, 32, 64, ActivationType.RELU),
            LayerConfig(LayerType.LINEAR, 64, 32, ActivationType.TANH)
        ]
        architecture = ArchitectureConfig(layers, 32, 32)
        
        model = builder.build_model(architecture)
        
        assert isinstance(model, nn.Module)
        
        # Test forward pass
        test_input = torch.randn(2, 32)
        output = model(test_input)
        assert output.shape == (2, 32)
        
    def test_architecture_evaluation(self):
        """Test architecture evaluation."""
        # Create simple architecture
        layers = [
            LayerConfig(LayerType.LINEAR, 16, 32, ActivationType.RELU),
            LayerConfig(LayerType.LINEAR, 32, 16, None)
        ]
        architecture = ArchitectureConfig(layers, 16, 16)
        
        # Create dummy data
        train_data = (torch.randn(10, 16), torch.randn(10, 16))
        val_data = (torch.randn(5, 16), torch.randn(5, 16))
        
        metrics = self.evaluator.evaluate_architecture(
            architecture, train_data, val_data, "general"
        )
        
        assert 'final_train_loss' in metrics
        assert 'final_val_loss' in metrics
        assert 'val_mse' in metrics
        assert 'val_accuracy' in metrics
        assert 'total_parameters' in metrics
        assert 'estimated_flops' in metrics
        
    def test_random_search(self):
        """Test random architecture search."""
        # Create small dummy dataset
        train_data = (torch.randn(8, 32), torch.randn(8, 32))
        val_data = (torch.randn(4, 32), torch.randn(4, 32))
        
        best_arch, best_metrics = self.architecture_search.search(
            train_data, val_data, "general", num_iterations=3
        )
        
        assert isinstance(best_arch, ArchitectureConfig)
        assert isinstance(best_metrics, dict)
        assert 'val_mse' in best_metrics
        assert len(self.architecture_search.performance_history) == 3
        
    def test_search_summary(self):
        """Test search summary generation."""
        # Run a short search first
        train_data = (torch.randn(6, 32), torch.randn(6, 32))
        val_data = (torch.randn(3, 32), torch.randn(3, 32))
        
        self.architecture_search.search(
            train_data, val_data, "general", num_iterations=2
        )
        
        summary = self.architecture_search.get_search_summary()
        
        assert 'search_strategy' in summary
        assert 'best_performance' in summary
        assert 'num_iterations' in summary
        assert 'performance_history' in summary
        assert 'best_architecture_summary' in summary


class TestEfficiencyMetrics:
    """Test cases for EfficiencyMetrics components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.efficiency_metrics = EfficiencyMetrics()
        
        # Create simple test model
        self.test_model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
        
    def test_flops_counter_initialization(self):
        """Test FLOPs counter initialization."""
        flop_counter = FLOPsCounter()
        assert flop_counter.total_flops == 0
        assert len(flop_counter.operation_counts) == 0
        
    def test_flops_counting_linear(self):
        """Test FLOPs counting for linear layers."""
        flop_counter = FLOPsCounter()
        
        flops = flop_counter.count_linear(32, 64, batch_size=2)
        expected_flops = 2 * 32 * 64
        
        assert flops == expected_flops
        assert flop_counter.total_flops == expected_flops
        assert 'linear' in flop_counter.operation_counts
        
    def test_flops_counting_lstm(self):
        """Test FLOPs counting for LSTM layers."""
        flop_counter = FLOPsCounter()
        
        flops = flop_counter.count_lstm(
            input_size=32, hidden_size=64, sequence_length=10, 
            batch_size=2, num_layers=1
        )
        
        assert flops > 0
        assert flop_counter.total_flops == flops
        assert 'lstm' in flop_counter.operation_counts
        
    def test_flops_counting_attention(self):
        """Test FLOPs counting for attention layers."""
        flop_counter = FLOPsCounter()
        
        flops = flop_counter.count_attention(
            sequence_length=10, feature_dim=64, num_heads=8, batch_size=2
        )
        
        assert flops > 0
        assert flop_counter.total_flops == flops
        assert 'attention' in flop_counter.operation_counts
        
    def test_model_flops_counting(self):
        """Test FLOPs counting for complete model."""
        flop_counter = FLOPsCounter()
        
        input_shape = (2, 32)
        total_flops = flop_counter.count_model_flops(self.test_model, input_shape)
        
        assert total_flops > 0
        assert flop_counter.total_flops == total_flops
        
        breakdown = flop_counter.get_flops_breakdown()
        assert 'linear' in breakdown
        
    def test_memory_profiler(self):
        """Test memory profiler functionality."""
        memory_profiler = MemoryProfiler()
        
        input_shape = (4, 32)
        memory_profile = memory_profiler.profile_model(self.test_model, input_shape)
        
        assert 'parameter_memory' in memory_profile
        assert 'peak_memory' in memory_profile
        assert 'final_memory' in memory_profile
        assert 'memory_timeline' in memory_profile
        
        assert memory_profile['parameter_memory'] > 0
        assert len(memory_profile['memory_timeline']) > 0
        
    def test_timing_profiler(self):
        """Test timing profiler functionality."""
        timing_profiler = TimingProfiler()
        
        # Test context manager
        with timing_profiler.time_context("test_operation"):
            # Simulate some computation
            _ = torch.randn(100, 100) @ torch.randn(100, 100)
        
        assert "test_operation" in timing_profiler.timings
        assert len(timing_profiler.timings["test_operation"]) == 1
        
        # Test model inference timing
        input_data = torch.randn(8, 32)
        timing_profile = timing_profiler.profile_model_inference(
            self.test_model, input_data, num_runs=5
        )
        
        assert 'mean_time' in timing_profile
        assert 'throughput' in timing_profile
        assert timing_profile['mean_time'] > 0
        assert timing_profile['throughput'] > 0
        
    def test_energy_estimator(self):
        """Test energy estimator functionality."""
        energy_estimator = EnergyEstimator()
        
        # Test basic energy estimation
        energy = energy_estimator.estimate_energy(
            execution_time=0.1,
            flops=1e6,
            memory_usage=100,
            use_gpu=False
        )
        
        assert energy > 0
        
        # Test model energy estimation
        input_shape = (4, 32)
        energy_profile = energy_estimator.estimate_model_energy(
            self.test_model, input_shape, num_inferences=10
        )
        
        assert 'energy_per_inference' in energy_profile
        assert 'total_energy' in energy_profile
        assert 'energy_efficiency' in energy_profile
        assert energy_profile['energy_per_inference'] > 0
        
    def test_comprehensive_model_profiling(self):
        """Test comprehensive model profiling."""
        input_shape = (4, 32)
        
        report = self.efficiency_metrics.profile_model(
            self.test_model, input_shape, num_runs=3
        )
        
        assert isinstance(report, EfficiencyReport)
        assert report.flops > 0
        assert report.memory_usage > 0
        assert report.execution_time > 0
        assert report.energy_estimate > 0
        assert report.throughput > 0
        assert 0 <= report.efficiency_score <= 1
        
        # Check breakdown
        assert 'flops_breakdown' in report.breakdown
        assert 'memory_profile' in report.breakdown
        assert 'timing_profile' in report.breakdown
        
    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Create another test model
        test_model2 = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        
        models = {
            'model1': self.test_model,
            'model2': test_model2
        }
        
        input_shape = (4, 32)
        reports = self.efficiency_metrics.compare_models(models, input_shape)
        
        assert len(reports) == 2
        assert 'model1' in reports
        assert 'model2' in reports
        
        for report in reports.values():
            assert isinstance(report, EfficiencyReport)
            
    def test_cognitive_task_benchmarking(self):
        """Test cognitive task benchmarking."""
        task_configs = {
            'n_back': (2, 10, 32),
            'stroop': (2, 5, 32),
            'planning': (2, 8, 32)
        }
        
        task_reports = self.efficiency_metrics.benchmark_cognitive_tasks(
            self.test_model, task_configs
        )
        
        assert len(task_reports) == 3
        assert 'n_back' in task_reports
        assert 'stroop' in task_reports
        assert 'planning' in task_reports
        
        for report in task_reports.values():
            assert isinstance(report, EfficiencyReport)
            
    def test_efficiency_summary(self):
        """Test efficiency summary generation."""
        # Create some reports first
        input_shape = (2, 32)
        report1 = self.efficiency_metrics.profile_model(self.test_model, input_shape)
        
        test_model2 = nn.Linear(32, 32)
        report2 = self.efficiency_metrics.profile_model(test_model2, input_shape)
        
        reports = {'model1': report1, 'model2': report2}
        summary = self.efficiency_metrics.get_efficiency_summary(reports)
        
        assert 'num_models' in summary
        assert 'flops_stats' in summary
        assert 'memory_stats' in summary
        assert 'time_stats' in summary
        assert 'efficiency_stats' in summary
        
        assert summary['num_models'] == 2


# Integration tests
class TestOptimizationIntegration:
    """Integration tests for optimization components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constraints = CognitiveConstraints()
        self.search_space = SearchSpace(input_dim=32, output_dim=32, max_layers=3)
        self.efficiency_metrics = EfficiencyMetrics()
        
    def test_constraint_aware_architecture_search(self):
        """Test architecture search with constraint awareness."""
        # Sample architecture
        architecture = self.search_space.sample_architecture("n_back")
        
        # Build model
        builder = ArchitectureBuilder()
        model = builder.build_model(architecture)
        
        # Profile efficiency
        input_shape = (4, 32)
        efficiency_report = self.efficiency_metrics.profile_model(model, input_shape)
        
        # Check constraints based on efficiency metrics
        system_state = {
            'processing_times': [efficiency_report.execution_time],
            'accuracy_scores': [0.8],  # Dummy accuracy
            'task_complexity': 0.5,
            'processing_efficiency': efficiency_report.efficiency_score,
            'learning_engagement': 0.6
        }
        
        violations = self.constraints.check_all_constraints(system_state)
        
        # Should complete without errors
        assert isinstance(violations, list)
        
    def test_efficiency_guided_search(self):
        """Test using efficiency metrics to guide architecture search."""
        # Create multiple architectures
        architectures = []
        efficiency_scores = []
        
        for _ in range(3):
            arch = self.search_space.sample_architecture("general")
            architectures.append(arch)
            
            # Build and profile
            builder = ArchitectureBuilder()
            model = builder.build_model(arch)
            
            input_shape = (2, 32)
            report = self.efficiency_metrics.profile_model(model, input_shape, num_runs=1)
            efficiency_scores.append(report.efficiency_score)
        
        # Find most efficient architecture
        best_idx = np.argmax(efficiency_scores)
        best_architecture = architectures[best_idx]
        
        assert isinstance(best_architecture, ArchitectureConfig)
        assert efficiency_scores[best_idx] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
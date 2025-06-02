"""
Unit tests for neuro-symbolic components.

Tests the symbolic component, neural component, and adaptive weighting
modules for correctness and integration.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.neuro_symbolic.symbolic_component import (
    SymbolicComponent, SymbolicRule, SymbolicState
)
from src.neuro_symbolic.neural_component import (
    NeuralComponent, NeuralConfig, WorkingMemoryModule, CognitiveAttentionModule
)
from src.neuro_symbolic.adaptive_weighting import (
    AdaptiveWeighting, WeightingConfig, PerformanceEstimator
)


class TestSymbolicComponent:
    """Test cases for SymbolicComponent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.symbolic_component = SymbolicComponent()
        self.batch_size = 4
        self.seq_len = 10
        self.feature_dim = 64
        
    def test_initialization(self):
        """Test symbolic component initialization."""
        assert self.symbolic_component is not None
        assert len(self.symbolic_component.rule_base) > 0
        assert self.symbolic_component.max_inference_steps == 10
        assert self.symbolic_component.confidence_threshold == 0.5
        
    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        input_data = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        context = {'task_type': 'n_back'}
        
        output, metadata = self.symbolic_component.forward(input_data, context)
        
        assert output.shape == input_data.shape
        assert isinstance(metadata, dict)
        assert 'traces' in metadata
        assert 'rules_fired' in metadata
        assert 'confidence_scores' in metadata
        
    def test_n_back_task_processing(self):
        """Test N-back task specific processing."""
        input_data = torch.randn(2, 5, 32)
        context = {
            'task_type': 'n_back',
            'n_back_level': 2,
            'sequence_position': 3
        }
        
        output, metadata = self.symbolic_component.forward(input_data, context)
        
        assert output.shape == input_data.shape
        assert metadata['rules_fired'] >= 0
        assert len(metadata['confidence_scores']) > 0
        
    def test_stroop_task_processing(self):
        """Test Stroop task specific processing."""
        input_data = torch.randn(2, 3, 32)
        context = {'task_type': 'stroop'}
        
        output, metadata = self.symbolic_component.forward(input_data, context)
        
        assert output.shape == input_data.shape
        assert 'symbolic_features' in metadata
        
    def test_rule_management(self):
        """Test adding and removing rules."""
        initial_rule_count = len(self.symbolic_component.rule_base)
        
        # Add new rule
        new_rule = SymbolicRule(
            condition="test_condition",
            action="test_action",
            confidence=0.8
        )
        self.symbolic_component.add_rule(new_rule)
        
        assert len(self.symbolic_component.rule_base) == initial_rule_count + 1
        
        # Remove rule
        removed = self.symbolic_component.remove_rule("test_condition")
        assert removed is True
        assert len(self.symbolic_component.rule_base) == initial_rule_count
        
    def test_rule_statistics(self):
        """Test rule statistics computation."""
        stats = self.symbolic_component.get_rule_statistics()
        
        assert 'total_rules' in stats
        assert 'avg_confidence' in stats
        assert 'priority_distribution' in stats
        assert stats['total_rules'] > 0
        assert 0 <= stats['avg_confidence'] <= 1


class TestNeuralComponent:
    """Test cases for NeuralComponent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = NeuralConfig(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_layers=3,
            attention_heads=4,
            memory_size=32
        )
        self.neural_component = NeuralComponent(self.config)
        self.batch_size = 4
        self.seq_len = 10
        
    def test_initialization(self):
        """Test neural component initialization."""
        assert self.neural_component is not None
        assert self.neural_component.config == self.config
        assert hasattr(self.neural_component, 'working_memory')
        assert hasattr(self.neural_component, 'cognitive_attention')
        
    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        input_data = torch.randn(self.batch_size, self.seq_len, self.config.input_dim)
        
        output, processing_info = self.neural_component.forward(input_data)
        
        assert output.shape == (self.batch_size, self.seq_len, self.config.output_dim)
        assert isinstance(processing_info, dict)
        assert 'layer_outputs' in processing_info
        assert 'memory_info' in processing_info
        assert 'attention_info' in processing_info
        
    def test_task_specific_processing(self):
        """Test task-specific processing."""
        input_data = torch.randn(2, 5, self.config.input_dim)
        
        # Test different task types
        for task_type in ['n_back', 'stroop', 'planning', 'general']:
            output, processing_info = self.neural_component.forward(
                input_data, task_type=task_type
            )
            
            assert output.shape[0] == 2
            assert output.shape[1] == 5
            assert processing_info['task_type'] == task_type
            
    def test_attention_patterns(self):
        """Test attention pattern extraction."""
        input_data = torch.randn(2, 8, self.config.input_dim)
        
        attention_patterns = self.neural_component.get_attention_patterns(
            input_data, task_type='n_back'
        )
        
        assert 'attention_weights' in attention_patterns
        assert 'attention_control' in attention_patterns
        assert 'attention_intensity' in attention_patterns
        assert 'memory_attention' in attention_patterns
        
    def test_memory_states(self):
        """Test working memory state extraction."""
        input_data = torch.randn(2, 6, self.config.input_dim)
        
        memory_states = self.neural_component.get_memory_states(
            input_data, task_type='planning'
        )
        
        assert 'final_memory' in memory_states
        assert 'memory_evolution' in memory_states
        assert 'controller_states' in memory_states
        
    def test_cognitive_load_analysis(self):
        """Test cognitive load analysis."""
        input_data = torch.randn(3, 7, self.config.input_dim)
        
        cognitive_load = self.neural_component.analyze_cognitive_load(
            input_data, task_type='stroop'
        )
        
        assert 'attention_load' in cognitive_load
        assert 'memory_load' in cognitive_load
        assert 'processing_load' in cognitive_load
        assert 'total_load' in cognitive_load
        
        # Check that all loads are non-negative
        for load_type, load_value in cognitive_load.items():
            assert load_value >= 0
            
    def test_model_info(self):
        """Test model information retrieval."""
        model_info = self.neural_component.get_model_info()
        
        assert 'config' in model_info
        assert 'total_parameters' in model_info
        assert 'memory_size' in model_info
        assert 'attention_heads' in model_info
        assert 'task_heads' in model_info
        
        assert model_info['total_parameters'] > 0


class TestWorkingMemoryModule:
    """Test cases for WorkingMemoryModule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory_module = WorkingMemoryModule(
            memory_size=16,
            memory_dim=32,
            controller_dim=64
        )
        
    def test_initialization(self):
        """Test working memory module initialization."""
        assert self.memory_module.memory_size == 16
        assert self.memory_module.memory_dim == 32
        assert self.memory_module.controller_dim == 64
        
    def test_forward_pass(self):
        """Test forward pass through working memory."""
        batch_size = 2
        seq_len = 5
        input_dim = 32
        
        input_seq = torch.randn(batch_size, seq_len, input_dim)
        
        output_seq, memory_info = self.memory_module.forward(input_seq)
        
        assert output_seq.shape == input_seq.shape
        assert 'final_memory' in memory_info
        assert 'memory_states' in memory_info
        assert 'final_hidden' in memory_info
        
        # Check memory states for each time step
        assert len(memory_info['memory_states']) == seq_len
        
    def test_memory_persistence(self):
        """Test memory persistence across sequences."""
        batch_size = 1
        seq_len = 3
        input_dim = 32
        
        input_seq1 = torch.randn(batch_size, seq_len, input_dim)
        input_seq2 = torch.randn(batch_size, seq_len, input_dim)
        
        # First sequence
        _, memory_info1 = self.memory_module.forward(input_seq1)
        final_hidden1 = memory_info1['final_hidden']
        
        # Second sequence with previous hidden state
        _, memory_info2 = self.memory_module.forward(input_seq2, final_hidden1)
        
        # Memory should be different due to persistence
        assert not torch.equal(
            memory_info1['final_memory'], 
            memory_info2['final_memory']
        )


class TestCognitiveAttentionModule:
    """Test cases for CognitiveAttentionModule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.attention_module = CognitiveAttentionModule(
            input_dim=64,
            num_heads=4,
            attention_dropout=0.1
        )
        
    def test_initialization(self):
        """Test cognitive attention module initialization."""
        assert self.attention_module.input_dim == 64
        assert self.attention_module.num_heads == 4
        assert self.attention_module.head_dim == 16
        
    def test_forward_pass(self):
        """Test forward pass through cognitive attention."""
        batch_size = 2
        seq_len = 8
        input_dim = 64
        
        input_seq = torch.randn(batch_size, seq_len, input_dim)
        
        output, attention_info = self.attention_module.forward(input_seq)
        
        assert output.shape == input_seq.shape
        assert 'attention_weights' in attention_info
        assert 'attention_control' in attention_info
        assert 'attention_intensity' in attention_info
        
    def test_task_specific_attention(self):
        """Test task-specific attention patterns."""
        batch_size = 2
        seq_len = 6
        input_dim = 64
        
        input_seq = torch.randn(batch_size, seq_len, input_dim)
        
        # Test different task types
        for task_type in ['n_back', 'stroop', 'planning']:
            output, attention_info = self.attention_module.forward(
                input_seq, task_type=task_type
            )
            
            assert output.shape == input_seq.shape
            assert attention_info['task_modulation'] == task_type


class TestAdaptiveWeighting:
    """Test cases for AdaptiveWeighting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = WeightingConfig(
            initial_alpha=0.5,
            learning_rate=0.01,
            adaptation_window=5
        )
        self.adaptive_weighting = AdaptiveWeighting(
            self.config,
            symbolic_dim=32,
            neural_dim=32
        )
        
    def test_initialization(self):
        """Test adaptive weighting initialization."""
        assert self.adaptive_weighting.config == self.config
        assert self.adaptive_weighting.state.alpha == 0.5
        assert hasattr(self.adaptive_weighting, 'performance_estimator')
        assert hasattr(self.adaptive_weighting, 'authenticity_estimator')
        
    def test_forward_pass(self):
        """Test forward pass through adaptive weighting."""
        batch_size = 2
        seq_len = 5
        feature_dim = 32
        
        symbolic_output = torch.randn(batch_size, seq_len, feature_dim)
        neural_output = torch.randn(batch_size, seq_len, feature_dim)
        
        symbolic_metadata = {'rules_fired': 3, 'traces': [{}]}
        neural_metadata = {'layer_outputs': [torch.randn(2, 5, 32)]}
        
        combined_output, weighting_info = self.adaptive_weighting.forward(
            symbolic_output, neural_output, symbolic_metadata, neural_metadata
        )
        
        assert combined_output.shape == symbolic_output.shape
        assert 'alpha' in weighting_info
        assert 'performance_score' in weighting_info
        assert 'authenticity_score' in weighting_info
        assert 'efficiency_score' in weighting_info
        
    def test_alpha_adaptation(self):
        """Test alpha parameter adaptation."""
        initial_alpha = self.adaptive_weighting.state.alpha
        
        # Simulate multiple forward passes to trigger adaptation
        for _ in range(10):
            symbolic_output = torch.randn(1, 3, 32)
            neural_output = torch.randn(1, 3, 32)
            symbolic_metadata = {'rules_fired': 2, 'traces': [{}]}
            neural_metadata = {'layer_outputs': [torch.randn(1, 3, 32)]}
            
            _, _ = self.adaptive_weighting.forward(
                symbolic_output, neural_output, symbolic_metadata, neural_metadata
            )
        
        # Alpha should have adapted (may increase or decrease)
        final_alpha = self.adaptive_weighting.state.alpha
        assert 0.1 <= final_alpha <= 0.9  # Within valid range
        
    def test_weighting_history(self):
        """Test weighting history tracking."""
        # Generate some history
        for _ in range(5):
            symbolic_output = torch.randn(1, 2, 32)
            neural_output = torch.randn(1, 2, 32)
            symbolic_metadata = {'rules_fired': 1, 'traces': [{}]}
            neural_metadata = {'layer_outputs': [torch.randn(1, 2, 32)]}
            
            self.adaptive_weighting.forward(
                symbolic_output, neural_output, symbolic_metadata, neural_metadata
            )
        
        history = self.adaptive_weighting.get_weighting_history()
        
        assert 'performance_history' in history
        assert 'authenticity_history' in history
        assert 'efficiency_history' in history
        assert len(history['performance_history']) == 5
        
    def test_manual_alpha_setting(self):
        """Test manual alpha setting."""
        self.adaptive_weighting.set_alpha(0.7)
        assert self.adaptive_weighting.state.alpha == 0.7
        
        # Test clamping
        self.adaptive_weighting.set_alpha(1.5)
        assert self.adaptive_weighting.state.alpha <= 0.9
        
        self.adaptive_weighting.set_alpha(-0.5)
        assert self.adaptive_weighting.state.alpha >= 0.1
        
    def test_adaptation_statistics(self):
        """Test adaptation statistics."""
        stats = self.adaptive_weighting.get_adaptation_statistics()
        
        assert 'current_alpha' in stats
        assert 'adaptation_count' in stats
        assert 'stability' in stats
        assert 'avg_performance' in stats
        assert 'avg_authenticity' in stats
        assert 'avg_efficiency' in stats
        
    def test_reset_adaptation(self):
        """Test adaptation reset."""
        # Generate some history
        for _ in range(3):
            symbolic_output = torch.randn(1, 2, 32)
            neural_output = torch.randn(1, 2, 32)
            symbolic_metadata = {'rules_fired': 1, 'traces': [{}]}
            neural_metadata = {'layer_outputs': [torch.randn(1, 2, 32)]}
            
            self.adaptive_weighting.forward(
                symbolic_output, neural_output, symbolic_metadata, neural_metadata
            )
        
        # Reset
        self.adaptive_weighting.reset_adaptation()
        
        assert self.adaptive_weighting.state.alpha == self.config.initial_alpha
        assert len(self.adaptive_weighting.state.performance_history) == 0
        assert self.adaptive_weighting.state.adaptation_count == 0


class TestPerformanceEstimator:
    """Test cases for PerformanceEstimator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.performance_estimator = PerformanceEstimator(
            input_dim=64,
            hidden_dim=32
        )
        
    def test_initialization(self):
        """Test performance estimator initialization."""
        assert self.performance_estimator is not None
        
    def test_forward_pass(self):
        """Test forward pass through performance estimator."""
        batch_size = 3
        feature_dim = 32
        
        symbolic_output = torch.randn(batch_size, feature_dim)
        neural_output = torch.randn(batch_size, feature_dim)
        
        performance = self.performance_estimator.forward(symbolic_output, neural_output)
        
        assert performance.shape == (batch_size, 1)
        assert torch.all(performance >= 0) and torch.all(performance <= 1)


# Integration tests
class TestNeuroSymbolicIntegration:
    """Integration tests for neuro-symbolic components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.symbolic_component = SymbolicComponent()
        
        neural_config = NeuralConfig(input_dim=64, output_dim=64)
        self.neural_component = NeuralComponent(neural_config)
        
        weighting_config = WeightingConfig()
        self.adaptive_weighting = AdaptiveWeighting(weighting_config)
        
    def test_end_to_end_integration(self):
        """Test end-to-end integration of all components."""
        batch_size = 2
        seq_len = 5
        feature_dim = 64
        
        input_data = torch.randn(batch_size, seq_len, feature_dim)
        context = {'task_type': 'n_back'}
        
        # Symbolic processing
        symbolic_output, symbolic_metadata = self.symbolic_component.forward(
            input_data, context
        )
        
        # Neural processing
        neural_output, neural_metadata = self.neural_component.forward(
            input_data, task_type='n_back'
        )
        
        # Adaptive weighting
        combined_output, weighting_info = self.adaptive_weighting.forward(
            symbolic_output, neural_output, symbolic_metadata, neural_metadata
        )
        
        # Verify shapes and outputs
        assert combined_output.shape == input_data.shape
        assert weighting_info['alpha'] > 0
        assert len(weighting_info['applied_biases']) >= 0
        
    def test_multiple_task_types(self):
        """Test integration across multiple task types."""
        batch_size = 1
        seq_len = 4
        feature_dim = 64
        
        task_types = ['n_back', 'stroop', 'planning']
        
        for task_type in task_types:
            input_data = torch.randn(batch_size, seq_len, feature_dim)
            context = {'task_type': task_type}
            
            # Process through all components
            symbolic_output, symbolic_metadata = self.symbolic_component.forward(
                input_data, context
            )
            
            neural_output, neural_metadata = self.neural_component.forward(
                input_data, task_type=task_type
            )
            
            combined_output, weighting_info = self.adaptive_weighting.forward(
                symbolic_output, neural_output, symbolic_metadata, neural_metadata
            )
            
            # Verify successful processing
            assert combined_output.shape == input_data.shape
            assert not torch.isnan(combined_output).any()
            assert not torch.isinf(combined_output).any()


if __name__ == "__main__":
    pytest.main([__file__])
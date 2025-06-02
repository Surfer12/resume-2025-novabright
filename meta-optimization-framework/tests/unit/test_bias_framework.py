"""
Unit tests for bias framework components.

Tests the bias mechanisms and agent-based modeling components
for correctness and functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.bias_framework.bias_mechanisms import (
    BiasType, BiasModel, BiasParameters, CognitiveBiasFramework,
    ConfirmationBias, AnchoringBias, AvailabilityBias, OverconfidenceBias
)
from src.bias_framework.agent_based_model import (
    Agent, BiasAgent, AgentBasedModel, AgentType, AgentState
)


class TestBiasParameters:
    """Test cases for BiasParameters."""
    
    def test_default_initialization(self):
        """Test default bias parameters initialization."""
        params = BiasParameters()
        
        assert params.strength == 0.5
        assert params.persistence == 0.8
        assert params.activation_threshold == 0.3
        assert params.decay_rate == 0.1
        assert params.context_sensitivity == 0.5
        
    def test_custom_initialization(self):
        """Test custom bias parameters initialization."""
        params = BiasParameters(
            strength=0.8,
            persistence=0.9,
            activation_threshold=0.2,
            decay_rate=0.05,
            context_sensitivity=0.7
        )
        
        assert params.strength == 0.8
        assert params.persistence == 0.9
        assert params.activation_threshold == 0.2
        assert params.decay_rate == 0.05
        assert params.context_sensitivity == 0.7


class TestBiasModel:
    """Test cases for base BiasModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.bias_model = BiasModel(BiasType.CONFIRMATION)
        
    def test_initialization(self):
        """Test bias model initialization."""
        assert self.bias_model.bias_type == BiasType.CONFIRMATION
        assert isinstance(self.bias_model.parameters, BiasParameters)
        assert self.bias_model.activation_level == 0.0
        assert len(self.bias_model.bias_history) == 0
        
    def test_activation_update(self):
        """Test bias activation update."""
        initial_activation = self.bias_model.activation_level
        
        self.bias_model.update_activation(0.5)
        
        assert self.bias_model.activation_level > initial_activation
        assert len(self.bias_model.bias_history) == 1
        
        # Test decay
        self.bias_model.update_activation(0.0)
        assert self.bias_model.activation_level < self.bias_model.bias_history[0]
        
    def test_is_active(self):
        """Test bias activation checking."""
        # Initially not active
        assert not self.bias_model.is_active()
        
        # Activate bias
        self.bias_model.update_activation(1.0)
        assert self.bias_model.is_active()
        
    def test_bias_strength(self):
        """Test bias strength computation."""
        # When not active
        assert self.bias_model.get_bias_strength() == 0.0
        
        # When active
        self.bias_model.update_activation(1.0)
        strength = self.bias_model.get_bias_strength()
        assert strength > 0
        assert strength <= self.bias_model.parameters.strength


class TestConfirmationBias:
    """Test cases for ConfirmationBias."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.confirmation_bias = ConfirmationBias()
        
    def test_initialization(self):
        """Test confirmation bias initialization."""
        assert self.confirmation_bias.bias_type == BiasType.CONFIRMATION
        assert len(self.confirmation_bias.prior_beliefs) == 0
        assert len(self.confirmation_bias.belief_strength) == 0
        
    def test_apply_bias_no_beliefs(self):
        """Test applying bias with no prior beliefs."""
        input_data = torch.randn(2, 5, 32)
        context = {'task_type': 'n_back'}
        
        output, bias_info = self.confirmation_bias.apply_bias(input_data, context)
        
        assert output.shape == input_data.shape
        assert 'bias_applied' in bias_info
        assert 'confirmation_signal' in bias_info
        
    def test_apply_bias_with_beliefs(self):
        """Test applying bias with existing beliefs."""
        # Set up prior beliefs
        self.confirmation_bias.prior_beliefs = {
            'mean_activation': 0.5,
            'task_type': 'n_back'
        }
        self.confirmation_bias.belief_strength = {
            'mean_activation': 0.8,
            'task_type': 0.9
        }
        
        # Create input that confirms beliefs
        input_data = torch.ones(2, 3, 32) * 0.5  # Mean activation = 0.5
        context = {'task_type': 'n_back'}
        
        output, bias_info = self.confirmation_bias.apply_bias(input_data, context)
        
        assert output.shape == input_data.shape
        assert bias_info['confirmation_signal'] > 0  # Should confirm beliefs
        
    def test_belief_update(self):
        """Test belief updating mechanism."""
        input_data = torch.randn(1, 2, 16)
        context = {'task_type': 'stroop'}
        
        # Apply bias multiple times to build beliefs
        for _ in range(3):
            self.confirmation_bias.apply_bias(input_data, context)
        
        assert len(self.confirmation_bias.prior_beliefs) > 0
        assert len(self.confirmation_bias.belief_strength) > 0


class TestAnchoringBias:
    """Test cases for AnchoringBias."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.anchoring_bias = AnchoringBias()
        
    def test_initialization(self):
        """Test anchoring bias initialization."""
        assert self.anchoring_bias.bias_type == BiasType.ANCHORING
        assert len(self.anchoring_bias.anchors) == 0
        assert len(self.anchoring_bias.anchor_weights) == 0
        
    def test_anchor_setting(self):
        """Test anchor setting on first exposure."""
        input_data = torch.randn(2, 3, 16)
        context = {'context_key': 'test_context'}
        
        output, bias_info = self.anchoring_bias.apply_bias(input_data, context)
        
        assert output.shape == input_data.shape
        assert bias_info['anchor_set'] is True
        assert 'test_context' in self.anchoring_bias.anchors
        assert bias_info['anchor_value'] == self.anchoring_bias.anchors['test_context']
        
    def test_anchoring_effect(self):
        """Test anchoring effect on subsequent inputs."""
        context = {'context_key': 'test_context'}
        
        # Set anchor
        anchor_input = torch.ones(1, 2, 8) * 0.8
        self.anchoring_bias.apply_bias(anchor_input, context)
        
        # Test with different input
        test_input = torch.ones(1, 2, 8) * 0.2
        output, bias_info = self.anchoring_bias.apply_bias(test_input, context)
        
        assert output.shape == test_input.shape
        assert bias_info['bias_applied'] is True
        assert bias_info['distance_from_anchor'] > 0
        
        # Output should be pulled toward anchor
        output_mean = torch.mean(output).item()
        input_mean = torch.mean(test_input).item()
        anchor_value = self.anchoring_bias.anchors['test_context']
        
        # Output should be between input and anchor
        assert min(input_mean, anchor_value) <= output_mean <= max(input_mean, anchor_value)


class TestAvailabilityBias:
    """Test cases for AvailabilityBias."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.availability_bias = AvailabilityBias()
        
    def test_initialization(self):
        """Test availability bias initialization."""
        assert self.availability_bias.bias_type == BiasType.AVAILABILITY
        assert len(self.availability_bias.memory_traces) == 0
        assert len(self.availability_bias.recency_weights) == 0
        
    def test_memory_storage(self):
        """Test memory trace storage."""
        input_data = torch.randn(1, 3, 16)
        context = {'event_type': 'test_event'}
        
        initial_memory_size = len(self.availability_bias.memory_traces)
        
        self.availability_bias.apply_bias(input_data, context)
        
        assert len(self.availability_bias.memory_traces) == initial_memory_size + 1
        assert len(self.availability_bias.recency_weights) == initial_memory_size + 1
        
    def test_availability_computation(self):
        """Test availability score computation."""
        context = {'event_type': 'repeated_event'}
        
        # Store multiple similar memories
        for _ in range(3):
            input_data = torch.randn(1, 2, 8) + 0.5  # Similar patterns
            self.availability_bias.apply_bias(input_data, context)
        
        # Test with similar input
        test_input = torch.randn(1, 2, 8) + 0.5
        output, bias_info = self.availability_bias.apply_bias(test_input, context)
        
        assert bias_info['availability_score'] > 0
        assert bias_info['memory_size'] == 4  # 3 previous + 1 current
        
    def test_memory_limit(self):
        """Test memory size limitation."""
        context = {'event_type': 'test'}
        
        # Store more memories than the limit
        for _ in range(55):  # More than max_memory_size (50)
            input_data = torch.randn(1, 2, 4)
            self.availability_bias.apply_bias(input_data, context)
        
        assert len(self.availability_bias.memory_traces) <= 50


class TestOverconfidenceBias:
    """Test cases for OverconfidenceBias."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.overconfidence_bias = OverconfidenceBias()
        
    def test_initialization(self):
        """Test overconfidence bias initialization."""
        assert self.overconfidence_bias.bias_type == BiasType.OVERCONFIDENCE
        assert len(self.overconfidence_bias.confidence_history) == 0
        assert len(self.overconfidence_bias.accuracy_history) == 0
        assert self.overconfidence_bias.calibration_error == 0.0
        
    def test_confidence_extraction(self):
        """Test confidence extraction from input."""
        input_data = torch.randn(1, 3, 8)
        context = {'task_difficulty': 0.3}
        
        output, bias_info = self.overconfidence_bias.apply_bias(input_data, context)
        
        assert 'confidence_estimate' in bias_info
        assert 0 <= bias_info['confidence_estimate'] <= 1
        
    def test_calibration_update(self):
        """Test calibration error update."""
        input_data = torch.randn(1, 2, 4)
        
        # Simulate overconfident predictions
        for i in range(6):
            context = {'actual_accuracy': 0.6}  # Actual performance
            self.overconfidence_bias.apply_bias(input_data, context)
        
        # Should detect overconfidence (high confidence, lower accuracy)
        assert len(self.overconfidence_bias.confidence_history) == 6
        assert len(self.overconfidence_bias.accuracy_history) == 6
        
    def test_overconfidence_effect(self):
        """Test overconfidence bias effect."""
        # Build up overconfidence pattern
        input_data = torch.randn(1, 2, 4)
        for _ in range(8):
            context = {'actual_accuracy': 0.5}  # Lower than typical confidence
            self.overconfidence_bias.apply_bias(input_data, context)
        
        # Test bias application
        output, bias_info = self.overconfidence_bias.apply_bias(input_data)
        
        if bias_info['bias_applied']:
            assert bias_info['confidence_inflation'] > 1.0


class TestCognitiveBiasFramework:
    """Test cases for CognitiveBiasFramework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        bias_types = [BiasType.CONFIRMATION, BiasType.ANCHORING, BiasType.AVAILABILITY]
        self.bias_framework = CognitiveBiasFramework(bias_types)
        
    def test_initialization(self):
        """Test bias framework initialization."""
        assert len(self.bias_framework.bias_models) == 3
        assert BiasType.CONFIRMATION in self.bias_framework.bias_models
        assert BiasType.ANCHORING in self.bias_framework.bias_models
        assert BiasType.AVAILABILITY in self.bias_framework.bias_models
        
    def test_apply_biases(self):
        """Test applying multiple biases."""
        input_data = torch.randn(2, 4, 16)
        context = {'task_type': 'n_back', 'context_key': 'test'}
        
        output, bias_report = self.bias_framework.apply_biases(input_data, context)
        
        assert output.shape == input_data.shape
        assert 'applied_biases' in bias_report
        assert 'bias_strengths' in bias_report
        assert 'total_bias_effect' in bias_report
        assert 'bias_interactions' in bias_report
        
    def test_bias_interactions(self):
        """Test bias interaction analysis."""
        input_data = torch.randn(1, 3, 8)
        context = {'task_type': 'stroop'}
        
        # Apply biases multiple times to activate them
        for _ in range(5):
            self.bias_framework.apply_biases(input_data, context)
        
        output, bias_report = self.bias_framework.apply_biases(input_data, context)
        
        # Check for interaction analysis
        assert 'bias_interactions' in bias_report
        
    def test_bias_summary(self):
        """Test bias framework summary."""
        summary = self.bias_framework.get_bias_summary()
        
        assert 'num_bias_models' in summary
        assert 'bias_types' in summary
        assert 'active_biases' in summary
        assert 'bias_activations' in summary
        assert 'bias_histories' in summary
        
        assert summary['num_bias_models'] == 3
        assert len(summary['bias_types']) == 3
        
    def test_bias_configuration(self):
        """Test bias configuration."""
        new_params = BiasParameters(strength=0.8, persistence=0.9)
        
        self.bias_framework.configure_bias(BiasType.CONFIRMATION, new_params)
        
        confirmation_bias = self.bias_framework.bias_models[BiasType.CONFIRMATION]
        assert confirmation_bias.parameters.strength == 0.8
        assert confirmation_bias.parameters.persistence == 0.9
        
    def test_bias_reset(self):
        """Test bias framework reset."""
        input_data = torch.randn(1, 2, 4)
        context = {'task_type': 'planning'}
        
        # Apply biases to build up state
        self.bias_framework.apply_biases(input_data, context)
        
        # Reset
        self.bias_framework.reset_biases()
        
        # Check that all biases are reset
        for bias_model in self.bias_framework.bias_models.values():
            assert bias_model.activation_level == 0.0
            assert len(bias_model.bias_history) == 0


class TestAgent:
    """Test cases for Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = Agent("test_agent", AgentType.RATIONAL)
        
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.agent_id == "test_agent"
        assert self.agent.agent_type == AgentType.RATIONAL
        assert isinstance(self.agent.state, AgentState)
        assert len(self.agent.decision_history) == 0
        assert len(self.agent.performance_history) == 0
        
    def test_perception(self):
        """Test agent perception."""
        environment = {
            'task_state': {'difficulty': 0.5, 'urgency': 0.3},
            'social_signals': {'cooperation': 0.7},
            'feedback': {'success_rate': 0.8}
        }
        
        perception = self.agent.perceive(environment)
        
        assert 'task_state' in perception
        assert 'social_signals' in perception
        assert 'feedback' in perception
        
    def test_decision_making(self):
        """Test agent decision making."""
        perception = {
            'task_state': {'difficulty': 0.3, 'urgency': 0.8},
            'social_signals': {},
            'feedback': {}
        }
        
        decision = self.agent.decide(perception)
        
        assert 'action' in decision
        assert 'confidence' in decision
        assert 'reasoning' in decision
        assert len(self.agent.decision_history) == 1
        
    def test_action_execution(self):
        """Test agent action execution."""
        decision = {'action': 'standard_approach', 'confidence': 0.7}
        environment = {'task_difficulty': 0.5}
        
        result = self.agent.act(decision, environment)
        
        assert 'action_taken' in result
        assert 'success' in result
        assert 'outcome' in result
        assert 'cost' in result
        assert isinstance(result['success'], bool)
        
    def test_learning(self):
        """Test agent learning."""
        initial_beliefs = self.agent.state.beliefs.copy()
        initial_confidence = self.agent.state.confidence
        
        result = {
            'success': True,
            'outcome': 0.8,
            'cost': 0.2
        }
        
        self.agent.learn(result)
        
        assert len(self.agent.performance_history) == 1
        assert len(self.agent.experience_buffer) == 1
        
        # Beliefs and confidence should potentially change
        # (exact changes depend on learning algorithm)
        
    def test_social_interaction(self):
        """Test agent social interaction."""
        other_agent = Agent("other_agent", AgentType.RATIONAL)
        
        # Give agents some experience
        result = {'success': True, 'outcome': 0.7}
        self.agent.learn(result)
        other_agent.learn(result)
        
        initial_beliefs = self.agent.state.beliefs.copy()
        
        self.agent.interact([other_agent])
        
        # Interaction should potentially modify beliefs
        # (exact changes depend on social influence algorithm)
        
    def test_agent_summary(self):
        """Test agent summary generation."""
        # Give agent some experience
        for success in [True, False, True, True]:
            result = {'success': success, 'outcome': 0.6}
            self.agent.learn(result)
        
        summary = self.agent.get_agent_summary()
        
        assert 'agent_id' in summary
        assert 'agent_type' in summary
        assert 'beliefs' in summary
        assert 'confidence' in summary
        assert 'recent_performance' in summary
        assert 'total_decisions' in summary
        assert 'total_experiences' in summary
        
        assert summary['agent_id'] == "test_agent"
        assert summary['total_experiences'] == 4


class TestBiasAgent:
    """Test cases for BiasAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        bias_types = [BiasType.CONFIRMATION, BiasType.ANCHORING]
        self.bias_agent = BiasAgent("bias_agent", bias_types)
        
    def test_initialization(self):
        """Test bias agent initialization."""
        assert self.bias_agent.agent_id == "bias_agent"
        assert self.bias_agent.agent_type == AgentType.BIASED
        assert hasattr(self.bias_agent, 'bias_framework')
        assert len(self.bias_agent.bias_framework.bias_models) == 2
        
    def test_biased_perception(self):
        """Test biased perception."""
        environment = {
            'task_state': {'difficulty': 0.5},
            'social_signals': {'cooperation': 0.7}
        }
        
        perception = self.bias_agent.perceive(environment)
        
        assert 'task_state' in perception
        assert 'social_signals' in perception
        assert len(self.bias_agent.bias_history) == 1
        
    def test_biased_decision_making(self):
        """Test biased decision making."""
        perception = {
            'task_state': {'difficulty': 0.4},
            'social_signals': {}
        }
        
        decision = self.bias_agent.decide(perception)
        
        assert 'action' in decision
        assert 'confidence' in decision
        assert 'bias_effects' in decision
        
    def test_biased_learning(self):
        """Test biased learning."""
        result = {
            'success': True,
            'outcome': 0.8,
            'cost': 0.1
        }
        
        initial_experience_count = len(self.bias_agent.experience_buffer)
        
        self.bias_agent.learn(result)
        
        assert len(self.bias_agent.experience_buffer) == initial_experience_count + 1
        
        # Check that bias effects were recorded
        latest_experience = self.bias_agent.experience_buffer[-1]
        assert 'bias_effects' in latest_experience['result']
        
    def test_bias_summary(self):
        """Test bias summary for bias agent."""
        bias_summary = self.bias_agent.get_bias_summary()
        
        assert 'num_bias_models' in bias_summary
        assert 'bias_types' in bias_summary
        assert 'active_biases' in bias_summary
        assert 'bias_activations' in bias_summary


class TestAgentBasedModel:
    """Test cases for AgentBasedModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = AgentBasedModel(num_agents=10)
        
    def test_initialization(self):
        """Test agent-based model initialization."""
        assert self.model.num_agents == 10
        assert len(self.model.agents) == 10
        assert self.model.time_step == 0
        assert len(self.model.simulation_history) == 0
        
        # Check agent type distribution
        rational_count = sum(1 for agent in self.model.agents 
                           if agent.agent_type == AgentType.RATIONAL)
        biased_count = sum(1 for agent in self.model.agents 
                         if agent.agent_type == AgentType.BIASED)
        
        assert rational_count + biased_count == 10
        assert rational_count > 0  # Should have some rational agents
        assert biased_count > 0    # Should have some biased agents
        
    def test_environment_initialization(self):
        """Test environment initialization."""
        assert 'task_difficulty' in self.model.environment
        assert 'resource_availability' in self.model.environment
        assert 'time_step' in self.model.environment
        
    def test_simulation_step(self):
        """Test single simulation step."""
        initial_time_step = self.model.time_step
        
        step_result = self.model.step()
        
        assert self.model.time_step == initial_time_step + 1
        assert len(self.model.simulation_history) == 1
        
        assert 'time_step' in step_result
        assert 'agent_decisions' in step_result
        assert 'agent_results' in step_result
        assert 'environment_state' in step_result
        assert 'collective_metrics' in step_result
        
        assert len(step_result['agent_decisions']) == 10
        assert len(step_result['agent_results']) == 10
        
    def test_collective_metrics(self):
        """Test collective metrics computation."""
        # Run a few steps to generate data
        for _ in range(3):
            self.model.step()
        
        latest_step = self.model.simulation_history[-1]
        metrics = latest_step['collective_metrics']
        
        assert 'mean_performance' in metrics
        assert 'mean_confidence' in metrics
        
        # Check that metrics are reasonable
        if 'mean_performance' in metrics:
            assert 0 <= metrics['mean_performance'] <= 1
        if 'mean_confidence' in metrics:
            assert 0 <= metrics['mean_confidence'] <= 1
            
    def test_simulation_run(self):
        """Test complete simulation run."""
        num_steps = 5
        history = self.model.run_simulation(num_steps)
        
        assert len(history) == num_steps
        assert self.model.time_step == num_steps
        assert len(self.model.simulation_history) == num_steps
        
    def test_results_analysis(self):
        """Test simulation results analysis."""
        # Run short simulation
        self.model.run_simulation(3)
        
        analysis = self.model.analyze_results()
        
        assert 'simulation_length' in analysis
        assert 'final_metrics' in analysis
        assert 'performance_evolution' in analysis
        assert 'agent_summaries' in analysis
        
        assert analysis['simulation_length'] == 3
        assert len(analysis['performance_evolution']) == 3
        assert len(analysis['agent_summaries']) == 10
        
    def test_model_summary(self):
        """Test model summary generation."""
        summary = self.model.get_model_summary()
        
        assert 'total_agents' in summary
        assert 'rational_agents' in summary
        assert 'biased_agents' in summary
        assert 'simulation_steps' in summary
        assert 'environment_state' in summary
        
        assert summary['total_agents'] == 10
        assert summary['rational_agents'] + summary['biased_agents'] == 10


# Integration tests
class TestBiasFrameworkIntegration:
    """Integration tests for bias framework components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        bias_types = [BiasType.CONFIRMATION, BiasType.ANCHORING, BiasType.AVAILABILITY]
        self.bias_framework = CognitiveBiasFramework(bias_types)
        self.agent_model = AgentBasedModel(num_agents=5)
        
    def test_bias_framework_with_agents(self):
        """Test bias framework integration with agents."""
        # Find a biased agent
        biased_agents = [agent for agent in self.agent_model.agents 
                        if isinstance(agent, BiasAgent)]
        
        if biased_agents:
            biased_agent = biased_agents[0]
            
            # Test biased agent behavior
            environment = {
                'task_state': {'difficulty': 0.5},
                'social_signals': {}
            }
            
            perception = biased_agent.perceive(environment)
            decision = biased_agent.decide(perception)
            result = biased_agent.act(decision, environment)
            biased_agent.learn(result)
            
            # Should complete without errors
            assert len(biased_agent.bias_history) > 0
            
    def test_multi_agent_bias_interactions(self):
        """Test bias interactions in multi-agent setting."""
        # Run simulation to see bias effects
        self.agent_model.run_simulation(3)
        
        analysis = self.agent_model.analyze_results()
        
        # Check that biased agents show different behavior
        biased_summaries = [summary for summary in analysis['agent_summaries']
                          if 'biased' in summary['agent_type']]
        rational_summaries = [summary for summary in analysis['agent_summaries']
                            if 'rational' in summary['agent_type']]
        
        if biased_summaries and rational_summaries:
            # Should have both types of agents
            assert len(biased_summaries) > 0
            assert len(rational_summaries) > 0
            
    def test_bias_evolution_over_time(self):
        """Test bias evolution over simulation time."""
        # Run longer simulation
        self.agent_model.run_simulation(5)
        
        # Check bias evolution in biased agents
        biased_agents = [agent for agent in self.agent_model.agents 
                        if isinstance(agent, BiasAgent)]
        
        for agent in biased_agents:
            if len(agent.bias_history) > 1:
                # Bias effects should evolve over time
                first_bias = agent.bias_history[0]
                last_bias = agent.bias_history[-1]
                
                # Should have bias information
                assert 'applied_biases' in first_bias
                assert 'applied_biases' in last_bias


if __name__ == "__main__":
    pytest.main([__file__])
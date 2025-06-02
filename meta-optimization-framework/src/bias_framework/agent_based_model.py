"""
Agent-Based Model Implementation

Implements agent-based modeling for studying cognitive biases
in multi-agent systems, including individual agents with biases
and emergent collective behaviors.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import logging

from .bias_mechanisms import BiasType, CognitiveBiasFramework, BiasParameters

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the model."""
    RATIONAL = "rational"
    BIASED = "biased"
    ADAPTIVE = "adaptive"
    SOCIAL = "social"


@dataclass
class AgentState:
    """State of an individual agent."""
    position: Tuple[float, float] = (0.0, 0.0)
    beliefs: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    social_influence: float = 0.3
    learning_rate: float = 0.1
    memory: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default beliefs if empty."""
        if not self.beliefs:
            self.beliefs = {
                'task_difficulty': 0.5,
                'success_probability': 0.5,
                'environment_stability': 0.5
            }


class Agent:
    """
    Base agent class for agent-based modeling.
    
    Represents an individual agent with cognitive capabilities,
    decision-making processes, and social interactions.
    """
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: AgentType = AgentType.RATIONAL,
                 initial_state: AgentState = None):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier for agent
            agent_type: Type of agent
            initial_state: Initial agent state
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = initial_state or AgentState()
        
        # Decision-making components
        self.decision_history = []
        self.performance_history = []
        self.social_network = []
        
        # Learning and adaptation
        self.experience_buffer = []
        self.adaptation_rate = 0.05
        
        logger.debug(f"Initialized {agent_type.value} agent {agent_id}")
    
    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive environment and extract relevant information.
        
        Args:
            environment: Environment state
            
        Returns:
            Perceived information
        """
        # Basic perception - can be overridden by subclasses
        perception = {
            'task_state': environment.get('task_state', {}),
            'social_signals': environment.get('social_signals', {}),
            'feedback': environment.get('feedback', {}),
            'resources': environment.get('resources', {})
        }
        
        # Add noise to perception based on agent capabilities
        perception = self._add_perception_noise(perception)
        
        return perception
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision based on perception.
        
        Args:
            perception: Perceived information
            
        Returns:
            Decision/action
        """
        # Basic rational decision-making
        decision = {
            'action': self._select_action(perception),
            'confidence': self.state.confidence,
            'reasoning': self._generate_reasoning(perception)
        }
        
        # Store decision in history
        self.decision_history.append({
            'perception': perception,
            'decision': decision,
            'timestamp': len(self.decision_history)
        })
        
        return decision
    
    def act(self, decision: Dict[str, Any], environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action in environment.
        
        Args:
            decision: Decision to execute
            environment: Environment state
            
        Returns:
            Action result
        """
        action = decision['action']
        
        # Execute action (simplified)
        result = {
            'action_taken': action,
            'success': self._evaluate_action_success(action, environment),
            'outcome': self._compute_outcome(action, environment),
            'cost': self._compute_action_cost(action)
        }
        
        return result
    
    def learn(self, result: Dict[str, Any]) -> None:
        """
        Learn from action result.
        
        Args:
            result: Result of action execution
        """
        # Update performance history
        self.performance_history.append(result['success'])
        
        # Update beliefs based on outcome
        self._update_beliefs(result)
        
        # Update confidence
        self._update_confidence(result)
        
        # Store experience
        experience = {
            'result': result,
            'previous_state': self.state.beliefs.copy(),
            'timestamp': len(self.experience_buffer)
        }
        self.experience_buffer.append(experience)
        
        # Limit experience buffer size
        if len(self.experience_buffer) > 100:
            self.experience_buffer.pop(0)
    
    def interact(self, other_agents: List['Agent']) -> None:
        """
        Interact with other agents.
        
        Args:
            other_agents: List of other agents to interact with
        """
        if not other_agents:
            return
        
        # Select agents to interact with based on social network
        interaction_partners = self._select_interaction_partners(other_agents)
        
        for partner in interaction_partners:
            # Exchange information
            self._exchange_information(partner)
            
            # Social influence
            self._apply_social_influence(partner)
    
    def _add_perception_noise(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to perception based on agent limitations."""
        noisy_perception = perception.copy()
        
        # Add Gaussian noise to numerical values
        noise_level = 0.1
        for key, value in perception.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        noise = np.random.normal(0, noise_level)
                        noisy_perception[key][subkey] = subvalue + noise
        
        return noisy_perception
    
    def _select_action(self, perception: Dict[str, Any]) -> str:
        """Select action based on perception."""
        # Simple action selection based on task state
        task_state = perception.get('task_state', {})
        
        if task_state.get('difficulty', 0.5) > 0.7:
            return "careful_approach"
        elif task_state.get('urgency', 0.5) > 0.7:
            return "quick_response"
        else:
            return "standard_approach"
    
    def _generate_reasoning(self, perception: Dict[str, Any]) -> str:
        """Generate reasoning for decision."""
        return f"Based on {self.agent_type.value} decision-making process"
    
    def _evaluate_action_success(self, action: str, environment: Dict[str, Any]) -> bool:
        """Evaluate if action was successful."""
        # Simplified success evaluation
        base_success_rate = 0.7
        
        # Adjust based on action type
        if action == "careful_approach":
            success_rate = base_success_rate + 0.1
        elif action == "quick_response":
            success_rate = base_success_rate - 0.1
        else:
            success_rate = base_success_rate
        
        # Add environmental factors
        difficulty = environment.get('task_difficulty', 0.5)
        success_rate *= (1.0 - difficulty * 0.3)
        
        return random.random() < success_rate
    
    def _compute_outcome(self, action: str, environment: Dict[str, Any]) -> float:
        """Compute numerical outcome of action."""
        base_outcome = 0.5
        
        if action == "careful_approach":
            outcome = base_outcome + 0.2
        elif action == "quick_response":
            outcome = base_outcome + 0.1
        else:
            outcome = base_outcome
        
        # Add noise
        outcome += np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, outcome))
    
    def _compute_action_cost(self, action: str) -> float:
        """Compute cost of action."""
        costs = {
            "careful_approach": 0.3,
            "quick_response": 0.1,
            "standard_approach": 0.2
        }
        return costs.get(action, 0.2)
    
    def _update_beliefs(self, result: Dict[str, Any]) -> None:
        """Update beliefs based on action result."""
        learning_rate = self.state.learning_rate
        
        # Update task difficulty belief
        if result['success']:
            # Success suggests task was easier than expected
            self.state.beliefs['task_difficulty'] *= (1 - learning_rate * 0.1)
        else:
            # Failure suggests task was harder than expected
            self.state.beliefs['task_difficulty'] += learning_rate * 0.1
        
        # Update success probability belief
        recent_successes = sum(self.performance_history[-10:]) if self.performance_history else 0
        recent_trials = min(len(self.performance_history), 10)
        if recent_trials > 0:
            empirical_success_rate = recent_successes / recent_trials
            self.state.beliefs['success_probability'] = (
                (1 - learning_rate) * self.state.beliefs['success_probability'] +
                learning_rate * empirical_success_rate
            )
        
        # Clamp beliefs to valid range
        for key in self.state.beliefs:
            self.state.beliefs[key] = max(0.0, min(1.0, self.state.beliefs[key]))
    
    def _update_confidence(self, result: Dict[str, Any]) -> None:
        """Update confidence based on action result."""
        if result['success']:
            self.state.confidence = min(1.0, self.state.confidence + 0.05)
        else:
            self.state.confidence = max(0.0, self.state.confidence - 0.1)
    
    def _select_interaction_partners(self, other_agents: List['Agent']) -> List['Agent']:
        """Select agents to interact with."""
        # Simple selection: interact with nearby agents
        max_interactions = 3
        
        if len(other_agents) <= max_interactions:
            return other_agents
        
        # Random selection for simplicity
        return random.sample(other_agents, max_interactions)
    
    def _exchange_information(self, partner: 'Agent') -> None:
        """Exchange information with another agent."""
        # Share recent experiences
        if self.experience_buffer and partner.experience_buffer:
            my_recent = self.experience_buffer[-1]
            partner_recent = partner.experience_buffer[-1]
            
            # Learn from partner's experience (simplified)
            if partner_recent['result']['success'] and not my_recent['result']['success']:
                # Partner succeeded where I failed - learn from them
                influence_rate = self.state.social_influence * 0.1
                for key, value in partner.state.beliefs.items():
                    if key in self.state.beliefs:
                        self.state.beliefs[key] = (
                            (1 - influence_rate) * self.state.beliefs[key] +
                            influence_rate * value
                        )
    
    def _apply_social_influence(self, partner: 'Agent') -> None:
        """Apply social influence from another agent."""
        influence_strength = self.state.social_influence
        
        # Confidence influence
        confidence_diff = partner.state.confidence - self.state.confidence
        self.state.confidence += influence_strength * confidence_diff * 0.1
        self.state.confidence = max(0.0, min(1.0, self.state.confidence))
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of agent state."""
        recent_performance = (
            np.mean(self.performance_history[-10:]) 
            if self.performance_history else 0.0
        )
        
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'beliefs': self.state.beliefs.copy(),
            'confidence': self.state.confidence,
            'recent_performance': recent_performance,
            'total_decisions': len(self.decision_history),
            'total_experiences': len(self.experience_buffer)
        }


class BiasAgent(Agent):
    """
    Agent with cognitive biases.
    
    Extends the base Agent class to include cognitive bias
    mechanisms that affect perception, decision-making, and learning.
    """
    
    def __init__(self, 
                 agent_id: str,
                 bias_types: List[BiasType] = None,
                 bias_parameters: Dict[BiasType, BiasParameters] = None,
                 initial_state: AgentState = None):
        """
        Initialize biased agent.
        
        Args:
            agent_id: Unique identifier for agent
            bias_types: List of bias types to include
            bias_parameters: Parameters for each bias type
            initial_state: Initial agent state
        """
        super().__init__(agent_id, AgentType.BIASED, initial_state)
        
        # Initialize bias framework
        self.bias_framework = CognitiveBiasFramework(bias_types)
        
        # Configure bias parameters
        if bias_parameters:
            for bias_type, parameters in bias_parameters.items():
                self.bias_framework.configure_bias(bias_type, parameters)
        
        self.bias_history = []
        
        logger.debug(f"Initialized BiasAgent {agent_id} with {len(self.bias_framework.bias_models)} biases")
    
    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive environment with bias effects."""
        # Get base perception
        base_perception = super().perceive(environment)
        
        # Convert perception to tensor for bias application
        perception_tensor = self._perception_to_tensor(base_perception)
        
        # Apply biases to perception
        biased_tensor, bias_report = self.bias_framework.apply_biases(
            perception_tensor, 
            context={'agent_id': self.agent_id, 'environment': environment}
        )
        
        # Convert back to perception format
        biased_perception = self._tensor_to_perception(biased_tensor, base_perception)
        
        # Store bias information
        self.bias_history.append(bias_report)
        
        return biased_perception
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision with bias effects."""
        # Get base decision
        base_decision = super().decide(perception)
        
        # Apply biases to decision confidence and reasoning
        decision_context = {
            'agent_beliefs': self.state.beliefs,
            'recent_performance': self.performance_history[-5:] if self.performance_history else [],
            'decision_type': base_decision.get('action', 'unknown')
        }
        
        # Convert decision confidence to tensor
        confidence_tensor = torch.tensor([base_decision['confidence']], dtype=torch.float32)
        
        # Apply biases
        biased_confidence_tensor, bias_report = self.bias_framework.apply_biases(
            confidence_tensor, context=decision_context
        )
        
        # Update decision with biased confidence
        biased_decision = base_decision.copy()
        biased_decision['confidence'] = float(biased_confidence_tensor[0])
        biased_decision['bias_effects'] = bias_report
        
        return biased_decision
    
    def learn(self, result: Dict[str, Any]) -> None:
        """Learn with bias effects."""
        # Apply biases to learning process
        learning_context = {
            'outcome': result.get('outcome', 0.5),
            'success': result.get('success', False),
            'agent_expectations': self.state.beliefs.copy()
        }
        
        # Convert outcome to tensor
        outcome_tensor = torch.tensor([result.get('outcome', 0.5)], dtype=torch.float32)
        
        # Apply biases to outcome interpretation
        biased_outcome_tensor, bias_report = self.bias_framework.apply_biases(
            outcome_tensor, context=learning_context
        )
        
        # Create biased result for learning
        biased_result = result.copy()
        biased_result['outcome'] = float(biased_outcome_tensor[0])
        biased_result['bias_effects'] = bias_report
        
        # Learn from biased result
        super().learn(biased_result)
    
    def _perception_to_tensor(self, perception: Dict[str, Any]) -> torch.Tensor:
        """Convert perception dictionary to tensor."""
        # Extract numerical values from perception
        values = []
        
        def extract_values(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    extract_values(v)
            elif isinstance(obj, (int, float)):
                values.append(float(obj))
            elif isinstance(obj, list):
                for item in obj:
                    extract_values(item)
        
        extract_values(perception)
        
        # Pad or truncate to fixed size
        target_size = 16
        if len(values) < target_size:
            values.extend([0.0] * (target_size - len(values)))
        else:
            values = values[:target_size]
        
        return torch.tensor(values, dtype=torch.float32)
    
    def _tensor_to_perception(self, 
                             tensor: torch.Tensor,
                             original_perception: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tensor back to perception format."""
        # For simplicity, just scale the original perception values
        scaling_factor = torch.mean(tensor).item() / max(0.1, self._get_perception_mean(original_perception))
        
        def scale_values(obj):
            if isinstance(obj, dict):
                return {k: scale_values(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float)):
                return obj * scaling_factor
            elif isinstance(obj, list):
                return [scale_values(item) for item in obj]
            else:
                return obj
        
        return scale_values(original_perception)
    
    def _get_perception_mean(self, perception: Dict[str, Any]) -> float:
        """Get mean of numerical values in perception."""
        values = []
        
        def extract_values(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    extract_values(v)
            elif isinstance(obj, (int, float)):
                values.append(float(obj))
            elif isinstance(obj, list):
                for item in obj:
                    extract_values(item)
        
        extract_values(perception)
        return np.mean(values) if values else 1.0
    
    def get_bias_summary(self) -> Dict[str, Any]:
        """Get summary of bias effects."""
        return self.bias_framework.get_bias_summary()


class AgentBasedModel:
    """
    Main agent-based model for studying cognitive biases.
    
    Manages multiple agents, environment dynamics, and
    collective behavior analysis.
    """
    
    def __init__(self, 
                 num_agents: int = 50,
                 environment_config: Dict[str, Any] = None):
        """
        Initialize agent-based model.
        
        Args:
            num_agents: Number of agents in the model
            environment_config: Environment configuration
        """
        self.num_agents = num_agents
        self.environment_config = environment_config or {}
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Environment state
        self.environment = self._initialize_environment()
        
        # Simulation state
        self.time_step = 0
        self.simulation_history = []
        
        logger.info(f"Initialized AgentBasedModel with {num_agents} agents")
    
    def _create_agents(self) -> List[Agent]:
        """Create initial population of agents."""
        agents = []
        
        # Create mix of agent types
        for i in range(self.num_agents):
            agent_id = f"agent_{i:03d}"
            
            # Determine agent type
            if i < self.num_agents * 0.3:  # 30% rational agents
                agent = Agent(agent_id, AgentType.RATIONAL)
            else:  # 70% biased agents
                # Random bias types for each agent
                bias_types = random.sample(
                    list(BiasType), 
                    random.randint(1, 3)
                )
                agent = BiasAgent(agent_id, bias_types)
            
            agents.append(agent)
        
        return agents
    
    def _initialize_environment(self) -> Dict[str, Any]:
        """Initialize environment state."""
        return {
            'task_difficulty': self.environment_config.get('task_difficulty', 0.5),
            'resource_availability': self.environment_config.get('resource_availability', 1.0),
            'social_signals': {},
            'global_feedback': {},
            'time_step': 0
        }
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Returns:
            Step results and statistics
        """
        step_results = {
            'time_step': self.time_step,
            'agent_decisions': [],
            'agent_results': [],
            'environment_state': self.environment.copy(),
            'collective_metrics': {}
        }
        
        # Agent perception and decision phase
        for agent in self.agents:
            # Agent perceives environment
            perception = agent.perceive(self.environment)
            
            # Agent makes decision
            decision = agent.decide(perception)
            step_results['agent_decisions'].append({
                'agent_id': agent.agent_id,
                'decision': decision
            })
            
            # Agent acts in environment
            result = agent.act(decision, self.environment)
            step_results['agent_results'].append({
                'agent_id': agent.agent_id,
                'result': result
            })
            
            # Agent learns from result
            agent.learn(result)
        
        # Social interaction phase
        self._social_interaction_phase()
        
        # Update environment
        self._update_environment(step_results)
        
        # Compute collective metrics
        step_results['collective_metrics'] = self._compute_collective_metrics()
        
        # Store in history
        self.simulation_history.append(step_results)
        self.time_step += 1
        
        return step_results
    
    def _social_interaction_phase(self) -> None:
        """Execute social interactions between agents."""
        # Random pairing for interactions
        shuffled_agents = self.agents.copy()
        random.shuffle(shuffled_agents)
        
        for i in range(0, len(shuffled_agents) - 1, 2):
            agent1 = shuffled_agents[i]
            agent2 = shuffled_agents[i + 1]
            
            # Agents interact with each other
            agent1.interact([agent2])
            agent2.interact([agent1])
    
    def _update_environment(self, step_results: Dict[str, Any]) -> None:
        """Update environment based on agent actions."""
        # Update task difficulty based on collective performance
        success_rate = np.mean([
            result['result']['success'] 
            for result in step_results['agent_results']
        ])
        
        # Adaptive difficulty
        if success_rate > 0.8:
            self.environment['task_difficulty'] = min(1.0, 
                self.environment['task_difficulty'] + 0.05)
        elif success_rate < 0.4:
            self.environment['task_difficulty'] = max(0.0,
                self.environment['task_difficulty'] - 0.05)
        
        # Update time step
        self.environment['time_step'] = self.time_step
    
    def _compute_collective_metrics(self) -> Dict[str, float]:
        """Compute collective behavior metrics."""
        metrics = {}
        
        # Performance metrics
        recent_performance = []
        for agent in self.agents:
            if agent.performance_history:
                recent_perf = np.mean(agent.performance_history[-5:])
                recent_performance.append(recent_perf)
        
        if recent_performance:
            metrics['mean_performance'] = np.mean(recent_performance)
            metrics['performance_std'] = np.std(recent_performance)
        
        # Confidence metrics
        confidences = [agent.state.confidence for agent in self.agents]
        metrics['mean_confidence'] = np.mean(confidences)
        metrics['confidence_std'] = np.std(confidences)
        
        # Bias metrics (for biased agents)
        biased_agents = [agent for agent in self.agents if isinstance(agent, BiasAgent)]
        if biased_agents:
            active_biases = []
            for agent in biased_agents:
                bias_summary = agent.get_bias_summary()
                active_biases.extend(bias_summary.get('active_biases', []))
            
            metrics['num_active_biases'] = len(active_biases)
            metrics['bias_diversity'] = len(set(active_biases))
        
        # Belief diversity
        belief_values = []
        for agent in self.agents:
            for belief_value in agent.state.beliefs.values():
                belief_values.append(belief_value)
        
        if belief_values:
            metrics['belief_diversity'] = np.std(belief_values)
        
        return metrics
    
    def run_simulation(self, num_steps: int = 100) -> List[Dict[str, Any]]:
        """
        Run complete simulation.
        
        Args:
            num_steps: Number of simulation steps
            
        Returns:
            Complete simulation history
        """
        logger.info(f"Running simulation for {num_steps} steps")
        
        for step in range(num_steps):
            step_result = self.step()
            
            if step % 10 == 0:
                logger.info(f"Step {step}: mean_performance={step_result['collective_metrics'].get('mean_performance', 0):.3f}")
        
        logger.info("Simulation completed")
        return self.simulation_history
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze simulation results."""
        if not self.simulation_history:
            return {}
        
        analysis = {
            'simulation_length': len(self.simulation_history),
            'final_metrics': self.simulation_history[-1]['collective_metrics'],
            'performance_evolution': [],
            'bias_evolution': [],
            'agent_summaries': []
        }
        
        # Extract performance evolution
        for step_result in self.simulation_history:
            metrics = step_result['collective_metrics']
            analysis['performance_evolution'].append(metrics.get('mean_performance', 0))
            analysis['bias_evolution'].append(metrics.get('num_active_biases', 0))
        
        # Agent summaries
        for agent in self.agents:
            analysis['agent_summaries'].append(agent.get_agent_summary())
        
        # Compute trends
        if len(analysis['performance_evolution']) > 10:
            recent_perf = analysis['performance_evolution'][-10:]
            early_perf = analysis['performance_evolution'][:10]
            analysis['performance_trend'] = np.mean(recent_perf) - np.mean(early_perf)
        
        return analysis
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model state."""
        rational_agents = sum(1 for agent in self.agents if agent.agent_type == AgentType.RATIONAL)
        biased_agents = sum(1 for agent in self.agents if agent.agent_type == AgentType.BIASED)
        
        return {
            'total_agents': len(self.agents),
            'rational_agents': rational_agents,
            'biased_agents': biased_agents,
            'simulation_steps': self.time_step,
            'environment_state': self.environment.copy(),
            'history_length': len(self.simulation_history)
        }
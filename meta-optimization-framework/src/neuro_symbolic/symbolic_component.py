"""
Symbolic Component Implementation

Implements symbolic reasoning logic S(x) for the neuro-symbolic framework.
Provides rule-based reasoning, logical inference, and symbolic pattern matching
for cognitive tasks.
"""

import torch
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SymbolicRule:
    """Represents a symbolic rule for reasoning."""
    condition: str
    action: str
    confidence: float = 1.0
    priority: int = 0
    
    def __str__(self) -> str:
        return f"IF {self.condition} THEN {self.action} (conf: {self.confidence:.3f})"


@dataclass
class SymbolicState:
    """Represents the current symbolic state."""
    facts: List[str]
    variables: Dict[str, Any]
    active_rules: List[SymbolicRule]
    
    def add_fact(self, fact: str) -> None:
        """Add a fact to the current state."""
        if fact not in self.facts:
            self.facts.append(fact)
    
    def remove_fact(self, fact: str) -> None:
        """Remove a fact from the current state."""
        if fact in self.facts:
            self.facts.remove(fact)


class SymbolicComponent:
    """
    Symbolic reasoning component S(x) for neuro-symbolic integration.
    
    Implements rule-based reasoning, pattern matching, and logical inference
    for cognitive tasks like N-back, Stroop, and planning.
    """
    
    def __init__(self, 
                 rule_base: Optional[List[SymbolicRule]] = None,
                 max_inference_steps: int = 10,
                 confidence_threshold: float = 0.5):
        """
        Initialize symbolic component.
        
        Args:
            rule_base: Initial set of symbolic rules
            max_inference_steps: Maximum steps for inference chain
            confidence_threshold: Minimum confidence for rule activation
        """
        self.rule_base = rule_base or []
        self.max_inference_steps = max_inference_steps
        self.confidence_threshold = confidence_threshold
        self.state = SymbolicState(facts=[], variables={}, active_rules=[])
        
        # Initialize cognitive task-specific rules
        self._initialize_cognitive_rules()
        
        logger.info(f"Initialized SymbolicComponent with {len(self.rule_base)} rules")
    
    def _initialize_cognitive_rules(self) -> None:
        """Initialize rules for cognitive tasks."""
        
        # N-back task rules
        n_back_rules = [
            SymbolicRule(
                condition="current_stimulus == stimulus_n_back",
                action="respond_match",
                confidence=0.9,
                priority=1
            ),
            SymbolicRule(
                condition="current_stimulus != stimulus_n_back",
                action="respond_no_match", 
                confidence=0.9,
                priority=1
            ),
            SymbolicRule(
                condition="working_memory_load > capacity",
                action="reduce_confidence",
                confidence=0.8,
                priority=2
            )
        ]
        
        # Stroop task rules
        stroop_rules = [
            SymbolicRule(
                condition="word_meaning == color_perception",
                action="respond_congruent",
                confidence=0.95,
                priority=1
            ),
            SymbolicRule(
                condition="word_meaning != color_perception",
                action="inhibit_word_response",
                confidence=0.7,
                priority=2
            ),
            SymbolicRule(
                condition="conflict_detected",
                action="increase_attention",
                confidence=0.8,
                priority=1
            )
        ]
        
        # Planning task rules
        planning_rules = [
            SymbolicRule(
                condition="goal_state_defined",
                action="decompose_into_subgoals",
                confidence=0.9,
                priority=1
            ),
            SymbolicRule(
                condition="subgoal_achieved",
                action="proceed_to_next_subgoal",
                confidence=0.95,
                priority=1
            ),
            SymbolicRule(
                condition="obstacle_detected",
                action="replan_strategy",
                confidence=0.8,
                priority=2
            )
        ]
        
        self.rule_base.extend(n_back_rules + stroop_rules + planning_rules)
    
    def forward(self, 
                input_data: torch.Tensor,
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through symbolic component.
        
        Args:
            input_data: Input tensor [batch_size, seq_len, feature_dim]
            context: Additional context information
            
        Returns:
            Tuple of (symbolic_output, reasoning_trace)
        """
        batch_size, seq_len, feature_dim = input_data.shape
        context = context or {}
        
        # Initialize output tensor
        symbolic_output = torch.zeros_like(input_data)
        reasoning_traces = []
        
        for batch_idx in range(batch_size):
            batch_trace = []
            
            for seq_idx in range(seq_len):
                # Extract current input
                current_input = input_data[batch_idx, seq_idx, :]
                
                # Convert to symbolic representation
                symbolic_state = self._tensorize_to_symbolic(current_input, context)
                
                # Perform symbolic reasoning
                reasoning_result = self._symbolic_reasoning(symbolic_state)
                
                # Convert back to tensor representation
                output_tensor = self._symbolic_to_tensor(reasoning_result, feature_dim)
                symbolic_output[batch_idx, seq_idx, :] = output_tensor
                
                batch_trace.append(reasoning_result)
            
            reasoning_traces.append(batch_trace)
        
        # Compile reasoning metadata
        reasoning_metadata = {
            'traces': reasoning_traces,
            'rules_fired': sum(len(trace.get('fired_rules', [])) if isinstance(trace, dict) else 0 for trace in reasoning_traces),
            'confidence_scores': self._extract_confidence_scores(reasoning_traces),
            'symbolic_features': self._extract_symbolic_features(reasoning_traces)
        }
        
        return symbolic_output, reasoning_metadata
    
    def _tensorize_to_symbolic(self, 
                              tensor_input: torch.Tensor,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tensor input to symbolic representation."""
        
        # Extract key features from tensor
        features = tensor_input.detach().cpu().numpy()
        
        # Create symbolic state based on input patterns
        symbolic_state = {
            'input_magnitude': float(torch.norm(tensor_input).item()),
            'dominant_features': self._extract_dominant_features(features),
            'pattern_type': self._classify_input_pattern(features),
            'context': context
        }
        
        # Add task-specific symbolic features
        if 'task_type' in context:
            if context['task_type'] == 'n_back':
                symbolic_state.update(self._extract_n_back_features(features, context))
            elif context['task_type'] == 'stroop':
                symbolic_state.update(self._extract_stroop_features(features, context))
            elif context['task_type'] == 'planning':
                symbolic_state.update(self._extract_planning_features(features, context))
        
        return symbolic_state
    
    def _extract_dominant_features(self, features: np.ndarray, top_k: int = 5) -> List[int]:
        """Extract indices of dominant features."""
        return np.argsort(np.abs(features))[-top_k:].tolist()
    
    def _classify_input_pattern(self, features: np.ndarray) -> str:
        """Classify the input pattern type."""
        # Simple pattern classification based on feature statistics
        mean_val = np.mean(features)
        std_val = np.std(features)
        
        if std_val < 0.1:
            return "uniform"
        elif mean_val > 0.5:
            return "high_activation"
        elif mean_val < -0.5:
            return "low_activation"
        else:
            return "mixed"
    
    def _extract_n_back_features(self, features: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract N-back specific symbolic features."""
        return {
            'stimulus_strength': float(np.max(features)),
            'memory_load': context.get('n_back_level', 2),
            'position_in_sequence': context.get('sequence_position', 0),
            'working_memory_items': context.get('working_memory', [])
        }
    
    def _extract_stroop_features(self, features: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Stroop-specific symbolic features."""
        # Assume features encode word and color information
        word_features = features[:len(features)//2]
        color_features = features[len(features)//2:]
        
        return {
            'word_activation': float(np.max(word_features)),
            'color_activation': float(np.max(color_features)),
            'conflict_level': float(np.abs(np.max(word_features) - np.max(color_features))),
            'response_tendency': 'word' if np.max(word_features) > np.max(color_features) else 'color'
        }
    
    def _extract_planning_features(self, features: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract planning-specific symbolic features."""
        return {
            'goal_clarity': float(np.std(features)),  # Higher std = clearer goal structure
            'subgoal_count': context.get('subgoal_count', 1),
            'planning_depth': context.get('planning_depth', 1),
            'current_goal_state': context.get('current_goal', 'undefined')
        }
    
    def _symbolic_reasoning(self, symbolic_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform symbolic reasoning on the current state."""
        
        reasoning_result = {
            'input_state': symbolic_state.copy(),
            'fired_rules': [],
            'inferences': [],
            'final_decision': None,
            'confidence': 0.0
        }
        
        # Apply inference rules
        for step in range(self.max_inference_steps):
            applicable_rules = self._find_applicable_rules(symbolic_state)
            
            if not applicable_rules:
                break
            
            # Select highest priority rule
            selected_rule = max(applicable_rules, key=lambda r: (r.priority, r.confidence))
            
            if selected_rule.confidence < self.confidence_threshold:
                break
            
            # Apply rule
            inference = self._apply_rule(selected_rule, symbolic_state)
            reasoning_result['fired_rules'].append(selected_rule)
            reasoning_result['inferences'].append(inference)
            
            # Update symbolic state
            symbolic_state.update(inference.get('state_updates', {}))
        
        # Make final decision
        reasoning_result['final_decision'] = self._make_decision(symbolic_state, reasoning_result)
        reasoning_result['confidence'] = self._compute_reasoning_confidence(reasoning_result)
        
        return reasoning_result
    
    def _find_applicable_rules(self, symbolic_state: Dict[str, Any]) -> List[SymbolicRule]:
        """Find rules applicable to current symbolic state."""
        applicable_rules = []
        
        for rule in self.rule_base:
            if self._evaluate_condition(rule.condition, symbolic_state):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _evaluate_condition(self, condition: str, symbolic_state: Dict[str, Any]) -> bool:
        """Evaluate if a rule condition is satisfied."""
        
        # Simple condition evaluation (can be extended with proper parser)
        try:
            # Replace symbolic variables with actual values
            eval_condition = condition
            
            for key, value in symbolic_state.items():
                if isinstance(value, (int, float)):
                    eval_condition = eval_condition.replace(key, str(value))
                elif isinstance(value, str):
                    eval_condition = eval_condition.replace(key, f"'{value}'")
            
            # Handle common patterns
            if "==" in eval_condition:
                parts = eval_condition.split("==")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # Get values from symbolic state
                    left_val = symbolic_state.get(left, left)
                    right_val = symbolic_state.get(right.strip("'"), right.strip("'"))
                    
                    return left_val == right_val
            
            elif "!=" in eval_condition:
                parts = eval_condition.split("!=")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    left_val = symbolic_state.get(left, left)
                    right_val = symbolic_state.get(right.strip("'"), right.strip("'"))
                    
                    return left_val != right_val
            
            elif ">" in eval_condition:
                parts = eval_condition.split(">")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    left_val = symbolic_state.get(left, 0)
                    right_val = float(right) if right.replace('.', '').isdigit() else symbolic_state.get(right, 0)
                    
                    return float(left_val) > float(right_val)
            
            # Default: check if condition exists as a fact
            return condition in symbolic_state.get('facts', [])
            
        except Exception as e:
            logger.warning(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _apply_rule(self, rule: SymbolicRule, symbolic_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a symbolic rule and return the inference result."""
        
        inference = {
            'rule': rule,
            'action_taken': rule.action,
            'state_updates': {},
            'confidence': rule.confidence
        }
        
        # Apply action based on rule
        action = rule.action
        
        if action == "respond_match":
            inference['state_updates'] = {'response': 'match', 'response_confidence': rule.confidence}
        elif action == "respond_no_match":
            inference['state_updates'] = {'response': 'no_match', 'response_confidence': rule.confidence}
        elif action == "reduce_confidence":
            current_conf = symbolic_state.get('response_confidence', 1.0)
            inference['state_updates'] = {'response_confidence': current_conf * 0.8}
        elif action == "respond_congruent":
            inference['state_updates'] = {'response': 'congruent', 'response_confidence': rule.confidence}
        elif action == "inhibit_word_response":
            inference['state_updates'] = {'inhibition_active': True, 'response_delay': 0.1}
        elif action == "increase_attention":
            current_attention = symbolic_state.get('attention_level', 0.5)
            inference['state_updates'] = {'attention_level': min(1.0, current_attention + 0.2)}
        elif action == "decompose_into_subgoals":
            inference['state_updates'] = {'subgoals_identified': True, 'planning_active': True}
        elif action == "proceed_to_next_subgoal":
            current_subgoal = symbolic_state.get('current_subgoal', 0)
            inference['state_updates'] = {'current_subgoal': current_subgoal + 1}
        elif action == "replan_strategy":
            inference['state_updates'] = {'replanning_active': True, 'strategy_changed': True}
        
        return inference
    
    def _make_decision(self, symbolic_state: Dict[str, Any], reasoning_result: Dict[str, Any]) -> str:
        """Make final decision based on reasoning process."""
        
        # Check if explicit response was generated
        if 'response' in symbolic_state:
            return symbolic_state['response']
        
        # Default decision based on input characteristics
        input_magnitude = symbolic_state.get('input_magnitude', 0)
        
        if input_magnitude > 0.7:
            return 'high_confidence_response'
        elif input_magnitude > 0.3:
            return 'medium_confidence_response'
        else:
            return 'low_confidence_response'
    
    def _compute_reasoning_confidence(self, reasoning_result: Dict[str, Any]) -> float:
        """Compute overall confidence in reasoning process."""
        
        if not reasoning_result['fired_rules']:
            return 0.1  # Low confidence if no rules fired
        
        # Average confidence of fired rules
        rule_confidences = [rule.confidence for rule in reasoning_result['fired_rules']]
        avg_confidence = np.mean(rule_confidences)
        
        # Adjust based on number of inference steps
        num_steps = len(reasoning_result['inferences'])
        step_penalty = max(0, (num_steps - 3) * 0.1)  # Penalty for too many steps
        
        final_confidence = max(0.1, avg_confidence - step_penalty)
        
        return float(final_confidence)
    
    def _symbolic_to_tensor(self, reasoning_result: Dict[str, Any], feature_dim: int) -> torch.Tensor:
        """Convert symbolic reasoning result back to tensor representation."""
        
        # Initialize output tensor
        output = torch.zeros(feature_dim)
        
        # Encode decision
        decision = reasoning_result.get('final_decision', 'no_decision')
        confidence = reasoning_result.get('confidence', 0.0)
        
        # Map decisions to tensor patterns
        if decision == 'match' or decision == 'high_confidence_response':
            output[0] = confidence
            output[1] = 1.0
        elif decision == 'no_match' or decision == 'low_confidence_response':
            output[0] = confidence
            output[1] = 0.0
        elif decision == 'congruent':
            output[0] = confidence
            output[2] = 1.0
        elif decision == 'medium_confidence_response':
            output[0] = confidence
            output[1] = 0.5
        
        # Encode reasoning metadata
        num_rules_fired = len(reasoning_result.get('fired_rules', []))
        output[3] = min(1.0, num_rules_fired / 5.0)  # Normalize to [0, 1]
        
        # Encode symbolic features
        input_state = reasoning_result.get('input_state', {})
        if 'attention_level' in input_state:
            output[4] = input_state['attention_level']
        if 'conflict_level' in input_state:
            output[5] = min(1.0, input_state['conflict_level'])
        
        return output
    
    def _extract_confidence_scores(self, reasoning_traces: List[List[Dict[str, Any]]]) -> List[float]:
        """Extract confidence scores from reasoning traces."""
        scores = []
        for batch_trace in reasoning_traces:
            batch_scores = [trace.get('confidence', 0.0) for trace in batch_trace]
            scores.extend(batch_scores)
        return scores
    
    def _extract_symbolic_features(self, reasoning_traces: List[List[Dict[str, Any]]]) -> Dict[str, List[Any]]:
        """Extract symbolic features from reasoning traces."""
        features = {
            'decisions': [],
            'rules_fired_count': [],
            'inference_steps': []
        }
        
        for batch_trace in reasoning_traces:
            for trace in batch_trace:
                features['decisions'].append(trace.get('final_decision', 'unknown'))
                features['rules_fired_count'].append(len(trace.get('fired_rules', [])))
                features['inference_steps'].append(len(trace.get('inferences', [])))
        
        return features
    
    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a new rule to the rule base."""
        self.rule_base.append(rule)
        logger.info(f"Added rule: {rule}")
    
    def remove_rule(self, rule_condition: str) -> bool:
        """Remove a rule by its condition."""
        for i, rule in enumerate(self.rule_base):
            if rule.condition == rule_condition:
                removed_rule = self.rule_base.pop(i)
                logger.info(f"Removed rule: {removed_rule}")
                return True
        return False
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about the rule base."""
        return {
            'total_rules': len(self.rule_base),
            'avg_confidence': np.mean([rule.confidence for rule in self.rule_base]),
            'priority_distribution': {
                priority: len([r for r in self.rule_base if r.priority == priority])
                for priority in set(rule.priority for rule in self.rule_base)
            }
        }
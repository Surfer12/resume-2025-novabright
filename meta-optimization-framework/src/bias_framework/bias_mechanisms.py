"""
Bias Mechanisms Implementation

Implements core cognitive bias models including confirmation bias,
anchoring bias, availability heuristic, and other systematic
deviations from rational decision-making.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Types of cognitive biases."""
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    REPRESENTATIVENESS = "representativeness"
    OVERCONFIDENCE = "overconfidence"
    LOSS_AVERSION = "loss_aversion"
    FRAMING = "framing"
    RECENCY = "recency"
    PRIMACY = "primacy"
    HINDSIGHT = "hindsight"


@dataclass
class BiasParameters:
    """Parameters for bias models."""
    strength: float = 0.5  # Bias strength [0, 1]
    persistence: float = 0.8  # How long bias persists
    activation_threshold: float = 0.3  # Threshold for bias activation
    decay_rate: float = 0.1  # Rate of bias decay over time
    context_sensitivity: float = 0.5  # Sensitivity to context


class BiasModel:
    """
    Base class for cognitive bias models.
    
    Implements the core structure for modeling systematic
    deviations from rational decision-making.
    """
    
    def __init__(self, 
                 bias_type: BiasType,
                 parameters: BiasParameters = None):
        """
        Initialize bias model.
        
        Args:
            bias_type: Type of cognitive bias
            parameters: Bias parameters
        """
        self.bias_type = bias_type
        self.parameters = parameters or BiasParameters()
        self.activation_level = 0.0
        self.bias_history = []
        self.context_memory = []
        
    def apply_bias(self, 
                   input_data: torch.Tensor,
                   context: Dict[str, Any] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply bias to input data.
        
        Args:
            input_data: Input tensor to bias
            context: Contextual information
            
        Returns:
            Tuple of (biased_output, bias_info)
        """
        raise NotImplementedError("Subclasses must implement apply_bias")
    
    def update_activation(self, trigger_strength: float) -> None:
        """Update bias activation level."""
        # Increase activation based on trigger strength
        activation_increase = trigger_strength * self.parameters.strength
        self.activation_level = min(1.0, self.activation_level + activation_increase)
        
        # Apply decay
        self.activation_level *= (1 - self.parameters.decay_rate)
        
        # Record in history
        self.bias_history.append(self.activation_level)
        
        # Trim history
        if len(self.bias_history) > 100:
            self.bias_history.pop(0)
    
    def is_active(self) -> bool:
        """Check if bias is currently active."""
        return self.activation_level > self.parameters.activation_threshold
    
    def get_bias_strength(self) -> float:
        """Get current effective bias strength."""
        if self.is_active():
            return self.activation_level * self.parameters.strength
        return 0.0


class ConfirmationBias(BiasModel):
    """
    Confirmation bias model.
    
    Models the tendency to search for, interpret, and recall
    information that confirms pre-existing beliefs.
    """
    
    def __init__(self, parameters: BiasParameters = None):
        """Initialize confirmation bias model."""
        super().__init__(BiasType.CONFIRMATION, parameters)
        self.prior_beliefs = {}
        self.belief_strength = {}
        
    def apply_bias(self, 
                   input_data: torch.Tensor,
                   context: Dict[str, Any] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply confirmation bias to input data."""
        context = context or {}
        
        # Extract belief-relevant features
        belief_features = self._extract_belief_features(input_data, context)
        
        # Check for belief confirmation/disconfirmation
        confirmation_signal = self._compute_confirmation_signal(belief_features, context)
        
        # Update activation based on confirmation
        self.update_activation(abs(confirmation_signal))
        
        if not self.is_active():
            return input_data, {'bias_applied': False, 'confirmation_signal': confirmation_signal}
        
        # Apply bias: amplify confirming evidence, diminish disconfirming evidence
        bias_strength = self.get_bias_strength()
        
        if confirmation_signal > 0:  # Confirming evidence
            bias_factor = 1.0 + bias_strength * confirmation_signal
        else:  # Disconfirming evidence
            bias_factor = 1.0 + bias_strength * confirmation_signal * 0.5  # Weaker effect
        
        biased_output = input_data * bias_factor
        
        # Update beliefs
        self._update_beliefs(belief_features, confirmation_signal)
        
        bias_info = {
            'bias_applied': True,
            'bias_strength': bias_strength,
            'confirmation_signal': confirmation_signal,
            'bias_factor': bias_factor,
            'belief_state': self.prior_beliefs.copy()
        }
        
        return biased_output, bias_info
    
    def _extract_belief_features(self, 
                                input_data: torch.Tensor,
                                context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features relevant to beliefs."""
        features = {}
        
        # Simple feature extraction based on input statistics
        features['mean_activation'] = torch.mean(input_data).item()
        features['max_activation'] = torch.max(input_data).item()
        features['activation_variance'] = torch.var(input_data).item()
        
        # Context-based features
        if 'task_type' in context:
            features['task_type'] = context['task_type']
        
        if 'previous_outcome' in context:
            features['previous_outcome'] = context['previous_outcome']
        
        return features
    
    def _compute_confirmation_signal(self, 
                                   belief_features: Dict[str, float],
                                   context: Dict[str, Any]) -> float:
        """Compute how much current input confirms existing beliefs."""
        if not self.prior_beliefs:
            return 0.0
        
        confirmation = 0.0
        
        for feature_name, feature_value in belief_features.items():
            if feature_name in self.prior_beliefs:
                prior_value = self.prior_beliefs[feature_name]
                belief_strength = self.belief_strength.get(feature_name, 0.5)
                
                # Compute similarity to prior belief
                if isinstance(feature_value, (int, float)) and isinstance(prior_value, (int, float)):
                    similarity = 1.0 - abs(feature_value - prior_value) / max(abs(prior_value), 1.0)
                    confirmation += similarity * belief_strength
                elif feature_value == prior_value:
                    confirmation += belief_strength
        
        # Normalize by number of beliefs
        if self.prior_beliefs:
            confirmation /= len(self.prior_beliefs)
        
        return confirmation
    
    def _update_beliefs(self, 
                       belief_features: Dict[str, float],
                       confirmation_signal: float) -> None:
        """Update prior beliefs based on new evidence."""
        learning_rate = 0.1 * (1 - self.parameters.persistence)
        
        for feature_name, feature_value in belief_features.items():
            if feature_name in self.prior_beliefs:
                # Update existing belief
                prior_value = self.prior_beliefs[feature_name]
                
                if confirmation_signal > 0:
                    # Confirming evidence: strengthen belief
                    update_rate = learning_rate * confirmation_signal
                else:
                    # Disconfirming evidence: weaker update
                    update_rate = learning_rate * abs(confirmation_signal) * 0.3
                
                self.prior_beliefs[feature_name] = (
                    (1 - update_rate) * prior_value + update_rate * feature_value
                )
                
                # Update belief strength
                if confirmation_signal > 0:
                    self.belief_strength[feature_name] = min(1.0, 
                        self.belief_strength.get(feature_name, 0.5) + 0.1)
                else:
                    self.belief_strength[feature_name] = max(0.1,
                        self.belief_strength.get(feature_name, 0.5) - 0.05)
            else:
                # New belief
                self.prior_beliefs[feature_name] = feature_value
                self.belief_strength[feature_name] = 0.5


class AnchoringBias(BiasModel):
    """
    Anchoring bias model.
    
    Models the tendency to rely heavily on the first piece
    of information encountered (the "anchor").
    """
    
    def __init__(self, parameters: BiasParameters = None):
        """Initialize anchoring bias model."""
        super().__init__(BiasType.ANCHORING, parameters)
        self.anchors = {}
        self.anchor_weights = {}
        
    def apply_bias(self, 
                   input_data: torch.Tensor,
                   context: Dict[str, Any] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply anchoring bias to input data."""
        context = context or {}
        
        # Determine context key for anchoring
        context_key = context.get('context_key', 'default')
        
        # Check if we have an anchor for this context
        if context_key not in self.anchors:
            # Set new anchor
            self.anchors[context_key] = torch.mean(input_data).item()
            self.anchor_weights[context_key] = 1.0
            self.update_activation(1.0)  # Strong activation when setting anchor
            
            return input_data, {
                'bias_applied': False,
                'anchor_set': True,
                'anchor_value': self.anchors[context_key]
            }
        
        # Apply anchoring bias
        anchor_value = self.anchors[context_key]
        anchor_weight = self.anchor_weights[context_key]
        
        # Compute distance from anchor
        current_value = torch.mean(input_data).item()
        distance_from_anchor = abs(current_value - anchor_value)
        
        # Update activation based on distance (closer = more bias)
        activation_trigger = 1.0 / (1.0 + distance_from_anchor)
        self.update_activation(activation_trigger)
        
        if not self.is_active():
            return input_data, {
                'bias_applied': False,
                'anchor_value': anchor_value,
                'distance_from_anchor': distance_from_anchor
            }
        
        # Apply bias: pull values toward anchor
        bias_strength = self.get_bias_strength()
        pull_factor = bias_strength * anchor_weight
        
        # Create bias tensor that pulls toward anchor
        bias_direction = torch.sign(anchor_value - input_data)
        bias_magnitude = torch.abs(anchor_value - input_data) * pull_factor
        bias_adjustment = bias_direction * bias_magnitude
        
        biased_output = input_data + bias_adjustment
        
        # Decay anchor weight over time
        self.anchor_weights[context_key] *= self.parameters.persistence
        
        bias_info = {
            'bias_applied': True,
            'bias_strength': bias_strength,
            'anchor_value': anchor_value,
            'anchor_weight': anchor_weight,
            'distance_from_anchor': distance_from_anchor,
            'bias_adjustment': torch.mean(torch.abs(bias_adjustment)).item()
        }
        
        return biased_output, bias_info


class AvailabilityBias(BiasModel):
    """
    Availability heuristic bias model.
    
    Models the tendency to overestimate the likelihood of events
    that are more easily recalled (more "available" in memory).
    """
    
    def __init__(self, parameters: BiasParameters = None):
        """Initialize availability bias model."""
        super().__init__(BiasType.AVAILABILITY, parameters)
        self.memory_traces = []
        self.recency_weights = []
        self.frequency_counts = {}
        
    def apply_bias(self, 
                   input_data: torch.Tensor,
                   context: Dict[str, Any] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply availability bias to input data."""
        context = context or {}
        
        # Extract memorable features
        memorable_features = self._extract_memorable_features(input_data, context)
        
        # Compute availability of similar patterns
        availability_score = self._compute_availability(memorable_features)
        
        # Update activation based on availability
        self.update_activation(availability_score)
        
        # Store current pattern in memory
        self._store_memory_trace(memorable_features)
        
        if not self.is_active():
            return input_data, {
                'bias_applied': False,
                'availability_score': availability_score
            }
        
        # Apply bias: amplify patterns that are more available in memory
        bias_strength = self.get_bias_strength()
        availability_amplification = 1.0 + bias_strength * availability_score
        
        biased_output = input_data * availability_amplification
        
        bias_info = {
            'bias_applied': True,
            'bias_strength': bias_strength,
            'availability_score': availability_score,
            'amplification_factor': availability_amplification,
            'memory_size': len(self.memory_traces)
        }
        
        return biased_output, bias_info
    
    def _extract_memorable_features(self, 
                                   input_data: torch.Tensor,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features that are likely to be memorable."""
        features = {}
        
        # Statistical features
        features['magnitude'] = torch.norm(input_data).item()
        features['peak_value'] = torch.max(torch.abs(input_data)).item()
        features['pattern_type'] = self._classify_pattern(input_data)
        
        # Context features
        if 'event_type' in context:
            features['event_type'] = context['event_type']
        if 'emotional_valence' in context:
            features['emotional_valence'] = context['emotional_valence']
        
        return features
    
    def _classify_pattern(self, input_data: torch.Tensor) -> str:
        """Classify input pattern for memory storage."""
        mean_val = torch.mean(input_data).item()
        std_val = torch.std(input_data).item()
        
        if std_val < 0.1:
            return "uniform"
        elif mean_val > 0.5:
            return "high_activation"
        elif mean_val < -0.5:
            return "low_activation"
        else:
            return "mixed"
    
    def _compute_availability(self, memorable_features: Dict[str, Any]) -> float:
        """Compute availability score based on memory traces."""
        if not self.memory_traces:
            return 0.0
        
        availability = 0.0
        total_weight = 0.0
        
        for i, (trace, weight) in enumerate(zip(self.memory_traces, self.recency_weights)):
            similarity = self._compute_similarity(memorable_features, trace)
            availability += similarity * weight
            total_weight += weight
        
        if total_weight > 0:
            availability /= total_weight
        
        return availability
    
    def _compute_similarity(self, 
                           features1: Dict[str, Any],
                           features2: Dict[str, Any]) -> float:
        """Compute similarity between feature sets."""
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        similarity = 0.0
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1.0)
                sim = 1.0 - abs(val1 - val2) / max_val
            elif val1 == val2:
                # Categorical similarity
                sim = 1.0
            else:
                sim = 0.0
            
            similarity += sim
        
        return similarity / len(common_keys)
    
    def _store_memory_trace(self, memorable_features: Dict[str, Any]) -> None:
        """Store memory trace with recency weighting."""
        self.memory_traces.append(memorable_features)
        self.recency_weights.append(1.0)  # New memories have full weight
        
        # Apply recency decay to existing memories
        for i in range(len(self.recency_weights) - 1):
            self.recency_weights[i] *= 0.95  # Decay factor
        
        # Limit memory size
        max_memory_size = 50
        if len(self.memory_traces) > max_memory_size:
            # Remove oldest, least weighted memories
            min_weight_idx = np.argmin(self.recency_weights)
            self.memory_traces.pop(min_weight_idx)
            self.recency_weights.pop(min_weight_idx)


class OverconfidenceBias(BiasModel):
    """
    Overconfidence bias model.
    
    Models the tendency to overestimate one's own abilities,
    knowledge, or chances of success.
    """
    
    def __init__(self, parameters: BiasParameters = None):
        """Initialize overconfidence bias model."""
        super().__init__(BiasType.OVERCONFIDENCE, parameters)
        self.confidence_history = []
        self.accuracy_history = []
        self.calibration_error = 0.0
        
    def apply_bias(self, 
                   input_data: torch.Tensor,
                   context: Dict[str, Any] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply overconfidence bias to input data."""
        context = context or {}
        
        # Extract confidence-related features
        confidence_estimate = self._extract_confidence(input_data, context)
        
        # Update calibration based on past performance
        if 'actual_accuracy' in context:
            self._update_calibration(confidence_estimate, context['actual_accuracy'])
        
        # Compute overconfidence trigger
        overconfidence_trigger = max(0, self.calibration_error)
        self.update_activation(overconfidence_trigger)
        
        if not self.is_active():
            return input_data, {
                'bias_applied': False,
                'confidence_estimate': confidence_estimate,
                'calibration_error': self.calibration_error
            }
        
        # Apply bias: inflate confidence/certainty
        bias_strength = self.get_bias_strength()
        confidence_inflation = 1.0 + bias_strength * overconfidence_trigger
        
        # Amplify high-confidence regions more than low-confidence regions
        confidence_mask = torch.abs(input_data) > torch.median(torch.abs(input_data))
        biased_output = input_data.clone()
        biased_output[confidence_mask] *= confidence_inflation
        
        bias_info = {
            'bias_applied': True,
            'bias_strength': bias_strength,
            'confidence_estimate': confidence_estimate,
            'calibration_error': self.calibration_error,
            'confidence_inflation': confidence_inflation
        }
        
        return biased_output, bias_info
    
    def _extract_confidence(self, 
                           input_data: torch.Tensor,
                           context: Dict[str, Any]) -> float:
        """Extract confidence estimate from input data."""
        # Use variance as inverse confidence measure
        variance = torch.var(input_data).item()
        confidence = 1.0 / (1.0 + variance)
        
        # Adjust based on context
        if 'task_difficulty' in context:
            difficulty = context['task_difficulty']
            confidence *= (1.0 - difficulty * 0.5)  # Harder tasks reduce confidence
        
        return confidence
    
    def _update_calibration(self, confidence: float, actual_accuracy: float) -> None:
        """Update calibration based on confidence vs actual performance."""
        self.confidence_history.append(confidence)
        self.accuracy_history.append(actual_accuracy)
        
        # Limit history size
        if len(self.confidence_history) > 20:
            self.confidence_history.pop(0)
            self.accuracy_history.pop(0)
        
        # Compute calibration error (overconfidence = positive error)
        if len(self.confidence_history) >= 5:
            avg_confidence = np.mean(self.confidence_history[-10:])
            avg_accuracy = np.mean(self.accuracy_history[-10:])
            self.calibration_error = avg_confidence - avg_accuracy


class CognitiveBiasFramework:
    """
    Main cognitive bias framework.
    
    Integrates multiple bias models and provides unified
    bias application and analysis capabilities.
    """
    
    def __init__(self, bias_types: List[BiasType] = None):
        """
        Initialize cognitive bias framework.
        
        Args:
            bias_types: List of bias types to include
        """
        if bias_types is None:
            bias_types = [BiasType.CONFIRMATION, BiasType.ANCHORING, 
                         BiasType.AVAILABILITY, BiasType.OVERCONFIDENCE]
        
        self.bias_models = {}
        
        # Initialize bias models
        for bias_type in bias_types:
            if bias_type == BiasType.CONFIRMATION:
                self.bias_models[bias_type] = ConfirmationBias()
            elif bias_type == BiasType.ANCHORING:
                self.bias_models[bias_type] = AnchoringBias()
            elif bias_type == BiasType.AVAILABILITY:
                self.bias_models[bias_type] = AvailabilityBias()
            elif bias_type == BiasType.OVERCONFIDENCE:
                self.bias_models[bias_type] = OverconfidenceBias()
            else:
                # Generic bias model for other types
                self.bias_models[bias_type] = BiasModel(bias_type)
        
        self.bias_interaction_matrix = self._initialize_interaction_matrix()
        self.global_bias_state = {}
        
        logger.info(f"Initialized CognitiveBiasFramework with {len(self.bias_models)} bias types")
    
    def apply_biases(self, 
                    input_data: torch.Tensor,
                    context: Dict[str, Any] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply all active biases to input data.
        
        Args:
            input_data: Input tensor
            context: Contextual information
            
        Returns:
            Tuple of (biased_output, bias_report)
        """
        context = context or {}
        
        current_data = input_data.clone()
        bias_report = {
            'applied_biases': [],
            'bias_strengths': {},
            'bias_interactions': {},
            'total_bias_effect': 0.0
        }
        
        # Apply each bias model
        for bias_type, bias_model in self.bias_models.items():
            try:
                biased_data, bias_info = bias_model.apply_bias(current_data, context)
                
                if bias_info.get('bias_applied', False):
                    bias_report['applied_biases'].append(bias_type.value)
                    bias_report['bias_strengths'][bias_type.value] = bias_info.get('bias_strength', 0.0)
                    
                    # Update current data for next bias
                    current_data = biased_data
                
                # Store detailed bias info
                bias_report[f'{bias_type.value}_info'] = bias_info
                
            except Exception as e:
                logger.warning(f"Error applying {bias_type.value} bias: {e}")
        
        # Compute total bias effect
        bias_effect = torch.norm(current_data - input_data).item()
        bias_report['total_bias_effect'] = bias_effect
        
        # Analyze bias interactions
        bias_report['bias_interactions'] = self._analyze_bias_interactions(bias_report)
        
        return current_data, bias_report
    
    def _initialize_interaction_matrix(self) -> Dict[Tuple[BiasType, BiasType], float]:
        """Initialize bias interaction matrix."""
        interactions = {}
        
        bias_types = list(self.bias_models.keys())
        
        for i, bias1 in enumerate(bias_types):
            for j, bias2 in enumerate(bias_types):
                if i != j:
                    # Define interaction strengths (simplified)
                    if (bias1 == BiasType.CONFIRMATION and bias2 == BiasType.AVAILABILITY) or \
                       (bias1 == BiasType.AVAILABILITY and bias2 == BiasType.CONFIRMATION):
                        interactions[(bias1, bias2)] = 0.7  # Strong positive interaction
                    elif (bias1 == BiasType.ANCHORING and bias2 == BiasType.OVERCONFIDENCE) or \
                         (bias1 == BiasType.OVERCONFIDENCE and bias2 == BiasType.ANCHORING):
                        interactions[(bias1, bias2)] = 0.5  # Moderate positive interaction
                    else:
                        interactions[(bias1, bias2)] = 0.1  # Weak interaction
        
        return interactions
    
    def _analyze_bias_interactions(self, bias_report: Dict[str, Any]) -> Dict[str, float]:
        """Analyze interactions between active biases."""
        interactions = {}
        applied_biases = bias_report['applied_biases']
        
        for i, bias1 in enumerate(applied_biases):
            for j, bias2 in enumerate(applied_biases):
                if i < j:  # Avoid duplicates
                    bias1_type = BiasType(bias1)
                    bias2_type = BiasType(bias2)
                    
                    interaction_key = (bias1_type, bias2_type)
                    if interaction_key in self.bias_interaction_matrix:
                        strength1 = bias_report['bias_strengths'].get(bias1, 0.0)
                        strength2 = bias_report['bias_strengths'].get(bias2, 0.0)
                        
                        interaction_strength = (
                            self.bias_interaction_matrix[interaction_key] * 
                            strength1 * strength2
                        )
                        
                        interactions[f"{bias1}_{bias2}"] = interaction_strength
        
        return interactions
    
    def get_bias_summary(self) -> Dict[str, Any]:
        """Get summary of bias framework state."""
        summary = {
            'num_bias_models': len(self.bias_models),
            'bias_types': [bt.value for bt in self.bias_models.keys()],
            'active_biases': [],
            'bias_activations': {},
            'bias_histories': {}
        }
        
        for bias_type, bias_model in self.bias_models.items():
            if bias_model.is_active():
                summary['active_biases'].append(bias_type.value)
            
            summary['bias_activations'][bias_type.value] = bias_model.activation_level
            summary['bias_histories'][bias_type.value] = bias_model.bias_history[-10:]  # Last 10
        
        return summary
    
    def reset_biases(self) -> None:
        """Reset all bias models to initial state."""
        for bias_model in self.bias_models.values():
            bias_model.activation_level = 0.0
            bias_model.bias_history = []
            bias_model.context_memory = []
            
            # Reset model-specific state
            if hasattr(bias_model, 'prior_beliefs'):
                bias_model.prior_beliefs = {}
            if hasattr(bias_model, 'anchors'):
                bias_model.anchors = {}
            if hasattr(bias_model, 'memory_traces'):
                bias_model.memory_traces = []
        
        logger.info("Reset all bias models")
    
    def configure_bias(self, 
                      bias_type: BiasType,
                      parameters: BiasParameters) -> None:
        """Configure parameters for specific bias type."""
        if bias_type in self.bias_models:
            self.bias_models[bias_type].parameters = parameters
            logger.info(f"Updated parameters for {bias_type.value} bias")
        else:
            logger.warning(f"Bias type {bias_type.value} not found in framework")
"""
Adaptive Weighting Implementation

Implements adaptive α negotiation routines for balancing symbolic and neural
components in the neuro-symbolic framework. Provides dynamic weighting based
on task demands, performance feedback, and cognitive authenticity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WeightingConfig:
    """Configuration for adaptive weighting."""
    initial_alpha: float = 0.5
    learning_rate: float = 0.01
    adaptation_window: int = 10
    min_alpha: float = 0.1
    max_alpha: float = 0.9
    stability_threshold: float = 0.05
    performance_weight: float = 0.4
    authenticity_weight: float = 0.3
    efficiency_weight: float = 0.3


@dataclass
class WeightingState:
    """Current state of the weighting system."""
    alpha: float
    performance_history: List[float]
    authenticity_history: List[float]
    efficiency_history: List[float]
    adaptation_count: int = 0
    last_update_step: int = 0


class PerformanceEstimator(nn.Module):
    """
    Estimates performance based on symbolic and neural outputs.
    
    Uses a small neural network to predict task performance from
    the outputs of symbolic and neural components.
    """
    
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 64):
        """
        Initialize performance estimator.
        
        Args:
            input_dim: Combined input dimension (symbolic + neural outputs)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                symbolic_output: torch.Tensor,
                neural_output: torch.Tensor) -> torch.Tensor:
        """
        Estimate performance from component outputs.
        
        Args:
            symbolic_output: Output from symbolic component
            neural_output: Output from neural component
            
        Returns:
            Performance estimate [0, 1]
        """
        # Combine outputs
        combined = torch.cat([symbolic_output, neural_output], dim=-1)
        
        # Estimate performance
        performance = self.estimator(combined)
        
        return performance


class AuthenticityEstimator(nn.Module):
    """
    Estimates cognitive authenticity of the combined system.
    
    Evaluates how well the system behavior matches human cognitive patterns.
    """
    
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 64):
        """
        Initialize authenticity estimator.
        
        Args:
            input_dim: Combined input dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.authenticity_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh for smoother gradients
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Cognitive pattern detectors
        self.pattern_detectors = nn.ModuleDict({
            'working_memory': nn.Linear(input_dim, 1),
            'attention_control': nn.Linear(input_dim, 1),
            'response_inhibition': nn.Linear(input_dim, 1),
            'cognitive_flexibility': nn.Linear(input_dim, 1)
        })
        
    def forward(self, 
                symbolic_output: torch.Tensor,
                neural_output: torch.Tensor,
                alpha: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Estimate cognitive authenticity.
        
        Args:
            symbolic_output: Output from symbolic component
            neural_output: Output from neural component
            alpha: Current weighting parameter
            
        Returns:
            Tuple of (authenticity_score, pattern_scores)
        """
        # Combine outputs with current weighting
        combined = torch.cat([
            alpha * symbolic_output,
            (1 - alpha) * neural_output
        ], dim=-1)
        
        # Overall authenticity
        authenticity = self.authenticity_net(combined)
        
        # Individual cognitive pattern scores
        pattern_scores = {}
        for pattern_name, detector in self.pattern_detectors.items():
            pattern_scores[pattern_name] = torch.sigmoid(detector(combined))
        
        return authenticity, pattern_scores


class EfficiencyEstimator:
    """
    Estimates computational efficiency of the system.
    
    Tracks FLOPs, memory usage, and processing time for efficiency assessment.
    """
    
    def __init__(self):
        """Initialize efficiency estimator."""
        self.flop_counter = 0
        self.memory_usage = 0
        self.processing_times = []
        
    def estimate_efficiency(self, 
                           symbolic_output: torch.Tensor,
                           neural_output: torch.Tensor,
                           symbolic_metadata: Dict[str, Any],
                           neural_metadata: Dict[str, Any]) -> float:
        """
        Estimate computational efficiency.
        
        Args:
            symbolic_output: Output from symbolic component
            neural_output: Output from neural component
            symbolic_metadata: Metadata from symbolic processing
            neural_metadata: Metadata from neural processing
            
        Returns:
            Efficiency score [0, 1]
        """
        # Estimate FLOPs
        symbolic_flops = self._estimate_symbolic_flops(symbolic_metadata)
        neural_flops = self._estimate_neural_flops(neural_metadata)
        total_flops = symbolic_flops + neural_flops
        
        # Estimate memory usage
        symbolic_memory = self._estimate_symbolic_memory(symbolic_output, symbolic_metadata)
        neural_memory = self._estimate_neural_memory(neural_output, neural_metadata)
        total_memory = symbolic_memory + neural_memory
        
        # Normalize efficiency metrics
        flop_efficiency = 1.0 / (1.0 + total_flops / 1e6)  # Normalize by 1M FLOPs
        memory_efficiency = 1.0 / (1.0 + total_memory / 1e6)  # Normalize by 1MB
        
        # Combined efficiency score
        efficiency = 0.6 * flop_efficiency + 0.4 * memory_efficiency
        
        return float(efficiency)
    
    def _estimate_symbolic_flops(self, metadata: Dict[str, Any]) -> float:
        """Estimate FLOPs for symbolic processing."""
        rules_fired = metadata.get('rules_fired', 0)
        traces = metadata.get('traces', [])
        if traces and isinstance(traces[-1], dict):
            inference_steps = len(traces[-1].get('inferences', []))
        else:
            inference_steps = 0
        
        # Rough estimate: each rule evaluation and inference step
        flops = rules_fired * 100 + inference_steps * 50
        return float(flops)
    
    def _estimate_neural_flops(self, metadata: Dict[str, Any]) -> float:
        """Estimate FLOPs for neural processing."""
        # Rough estimate based on layer outputs and attention operations
        layer_outputs = metadata.get('layer_outputs', [])
        attention_ops = 1 if 'attention_info' in metadata else 0
        memory_ops = 1 if 'memory_info' in metadata else 0
        
        # Estimate based on tensor operations
        flops = len(layer_outputs) * 1000 + attention_ops * 5000 + memory_ops * 3000
        return float(flops)
    
    def _estimate_symbolic_memory(self, output: torch.Tensor, metadata: Dict[str, Any]) -> float:
        """Estimate memory usage for symbolic processing."""
        # Memory for rules, facts, and inference traces
        traces = metadata.get('traces', [])
        if traces and isinstance(traces[-1], dict):
            num_rules = len(traces[-1].get('fired_rules', []))
        else:
            num_rules = 0
        memory = output.numel() * 4 + num_rules * 100  # 4 bytes per float + rule overhead
        return float(memory)
    
    def _estimate_neural_memory(self, output: torch.Tensor, metadata: Dict[str, Any]) -> float:
        """Estimate memory usage for neural processing."""
        # Memory for activations and parameters
        activation_memory = output.numel() * 4  # 4 bytes per float
        
        # Additional memory for attention and working memory
        if 'attention_info' in metadata:
            activation_memory += output.numel() * 2  # Attention weights
        if 'memory_info' in metadata:
            activation_memory += output.numel() * 4  # Working memory states
        
        return float(activation_memory)


class AdaptiveWeighting(nn.Module):
    """
    Adaptive weighting system for neuro-symbolic integration.
    
    Dynamically adjusts the balance parameter α between symbolic and neural
    components based on performance, cognitive authenticity, and efficiency.
    """
    
    def __init__(self, 
                 config: WeightingConfig,
                 symbolic_dim: int = 64,
                 neural_dim: int = 64):
        """
        Initialize adaptive weighting system.
        
        Args:
            config: Weighting configuration
            symbolic_dim: Dimension of symbolic component output
            neural_dim: Dimension of neural component output
        """
        super().__init__()
        
        self.config = config
        self.symbolic_dim = symbolic_dim
        self.neural_dim = neural_dim
        self.combined_dim = symbolic_dim + neural_dim
        
        # Initialize weighting state
        self.state = WeightingState(
            alpha=config.initial_alpha,
            performance_history=[],
            authenticity_history=[],
            efficiency_history=[]
        )
        
        # Estimators
        self.performance_estimator = PerformanceEstimator(
            input_dim=self.combined_dim,
            hidden_dim=64
        )
        
        self.authenticity_estimator = AuthenticityEstimator(
            input_dim=self.combined_dim,
            hidden_dim=64
        )
        
        self.efficiency_estimator = EfficiencyEstimator()
        
        # Alpha adaptation network
        self.alpha_adapter = nn.Sequential(
            nn.Linear(3, 16),  # 3 inputs: performance, authenticity, efficiency
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()  # Output in [-1, 1] for alpha adjustment
        )
        
        # Stability tracker
        self.stability_window = []
        
        logger.info(f"Initialized AdaptiveWeighting with α={config.initial_alpha}")
    
    def forward(self, 
                symbolic_output: torch.Tensor,
                neural_output: torch.Tensor,
                symbolic_metadata: Dict[str, Any],
                neural_metadata: Dict[str, Any],
                target_performance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute adaptive weighting and combined output.
        
        Args:
            symbolic_output: Output from symbolic component
            neural_output: Output from neural component
            symbolic_metadata: Metadata from symbolic processing
            neural_metadata: Metadata from neural processing
            target_performance: Optional target performance for training
            
        Returns:
            Tuple of (combined_output, weighting_info)
        """
        batch_size = symbolic_output.shape[0]
        
        # Current alpha value
        current_alpha = self.state.alpha
        
        # Estimate performance
        performance_score = self.performance_estimator(symbolic_output, neural_output)
        avg_performance = torch.mean(performance_score).item()
        
        # Estimate authenticity
        authenticity_score, pattern_scores = self.authenticity_estimator(
            symbolic_output, neural_output, current_alpha
        )
        avg_authenticity = torch.mean(authenticity_score).item()
        
        # Estimate efficiency
        efficiency_score = self.efficiency_estimator.estimate_efficiency(
            symbolic_output, neural_output, symbolic_metadata, neural_metadata
        )
        
        # Update histories
        self.state.performance_history.append(avg_performance)
        self.state.authenticity_history.append(avg_authenticity)
        self.state.efficiency_history.append(efficiency_score)
        
        # Trim histories to adaptation window
        if len(self.state.performance_history) > self.config.adaptation_window:
            self.state.performance_history.pop(0)
            self.state.authenticity_history.pop(0)
            self.state.efficiency_history.pop(0)
        
        # Adapt alpha if enough history
        if len(self.state.performance_history) >= self.config.adaptation_window:
            new_alpha = self._adapt_alpha(avg_performance, avg_authenticity, efficiency_score)
            self.state.alpha = new_alpha
            self.state.adaptation_count += 1
        
        # Combine outputs with current alpha
        combined_output = (current_alpha * symbolic_output + 
                          (1 - current_alpha) * neural_output)
        
        # Compile weighting information
        weighting_info = {
            'alpha': current_alpha,
            'performance_score': avg_performance,
            'authenticity_score': avg_authenticity,
            'efficiency_score': efficiency_score,
            'pattern_scores': {k: torch.mean(v).item() for k, v in pattern_scores.items()},
            'adaptation_count': self.state.adaptation_count,
            'stability': self._compute_stability(),
            'component_contributions': {
                'symbolic': current_alpha,
                'neural': 1 - current_alpha
            }
        }
        
        # Training mode: compute loss for alpha adaptation
        if self.training and target_performance is not None:
            adaptation_loss = self._compute_adaptation_loss(
                performance_score, authenticity_score, efficiency_score, target_performance
            )
            weighting_info['adaptation_loss'] = adaptation_loss
        
        return combined_output, weighting_info
    
    def _adapt_alpha(self, 
                    performance: float,
                    authenticity: float,
                    efficiency: float) -> float:
        """
        Adapt alpha based on performance metrics.
        
        Args:
            performance: Current performance score
            authenticity: Current authenticity score
            efficiency: Current efficiency score
            
        Returns:
            New alpha value
        """
        # Compute recent trends
        perf_trend = self._compute_trend(self.state.performance_history)
        auth_trend = self._compute_trend(self.state.authenticity_history)
        eff_trend = self._compute_trend(self.state.efficiency_history)
        
        # Weighted combination of metrics
        combined_score = (self.config.performance_weight * performance +
                         self.config.authenticity_weight * authenticity +
                         self.config.efficiency_weight * efficiency)
        
        # Use neural network to suggest alpha adjustment
        metrics_tensor = torch.tensor([performance, authenticity, efficiency], dtype=torch.float32)
        alpha_adjustment = self.alpha_adapter(metrics_tensor).item()
        
        # Apply adjustment with learning rate
        alpha_delta = self.config.learning_rate * alpha_adjustment
        
        # Consider trends for adjustment direction
        if perf_trend < 0 or auth_trend < 0:  # Performance or authenticity declining
            # Adjust alpha in direction that might help
            if performance < authenticity:
                alpha_delta += 0.01  # Increase symbolic contribution
            else:
                alpha_delta -= 0.01  # Increase neural contribution
        
        # Update alpha
        new_alpha = self.state.alpha + alpha_delta
        
        # Clamp to valid range
        new_alpha = max(self.config.min_alpha, min(self.config.max_alpha, new_alpha))
        
        # Track stability
        self.stability_window.append(abs(alpha_delta))
        if len(self.stability_window) > self.config.adaptation_window:
            self.stability_window.pop(0)
        
        return new_alpha
    
    def _compute_trend(self, history: List[float], window: int = 5) -> float:
        """Compute trend in recent history."""
        if len(history) < window:
            return 0.0
        
        recent = history[-window:]
        if len(recent) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(recent))
        y = np.array(recent)
        trend = np.polyfit(x, y, 1)[0]  # Slope of linear fit
        
        return float(trend)
    
    def _compute_stability(self) -> float:
        """Compute stability metric based on recent alpha changes."""
        if len(self.stability_window) < 2:
            return 1.0
        
        avg_change = np.mean(self.stability_window)
        stability = max(0.0, 1.0 - avg_change / self.config.stability_threshold)
        
        return float(stability)
    
    def _compute_adaptation_loss(self, 
                                performance_score: torch.Tensor,
                                authenticity_score: torch.Tensor,
                                efficiency_score: float,
                                target_performance: torch.Tensor) -> torch.Tensor:
        """Compute loss for training the adaptation mechanism."""
        
        # Performance loss
        perf_loss = F.mse_loss(performance_score.squeeze(), target_performance.squeeze())
        
        # Authenticity regularization (encourage high authenticity)
        auth_loss = -torch.mean(torch.log(authenticity_score + 1e-8))
        
        # Efficiency regularization
        eff_tensor = torch.tensor(efficiency_score, dtype=torch.float32)
        eff_loss = -torch.log(eff_tensor + 1e-8)
        
        # Combined loss
        total_loss = (self.config.performance_weight * perf_loss +
                     self.config.authenticity_weight * auth_loss +
                     self.config.efficiency_weight * eff_loss)
        
        return total_loss
    
    def get_weighting_history(self) -> Dict[str, List[float]]:
        """Get complete weighting history."""
        return {
            'performance_history': self.state.performance_history.copy(),
            'authenticity_history': self.state.authenticity_history.copy(),
            'efficiency_history': self.state.efficiency_history.copy(),
            'alpha_history': [self.state.alpha]  # Current alpha
        }
    
    def reset_adaptation(self) -> None:
        """Reset adaptation state."""
        self.state = WeightingState(
            alpha=self.config.initial_alpha,
            performance_history=[],
            authenticity_history=[],
            efficiency_history=[]
        )
        self.stability_window = []
        logger.info("Reset adaptive weighting state")
    
    def set_alpha(self, alpha: float) -> None:
        """Manually set alpha value."""
        alpha = max(self.config.min_alpha, min(self.config.max_alpha, alpha))
        self.state.alpha = alpha
        logger.info(f"Manually set α={alpha}")
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        return {
            'current_alpha': self.state.alpha,
            'adaptation_count': self.state.adaptation_count,
            'stability': self._compute_stability(),
            'performance_trend': self._compute_trend(self.state.performance_history),
            'authenticity_trend': self._compute_trend(self.state.authenticity_history),
            'efficiency_trend': self._compute_trend(self.state.efficiency_history),
            'avg_performance': np.mean(self.state.performance_history) if self.state.performance_history else 0.0,
            'avg_authenticity': np.mean(self.state.authenticity_history) if self.state.authenticity_history else 0.0,
            'avg_efficiency': np.mean(self.state.efficiency_history) if self.state.efficiency_history else 0.0
        }
"""
Architecture Search Implementation

Implements neural architecture search for cognitive models, including
search space definition, optimization strategies, and performance evaluation
for cognitive task architectures.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import logging

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Types of neural network layers."""
    LINEAR = "linear"
    CONV1D = "conv1d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    MEMORY = "memory"
    DROPOUT = "dropout"
    NORM = "norm"


class ActivationType(Enum):
    """Types of activation functions."""
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SWISH = "swish"


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    layer_type: LayerType
    input_dim: int
    output_dim: int
    activation: Optional[ActivationType] = None
    dropout_rate: float = 0.0
    layer_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate layer configuration."""
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("Input and output dimensions must be positive")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("Dropout rate must be between 0 and 1")


@dataclass
class ArchitectureConfig:
    """Configuration for a complete architecture."""
    layers: List[LayerConfig]
    input_dim: int
    output_dim: int
    task_type: str = "general"
    use_residual: bool = True
    use_attention: bool = True
    use_memory: bool = True
    
    def __post_init__(self):
        """Validate architecture configuration."""
        if not self.layers:
            raise ValueError("Architecture must have at least one layer")
        
        # Check dimension compatibility
        if self.layers[0].input_dim != self.input_dim:
            raise ValueError("First layer input dimension must match architecture input dimension")
        if self.layers[-1].output_dim != self.output_dim:
            raise ValueError("Last layer output dimension must match architecture output dimension")


class SearchSpace:
    """
    Defines the search space for neural architecture search.
    
    Specifies the possible layer types, dimensions, and configurations
    that can be explored during architecture search.
    """
    
    def __init__(self, 
                 input_dim: int = 64,
                 output_dim: int = 64,
                 max_layers: int = 10,
                 min_layers: int = 2):
        """
        Initialize search space.
        
        Args:
            input_dim: Input dimension for architectures
            output_dim: Output dimension for architectures
            max_layers: Maximum number of layers
            min_layers: Minimum number of layers
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_layers = max_layers
        self.min_layers = min_layers
        
        # Define possible layer configurations
        self.layer_types = [
            LayerType.LINEAR,
            LayerType.LSTM,
            LayerType.GRU,
            LayerType.ATTENTION,
            LayerType.MEMORY
        ]
        
        self.activations = [
            ActivationType.RELU,
            ActivationType.GELU,
            ActivationType.TANH
        ]
        
        self.hidden_dims = [32, 64, 128, 256, 512]
        self.dropout_rates = [0.0, 0.1, 0.2, 0.3]
        
        # Task-specific constraints
        self.task_constraints = {
            'n_back': {
                'required_layers': [LayerType.MEMORY],
                'preferred_activations': [ActivationType.TANH, ActivationType.SIGMOID]
            },
            'stroop': {
                'required_layers': [LayerType.ATTENTION],
                'preferred_activations': [ActivationType.RELU, ActivationType.GELU]
            },
            'planning': {
                'required_layers': [LayerType.MEMORY, LayerType.ATTENTION],
                'preferred_activations': [ActivationType.GELU, ActivationType.TANH]
            }
        }
        
        logger.info(f"Initialized SearchSpace with {len(self.layer_types)} layer types")
    
    def sample_architecture(self, task_type: str = "general") -> ArchitectureConfig:
        """
        Sample a random architecture from the search space.
        
        Args:
            task_type: Type of cognitive task
            
        Returns:
            Sampled architecture configuration
        """
        # Determine number of layers
        num_layers = random.randint(self.min_layers, self.max_layers)
        
        # Get task constraints
        constraints = self.task_constraints.get(task_type, {})
        required_layers = constraints.get('required_layers', [])
        preferred_activations = constraints.get('preferred_activations', self.activations)
        
        layers = []
        current_dim = self.input_dim
        
        for i in range(num_layers):
            # Choose layer type
            if i < len(required_layers):
                # Use required layers first
                layer_type = required_layers[i]
            else:
                # Sample from available types
                layer_type = random.choice(self.layer_types)
            
            # Choose output dimension
            if i == num_layers - 1:
                # Last layer must match output dimension
                output_dim = self.output_dim
            else:
                output_dim = random.choice(self.hidden_dims)
            
            # Choose activation
            activation = random.choice(preferred_activations)
            
            # Choose dropout rate
            dropout_rate = random.choice(self.dropout_rates)
            
            # Create layer configuration
            layer_config = LayerConfig(
                layer_type=layer_type,
                input_dim=current_dim,
                output_dim=output_dim,
                activation=activation,
                dropout_rate=dropout_rate,
                layer_params=self._sample_layer_params(layer_type)
            )
            
            layers.append(layer_config)
            current_dim = output_dim
        
        # Create architecture configuration
        architecture = ArchitectureConfig(
            layers=layers,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            task_type=task_type,
            use_residual=random.choice([True, False]),
            use_attention=LayerType.ATTENTION in [l.layer_type for l in layers],
            use_memory=LayerType.MEMORY in [l.layer_type for l in layers]
        )
        
        return architecture
    
    def _sample_layer_params(self, layer_type: LayerType) -> Dict[str, Any]:
        """Sample layer-specific parameters."""
        params = {}
        
        if layer_type == LayerType.LSTM:
            params['num_layers'] = random.choice([1, 2, 3])
            params['bidirectional'] = random.choice([True, False])
            
        elif layer_type == LayerType.GRU:
            params['num_layers'] = random.choice([1, 2, 3])
            params['bidirectional'] = random.choice([True, False])
            
        elif layer_type == LayerType.ATTENTION:
            params['num_heads'] = random.choice([1, 2, 4, 8])
            params['dropout'] = random.choice([0.0, 0.1, 0.2])
            
        elif layer_type == LayerType.MEMORY:
            params['memory_size'] = random.choice([16, 32, 64, 128])
            params['num_heads'] = random.choice([1, 2, 4])
            
        elif layer_type == LayerType.CONV1D:
            params['kernel_size'] = random.choice([3, 5, 7])
            params['stride'] = random.choice([1, 2])
            params['padding'] = random.choice([0, 1, 2])
        
        return params
    
    def mutate_architecture(self, 
                           architecture: ArchitectureConfig,
                           mutation_rate: float = 0.1) -> ArchitectureConfig:
        """
        Mutate an existing architecture.
        
        Args:
            architecture: Architecture to mutate
            mutation_rate: Probability of mutating each component
            
        Returns:
            Mutated architecture
        """
        new_layers = []
        
        for layer in architecture.layers:
            if random.random() < mutation_rate:
                # Mutate this layer
                new_layer = self._mutate_layer(layer)
                new_layers.append(new_layer)
            else:
                new_layers.append(layer)
        
        # Possibly add or remove layers
        if random.random() < mutation_rate:
            if len(new_layers) < self.max_layers and random.random() < 0.5:
                # Add a layer
                insert_pos = random.randint(0, len(new_layers))
                new_layer = self._create_random_layer(
                    new_layers[insert_pos-1].output_dim if insert_pos > 0 else self.input_dim,
                    new_layers[insert_pos].input_dim if insert_pos < len(new_layers) else self.output_dim
                )
                new_layers.insert(insert_pos, new_layer)
                
            elif len(new_layers) > self.min_layers and random.random() < 0.5:
                # Remove a layer
                remove_pos = random.randint(0, len(new_layers) - 1)
                new_layers.pop(remove_pos)
        
        # Fix dimension compatibility
        new_layers = self._fix_dimensions(new_layers)
        
        return ArchitectureConfig(
            layers=new_layers,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            task_type=architecture.task_type,
            use_residual=architecture.use_residual,
            use_attention=architecture.use_attention,
            use_memory=architecture.use_memory
        )
    
    def _mutate_layer(self, layer: LayerConfig) -> LayerConfig:
        """Mutate a single layer configuration."""
        # Choose what to mutate
        mutation_type = random.choice(['type', 'activation', 'dropout', 'params'])
        
        new_layer = LayerConfig(
            layer_type=layer.layer_type,
            input_dim=layer.input_dim,
            output_dim=layer.output_dim,
            activation=layer.activation,
            dropout_rate=layer.dropout_rate,
            layer_params=layer.layer_params.copy()
        )
        
        if mutation_type == 'type':
            new_layer.layer_type = random.choice(self.layer_types)
            new_layer.layer_params = self._sample_layer_params(new_layer.layer_type)
        elif mutation_type == 'activation':
            new_layer.activation = random.choice(self.activations)
        elif mutation_type == 'dropout':
            new_layer.dropout_rate = random.choice(self.dropout_rates)
        elif mutation_type == 'params':
            new_layer.layer_params = self._sample_layer_params(new_layer.layer_type)
        
        return new_layer
    
    def _create_random_layer(self, input_dim: int, output_dim: int) -> LayerConfig:
        """Create a random layer with specified dimensions."""
        return LayerConfig(
            layer_type=random.choice(self.layer_types),
            input_dim=input_dim,
            output_dim=output_dim,
            activation=random.choice(self.activations),
            dropout_rate=random.choice(self.dropout_rates),
            layer_params=self._sample_layer_params(random.choice(self.layer_types))
        )
    
    def _fix_dimensions(self, layers: List[LayerConfig]) -> List[LayerConfig]:
        """Fix dimension compatibility between layers."""
        if not layers:
            return layers
        
        # Fix first layer
        layers[0].input_dim = self.input_dim
        
        # Fix intermediate layers
        for i in range(1, len(layers)):
            layers[i].input_dim = layers[i-1].output_dim
        
        # Fix last layer
        layers[-1].output_dim = self.output_dim
        
        return layers


class ArchitectureBuilder:
    """
    Builds PyTorch models from architecture configurations.
    
    Converts abstract architecture descriptions into executable
    PyTorch neural network modules.
    """
    
    def __init__(self):
        """Initialize architecture builder."""
        self.activation_map = {
            ActivationType.RELU: nn.ReLU,
            ActivationType.GELU: nn.GELU,
            ActivationType.TANH: nn.Tanh,
            ActivationType.SIGMOID: nn.Sigmoid,
            ActivationType.SWISH: lambda: nn.SiLU()  # Swish is SiLU in PyTorch
        }
    
    def build_model(self, architecture: ArchitectureConfig) -> nn.Module:
        """
        Build PyTorch model from architecture configuration.
        
        Args:
            architecture: Architecture configuration
            
        Returns:
            PyTorch model
        """
        layers = []
        
        for i, layer_config in enumerate(architecture.layers):
            # Build layer
            layer = self._build_layer(layer_config)
            layers.append(layer)
            
            # Add activation if specified
            if layer_config.activation:
                activation = self.activation_map[layer_config.activation]()
                layers.append(activation)
            
            # Add dropout if specified
            if layer_config.dropout_rate > 0:
                dropout = nn.Dropout(layer_config.dropout_rate)
                layers.append(dropout)
        
        # Create sequential model
        model = nn.Sequential(*layers)
        
        return model
    
    def _build_layer(self, layer_config: LayerConfig) -> nn.Module:
        """Build a single layer from configuration."""
        layer_type = layer_config.layer_type
        input_dim = layer_config.input_dim
        output_dim = layer_config.output_dim
        params = layer_config.layer_params
        
        if layer_type == LayerType.LINEAR:
            return nn.Linear(input_dim, output_dim)
            
        elif layer_type == LayerType.CONV1D:
            kernel_size = params.get('kernel_size', 3)
            stride = params.get('stride', 1)
            padding = params.get('padding', 1)
            return nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding)
            
        elif layer_type == LayerType.LSTM:
            num_layers = params.get('num_layers', 1)
            bidirectional = params.get('bidirectional', False)
            return nn.LSTM(input_dim, output_dim, num_layers, 
                          batch_first=True, bidirectional=bidirectional)
            
        elif layer_type == LayerType.GRU:
            num_layers = params.get('num_layers', 1)
            bidirectional = params.get('bidirectional', False)
            return nn.GRU(input_dim, output_dim, num_layers,
                         batch_first=True, bidirectional=bidirectional)
            
        elif layer_type == LayerType.ATTENTION:
            num_heads = params.get('num_heads', 8)
            dropout = params.get('dropout', 0.0)
            return nn.MultiheadAttention(input_dim, num_heads, dropout, batch_first=True)
            
        elif layer_type == LayerType.MEMORY:
            # Simplified memory layer (would need full implementation)
            return nn.Linear(input_dim, output_dim)
            
        elif layer_type == LayerType.DROPOUT:
            dropout_rate = params.get('dropout_rate', 0.1)
            return nn.Dropout(dropout_rate)
            
        elif layer_type == LayerType.NORM:
            return nn.LayerNorm(input_dim)
            
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")


class ArchitectureEvaluator:
    """
    Evaluates architecture performance on cognitive tasks.
    
    Provides methods for training and evaluating architectures
    on specific cognitive tasks with performance metrics.
    """
    
    def __init__(self, 
                 device: torch.device = None,
                 max_epochs: int = 10,
                 early_stopping_patience: int = 3):
        """
        Initialize architecture evaluator.
        
        Args:
            device: Device for training (CPU/GPU)
            max_epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
        """
        self.device = device or torch.device('cpu')
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
    def evaluate_architecture(self, 
                             architecture: ArchitectureConfig,
                             train_data: Tuple[torch.Tensor, torch.Tensor],
                             val_data: Tuple[torch.Tensor, torch.Tensor],
                             task_type: str = "general") -> Dict[str, float]:
        """
        Evaluate architecture performance.
        
        Args:
            architecture: Architecture to evaluate
            train_data: Training data (inputs, targets)
            val_data: Validation data (inputs, targets)
            task_type: Type of cognitive task
            
        Returns:
            Performance metrics
        """
        # Build model
        builder = ArchitectureBuilder()
        model = builder.build_model(architecture)
        model.to(self.device)
        
        # Train model
        train_metrics = self._train_model(model, train_data, val_data)
        
        # Evaluate on validation set
        val_metrics = self._evaluate_model(model, val_data)
        
        # Compute architecture-specific metrics
        arch_metrics = self._compute_architecture_metrics(model, architecture)
        
        # Combine all metrics
        all_metrics = {
            **train_metrics,
            **val_metrics,
            **arch_metrics,
            'task_type': task_type
        }
        
        return all_metrics
    
    def _train_model(self, 
                    model: nn.Module,
                    train_data: Tuple[torch.Tensor, torch.Tensor],
                    val_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Train the model and return training metrics."""
        train_inputs, train_targets = train_data
        val_inputs, val_targets = val_data
        
        # Move data to device
        train_inputs = train_inputs.to(self.device)
        train_targets = train_targets.to(self.device)
        val_inputs = val_inputs.to(self.device)
        val_targets = val_targets.to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.max_epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            train_outputs = model(train_inputs)
            train_loss = criterion(train_outputs, train_targets)
            train_loss.backward()
            optimizer.step()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'converged': patience_counter < self.early_stopping_patience
        }
    
    def _evaluate_model(self, 
                       model: nn.Module,
                       val_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model on validation data."""
        val_inputs, val_targets = val_data
        val_inputs = val_inputs.to(self.device)
        val_targets = val_targets.to(self.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(val_inputs)
            
            # Compute various metrics
            mse_loss = nn.MSELoss()(outputs, val_targets).item()
            mae_loss = nn.L1Loss()(outputs, val_targets).item()
            
            # Compute correlation
            outputs_flat = outputs.view(-1)
            targets_flat = val_targets.view(-1)
            correlation = torch.corrcoef(torch.stack([outputs_flat, targets_flat]))[0, 1].item()
            
            # Compute accuracy for classification-like tasks
            accuracy = self._compute_accuracy(outputs, val_targets)
        
        return {
            'val_mse': mse_loss,
            'val_mae': mae_loss,
            'val_correlation': correlation,
            'val_accuracy': accuracy
        }
    
    def _compute_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy for classification-like tasks."""
        # Simple threshold-based accuracy
        predictions = (outputs > 0.5).float()
        targets_binary = (targets > 0.5).float()
        accuracy = (predictions == targets_binary).float().mean().item()
        return accuracy
    
    def _compute_architecture_metrics(self, 
                                    model: nn.Module,
                                    architecture: ArchitectureConfig) -> Dict[str, float]:
        """Compute architecture-specific metrics."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate FLOPs (simplified)
        flops = self._estimate_flops(model, architecture)
        
        # Compute complexity metrics
        num_layers = len(architecture.layers)
        avg_layer_size = np.mean([layer.output_dim for layer in architecture.layers])
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_flops': flops,
            'num_layers': num_layers,
            'avg_layer_size': avg_layer_size,
            'parameter_efficiency': trainable_params / max(1, flops),
            'architecture_complexity': num_layers * avg_layer_size
        }
    
    def _estimate_flops(self, model: nn.Module, architecture: ArchitectureConfig) -> float:
        """Estimate FLOPs for the model (simplified)."""
        total_flops = 0
        
        for layer_config in architecture.layers:
            if layer_config.layer_type == LayerType.LINEAR:
                flops = layer_config.input_dim * layer_config.output_dim
            elif layer_config.layer_type in [LayerType.LSTM, LayerType.GRU]:
                # Simplified FLOP estimation for RNNs
                flops = 4 * layer_config.input_dim * layer_config.output_dim
            elif layer_config.layer_type == LayerType.ATTENTION:
                # Simplified FLOP estimation for attention
                flops = 3 * layer_config.input_dim * layer_config.output_dim
            else:
                flops = layer_config.input_dim * layer_config.output_dim
            
            total_flops += flops
        
        return total_flops


class ArchitectureSearch:
    """
    Main neural architecture search system.
    
    Implements search strategies for finding optimal cognitive architectures
    using evolutionary algorithms, random search, and Bayesian optimization.
    """
    
    def __init__(self, 
                 search_space: SearchSpace,
                 evaluator: ArchitectureEvaluator,
                 search_strategy: str = "evolutionary"):
        """
        Initialize architecture search.
        
        Args:
            search_space: Search space definition
            evaluator: Architecture evaluator
            search_strategy: Search strategy ("evolutionary", "random", "bayesian")
        """
        self.search_space = search_space
        self.evaluator = evaluator
        self.search_strategy = search_strategy
        
        # Search state
        self.population = []
        self.performance_history = []
        self.best_architecture = None
        self.best_performance = -float('inf')
        
        logger.info(f"Initialized ArchitectureSearch with {search_strategy} strategy")
    
    def search(self, 
               train_data: Tuple[torch.Tensor, torch.Tensor],
               val_data: Tuple[torch.Tensor, torch.Tensor],
               task_type: str = "general",
               num_iterations: int = 50,
               population_size: int = 20) -> Tuple[ArchitectureConfig, Dict[str, float]]:
        """
        Perform architecture search.
        
        Args:
            train_data: Training data
            val_data: Validation data
            task_type: Type of cognitive task
            num_iterations: Number of search iterations
            population_size: Population size for evolutionary search
            
        Returns:
            Tuple of (best_architecture, best_metrics)
        """
        if self.search_strategy == "evolutionary":
            return self._evolutionary_search(train_data, val_data, task_type, 
                                           num_iterations, population_size)
        elif self.search_strategy == "random":
            return self._random_search(train_data, val_data, task_type, num_iterations)
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")
    
    def _evolutionary_search(self, 
                           train_data: Tuple[torch.Tensor, torch.Tensor],
                           val_data: Tuple[torch.Tensor, torch.Tensor],
                           task_type: str,
                           num_iterations: int,
                           population_size: int) -> Tuple[ArchitectureConfig, Dict[str, float]]:
        """Perform evolutionary architecture search."""
        
        # Initialize population
        self.population = []
        for _ in range(population_size):
            architecture = self.search_space.sample_architecture(task_type)
            self.population.append(architecture)
        
        for iteration in range(num_iterations):
            logger.info(f"Evolution iteration {iteration + 1}/{num_iterations}")
            
            # Evaluate population
            fitness_scores = []
            for i, architecture in enumerate(self.population):
                try:
                    metrics = self.evaluator.evaluate_architecture(
                        architecture, train_data, val_data, task_type
                    )
                    fitness = self._compute_fitness(metrics)
                    fitness_scores.append(fitness)
                    
                    # Update best architecture
                    if fitness > self.best_performance:
                        self.best_performance = fitness
                        self.best_architecture = architecture
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate architecture {i}: {e}")
                    fitness_scores.append(-float('inf'))
            
            # Selection and reproduction
            new_population = self._evolve_population(self.population, fitness_scores)
            self.population = new_population
            
            # Track progress
            avg_fitness = np.mean([f for f in fitness_scores if f != -float('inf')])
            self.performance_history.append(avg_fitness)
            
            logger.info(f"Iteration {iteration + 1}: avg_fitness={avg_fitness:.4f}, "
                       f"best_fitness={self.best_performance:.4f}")
        
        # Get final metrics for best architecture
        best_metrics = self.evaluator.evaluate_architecture(
            self.best_architecture, train_data, val_data, task_type
        )
        
        return self.best_architecture, best_metrics
    
    def _random_search(self, 
                      train_data: Tuple[torch.Tensor, torch.Tensor],
                      val_data: Tuple[torch.Tensor, torch.Tensor],
                      task_type: str,
                      num_iterations: int) -> Tuple[ArchitectureConfig, Dict[str, float]]:
        """Perform random architecture search."""
        
        for iteration in range(num_iterations):
            logger.info(f"Random search iteration {iteration + 1}/{num_iterations}")
            
            # Sample random architecture
            architecture = self.search_space.sample_architecture(task_type)
            
            try:
                # Evaluate architecture
                metrics = self.evaluator.evaluate_architecture(
                    architecture, train_data, val_data, task_type
                )
                fitness = self._compute_fitness(metrics)
                
                # Update best architecture
                if fitness > self.best_performance:
                    self.best_performance = fitness
                    self.best_architecture = architecture
                
                self.performance_history.append(fitness)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate architecture {iteration}: {e}")
                self.performance_history.append(-float('inf'))
        
        # Get final metrics for best architecture
        best_metrics = self.evaluator.evaluate_architecture(
            self.best_architecture, train_data, val_data, task_type
        )
        
        return self.best_architecture, best_metrics
    
    def _compute_fitness(self, metrics: Dict[str, float]) -> float:
        """Compute fitness score from evaluation metrics."""
        # Weighted combination of metrics
        performance_weight = 0.4
        efficiency_weight = 0.3
        complexity_weight = 0.3
        
        # Performance component (higher is better)
        performance = metrics.get('val_correlation', 0.0)
        if performance < 0:
            performance = 0.0
        
        # Efficiency component (lower parameters is better)
        total_params = metrics.get('total_parameters', 1e6)
        efficiency = 1.0 / (1.0 + total_params / 1e6)  # Normalize by 1M parameters
        
        # Complexity component (moderate complexity is better)
        complexity = metrics.get('architecture_complexity', 1000)
        complexity_score = 1.0 / (1.0 + abs(complexity - 5000) / 5000)  # Target ~5000
        
        # Combined fitness
        fitness = (performance_weight * performance +
                  efficiency_weight * efficiency +
                  complexity_weight * complexity_score)
        
        return fitness
    
    def _evolve_population(self, 
                          population: List[ArchitectureConfig],
                          fitness_scores: List[float]) -> List[ArchitectureConfig]:
        """Evolve population using selection, crossover, and mutation."""
        
        # Selection: tournament selection
        selected = self._tournament_selection(population, fitness_scores, len(population))
        
        # Crossover and mutation
        new_population = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            # Crossover (simplified: just pick one parent)
            child = parent1 if random.random() < 0.5 else parent2
            
            # Mutation
            if random.random() < 0.3:  # Mutation probability
                child = self.search_space.mutate_architecture(child, mutation_rate=0.1)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, 
                            population: List[ArchitectureConfig],
                            fitness_scores: List[float],
                            num_selected: int,
                            tournament_size: int = 3) -> List[ArchitectureConfig]:
        """Tournament selection for evolutionary algorithm."""
        selected = []
        
        for _ in range(num_selected):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), 
                                             min(tournament_size, len(population)))
            
            # Find best individual in tournament
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[best_idx])
        
        return selected
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of search process."""
        return {
            'search_strategy': self.search_strategy,
            'best_performance': self.best_performance,
            'num_iterations': len(self.performance_history),
            'performance_history': self.performance_history,
            'best_architecture_summary': {
                'num_layers': len(self.best_architecture.layers) if self.best_architecture else 0,
                'task_type': self.best_architecture.task_type if self.best_architecture else None,
                'use_attention': self.best_architecture.use_attention if self.best_architecture else False,
                'use_memory': self.best_architecture.use_memory if self.best_architecture else False
            }
        }
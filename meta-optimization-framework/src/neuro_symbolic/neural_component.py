"""
Neural Component Implementation

Implements neural computation N(x) for the neuro-symbolic framework.
Provides deep learning models optimized for cognitive tasks with
attention mechanisms and working memory modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NeuralConfig:
    """Configuration for neural component."""
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 64
    num_layers: int = 3
    dropout_rate: float = 0.1
    attention_heads: int = 8
    memory_size: int = 32
    use_residual: bool = True
    activation: str = "relu"


class WorkingMemoryModule(nn.Module):
    """
    Working memory module for cognitive tasks.
    
    Implements a differentiable working memory with attention-based
    read/write operations, inspired by Neural Turing Machines.
    """
    
    def __init__(self, 
                 memory_size: int = 32,
                 memory_dim: int = 64,
                 controller_dim: int = 128):
        """
        Initialize working memory module.
        
        Args:
            memory_size: Number of memory slots
            memory_dim: Dimensionality of each memory slot
            controller_dim: Dimensionality of controller hidden state
        """
        super().__init__()
        
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.controller_dim = controller_dim
        
        # Memory matrix
        self.register_buffer('memory', torch.zeros(memory_size, memory_dim))
        
        # Controller network
        self.controller = nn.LSTM(memory_dim, controller_dim, batch_first=True)
        
        # Read/Write heads
        self.read_head = nn.Linear(controller_dim, memory_dim)
        self.write_head = nn.Linear(controller_dim, memory_dim * 2)  # key + value
        self.erase_head = nn.Linear(controller_dim, memory_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(memory_dim, num_heads=4, batch_first=True)
        
        # Gating mechanisms
        self.read_gate = nn.Linear(controller_dim, 1)
        self.write_gate = nn.Linear(controller_dim, 1)
        
        self._reset_memory()
    
    def _reset_memory(self) -> None:
        """Reset memory to initial state."""
        nn.init.xavier_uniform_(self.memory)
    
    def forward(self, 
                input_seq: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through working memory.
        
        Args:
            input_seq: Input sequence [batch_size, seq_len, input_dim]
            hidden_state: Previous hidden state for LSTM
            
        Returns:
            Tuple of (output_sequence, memory_info)
        """
        batch_size, seq_len, input_dim = input_seq.shape
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h0 = torch.zeros(1, batch_size, self.controller_dim, device=input_seq.device)
            c0 = torch.zeros(1, batch_size, self.controller_dim, device=input_seq.device)
            hidden_state = (h0, c0)
        
        # Expand memory for batch processing
        batch_memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        outputs = []
        memory_states = []
        
        for t in range(seq_len):
            # Current input
            current_input = input_seq[:, t:t+1, :]  # [batch_size, 1, input_dim]
            
            # Controller step
            controller_out, hidden_state = self.controller(current_input, hidden_state)
            controller_hidden = controller_out.squeeze(1)  # [batch_size, controller_dim]
            
            # Read from memory
            read_output, read_attention = self._read_memory(controller_hidden, batch_memory)
            
            # Write to memory
            batch_memory = self._write_memory(controller_hidden, batch_memory)
            
            # Combine input with memory read
            combined_output = current_input.squeeze(1) + read_output
            outputs.append(combined_output)
            
            # Store memory state for analysis
            memory_states.append({
                'memory_state': batch_memory.clone(),
                'read_attention': read_attention,
                'controller_hidden': controller_hidden.clone()
            })
        
        # Stack outputs
        output_sequence = torch.stack(outputs, dim=1)  # [batch_size, seq_len, input_dim]
        
        memory_info = {
            'final_memory': batch_memory,
            'memory_states': memory_states,
            'final_hidden': hidden_state
        }
        
        return output_sequence, memory_info
    
    def _read_memory(self, 
                    controller_hidden: torch.Tensor,
                    memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from memory using attention mechanism."""
        
        # Generate read key
        read_key = self.read_head(controller_hidden).unsqueeze(1)  # [batch_size, 1, memory_dim]
        
        # Ensure memory has correct dimensions for attention
        # memory should be [batch_size, memory_size, memory_dim]
        if memory.dim() == 4:
            memory = memory.squeeze(1)  # Remove extra dimension if present
        elif memory.dim() == 2:
            memory = memory.unsqueeze(0)  # Add batch dimension if missing
        
        # Debug: print shapes
        # print(f"read_key shape: {read_key.shape}, memory shape: {memory.shape}")
        
        # Attention over memory
        read_output, attention_weights = self.attention(
            read_key, memory, memory
        )
        
        # Apply read gate
        read_gate = torch.sigmoid(self.read_gate(controller_hidden))
        gated_output = read_gate.unsqueeze(1) * read_output
        
        return gated_output.squeeze(1), attention_weights.squeeze(1)
    
    def _write_memory(self, 
                     controller_hidden: torch.Tensor,
                     memory: torch.Tensor) -> torch.Tensor:
        """Write to memory with erase and add operations."""
        
        # Generate write parameters
        write_params = self.write_head(controller_hidden)  # [batch_size, memory_dim * 2]
        write_key = write_params[:, :self.memory_dim]
        write_value = write_params[:, self.memory_dim:]
        
        erase_vector = torch.sigmoid(self.erase_head(controller_hidden))
        write_gate = torch.sigmoid(self.write_gate(controller_hidden))
        
        # Compute attention weights for writing
        write_attention = F.softmax(
            torch.bmm(write_key.unsqueeze(1), memory.transpose(1, 2)), dim=-1
        )  # [batch_size, 1, memory_size]
        
        # Erase operation
        erase_weights = write_attention.transpose(1, 2)  # [batch_size, memory_size, 1]
        erase_matrix = erase_weights * erase_vector.unsqueeze(1)
        memory_after_erase = memory * (1 - erase_matrix)
        
        # Add operation
        add_matrix = erase_weights * write_value.unsqueeze(1)
        updated_memory = memory_after_erase + write_gate.unsqueeze(1) * add_matrix
        
        return updated_memory


class CognitiveAttentionModule(nn.Module):
    """
    Cognitive attention module with task-specific attention patterns.
    
    Implements attention mechanisms that model cognitive processes
    like selective attention, divided attention, and attention switching.
    """
    
    def __init__(self, 
                 input_dim: int = 64,
                 num_heads: int = 8,
                 attention_dropout: float = 0.1):
        """
        Initialize cognitive attention module.
        
        Args:
            input_dim: Input feature dimension
            num_heads: Number of attention heads
            attention_dropout: Dropout rate for attention weights
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        
        # Task-specific attention patterns
        self.task_attention = nn.ModuleDict({
            'n_back': nn.Linear(input_dim, input_dim),
            'stroop': nn.Linear(input_dim, input_dim),
            'planning': nn.Linear(input_dim, input_dim)
        })
        
        # Attention control mechanisms
        self.attention_control = nn.Linear(input_dim, 3)  # Focus, divide, switch
        self.attention_intensity = nn.Linear(input_dim, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, 
                input_seq: torch.Tensor,
                task_type: str = 'general',
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through cognitive attention.
        
        Args:
            input_seq: Input sequence [batch_size, seq_len, input_dim]
            task_type: Type of cognitive task
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (attended_output, attention_info)
        """
        batch_size, seq_len, input_dim = input_seq.shape
        
        # Apply task-specific attention modulation
        if task_type in self.task_attention:
            task_modulated = self.task_attention[task_type](input_seq)
            modulated_input = input_seq + 0.1 * task_modulated
        else:
            modulated_input = input_seq
        
        # Multi-head attention
        attended_output, attention_weights = self.attention(
            modulated_input, modulated_input, modulated_input,
            attn_mask=attention_mask
        )
        
        # Attention control
        attention_control = F.softmax(self.attention_control(modulated_input), dim=-1)
        attention_intensity = torch.sigmoid(self.attention_intensity(modulated_input))
        
        # Apply attention control
        controlled_output = attended_output * attention_intensity
        
        # Residual connection and layer norm
        output = self.layer_norm(controlled_output + input_seq)
        
        attention_info = {
            'attention_weights': attention_weights,
            'attention_control': attention_control,
            'attention_intensity': attention_intensity,
            'task_modulation': task_type
        }
        
        return output, attention_info


class NeuralComponent(nn.Module):
    """
    Neural component N(x) for neuro-symbolic integration.
    
    Implements deep neural networks with cognitive-inspired architectures
    including working memory, attention mechanisms, and task-specific processing.
    """
    
    def __init__(self, config: NeuralConfig):
        """
        Initialize neural component.
        
        Args:
            config: Neural component configuration
        """
        super().__init__()
        
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Core neural layers
        self.neural_layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer = nn.ModuleDict({
                'linear': nn.Linear(config.hidden_dim, config.hidden_dim),
                'norm': nn.LayerNorm(config.hidden_dim),
                'dropout': nn.Dropout(config.dropout_rate)
            })
            self.neural_layers.append(layer)
        
        # Working memory module
        self.working_memory = WorkingMemoryModule(
            memory_size=config.memory_size,
            memory_dim=config.hidden_dim,
            controller_dim=config.hidden_dim
        )
        
        # Cognitive attention module
        self.cognitive_attention = CognitiveAttentionModule(
            input_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            attention_dropout=config.dropout_rate
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            'n_back': nn.Linear(config.hidden_dim, 2),  # match/no-match
            'stroop': nn.Linear(config.hidden_dim, 3),  # congruent/incongruent/neutral
            'planning': nn.Linear(config.hidden_dim, config.hidden_dim),  # goal representation
            'general': nn.Linear(config.hidden_dim, config.output_dim)
        })
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Activation function
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "gelu":
            self.activation = F.gelu
        elif config.activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = F.relu
        
        logger.info(f"Initialized NeuralComponent with {self._count_parameters()} parameters")
    
    def forward(self, 
                input_data: torch.Tensor,
                task_type: str = 'general',
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through neural component.
        
        Args:
            input_data: Input tensor [batch_size, seq_len, input_dim]
            task_type: Type of cognitive task
            context: Additional context information
            
        Returns:
            Tuple of (neural_output, processing_info)
        """
        context = context or {}
        
        # Input projection
        hidden = self.input_projection(input_data)
        hidden = self.activation(hidden)
        
        # Core neural processing
        layer_outputs = []
        for i, layer in enumerate(self.neural_layers):
            # Linear transformation
            layer_input = hidden
            hidden = layer['linear'](hidden)
            hidden = self.activation(hidden)
            
            # Normalization and dropout
            hidden = layer['norm'](hidden)
            hidden = layer['dropout'](hidden)
            
            # Residual connection
            if self.config.use_residual and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            
            layer_outputs.append(hidden.clone())
        
        # Working memory processing
        memory_output, memory_info = self.working_memory(hidden)
        
        # Cognitive attention
        attended_output, attention_info = self.cognitive_attention(
            memory_output, task_type=task_type
        )
        
        # Task-specific processing
        if task_type in self.task_heads:
            task_output = self.task_heads[task_type](attended_output)
        else:
            task_output = self.task_heads['general'](attended_output)
        
        # Final output projection
        neural_output = self.output_projection(attended_output)
        
        # Compile processing information
        processing_info = {
            'layer_outputs': layer_outputs,
            'memory_info': memory_info,
            'attention_info': attention_info,
            'task_output': task_output,
            'task_type': task_type,
            'hidden_states': {
                'input_projected': self.input_projection(input_data),
                'final_hidden': attended_output,
                'memory_enhanced': memory_output
            }
        }
        
        return neural_output, processing_info
    
    def get_attention_patterns(self, 
                              input_data: torch.Tensor,
                              task_type: str = 'general') -> Dict[str, torch.Tensor]:
        """Extract attention patterns for analysis."""
        
        with torch.no_grad():
            _, processing_info = self.forward(input_data, task_type)
            
            attention_patterns = {
                'attention_weights': processing_info['attention_info']['attention_weights'],
                'attention_control': processing_info['attention_info']['attention_control'],
                'attention_intensity': processing_info['attention_info']['attention_intensity'],
                'memory_attention': [
                    state['read_attention'] for state in processing_info['memory_info']['memory_states']
                ]
            }
        
        return attention_patterns
    
    def get_memory_states(self, 
                         input_data: torch.Tensor,
                         task_type: str = 'general') -> Dict[str, torch.Tensor]:
        """Extract working memory states for analysis."""
        
        with torch.no_grad():
            _, processing_info = self.forward(input_data, task_type)
            
            memory_states = {
                'final_memory': processing_info['memory_info']['final_memory'],
                'memory_evolution': [
                    state['memory_state'] for state in processing_info['memory_info']['memory_states']
                ],
                'controller_states': [
                    state['controller_hidden'] for state in processing_info['memory_info']['memory_states']
                ]
            }
        
        return memory_states
    
    def analyze_cognitive_load(self, 
                              input_data: torch.Tensor,
                              task_type: str = 'general') -> Dict[str, float]:
        """Analyze cognitive load metrics."""
        
        with torch.no_grad():
            _, processing_info = self.forward(input_data, task_type)
            
            # Attention intensity as cognitive load indicator
            attention_intensity = processing_info['attention_info']['attention_intensity']
            avg_attention_load = torch.mean(attention_intensity).item()
            
            # Memory usage as cognitive load indicator
            memory_states = processing_info['memory_info']['memory_states']
            memory_activations = [torch.norm(state['memory_state']).item() for state in memory_states]
            avg_memory_load = np.mean(memory_activations)
            
            # Layer activation magnitudes
            layer_outputs = processing_info['layer_outputs']
            layer_activations = [torch.norm(output).item() for output in layer_outputs]
            avg_layer_load = np.mean(layer_activations)
            
            cognitive_load = {
                'attention_load': avg_attention_load,
                'memory_load': avg_memory_load,
                'processing_load': avg_layer_load,
                'total_load': (avg_attention_load + avg_memory_load + avg_layer_load) / 3
            }
        
        return cognitive_load
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'config': self.config,
            'total_parameters': self._count_parameters(),
            'memory_size': self.config.memory_size,
            'attention_heads': self.config.attention_heads,
            'num_layers': self.config.num_layers,
            'task_heads': list(self.task_heads.keys())
        }
"""
Efficiency Metrics Implementation

Implements comprehensive efficiency tracking for the meta-optimization framework,
including FLOPs counting, memory usage monitoring, timing analysis, and
energy consumption estimation.
"""

import torch
import torch.nn as nn
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyReport:
    """Comprehensive efficiency report."""
    flops: float
    memory_usage: float  # MB
    execution_time: float  # seconds
    energy_estimate: float  # Joules (estimated)
    throughput: float  # samples/second
    efficiency_score: float  # Combined efficiency metric
    breakdown: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return (f"EfficiencyReport(FLOPs: {self.flops:.2e}, "
                f"Memory: {self.memory_usage:.1f}MB, "
                f"Time: {self.execution_time:.3f}s, "
                f"Throughput: {self.throughput:.1f} samples/s, "
                f"Score: {self.efficiency_score:.3f})")


class FLOPsCounter:
    """
    Floating Point Operations counter for neural networks.
    
    Tracks computational complexity by counting FLOPs for different
    layer types and operations.
    """
    
    def __init__(self):
        """Initialize FLOPs counter."""
        self.total_flops = 0
        self.layer_flops = {}
        self.operation_counts = {}
        
    def reset(self) -> None:
        """Reset all counters."""
        self.total_flops = 0
        self.layer_flops = {}
        self.operation_counts = {}
    
    def count_linear(self, input_features: int, output_features: int, batch_size: int = 1) -> float:
        """Count FLOPs for linear layer."""
        flops = batch_size * input_features * output_features
        self.total_flops += flops
        self.operation_counts['linear'] = self.operation_counts.get('linear', 0) + flops
        return flops
    
    def count_conv1d(self, 
                     input_channels: int,
                     output_channels: int,
                     kernel_size: int,
                     input_length: int,
                     batch_size: int = 1) -> float:
        """Count FLOPs for 1D convolution."""
        output_length = input_length  # Assuming same padding
        flops = batch_size * output_channels * output_length * input_channels * kernel_size
        self.total_flops += flops
        self.operation_counts['conv1d'] = self.operation_counts.get('conv1d', 0) + flops
        return flops
    
    def count_lstm(self, 
                   input_size: int,
                   hidden_size: int,
                   sequence_length: int,
                   batch_size: int = 1,
                   num_layers: int = 1) -> float:
        """Count FLOPs for LSTM layer."""
        # LSTM has 4 gates, each with input and hidden transformations
        flops_per_gate = input_size * hidden_size + hidden_size * hidden_size
        flops_per_timestep = 4 * flops_per_gate
        total_flops = batch_size * sequence_length * flops_per_timestep * num_layers
        
        self.total_flops += total_flops
        self.operation_counts['lstm'] = self.operation_counts.get('lstm', 0) + total_flops
        return total_flops
    
    def count_attention(self, 
                       sequence_length: int,
                       feature_dim: int,
                       num_heads: int,
                       batch_size: int = 1) -> float:
        """Count FLOPs for multi-head attention."""
        # Query, Key, Value projections
        qkv_flops = 3 * batch_size * sequence_length * feature_dim * feature_dim
        
        # Attention computation
        attention_flops = batch_size * num_heads * sequence_length * sequence_length * (feature_dim // num_heads)
        
        # Output projection
        output_flops = batch_size * sequence_length * feature_dim * feature_dim
        
        total_flops = qkv_flops + attention_flops + output_flops
        self.total_flops += total_flops
        self.operation_counts['attention'] = self.operation_counts.get('attention', 0) + total_flops
        return total_flops
    
    def count_activation(self, num_elements: int) -> float:
        """Count FLOPs for activation functions."""
        # Most activations are ~1 FLOP per element
        flops = num_elements
        self.total_flops += flops
        self.operation_counts['activation'] = self.operation_counts.get('activation', 0) + flops
        return flops
    
    def count_model_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Count FLOPs for entire model with given input shape."""
        self.reset()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Hook to count FLOPs
        def flop_hook(module, input, output):
            if isinstance(module, nn.Linear):
                batch_size = input[0].shape[0]
                self.count_linear(module.in_features, module.out_features, batch_size)
            elif isinstance(module, nn.Conv1d):
                batch_size = input[0].shape[0]
                input_length = input[0].shape[2]
                self.count_conv1d(
                    module.in_channels, module.out_channels,
                    module.kernel_size[0], input_length, batch_size
                )
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                batch_size = input[0].shape[0]
                sequence_length = input[0].shape[1]
                self.count_lstm(
                    module.input_size, module.hidden_size,
                    sequence_length, batch_size, module.num_layers
                )
            elif isinstance(module, nn.MultiheadAttention):
                batch_size = input[0].shape[0]
                sequence_length = input[0].shape[1]
                feature_dim = input[0].shape[2]
                self.count_attention(
                    sequence_length, feature_dim, module.num_heads, batch_size
                )
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
                if hasattr(output, 'numel'):
                    self.count_activation(output.numel())
        
        # Register hooks
        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(flop_hook))
        
        try:
            # Forward pass to count FLOPs
            with torch.no_grad():
                model(dummy_input)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return self.total_flops
    
    def get_flops_breakdown(self) -> Dict[str, float]:
        """Get breakdown of FLOPs by operation type."""
        return self.operation_counts.copy()


class MemoryProfiler:
    """
    Memory usage profiler for neural networks.
    
    Tracks memory consumption including model parameters,
    activations, gradients, and temporary buffers.
    """
    
    def __init__(self):
        """Initialize memory profiler."""
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_timeline = []
        self.baseline_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def start_profiling(self) -> None:
        """Start memory profiling."""
        self.baseline_memory = self._get_memory_usage()
        self.peak_memory = 0
        self.memory_timeline = []
        
    def record_memory(self, label: str = "") -> float:
        """Record current memory usage."""
        cpu_memory = self._get_memory_usage() - self.baseline_memory
        gpu_memory = self._get_gpu_memory()
        total_memory = cpu_memory + gpu_memory
        
        self.current_memory = total_memory
        self.peak_memory = max(self.peak_memory, total_memory)
        
        self.memory_timeline.append({
            'label': label,
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'total_memory': total_memory,
            'timestamp': time.time()
        })
        
        return total_memory
    
    def profile_model(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """Profile memory usage of model."""
        self.start_profiling()
        
        # Model parameters
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        self.record_memory("parameters")
        
        # Create input
        dummy_input = torch.randn(input_shape)
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        self.record_memory("input_created")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        self.record_memory("forward_pass")
        
        # Backward pass (if training)
        if model.training:
            loss = output.sum()
            loss.backward()
            self.record_memory("backward_pass")
        
        return {
            'parameter_memory': param_memory,
            'peak_memory': self.peak_memory,
            'final_memory': self.current_memory,
            'memory_timeline': self.memory_timeline
        }


class TimingProfiler:
    """
    Timing profiler for performance analysis.
    
    Measures execution times for different components
    and operations in the neural network.
    """
    
    def __init__(self):
        """Initialize timing profiler."""
        self.timings = {}
        self.start_times = {}
        self.total_time = 0
        
    def start_timer(self, name: str) -> None:
        """Start timing for named operation."""
        self.start_times[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        """End timing for named operation."""
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.perf_counter() - self.start_times[name]
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
        
        del self.start_times[name]
        return elapsed
    
    @contextmanager
    def time_context(self, name: str):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)
    
    def profile_model_inference(self, 
                               model: nn.Module,
                               input_data: torch.Tensor,
                               num_runs: int = 100) -> Dict[str, float]:
        """Profile model inference time."""
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_data)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'throughput': input_data.shape[0] / np.mean(times)  # samples/second
        }
    
    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all timings."""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times)
            }
        return summary


class EnergyEstimator:
    """
    Energy consumption estimator.
    
    Estimates energy consumption based on computational
    complexity and hardware characteristics.
    """
    
    def __init__(self, 
                 cpu_power: float = 15.0,  # Watts
                 gpu_power: float = 250.0,  # Watts
                 memory_power: float = 5.0):  # Watts
        """
        Initialize energy estimator.
        
        Args:
            cpu_power: CPU power consumption in Watts
            gpu_power: GPU power consumption in Watts
            memory_power: Memory power consumption in Watts
        """
        self.cpu_power = cpu_power
        self.gpu_power = gpu_power
        self.memory_power = memory_power
        
    def estimate_energy(self, 
                       execution_time: float,
                       flops: float,
                       memory_usage: float,
                       use_gpu: bool = False) -> float:
        """
        Estimate energy consumption.
        
        Args:
            execution_time: Execution time in seconds
            flops: Number of floating point operations
            memory_usage: Memory usage in MB
            use_gpu: Whether GPU is used
            
        Returns:
            Energy consumption in Joules
        """
        # Base power consumption
        if use_gpu:
            compute_power = self.gpu_power
        else:
            compute_power = self.cpu_power
        
        # Memory power based on usage
        memory_power = self.memory_power * (memory_usage / 1000)  # Scale by GB
        
        # Total power
        total_power = compute_power + memory_power
        
        # Energy = Power × Time
        energy = total_power * execution_time
        
        return energy
    
    def estimate_model_energy(self, 
                             model: nn.Module,
                             input_shape: Tuple[int, ...],
                             num_inferences: int = 1000) -> Dict[str, float]:
        """Estimate energy consumption for model inferences."""
        
        # Count FLOPs
        flop_counter = FLOPsCounter()
        flops_per_inference = flop_counter.count_model_flops(model, input_shape)
        
        # Profile memory
        memory_profiler = MemoryProfiler()
        memory_profile = memory_profiler.profile_model(model, input_shape)
        
        # Profile timing
        timing_profiler = TimingProfiler()
        dummy_input = torch.randn(input_shape)
        timing_profile = timing_profiler.profile_model_inference(model, dummy_input)
        
        # Estimate energy per inference
        time_per_inference = timing_profile['mean_time']
        memory_usage = memory_profile['peak_memory']
        use_gpu = next(model.parameters()).is_cuda if list(model.parameters()) else False
        
        energy_per_inference = self.estimate_energy(
            time_per_inference, flops_per_inference, memory_usage, use_gpu
        )
        
        # Total energy for all inferences
        total_energy = energy_per_inference * num_inferences
        
        return {
            'energy_per_inference': energy_per_inference,
            'total_energy': total_energy,
            'flops_per_inference': flops_per_inference,
            'time_per_inference': time_per_inference,
            'memory_usage': memory_usage,
            'energy_efficiency': flops_per_inference / energy_per_inference  # FLOPs/Joule
        }


class EfficiencyMetrics:
    """
    Main efficiency metrics system.
    
    Integrates FLOPs counting, memory profiling, timing analysis,
    and energy estimation for comprehensive efficiency evaluation.
    """
    
    def __init__(self):
        """Initialize efficiency metrics system."""
        self.flop_counter = FLOPsCounter()
        self.memory_profiler = MemoryProfiler()
        self.timing_profiler = TimingProfiler()
        self.energy_estimator = EnergyEstimator()
        
        logger.info("Initialized EfficiencyMetrics system")
    
    def profile_model(self, 
                     model: nn.Module,
                     input_shape: Tuple[int, ...],
                     num_runs: int = 100) -> EfficiencyReport:
        """
        Comprehensive efficiency profiling of model.
        
        Args:
            model: PyTorch model to profile
            input_shape: Input tensor shape
            num_runs: Number of runs for timing
            
        Returns:
            Comprehensive efficiency report
        """
        # Count FLOPs
        flops = self.flop_counter.count_model_flops(model, input_shape)
        flops_breakdown = self.flop_counter.get_flops_breakdown()
        
        # Profile memory
        memory_profile = self.memory_profiler.profile_model(model, input_shape)
        peak_memory = memory_profile['peak_memory']
        
        # Profile timing
        dummy_input = torch.randn(input_shape)
        if torch.cuda.is_available() and next(model.parameters(), torch.tensor(0)).is_cuda:
            dummy_input = dummy_input.cuda()
        
        timing_profile = self.timing_profiler.profile_model_inference(
            model, dummy_input, num_runs
        )
        execution_time = timing_profile['mean_time']
        throughput = timing_profile['throughput']
        
        # Estimate energy
        use_gpu = next(model.parameters(), torch.tensor(0)).is_cuda
        energy = self.energy_estimator.estimate_energy(
            execution_time, flops, peak_memory, use_gpu
        )
        
        # Compute efficiency score
        efficiency_score = self._compute_efficiency_score(
            flops, peak_memory, execution_time, energy
        )
        
        # Create comprehensive report
        report = EfficiencyReport(
            flops=flops,
            memory_usage=peak_memory,
            execution_time=execution_time,
            energy_estimate=energy,
            throughput=throughput,
            efficiency_score=efficiency_score,
            breakdown={
                'flops_breakdown': flops_breakdown,
                'memory_profile': memory_profile,
                'timing_profile': timing_profile,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            }
        )
        
        return report
    
    def compare_models(self, 
                      models: Dict[str, nn.Module],
                      input_shape: Tuple[int, ...]) -> Dict[str, EfficiencyReport]:
        """
        Compare efficiency of multiple models.
        
        Args:
            models: Dictionary of model name to model
            input_shape: Input tensor shape
            
        Returns:
            Dictionary of model name to efficiency report
        """
        reports = {}
        
        for name, model in models.items():
            logger.info(f"Profiling model: {name}")
            try:
                report = self.profile_model(model, input_shape)
                reports[name] = report
            except Exception as e:
                logger.error(f"Failed to profile model {name}: {e}")
                # Create dummy report for failed models
                reports[name] = EfficiencyReport(
                    flops=float('inf'),
                    memory_usage=float('inf'),
                    execution_time=float('inf'),
                    energy_estimate=float('inf'),
                    throughput=0.0,
                    efficiency_score=0.0
                )
        
        return reports
    
    def _compute_efficiency_score(self, 
                                 flops: float,
                                 memory: float,
                                 time: float,
                                 energy: float) -> float:
        """
        Compute overall efficiency score.
        
        Args:
            flops: Number of FLOPs
            memory: Memory usage in MB
            time: Execution time in seconds
            energy: Energy consumption in Joules
            
        Returns:
            Efficiency score (higher is better)
        """
        # Normalize metrics (lower is better for all except throughput)
        flop_score = 1.0 / (1.0 + flops / 1e9)  # Normalize by 1G FLOPs
        memory_score = 1.0 / (1.0 + memory / 1000)  # Normalize by 1GB
        time_score = 1.0 / (1.0 + time)  # Normalize by 1 second
        energy_score = 1.0 / (1.0 + energy / 100)  # Normalize by 100 Joules
        
        # Weighted combination
        efficiency_score = (0.3 * flop_score +
                           0.25 * memory_score +
                           0.25 * time_score +
                           0.2 * energy_score)
        
        return efficiency_score
    
    def benchmark_cognitive_tasks(self, 
                                 model: nn.Module,
                                 task_configs: Dict[str, Tuple[int, ...]]) -> Dict[str, EfficiencyReport]:
        """
        Benchmark model efficiency on different cognitive tasks.
        
        Args:
            model: Model to benchmark
            task_configs: Dictionary of task name to input shape
            
        Returns:
            Dictionary of task name to efficiency report
        """
        task_reports = {}
        
        for task_name, input_shape in task_configs.items():
            logger.info(f"Benchmarking task: {task_name}")
            report = self.profile_model(model, input_shape)
            task_reports[task_name] = report
        
        return task_reports
    
    def get_efficiency_summary(self, reports: Dict[str, EfficiencyReport]) -> Dict[str, Any]:
        """Get summary statistics across multiple efficiency reports."""
        
        if not reports:
            return {}
        
        # Extract metrics
        flops = [r.flops for r in reports.values() if r.flops != float('inf')]
        memory = [r.memory_usage for r in reports.values() if r.memory_usage != float('inf')]
        times = [r.execution_time for r in reports.values() if r.execution_time != float('inf')]
        energy = [r.energy_estimate for r in reports.values() if r.energy_estimate != float('inf')]
        scores = [r.efficiency_score for r in reports.values()]
        
        summary = {
            'num_models': len(reports),
            'flops_stats': {
                'mean': np.mean(flops) if flops else 0,
                'std': np.std(flops) if flops else 0,
                'min': np.min(flops) if flops else 0,
                'max': np.max(flops) if flops else 0
            },
            'memory_stats': {
                'mean': np.mean(memory) if memory else 0,
                'std': np.std(memory) if memory else 0,
                'min': np.min(memory) if memory else 0,
                'max': np.max(memory) if memory else 0
            },
            'time_stats': {
                'mean': np.mean(times) if times else 0,
                'std': np.std(times) if times else 0,
                'min': np.min(times) if times else 0,
                'max': np.max(times) if times else 0
            },
            'energy_stats': {
                'mean': np.mean(energy) if energy else 0,
                'std': np.std(energy) if energy else 0,
                'min': np.min(energy) if energy else 0,
                'max': np.max(energy) if energy else 0
            },
            'efficiency_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        }
        
        return summary
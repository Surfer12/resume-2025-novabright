"""
End-to-End Demo of Meta-Optimization Framework

Demonstrates the complete pipeline from data generation through
neuro-symbolic processing, optimization, and bias modeling.
"""

import torch
import numpy as np
import logging

# Import framework components
from src.utils.data_processing import DataProcessor
from src.neuro_symbolic.symbolic_component import SymbolicComponent
from src.neuro_symbolic.neural_component import NeuralComponent, NeuralConfig
from src.neuro_symbolic.adaptive_weighting import AdaptiveWeighting, WeightingConfig
from src.optimization.cognitive_constraints import CognitiveConstraints
from src.bias_framework.bias_mechanisms import CognitiveBiasFramework, BiasType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run end-to-end demonstration."""
    logger.info("Starting Meta-Optimization Framework Demo")
    
    # Step 1: Data Generation
    logger.info("Step 1: Generating cognitive task data")
    data_processor = DataProcessor(random_seed=42)
    
    # Generate N-back task data
    train_inputs, train_targets = data_processor.generate_n_back_data(
        batch_size=16, sequence_length=10, n_back=2, feature_dim=32
    )
    logger.info(f"Generated N-back data: {train_inputs.shape}")
    
    # Step 2: Initialize Components
    logger.info("Step 2: Initializing framework components")
    
    # Symbolic component
    symbolic_component = SymbolicComponent()
    
    # Neural component
    neural_config = NeuralConfig(
        input_dim=32,
        hidden_dim=64,
        output_dim=32,
        num_layers=3,
        attention_heads=4,
        memory_size=16
    )
    neural_component = NeuralComponent(neural_config)
    
    # Adaptive weighting
    weighting_config = WeightingConfig(
        initial_alpha=0.5,
        learning_rate=0.01,
        adaptation_window=5
    )
    adaptive_weighting = AdaptiveWeighting(weighting_config, 32, 32)
    
    # Cognitive constraints
    constraints = CognitiveConstraints()
    
    # Bias framework
    bias_types = [BiasType.CONFIRMATION, BiasType.ANCHORING, BiasType.AVAILABILITY]
    bias_framework = CognitiveBiasFramework(bias_types)
    
    logger.info("All components initialized successfully")
    
    # Step 3: Process Through Pipeline
    logger.info("Step 3: Processing data through complete pipeline")
    
    context = {
        'task_type': 'n_back',
        'n_back_level': 2,
        'sequence_position': 5
    }
    
    # Symbolic processing
    logger.info("  3a: Symbolic reasoning...")
    symbolic_output, symbolic_metadata = symbolic_component.forward(train_inputs, context)
    logger.info(f"     Rules fired: {symbolic_metadata['rules_fired']}")
    logger.info(f"     Confidence scores: {len(symbolic_metadata['confidence_scores'])}")
    
    # Neural processing
    logger.info("  3b: Neural computation...")
    neural_output, neural_metadata = neural_component.forward(train_inputs, task_type='n_back')
    logger.info(f"     Layers processed: {len(neural_metadata['layer_outputs'])}")
    logger.info(f"     Memory states: {len(neural_metadata['memory_info']['memory_states'])}")
    
    # Adaptive weighting
    logger.info("  3c: Adaptive integration...")
    combined_output, weighting_info = adaptive_weighting.forward(
        symbolic_output, neural_output, symbolic_metadata, neural_metadata
    )
    logger.info(f"     Alpha (symbolic weight): {weighting_info['alpha']:.3f}")
    logger.info(f"     Performance score: {weighting_info['performance_score']:.3f}")
    logger.info(f"     Authenticity score: {weighting_info['authenticity_score']:.3f}")
    
    # Bias application
    logger.info("  3d: Applying cognitive biases...")
    biased_output, bias_report = bias_framework.apply_biases(combined_output, context)
    logger.info(f"     Applied biases: {bias_report['applied_biases']}")
    logger.info(f"     Total bias effect: {bias_report['total_bias_effect']:.3f}")
    
    # Step 4: Constraint Checking
    logger.info("Step 4: Checking cognitive constraints")
    
    # Extract memory states for constraint checking
    memory_states = []
    for state in neural_metadata['memory_info']['memory_states']:
        memory_states.append(state['memory_state'])
    
    system_state = {
        'memory_states': memory_states,
        'attention_weights': neural_metadata['attention_info']['attention_weights'],
        'attention_control': neural_metadata['attention_info']['attention_control'],
        'sequence_length': train_inputs.shape[1],
        'task_complexity': 0.6,
        'processing_efficiency': weighting_info['efficiency_score'],
        'learning_engagement': 0.7
    }
    
    violations = constraints.check_all_constraints(system_state)
    logger.info(f"Constraint violations detected: {len(violations)}")
    
    for violation in violations[:3]:  # Show first 3 violations
        logger.info(f"  - {violation.constraint_type.value}: {violation.description}")
    
    # Step 5: Performance Analysis
    logger.info("Step 5: Analyzing performance")
    
    # Compute performance metrics
    mse_loss = torch.nn.MSELoss()(biased_output, train_targets)
    mae_loss = torch.nn.L1Loss()(biased_output, train_targets)
    
    # Compute correlation (handle dimension mismatch)
    output_flat = biased_output.view(-1)
    target_flat = train_targets.view(-1)
    
    # Ensure same size for correlation
    min_size = min(output_flat.size(0), target_flat.size(0))
    output_flat = output_flat[:min_size]
    target_flat = target_flat[:min_size]
    
    if min_size > 1:
        correlation = torch.corrcoef(torch.stack([output_flat, target_flat]))[0, 1]
    else:
        correlation = torch.tensor(0.0)  # Default if insufficient data
    
    logger.info(f"Performance Metrics:")
    logger.info(f"  - MSE Loss: {mse_loss.item():.4f}")
    logger.info(f"  - MAE Loss: {mae_loss.item():.4f}")
    logger.info(f"  - Correlation: {correlation.item():.4f}")
    
    # Step 6: Adaptive Learning Demonstration
    logger.info("Step 6: Demonstrating adaptive learning")
    
    alpha_history = [weighting_info['alpha']]
    performance_history = [weighting_info['performance_score']]
    
    # Simulate multiple learning iterations
    for iteration in range(4):
        logger.info(f"  Iteration {iteration + 1}:")
        
        # Add noise to simulate different conditions
        noisy_inputs = train_inputs + torch.randn_like(train_inputs) * 0.1
        
        # Process through pipeline
        symbolic_out, symbolic_meta = symbolic_component.forward(noisy_inputs, context)
        neural_out, neural_meta = neural_component.forward(noisy_inputs, task_type='n_back')
        combined_out, weight_info = adaptive_weighting.forward(
            symbolic_out, neural_out, symbolic_meta, neural_meta
        )
        
        alpha_history.append(weight_info['alpha'])
        performance_history.append(weight_info['performance_score'])
        
        logger.info(f"    Alpha: {weight_info['alpha']:.3f}")
        logger.info(f"    Performance: {weight_info['performance_score']:.3f}")
    
    # Analyze adaptation
    alpha_change = alpha_history[-1] - alpha_history[0]
    perf_change = performance_history[-1] - performance_history[0]
    
    logger.info(f"Adaptation Summary:")
    logger.info(f"  - Alpha change: {alpha_change:+.3f}")
    logger.info(f"  - Performance change: {perf_change:+.3f}")
    logger.info(f"  - Final symbolic/neural balance: {alpha_history[-1]:.1%}/{1-alpha_history[-1]:.1%}")
    
    # Step 7: Summary
    logger.info("Step 7: Demo Summary")
    logger.info("="*50)
    logger.info("Meta-Optimization Framework Demo Completed Successfully!")
    logger.info(f"✓ Processed {train_inputs.shape[0]} samples through complete pipeline")
    logger.info(f"✓ Symbolic reasoning: {symbolic_metadata['rules_fired']} rules activated")
    logger.info(f"✓ Neural processing: {len(neural_metadata['layer_outputs'])} layers")
    logger.info(f"✓ Adaptive weighting: α converged to {alpha_history[-1]:.3f}")
    logger.info(f"✓ Bias modeling: {len(bias_report['applied_biases'])} biases applied")
    logger.info(f"✓ Constraint checking: {len(violations)} violations detected")
    logger.info(f"✓ Final performance: MSE={mse_loss.item():.4f}, Correlation={correlation.item():.3f}")
    logger.info("="*50)
    
    return {
        'final_output': biased_output,
        'performance_metrics': {
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'correlation': correlation.item()
        },
        'adaptation_history': {
            'alpha_history': alpha_history,
            'performance_history': performance_history
        },
        'constraint_violations': len(violations),
        'bias_effects': bias_report['total_bias_effect']
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final MSE Loss: {results['performance_metrics']['mse_loss']:.4f}")
        print(f"Final Correlation: {results['performance_metrics']['correlation']:.3f}")
        print(f"Alpha Evolution: {results['adaptation_history']['alpha_history'][0]:.3f} → {results['adaptation_history']['alpha_history'][-1]:.3f}")
        print(f"Constraint Violations: {results['constraint_violations']}")
        print(f"Bias Effects: {results['bias_effects']:.3f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
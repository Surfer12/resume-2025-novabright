"""
Meta-Optimization Framework: Bridging Minds and Machines

A comprehensive framework for cognitive-inspired deep learning optimization.
"""

__version__ = "0.1.0"
__author__ = "Ryan Oates"
__email__ = "ryan.oates@ucsb.edu"

from .core.meta_optimization import MetaOptimizer
from .core.dynamic_integration import DynamicIntegrator
from .core.cognitive_regularization import CognitiveRegularizer
from .core.bias_modeling import BiasModeler

__all__ = [
    "MetaOptimizer",
    "DynamicIntegrator", 
    "CognitiveRegularizer",
    "BiasModeler",
]
"""
Core algorithms and frameworks for meta-optimization.

This module implements the fundamental mathematical framework:
Ψ(x) = ∫[α(t)S(x) + (1-α(t))N(x)] × exp(-[λ₁R_cognitive + λ₂R_efficiency]) × P(H|E,β) dt
"""

from .meta_optimization import MetaOptimizer
from .dynamic_integration import DynamicIntegrator
from .cognitive_regularization import CognitiveRegularizer
from .bias_modeling import BiasModeler

__all__ = [
    "MetaOptimizer",
    "DynamicIntegrator",
    "CognitiveRegularizer", 
    "BiasModeler",
]
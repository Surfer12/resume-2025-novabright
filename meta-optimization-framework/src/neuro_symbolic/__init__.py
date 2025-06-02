"""
Neuro-Symbolic Integration Module

This module implements the core neuro-symbolic integration framework
for the meta-optimization system, combining symbolic reasoning with
neural computation through adaptive weighting mechanisms.
"""

from .symbolic_component import SymbolicComponent
from .neural_component import NeuralComponent
from .adaptive_weighting import AdaptiveWeighting

__all__ = [
    'SymbolicComponent',
    'NeuralComponent', 
    'AdaptiveWeighting'
]
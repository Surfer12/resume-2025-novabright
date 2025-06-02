"""
Utility modules for the meta-optimization framework.

Provides shared functionality for statistical analysis, data processing,
visualization, and failure documentation.
"""

from .statistical_analysis import ConfidenceInterval, EffectSize, StatisticalAnalyzer
from .failure_documentation import FailureDocumenter, FailureMode
from .data_processing import DataProcessor
from .visualization import Visualizer

__all__ = [
    "ConfidenceInterval",
    "EffectSize", 
    "StatisticalAnalyzer",
    "FailureDocumenter",
    "FailureMode",
    "DataProcessor",
    "Visualizer",
]
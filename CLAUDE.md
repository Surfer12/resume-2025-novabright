# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic research repository for cognitive-inspired deep learning optimization targeting top-tier ML conferences (ICML, NeurIPS, ICLR). The project implements a mathematical framework bridging cognitive science and AI: `Ψ(x) = ∫[α(t)S(x) + (1-α(t))N(x)] × exp(-[λ₁R_cognitive + λ₂R_efficiency]) × P(H|E,β) dt`

## Development Commands

### Environment Setup
```bash
cd meta-optimization-framework
pip install -r requirements.txt
pip install -e .  # Editable install for development
```

### Testing
```bash
pytest tests/ --cov=src --cov-report=html  # Run tests with coverage
pytest tests/unit/ -v  # Unit tests only
pytest tests/integration/ -v  # Integration tests only
pytest -m "not slow"  # Skip slow tests
```

### Code Quality
```bash
black .  # Format code (127 char line limit)
flake8 .  # Linting
mypy src/ --ignore-missing-imports  # Type checking
```

### Documentation
```bash
cd docs/
make html  # Build Sphinx documentation
```

### Running the Framework
```bash
python -m src.core.meta_optimization --task n_back --target-improvement 0.19
meta-optimize  # Via console script
cognitive-benchmark  # Run cognitive authenticity tests
```

### Local Research Server
```bash
python start-local-server.py  # Serves research content at http://localhost:8080
```

## Architecture Overview

### Dual Repository Structure
- **Academic Content**: 12 chapters + 4 appendices in `chapters/` and `appendices/`
- **Implementation**: Complete Python package in `meta-optimization-framework/`

### Core Implementation (`meta-optimization-framework/src/`)
- `core/meta_optimization.py`: Main optimization framework implementing the grand unified equation
- `core/dynamic_integration.py`: α-parameter adaptation for symbolic-neural balance
- `core/cognitive_regularization.py`: λ-parameter optimization for cognitive constraints
- `core/bias_modeling.py`: β-parameter bias modeling and correction

### Key Algorithms
- **MetaOptimizer**: Primary optimization loop with 19% ± 8% target improvement
- **CognitiveSymbolicReasoner**: Task-specific symbolic reasoning (N-back, Stroop, planning)
- **CognitiveNeuralProcessor**: Adaptive neural networks for cognitive tasks
- **Dynamic Integration**: Real-time symbolic-neural balance adjustment

### Package Management
- **Primary**: `pyproject.toml` with modern Python packaging
- **Environment**: `pixi.toml` using Pixi package manager with Modular Max integration
- **Dependencies**: PyTorch, NumPy, SciPy, Optuna, Sphinx, Jupyter

### Testing Framework
- **Structure**: `tests/{unit,integration,validation,benchmarks}/`
- **Coverage**: Minimum 80% (100% for core math functions)
- **Markers**: `slow`, `integration`, `performance`, `cognitive`
- **Performance**: Validates 19% accuracy improvement, 12% efficiency gains

### Documentation Standards
- **API Docs**: Sphinx with mathematical notation support
- **Docstrings**: Google-style with comprehensive type hints
- **Academic**: LaTeX support for mathematical equations
- **Research**: Failure Museum for systematic failure documentation

### Quality Requirements
- **Style**: Black formatting, 127-character line limit
- **Types**: MyPy with strict configuration
- **Performance**: Statistical validation with 95% confidence intervals
- **Reproducibility**: Fixed random seeds and experiment tracking

## Development Workflow

1. **Research Phase**: Work in `notebooks/` for exploration
2. **Implementation**: Add algorithms to `src/core/` following existing patterns
3. **Testing**: Write comprehensive tests in `tests/` with proper fixtures
4. **Documentation**: Update both code docs and academic chapters
5. **Validation**: Run full test suite and performance benchmarks

## Special Considerations

- **Academic Standards**: All changes must maintain publication readiness
- **Performance Targets**: Code must meet statistical significance thresholds
- **Mathematical Correctness**: Core algorithms require mathematical validation
- **Cognitive Authenticity**: Implementations must align with cognitive science principles
- **Interdisciplinary Integration**: Balance theoretical rigor with practical implementation
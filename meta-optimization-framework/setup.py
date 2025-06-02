"""
Meta-Optimization Framework: Bridging Minds and Machines
A comprehensive framework for cognitive-inspired deep learning optimization.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="meta-optimization-framework",
    version="0.1.0",
    author="Ryan Oates",
    author_email="ryan.oates@ucsb.edu",
    description="Cognitive-inspired deep learning optimization framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Surfer12/meta-optimization-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "experiments": [
            "wandb>=0.12.0",
            "tensorboard>=2.8.0",
            "psychopy>=2022.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meta-optimize=src.core.meta_optimization:main",
            "cognitive-benchmark=src.evaluation.cognitive_authenticity:main",
            "failure-analyze=src.utils.failure_documentation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt", "*.md"],
    },
    keywords="cognitive-science, deep-learning, optimization, neuro-symbolic, bias-modeling",
    project_urls={
        "Bug Reports": "https://github.com/Surfer12/meta-optimization-framework/issues",
        "Source": "https://github.com/Surfer12/meta-optimization-framework",
        "Documentation": "https://meta-optimization-framework.readthedocs.io/",
    },
)
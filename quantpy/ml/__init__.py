"""
ml
==
Deep-learning helpers for quantitative finance:

  - NeuralNetworkPricer  – neural-network surrogate / fast pricer
  - VolSurfaceInterpolator  – ML-based vol surface completion
  - DeepHedging  – conceptual deep hedging framework
  - RiskPredictionModel  – predict risk sensitivities from market data features
"""

from .suggestions import (
    NeuralNetworkPricer,
    VolSurfaceInterpolator,
    build_training_data,
    DEEP_LEARNING_SUGGESTIONS,
)

__all__ = [
    "NeuralNetworkPricer",
    "VolSurfaceInterpolator",
    "build_training_data",
    "DEEP_LEARNING_SUGGESTIONS",
]

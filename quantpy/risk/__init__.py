"""
risk
====
Scenario analysis and risk sensitivities:
  - RiskEngine  – computes delta, gamma, vega, nu, and parallel/bucket shift risks
"""

from .scenarios import RiskEngine, RiskResult

__all__ = ["RiskEngine", "RiskResult"]

"""
instruments
===========
Pricing of fixed-income derivatives:
  - InterestRateSwap
  - Cap / Floor
  - EuropeanSwaption
  - BermudanSwaption
  - AutoCall (auto-callable structured note)
"""

from .swap import InterestRateSwap
from .cap_floor import Cap, Floor
from .swaption import EuropeanSwaption, BermudanSwaption
from .autocall import AutoCall

__all__ = [
    "InterestRateSwap",
    "Cap",
    "Floor",
    "EuropeanSwaption",
    "BermudanSwaption",
    "AutoCall",
]

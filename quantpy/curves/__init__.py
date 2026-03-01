"""
curves
======
Interest-rate curve construction and interpolation:
  - DiscountCurve  – piecewise-log-linear / cubic-spline discount-factor curve
  - CurveStripper  – bootstrap from MM, futures, FRAs, and swap rates
  - NSSModel       – Nelson-Siegel-Svensson parametric curve
"""

from .discount_curve import DiscountCurve
from .interpolation import (
    linear_interpolate,
    cubic_spline_interpolate,
    log_linear_interpolate,
)
from .nss_model import NSSModel
from .curve_stripper import CurveStripper

__all__ = [
    "DiscountCurve",
    "linear_interpolate",
    "cubic_spline_interpolate",
    "log_linear_interpolate",
    "NSSModel",
    "CurveStripper",
]

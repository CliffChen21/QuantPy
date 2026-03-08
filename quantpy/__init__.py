"""
QuantPy – FIC AI Python Quantitative Finance Library
=====================================================

Modules
-------
market_data : Market data generation (swap rates, MM quotes, vol surfaces)
curves      : Curve stripping and interpolation (cubic spline, NSS)
models      : Hull White and SABR model calibration & pricing
instruments : Swap, Cap/Floor, Swaption, AutoCall
risk        : Scenario analysis (delta, vega, gamma, nu, shift)
ml          : Deep-learning helpers / neural-network surrogate models
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("quantpy")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "market_data",
    "curves",
    "models",
    "instruments",
    "risk",
    "ml",
    "reports",
]

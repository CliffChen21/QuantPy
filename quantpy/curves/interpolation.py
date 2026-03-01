"""
Interpolation utilities for yield-curve construction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline


def linear_interpolate(
    x: Sequence[float],
    y: Sequence[float],
    xi: float,
    extrapolate: bool = True,
) -> float:
    """
    Linear interpolation (and flat extrapolation by default).

    Parameters
    ----------
    x, y : array-like
        Known (x, y) pairs, x must be strictly increasing.
    xi : float
        Query point.
    extrapolate : bool
        If True, use flat extrapolation outside [x[0], x[-1]].

    Returns
    -------
    float
    """
    xarr = np.asarray(x, dtype=float)
    yarr = np.asarray(y, dtype=float)
    if extrapolate:
        xi = float(np.clip(xi, xarr[0], xarr[-1]))
    return float(np.interp(xi, xarr, yarr))


def log_linear_interpolate(
    x: Sequence[float],
    y: Sequence[float],
    xi: float,
    extrapolate: bool = True,
) -> float:
    """
    Log-linear interpolation – standard for discount factors.

    Interpolates ln(y) linearly, so that:
        y(xi) = exp( linear(ln(y), xi) )

    Parameters
    ----------
    x, y : array-like
        Known points; y must be positive.
    xi : float
        Query point.
    extrapolate : bool

    Returns
    -------
    float
    """
    yarr = np.asarray(y, dtype=float)
    log_y = np.log(np.maximum(yarr, 1e-15))
    log_yi = linear_interpolate(x, log_y, xi, extrapolate=extrapolate)
    return float(np.exp(log_yi))


def cubic_spline_interpolate(
    x: Sequence[float],
    y: Sequence[float],
    xi: float,
    bc_type: str = "not-a-knot",
) -> float:
    """
    Cubic-spline interpolation via scipy.

    Parameters
    ----------
    x, y : array-like
        Known (x, y) pairs.
    xi : float
        Query point.
    bc_type : str
        Boundary condition passed to :class:`scipy.interpolate.CubicSpline`.

    Returns
    -------
    float
    """
    cs = CubicSpline(np.asarray(x, dtype=float), np.asarray(y, dtype=float), bc_type=bc_type)
    return float(cs(xi))

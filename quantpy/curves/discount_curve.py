"""
DiscountCurve
=============
A discount-factor curve built from (time, discount-factor) pairs.
Supports log-linear (default) and cubic-spline interpolation.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from .interpolation import log_linear_interpolate, cubic_spline_interpolate


class DiscountCurve:
    """
    Piecewise-smooth discount factor curve.

    Parameters
    ----------
    times : list of float
        Pillar times in years.  Must start with 0 (or include it).
    discount_factors : list of float
        Discount factors DF(0, T) for each pillar.  DF(0,0) = 1.0.
    method : str
        Interpolation method: "log_linear" (default) or "cubic_spline".
    """

    def __init__(
        self,
        times: Sequence[float],
        discount_factors: Sequence[float],
        method: str = "log_linear",
    ) -> None:
        times = list(times)
        dfs = list(discount_factors)
        if not times or times[0] != 0.0:
            times.insert(0, 0.0)
            dfs.insert(0, 1.0)
        self._times = np.array(times, dtype=float)
        self._dfs = np.array(dfs, dtype=float)
        if method not in ("log_linear", "cubic_spline"):
            raise ValueError("method must be 'log_linear' or 'cubic_spline'")
        self._method = method
        if method == "cubic_spline":
            log_dfs = np.log(np.maximum(self._dfs, 1e-15))
            self._cs = CubicSpline(self._times, log_dfs, bc_type="not-a-knot")

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def discount_factor(self, t: float) -> float:
        """
        Return the discount factor DF(0, t).

        Parameters
        ----------
        t : float  Time in years.

        Returns
        -------
        float
        """
        if t <= 0.0:
            return 1.0
        if self._method == "cubic_spline":
            return float(np.exp(self._cs(t)))
        return log_linear_interpolate(self._times, self._dfs, t)

    def zero_rate(self, t: float, compounding: str = "continuous") -> float:
        """
        Continuously compounded (default) or annually compounded zero rate.

        Parameters
        ----------
        t : float
        compounding : str  "continuous" | "annual"

        Returns
        -------
        float
        """
        if t <= 1e-10:
            return self.zero_rate(1e-4, compounding)
        df = self.discount_factor(t)
        df = max(df, 1e-15)
        r_cont = -math.log(df) / t
        if compounding == "continuous":
            return r_cont
        # convert to annual compounding: (1+r)^t = exp(r_cont * t)
        return math.exp(r_cont) - 1.0

    def forward_rate(
        self,
        t1: float,
        t2: float,
        compounding: str = "continuous",
    ) -> float:
        """
        Simply-compounded or continuously-compounded forward rate F(t1, t2).

        Parameters
        ----------
        t1, t2 : float  Start and end times (years).
        compounding : str  "continuous" | "simple"

        Returns
        -------
        float
        """
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        dt = t2 - t1
        if compounding == "continuous":
            return math.log(df1 / max(df2, 1e-15)) / dt
        # simply-compounded
        return (df1 / max(df2, 1e-15) - 1.0) / dt

    def par_swap_rate(
        self,
        maturity: float,
        payment_frequency: float = 0.5,
    ) -> float:
        """
        Compute the par fixed rate of a vanilla interest-rate swap.

        Parameters
        ----------
        maturity : float    Swap maturity in years.
        payment_frequency : float  Year-fraction between coupon payments (0.5 = semi-annual).

        Returns
        -------
        float
        """
        n = int(round(maturity / payment_frequency))
        times = [i * payment_frequency for i in range(1, n + 1)]
        annuity = sum(payment_frequency * self.discount_factor(t) for t in times)
        if annuity < 1e-12:
            return 0.0
        df_mat = self.discount_factor(maturity)
        return (1.0 - df_mat) / annuity

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._times)
        return (
            f"DiscountCurve(pillars={n}, method={self._method!r}, "
            f"maxTenor={self._times[-1]:.1f}Y)"
        )

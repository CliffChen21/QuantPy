"""
Curve Stripper
==============
Bootstrap a piecewise-smooth discount curve from:
  * Money-market (deposit) quotes
  * Interest-rate futures
  * Forward-rate agreements (FRAs)
  * Par-swap rates

The bootstrapped pillars are stored as a :class:`~quantpy.curves.DiscountCurve`.

Convention: ACT/360 for MM deposits, quarterly futures, ACT/360 or 30/360
for swaps (simplified to ACT/365 fixed year fractions here).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from quantpy.market_data.generators import (
    FRAQuote,
    FutureQuote,
    MoneyMarketQuote,
    SwapRateQuote,
)
from .discount_curve import DiscountCurve


class CurveStripper:
    """
    Bootstrap an OIS / LIBOR discount curve from market instruments.

    Parameters
    ----------
    method : str
        Interpolation method for the resulting curve:
        "log_linear" (default) or "cubic_spline".
    """

    def __init__(self, method: str = "log_linear") -> None:
        if method not in ("log_linear", "cubic_spline"):
            raise ValueError("method must be 'log_linear' or 'cubic_spline'")
        self._method = method
        self._times: List[float] = [0.0]
        self._dfs: List[float] = [1.0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_curve(self) -> DiscountCurve:
        return DiscountCurve(self._times, self._dfs, method=self._method)

    def _add_pillar(self, t: float, df: float) -> None:
        """Insert (t, df) keeping times sorted."""
        if t in self._times:
            idx = self._times.index(t)
            self._dfs[idx] = df
            return
        # find insertion point
        for i, ti in enumerate(self._times):
            if ti > t:
                self._times.insert(i, t)
                self._dfs.insert(i, df)
                return
        self._times.append(t)
        self._dfs.append(df)

    # ------------------------------------------------------------------
    # Bootstrapping methods
    # ------------------------------------------------------------------

    def strip_mm(self, quotes: Sequence[MoneyMarketQuote]) -> "CurveStripper":
        """
        Bootstrap money-market deposit quotes.

        Assumes ACT/360 day count (approximated by year fractions).

        Parameters
        ----------
        quotes : list of MoneyMarketQuote

        Returns
        -------
        self  (for chaining)
        """
        for q in sorted(quotes, key=lambda x: x.tenor):
            t = q.tenor
            # DF = 1 / (1 + r * t)  (simple interest)
            df = 1.0 / (1.0 + q.rate * t)
            self._add_pillar(t, df)
        return self

    def strip_futures(self, quotes: Sequence[FutureQuote]) -> "CurveStripper":
        """
        Bootstrap interest-rate futures (convexity-adjusted if needed).

        We use the implied forward rate directly (zero convexity adjustment
        for simplicity).

        Parameters
        ----------
        quotes : list of FutureQuote

        Returns
        -------
        self
        """
        for q in sorted(quotes, key=lambda x: x.start):
            t1, t2 = q.start, q.end
            dt = t2 - t1
            # If t1 < max known pillar we already have DF(t1)
            curve = self._current_curve()
            df1 = curve.discount_factor(t1)
            # DF(t1, t2) from futures implied fwd rate
            df_fwd = 1.0 / (1.0 + q.implied_rate * dt)
            df2 = df1 * df_fwd
            self._add_pillar(t2, df2)
        return self

    def strip_fra(self, quotes: Sequence[FRAQuote]) -> "CurveStripper":
        """
        Bootstrap FRA quotes.

        Parameters
        ----------
        quotes : list of FRAQuote

        Returns
        -------
        self
        """
        for q in sorted(quotes, key=lambda x: x.end):
            t1, t2 = q.start, q.end
            dt = t2 - t1
            curve = self._current_curve()
            df1 = curve.discount_factor(t1)
            df2 = df1 / (1.0 + q.rate * dt)
            self._add_pillar(t2, df2)
        return self

    def strip_swaps(
        self,
        quotes: Sequence[SwapRateQuote],
        payment_frequency: float = 0.5,
    ) -> "CurveStripper":
        """
        Bootstrap par-swap rates via sequential root-finding.

        For each new swap maturity T_n:
          Solve for DF(T_n) such that:
            par_rate = (1 - DF(T_n)) / (Σ δ_i * DF(T_i))

        Parameters
        ----------
        quotes : list of SwapRateQuote
        payment_frequency : float  Year-fraction between coupons (0.5 = semi-annual).

        Returns
        -------
        self
        """
        for q in sorted(quotes, key=lambda x: x.tenor):
            mat = q.tenor
            par = q.rate
            n = int(round(mat / payment_frequency))
            coupon_times = [i * payment_frequency for i in range(1, n + 1)]

            def residual(df_last: float) -> float:
                # Temporarily add pillar for current maturity
                curve = self._current_curve()
                # annuity up to last coupon using current pillars
                ann = 0.0
                for k, ct in enumerate(coupon_times):
                    if ct < mat - 1e-8:
                        ann += payment_frequency * curve.discount_factor(ct)
                    else:
                        ann += payment_frequency * df_last
                pv01 = ann
                if pv01 < 1e-12:
                    return 0.0
                par_model = (1.0 - df_last) / pv01
                return par_model - par

            # Bracket: df must be positive and < 1
            try:
                df_solution = brentq(residual, 1e-6, 1.0 - 1e-8, xtol=1e-12, maxiter=200)
            except ValueError:
                # Fall back: use approximation
                curve = self._current_curve()
                ann_approx = sum(
                    payment_frequency * curve.discount_factor(ct)
                    for ct in coupon_times[:-1]
                )
                # (1 - df) = par * (ann_approx + freq * df)
                # 1 - df = par*ann_approx + par*freq*df
                # 1 - par*ann_approx = df*(1 + par*freq)
                df_solution = (1.0 - par * ann_approx) / (1.0 + par * payment_frequency)

            self._add_pillar(mat, max(df_solution, 1e-10))
        return self

    def build(self) -> DiscountCurve:
        """
        Return the bootstrapped :class:`DiscountCurve`.

        Returns
        -------
        DiscountCurve
        """
        return DiscountCurve(self._times, self._dfs, method=self._method)

    # ------------------------------------------------------------------
    # Convenience class method
    # ------------------------------------------------------------------

    @classmethod
    def from_market_data(
        cls,
        mm_quotes: Optional[Sequence[MoneyMarketQuote]] = None,
        future_quotes: Optional[Sequence[FutureQuote]] = None,
        fra_quotes: Optional[Sequence[FRAQuote]] = None,
        swap_quotes: Optional[Sequence[SwapRateQuote]] = None,
        method: str = "log_linear",
    ) -> DiscountCurve:
        """
        Convenience factory: strip the curve from all available instruments.

        Parameters
        ----------
        mm_quotes : list of MoneyMarketQuote, optional
        future_quotes : list of FutureQuote, optional
        fra_quotes : list of FRAQuote, optional
        swap_quotes : list of SwapRateQuote, optional
        method : str

        Returns
        -------
        DiscountCurve
        """
        stripper = cls(method=method)
        if mm_quotes:
            stripper.strip_mm(mm_quotes)
        if future_quotes:
            stripper.strip_futures(future_quotes)
        if fra_quotes:
            stripper.strip_fra(fra_quotes)
        if swap_quotes:
            stripper.strip_swaps(swap_quotes)
        return stripper.build()

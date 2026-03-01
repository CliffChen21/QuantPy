"""
Interest Rate Swap
==================
Vanilla fixed-float interest rate swap pricing.

Conventions
-----------
* Pay fixed / receive floating (payer swap) or vice-versa.
* Semi-annual fixed leg, quarterly float leg (defaults).
* Simplified ACT/365 day-count.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from quantpy.curves.discount_curve import DiscountCurve


class InterestRateSwap:
    """
    Vanilla interest rate swap.

    Parameters
    ----------
    maturity : float
        Swap maturity (years from today).
    fixed_rate : float
        Fixed coupon rate (decimal).  Set to None for par pricing.
    notional : float
        Notional amount.
    is_payer : bool
        True = pay fixed / receive floating (payer swap).
    fixed_frequency : float
        Year-fraction between fixed coupons (default 0.5 = semi-annual).
    float_frequency : float
        Year-fraction between float coupons (default 0.25 = quarterly).
    start : float
        Swap start (years from today, default = 0 = spot starting).
    """

    def __init__(
        self,
        maturity: float,
        fixed_rate: float,
        notional: float = 1_000_000.0,
        is_payer: bool = True,
        fixed_frequency: float = 0.5,
        float_frequency: float = 0.25,
        start: float = 0.0,
    ) -> None:
        self.maturity = maturity
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.is_payer = is_payer
        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency
        self.start = start

    # ------------------------------------------------------------------
    # Payment schedules
    # ------------------------------------------------------------------

    def fixed_payment_times(self) -> List[float]:
        """Return list of fixed coupon payment times."""
        n = int(round((self.maturity - self.start) / self.fixed_frequency))
        return [self.start + (i + 1) * self.fixed_frequency for i in range(n)]

    def float_payment_times(self) -> List[float]:
        """Return list of float coupon payment times."""
        n = int(round((self.maturity - self.start) / self.float_frequency))
        return [self.start + (i + 1) * self.float_frequency for i in range(n)]

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def fixed_leg_pv(self, curve: DiscountCurve) -> float:
        """
        PV of the fixed leg.

        PV_fixed = rate * Σ_i δ_i · DF(T_i) * N
        """
        coupon = self.fixed_rate * self.fixed_frequency * self.notional
        pv = sum(coupon * curve.discount_factor(t) for t in self.fixed_payment_times())
        # Add notional at maturity (if computing par)
        return pv

    def float_leg_pv(self, curve: DiscountCurve) -> float:
        """
        PV of the floating leg using the replication argument.

        PV_float = N * [DF(start) – DF(maturity)]
        (under simple replication, ignoring spreads).
        """
        df_start = curve.discount_factor(self.start)
        df_mat = curve.discount_factor(self.maturity)
        return self.notional * (df_start - df_mat)

    def npv(self, curve: DiscountCurve) -> float:
        """
        Net present value of the swap from payer's perspective.

        NPV_payer = PV_float – PV_fixed
        NPV_receiver = PV_fixed – PV_float

        Parameters
        ----------
        curve : DiscountCurve

        Returns
        -------
        float
        """
        fixed_pv = self.fixed_leg_pv(curve)
        float_pv = self.float_leg_pv(curve)
        if self.is_payer:
            return float_pv - fixed_pv
        return fixed_pv - float_pv

    def par_rate(self, curve: DiscountCurve) -> float:
        """
        Compute the par (fair) fixed rate.

        par = PV_float / Annuity
            = (DF(start) – DF(maturity)) / Σ_i δ_i · DF(T_i)

        Parameters
        ----------
        curve : DiscountCurve

        Returns
        -------
        float
        """
        annuity = sum(
            self.fixed_frequency * curve.discount_factor(t)
            for t in self.fixed_payment_times()
        )
        if annuity < 1e-12:
            return 0.0
        df_start = curve.discount_factor(self.start)
        df_mat = curve.discount_factor(self.maturity)
        return (df_start - df_mat) / annuity

    def annuity(self, curve: DiscountCurve) -> float:
        """
        Dollar value of 1 bp (DV01) annuity.

        annuity = Σ_i δ_i · DF(T_i) · N
        """
        return sum(
            self.fixed_frequency * curve.discount_factor(t)
            for t in self.fixed_payment_times()
        ) * self.notional

    def dv01(self, curve: DiscountCurve, bump_size: float = 1e-4) -> float:
        """
        DV01: sensitivity of NPV to a parallel shift of 1 bp in rates.

        Parameters
        ----------
        curve : DiscountCurve
        bump_size : float  Bump size (default 1e-4 = 1 bp).

        Returns
        -------
        float
        """
        from quantpy.curves.discount_curve import DiscountCurve
        # Bump all discount factors
        times = curve._times
        dfs_up = curve._dfs * [
            math.exp(-bump_size * t) for t in times
        ]
        curve_up = DiscountCurve(list(times), list(dfs_up), method=curve._method)
        npv_up = self.npv(curve_up)
        dfs_dn = curve._dfs * [
            math.exp(+bump_size * t) for t in times
        ]
        curve_dn = DiscountCurve(list(times), list(dfs_dn), method=curve._method)
        npv_dn = self.npv(curve_dn)
        return (npv_up - npv_dn) / 2.0

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        direction = "Payer" if self.is_payer else "Receiver"
        return (
            f"InterestRateSwap({direction}, maturity={self.maturity}Y, "
            f"fixedRate={self.fixed_rate:.4%}, notional={self.notional:,.0f})"
        )

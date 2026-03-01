"""
Cap and Floor Instruments
=========================
Vanilla interest-rate cap and floor pricing using:
  * Black (market) model
  * Hull-White model (delegated to HullWhiteModel)
"""

from __future__ import annotations

import math
from typing import Optional

from scipy.stats import norm

from quantpy.curves.discount_curve import DiscountCurve


class Cap:
    """
    Interest-rate cap (portfolio of caplets).

    A cap pays max(L(T_{i-1}, T_i) – K, 0) * δ * N at each payment date.

    Parameters
    ----------
    maturity : float     Cap maturity (years).
    strike : float       Cap rate (decimal).
    notional : float
    payment_frequency : float  0.25 = quarterly (default).
    start : float        Cap start (years, default 0).
    """

    def __init__(
        self,
        maturity: float,
        strike: float,
        notional: float = 1_000_000.0,
        payment_frequency: float = 0.25,
        start: float = 0.0,
    ) -> None:
        self.maturity = maturity
        self.strike = strike
        self.notional = notional
        self.payment_frequency = payment_frequency
        self.start = start

    # ------------------------------------------------------------------
    # Black model pricing
    # ------------------------------------------------------------------

    def black_price(self, curve: DiscountCurve, vol: float) -> float:
        """
        Price the cap using Black's formula for each caplet.

        Parameters
        ----------
        curve : DiscountCurve
        vol : float  Flat Black implied vol.

        Returns
        -------
        float  Present value.
        """
        total = 0.0
        n = int(round((self.maturity - self.start) / self.payment_frequency))
        for i in range(n):
            t_reset = self.start + i * self.payment_frequency
            t_pay = self.start + (i + 1) * self.payment_frequency
            total += self._black_caplet(curve, vol, t_reset, t_pay, is_cap=True)
        return total

    def hull_white_price(self, hw_model) -> float:
        """
        Price using Hull-White caplet formula.

        Parameters
        ----------
        hw_model : HullWhiteModel

        Returns
        -------
        float
        """
        return hw_model.cap_price(
            self.maturity, self.strike, self.payment_frequency, self.notional, is_cap=True
        )

    def _black_caplet(
        self,
        curve: DiscountCurve,
        vol: float,
        t_reset: float,
        t_pay: float,
        is_cap: bool = True,
    ) -> float:
        """Price a single caplet/floorlet using Black's formula."""
        delta = t_pay - t_reset
        fwd = curve.forward_rate(t_reset, t_pay, compounding="simple")
        df = curve.discount_factor(t_pay)

        if t_reset <= 1e-10 or vol < 1e-10:
            if is_cap:
                return max(fwd - self.strike, 0.0) * delta * self.notional * df
            return max(self.strike - fwd, 0.0) * delta * self.notional * df

        sqrt_T = math.sqrt(t_reset)
        d1 = (math.log(max(fwd / max(self.strike, 1e-8), 1e-15)) + 0.5 * vol ** 2 * t_reset) / (vol * sqrt_T)
        d2 = d1 - vol * sqrt_T

        if is_cap:
            price = df * delta * self.notional * (fwd * norm.cdf(d1) - self.strike * norm.cdf(d2))
        else:
            price = df * delta * self.notional * (self.strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))
        return max(price, 0.0)

    def implied_vol(
        self,
        curve: DiscountCurve,
        market_price: float,
        vol_init: float = 0.30,
    ) -> float:
        """
        Compute the flat Black implied vol from a market cap price.

        Parameters
        ----------
        curve : DiscountCurve
        market_price : float
        vol_init : float  Initial guess.

        Returns
        -------
        float
        """
        from scipy.optimize import brentq

        def residual(v: float) -> float:
            return self.black_price(curve, v) - market_price

        try:
            return brentq(residual, 1e-6, 10.0, xtol=1e-8, maxiter=200)
        except ValueError:
            return vol_init

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Cap(maturity={self.maturity}Y, strike={self.strike:.2%}, "
            f"notional={self.notional:,.0f})"
        )


class Floor(Cap):
    """
    Interest-rate floor (portfolio of floorlets).

    Identical to :class:`Cap` but prices floorlets.
    """

    def black_price(self, curve: DiscountCurve, vol: float) -> float:
        """Price using Black's formula (floorlets)."""
        total = 0.0
        n = int(round((self.maturity - self.start) / self.payment_frequency))
        for i in range(n):
            t_reset = self.start + i * self.payment_frequency
            t_pay = self.start + (i + 1) * self.payment_frequency
            total += self._black_caplet(curve, vol, t_reset, t_pay, is_cap=False)
        return total

    def hull_white_price(self, hw_model) -> float:
        return hw_model.cap_price(
            self.maturity, self.strike, self.payment_frequency, self.notional, is_cap=False
        )

    def __repr__(self) -> str:
        return (
            f"Floor(maturity={self.maturity}Y, strike={self.strike:.2%}, "
            f"notional={self.notional:,.0f})"
        )

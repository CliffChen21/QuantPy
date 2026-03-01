"""
Swaption Instruments
====================
European and Bermudan swaption pricing.

European swaption: Black model and Hull-White (Jamshidian decomposition).
Bermudan swaption:  Hull-White trinomial tree.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from scipy.stats import norm

from quantpy.curves.discount_curve import DiscountCurve


class EuropeanSwaption:
    """
    European swaption: option to enter an interest rate swap.

    Parameters
    ----------
    expiry : float         Option expiry (years).
    swap_maturity : float  Underlying swap maturity from today.
    strike : float         Fixed rate.
    notional : float
    is_payer : bool        True = option to pay fixed (payer swaption).
    payment_frequency : float  Fixed leg payment frequency (0.5 = semi-annual).
    """

    def __init__(
        self,
        expiry: float,
        swap_maturity: float,
        strike: float,
        notional: float = 1_000_000.0,
        is_payer: bool = True,
        payment_frequency: float = 0.5,
    ) -> None:
        if expiry >= swap_maturity:
            raise ValueError("expiry must be before swap_maturity")
        self.expiry = expiry
        self.swap_maturity = swap_maturity
        self.strike = strike
        self.notional = notional
        self.is_payer = is_payer
        self.payment_frequency = payment_frequency

    def _coupon_times(self) -> List[float]:
        n = int(round((self.swap_maturity - self.expiry) / self.payment_frequency))
        return [self.expiry + (i + 1) * self.payment_frequency for i in range(n)]

    def annuity(self, curve: DiscountCurve) -> float:
        """PV01 / annuity of the underlying swap."""
        return sum(
            self.payment_frequency * curve.discount_factor(t)
            for t in self._coupon_times()
        ) * self.notional

    def black_price(self, curve: DiscountCurve, vol: float) -> float:
        """
        Price using Black's swaption formula.

        σ_Black convention: annuity * [S·N(d1) – K·N(d2)]  (payer).

        Parameters
        ----------
        curve : DiscountCurve
        vol : float  Black implied vol (lognormal).

        Returns
        -------
        float
        """
        S = curve.par_swap_rate(
            self.swap_maturity, payment_frequency=self.payment_frequency
        )
        ann = self.annuity(curve)
        T = self.expiry

        if T <= 1e-10 or vol < 1e-10:
            if self.is_payer:
                return max(S - self.strike, 0.0) * ann
            return max(self.strike - S, 0.0) * ann

        sqrt_T = math.sqrt(T)
        d1 = (math.log(max(S / max(self.strike, 1e-8), 1e-15)) + 0.5 * vol ** 2 * T) / (vol * sqrt_T)
        d2 = d1 - vol * sqrt_T

        if self.is_payer:
            return ann * (S * norm.cdf(d1) - self.strike * norm.cdf(d2))
        return ann * (self.strike * norm.cdf(-d2) - S * norm.cdf(-d1))

    def hull_white_price(self, hw_model) -> float:
        """
        Price using Hull-White Jamshidian decomposition.

        Parameters
        ----------
        hw_model : HullWhiteModel

        Returns
        -------
        float
        """
        return hw_model.european_swaption_price(
            expiry=self.expiry,
            swap_maturity=self.swap_maturity,
            strike=self.strike,
            notional=self.notional,
            is_payer=self.is_payer,
            payment_frequency=self.payment_frequency,
        )

    def implied_vol(
        self,
        curve: DiscountCurve,
        market_price: float,
        vol_init: float = 0.30,
    ) -> float:
        """
        Back out Black implied vol from a market swaption price.

        Parameters
        ----------
        curve : DiscountCurve
        market_price : float
        vol_init : float

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

    def __repr__(self) -> str:
        kind = "Payer" if self.is_payer else "Receiver"
        return (
            f"EuropeanSwaption({kind}, expiry={self.expiry}Y, "
            f"swapMaturity={self.swap_maturity}Y, strike={self.strike:.4%})"
        )


class BermudanSwaption:
    """
    Bermudan swaption: exercisable on multiple dates.

    Parameters
    ----------
    expiries : list of float  Exercise dates (years).
    swap_maturity : float
    strike : float
    notional : float
    is_payer : bool
    payment_frequency : float
    """

    def __init__(
        self,
        expiries: Sequence[float],
        swap_maturity: float,
        strike: float,
        notional: float = 1_000_000.0,
        is_payer: bool = True,
        payment_frequency: float = 0.5,
    ) -> None:
        self.expiries = list(expiries)
        self.swap_maturity = swap_maturity
        self.strike = strike
        self.notional = notional
        self.is_payer = is_payer
        self.payment_frequency = payment_frequency

    def hull_white_price(self, hw_model, n_time_steps: int = 100) -> float:
        """
        Price via Hull-White trinomial tree.

        Parameters
        ----------
        hw_model : HullWhiteModel
        n_time_steps : int

        Returns
        -------
        float
        """
        return hw_model.bermudan_swaption_price(
            expiries=self.expiries,
            swap_maturity=self.swap_maturity,
            strike=self.strike,
            notional=self.notional,
            is_payer=self.is_payer,
            payment_frequency=self.payment_frequency,
            n_time_steps=n_time_steps,
        )

    def lower_bound(self, hw_model) -> float:
        """
        Lower bound: max of all European swaption prices.

        (A Bermudan swaption is always worth at least the most valuable
        European swaption co-terminal with one of the exercise dates.)
        """
        best = 0.0
        for exp in self.expiries:
            if exp >= self.swap_maturity:
                continue
            eur = EuropeanSwaption(
                expiry=exp,
                swap_maturity=self.swap_maturity,
                strike=self.strike,
                notional=self.notional,
                is_payer=self.is_payer,
                payment_frequency=self.payment_frequency,
            )
            price = eur.hull_white_price(hw_model)
            best = max(best, price)
        return best

    def __repr__(self) -> str:
        kind = "Payer" if self.is_payer else "Receiver"
        return (
            f"BermudanSwaption({kind}, exercises={len(self.expiries)}, "
            f"swapMaturity={self.swap_maturity}Y, strike={self.strike:.4%})"
        )

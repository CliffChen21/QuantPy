"""
Market Data Generation
======================
Provides realistic simulated market quotes for:
  * Swap rates (1Y – 30Y)
  * Money-market deposits (O/N – 12M)
  * Interest-rate futures (IMM dates)
  * Forward-rate agreements (FRA)
  * Interest-rate cap/floor volatility surface
  * FX option volatility surface (delta-space)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SwapRateQuote:
    """A par-swap rate quote for a given tenor."""

    tenor: float        # years
    rate: float         # decimal (e.g. 0.025 = 2.5 %)
    currency: str = "USD"

    def __repr__(self) -> str:
        return f"SwapRateQuote(tenor={self.tenor}Y, rate={self.rate:.4%}, ccy={self.currency})"


@dataclass
class MoneyMarketQuote:
    """A money-market deposit quote."""

    tenor: float        # years
    rate: float         # decimal
    currency: str = "USD"

    def __repr__(self) -> str:
        return f"MMQuote(tenor={self.tenor}Y, rate={self.rate:.4%}, ccy={self.currency})"


@dataclass
class FutureQuote:
    """An interest-rate future quote (e.g. Eurodollar / SOFR future)."""

    start: float        # years from today
    end: float          # years from today
    price: float        # e.g. 97.50 => implied rate 2.50 %
    currency: str = "USD"

    @property
    def implied_rate(self) -> float:
        return (100.0 - self.price) / 100.0

    def __repr__(self) -> str:
        return (
            f"FutureQuote(start={self.start:.2f}Y, end={self.end:.2f}Y, "
            f"price={self.price:.4f}, impliedRate={self.implied_rate:.4%})"
        )


@dataclass
class FRAQuote:
    """A forward-rate agreement (FRA) quote."""

    start: float        # years from today
    end: float          # years from today
    rate: float         # decimal
    currency: str = "USD"

    def __repr__(self) -> str:
        return (
            f"FRAQuote(start={self.start:.2f}Y, end={self.end:.2f}Y, "
            f"rate={self.rate:.4%})"
        )


@dataclass
class VolatilityQuote:
    """An interest-rate cap/floor implied-volatility quote."""

    tenor: float        # cap maturity (years)
    strike: float       # decimal (e.g. 0.02 = 2 %)
    vol: float          # log-normal implied vol (decimal, e.g. 0.30 = 30 %)
    vol_type: str = "lognormal"   # "lognormal" | "normal"

    def __repr__(self) -> str:
        return (
            f"VolQuote(tenor={self.tenor}Y, K={self.strike:.2%}, "
            f"vol={self.vol:.2%})"
        )


@dataclass
class FXVolatilityQuote:
    """An FX option implied-volatility quote in delta space."""

    expiry: float       # years
    delta: float        # e.g. 0.25, 0.50 (ATM), 0.75
    vol: float          # decimal
    currency_pair: str = "EURUSD"

    def __repr__(self) -> str:
        return (
            f"FXVolQuote(expiry={self.expiry}Y, delta={self.delta:.0%}, "
            f"vol={self.vol:.2%}, pair={self.currency_pair})"
        )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class MarketDataGenerator:
    """
    Generates a self-consistent set of simulated market data.

    The generated data follows typical USD interest-rate market
    conventions and can be used for curve stripping and model
    calibration demonstrations.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    base_rate : float
        Approximate short-rate level (default 4 % – typical recent USD).
    """

    # Standard swap tenors in years
    SWAP_TENORS: List[float] = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30]

    # Money-market tenors in years (overnight, 1W, 1M, 3M, 6M, 12M)
    MM_TENORS: List[float] = [
        1 / 360,   # O/N
        7 / 360,   # 1W
        1 / 12,    # 1M
        3 / 12,    # 3M
        6 / 12,    # 6M
        1.0,       # 12M
    ]

    # IMM-style futures: 4 quarterly contracts starting at ~3M
    FUTURE_STARTS: List[float] = [0.25, 0.50, 0.75, 1.00]

    # Cap tenors for vol surface
    CAP_TENORS: List[float] = [1, 2, 3, 5, 7, 10, 15, 20]

    # Cap strikes relative to ATM
    CAP_STRIKES_OFFSET: List[float] = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]

    # FX option expiries
    FX_EXPIRIES: List[float] = [1 / 12, 3 / 12, 6 / 12, 1, 2, 5]

    def __init__(
        self,
        seed: int = 42,
        base_rate: float = 0.04,
        currency: str = "USD",
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.base_rate = base_rate
        self.currency = currency

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nss_rate(
        self,
        tau: float,
        beta0: float = None,
        beta1: float = None,
        beta2: float = None,
        beta3: float = None,
        lambda1: float = 1.5,
        lambda2: float = 0.5,
    ) -> float:
        """Nelson-Siegel-Svensson curve used to generate realistic rates."""
        b0 = beta0 if beta0 is not None else self.base_rate
        b1 = beta1 if beta1 is not None else -0.005
        b2 = beta2 if beta2 is not None else 0.02
        b3 = beta3 if beta3 is not None else -0.01
        if tau < 1e-6:
            return b0 + b1
        e1 = math.exp(-tau / lambda1)
        e2 = math.exp(-tau / lambda2)
        t1 = (1 - e1) / (tau / lambda1)
        t2 = (1 - e2) / (tau / lambda2)
        return b0 + b1 * t1 + b2 * (t1 - e1) + b3 * (t2 - e2)

    def _add_noise(self, value: float, scale: float = 0.0005) -> float:
        """Add small random noise to a rate."""
        return value + float(self.rng.normal(0, scale))

    # ------------------------------------------------------------------
    # Public generators
    # ------------------------------------------------------------------

    def generate_swap_rates(self) -> List[SwapRateQuote]:
        """
        Generate par-swap rates for standard tenors (1Y – 30Y).

        Returns
        -------
        list of SwapRateQuote
        """
        quotes = []
        for tenor in self.SWAP_TENORS:
            rate = self._nss_rate(tenor)
            rate = max(self._add_noise(rate, scale=0.0003), 0.0001)
            quotes.append(SwapRateQuote(tenor=tenor, rate=rate, currency=self.currency))
        return quotes

    def generate_mm_quotes(self) -> List[MoneyMarketQuote]:
        """
        Generate money-market deposit quotes for short tenors.

        Returns
        -------
        list of MoneyMarketQuote
        """
        quotes = []
        for tenor in self.MM_TENORS:
            rate = self._nss_rate(tenor)
            rate = max(self._add_noise(rate, scale=0.0002), 0.0001)
            quotes.append(MoneyMarketQuote(tenor=tenor, rate=rate, currency=self.currency))
        return quotes

    def generate_future_quotes(self) -> List[FutureQuote]:
        """
        Generate interest-rate future quotes (quarterly IMM contracts).

        Returns
        -------
        list of FutureQuote
        """
        quotes = []
        for start in self.FUTURE_STARTS:
            end = start + 0.25
            fwd_rate = self._nss_rate((start + end) / 2)
            fwd_rate = max(self._add_noise(fwd_rate, scale=0.0003), 0.0001)
            price = 100.0 * (1 - fwd_rate)
            quotes.append(
                FutureQuote(start=start, end=end, price=price, currency=self.currency)
            )
        return quotes

    def generate_fra_quotes(self) -> List[FRAQuote]:
        """
        Generate FRA quotes for standard periods (e.g. 3x6, 6x9 ...).

        Returns
        -------
        list of FRAQuote
        """
        fra_periods = [
            (0.25, 0.50),
            (0.50, 0.75),
            (0.75, 1.00),
            (1.00, 1.25),
            (1.25, 1.50),
            (1.50, 1.75),
            (1.75, 2.00),
        ]
        quotes = []
        for start, end in fra_periods:
            fwd_rate = self._nss_rate((start + end) / 2)
            fwd_rate = max(self._add_noise(fwd_rate, scale=0.0003), 0.0001)
            quotes.append(
                FRAQuote(start=start, end=end, rate=fwd_rate, currency=self.currency)
            )
        return quotes

    def generate_rates_vol_surface(self) -> List[VolatilityQuote]:
        """
        Generate an interest-rate cap/floor implied-volatility surface.

        Volatility follows a typical smile pattern: higher vol for
        out-of-the-money strikes, decreasing with tenor (vol term structure).

        Returns
        -------
        list of VolatilityQuote
        """
        quotes = []
        # ATM vol term structure – roughly 50 % for short, 30 % for long tenors
        atm_vol_ts = {
            1: 0.50,
            2: 0.45,
            3: 0.42,
            5: 0.38,
            7: 0.35,
            10: 0.30,
            15: 0.28,
            20: 0.27,
        }
        for tenor in self.CAP_TENORS:
            atm_rate = self._nss_rate(tenor)
            atm_vol = atm_vol_ts.get(tenor, 0.35)
            for offset in self.CAP_STRIKES_OFFSET:
                strike = max(atm_rate + offset, 0.0025)
                # Simple smile: vol increases for OTM strikes
                smile_adj = 0.05 * (offset / 0.01) ** 2
                vol = max(
                    self._add_noise(atm_vol + smile_adj, scale=0.005),
                    0.05,
                )
                quotes.append(
                    VolatilityQuote(tenor=tenor, strike=strike, vol=vol)
                )
        return quotes

    def generate_fx_vol_surface(
        self,
        currency_pair: str = "EURUSD",
        spot: float = 1.08,
    ) -> List[FXVolatilityQuote]:
        """
        Generate an FX option implied-volatility surface in delta space.

        Uses a standard 5-point smile per expiry: 10-delta put, 25-delta
        put, ATM (50-delta), 25-delta call, 10-delta call.

        Parameters
        ----------
        currency_pair : str
        spot : float
            Current spot rate.

        Returns
        -------
        list of FXVolatilityQuote
        """
        deltas = [0.10, 0.25, 0.50, 0.75, 0.90]
        # ATM vol for each expiry
        atm_vols = {
            1 / 12: 0.08,
            3 / 12: 0.09,
            6 / 12: 0.095,
            1: 0.10,
            2: 0.105,
            5: 0.12,
        }
        quotes = []
        for expiry in self.FX_EXPIRIES:
            atm_vol = atm_vols.get(expiry, 0.10)
            for delta in deltas:
                # Smile: higher vol away from ATM
                smile = 0.02 * (2 * delta - 1) ** 2
                vol = max(
                    self._add_noise(atm_vol + smile, scale=0.002),
                    0.02,
                )
                quotes.append(
                    FXVolatilityQuote(
                        expiry=expiry, delta=delta, vol=vol,
                        currency_pair=currency_pair,
                    )
                )
        return quotes

    def generate_all(self) -> Dict[str, list]:
        """
        Generate the full market data set.

        Returns
        -------
        dict with keys: "swap_rates", "mm_quotes", "futures",
                        "fras", "rates_vol", "fx_vol"
        """
        return {
            "swap_rates": self.generate_swap_rates(),
            "mm_quotes": self.generate_mm_quotes(),
            "futures": self.generate_future_quotes(),
            "fras": self.generate_fra_quotes(),
            "rates_vol": self.generate_rates_vol_surface(),
            "fx_vol": self.generate_fx_vol_surface(),
        }

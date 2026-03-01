"""
Auto-Callable Structured Note
==============================
An auto-call pays a coupon and potentially redeems early if the underlying
rate (or index level) exceeds a barrier on observation dates.  If the
underlying never triggers, a final payoff (possibly protected) is paid
at maturity.

Pricing is performed via Monte Carlo simulation of the Hull-White model
for the underlying (rates-linked) auto-call.

Payoff description
------------------
On each observation date t_i (i = 1, ..., N):
  * If S(t_i) ≥ autocall_barrier:
      - Pay coupon_i and redeem at par (early termination).
On final date t_N (if not called):
  * If S(t_N) ≥ final_barrier: pay par + coupon_N
  * Else: pay par * max(S(t_N) / S(0), capital_protection)

where S(t) is the underlying index level (simulated).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


class AutoCall:
    """
    Auto-callable note pricing via Monte Carlo.

    Parameters
    ----------
    observation_dates : list of float  Observation / potential call dates (years).
    autocall_barrier : float           Autocall barrier as fraction of initial (e.g. 1.0 = 100 %).
    coupon : float                     Coupon per observation period (decimal).
    final_barrier : float              Final barrier for full redemption (e.g. 1.0).
    capital_protection : float         Capital protection level (e.g. 0.0 = no protection).
    notional : float
    """

    def __init__(
        self,
        observation_dates: Sequence[float],
        autocall_barrier: float = 1.0,
        coupon: float = 0.05,
        final_barrier: float = 0.8,
        capital_protection: float = 0.0,
        notional: float = 1_000_000.0,
    ) -> None:
        self.observation_dates = sorted(observation_dates)
        self.autocall_barrier = autocall_barrier
        self.coupon = coupon
        self.final_barrier = final_barrier
        self.capital_protection = capital_protection
        self.notional = notional

    def price(
        self,
        hw_model,
        n_paths: int = 50_000,
        n_steps: int = 200,
        seed: int = 42,
        underlying_vol: float = 0.20,
        spot: float = 1.0,
        dividend_yield: float = 0.0,
    ) -> float:
        """
        Price the auto-call via Monte Carlo under the Hull-White model.

        The underlying index is assumed to follow a log-normal process
        with the Hull-White short rate as the risk-free rate:

            dS/S = (r(t) – q) dt + σ_S dW_S

        where W_S is independent of the rate Brownian motion.

        Parameters
        ----------
        hw_model : HullWhiteModel
        n_paths : int        Number of MC paths.
        n_steps : int        Time steps per path.
        seed : int
        underlying_vol : float  Equity / index vol (σ_S).
        spot : float         Initial index level (normalised to 1.0 by default).
        dividend_yield : float

        Returns
        -------
        float  Present value.
        """
        rng = np.random.default_rng(seed)
        T_max = self.observation_dates[-1]
        dt = T_max / n_steps
        times = np.linspace(0.0, T_max, n_steps + 1)

        # Simulate r(t) paths
        _times_hw, r_paths = hw_model.simulate(
            n_paths=n_paths, n_steps=n_steps, T=T_max, seed=seed
        )

        # Simulate S(t) given r(t): Euler on S
        a_eq = hw_model.a
        sig_S = underlying_vol
        S_paths = np.ones((n_paths, n_steps + 1)) * spot
        Z_S = rng.standard_normal((n_paths, n_steps))

        for i in range(n_steps):
            r_i = r_paths[:, i]
            S_paths[:, i + 1] = S_paths[:, i] * np.exp(
                (r_i - dividend_yield - 0.5 * sig_S ** 2) * dt + sig_S * math.sqrt(dt) * Z_S[:, i]
            )

        # Compute discount factors along each path
        # DF(0, T) = E[exp(-integral_0^T r(t) dt)]
        # Approximate: integral ≈ Σ r_i * dt (trapezoid)
        integral_r = np.trapezoid(r_paths, dx=dt, axis=1)  # shape (n_paths,)
        df_paths = np.exp(-integral_r)

        # Get observation-step indices
        obs_indices = [int(round(t / dt)) for t in self.observation_dates]
        obs_indices = [min(idx, n_steps) for idx in obs_indices]

        payoffs = np.zeros(n_paths)
        called = np.zeros(n_paths, dtype=bool)

        for k, (t_obs, step) in enumerate(zip(self.observation_dates, obs_indices)):
            S_obs = S_paths[:, step]
            trigger = (~called) & (S_obs >= self.autocall_barrier * spot)

            # Discount to today for triggered paths
            # DF(0, t_obs) from the path integral up to t_obs
            integral_to_obs = np.trapezoid(r_paths[:, : step + 1], dx=dt, axis=1)
            df_obs = np.exp(-integral_to_obs)

            payoff_triggered = (1.0 + self.coupon) * self.notional
            payoffs = np.where(trigger, payoffs + df_obs * payoff_triggered, payoffs)
            called = called | trigger

        # Final payoff for uncalled paths
        k_final = len(self.observation_dates) - 1
        t_final = self.observation_dates[k_final]
        step_final = obs_indices[k_final]
        S_final = S_paths[:, step_final]
        integral_to_final = np.trapezoid(r_paths[:, : step_final + 1], dx=dt, axis=1)
        df_final = np.exp(-integral_to_final)

        above_final_barrier = (~called) & (S_final >= self.final_barrier * spot)
        below_final_barrier = (~called) & (S_final < self.final_barrier * spot)

        # Above barrier: full redemption + coupon
        payoffs = np.where(
            above_final_barrier,
            payoffs + df_final * (1.0 + self.coupon) * self.notional,
            payoffs,
        )
        # Below barrier: capital protection
        redemption = np.maximum(S_final / spot, self.capital_protection)
        payoffs = np.where(
            below_final_barrier,
            payoffs + df_final * redemption * self.notional,
            payoffs,
        )

        return float(np.mean(payoffs))

    def __repr__(self) -> str:
        return (
            f"AutoCall(observations={len(self.observation_dates)}, "
            f"barrier={self.autocall_barrier:.0%}, "
            f"coupon={self.coupon:.2%}, notional={self.notional:,.0f})"
        )

"""
SABR Stochastic-Alpha-Beta-Rho Model
=====================================
Implements the Hagan et al. (2002) approximation for implied Black volatility
under the SABR model:

    dF = α·F^β·dW₁
    dα = ν·α·dW₂
    dW₁·dW₂ = ρ·dt

Reference
---------
Hagan, P.S., Kumar, D., Lesniewski, A.S. & Woodward, D.E. (2002).
Managing Smile Risk. Wilmott Magazine, pp. 84–108.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm


class SABRModel:
    """
    SABR model for forward-rate smile dynamics.

    Parameters
    ----------
    alpha : float  Initial vol level (σ₀ ≡ α in SABR notation).
    beta  : float  CEV exponent, 0 ≤ β ≤ 1.
                   β = 0: normal SABR, β = 1: log-normal SABR.
    rho   : float  Correlation between F and α, -1 < ρ < 1.
    nu    : float  Vol-of-vol (ν > 0).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.5,
        rho: float = -0.3,
        nu: float = 0.4,
    ) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if nu <= 0.0:
            raise ValueError("nu must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    # ------------------------------------------------------------------
    # Hagan implied vol formula
    # ------------------------------------------------------------------

    def implied_vol(
        self,
        F: float,
        K: float,
        T: float,
        vol_type: str = "lognormal",
    ) -> float:
        """
        SABR implied (Black log-normal) volatility via Hagan approximation.

        Parameters
        ----------
        F : float  Current forward rate.
        K : float  Strike.
        T : float  Time to expiry (years).
        vol_type : str  "lognormal" (default) | "normal"

        Returns
        -------
        float  Implied volatility.
        """
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        if F <= 0 or K <= 0:
            raise ValueError("F and K must be positive for lognormal SABR")

        if vol_type == "normal":
            return self._normal_implied_vol(F, K, T)

        # Handle ATM case separately
        if abs(F - K) < 1e-10:
            return self._atm_implied_vol(F, T)

        FK = F * K
        FK_beta = FK ** ((1.0 - beta) / 2.0)
        log_FK = math.log(F / K)

        z = (nu / alpha) * FK_beta * log_FK

        # x(z) function
        if abs(z) < 1e-6:
            x_z = 1.0
        else:
            chi = math.log(
                (math.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho)
            )
            x_z = z / chi

        # Leading term
        log_FK_sq = log_FK ** 2
        b2 = (1.0 - beta) ** 2
        D1 = 1.0 + b2 / 24.0 * log_FK_sq + b2 ** 2 / 1920.0 * log_FK_sq ** 2

        # Correction
        C1 = (
            b2 * alpha ** 2 / (24.0 * FK ** (1.0 - beta))
            + rho * beta * nu * alpha / (4.0 * FK_beta)
            + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
        )

        sigma_B = (
            (alpha / (FK_beta * D1))
            * x_z
            * (1.0 + C1 * T)
        )

        return max(sigma_B, 1e-6)

    def _atm_implied_vol(self, F: float, T: float) -> float:
        """ATM Black implied vol (closed form)."""
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu
        F_b = F ** (1.0 - beta)
        C1 = (
            (1.0 - beta) ** 2 * alpha ** 2 / (24.0 * F_b ** 2)
            + rho * beta * nu * alpha / (4.0 * F_b)
            + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
        )
        return (alpha / F_b) * (1.0 + C1 * T)

    def _normal_implied_vol(self, F: float, K: float, T: float) -> float:
        """Bachelier / normal implied vol approximation."""
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu
        if abs(F - K) < 1e-10:
            F_b = F ** beta
            return alpha * F_b * (1.0 + ((1.0 - beta) ** 2 * alpha ** 2 / (24.0 * F ** (2.0 - 2.0 * beta)) + rho * nu * alpha * beta / (4.0 * F ** (1.0 - beta)) + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0) * T)
        log_FK = math.log(F / K)
        FK_beta = (F * K) ** (beta / 2.0)
        z = nu * (F - K) / (alpha * FK_beta)
        if abs(z) < 1e-8:
            chi_ratio = 1.0
        else:
            chi = math.log((math.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
            chi_ratio = z / chi
        C1 = (1.0 - beta) ** 2 * alpha ** 2 / (24.0 * (F * K) ** (1.0 - beta)) + rho * nu * alpha * beta / (4.0 * FK_beta) + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
        return alpha * FK_beta * chi_ratio * (1.0 + C1 * T)

    # ------------------------------------------------------------------
    # Option pricing
    # ------------------------------------------------------------------

    def option_price(
        self,
        F: float,
        K: float,
        T: float,
        df: float = 1.0,
        is_call: bool = True,
    ) -> float:
        """
        Price a European call or put using SABR implied vol in Black formula.

        Parameters
        ----------
        F : float  Forward rate.
        K : float  Strike.
        T : float  Expiry (years).
        df : float Discount factor.
        is_call : bool

        Returns
        -------
        float  Present value.
        """
        vol = self.implied_vol(F, K, T)
        if T <= 1e-10 or vol < 1e-10:
            if is_call:
                return max(F - K, 0.0) * df
            return max(K - F, 0.0) * df

        d1 = (math.log(F / K) + 0.5 * vol ** 2 * T) / (vol * math.sqrt(T))
        d2 = d1 - vol * math.sqrt(T)

        if is_call:
            return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    # ------------------------------------------------------------------
    # Volatility surface
    # ------------------------------------------------------------------

    def vol_surface(
        self,
        F: float,
        strikes: Sequence[float],
        expiry: float,
    ) -> np.ndarray:
        """
        Compute implied-vol smile for a given expiry and set of strikes.

        Parameters
        ----------
        F : float  ATM forward.
        strikes : array-like
        expiry : float

        Returns
        -------
        np.ndarray  Implied vols.
        """
        return np.array([self.implied_vol(F, K, expiry) for K in strikes])

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(
        cls,
        F: float,
        strikes: Sequence[float],
        expiry: float,
        market_vols: Sequence[float],
        beta: float = 0.5,
        method: str = "minimize",
    ) -> "SABRModel":
        """
        Calibrate α, ρ, ν to match market implied vols (β is fixed).

        Parameters
        ----------
        F : float            ATM forward rate.
        strikes : array-like
        expiry : float
        market_vols : array-like  Market Black implied vols.
        beta : float         Fixed CEV exponent (default 0.5).
        method : str         "minimize" | "de" (differential evolution).

        Returns
        -------
        SABRModel
        """
        K_arr = np.asarray(strikes, dtype=float)
        v_arr = np.asarray(market_vols, dtype=float)

        def objective(params: np.ndarray) -> float:
            alpha = max(params[0], 1e-5)
            rho = max(min(params[1], 0.9999), -0.9999)
            nu = max(params[2], 1e-5)
            model = cls(alpha=alpha, beta=beta, rho=rho, nu=nu)
            err = 0.0
            for K, v_mkt in zip(K_arr, v_arr):
                try:
                    v_mod = model.implied_vol(F, K, expiry)
                except Exception:
                    v_mod = 0.0
                err += (v_mod - v_mkt) ** 2
            return err

        bounds = [(1e-5, 2.0), (-0.9999, 0.9999), (1e-5, 5.0)]
        # Use ATM vol to guess alpha
        atm_vol = float(np.interp(F, K_arr, v_arr))
        x0 = [atm_vol * F ** (1.0 - beta), -0.3, 0.4]

        if method == "de":
            res = differential_evolution(objective, bounds, seed=42, maxiter=500, tol=1e-10)
        else:
            res = minimize(
                objective, x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 2000},
            )

        alpha_opt = max(res.x[0], 1e-5)
        rho_opt = max(min(res.x[1], 0.9999), -0.9999)
        nu_opt = max(res.x[2], 1e-5)
        return cls(alpha=alpha_opt, beta=beta, rho=rho_opt, nu=nu_opt)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SABRModel(α={self.alpha:.4f}, β={self.beta:.2f}, "
            f"ρ={self.rho:.4f}, ν={self.nu:.4f})"
        )

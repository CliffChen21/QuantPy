"""
Nelson-Siegel-Svensson (NSS) Parametric Curve Model
====================================================
Fits the NSS model to par-swap rates or zero rates.

Reference
---------
Nelson, C.R. & Siegel, A.F. (1987).
Svensson, L. (1994) extended version with second hump term.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution

from .discount_curve import DiscountCurve


class NSSModel:
    """
    Nelson-Siegel-Svensson parametric yield-curve model.

    Parameters
    ----------
    beta0, beta1, beta2, beta3 : float
        Level, slope, first curvature, second curvature.
    lambda1, lambda2 : float
        Decay parameters (> 0).
    """

    def __init__(
        self,
        beta0: float = 0.04,
        beta1: float = -0.01,
        beta2: float = 0.02,
        beta3: float = -0.01,
        lambda1: float = 1.5,
        lambda2: float = 0.5,
    ) -> None:
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    # ------------------------------------------------------------------
    # Core formula
    # ------------------------------------------------------------------

    def zero_rate(self, tau: float) -> float:
        """
        NSS zero rate for maturity *tau* (years).

        Parameters
        ----------
        tau : float  Time in years.

        Returns
        -------
        float  Continuously-compounded zero rate.
        """
        if tau < 1e-8:
            return self.beta0 + self.beta1

        l1, l2 = self.lambda1, self.lambda2
        e1 = math.exp(-tau / l1)
        e2 = math.exp(-tau / l2)

        t1 = (1 - e1) / (tau / l1)
        t2 = (1 - e2) / (tau / l2)

        return (
            self.beta0
            + self.beta1 * t1
            + self.beta2 * (t1 - e1)
            + self.beta3 * (t2 - e2)
        )

    def discount_factor(self, tau: float) -> float:
        """DF(0, tau) = exp(-r(tau) * tau)."""
        return math.exp(-self.zero_rate(tau) * tau)

    def forward_rate(self, tau: float) -> float:
        """
        Instantaneous forward rate f(0, tau).

        Derived analytically from the NSS zero-rate formula.
        """
        if tau < 1e-8:
            return self.beta0 + self.beta1

        l1, l2 = self.lambda1, self.lambda2
        e1 = math.exp(-tau / l1)
        e2 = math.exp(-tau / l2)

        return (
            self.beta0
            + self.beta1 * e1
            + self.beta2 * (tau / l1) * e1
            + self.beta3 * (tau / l2) * e2
        )

    def to_discount_curve(
        self,
        tenors: Optional[Sequence[float]] = None,
        method: str = "log_linear",
    ) -> DiscountCurve:
        """
        Convert to a :class:`DiscountCurve` sampled at *tenors*.

        Parameters
        ----------
        tenors : list of float, optional
            Sample tenors. Defaults to a standard grid.
        method : str
            Interpolation method for the resulting DiscountCurve.

        Returns
        -------
        DiscountCurve
        """
        if tenors is None:
            tenors = [
                0.0, 0.083, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4,
                5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30,
            ]
        dfs = [self.discount_factor(t) for t in tenors]
        # ensure t=0 -> DF=1
        times = list(tenors)
        if times[0] != 0.0:
            times.insert(0, 0.0)
            dfs.insert(0, 1.0)
        return DiscountCurve(times=times, discount_factors=dfs, method=method)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(
        cls,
        tenors: Sequence[float],
        rates: Sequence[float],
        rate_type: str = "zero",
        method: str = "minimize",
    ) -> "NSSModel":
        """
        Fit the NSS model to observed rates.

        Parameters
        ----------
        tenors : array-like
            Maturities in years.
        rates : array-like
            Observed rates (decimal).
        rate_type : str
            "zero" (continuously compounded) or "par" (par swap rates).
        method : str
            "minimize" (L-BFGS-B) or "de" (differential evolution).

        Returns
        -------
        NSSModel
        """
        tenors = np.asarray(tenors, dtype=float)
        rates = np.asarray(rates, dtype=float)

        def _par_rate(model: "NSSModel", tau: float) -> float:
            """Approximate par swap rate from NSS model (semi-annual coupons)."""
            n = max(int(round(tau * 2)), 1)
            t_list = [(i + 1) * 0.5 for i in range(n)]
            annuity = sum(0.5 * model.discount_factor(t) for t in t_list)
            if annuity < 1e-12:
                return 0.0
            return (1.0 - model.discount_factor(tau)) / annuity

        def _objective(params: np.ndarray) -> float:
            b0, b1, b2, b3 = params[:4]
            l1 = max(params[4], 0.01)
            l2 = max(params[5], 0.01)
            model = cls(b0, b1, b2, b3, l1, l2)
            errors = []
            for tau, r_obs in zip(tenors, rates):
                if rate_type == "zero":
                    r_model = model.zero_rate(tau)
                else:
                    r_model = _par_rate(model, tau)
                errors.append((r_model - r_obs) ** 2)
            return float(np.sum(errors))

        x0 = np.array([0.04, -0.01, 0.02, -0.01, 1.5, 0.5])
        bounds = [
            (0.0, 0.20), (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15),
            (0.01, 10.0), (0.01, 5.0),
        ]

        if method == "de":
            res = differential_evolution(_objective, bounds, seed=42, maxiter=500, tol=1e-8)
        else:
            res = minimize(
                _objective, x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
            )

        b0, b1, b2, b3 = res.x[:4]
        l1 = max(res.x[4], 0.01)
        l2 = max(res.x[5], 0.01)
        return cls(b0, b1, b2, b3, l1, l2)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NSSModel(β0={self.beta0:.4f}, β1={self.beta1:.4f}, "
            f"β2={self.beta2:.4f}, β3={self.beta3:.4f}, "
            f"λ1={self.lambda1:.4f}, λ2={self.lambda2:.4f})"
        )

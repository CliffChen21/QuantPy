"""
Risk Scenario Engine
====================
Computes first- and second-order sensitivities via bump-and-reprice:

  Delta (Δ)  – sensitivity to a parallel 1bp shift in zero rates
  Gamma (Γ)  – second-order rate sensitivity
  Vega  (ν)  – sensitivity to a 1-vol-point shift in flat vol
  Nu         – sensitivity to vol-of-vol (ν in SABR)
  Rho        – sensitivity to SABR correlation parameter
  Bucket DV01 – sensitivity to each pillar of the discount curve

All sensitivities are computed by central finite differences unless noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from quantpy.curves.discount_curve import DiscountCurve


@dataclass
class RiskResult:
    """Container for all computed risk sensitivities."""

    base_price: float
    delta: float = 0.0                      # 1bp parallel bump
    gamma: float = 0.0                      # 1bp^2 convexity
    vega: float = 0.0                       # 1 vol-point bump
    nu: float = 0.0                         # SABR nu sensitivity
    rho_sabr: float = 0.0                   # SABR rho sensitivity
    bucket_dv01: Dict[float, float] = field(default_factory=dict)  # per-pillar DV01
    parallel_shift: Dict[float, float] = field(default_factory=dict)  # NPV at shifted rates

    def __repr__(self) -> str:
        lines = [
            f"RiskResult(",
            f"  base_price   = {self.base_price:,.4f}",
            f"  delta (1bp)  = {self.delta:,.4f}",
            f"  gamma        = {self.gamma:,.6f}",
            f"  vega (1vp)   = {self.vega:,.4f}",
            f"  nu           = {self.nu:,.4f}",
            f"  rho_sabr     = {self.rho_sabr:,.4f}",
            f")",
        ]
        return "\n".join(lines)


class RiskEngine:
    """
    General-purpose bump-and-reprice risk engine.

    Parameters
    ----------
    rate_bump : float  Parallel bump size for delta/gamma (default 1bp = 1e-4).
    vol_bump  : float  Vol bump for vega (default 1 vol-point = 0.01).
    """

    def __init__(
        self,
        rate_bump: float = 1e-4,
        vol_bump: float = 0.01,
    ) -> None:
        self.rate_bump = rate_bump
        self.vol_bump = vol_bump

    # ------------------------------------------------------------------
    # Curve bumping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bump_curve(
        curve: DiscountCurve, bump: float, bucket: Optional[int] = None
    ) -> DiscountCurve:
        """
        Return a bumped curve.

        Parameters
        ----------
        curve  : DiscountCurve
        bump   : float  Rate bump (e.g. 1e-4 for 1bp).
        bucket : int, optional  If given, only bump pillar at this index.

        Returns
        -------
        DiscountCurve
        """
        times = list(curve._times)
        dfs = list(curve._dfs)
        new_dfs = []
        for i, (t, df) in enumerate(zip(times, dfs)):
            if t <= 0.0:
                new_dfs.append(1.0)
                continue
            if bucket is None or i == bucket:
                new_dfs.append(df * math.exp(-bump * t))
            else:
                new_dfs.append(df)
        return DiscountCurve(times, new_dfs, method=curve._method)

    # ------------------------------------------------------------------
    # Core risk functions
    # ------------------------------------------------------------------

    def compute_delta(
        self,
        pricer: Callable[[DiscountCurve], float],
        curve: DiscountCurve,
    ) -> float:
        """
        Delta: dV/dr (parallel 1bp shift).

        Uses central finite difference.

        Parameters
        ----------
        pricer : callable  Function (curve) -> price.
        curve  : DiscountCurve

        Returns
        -------
        float
        """
        c_up = self._bump_curve(curve, +self.rate_bump)
        c_dn = self._bump_curve(curve, -self.rate_bump)
        return (pricer(c_up) - pricer(c_dn)) / (2.0 * self.rate_bump)

    def compute_gamma(
        self,
        pricer: Callable[[DiscountCurve], float],
        curve: DiscountCurve,
        base_price: float,
    ) -> float:
        """
        Gamma: d²V/dr² (parallel 1bp shift).

        Parameters
        ----------
        pricer : callable
        curve  : DiscountCurve
        base_price : float  V(0) already computed.

        Returns
        -------
        float
        """
        c_up = self._bump_curve(curve, +self.rate_bump)
        c_dn = self._bump_curve(curve, -self.rate_bump)
        return (pricer(c_up) - 2.0 * base_price + pricer(c_dn)) / (self.rate_bump ** 2)

    def compute_vega(
        self,
        pricer_with_vol: Callable[[float], float],
        base_vol: float,
    ) -> float:
        """
        Vega: dV/dσ (1 vol-point shift in flat Black vol).

        Parameters
        ----------
        pricer_with_vol : callable  Function (vol) -> price.
        base_vol : float

        Returns
        -------
        float
        """
        return (
            pricer_with_vol(base_vol + self.vol_bump) - pricer_with_vol(base_vol - self.vol_bump)
        ) / (2.0 * self.vol_bump)

    def compute_nu(
        self,
        pricer_with_nu: Callable[[float], float],
        base_nu: float,
        bump: float = 0.01,
    ) -> float:
        """
        Nu (SABR): dV/dν.

        Parameters
        ----------
        pricer_with_nu : callable  Function (nu) -> price.
        base_nu : float
        bump : float

        Returns
        -------
        float
        """
        return (
            pricer_with_nu(base_nu + bump) - pricer_with_nu(base_nu - bump)
        ) / (2.0 * bump)

    def compute_rho_sabr(
        self,
        pricer_with_rho: Callable[[float], float],
        base_rho: float,
        bump: float = 0.01,
    ) -> float:
        """
        Rho (SABR correlation): dV/dρ.

        Parameters
        ----------
        pricer_with_rho : callable  Function (rho) -> price.
        base_rho : float
        bump : float

        Returns
        -------
        float
        """
        rho_up = min(base_rho + bump, 0.9999)
        rho_dn = max(base_rho - bump, -0.9999)
        return (pricer_with_rho(rho_up) - pricer_with_rho(rho_dn)) / (rho_up - rho_dn)

    def compute_bucket_dv01(
        self,
        pricer: Callable[[DiscountCurve], float],
        curve: DiscountCurve,
    ) -> Dict[float, float]:
        """
        Bucket DV01: sensitivity to each curve pillar individually.

        Returns
        -------
        dict {pillar_time: dv01}
        """
        result = {}
        times = list(curve._times)
        for i, t in enumerate(times):
            if t <= 0.0:
                result[t] = 0.0
                continue
            c_up = self._bump_curve(curve, +self.rate_bump, bucket=i)
            c_dn = self._bump_curve(curve, -self.rate_bump, bucket=i)
            dv01 = (pricer(c_up) - pricer(c_dn)) / (2.0 * self.rate_bump)
            result[t] = dv01
        return result

    def compute_parallel_shift_scenario(
        self,
        pricer: Callable[[DiscountCurve], float],
        curve: DiscountCurve,
        shifts: Sequence[float] = None,
    ) -> Dict[float, float]:
        """
        Revalue the instrument at a range of parallel rate shifts.

        Parameters
        ----------
        pricer : callable
        curve  : DiscountCurve
        shifts : list of float  Shifts in decimal (e.g. [-0.01, -0.005, 0, 0.005, 0.01]).

        Returns
        -------
        dict {shift: price}
        """
        if shifts is None:
            shifts = [-0.02, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, 0.02]
        result = {}
        for s in shifts:
            c_shifted = self._bump_curve(curve, s)
            result[s] = pricer(c_shifted)
        return result

    # ------------------------------------------------------------------
    # All-in-one
    # ------------------------------------------------------------------

    def full_risk(
        self,
        pricer: Callable[[DiscountCurve], float],
        curve: DiscountCurve,
        pricer_with_vol: Optional[Callable[[float], float]] = None,
        base_vol: Optional[float] = None,
        pricer_with_nu: Optional[Callable[[float], float]] = None,
        base_nu: Optional[float] = None,
        pricer_with_rho: Optional[Callable[[float], float]] = None,
        base_rho: Optional[float] = None,
        compute_buckets: bool = True,
        compute_scenarios: bool = True,
    ) -> RiskResult:
        """
        Compute all risk sensitivities in one call.

        Parameters
        ----------
        pricer : callable  Function (curve) -> price.
        curve  : DiscountCurve
        pricer_with_vol : callable, optional  Function (vol) -> price for vega.
        base_vol : float, optional
        pricer_with_nu : callable, optional  Function (nu) -> price for nu.
        base_nu : float, optional
        pricer_with_rho : callable, optional  Function (rho) -> price for rho.
        base_rho : float, optional
        compute_buckets : bool
        compute_scenarios : bool

        Returns
        -------
        RiskResult
        """
        base_price = pricer(curve)
        delta = self.compute_delta(pricer, curve)
        gamma = self.compute_gamma(pricer, curve, base_price)

        vega = 0.0
        if pricer_with_vol is not None and base_vol is not None:
            vega = self.compute_vega(pricer_with_vol, base_vol)

        nu = 0.0
        if pricer_with_nu is not None and base_nu is not None:
            nu = self.compute_nu(pricer_with_nu, base_nu)

        rho_sabr = 0.0
        if pricer_with_rho is not None and base_rho is not None:
            rho_sabr = self.compute_rho_sabr(pricer_with_rho, base_rho)

        buckets = {}
        if compute_buckets:
            buckets = self.compute_bucket_dv01(pricer, curve)

        scenarios = {}
        if compute_scenarios:
            scenarios = self.compute_parallel_shift_scenario(pricer, curve)

        return RiskResult(
            base_price=base_price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            nu=nu,
            rho_sabr=rho_sabr,
            bucket_dv01=buckets,
            parallel_shift=scenarios,
        )

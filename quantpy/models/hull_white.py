"""
Hull-White One-Factor Model
===========================
Short-rate process:
    dr(t) = [θ(t) – a·r(t)] dt + σ·dW(t)

where θ(t) is chosen to fit the initial discount curve exactly.

Key analytics:
  * Zero-coupon bond price P(t, T)
  * Cap / Floor pricing (each caplet = put on zero-coupon bond)
  * European swaption pricing (Jamshidian decomposition)
  * Bermudan swaption pricing (trinomial tree)
  * Monte-Carlo simulation

References
----------
Hull, J. & White, A. (1994). Numerical Procedures for Implementing
Term Structure Models I: Single-Factor Models.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm

from quantpy.curves.discount_curve import DiscountCurve


class HullWhiteModel:
    """
    1-Factor Hull-White model.

    Parameters
    ----------
    curve : DiscountCurve
        The initial discount curve P(0, T) used to calibrate θ(t).
    a : float
        Mean-reversion speed (> 0).
    sigma : float
        Short-rate volatility (> 0).
    """

    def __init__(
        self,
        curve: DiscountCurve,
        a: float = 0.05,
        sigma: float = 0.01,
    ) -> None:
        if a <= 0:
            raise ValueError("a (mean reversion) must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.curve = curve
        self.a = a
        self.sigma = sigma

    # ------------------------------------------------------------------
    # Bond-pricing analytics
    # ------------------------------------------------------------------

    def _B(self, t: float, T: float) -> float:
        """B(t,T) factor in the bond-price formula."""
        return (1.0 - math.exp(-self.a * (T - t))) / self.a

    def _ln_A(self, t: float, T: float) -> float:
        """
        ln A(t,T) derived from the initial curve via Girsanov / HJM consistency.

        ln A(t,T) = ln P(0,T)/P(0,t) + B(t,T)*f(0,t)
                    – (σ²/4a)(1–exp(–2at)) B²(t,T)
        """
        P0T = self.curve.discount_factor(T)
        P0t = self.curve.discount_factor(t)
        f0t = self.curve.forward_rate(max(t, 1e-6), max(t + 1e-4, 1e-5))
        B = self._B(t, T)
        corr = (self.sigma ** 2 / (4.0 * self.a)) * (1.0 - math.exp(-2.0 * self.a * t)) * B ** 2
        return math.log(P0T / max(P0t, 1e-15)) + B * f0t - corr

    def bond_price(self, t: float, T: float, r_t: float) -> float:
        """
        Zero-coupon bond price P(t, T | r(t) = r_t).

        Parameters
        ----------
        t, T : float  Current time and maturity (years).
        r_t : float   Short rate at time t.

        Returns
        -------
        float
        """
        B = self._B(t, T)
        lnA = self._ln_A(t, T)
        return math.exp(lnA - B * r_t)

    def mean_r(self, t: float, r0: float = None) -> float:
        """
        Conditional mean E[r(t) | r(0)].

        Under HW, E[r(t)] = r(0)*exp(-at) + integral_0^t theta(s)*exp(-a(t-s))ds

        For consistent fitting with the initial curve:
        E[r(t)] = f(0,t) + (σ²/2a²)(1 – exp(–at))²
        (where f(0,t) is the market instantaneous forward rate).
        """
        f0t = self.curve.forward_rate(max(t - 1e-4, 1e-6), t + 1e-4)
        mean = f0t + (self.sigma ** 2 / (2.0 * self.a ** 2)) * (1.0 - math.exp(-self.a * t)) ** 2
        if r0 is not None:
            # E[r(t)|r(0)] exactly
            mean_r0 = f0t + (self.sigma ** 2 / (2.0 * self.a ** 2)) * (1.0 - math.exp(-self.a * t)) ** 2
            # Adjust for r(0) deviation
            f00 = self.curve.forward_rate(1e-6, 2e-4)
            mean = mean_r0 + (r0 - f00) * math.exp(-self.a * t)
        return mean

    def variance_r(self, t: float, s: float = 0.0) -> float:
        """
        Var[r(t) | F_s] = (σ²/2a)(1 – exp(–2a(t–s))).

        Parameters
        ----------
        t : float  Future time.
        s : float  Current time (default 0).

        Returns
        -------
        float
        """
        return (self.sigma ** 2 / (2.0 * self.a)) * (1.0 - math.exp(-2.0 * self.a * (t - s)))

    # ------------------------------------------------------------------
    # Caplet pricing
    # ------------------------------------------------------------------

    def caplet_price(
        self,
        reset: float,
        payment: float,
        strike: float,
        notional: float = 1.0,
        is_call: bool = True,
    ) -> float:
        """
        Price a caplet (or floorlet) using Hull-White analytics.

        A caplet [reset, payment] with strike K is a call on a ZCB:
          payoff at payment = δ·max(L – K, 0)
               = (1 + δ·K)·max(P(reset, payment)_inv – K*, 0)

        We price it as a put on the ZCB P(reset, payment).

        Parameters
        ----------
        reset   : float  Fixing date (years).
        payment : float  Payment date (years).
        strike  : float  Cap strike rate (decimal).
        notional : float
        is_call : bool   True = caplet, False = floorlet.

        Returns
        -------
        float  Present value.
        """
        delta = payment - reset
        # Strike on the bond = 1 / (1 + strike * delta)
        X = 1.0 / (1.0 + strike * delta)
        # Scale factor
        M = (1.0 + strike * delta) * notional

        # sigma_p: std dev of ln P(reset, payment)
        sigma_p = (
            self.sigma
            * math.sqrt((1.0 - math.exp(-2.0 * self.a * reset)) / (2.0 * self.a))
            * self._B(reset, payment)
        )

        if sigma_p < 1e-12:
            # Deterministic limit
            bond = self.curve.discount_factor(payment)
            if is_call:
                return max(bond / max(self.curve.discount_factor(reset), 1e-15) - X, 0.0) * M * self.curve.discount_factor(reset)
            else:
                return max(X - bond / max(self.curve.discount_factor(reset), 1e-15), 0.0) * M * self.curve.discount_factor(reset)

        P0t = self.curve.discount_factor(reset)
        P0T = self.curve.discount_factor(payment)

        h = math.log(P0T / (P0t * X)) / sigma_p + sigma_p / 2.0

        if is_call:
            # Caplet = ZCB put = M * [X * P0t * N(-h+sigma_p) - P0T * N(-h)]
            price = M * (X * P0t * norm.cdf(-h + sigma_p) - P0T * norm.cdf(-h))
        else:
            # Floorlet = ZCB call = M * [P0T * N(h) - X * P0t * N(h - sigma_p)]
            price = M * (P0T * norm.cdf(h) - X * P0t * norm.cdf(h - sigma_p))

        return max(price, 0.0)

    def cap_price(
        self,
        maturity: float,
        strike: float,
        payment_frequency: float = 0.25,
        notional: float = 1.0,
        is_cap: bool = True,
    ) -> float:
        """
        Price a cap (or floor) as a sum of caplets.

        Parameters
        ----------
        maturity : float
        strike : float
        payment_frequency : float  0.25 = quarterly, 0.5 = semi-annual.
        notional : float
        is_cap : bool

        Returns
        -------
        float
        """
        n = int(round(maturity / payment_frequency))
        resets = [(i) * payment_frequency for i in range(1, n + 1)]
        payments = [(i + 1) * payment_frequency for i in range(1, n + 1)]
        total = 0.0
        for reset, payment in zip(resets, payments):
            total += self.caplet_price(reset, payment, strike, notional, is_call=is_cap)
        return total

    # ------------------------------------------------------------------
    # European swaption (Jamshidian decomposition)
    # ------------------------------------------------------------------

    def european_swaption_price(
        self,
        expiry: float,
        swap_maturity: float,
        strike: float,
        notional: float = 1.0,
        is_payer: bool = True,
        payment_frequency: float = 0.5,
    ) -> float:
        """
        Price a European swaption via Jamshidian decomposition.

        A payer swaption gives the right to pay fixed and receive floating.

        Parameters
        ----------
        expiry : float          Option expiry (years).
        swap_maturity : float   Underlying swap maturity from today.
        strike : float          Fixed rate.
        notional : float
        is_payer : bool         True = payer, False = receiver.
        payment_frequency : float

        Returns
        -------
        float  Present value.
        """
        n = int(round((swap_maturity - expiry) / payment_frequency))
        coupon_times = [expiry + (i + 1) * payment_frequency for i in range(n)]
        delta = payment_frequency

        # Jamshidian: find r* such that sum of bond prices equals 1
        # (swap value = 0) at the optimal exercise boundary

        coupon_flows = [strike * delta * notional] * n
        coupon_flows[-1] += notional  # add notional on last payment

        def swap_value_at_r(r_star: float) -> float:
            total = 0.0
            for ci, ct in zip(coupon_flows, coupon_times):
                total += ci * self.bond_price(expiry, ct, r_star)
            return total - notional

        # Find r* (root of swap_value = 0)
        try:
            r_lo, r_hi = -0.5, 1.0
            f_lo = swap_value_at_r(r_lo)
            f_hi = swap_value_at_r(r_hi)
            if f_lo * f_hi > 0:
                # Widen bracket
                r_lo, r_hi = -1.0, 2.0
            r_star = brentq(swap_value_at_r, r_lo, r_hi, xtol=1e-10, maxiter=300)
        except ValueError:
            # Approximate: use current forward rate
            r_star = self.curve.forward_rate(expiry - 1e-3, expiry + 1e-3)

        # Strike prices for individual bond options
        X_i = [self.bond_price(expiry, ct, r_star) for ct in coupon_times]

        # Sum of bond options
        price = 0.0
        for ci, ct, Xi in zip(coupon_flows, coupon_times, X_i):
            # Each bond option is like a caplet/floorlet on a ZCB
            sigma_p = (
                self.sigma
                * math.sqrt((1.0 - math.exp(-2.0 * self.a * expiry)) / (2.0 * self.a))
                * self._B(expiry, ct)
            )
            P0t = self.curve.discount_factor(expiry)
            P0T = self.curve.discount_factor(ct)

            if sigma_p < 1e-12:
                if is_payer:
                    option_price = max(Xi - P0T / P0t, 0.0) * ci * P0t
                else:
                    option_price = max(P0T / P0t - Xi, 0.0) * ci * P0t
            else:
                h = math.log(P0T / (P0t * Xi)) / sigma_p + sigma_p / 2.0
                if is_payer:
                    # Receiver bond option (put on ZCB)
                    option_price = ci * (Xi * P0t * norm.cdf(-h + sigma_p) - P0T * norm.cdf(-h))
                else:
                    # Payer bond option (call on ZCB)
                    option_price = ci * (P0T * norm.cdf(h) - Xi * P0t * norm.cdf(h - sigma_p))
            price += max(option_price, 0.0)

        return price

    # ------------------------------------------------------------------
    # Bermudan swaption (trinomial tree)
    # ------------------------------------------------------------------

    def bermudan_swaption_price(
        self,
        expiries: Sequence[float],
        swap_maturity: float,
        strike: float,
        notional: float = 1.0,
        is_payer: bool = True,
        payment_frequency: float = 0.5,
        n_time_steps: int = 100,
    ) -> float:
        """
        Price a Bermudan swaption on a trinomial tree.

        Parameters
        ----------
        expiries : list of float
            Exercise dates (years from today).
        swap_maturity : float
            Final swap maturity.
        strike : float
            Fixed rate.
        notional : float
        is_payer : bool
        payment_frequency : float
        n_time_steps : int
            Number of tree steps (more = more accurate).

        Returns
        -------
        float
        """
        # Build trinomial tree up to swap_maturity
        T_max = swap_maturity
        dt = T_max / n_time_steps
        dx = self.sigma * math.sqrt(3.0 * dt)  # Typical Hull-White spacing

        # Branching probabilities (Hull-White symmetric branching)
        a, sigma = self.a, self.sigma
        V = sigma ** 2 * dt
        # j_max: limits tree size
        j_max = max(int(math.ceil(0.1844 / (a * dt))), 2)

        # Time grid
        times = [i * dt for i in range(n_time_steps + 1)]

        # Alpha (θ adjustment to fit initial curve)
        # alpha(t) = E^Q[r(t)] = d/dt[-ln P(0,t)]
        def alpha(t: float) -> float:
            """Mean of r*(t) = r(t) - alpha(t) process."""
            f0t = self.curve.forward_rate(max(t - 1e-4, 1e-6), t + 1e-4)
            return f0t + (sigma ** 2 / (2.0 * a ** 2)) * (1.0 - math.exp(-a * t)) ** 2

        # State: r*(t) = j * dx  (centered process)
        # r(t) = r*(t) + alpha(t)

        # Initialize tree: node values are option price
        # Forward sweep: compute Arrow-Debreu prices Q_{i,j}
        # Backward sweep: compute option values

        # For efficiency, store Arrow-Debreu prices (state prices)
        max_nodes = 2 * j_max + 1
        # Q[j + j_max] = Arrow-Debreu price at current time slice
        Q = np.zeros(max_nodes)
        Q[j_max] = 1.0  # Start: Q(0,0) = 1

        # We'll store all Q values at each time for backward pass
        # Memory-efficient: only need current slice
        # For backward pass we need option values at each time slice

        # Backward pass: option value V[j]
        # At maturity T_max, V = 0 (swap expired, no exercise)
        V_opt = np.zeros(max_nodes)

        # We need to handle exercise at each expiry date
        # Collect exercise indices
        exercise_set = set()
        for ex in expiries:
            idx = int(round(ex / dt))
            exercise_set.add(min(idx, n_time_steps))

        # Swap coupon times
        n_coup = int(round((swap_maturity) / payment_frequency))
        coupon_times_set = {
            int(round((i + 1) * payment_frequency / dt))
            for i in range(n_coup)
        }
        coupon_delta = payment_frequency

        def intrinsic_value(t: float, r: float) -> float:
            """
            Intrinsic value of holding the swaption at time t with r(t)=r.
            = value of immediate exercise = value of the underlying swap.
            """
            n_c = int(round((swap_maturity - t) / payment_frequency))
            if n_c <= 0:
                return 0.0
            c_times = [t + (k + 1) * payment_frequency for k in range(n_c)]
            # Swap PV from perspective of payer
            ann = sum(
                coupon_delta * self.bond_price(t, ct, r)
                for ct in c_times
            )
            swap_pv = (self.curve.discount_factor(t) / max(self.curve.discount_factor(swap_maturity), 1e-15) *
                       (1.0 - self.bond_price(t, swap_maturity, r)) - strike * ann) * notional
            if not is_payer:
                swap_pv = -swap_pv
            return max(swap_pv, 0.0)

        # Backward induction
        # Store option values V at each time slice
        V_opt = np.zeros(max_nodes)

        # We need Arrow-Debreu prices at each time slice for discounting
        # Two-pass: forward to build AD prices, backward to build option values

        # --- Forward pass: build Arrow-Debreu prices ---
        # Q_all[i] = array of AD prices at time step i
        Q_all = [np.zeros(max_nodes) for _ in range(n_time_steps + 1)]
        Q_all[0][j_max] = 1.0

        for i in range(n_time_steps):
            t_i = times[i]
            alp_i = alpha(t_i)
            Q_curr = Q_all[i]
            Q_next = np.zeros(max_nodes)
            for j in range(-j_max, j_max + 1):
                jj = j + j_max
                if Q_curr[jj] == 0.0:
                    continue
                r_j = j * dx + alp_i
                disc = math.exp(-r_j * dt)
                pu, pm, pd, j_up, j_mid, j_down = self._trinomial_probs(j, j_max, a, dt, dx)
                if 0 <= j_up + j_max < max_nodes:
                    Q_next[j_up + j_max] += Q_curr[jj] * disc * pu
                if 0 <= j_mid + j_max < max_nodes:
                    Q_next[j_mid + j_max] += Q_curr[jj] * disc * pm
                if 0 <= j_down + j_max < max_nodes:
                    Q_next[j_down + j_max] += Q_curr[jj] * disc * pd
            Q_all[i + 1] = Q_next

        # --- Backward pass ---
        V_opt = np.zeros(max_nodes)
        for i in range(n_time_steps, -1, -1):
            t_i = times[i]
            alp_i = alpha(t_i)
            V_new = np.zeros(max_nodes)

            if i == n_time_steps:
                # Terminal: no more cash flows
                V_opt = np.zeros(max_nodes)
                continue

            # Propagate backward
            t_next = times[i + 1]
            alp_next = alpha(t_next)
            V_hold = np.zeros(max_nodes)
            for j in range(-j_max, j_max + 1):
                jj = j + j_max
                r_j = j * dx + alp_i
                disc = math.exp(-r_j * dt)
                pu, pm, pd, j_up, j_mid, j_down = self._trinomial_probs(j, j_max, a, dt, dx)
                v_up = V_opt[j_up + j_max] if 0 <= j_up + j_max < max_nodes else 0.0
                v_mid = V_opt[j_mid + j_max] if 0 <= j_mid + j_max < max_nodes else 0.0
                v_dn = V_opt[j_down + j_max] if 0 <= j_down + j_max < max_nodes else 0.0
                V_hold[jj] = disc * (pu * v_up + pm * v_mid + pd * v_dn)

            # Exercise
            if i in exercise_set:
                for j in range(-j_max, j_max + 1):
                    jj = j + j_max
                    r_j = j * dx + alp_i
                    iv = intrinsic_value(t_i, r_j)
                    V_hold[jj] = max(V_hold[jj], iv)

            V_opt = V_hold

        # Price = sum_j Q_all[1][j] * V_opt[j] at t=0
        # Actually price = V_opt at node (0, j_max)
        return max(V_opt[j_max], 0.0)

    def _trinomial_probs(
        self, j: int, j_max: int, a: float, dt: float, dx: float
    ) -> Tuple[float, float, float, int, int, int]:
        """
        Hull-White trinomial tree branching probabilities.

        Returns (pu, pm, pd, j_up, j_mid, j_down).
        """
        eta = a * j * dt  # E[dX] = -a * j * dx * dt => drift factor
        V_dt = (self.sigma * math.sqrt(dt)) ** 2 / (dx ** 2)

        if j == j_max:
            # Top: branch down
            j_up = j - 1
            j_mid = j - 1
            j_down = j - 2  # extra-down branching
            pu = (1.0 / 6.0) + (eta ** 2 + eta) / 2.0
            pm = (2.0 / 3.0) - eta ** 2
            pd = (1.0 / 6.0) + (eta ** 2 - eta) / 2.0
            pu = max(min(pu, 1.0), 0.0)
            pm = max(min(pm, 1.0), 0.0)
            pd = max(min(pd, 1.0), 0.0)
            s = pu + pm + pd
            if s > 0:
                pu, pm, pd = pu / s, pm / s, pd / s
        elif j == -j_max:
            # Bottom: branch up
            j_up = j + 2
            j_mid = j + 1
            j_down = j + 1
            pu = (1.0 / 6.0) + (eta ** 2 + eta) / 2.0
            pm = (2.0 / 3.0) - eta ** 2
            pd = (1.0 / 6.0) + (eta ** 2 - eta) / 2.0
            pu = max(min(pu, 1.0), 0.0)
            pm = max(min(pm, 1.0), 0.0)
            pd = max(min(pd, 1.0), 0.0)
            s = pu + pm + pd
            if s > 0:
                pu, pm, pd = pu / s, pm / s, pd / s
        else:
            # Normal branching
            j_up = j + 1
            j_mid = j
            j_down = j - 1
            pu = (1.0 / 6.0) + (eta ** 2 - eta) / 2.0  # Note: sign convention
            pm = (2.0 / 3.0) - eta ** 2
            pd = (1.0 / 6.0) + (eta ** 2 + eta) / 2.0
            # Drift: E[r*] = r* - a*r*dt => E[j] = j*(1-a*dt)
            # Standard HW: pu=1/6+(η²+η)/2, pm=2/3-η², pd=1/6+(η²-η)/2
            # where η = a*j*sqrt(dt)/sqrt(3)
            eta2 = a * j * math.sqrt(dt / 3.0)
            pu = 1.0 / 6.0 + (eta2 ** 2 + eta2) / 2.0
            pm = 2.0 / 3.0 - eta2 ** 2
            pd = 1.0 / 6.0 + (eta2 ** 2 - eta2) / 2.0
            pu = max(min(pu, 1.0), 0.0)
            pm = max(min(pm, 1.0), 0.0)
            pd = max(min(pd, 1.0), 0.0)
            s = pu + pm + pd
            if s > 0:
                pu, pm, pd = pu / s, pm / s, pd / s

        return pu, pm, pd, j_up, j_mid, j_down

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_paths: int = 10_000,
        n_steps: int = 100,
        T: float = 10.0,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate short-rate paths under the risk-neutral measure.

        Uses an exact discretisation of the Hull-White SDE:
          r(t+dt) = r(t)*exp(-a*dt) + alpha(t+dt) - alpha(t)*exp(-a*dt)
                    + sigma*sqrt((1-exp(-2a*dt))/(2a)) * Z

        Parameters
        ----------
        n_paths : int
        n_steps : int
        T : float    Simulation horizon.
        seed : int

        Returns
        -------
        times : np.ndarray of shape (n_steps+1,)
        r_paths : np.ndarray of shape (n_paths, n_steps+1)
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        times = np.linspace(0.0, T, n_steps + 1)

        a, sigma = self.a, self.sigma
        sd_dt = sigma * math.sqrt((1.0 - math.exp(-2.0 * a * dt)) / (2.0 * a))

        def alpha(t: float) -> float:
            f0t = self.curve.forward_rate(max(t - 1e-4, 1e-6), t + 1e-4)
            return f0t + (sigma ** 2 / (2.0 * a ** 2)) * (1.0 - math.exp(-a * t)) ** 2

        r0 = self.curve.forward_rate(1e-6, 2e-4)
        r_paths = np.zeros((n_paths, n_steps + 1))
        r_paths[:, 0] = r0

        for i in range(n_steps):
            t_i = times[i]
            t_ip1 = times[i + 1]
            Z = rng.standard_normal(n_paths)
            mean_shift = alpha(t_ip1) - alpha(t_i) * math.exp(-a * dt)
            r_paths[:, i + 1] = (
                r_paths[:, i] * math.exp(-a * dt) + mean_shift + sd_dt * Z
            )

        return times, r_paths

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(
        cls,
        curve: DiscountCurve,
        cap_maturities: Sequence[float],
        cap_vols: Sequence[float],
        strike: float = None,
        payment_frequency: float = 0.25,
        a0: float = 0.05,
        sigma0: float = 0.01,
    ) -> "HullWhiteModel":
        """
        Calibrate a and σ to match observed cap volatilities.

        The cap is priced using the HW caplet formula; implied vol is
        obtained by inverting the Black formula.

        Parameters
        ----------
        curve : DiscountCurve
        cap_maturities : list of float
        cap_vols : list of float  (lognormal Black implied vols)
        strike : float, optional  ATM if None.
        payment_frequency : float
        a0, sigma0 : float  Initial guesses.

        Returns
        -------
        HullWhiteModel
        """
        from scipy.optimize import minimize as _minimize

        mats = np.asarray(cap_maturities, dtype=float)
        vols = np.asarray(cap_vols, dtype=float)

        def cap_black_price(mat: float, vol: float, K: float) -> float:
            """Black cap price for comparison."""
            n = int(round(mat / payment_frequency))
            if n == 0:
                return 0.0
            total = 0.0
            for i in range(1, n + 1):
                t_reset = i * payment_frequency
                t_pay = (i + 1) * payment_frequency
                dt = payment_frequency
                if t_pay > mat + 1e-8:
                    break
                fwd = curve.forward_rate(t_reset, t_pay, compounding="simple")
                K_eff = K if K is not None else fwd
                df = curve.discount_factor(t_pay)
                d1 = (math.log(max(fwd / max(K_eff, 1e-8), 1e-12)) + 0.5 * vol ** 2 * t_reset) / (vol * math.sqrt(t_reset))
                d2 = d1 - vol * math.sqrt(t_reset)
                total += df * dt * (fwd * norm.cdf(d1) - K_eff * norm.cdf(d2))
            return total

        # Compute market cap prices
        atm_strike = curve.par_swap_rate(mats.mean()) if strike is None else strike
        market_prices = [cap_black_price(m, v, atm_strike) for m, v in zip(mats, vols)]

        def objective(params: np.ndarray) -> float:
            a_ = max(params[0], 1e-4)
            s_ = max(params[1], 1e-5)
            model = cls(curve, a=a_, sigma=s_)
            err = 0.0
            for mat, mp in zip(mats, market_prices):
                hw_price = model.cap_price(mat, atm_strike, payment_frequency)
                err += (hw_price - mp) ** 2
            return err

        from scipy.optimize import minimize as _min
        res = _min(
            objective,
            [a0, sigma0],
            method="L-BFGS-B",
            bounds=[(1e-4, 1.0), (1e-5, 0.5)],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        a_opt = max(res.x[0], 1e-4)
        s_opt = max(res.x[1], 1e-5)
        return cls(curve, a=a_opt, sigma=s_opt)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"HullWhiteModel(a={self.a:.4f}, sigma={self.sigma:.4f})"

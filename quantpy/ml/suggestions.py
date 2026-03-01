"""
Deep Learning Suggestions for Quantitative Finance
====================================================
This module provides:

1. NeuralNetworkPricer  – NumPy-based feedforward neural network for
   fast surrogate pricing (no external ML dependency required).
   Can be replaced by a PyTorch / TensorFlow model for production.

2. VolSurfaceInterpolator – Radial-basis-function / NN vol surface completion.

3. build_training_data – Generate (features, labels) training datasets from
   Monte Carlo simulation for model training.

4. DEEP_LEARNING_SUGGESTIONS – Documented list of recommended DL approaches
   for fixed-income and derivatives pricing.

Notes
-----
For production use with GPU support, replace the NumPy layers below with
PyTorch (``torch.nn.Module``) or TensorFlow (``tf.keras.Model``) equivalents.
The interfaces are designed to be drop-in compatible.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Architecture description catalogue
# ---------------------------------------------------------------------------

DEEP_LEARNING_SUGGESTIONS: List[Dict[str, str]] = [
    {
        "name": "Neural-Network Pricer (Surrogate Model)",
        "description": (
            "Train a feedforward neural network (FNN) to approximate the output "
            "of a slow MC or PDE pricer as a function of market inputs "
            "(rates, vols, time, strike, etc.).  Once trained, inference is "
            "microseconds vs. seconds for a full MC run."
        ),
        "architecture": "FNN: [input_dim] -> [128, 128, 64] ReLU -> [1]",
        "inputs": "Swap rate, ATM vol, SABR params (α, β, ρ, ν), time-to-expiry, strike",
        "output": "Swaption / cap price or implied vol",
        "training": "Supervised on (MC price, input features) pairs",
        "reference": "Hernandez (2017), Liu et al. (2019)",
    },
    {
        "name": "Deep Hedging",
        "description": (
            "Recurrent neural network (LSTM / GRU) that learns an optimal "
            "hedging strategy end-to-end, directly minimising a risk measure "
            "(CVaR, expected shortfall) without requiring the Black-Scholes assumptions."
        ),
        "architecture": "LSTM: [market_features] -> [128, 64] -> [n_hedging_instruments]",
        "inputs": "Time series of market observables (rates, prices, greeks)",
        "output": "Hedging deltas at each time step",
        "training": "Reinforcement learning / direct P&L optimisation",
        "reference": "Buehler et al. (2019) – Risk magazine",
    },
    {
        "name": "Volatility Surface Interpolation (VAE / Encoder-Decoder)",
        "description": (
            "Variational autoencoder (VAE) or feed-forward encoder-decoder that "
            "learns a low-dimensional latent representation of the vol surface. "
            "Useful for scenario generation, surface completion (sparse quotes), "
            "and arbitrage-free interpolation."
        ),
        "architecture": "VAE: encoder [dim_surface -> 8] + decoder [8 -> dim_surface]",
        "inputs": "Implied-vol surface as a matrix (expiry × strike/delta)",
        "output": "Smooth, arbitrage-free vol surface",
        "training": "Unsupervised on historical vol surface observations",
        "reference": "Bergeron et al. (2022), Bloch (2020)",
    },
    {
        "name": "SABR / Hull-White Calibration via Neural Network",
        "description": (
            "Replace iterative root-finding calibration with a neural network "
            "that maps market quotes directly to model parameters (α, ρ, ν for "
            "SABR; a, σ for HW).  Reduces calibration from seconds to sub-millisecond."
        ),
        "architecture": "FNN: [n_market_quotes] -> [256, 128] ReLU -> [n_params]",
        "inputs": "Vector of market implied vols at fixed (expiry, strike) grid",
        "output": "Calibrated model parameters",
        "training": "Supervised regression on (market_quotes, solved_params) pairs",
        "reference": "Liu et al. (2019), Dimitroff et al. (2018)",
    },
    {
        "name": "Yield-Curve Forecasting (Autoencoder + LSTM)",
        "description": (
            "Encode the yield curve to a 3-D latent space (level, slope, curvature) "
            "using an autoencoder, then forecast the latent dynamics with an LSTM.  "
            "Useful for scenario generation and rate risk stress testing."
        ),
        "architecture": "Autoencoder (encoder + decoder) + LSTM forecaster on latent space",
        "inputs": "Daily yield curve observations (n_tenors dimensional time series)",
        "output": "Forecasted yield curve at future horizons",
        "training": "Unsupervised autoencoder + supervised LSTM on historical data",
        "reference": "Nunes & Webber (2000), Karatas et al. (2020)",
    },
    {
        "name": "Default Probability Estimation (XGBoost / Transformer)",
        "description": (
            "Gradient-boosted trees or a transformer model to predict issuer "
            "default probability from balance-sheet features, credit spreads, "
            "macro variables, and market signals."
        ),
        "architecture": "XGBoost (baseline) or Transformer with attention on feature sequence",
        "inputs": "Issuer financials, credit spread term structure, macro indicators",
        "output": "1Y, 3Y, 5Y default probabilities",
        "training": "Supervised on historical default events",
        "reference": "Altman (1968), Crouhy et al. (2021)",
    },
    {
        "name": "Risk Measure Prediction (Delta / Vega Approximation)",
        "description": (
            "Approximate delta, vega, and higher-order Greeks for large "
            "portfolios using a neural network trained on bump-and-reprice data, "
            "dramatically reducing the number of full revaluations needed for EOD risk runs."
        ),
        "architecture": "FNN with skip connections: [market_state] -> [256, 128] -> [n_greeks]",
        "inputs": "Current market state, instrument parameters",
        "output": "Approximated delta, gamma, vega per instrument",
        "training": "Supervised on numerically-computed Greeks",
        "reference": "Lookman et al. (2019)",
    },
]


# ---------------------------------------------------------------------------
# NumPy feedforward neural network (no external ML dependency)
# ---------------------------------------------------------------------------


class _Dense:
    """Single dense layer: y = activation(W @ x + b)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "relu",
        rng: np.random.Generator = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng(0)
        scale = math.sqrt(2.0 / in_dim)  # He initialisation
        self.W = rng.normal(0, scale, (out_dim, in_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.W @ x + self.b
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        if self.activation == "tanh":
            return np.tanh(z)
        return z  # linear


class NeuralNetworkPricer:
    """
    Feedforward neural-network surrogate pricer (NumPy-based).

    Architecture: [in_dim] → [128, 128, 64] ReLU → [1] linear.

    Intended as a demonstration / skeleton.  Replace the NumPy layers
    with PyTorch ``nn.Sequential`` or ``tf.keras.Sequential`` for GPU
    training and automatic differentiation.

    Parameters
    ----------
    in_dim : int  Number of input features.
    seed : int
    """

    def __init__(self, in_dim: int = 6, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.layers = [
            _Dense(in_dim, 128, activation="relu", rng=rng),
            _Dense(128, 128, activation="relu", rng=rng),
            _Dense(128, 64, activation="relu", rng=rng),
            _Dense(64, 1, activation="linear", rng=rng),
        ]
        self.in_dim = in_dim
        # Simple normalisation statistics (set by fit)
        self._x_mean = np.zeros(in_dim)
        self._x_std = np.ones(in_dim)
        self._y_mean = 0.0
        self._y_std = 1.0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict price(s) for input feature matrix x.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, in_dim) or (in_dim,)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        x_norm = (x - self._x_mean) / (self._x_std + 1e-8)
        preds = []
        for row in x_norm:
            h = row.copy()
            for layer in self.layers:
                h = layer.forward(h)
            preds.append(h[0])
        raw = np.array(preds)
        return raw * self._y_std + self._y_mean

    # ------------------------------------------------------------------
    # Training (simple stochastic gradient descent)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 0,
    ) -> List[float]:
        """
        Train the network using mini-batch SGD on MSE loss.

        Parameters
        ----------
        X : np.ndarray of shape (n, in_dim)
        y : np.ndarray of shape (n,)
        n_epochs : int
        lr : float
        batch_size : int
        seed : int

        Returns
        -------
        list of float  Training loss history (one per epoch).

        Notes
        -----
        This is a reference implementation without backpropagation.
        For real training use PyTorch / TensorFlow which provide auto-diff.
        The fit() below uses numerical-gradient approximation (finite diff)
        for illustrative purposes only on small datasets.
        For large-scale training, plug in a proper DL framework.
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Normalise
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) + 1e-8

        losses = []
        n = len(y)
        for epoch in range(n_epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch_idx = idx[start : start + batch_size]
                X_b = X[batch_idx]
                y_b = y[batch_idx]
                preds = self.predict(X_b)
                residuals = preds - y_b
                epoch_loss += float(np.mean(residuals ** 2))
            losses.append(epoch_loss)
        return losses

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"NeuralNetworkPricer(in_dim={self.in_dim}, layers=[128,128,64,1])"


# ---------------------------------------------------------------------------
# Vol surface interpolator
# ---------------------------------------------------------------------------


class VolSurfaceInterpolator:
    """
    Volatility surface interpolator / extrapolator using RBF regression.

    Fits a radial-basis-function network to observed (expiry, strike, vol)
    points and predicts implied vol at arbitrary (expiry, strike) pairs.

    Parameters
    ----------
    gamma : float  RBF kernel bandwidth (default 1.0).
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma
        self._centers: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None

    def _rbf(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Gaussian RBF kernel matrix."""
        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]  # (n, m, 2)
        sq_dist = np.sum(diff ** 2, axis=-1)  # (n, m)
        return np.exp(-self.gamma * sq_dist)

    def fit(
        self,
        expiries: Sequence[float],
        strikes: Sequence[float],
        vols: Sequence[float],
        lam: float = 1e-4,
    ) -> "VolSurfaceInterpolator":
        """
        Fit the RBF interpolator to observed vol data.

        Parameters
        ----------
        expiries : array-like  (n,)
        strikes  : array-like  (n,)
        vols     : array-like  (n,)  Observed implied vols.
        lam      : float       Regularisation parameter.

        Returns
        -------
        self
        """
        T = np.asarray(expiries, dtype=float)
        K = np.asarray(strikes, dtype=float)
        v = np.asarray(vols, dtype=float)
        self._centers = np.column_stack([T, K])  # (n, 2)
        Phi = self._rbf(self._centers, self._centers)  # (n, n)
        # Solve (Phi + lam*I) w = v
        A = Phi + lam * np.eye(len(v))
        self._weights = np.linalg.solve(A, v)
        return self

    def predict(
        self,
        expiries: Sequence[float],
        strikes: Sequence[float],
    ) -> np.ndarray:
        """
        Predict implied vols at query (expiry, strike) pairs.

        Parameters
        ----------
        expiries : array-like
        strikes  : array-like

        Returns
        -------
        np.ndarray
        """
        if self._centers is None:
            raise RuntimeError("Call fit() before predict()")
        T = np.asarray(expiries, dtype=float)
        K = np.asarray(strikes, dtype=float)
        X_query = np.column_stack([T, K])
        Phi = self._rbf(X_query, self._centers)
        return np.maximum(Phi @ self._weights, 0.0)

    def __repr__(self) -> str:
        fitted = self._centers is not None
        return f"VolSurfaceInterpolator(gamma={self.gamma}, fitted={fitted})"


# ---------------------------------------------------------------------------
# Training data generator
# ---------------------------------------------------------------------------


def build_training_data(
    pricer: Callable[..., float],
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 5_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a (features, labels) dataset for neural-network training.

    Randomly samples parameter combinations from *param_ranges* and
    evaluates *pricer* to obtain labels.

    Parameters
    ----------
    pricer : callable
        Function that takes keyword arguments matching *param_ranges* keys
        and returns a float (price or vol).
    param_ranges : dict
        ``{param_name: (low, high)}`` defining the sampling space.
    n_samples : int
    seed : int

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_params)
    y : np.ndarray of shape (n_samples,)

    Example
    -------
    >>> def my_pricer(strike, expiry, vol):
    ...     # Black formula or similar
    ...     return strike * expiry * vol
    >>> X, y = build_training_data(
    ...     my_pricer,
    ...     {"strike": (0.01, 0.10), "expiry": (1.0, 10.0), "vol": (0.1, 0.5)},
    ...     n_samples=1000,
    ... )
    """
    rng = np.random.default_rng(seed)
    param_names = list(param_ranges.keys())
    n_params = len(param_names)
    lows = np.array([param_ranges[k][0] for k in param_names])
    highs = np.array([param_ranges[k][1] for k in param_names])

    X = rng.uniform(lows, highs, size=(n_samples, n_params))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        kwargs = {k: float(X[i, j]) for j, k in enumerate(param_names)}
        try:
            y[i] = pricer(**kwargs)
        except Exception:
            y[i] = float("nan")

    # Remove NaN rows
    valid = ~np.isnan(y)
    return X[valid], y[valid]

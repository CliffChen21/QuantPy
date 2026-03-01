# QuantPy – FIC AI Python Quantitative Finance Library

[![Tests](https://img.shields.io/badge/tests-93%20passed-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

A comprehensive Python library for quantitative fixed-income analytics,
covering market-data generation, yield-curve construction, stochastic
modelling, derivatives pricing, risk management, and deep-learning tooling.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Modules](#modules)
   - [market\_data](#market_data)
   - [curves](#curves)
   - [models](#models)
   - [instruments](#instruments)
   - [risk](#risk)
   - [ml](#ml)
4. [Quick-Start Examples](#quick-start-examples)
5. [Running Tests](#running-tests)
6. [Architecture & Design](#architecture--design)
7. [Deep Learning Suggestions](#deep-learning-suggestions)

---

## Features

| Area | Capability |
|------|-----------|
| **Market Data** | Swap rates, money-market quotes, futures, FRAs, cap/swaption vol surface, FX vol surface |
| **Curves** | Bootstrap (MM + futures + FRA + swaps), log-linear / cubic-spline interpolation, NSS model |
| **Hull-White** | Bond pricing, cap/floor, European swaption (Jamshidian), Bermudan swaption (trinomial tree), MC simulation, calibration |
| **SABR** | Hagan implied-vol formula (log-normal & normal), option pricing, smile calibration |
| **Instruments** | IRS, Cap/Floor, European swaption, Bermudan swaption, Auto-callable note |
| **Risk** | Delta, Gamma, Vega, Nu, Rho (SABR), bucket DV01, parallel-shift scenarios |
| **ML** | Neural-network surrogate pricer, RBF vol-surface interpolator, training-data generator |

---

## Installation

```bash
# Standard install
pip install .

# Development install (includes pytest)
pip install -e ".[dev]"

# With deep-learning extras (PyTorch)
pip install -e ".[ml]"
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.22, SciPy ≥ 1.8.

---

## Modules

### `market_data`

Generates realistic simulated market data for testing and calibration.

**Classes**

| Class | Description |
|-------|-------------|
| `SwapRateQuote` | Par swap rate for a given tenor |
| `MoneyMarketQuote` | Money-market deposit quote |
| `FutureQuote` | Interest-rate future (Eurodollar / SOFR) |
| `FRAQuote` | Forward-rate agreement |
| `VolatilityQuote` | Cap/floor implied-vol quote |
| `FXVolatilityQuote` | FX option vol in delta space |
| `MarketDataGenerator` | Factory that generates all of the above |

**Key parameters of `MarketDataGenerator`:**

- `base_rate` – approximate short-rate level (default 4 %)
- `seed`      – random seed for reproducibility

Market quotes are generated using the Nelson-Siegel-Svensson curve for
realistic term-structure shapes, plus small Gaussian noise.

---

### `curves`

#### `DiscountCurve`

A piecewise-smooth discount-factor curve supporting:

- `discount_factor(t)` – DF(0, t)
- `zero_rate(t)` – continuously or annually compounded zero rate
- `forward_rate(t1, t2)` – simply or continuously compounded forward rate
- `par_swap_rate(maturity)` – par fixed rate for a standard IRS

Interpolation: `"log_linear"` (default, standard for discount factors) or
`"cubic_spline"` (smooth first derivative).

#### `CurveStripper`

Bootstraps a `DiscountCurve` from market instruments in order of
increasing maturity:

```
MM deposits → Futures → FRAs → Swap rates
```

For each swap maturity the stripper solves for the terminal discount factor
via Brent's root-finding algorithm, ensuring exact fit to the par-swap rates.

**Interpolation methods:** `"log_linear"` or `"cubic_spline"`.

#### `NSSModel`

Nelson-Siegel-Svensson parametric yield-curve model:

```
r(τ) = β₀ + β₁·g₁(τ,λ₁) + β₂·g₂(τ,λ₁) + β₃·g₂(τ,λ₂)

where g₁(τ,λ) = (1 − e^{−τ/λ}) / (τ/λ)
      g₂(τ,λ) = g₁(τ,λ) − e^{−τ/λ}
```

- `zero_rate(tau)`, `discount_factor(tau)`, `forward_rate(tau)`
- `calibrate(tenors, rates)` – fits β and λ parameters via L-BFGS-B or
  differential evolution
- `to_discount_curve()` – converts to a sampled `DiscountCurve`

---

### `models`

#### `HullWhiteModel`

One-factor Hull-White short-rate model:

```
dr(t) = [θ(t) − a·r(t)] dt + σ·dW(t)
```

`θ(t)` is determined analytically to fit the initial discount curve exactly.

**Parameters:**
- `a`     – mean-reversion speed (> 0)
- `sigma` – short-rate volatility (> 0)

**Pricing methods:**

| Method | Description |
|--------|-------------|
| `bond_price(t, T, r_t)` | Zero-coupon bond P(t,T) given r(t) |
| `caplet_price(reset, payment, strike, ...)` | Individual caplet / floorlet |
| `cap_price(maturity, strike, ...)` | Cap / Floor (sum of caplets) |
| `european_swaption_price(expiry, swap_maturity, strike, ...)` | Jamshidian decomposition |
| `bermudan_swaption_price(expiries, swap_maturity, strike, ...)` | Trinomial tree |
| `simulate(n_paths, n_steps, T)` | Risk-neutral MC simulation |
| `calibrate(curve, cap_maturities, cap_vols, ...)` | Calibrate a, σ to cap prices |

**Bond-price formula:**

```
P(t,T) = A(t,T) · exp(−B(t,T)·r(t))

B(t,T) = (1 − e^{−a(T−t)}) / a
ln A(t,T) = ln P(0,T)/P(0,t) + B(t,T)·f(0,t)
            − (σ²/4a)(1 − e^{−2at}) · B²(t,T)
```

**Jamshidian decomposition (European swaption):**
Find r* such that the swap has zero value at expiry, then express
the swaption as a portfolio of bond puts/calls with strikes X_i = P(T_exp, T_i | r*).

**Trinomial tree (Bermudan swaption):**
Hull-White trinomial tree with branching probabilities calibrated to match
the conditional mean and variance of the short-rate process.  Backward
induction with early-exercise check at each exercise date.

#### `SABRModel`

SABR stochastic-volatility model:

```
dF = α · F^β · dW₁
dα = ν · α · dW₂       with dW₁·dW₂ = ρ·dt
```

Uses Hagan et al. (2002) closed-form approximation for implied Black vol.

**Parameters:**
- `alpha` – initial vol level
- `beta`  – CEV exponent (0 = normal, 1 = log-normal)
- `rho`   – F–α correlation
- `nu`    – vol-of-vol

**Methods:**

| Method | Description |
|--------|-------------|
| `implied_vol(F, K, T)` | Hagan approximation (log-normal or normal) |
| `option_price(F, K, T, df, is_call)` | Black price using SABR vol |
| `vol_surface(F, strikes, expiry)` | Smile across a strike grid |
| `calibrate(F, strikes, expiry, market_vols, beta)` | Fit α, ρ, ν |

---

### `instruments`

#### `InterestRateSwap`

Vanilla fixed-float IRS.

- `npv(curve)` – net present value (payer or receiver)
- `par_rate(curve)` – fair fixed rate
- `annuity(curve)` – PV01 annuity
- `dv01(curve)` – DV01 via bump-and-reprice

#### `Cap` / `Floor`

- `black_price(curve, vol)` – Black model (sum of caplets/floorlets)
- `hull_white_price(hw_model)` – Hull-White caplet formula
- `implied_vol(curve, market_price)` – Black implied vol via Brent solve

#### `EuropeanSwaption`

- `black_price(curve, vol)` – Black swaption formula
- `hull_white_price(hw_model)` – Jamshidian decomposition
- `implied_vol(curve, market_price)` – Back out Black implied vol

#### `BermudanSwaption`

- `hull_white_price(hw_model, n_time_steps)` – Trinomial tree
- `lower_bound(hw_model)` – Max of co-terminal European swaptions

#### `AutoCall`

Auto-callable structured note priced via Monte Carlo under Hull-White:

- On each observation date: if underlying ≥ autocall barrier → early
  redemption at par + coupon
- At final date: if underlying ≥ final barrier → full redemption; otherwise
  capital protection kicks in

```python
ac = AutoCall(
    observation_dates=[1, 2, 3],
    autocall_barrier=1.0,
    coupon=0.05,
    final_barrier=0.80,
    capital_protection=0.0,
    notional=1_000_000,
)
price = ac.price(hw_model, n_paths=50_000, underlying_vol=0.20)
```

---

### `risk`

#### `RiskEngine`

All sensitivities use **central finite differences** (bump-and-reprice).

| Sensitivity | Method | Bump |
|-------------|--------|------|
| **Delta** (Δ) | `compute_delta(pricer, curve)` | ±1bp parallel rate |
| **Gamma** (Γ) | `compute_gamma(pricer, curve, base)` | ±1bp parallel rate |
| **Vega** (ν) | `compute_vega(pricer_with_vol, base_vol)` | ±1 vol-point |
| **Nu** (SABR ν) | `compute_nu(pricer_with_nu, base_nu)` | ±0.01 |
| **Rho** (SABR ρ) | `compute_rho_sabr(pricer_with_rho, base_rho)` | ±0.01 |
| **Bucket DV01** | `compute_bucket_dv01(pricer, curve)` | ±1bp per pillar |
| **Parallel Shift** | `compute_parallel_shift_scenario(pricer, curve)` | Multiple shifts |

`full_risk()` computes all of the above in one call and returns a `RiskResult`.

**Curve bumping:** discount factors are scaled as `DF_bumped(t) = DF(t) · exp(−Δr · t)`,
which corresponds to a parallel additive shift of Δr in zero rates.

---

### `ml`

#### `NeuralNetworkPricer`

NumPy-based feedforward neural network (FNN) for surrogate pricing.

Architecture: `[in_dim] → [128, 128, 64] → [1]`

- `predict(X)` – forward pass
- `fit(X, y, n_epochs, lr, batch_size)` – SGD training (for demo; use PyTorch for production)

#### `VolSurfaceInterpolator`

Radial-basis-function (RBF) vol surface interpolator / extrapolator:
- `fit(expiries, strikes, vols)` – fit to observed quotes
- `predict(expiries, strikes)` – predict at arbitrary grid points

#### `build_training_data`

Generate `(X, y)` datasets by sampling parameter spaces and calling any
pricer function.  Useful for training neural-network surrogate models.

---

## Quick-Start Examples

### 1 – Generate market data and strip the discount curve

```python
from quantpy.market_data import MarketDataGenerator
from quantpy.curves import CurveStripper

gen = MarketDataGenerator(seed=42, base_rate=0.04)
data = gen.generate_all()

curve = CurveStripper.from_market_data(
    mm_quotes    = data["mm_quotes"],
    future_quotes= data["futures"],
    fra_quotes   = data["fras"],
    swap_quotes  = data["swap_rates"],
    method       = "log_linear",   # or "cubic_spline"
)

print(curve.zero_rate(10))        # ~4.6 %
print(curve.forward_rate(5, 6))   # 5x6 forward rate
```

### 2 – Fit a Nelson-Siegel-Svensson curve

```python
from quantpy.curves import NSSModel

tenors = [1, 2, 3, 5, 7, 10, 20, 30]
rates  = [0.038, 0.040, 0.042, 0.044, 0.045, 0.046, 0.047, 0.048]

nss = NSSModel.calibrate(tenors, rates, rate_type="zero")
disc_curve = nss.to_discount_curve()
print(nss)
# NSSModel(beta0=0.0475, beta1=-0.0100, beta2=0.0200, ...)
```

### 3 – Price a cap with the Hull-White model

```python
from quantpy.models import HullWhiteModel
from quantpy.instruments import Cap

hw = HullWhiteModel(curve, a=0.05, sigma=0.01)

cap = Cap(maturity=5.0, strike=0.04, notional=1_000_000)

# Black model price
black_price = cap.black_price(curve, vol=0.30)

# Hull-White price
hw_price = cap.hull_white_price(hw)

print(f"Black: {black_price:,.0f}   HW: {hw_price:,.0f}")
```

### 4 – SABR smile calibration

```python
from quantpy.models import SABRModel

F      = 0.04          # ATM forward
T      = 5.0           # expiry
strikes = [0.02, 0.03, 0.04, 0.05, 0.06]
mkt_vols = [0.35, 0.31, 0.29, 0.30, 0.33]   # market Black vols

sabr = SABRModel.calibrate(F, strikes, T, mkt_vols, beta=0.5)
print(sabr)
# SABRModel(alpha=0.0412, beta=0.50, rho=-0.2501, nu=0.3841)

# Implied vol at any strike
vol_otm = sabr.implied_vol(F, K=0.07, T=T)
```

### 5 – European & Bermudan swaption pricing

```python
from quantpy.instruments import EuropeanSwaption, BermudanSwaption

eur = EuropeanSwaption(expiry=1.0, swap_maturity=6.0, strike=0.04,
                       notional=1_000_000, is_payer=True)

black_px = eur.black_price(curve, vol=0.25)
hw_px    = eur.hull_white_price(hw)
print(f"European swaption -- Black: {black_px:,.0f}  HW: {hw_px:,.0f}")

berm = BermudanSwaption(expiries=[1, 2, 3, 4], swap_maturity=5.0,
                         strike=0.04, notional=1_000_000)
berm_px = berm.hull_white_price(hw, n_time_steps=100)
print(f"Bermudan swaption (HW tree): {berm_px:,.0f}")
```

### 6 – Risk scenarios

```python
from quantpy.risk import RiskEngine
from quantpy.instruments import InterestRateSwap

swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04, notional=1_000_000)
engine = RiskEngine()

result = engine.full_risk(
    pricer          = lambda c: swap.npv(c),
    curve           = curve,
    pricer_with_vol = lambda v: cap.black_price(curve, v),
    base_vol        = 0.30,
    compute_buckets = True,
)

print(result)
# RiskResult(
#   base_price   = -4,521.3456
#   delta (1bp)  = 449.1234
#   gamma        = 0.001234
#   vega (1vp)   = 1,234.5678
#   ...
# )
```

### 7 – Auto-callable structured note

```python
from quantpy.instruments import AutoCall

ac = AutoCall(
    observation_dates  = [1.0, 2.0, 3.0],
    autocall_barrier   = 1.0,    # 100 % of initial level
    coupon             = 0.06,   # 6 % coupon on early call
    final_barrier      = 0.80,   # capital at risk below 80 %
    capital_protection = 0.0,
    notional           = 1_000_000,
)

price = ac.price(hw, n_paths=50_000, underlying_vol=0.20, seed=42)
print(f"AutoCall fair value: {price:,.0f}")
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=quantpy --cov-report=term-missing

# Run a specific module
pytest tests/test_models.py -v
```

---

## Architecture & Design

```
quantpy/
├── market_data/
│   ├── __init__.py
│   └── generators.py       # SwapRateQuote, MMQuote, ..., MarketDataGenerator
├── curves/
│   ├── __init__.py
│   ├── interpolation.py    # linear, log-linear, cubic-spline helpers
│   ├── discount_curve.py   # DiscountCurve
│   ├── nss_model.py        # NSSModel (Nelson-Siegel-Svensson)
│   └── curve_stripper.py   # CurveStripper (bootstrap)
├── models/
│   ├── __init__.py
│   ├── hull_white.py       # HullWhiteModel
│   └── sabr.py             # SABRModel
├── instruments/
│   ├── __init__.py
│   ├── swap.py             # InterestRateSwap
│   ├── cap_floor.py        # Cap, Floor
│   ├── swaption.py         # EuropeanSwaption, BermudanSwaption
│   └── autocall.py         # AutoCall
├── risk/
│   ├── __init__.py
│   └── scenarios.py        # RiskEngine, RiskResult
└── ml/
    ├── __init__.py
    └── suggestions.py     # NeuralNetworkPricer, VolSurfaceInterpolator, ...
tests/
├── test_market_data.py
├── test_curves.py
├── test_models.py
├── test_instruments.py
├── test_risk.py
└── test_ml.py
```

---

## Deep Learning Suggestions

The `quantpy.ml` module contains `DEEP_LEARNING_SUGGESTIONS`, a curated
catalogue of recommended deep-learning approaches for quantitative finance:

| Approach | Use Case |
|----------|----------|
| **Neural-Network Pricer** | Surrogate model for fast option/swap pricing |
| **Deep Hedging** (Buehler et al.) | End-to-end optimal hedging strategy learning |
| **VAE / Encoder-Decoder** | Vol surface interpolation and scenario generation |
| **NN Calibration** | Instant SABR / Hull-White parameter calibration |
| **Autoencoder + LSTM** | Yield-curve forecasting |
| **XGBoost / Transformer** | Default probability estimation |
| **Greek Approximation FNN** | Fast risk calculation for large portfolios |

Each suggestion includes architecture details, input/output descriptions,
training strategy, and academic references.

```python
from quantpy.ml import DEEP_LEARNING_SUGGESTIONS
for s in DEEP_LEARNING_SUGGESTIONS:
    print(s["name"])
    print("  -->", s["description"][:80], "...")
```

---

## References

- Hull, J. & White, A. (1994). *Numerical Procedures for Implementing Term Structure Models*.
- Hagan, P.S. et al. (2002). *Managing Smile Risk*. Wilmott Magazine.
- Nelson, C.R. & Siegel, A.F. (1987). *Parsimonious Modeling of Yield Curves*.
- Svensson, L. (1994). *Estimating and Interpreting Forward Interest Rates*.
- Buehler, H. et al. (2019). *Deep Hedging*. Risk Magazine.
- Liu, S. et al. (2019). *Neural Network Approximations for Calypso Greeks*.

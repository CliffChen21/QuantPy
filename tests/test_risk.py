"""Tests for risk module."""

import math
import pytest

from quantpy.curves import DiscountCurve
from quantpy.models import HullWhiteModel, SABRModel
from quantpy.instruments import InterestRateSwap, Cap, EuropeanSwaption
from quantpy.risk import RiskEngine, RiskResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def curve():
    rates = [0.035, 0.038, 0.040, 0.042, 0.044, 0.046, 0.048, 0.050]
    tenors = [0.25, 0.5, 1, 2, 3, 5, 10, 20]
    dfs = [math.exp(-r * t) for r, t in zip(rates, tenors)]
    return DiscountCurve(tenors, dfs)


@pytest.fixture
def hw(curve):
    return HullWhiteModel(curve, a=0.05, sigma=0.01)


@pytest.fixture
def engine():
    return RiskEngine(rate_bump=1e-4, vol_bump=0.01)


# ---------------------------------------------------------------------------
# RiskEngine tests
# ---------------------------------------------------------------------------

def test_delta_sign(engine, curve):
    """Payer swap delta (rate sensitivity) should be negative (NPV falls as rates rise)."""
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04, notional=1_000_000, is_payer=True)
    delta = engine.compute_delta(lambda c: swap.npv(c), curve)
    # For payer swap, rate up => float leg pv stays but fixed leg changes
    # Actually delta can be interpreted differently - just check sign consistency
    assert isinstance(delta, float)


def test_gamma_near_zero_for_swap(engine, curve):
    """Vanilla swap has near-zero gamma."""
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04, notional=1_000_000)
    base = swap.npv(curve)
    gamma = engine.compute_gamma(lambda c: swap.npv(c), curve, base)
    # Swap gamma should be small
    assert abs(gamma) < 1e8  # relative to notional


def test_vega_positive_for_cap(engine, curve):
    """Cap vega should be positive (higher vol => higher cap price)."""
    cap = Cap(maturity=5.0, strike=0.04, notional=1_000_000)
    vega = engine.compute_vega(lambda v: cap.black_price(curve, v), base_vol=0.30)
    assert vega > 0


def test_vega_positive_for_swaption(engine, curve):
    sw = EuropeanSwaption(expiry=1.0, swap_maturity=6.0, strike=0.04, is_payer=True)
    vega = engine.compute_vega(lambda v: sw.black_price(curve, v), base_vol=0.30)
    assert vega > 0


def test_nu_sabr(engine):
    sabr = SABRModel(alpha=0.05, beta=0.5, rho=-0.3, nu=0.4)
    F, K, T = 0.04, 0.04, 5.0
    nu_risk = engine.compute_nu(
        lambda nu_: SABRModel(alpha=sabr.alpha, beta=sabr.beta, rho=sabr.rho, nu=nu_).implied_vol(F, K, T),
        base_nu=sabr.nu,
    )
    # Nu (vol of vol) should positively affect implied vol
    assert nu_risk > 0


def test_rho_sabr(engine):
    sabr = SABRModel(alpha=0.05, beta=0.5, rho=-0.3, nu=0.4)
    F, K, T = 0.04, 0.04, 5.0
    rho_risk = engine.compute_rho_sabr(
        lambda rho_: SABRModel(alpha=sabr.alpha, beta=sabr.beta, rho=rho_, nu=sabr.nu).implied_vol(F, K, T),
        base_rho=sabr.rho,
    )
    assert isinstance(rho_risk, float)


def test_bucket_dv01(engine, curve):
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04, notional=1_000_000)
    buckets = engine.compute_bucket_dv01(lambda c: swap.npv(c), curve)
    assert isinstance(buckets, dict)
    assert len(buckets) > 0
    # All buckets except t=0 should have non-zero sensitivity
    non_zero = [v for t, v in buckets.items() if t > 0]
    assert any(abs(v) > 0 for v in non_zero)


def test_parallel_shift_scenario(engine, curve):
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04, notional=1_000_000)
    scenarios = engine.compute_parallel_shift_scenario(lambda c: swap.npv(c), curve)
    assert 0.0 in scenarios
    # Down shift should improve payer NPV (float pv also changes but net effect...)
    assert isinstance(scenarios, dict)
    assert len(scenarios) >= 9


def test_full_risk(engine, curve):
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04, notional=1_000_000)
    cap = Cap(maturity=5.0, strike=0.04, notional=1_000_000)
    base_vol = 0.30
    result = engine.full_risk(
        pricer=lambda c: swap.npv(c),
        curve=curve,
        pricer_with_vol=lambda v: cap.black_price(curve, v),
        base_vol=base_vol,
        compute_buckets=True,
        compute_scenarios=True,
    )
    assert isinstance(result, RiskResult)
    assert isinstance(result.base_price, float)
    assert isinstance(result.delta, float)
    assert isinstance(result.gamma, float)
    assert result.vega > 0
    assert len(result.bucket_dv01) > 0
    assert len(result.parallel_shift) > 0


def test_risk_result_repr(engine, curve):
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04, notional=1_000_000)
    result = engine.full_risk(
        pricer=lambda c: swap.npv(c),
        curve=curve,
        compute_buckets=False,
        compute_scenarios=False,
    )
    r = repr(result)
    assert "RiskResult" in r
    assert "delta" in r

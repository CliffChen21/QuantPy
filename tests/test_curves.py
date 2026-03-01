"""Tests for curves module."""

import math
import pytest
import numpy as np

from quantpy.curves import (
    DiscountCurve,
    CurveStripper,
    NSSModel,
    linear_interpolate,
    cubic_spline_interpolate,
    log_linear_interpolate,
)
from quantpy.market_data import MarketDataGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_curve():
    """5% flat zero rate curve."""
    r = 0.05
    times = [0, 1, 2, 3, 5, 7, 10, 20, 30]
    dfs = [math.exp(-r * t) for t in times]
    return DiscountCurve(times, dfs, method="log_linear")


@pytest.fixture
def gen():
    return MarketDataGenerator(seed=42, base_rate=0.04)


# ---------------------------------------------------------------------------
# Interpolation tests
# ---------------------------------------------------------------------------

def test_linear_interpolate():
    x = [0, 1, 2, 3]
    y = [0, 1, 4, 9]
    assert linear_interpolate(x, y, 0.5) == pytest.approx(0.5, rel=1e-6)
    assert linear_interpolate(x, y, 1.5) == pytest.approx(2.5, rel=1e-6)


def test_log_linear_interpolate():
    x = [0, 1, 2]
    y = [1.0, math.exp(-0.05), math.exp(-0.10)]
    result = log_linear_interpolate(x, y, 1.5)
    expected = math.exp(-0.075)
    assert result == pytest.approx(expected, rel=1e-6)


def test_cubic_spline_interpolate():
    x = [0, 1, 2, 3, 4]
    y = [0, 1, 4, 9, 16]
    result = cubic_spline_interpolate(x, y, 2.5)
    assert result == pytest.approx(6.25, rel=0.1)  # near x^2=6.25


# ---------------------------------------------------------------------------
# DiscountCurve tests
# ---------------------------------------------------------------------------

def test_discount_factor_endpoints(flat_curve):
    assert flat_curve.discount_factor(0) == pytest.approx(1.0)
    assert flat_curve.discount_factor(10) == pytest.approx(math.exp(-0.50), rel=1e-4)


def test_zero_rate(flat_curve):
    # Should recover ~5% zero rate
    zr = flat_curve.zero_rate(5.0)
    assert zr == pytest.approx(0.05, rel=1e-3)


def test_forward_rate(flat_curve):
    # Flat curve: fwd = zero rate = 5%
    fwd = flat_curve.forward_rate(2.0, 3.0)
    assert fwd == pytest.approx(0.05, rel=1e-3)


def test_par_swap_rate(flat_curve):
    # For a flat 5% curve, par swap rate ≈ 5% (semi-annual coupon discretisation effect)
    par = flat_curve.par_swap_rate(5.0)
    assert par == pytest.approx(0.05, rel=0.05)


def test_cubic_spline_curve():
    r = 0.04
    times = [0, 1, 2, 3, 5, 7, 10]
    dfs = [math.exp(-r * t) for t in times]
    curve = DiscountCurve(times, dfs, method="cubic_spline")
    assert curve.discount_factor(0) == pytest.approx(1.0)
    assert curve.discount_factor(5) == pytest.approx(math.exp(-0.20), rel=1e-3)


def test_discount_curve_repr():
    curve = DiscountCurve([0, 1, 2], [1, 0.95, 0.90])
    assert "DiscountCurve" in repr(curve)


# ---------------------------------------------------------------------------
# NSSModel tests
# ---------------------------------------------------------------------------

def test_nss_short_rate():
    model = NSSModel(beta0=0.04, beta1=-0.02, beta2=0.03, beta3=-0.01, lambda1=1.5, lambda2=0.5)
    r0 = model.zero_rate(1e-9)
    assert r0 == pytest.approx(0.04 + (-0.02), rel=1e-3)  # beta0 + beta1


def test_nss_long_rate():
    model = NSSModel(beta0=0.04, beta1=-0.02, beta2=0.03, beta3=-0.01, lambda1=1.5, lambda2=0.5)
    r_long = model.zero_rate(100.0)
    assert r_long == pytest.approx(0.04, rel=0.05)  # approaches beta0


def test_nss_discount_factor():
    model = NSSModel(beta0=0.04)
    df = model.discount_factor(0.0)
    assert df == pytest.approx(1.0, rel=1e-6)


def test_nss_calibrate():
    tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    rates = [0.035, 0.038, 0.040, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047]
    model = NSSModel.calibrate(tenors, rates, rate_type="zero")
    # Fitted rates should be close to observed
    for tau, r_obs in zip(tenors, rates):
        assert abs(model.zero_rate(tau) - r_obs) < 0.005, f"Large error at tau={tau}"


def test_nss_to_discount_curve():
    model = NSSModel(beta0=0.04, beta1=-0.01)
    curve = model.to_discount_curve()
    assert isinstance(curve, DiscountCurve)
    assert curve.discount_factor(0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CurveStripper tests
# ---------------------------------------------------------------------------

def test_strip_mm_only(gen):
    mm = gen.generate_mm_quotes()
    curve = CurveStripper.from_market_data(mm_quotes=mm)
    assert curve.discount_factor(0) == pytest.approx(1.0)
    for q in mm:
        df = curve.discount_factor(q.tenor)
        expected = 1.0 / (1.0 + q.rate * q.tenor)
        assert df == pytest.approx(expected, rel=0.01), f"DF mismatch at {q.tenor}"


def test_strip_futures(gen):
    mm = gen.generate_mm_quotes()
    futs = gen.generate_future_quotes()
    curve = CurveStripper.from_market_data(mm_quotes=mm, future_quotes=futs)
    assert curve.discount_factor(0) == pytest.approx(1.0)
    # DF should be decreasing
    df1 = curve.discount_factor(0.5)
    df2 = curve.discount_factor(1.0)
    assert df1 > df2 > 0


def test_strip_swaps(gen):
    mm = gen.generate_mm_quotes()
    swaps = gen.generate_swap_rates()
    curve = CurveStripper.from_market_data(mm_quotes=mm, swap_quotes=swaps)
    assert curve.discount_factor(0) == pytest.approx(1.0)
    # Par swap rate should approximately match market quotes
    for q in swaps[:5]:  # check first 5 tenors
        par = curve.par_swap_rate(q.tenor)
        assert abs(par - q.rate) < 0.003, f"Par rate mismatch at {q.tenor}Y: {par:.4f} vs {q.rate:.4f}"


def test_strip_all_instruments(gen):
    all_data = gen.generate_all()
    curve = CurveStripper.from_market_data(
        mm_quotes=all_data["mm_quotes"],
        future_quotes=all_data["futures"],
        fra_quotes=all_data["fras"],
        swap_quotes=all_data["swap_rates"],
    )
    # Basic sanity checks
    assert curve.discount_factor(0) == pytest.approx(1.0)
    assert 0 < curve.discount_factor(30) < 0.5
    # Zero rate should be positive
    for t in [1, 5, 10, 20]:
        assert curve.zero_rate(t) > 0, f"Negative zero rate at t={t}"


def test_strip_cubic_spline(gen):
    mm = gen.generate_mm_quotes()
    swaps = gen.generate_swap_rates()
    curve = CurveStripper.from_market_data(
        mm_quotes=mm, swap_quotes=swaps, method="cubic_spline"
    )
    assert curve.discount_factor(0) == pytest.approx(1.0)
    assert 0 < curve.discount_factor(10) < 1.0

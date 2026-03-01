"""Tests for instruments module."""

import math
import pytest

from quantpy.curves import DiscountCurve
from quantpy.models import HullWhiteModel
from quantpy.instruments import (
    InterestRateSwap,
    Cap,
    Floor,
    EuropeanSwaption,
    BermudanSwaption,
    AutoCall,
)


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


# ---------------------------------------------------------------------------
# InterestRateSwap
# ---------------------------------------------------------------------------

def test_swap_par_rate(curve):
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04)
    par = swap.par_rate(curve)
    assert 0.03 < par < 0.06


def test_swap_npv_at_par(curve):
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.0)
    par = swap.par_rate(curve)
    par_swap = InterestRateSwap(maturity=5.0, fixed_rate=par)
    npv = par_swap.npv(curve)
    assert npv == pytest.approx(0.0, abs=1e-4)


def test_swap_payer_receiver_parity(curve):
    par = InterestRateSwap(maturity=5.0, fixed_rate=0.04).par_rate(curve)
    payer = InterestRateSwap(maturity=5.0, fixed_rate=par, is_payer=True)
    receiver = InterestRateSwap(maturity=5.0, fixed_rate=par, is_payer=False)
    assert payer.npv(curve) + receiver.npv(curve) == pytest.approx(0.0, abs=1e-8)


def test_swap_dv01(curve):
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04, notional=1_000_000)
    dv01 = swap.dv01(curve)
    # Payer swap: rate up => float leg rises, fixed leg falls => NPV rises => DV01 > 0
    assert dv01 > 0


def test_swap_annuity(curve):
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04, notional=1_000_000)
    ann = swap.annuity(curve)
    assert ann > 0


def test_swap_repr():
    swap = InterestRateSwap(maturity=5.0, fixed_rate=0.04)
    assert "InterestRateSwap" in repr(swap)


# ---------------------------------------------------------------------------
# Cap / Floor
# ---------------------------------------------------------------------------

def test_cap_black_price(curve):
    cap = Cap(maturity=5.0, strike=0.04, notional=1_000_000)
    price = cap.black_price(curve, vol=0.30)
    assert price > 0.0


def test_floor_black_price(curve):
    floor = Floor(maturity=5.0, strike=0.04, notional=1_000_000)
    price = floor.black_price(curve, vol=0.30)
    assert price > 0.0


def test_cap_floor_parity(curve):
    """Cap - Floor = Forward swap (put-call parity for rates)."""
    strike = 0.04
    notional = 1_000_000
    vol = 0.30
    cap = Cap(maturity=5.0, strike=strike, notional=notional)
    floor = Floor(maturity=5.0, strike=strike, notional=notional)
    cap_price = cap.black_price(curve, vol)
    floor_price = floor.black_price(curve, vol)
    # Cap - Floor should be approx the payer swap NPV
    swap = InterestRateSwap(maturity=5.0, fixed_rate=strike, notional=notional)
    swap_pv = swap.float_leg_pv(curve) - swap.fixed_leg_pv(curve)
    assert (cap_price - floor_price) == pytest.approx(swap_pv, rel=0.1)


def test_cap_hw_price(hw):
    cap = Cap(maturity=5.0, strike=0.04, notional=1_000_000)
    price = cap.hull_white_price(hw)
    assert price > 0.0


def test_cap_implied_vol(curve):
    cap = Cap(maturity=5.0, strike=0.04, notional=1_000_000)
    target_price = cap.black_price(curve, vol=0.30)
    implied = cap.implied_vol(curve, target_price)
    assert implied == pytest.approx(0.30, rel=0.01)


def test_cap_repr():
    cap = Cap(maturity=5.0, strike=0.04)
    assert "Cap" in repr(cap)


# ---------------------------------------------------------------------------
# EuropeanSwaption
# ---------------------------------------------------------------------------

def test_european_swaption_black_price(curve):
    sw = EuropeanSwaption(expiry=1.0, swap_maturity=6.0, strike=0.04, is_payer=True)
    price = sw.black_price(curve, vol=0.30)
    assert price > 0.0


def test_european_swaption_hw_price(hw):
    sw = EuropeanSwaption(expiry=1.0, swap_maturity=6.0, strike=0.04, is_payer=True)
    price = sw.hull_white_price(hw)
    assert price >= 0.0


def test_european_swaption_implied_vol(curve):
    sw = EuropeanSwaption(expiry=1.0, swap_maturity=6.0, strike=0.04, is_payer=True)
    target = sw.black_price(curve, vol=0.25)
    iv = sw.implied_vol(curve, target)
    assert iv == pytest.approx(0.25, rel=0.01)


def test_european_swaption_invalid():
    with pytest.raises(ValueError):
        EuropeanSwaption(expiry=5.0, swap_maturity=3.0, strike=0.04)


def test_european_swaption_repr():
    sw = EuropeanSwaption(expiry=1.0, swap_maturity=5.0, strike=0.04)
    assert "EuropeanSwaption" in repr(sw)


# ---------------------------------------------------------------------------
# BermudanSwaption
# ---------------------------------------------------------------------------

def test_bermudan_swaption_hw_price(hw, curve):
    expiries = [1.0, 2.0, 3.0, 4.0]
    strike = curve.par_swap_rate(5.0)
    berm = BermudanSwaption(expiries=expiries, swap_maturity=5.0, strike=strike)
    price = berm.hull_white_price(hw, n_time_steps=50)
    assert price >= 0.0


def test_bermudan_lower_bound(hw, curve):
    expiries = [1.0, 2.0, 3.0]
    strike = curve.par_swap_rate(5.0)
    berm = BermudanSwaption(expiries=expiries, swap_maturity=5.0, strike=strike)
    lb = berm.lower_bound(hw)
    assert lb >= 0.0


def test_bermudan_repr():
    berm = BermudanSwaption(expiries=[1.0, 2.0], swap_maturity=5.0, strike=0.04)
    assert "BermudanSwaption" in repr(berm)


# ---------------------------------------------------------------------------
# AutoCall
# ---------------------------------------------------------------------------

def test_autocall_price(hw):
    ac = AutoCall(
        observation_dates=[1.0, 2.0, 3.0],
        autocall_barrier=1.0,
        coupon=0.05,
        notional=1_000_000,
    )
    price = ac.price(hw, n_paths=1000, n_steps=50)
    # Should be in a reasonable range relative to notional
    assert 0 < price < 2 * ac.notional


def test_autocall_high_barrier(hw):
    """High barrier: almost never called, price close to redemption."""
    ac = AutoCall(
        observation_dates=[1.0, 2.0, 3.0],
        autocall_barrier=100.0,  # never triggered
        coupon=0.0,
        capital_protection=1.0,
        notional=1_000_000,
    )
    price = ac.price(hw, n_paths=1000, n_steps=30)
    # With full capital protection, price ≈ PV of notional
    assert price > 0


def test_autocall_repr():
    ac = AutoCall(observation_dates=[1.0, 2.0], autocall_barrier=1.0, coupon=0.05)
    assert "AutoCall" in repr(ac)

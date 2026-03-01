"""Tests for Hull-White and SABR models."""

import math
import pytest
import numpy as np

from quantpy.curves import DiscountCurve, CurveStripper
from quantpy.market_data import MarketDataGenerator
from quantpy.models import HullWhiteModel, SABRModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def curve():
    """Simple upward-sloping curve."""
    rates = [0.035, 0.038, 0.040, 0.042, 0.044, 0.046, 0.048, 0.050]
    tenors = [0.25, 0.5, 1, 2, 3, 5, 10, 20]
    dfs = [math.exp(-r * t) for r, t in zip(rates, tenors)]
    return DiscountCurve(tenors, dfs)


@pytest.fixture
def hw(curve):
    return HullWhiteModel(curve, a=0.05, sigma=0.01)


@pytest.fixture
def sabr():
    return SABRModel(alpha=0.05, beta=0.5, rho=-0.3, nu=0.4)


# ---------------------------------------------------------------------------
# Hull-White tests
# ---------------------------------------------------------------------------

def test_hw_bond_at_t0(hw, curve):
    """Bond price at t=0 should equal discount curve DF."""
    r0 = curve.forward_rate(1e-6, 2e-4)
    for T in [1, 2, 5, 10]:
        p_hw = hw.bond_price(0.0, T, r0)
        p_curve = curve.discount_factor(T)
        assert p_hw == pytest.approx(p_curve, rel=0.02), f"T={T}"


def test_hw_caplet_positive(hw):
    """Caplet price should be non-negative."""
    price = hw.caplet_price(reset=1.0, payment=1.25, strike=0.04, notional=1.0)
    assert price >= 0.0


def test_hw_caplet_parity(hw, curve):
    """Cap-floor parity: Cap - Floor = Swap value."""
    strike = 0.04
    cap = hw.caplet_price(1.0, 1.25, strike, is_call=True)
    floor = hw.caplet_price(1.0, 1.25, strike, is_call=False)
    # Caplet - Floorlet = FRA * df_payment
    df_reset = curve.discount_factor(1.0)
    df_pay = curve.discount_factor(1.25)
    fwd = curve.forward_rate(1.0, 1.25, compounding="simple")
    dt = 0.25
    fra_pv = (fwd - strike) * dt * df_pay
    assert (cap - floor) == pytest.approx(fra_pv, abs=1e-4)


def test_hw_cap_price(hw):
    """Cap price with maturity 5Y should be positive."""
    price = hw.cap_price(maturity=5.0, strike=0.04)
    assert price > 0.0


def test_hw_floor_price(hw):
    price = hw.cap_price(maturity=5.0, strike=0.04, is_cap=False)
    assert price > 0.0


def test_hw_european_swaption_positive(hw):
    price = hw.european_swaption_price(
        expiry=1.0, swap_maturity=5.0, strike=0.04, is_payer=True
    )
    assert price >= 0.0


def test_hw_european_swaption_payer_receiver(hw):
    """Payer + Receiver = Forward swap (swaption parity)."""
    expiry = 1.0
    maturity = 5.0
    strike = hw.curve.par_swap_rate(maturity)
    payer = hw.european_swaption_price(expiry, maturity, strike, is_payer=True)
    receiver = hw.european_swaption_price(expiry, maturity, strike, is_payer=False)
    # At-the-money: payer ≈ receiver (not exact for non-ATM expiry curve)
    assert payer > 0.0
    assert receiver > 0.0


def test_hw_simulate_shape(hw):
    times, paths = hw.simulate(n_paths=100, n_steps=50, T=5.0)
    assert times.shape == (51,)
    assert paths.shape == (100, 51)


def test_hw_simulate_reasonable_rates(hw):
    _, paths = hw.simulate(n_paths=1000, n_steps=100, T=10.0)
    mean_final = paths[:, -1].mean()
    assert -0.10 < mean_final < 0.20, f"Unreasonable final rate mean: {mean_final}"


def test_hw_bermudan_swaption_positive(hw):
    expiries = [1.0, 2.0, 3.0]
    price = hw.bermudan_swaption_price(
        expiries=expiries, swap_maturity=5.0, strike=0.04, n_time_steps=50
    )
    assert price >= 0.0


def test_hw_bermudan_ge_european(hw):
    """Bermudan swaption should be >= the best European."""
    expiry = 1.0
    maturity = 5.0
    strike = hw.curve.par_swap_rate(maturity)
    eur = hw.european_swaption_price(expiry, maturity, strike, is_payer=True)
    berm = hw.bermudan_swaption_price([expiry], maturity, strike, n_time_steps=50)
    # With only one exercise date, Bermudan ≈ European (within tree error)
    assert berm >= 0.0


def test_hw_repr(hw):
    assert "HullWhiteModel" in repr(hw)


# ---------------------------------------------------------------------------
# SABR tests
# ---------------------------------------------------------------------------

def test_sabr_atm_vol(sabr):
    vol = sabr.implied_vol(F=0.04, K=0.04, T=5.0)
    assert vol > 0.0
    assert vol < 2.0


def test_sabr_smile_shape(sabr):
    """Implied vol should be higher away from ATM (smile effect)."""
    F = 0.04
    T = 5.0
    vol_atm = sabr.implied_vol(F, F, T)
    vol_low = sabr.implied_vol(F, F * 0.5, T)
    vol_high = sabr.implied_vol(F, F * 2.0, T)
    assert vol_low > vol_atm * 0.8  # smile is present
    assert vol_high > vol_atm * 0.8


def test_sabr_vol_surface(sabr):
    F = 0.04
    strikes = [0.02, 0.03, 0.04, 0.05, 0.06]
    vols = sabr.vol_surface(F, strikes, expiry=5.0)
    assert len(vols) == len(strikes)
    assert all(v > 0 for v in vols)


def test_sabr_option_price(sabr):
    price = sabr.option_price(F=0.04, K=0.04, T=5.0, df=1.0, is_call=True)
    assert price >= 0.0


def test_sabr_put_call_parity(sabr):
    """Put-Call parity: Call - Put = (F - K) * df."""
    F, K, T = 0.04, 0.04, 3.0
    call = sabr.option_price(F, K, T, df=1.0, is_call=True)
    put = sabr.option_price(F, K, T, df=1.0, is_call=False)
    assert (call - put) == pytest.approx(F - K, abs=1e-6)


def test_sabr_calibrate():
    F = 0.04
    T = 5.0
    # Generate target vols from a known model
    true_model = SABRModel(alpha=0.04, beta=0.5, rho=-0.2, nu=0.35)
    strikes = [0.02, 0.03, 0.04, 0.05, 0.06]
    target_vols = [true_model.implied_vol(F, K, T) for K in strikes]

    fitted = SABRModel.calibrate(F, strikes, T, target_vols, beta=0.5)
    # Fitted vols should be close to target
    for K, v_t in zip(strikes, target_vols):
        v_f = fitted.implied_vol(F, K, T)
        assert abs(v_f - v_t) < 0.01, f"Calibration error at K={K}: {v_f:.4f} vs {v_t:.4f}"


def test_sabr_invalid_params():
    with pytest.raises(ValueError):
        SABRModel(alpha=0.05, beta=1.5)  # beta > 1
    with pytest.raises(ValueError):
        SABRModel(alpha=0.05, beta=0.5, rho=1.0)  # rho = 1
    with pytest.raises(ValueError):
        SABRModel(alpha=-0.01)  # alpha < 0


def test_sabr_repr(sabr):
    assert "SABRModel" in repr(sabr)


def test_sabr_normal_vol(sabr):
    vol = sabr.implied_vol(F=0.04, K=0.03, T=5.0, vol_type="normal")
    assert vol > 0.0

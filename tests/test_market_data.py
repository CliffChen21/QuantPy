"""Tests for market_data module."""

import pytest
from quantpy.market_data import (
    MarketDataGenerator,
    SwapRateQuote,
    MoneyMarketQuote,
    FutureQuote,
    FRAQuote,
    VolatilityQuote,
    FXVolatilityQuote,
)


@pytest.fixture
def generator():
    return MarketDataGenerator(seed=42, base_rate=0.04)


def test_swap_rates(generator):
    quotes = generator.generate_swap_rates()
    assert len(quotes) == len(MarketDataGenerator.SWAP_TENORS)
    for q in quotes:
        assert isinstance(q, SwapRateQuote)
        assert q.rate > 0
        assert q.tenor > 0
        assert q.currency == "USD"


def test_mm_quotes(generator):
    quotes = generator.generate_mm_quotes()
    assert len(quotes) == len(MarketDataGenerator.MM_TENORS)
    for q in quotes:
        assert isinstance(q, MoneyMarketQuote)
        assert q.rate > 0


def test_future_quotes(generator):
    quotes = generator.generate_future_quotes()
    assert len(quotes) == len(MarketDataGenerator.FUTURE_STARTS)
    for q in quotes:
        assert isinstance(q, FutureQuote)
        assert 90 < q.price < 100  # sanity
        assert q.implied_rate > 0


def test_fra_quotes(generator):
    quotes = generator.generate_fra_quotes()
    assert len(quotes) > 0
    for q in quotes:
        assert isinstance(q, FRAQuote)
        assert q.rate > 0
        assert q.end > q.start


def test_rates_vol_surface(generator):
    quotes = generator.generate_rates_vol_surface()
    assert len(quotes) > 0
    for q in quotes:
        assert isinstance(q, VolatilityQuote)
        assert q.vol > 0
        assert q.tenor > 0


def test_fx_vol_surface(generator):
    quotes = generator.generate_fx_vol_surface()
    assert len(quotes) > 0
    for q in quotes:
        assert isinstance(q, FXVolatilityQuote)
        assert q.vol > 0
        assert q.expiry > 0


def test_generate_all(generator):
    all_data = generator.generate_all()
    assert set(all_data.keys()) == {
        "swap_rates", "mm_quotes", "futures", "fras", "rates_vol", "fx_vol"
    }
    for key, data in all_data.items():
        assert len(data) > 0, f"Empty data for {key}"


def test_repr(generator):
    q = generator.generate_swap_rates()[0]
    assert "SwapRateQuote" in repr(q)
    m = generator.generate_mm_quotes()[0]
    assert "MMQuote" in repr(m)


def test_reproducibility():
    g1 = MarketDataGenerator(seed=99)
    g2 = MarketDataGenerator(seed=99)
    r1 = [q.rate for q in g1.generate_swap_rates()]
    r2 = [q.rate for q in g2.generate_swap_rates()]
    assert r1 == r2


def test_different_seeds():
    g1 = MarketDataGenerator(seed=1)
    g2 = MarketDataGenerator(seed=2)
    r1 = [q.rate for q in g1.generate_swap_rates()]
    r2 = [q.rate for q in g2.generate_swap_rates()]
    assert r1 != r2

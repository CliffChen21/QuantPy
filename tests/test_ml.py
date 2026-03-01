"""Tests for ml module."""

import pytest
import numpy as np

from quantpy.ml import (
    NeuralNetworkPricer,
    VolSurfaceInterpolator,
    build_training_data,
    DEEP_LEARNING_SUGGESTIONS,
)


def test_dl_suggestions_list():
    assert isinstance(DEEP_LEARNING_SUGGESTIONS, list)
    assert len(DEEP_LEARNING_SUGGESTIONS) >= 5
    for entry in DEEP_LEARNING_SUGGESTIONS:
        assert "name" in entry
        assert "description" in entry


def test_nn_pricer_predict():
    model = NeuralNetworkPricer(in_dim=4)
    X = np.array([[0.04, 5.0, 0.30, 0.5],
                  [0.05, 3.0, 0.25, 0.4]])
    preds = model.predict(X)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))


def test_nn_pricer_fit():
    model = NeuralNetworkPricer(in_dim=3)
    X = np.random.default_rng(0).uniform(0, 1, (50, 3))
    y = X[:, 0] * X[:, 1] + X[:, 2]
    losses = model.fit(X, y, n_epochs=10, batch_size=16)
    assert len(losses) == 10
    assert all(isinstance(l, float) for l in losses)


def test_nn_pricer_single_sample():
    model = NeuralNetworkPricer(in_dim=2)
    pred = model.predict(np.array([0.04, 5.0]))
    assert np.isfinite(pred[0])


def test_nn_pricer_repr():
    model = NeuralNetworkPricer(in_dim=5)
    assert "NeuralNetworkPricer" in repr(model)


def test_vol_surface_fit_predict():
    rng = np.random.default_rng(42)
    expiries = rng.uniform(0.25, 5.0, 20)
    strikes = rng.uniform(0.02, 0.08, 20)
    vols = 0.3 + 0.1 * rng.standard_normal(20)
    vols = np.abs(vols)

    interp = VolSurfaceInterpolator(gamma=1.0)
    interp.fit(expiries, strikes, vols)

    # Predict at training points (should be close)
    preds = interp.predict(expiries, strikes)
    assert preds.shape == (20,)
    assert np.all(preds >= 0)


def test_vol_surface_not_fitted():
    interp = VolSurfaceInterpolator()
    with pytest.raises(RuntimeError):
        interp.predict([1.0], [0.04])


def test_vol_surface_repr():
    interp = VolSurfaceInterpolator()
    assert "VolSurfaceInterpolator" in repr(interp)


def test_build_training_data():
    def simple_pricer(strike, expiry):
        return strike * expiry

    X, y = build_training_data(
        simple_pricer,
        {"strike": (0.01, 0.10), "expiry": (1.0, 10.0)},
        n_samples=100,
        seed=0,
    )
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert np.all(np.isfinite(y))


def test_build_training_data_with_errors():
    """Pricer that raises occasionally should not include NaN rows."""
    call_count = [0]

    def flaky_pricer(x):
        call_count[0] += 1
        if call_count[0] % 5 == 0:
            raise ValueError("Occasional error")
        return x ** 2

    X, y = build_training_data(
        flaky_pricer, {"x": (0.0, 1.0)}, n_samples=50, seed=0
    )
    assert np.all(np.isfinite(y))
    assert len(X) <= 50

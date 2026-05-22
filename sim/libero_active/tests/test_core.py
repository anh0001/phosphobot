"""Unit tests for the dependency-independent research core (uncertainty + query methods).

These run without LeRobot/LIBERO and verify the conformal math is correct:
    pytest sim/libero_active/tests/test_core.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_active.config import ConformalConfig
from conformal_active.query_methods import StepContext, build_query_method
from conformal_active.uncertainty import (
    ConformalActionUncertainty,
    IQTTracker,
    _conformal_quantile,
    raw_dispersion,
)


# -- raw_dispersion ------------------------------------------------------------
def test_dispersion_zero_for_identical_samples():
    chunk = np.ones((1, 8, 7))
    samples = np.repeat(chunk, 5, axis=0)  # 5 identical chunks
    assert raw_dispersion(samples) == pytest.approx(0.0)


def test_dispersion_positive_for_varied_samples():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(8, 8, 7))
    assert raw_dispersion(samples) > 0.0


def test_dispersion_rejects_wrong_rank():
    with pytest.raises(ValueError):
        raw_dispersion(np.zeros((8, 7)))


# -- conformal quantile --------------------------------------------------------
def test_conformal_quantile_coverage():
    """A fresh score should exceed the (1-alpha) quantile with prob ~ alpha."""
    rng = np.random.default_rng(42)
    alpha = 0.1
    exceed = 0
    trials = 2000
    for _ in range(trials):
        calib = rng.normal(size=200)
        thr = _conformal_quantile(calib, alpha)
        fresh = rng.normal()
        exceed += fresh > thr
    rate = exceed / trials
    # split-conformal guarantee: empirical exceed-rate ~ alpha (within sampling noise)
    assert 0.06 < rate < 0.14, f"exceed rate {rate} not near alpha={alpha}"


def test_conformal_quantile_empty():
    assert _conformal_quantile(np.array([]), 0.1) == float("inf")


# -- IQT tracker ---------------------------------------------------------------
def test_iqt_raises_alpha_when_overcovering():
    """If labels never miscover (err=0), alpha_t should rise toward querying more."""
    iqt = IQTTracker(target_alpha=0.1, gamma=0.05)
    start = iqt.effective_alpha
    for _ in range(50):
        iqt.update(miscovered=False)
    assert iqt.effective_alpha > start


def test_iqt_converges_around_target():
    """ACI self-corrects only when miscoverage is *coupled* to alpha_t.

    Under exchangeability the prediction set at level alpha_t miscovers with
    probability alpha_t, so err_t ~ Bernoulli(alpha_t). The update then has a fixed
    point at alpha_t == target (drift = gamma*(target - alpha_t)), so alpha_t is
    mean-reverting and converges. (Feeding an exogenous err stream instead would make
    it an undamped random walk — that is a property of ACI, not a bug.)
    """
    rng = np.random.default_rng(1)
    iqt = IQTTracker(target_alpha=0.1, gamma=0.02)
    trace = []
    for _ in range(8000):
        miscovered = bool(rng.random() < iqt.effective_alpha)  # coupled to alpha_t
        iqt.update(miscovered=miscovered)
        trace.append(iqt.effective_alpha)
    # Long-run average of the coupled system sits at the target.
    assert np.mean(trace[-4000:]) == pytest.approx(0.1, abs=0.03)


# -- ConformalActionUncertainty ------------------------------------------------
def test_uncertainty_never_queries_before_calibration():
    cu = ConformalActionUncertainty(calib_min=20)
    assert not cu.is_calibrated
    assert cu.is_uncertain(raw_score=1e9) is False  # uncalibrated -> never query


def test_uncertainty_queries_high_scores_after_calibration():
    cu = ConformalActionUncertainty(target_alpha=0.1, calib_min=20, use_iqt=False)
    rng = np.random.default_rng(7)
    for _ in range(200):
        cu.add_calibration(abs(rng.normal()))
    assert cu.is_calibrated
    # A score far above the calibration range must trigger a query.
    assert cu.is_uncertain(raw_score=100.0) is True
    # A tiny score must not.
    assert cu.is_uncertain(raw_score=0.0) is False


# -- query method factory -----------------------------------------------------
@pytest.mark.parametrize("name", ["random", "entropy", "knn", "human_gated", "conformal"])
def test_build_query_method(name):
    m = build_query_method(name, seed=0, conformal=ConformalConfig())
    assert m.name == name


def test_query_methods_produce_scores():
    rng = np.random.default_rng(3)
    ctx = StepContext(
        step=0,
        action_samples=rng.normal(size=(8, 8, 7)),
        feature=rng.normal(size=32),
        oracle_disagreement=0.7,
    )
    for name in ["random", "entropy", "human_gated", "conformal"]:
        m = build_query_method(name, seed=0, conformal=ConformalConfig())
        assert isinstance(m.score(ctx), float)

    knn = build_query_method("knn", seed=0, conformal=ConformalConfig())
    assert knn.score(ctx) == 1.0  # no demos registered yet -> maximally novel
    knn.register_demo_state(ctx.feature)
    assert knn.score(ctx) == pytest.approx(0.0, abs=1e-9)  # exact match -> zero distance

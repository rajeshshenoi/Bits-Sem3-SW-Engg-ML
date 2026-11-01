"""ML tests - model inference behaviour.

Covers output shape/range checks, minimum-functionality tests on
domain-obvious cases, invariance tests (changes that must NOT change the
prediction) and directional-expectation tests (changes that must move the
prediction in a known direction), following the CheckList methodology.
"""

from __future__ import annotations

import numpy as np
import pytest

from brakeguard.data import SyntheticTelemetrySource
from brakeguard.models import BrakeFaultPredictor
from brakeguard.schemas import RAW_FEATURES
from tests.conftest import HIGH_RISK, LOW_RISK


@pytest.fixture(scope="module")
def predictor(trained_bundle):
    return BrakeFaultPredictor(trained_bundle)


@pytest.fixture(scope="module")
def fleet():
    return SyntheticTelemetrySource(n_samples=200, seed=99).load()[list(RAW_FEATURES)].to_dict("records")


# -- shape and range -------------------------------------------------------------------
def test_batch_output_shape(predictor, fleet):
    probabilities = predictor.predict_proba(fleet)
    assert probabilities.shape == (len(fleet),)
    assert len(predictor.predict_batch(fleet)) == len(fleet)


def test_probabilities_are_valid(predictor, fleet):
    probabilities = predictor.predict_proba(fleet)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert not np.any(np.isnan(probabilities))


def test_risk_band_consistent_with_probability(predictor, fleet):
    for pred in predictor.predict_batch(fleet):
        assert pred.risk == predictor.thresholds.classify(pred.fault_probability)


def test_explanation_has_top_k_known_features(predictor, trained_bundle):
    pred = predictor.predict(HIGH_RISK, top_k=3)
    assert len(pred.top_factors) == 3
    assert {f["feature"] for f in pred.top_factors} <= set(trained_bundle.feature_names)


def test_predictions_are_deterministic(predictor):
    assert predictor.predict(HIGH_RISK).fault_probability == predictor.predict(HIGH_RISK).fault_probability


def test_batch_and_single_predictions_agree(predictor, fleet):
    batch = predictor.predict_proba(fleet[:10])
    single = [predictor.predict(r).fault_probability for r in fleet[:10]]
    assert np.allclose(batch, single)


# -- minimum functionality ------------------------------------------------------------------
def test_obviously_failing_brakes_are_high_risk(predictor):
    assert predictor.predict(HIGH_RISK).risk == "HIGH"


def test_new_brakes_are_low_risk(predictor):
    assert predictor.predict(LOW_RISK).risk == "LOW"


# -- invariance ----------------------------------------------------------------------------
def test_invariant_to_key_order(predictor):
    reordered = dict(reversed(list(HIGH_RISK.items())))
    assert predictor.predict(reordered).fault_probability == predictor.predict(HIGH_RISK).fault_probability


def test_invariant_to_extra_metadata(predictor):
    with_meta = {**HIGH_RISK, "vehicle_id": "BG-1", "driver_name": "anyone"}
    assert predictor.predict(with_meta).fault_probability == predictor.predict(HIGH_RISK).fault_probability


def test_invariant_to_negligible_sensor_noise(predictor, fleet):
    rng = np.random.default_rng(0)
    base = predictor.predict_proba(fleet)
    noisy = [{k: v * (1 + rng.normal(0, 0.001)) for k, v in r.items()} for r in fleet]
    assert np.mean(np.abs(predictor.predict_proba(noisy) - base)) < 0.01


# -- directional expectations ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("feature", "low", "high"),
    [
        ("pad_wear_pct", 15.0, 90.0),
        ("vibration_rms", 0.5, 5.0),
        ("brake_temperature", 100.0, 240.0),
        ("fluid_age_days", 30.0, 800.0),
    ],
)
def test_worse_condition_increases_risk(predictor, fleet, feature, low, high):
    """Averaged over the fleet, a worse value must not lower the fault probability."""
    better = predictor.predict_proba([{**r, feature: low} for r in fleet])
    worse = predictor.predict_proba([{**r, feature: high} for r in fleet])
    assert worse.mean() > better.mean() + 0.02


def test_pressure_deviation_in_either_direction_increases_risk(predictor, fleet):
    nominal = predictor.predict_proba([{**r, "brake_pressure": 65.0} for r in fleet]).mean()
    too_low = predictor.predict_proba([{**r, "brake_pressure": 25.0} for r in fleet]).mean()
    too_high = predictor.predict_proba([{**r, "brake_pressure": 105.0} for r in fleet]).mean()
    assert too_low > nominal and too_high > nominal

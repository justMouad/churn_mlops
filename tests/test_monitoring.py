import numpy as np

from churnops.config import CATEGORICAL, NUMERIC
from churnops.data import make_customers
from churnops.monitoring import (
    churn_performance,
    drift_report,
    forecast_performance,
    psi_categorical,
    psi_numeric,
    status_from_psi,
)


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi_numeric(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi_numeric(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_psi_categorical():
    assert psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50) < 1e-6
    assert psi_categorical(["a"] * 90 + ["b"] * 10, ["a"] * 10 + ["b"] * 90) > 0.25


def test_status_thresholds():
    assert status_from_psi(0.05) == "ok" and status_from_psi(0.15) == "warning" and status_from_psi(0.4) == "alert"


def test_drift_report_stable_vs_shifted():
    ref = make_customers(4000, seed=1)
    stable = drift_report(ref, make_customers(3000, seed=2), NUMERIC, CATEGORICAL)
    shifted = drift_report(ref, make_customers(3000, seed=3, drift=True), NUMERIC, CATEGORICAL)
    assert stable["overall_status"] == "ok" and not stable["retrain_recommended"]
    assert shifted["overall_status"] == "alert" and shifted["retrain_recommended"]
    assert shifted["features"]["monthly_charges"]["status"] == "alert"


def test_binary_feature_no_false_alarm():
    ref = make_customers(4000, seed=1)
    r = drift_report(ref, make_customers(3000, seed=2), ["has_discount"], [])
    assert r["features"]["has_discount"]["status"] == "ok"


def test_performance_monitors():
    y = np.full(20, 100.0)
    assert forecast_performance(y, y * 1.03, baseline_mape=3.0)["status"] == "ok"
    assert forecast_performance(y, y * 1.20, baseline_mape=3.0)["retrain_recommended"]
    rng = np.random.default_rng(0)
    yt = rng.binomial(1, 0.5, 2000)
    assert churn_performance(yt, rng.random(2000), ref_roc_auc=0.80)["retrain_recommended"]

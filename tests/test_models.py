import numpy as np
import pandas as pd
import pytest

from churnops import forecasting as fc
from churnops.churn import train_churn
from churnops.data import make_customers, make_timeseries


def test_metrics():
    assert fc.mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert fc.mae([1, 2], [2, 4]) == 1.5
    assert fc.smape([100], [100]) == 0


def test_seasonal_naive_repeats_week():
    df = make_timeseries(60)
    p = fc.SeasonalNaive().fit(df).predict(14)
    assert np.allclose(p["yhat"].values[:7], df["y"].values[-7:])
    assert p["ds"].iloc[0] == df["ds"].iloc[-1] + pd.Timedelta(days=1)


def test_prophet_beats_naive_and_roundtrips(tmp_path):
    df = make_timeseries(500)
    train, test = df.iloc[:-28], df.iloc[-28:]
    m = fc.ProphetForecaster().fit(train)
    pred = m.predict(28)
    assert fc.mape(test["y"], pred["yhat"]) < fc.mape(test["y"], fc.SeasonalNaive().fit(train).predict(28)["yhat"])
    m.save(tmp_path / "p.json")
    again = fc.ProphetForecaster.load(tmp_path / "p.json").predict(28)
    assert np.allclose(pred["yhat"], again["yhat"], rtol=1e-6)


def test_lstm_trains_and_roundtrips(tmp_path):
    pytest.importorskip("torch")
    df = make_timeseries(400)
    m = fc.LSTMForecaster(epochs=15).fit(df)
    p = m.predict(40)                                  # > horizon : prévision récursive
    assert len(p) == 40 and p["yhat"].notna().all()
    assert 0.5 * df["y"].iloc[-30:].mean() < p["yhat"].mean() < 1.5 * df["y"].iloc[-30:].mean()
    m.save(tmp_path / "l.pt")
    assert np.allclose(fc.LSTMForecaster.load(tmp_path / "l.pt").predict(10)["yhat"], p["yhat"][:10], rtol=1e-4)


def test_churn_model_beats_baseline():
    df = make_customers(6000)
    model, meta, ref = train_churn(df, quick=True)
    tm = meta["test_metrics"]
    assert tm["roc_auc"] > 0.75 and tm["pr_auc"] > 2 * tm["positive_rate"]
    assert 0.05 <= meta["threshold"] <= 0.95 and "churn_probability" in ref

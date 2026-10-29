import pandas as pd

from churnops.data import make_customers

CUSTOMER = {"customer_id": "C-1", "tenure_months": 3, "monthly_charges": 95.0, "contract_type": "month-to-month",
            "payment_method": "electronic_check", "support_tickets_90d": 5, "usage_hours_30d": 15.0,
            "late_payments_6m": 2, "num_products": 1, "has_discount": 0, "age": 30}
LOYAL = {**CUSTOMER, "customer_id": "C-2", "tenure_months": 60, "contract_type": "two-year",
         "payment_method": "credit_card", "support_tickets_90d": 0, "usage_hours_30d": 110.0,
         "late_payments_6m": 0, "num_products": 5, "has_discount": 1, "monthly_charges": 45.0}


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["models_loaded"]["churn"] and r["models_loaded"]["prophet"]


def test_churn_single_and_ordering(client):
    risky = client.post("/predict/churn", json=CUSTOMER).json()
    loyal = client.post("/predict/churn", json=LOYAL).json()
    assert 0 <= loyal["churn_probability"] <= 1
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["risk_tier"] in {"medium", "high"} and loyal["risk_tier"] == "low"


def test_churn_validation_error(client):
    bad = {**CUSTOMER, "contract_type": "lifetime"}
    assert client.post("/predict/churn", json=bad).status_code == 422
    assert client.post("/predict/churn", json={**CUSTOMER, "age": 5}).status_code == 422


def test_churn_batch(client):
    r = client.post("/predict/churn/batch", json={"customers": [CUSTOMER, LOYAL]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_forecast(client):
    r = client.post("/forecast", json={"horizon": 14, "model": "prophet"}).json()
    assert len(r["forecast"]) == 14 and r["forecast"][0]["yhat_lower"] < r["forecast"][0]["yhat_upper"]
    assert client.post("/forecast", json={"horizon": 0}).status_code == 422
    assert client.post("/forecast", json={"horizon": 400}).status_code == 422
    r2 = client.post("/forecast", json={"horizon": 10}).json()      # champion
    assert r2["model"] in {"prophet", "lstm"}


def test_live_drift_needs_data_then_detects_shift(client):
    shifted = make_customers(400, seed=9, drift=True).drop(columns=["churned"])
    rows = shifted.to_dict("records")
    r = client.post("/monitoring/drift", json={"customers": rows}).json()
    assert r["overall_status"] == "alert" and r["retrain_recommended"]
    stable = make_customers(400, seed=10).drop(columns=["churned"]).to_dict("records")
    assert client.post("/monitoring/drift", json={"customers": stable}).json()["overall_status"] in {"ok", "warning"}


def test_live_buffer_drift(client):
    rows = make_customers(300, seed=11, drift=True).drop(columns=["churned"]).to_dict("records")
    client.post("/predict/churn/batch", json={"customers": rows[:300]})
    r = client.get("/monitoring/drift", params={"min_samples": 100}).json()
    assert r["overall_status"] in {"warning", "alert"}
    assert client.get("/monitoring/stats").json()["churn_predictions_served"] >= 300


def test_forecast_performance_endpoint(client):
    f = client.post("/forecast", json={"horizon": 14}).json()["forecast"]
    actuals = [{"ds": p["ds"], "y": p["yhat"] * 1.01} for p in f]
    r = client.post("/monitoring/forecast-performance", json={"actuals": actuals}).json()
    assert r["current_mape"] < 2 and r["status"] == "ok"
    bad = [{"ds": p["ds"], "y": p["yhat"] * 1.5} for p in f]
    assert client.post("/monitoring/forecast-performance", json={"actuals": bad}).json()["retrain_recommended"]
    old = [{"ds": str(d.date()), "y": 5000} for d in pd.date_range("2020-01-01", periods=10)]
    assert client.post("/monitoring/forecast-performance", json={"actuals": old}).status_code == 422

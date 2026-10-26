"""API FastAPI : prédiction du churn, prévision de série temporelle, monitoring de dérive."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request

from churnops.config import CATEGORICAL, FEATURES, MAX_API_HORIZON, MIN_LIVE_SAMPLES, NUMERIC
from churnops.monitoring import drift_report, forecast_performance

from .schemas import (
    BatchRequest,
    BatchResponse,
    ChurnPrediction,
    Customer,
    ForecastPerfRequest,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
)
from .state import ModelState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("churn-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelState().load()
    log.info("Artefacts chargés (version %s)", app.state.models.version)
    yield


app = FastAPI(title="Churn & Forecast API", version="1.0.0", lifespan=lifespan,
              description="Prédiction du churn, prévision de la série d'activité et suivi de dérive.")


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def _state(request: Request) -> ModelState:
    return request.app.state.models


def _tier(p: float) -> str:
    return "high" if p >= 0.5 else "medium" if p >= 0.2 else "low"


def _predict_customers(st: ModelState, customers: list[Customer]) -> list[ChurnPrediction]:
    if st.churn is None:
        raise HTTPException(503, "Modèle churn non chargé")
    rows = [c.model_dump() for c in customers]
    probas = st.churn.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]
    st.record(rows, probas.tolist())
    st.n_churn_requests += len(rows)
    thr = st.churn_meta["threshold"]
    return [ChurnPrediction(customer_id=r.get("customer_id"), churn_probability=round(float(p), 4),
                            churn_predicted=bool(p >= thr), risk_tier=_tier(float(p)), threshold=thr,
                            model_version=st.version) for r, p in zip(rows, probas, strict=True)]


# ------------------------------------------------------------------ santé / info
@app.get("/health", tags=["ops"])
def health(request: Request):
    st = _state(request)
    loaded = {"churn": st.churn is not None, **{k: True for k in st.forecasters}}
    ok = st.churn is not None and "prophet" in st.forecasters
    return {"status": "ok" if ok else "degraded", "models_loaded": loaded, "version": st.version}


@app.get("/model/info", tags=["ops"])
def model_info(request: Request):
    st = _state(request)
    return {"registry": st.registry, "churn": st.churn_meta, "forecast": st.forecast_meta,
            "available_forecasters": list(st.forecasters)}


# ------------------------------------------------------------------ churn
@app.post("/predict/churn", response_model=ChurnPrediction, tags=["churn"])
def predict_churn(customer: Customer, request: Request):
    return _predict_customers(_state(request), [customer])[0]


@app.post("/predict/churn/batch", response_model=BatchResponse, tags=["churn"])
def predict_churn_batch(body: BatchRequest, request: Request):
    return BatchResponse(predictions=_predict_customers(_state(request), body.customers))


# ------------------------------------------------------------------ prévision
def _get_forecaster(st: ModelState, name: str):
    if name == "champion":
        name = st.forecast_meta.get("champion", "prophet")
    if name not in st.forecasters:
        if name == "lstm" and "prophet" in st.forecasters:
            log.warning("LSTM indisponible : repli sur Prophet")
            name = "prophet"
        else:
            raise HTTPException(503, f"Modèle de prévision '{name}' indisponible")
    return name, st.forecasters[name]


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(body: ForecastRequest, request: Request):
    st = _state(request)
    if body.horizon > MAX_API_HORIZON:
        raise HTTPException(422, f"horizon max {MAX_API_HORIZON}")
    name, model = _get_forecaster(st, body.model)
    df = model.predict(body.horizon)
    st.n_forecast_requests += 1
    pts = [ForecastPoint(ds=r.ds.date(), yhat=round(float(r.yhat), 1),
                         yhat_lower=None if np.isnan(r.yhat_lower) else round(float(r.yhat_lower), 1),
                         yhat_upper=None if np.isnan(r.yhat_upper) else round(float(r.yhat_upper), 1))
           for r in df.itertuples()]
    return ForecastResponse(model=name, horizon=body.horizon, last_observation=model.last_date_.date(), forecast=pts)


# ------------------------------------------------------------------ monitoring
@app.get("/monitoring/drift", tags=["monitoring"])
def live_drift(request: Request, min_samples: int = Query(MIN_LIVE_SAMPLES, ge=10)):
    """Dérive du trafic réel récent (tampon mémoire) vs données d'entraînement."""
    st = _state(request)
    live = st.live_frame()
    if st.reference is None:
        raise HTTPException(503, "Référence indisponible")
    if len(live) < min_samples:
        return {"overall_status": "insufficient_data", "n_samples_current": int(len(live)), "min_samples": min_samples}
    return drift_report(st.reference, live, NUMERIC + ["churn_probability"], CATEGORICAL)


@app.post("/monitoring/drift", tags=["monitoring"])
def batch_drift(body: BatchRequest, request: Request):
    """Dérive d'un lot fourni par l'appelant (inclut la dérive des prédictions)."""
    st = _state(request)
    if st.reference is None or st.churn is None:
        raise HTTPException(503, "Modèle/référence indisponibles")
    df = pd.DataFrame([c.model_dump() for c in body.customers])[FEATURES]
    df["churn_probability"] = st.churn.predict_proba(df)[:, 1]
    return drift_report(st.reference, df, NUMERIC + ["churn_probability"], CATEGORICAL)


@app.post("/monitoring/forecast-performance", tags=["monitoring"])
def forecast_perf(body: ForecastPerfRequest, request: Request):
    """Compare les valeurs réellement observées aux prévisions du champion (MAPE vs backtest)."""
    st = _state(request)
    name, model = _get_forecaster(st, "champion")
    actual = pd.DataFrame([a.model_dump() for a in body.actuals]).assign(ds=lambda d: pd.to_datetime(d.ds))
    horizon = (actual.ds.max() - model.last_date_).days
    if horizon < 1 or horizon > MAX_API_HORIZON:
        raise HTTPException(422, "Les dates observées doivent suivre la fin de l'entraînement "
                                 f"({model.last_date_.date()}) dans la limite de {MAX_API_HORIZON} jours")
    pred = model.predict(horizon).merge(actual, on="ds", how="inner")
    if len(pred) < 7:
        raise HTTPException(422, "Moins de 7 dates observées dans la fenêtre prévue")
    return {"model": name, **forecast_performance(pred["y"], pred["yhat"], st.forecast_meta["baseline_mape"]),
            "n_points": int(len(pred))}


@app.get("/monitoring/stats", tags=["monitoring"])
def stats(request: Request):
    st = _state(request)
    return {"churn_predictions_served": st.n_churn_requests, "forecast_requests": st.n_forecast_requests,
            "live_buffer_size": int(len(st.live_frame())), "model_version": st.version}

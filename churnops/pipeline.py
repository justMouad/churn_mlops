"""Orchestration de l'entraînement : produit tous les artefacts consommés par l'API."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from . import forecasting as fc
from .churn import train_churn
from .config import ARTIFACTS_DIR, FORECAST_HORIZON
from .data import make_customers, make_timeseries


def _hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]


def run_training(out_dir: Path = ARTIFACTS_DIR, quick: bool = False, customers: pd.DataFrame | None = None,
                 series: pd.DataFrame | None = None, seed: int = 42, verbose: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    # ---------------- churn
    customers = make_customers(8000 if quick else 20000, seed=seed) if customers is None else customers
    model, cmeta, ref = train_churn(customers, quick=quick)
    joblib.dump(model, out / "churn_model.joblib")
    ref.to_csv(out / "churn_reference.csv", index=False)
    (out / "churn_meta.json").write_text(json.dumps(cmeta, indent=2))
    tm = cmeta["test_metrics"]
    log(f"[churn] modèle={cmeta['model']} PR-AUC={tm['pr_auc']:.3f} ROC-AUC={tm['roc_auc']:.3f} "
        f"F1={tm['f1']:.3f} seuil={tm['threshold']:.2f}")

    # ---------------- séries temporelles
    series = make_timeseries(seed=seed) if series is None else series
    bt = fc.backtest(series, horizon=FORECAST_HORIZON, n_folds=2 if quick else 3, lstm_epochs=25 if quick else 80)
    champion = fc.pick_champion(bt)
    for name, r in bt.items():
        log(f"[forecast] {name:8s} MAPE={r['mape']:.2f}%  sMAPE={r['smape']:.2f}%  MAE={r['mae']:.0f}")
    log(f"[forecast] champion = {champion}")

    fc.ProphetForecaster().fit(series).save(out / "prophet_model.json")
    if fc.lstm_available():
        fc.LSTMForecaster(epochs=25 if quick else 80).fit(series).save(out / "lstm_model.pt")
    fc.save_forecast_meta(out / "forecast_meta.json", champion=champion, backtest=bt, horizon=FORECAST_HORIZON,
                          train_end=str(series["ds"].iloc[-1].date()), n_obs=len(series),
                          baseline_mape=bt[champion]["mape"])
    series.to_csv(out / "series_history.csv", index=False)

    registry = {"version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "data_hash": {"customers": _hash(customers), "series": _hash(series)},
                "churn": {"model": cmeta["model"], **tm},
                "forecast": {"champion": champion, "mape": bt[champion]["mape"],
                             "naive_mape": bt["naive"]["mape"], "models": {k: v["mape"] for k, v in bt.items()}}}
    (out / "registry.json").write_text(json.dumps(registry, indent=2))
    return registry

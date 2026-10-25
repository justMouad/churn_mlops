"""Chargement des artefacts et tampon des requêtes récentes (pour le suivi de dérive)."""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path

import joblib
import pandas as pd

from churnops import forecasting as fc
from churnops.config import ARTIFACTS_DIR, FEATURES, LIVE_BUFFER_SIZE

log = logging.getLogger("churn-api")


class ModelState:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.dir = Path(artifacts_dir)
        self.churn = None
        self.churn_meta: dict = {}
        self.reference: pd.DataFrame | None = None
        self.forecasters: dict = {}
        self.forecast_meta: dict = {}
        self.registry: dict = {}
        self._live = deque(maxlen=LIVE_BUFFER_SIZE)
        self._lock = threading.Lock()
        self.n_churn_requests = 0
        self.n_forecast_requests = 0

    # ------------------------------------------------------------------ chargement
    def load(self):
        d = self.dir
        if (d / "churn_model.joblib").exists():
            self.churn = joblib.load(d / "churn_model.joblib")
            self.churn_meta = json.loads((d / "churn_meta.json").read_text())
            self.reference = pd.read_csv(d / "churn_reference.csv")
        else:
            log.warning("Modèle churn absent de %s", d)
        if (d / "forecast_meta.json").exists():
            self.forecast_meta = json.loads((d / "forecast_meta.json").read_text())
            hist = pd.read_csv(d / "series_history.csv", parse_dates=["ds"])
            self.forecasters["naive"] = fc.SeasonalNaive().fit(hist)
            if (d / "prophet_model.json").exists():
                self.forecasters["prophet"] = fc.ProphetForecaster.load(d / "prophet_model.json")
            if (d / "lstm_model.pt").exists() and fc.lstm_available():
                self.forecasters["lstm"] = fc.LSTMForecaster.load(d / "lstm_model.pt")
        if (d / "registry.json").exists():
            self.registry = json.loads((d / "registry.json").read_text())
        return self

    @property
    def version(self) -> str:
        return self.registry.get("version", "unknown")

    # ------------------------------------------------------------------ tampon live
    def record(self, rows: list[dict], probas: list[float]):
        with self._lock:
            for r, p in zip(rows, probas, strict=True):
                self._live.append({**{k: r[k] for k in FEATURES}, "churn_probability": p})

    def live_frame(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(list(self._live))

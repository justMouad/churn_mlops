"""Prévision de séries temporelles : baseline saisonnière, Prophet, LSTM (PyTorch) + backtest glissant."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FORECAST_HORIZON, SEED

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.WARNING)


# ----------------------------------------------------------------------------- métriques
def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100)


def smape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-9)) * 100)


def _future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


# ----------------------------------------------------------------------------- baseline
class SeasonalNaive:
    """Répète la dernière semaine observée : référence minimale à battre."""
    name = "naive"

    def __init__(self, season: int = 7):
        self.season = season

    def fit(self, df: pd.DataFrame):
        self.tail_ = df["y"].values[-self.season:]
        self.last_date_ = pd.Timestamp(df["ds"].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        yhat = np.resize(self.tail_, horizon).astype(float)
        return pd.DataFrame({"ds": _future_dates(self.last_date_, horizon), "yhat": yhat,
                             "yhat_lower": np.nan, "yhat_upper": np.nan})


# ----------------------------------------------------------------------------- Prophet
class ProphetForecaster:
    name = "prophet"

    def fit(self, df: pd.DataFrame):
        from prophet import Prophet
        self.model_ = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                              seasonality_mode="multiplicative", interval_width=0.9)
        self.model_.fit(df[["ds", "y"]])
        self.last_date_ = pd.Timestamp(df["ds"].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        future = pd.DataFrame({"ds": _future_dates(self.last_date_, horizon)})
        f = self.model_.predict(future)
        return f[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)

    def save(self, path: Path):
        from prophet.serialize import model_to_json
        Path(path).write_text(model_to_json(self.model_))

    @classmethod
    def load(cls, path: Path):
        from prophet.serialize import model_from_json
        obj = cls()
        obj.model_ = model_from_json(Path(path).read_text())
        obj.last_date_ = pd.Timestamp(obj.model_.history_dates.max())
        return obj


# ----------------------------------------------------------------------------- LSTM
class LSTMForecaster:
    """LSTM multi-horizon direct.

    Chaque fenêtre est normalisée par sa moyenne (le réseau apprend des ratios, donc la
    tendance/niveau ne gêne pas l'apprentissage). Entrées : valeur normalisée + jour de semaine (sin/cos).
    """
    name = "lstm"

    def __init__(self, window: int = 56, horizon: int = FORECAST_HORIZON, hidden: int = 48,
                 epochs: int = 80, lr: float = 3e-3, seed: int = SEED):
        self.window, self.horizon, self.hidden, self.epochs, self.lr, self.seed = window, horizon, hidden, epochs, lr, seed

    @staticmethod
    def _cal(dates) -> np.ndarray:
        dow = pd.DatetimeIndex(dates).dayofweek.values
        return np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=1)

    def _net(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s, hidden, horizon):
                super().__init__()
                s.lstm = nn.LSTM(3, hidden, batch_first=True)
                s.head = nn.Linear(hidden, horizon)

            def forward(s, x):
                out, _ = s.lstm(x)
                return s.head(out[:, -1])
        return Net(self.hidden, self.horizon)

    def fit(self, df: pd.DataFrame):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        y = df["y"].values.astype(np.float32)
        cal = self._cal(df["ds"])
        X, Y = [], []
        for i in range(self.window, len(y) - self.horizon + 1):
            w = y[i - self.window:i]
            m = w.mean()
            X.append(np.column_stack([w / m, cal[i - self.window:i]]))
            Y.append(y[i:i + self.horizon] / m)
        X = torch.tensor(np.array(X), dtype=torch.float32)
        Y = torch.tensor(np.array(Y), dtype=torch.float32)
        self.net_ = self._net()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=g)
            for b in range(0, len(X), 64):
                idx = perm[b:b + 64]
                opt.zero_grad()
                loss = loss_fn(self.net_(X[idx]), Y[idx])
                loss.backward()
                opt.step()
        self.tail_y_ = y[-self.window:].astype(float).tolist()
        self.last_date_ = pd.Timestamp(df["ds"].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        import torch
        self.net_.eval()
        hist = list(self.tail_y_)
        dates = list(pd.date_range(end=self.last_date_, periods=self.window, freq="D"))
        preds: list[float] = []
        with torch.no_grad():
            while len(preds) < horizon:           # horizon > H : prévision récursive par blocs
                w = np.array(hist[-self.window:])
                m = w.mean()
                x = np.column_stack([w / m, self._cal(dates[-self.window:])])
                out = self.net_(torch.tensor(x[None], dtype=torch.float32))[0].numpy() * m
                preds.extend(out.tolist())
                hist.extend(out.tolist())
                dates.extend(_future_dates(dates[-1], self.horizon))
        return pd.DataFrame({"ds": _future_dates(self.last_date_, horizon), "yhat": preds[:horizon],
                             "yhat_lower": np.nan, "yhat_upper": np.nan})

    def save(self, path: Path):
        import torch
        torch.save({"state_dict": self.net_.state_dict(),
                    "config": {"window": self.window, "horizon": self.horizon, "hidden": self.hidden},
                    "tail_y": self.tail_y_, "last_date": self.last_date_.isoformat()}, path)

    @classmethod
    def load(cls, path: Path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(**ck["config"])
        obj.net_ = obj._net()
        obj.net_.load_state_dict(ck["state_dict"])
        obj.tail_y_ = ck["tail_y"]
        obj.last_date_ = pd.Timestamp(ck["last_date"])
        return obj


# ----------------------------------------------------------------------------- backtest
def lstm_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def backtest(df: pd.DataFrame, horizon: int = FORECAST_HORIZON, n_folds: int = 3, lstm_epochs: int = 80) -> dict:
    """Validation à origine glissante : n_folds fenêtres de test consécutives de `horizon` jours."""
    factories = {"naive": SeasonalNaive, "prophet": ProphetForecaster}
    if lstm_available():
        factories["lstm"] = lambda: LSTMForecaster(horizon=horizon, epochs=lstm_epochs)
    results = {k: {"folds": []} for k in factories}
    for k in range(n_folds):
        cut = len(df) - horizon * (n_folds - k)
        train, test = df.iloc[:cut], df.iloc[cut:cut + horizon]
        for name, make in factories.items():
            pred = make().fit(train).predict(horizon)
            results[name]["folds"].append({"mape": mape(test["y"], pred["yhat"]),
                                           "smape": smape(test["y"], pred["yhat"]),
                                           "mae": mae(test["y"], pred["yhat"])})
    for r in results.values():
        for m in ("mape", "smape", "mae"):
            r[m] = float(np.mean([f[m] for f in r["folds"]]))
    return results


def pick_champion(bt: dict) -> str:
    """Meilleur modèle non-baseline selon le MAPE moyen (la baseline sert de garde-fou)."""
    cands = {k: v["mape"] for k, v in bt.items() if k != "naive"}
    return min(cands, key=cands.get)


def save_forecast_meta(path: Path, **meta):
    Path(path).write_text(json.dumps(meta, indent=2, default=str))

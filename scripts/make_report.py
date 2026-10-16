"""Génère reports/forecast_comparison.png : 28 derniers jours réels vs prévisions de chaque modèle."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from churnops import forecasting as fc  # noqa: E402
from churnops.config import FORECAST_HORIZON, REPORTS_DIR  # noqa: E402
from churnops.data import make_timeseries  # noqa: E402

df = make_timeseries()
train, test = df.iloc[:-FORECAST_HORIZON], df.iloc[-FORECAST_HORIZON:]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(train["ds"].iloc[-60:], train["y"].iloc[-60:], c="grey", label="historique")
ax.plot(test["ds"], test["y"], c="black", lw=2, label="réel")
models = {"naive": fc.SeasonalNaive()}
models["prophet"] = fc.ProphetForecaster()
if fc.lstm_available():
    models["lstm"] = fc.LSTMForecaster()
for name, m in models.items():
    p = m.fit(train).predict(FORECAST_HORIZON)
    ax.plot(p["ds"], p["yhat"], label=f"{name} (MAPE {fc.mape(test['y'], p['yhat']):.1f}%)")
ax.set_title("Prévision sur 28 jours : comparaison des modèles")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
REPORTS_DIR.mkdir(exist_ok=True)
fig.savefig(REPORTS_DIR / "forecast_comparison.png", dpi=140)
print("écrit", REPORTS_DIR / "forecast_comparison.png")

"""Configuration centrale (surchargeable par variables d'environnement)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))
SEED = 42

# --- Prévision ---
FORECAST_HORIZON = 28
MAX_API_HORIZON = 180

# --- Churn ---
TARGET = "churned"
NUMERIC = ["tenure_months", "monthly_charges", "support_tickets_90d", "usage_hours_30d",
           "late_payments_6m", "num_products", "has_discount", "age"]
CATEGORICAL = ["contract_type", "payment_method"]
FEATURES = NUMERIC + CATEGORICAL

# --- Monitoring (seuils PSI usuels : <0.1 stable, 0.1-0.25 à surveiller, >0.25 dérive forte) ---
PSI_WARN = 0.10
PSI_ALERT = 0.25
MIN_LIVE_SAMPLES = 100
LIVE_BUFFER_SIZE = 5000

"""Générateurs de données synthétiques : série temporelle d'activité + base clients (churn)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED, TARGET

CONTRACTS = ["month-to-month", "one-year", "two-year"]
PAYMENTS = ["credit_card", "bank_transfer", "electronic_check", "mailed_check"]


def make_timeseries(n_days: int = 1095, end: str = "2026-10-31", seed: int = SEED) -> pd.DataFrame:
    """Clients actifs quotidiens : tendance + saisonnalité hebdo/annuelle + promotions + bruit."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range(end=end, periods=n_days, freq="D")
    t = np.arange(n_days)
    trend = 5000 + 2.2 * t
    weekly = np.array([1.00, 1.03, 1.04, 1.04, 1.02, 0.90, 0.85])[ds.dayofweek]
    yearly = 1 + 0.08 * np.sin(2 * np.pi * (ds.dayofyear - 80) / 365.25)
    promo = np.ones(n_days)
    for start in rng.choice(n_days - 8, size=12, replace=False):
        promo[start:start + 7] *= 1.12
    y = trend * weekly * yearly * promo * rng.normal(1, 0.025, n_days)
    return pd.DataFrame({"ds": ds, "y": y.round(0)})


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_customers(n: int = 20_000, seed: int = SEED, drift: bool = False, churn_rate: float = 0.20) -> pd.DataFrame:
    """Base clients. drift=True décale les distributions (clients plus récents, plus chers, plus de tickets)."""
    rng = np.random.default_rng(seed)
    tenure = np.clip(rng.exponential(14 if drift else 24, n) + 1, 1, 72).round().astype(int)
    charges = np.clip(rng.normal(88 if drift else 70, 25, n), 15, 200).round(2)
    contract = rng.choice(CONTRACTS, n, p=[0.72, 0.18, 0.10] if drift else [0.55, 0.25, 0.20])
    payment = rng.choice(PAYMENTS, n, p=[0.3, 0.25, 0.3, 0.15])
    tickets = rng.poisson(2.0 if drift else 1.2, n)
    usage = np.clip(rng.normal(50 if drift else 60, 25, n), 0, None).round(1)
    late = rng.poisson(0.6, n)
    products = np.clip(1 + rng.poisson(1.5, n), 1, 8)
    discount = rng.binomial(1, 0.3, n)
    age = np.clip(rng.normal(42, 13, n), 18, 90).round().astype(int)

    z = (-0.04 * tenure + 0.015 * (charges - 70)
         + np.select([contract == "month-to-month", contract == "one-year"], [1.0, -0.2], -1.2)
         + 0.35 * tickets - 0.012 * (usage - 60) + 0.4 * late - 0.25 * (products - 2)
         - 0.5 * discount + 0.9 * (payment == "electronic_check")
         + 0.5 * (tickets >= 3) * (contract == "month-to-month") + rng.normal(0, 0.6, n))
    lo, hi = -15.0, 15.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > churn_rate:
            hi = mid
        else:
            lo = mid
    y = rng.binomial(1, _sigmoid(z + (lo + hi) / 2))
    return pd.DataFrame({
        "tenure_months": tenure, "monthly_charges": charges, "contract_type": contract,
        "payment_method": payment, "support_tickets_90d": tickets, "usage_hours_30d": usage,
        "late_payments_6m": late, "num_products": products, "has_discount": discount,
        "age": age, TARGET: y,
    })

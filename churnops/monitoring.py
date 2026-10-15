"""Suivi de base de la dérive des données et des performances (PSI, KS, dégradation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import PSI_ALERT, PSI_WARN

EPS = 1e-4


def psi_numeric(ref, cur, bins: int = 10) -> float:
    """Population Stability Index avec bornes en quantiles de la référence."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # variable quasi constante
        return 0.0 if np.allclose(np.mean(ref), np.mean(cur)) else PSI_ALERT + 1
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorical(ref, cur) -> float:
    ref, cur = pd.Series(ref), pd.Series(cur)
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS).values
    q = cur.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS).values
    return float(np.sum((q - p) * np.log(q / p)))


def status_from_psi(psi: float) -> str:
    return "alert" if psi >= PSI_ALERT else "warning" if psi >= PSI_WARN else "ok"


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    """Rapport de dérive par variable + statut global + recommandation de réentraînement."""
    features = {}
    for c in numeric:
        if c not in cur or c not in ref:
            continue
        if ref[c].nunique() <= 10:       # binaire / faible cardinalité : PSI par modalité
            psi = psi_categorical(ref[c], cur[c])
        else:
            psi = psi_numeric(ref[c], cur[c])
        features[c] = {"type": "numeric", "psi": round(psi, 4), "ks_pvalue": float(ks_2samp(ref[c], cur[c]).pvalue),
                       "ref_mean": float(ref[c].mean()), "cur_mean": float(cur[c].mean()), "status": status_from_psi(psi)}
    for c in categorical:
        if c not in cur or c not in ref:
            continue
        psi = psi_categorical(ref[c], cur[c])
        features[c] = {"type": "categorical", "psi": round(psi, 4), "status": status_from_psi(psi)}
    statuses = [f["status"] for f in features.values()]
    n_alert, n_warn = statuses.count("alert"), statuses.count("warning")
    overall = "alert" if n_alert else "warning" if n_warn else "ok"
    return {"overall_status": overall, "n_features": len(features), "n_alert": n_alert, "n_warning": n_warn,
            "n_samples_current": int(len(cur)), "n_samples_reference": int(len(ref)),
            "retrain_recommended": bool(n_alert >= 1 or n_warn >= max(2, int(0.3 * len(features)))),
            "features": dict(sorted(features.items(), key=lambda kv: -kv[1]["psi"]))}


def forecast_performance(y_true, y_pred, baseline_mape: float, ratio_threshold: float = 1.5) -> dict:
    """Compare le MAPE récent à celui mesuré lors du backtest d'entraînement."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = y_true != 0
    cur = float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)
    ratio = cur / baseline_mape if baseline_mape else float("inf")
    return {"current_mape": cur, "baseline_mape": baseline_mape, "ratio": float(ratio),
            "status": "alert" if ratio >= ratio_threshold else "warning" if ratio >= 1.2 else "ok",
            "retrain_recommended": bool(ratio >= ratio_threshold)}


def churn_performance(y_true, proba, ref_roc_auc: float, max_drop: float = 0.05) -> dict:
    """Quand les vrais labels de churn arrivent (avec retard), vérifie la perte d'AUC."""
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, proba))
    drop = ref_roc_auc - auc
    return {"current_roc_auc": auc, "reference_roc_auc": ref_roc_auc, "drop": float(drop),
            "status": "alert" if drop >= max_drop else "warning" if drop >= max_drop / 2 else "ok",
            "retrain_recommended": bool(drop >= max_drop)}

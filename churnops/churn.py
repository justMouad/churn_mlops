"""Prédiction du churn : sélection de modèle (LogReg vs Gradient Boosting) + seuil optimisé F1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, FEATURES, NUMERIC, SEED, TARGET


def _prep() -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])


def candidates(quick: bool = False) -> dict:
    hgb_grid = {"clf__learning_rate": [0.1], "clf__max_depth": [3]} if quick else \
               {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5], "clf__max_iter": [150]}
    return {
        "logreg": (Pipeline([("prep", _prep()), ("clf", LogisticRegression(max_iter=1000))]),
                   {"clf__C": [1.0] if quick else [0.1, 1.0, 10.0]}),
        "hist_gb": (Pipeline([("prep", _prep()), ("clf", HistGradientBoostingClassifier(random_state=SEED))]),
                    hgb_grid),
    }


def best_f1_threshold(y, proba) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([f1_score(y, proba >= t, zero_division=0) for t in grid]))])


def evaluate(y, proba, threshold: float) -> dict:
    pred = proba >= threshold
    return {"roc_auc": float(roc_auc_score(y, proba)), "pr_auc": float(average_precision_score(y, proba)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "brier": float(brier_score_loss(y, proba)),
            "threshold": float(threshold), "positive_rate": float(np.mean(y))}


def train_churn(df: pd.DataFrame, quick: bool = False):
    """Retourne (pipeline, metadata, reference_df).

    train 60 % (CV + tuning) / val 20 % (seuil, choix du modèle) / test 20 % (métriques finales).
    """
    X, y = df[FEATURES], df[TARGET]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)

    fitted, comparison = {}, {}
    for name, (pipe, grid) in candidates(quick).items():
        gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=3, n_jobs=1).fit(X_tr, y_tr)
        p_va = gs.predict_proba(X_va)[:, 1]
        comparison[name] = {"cv_pr_auc": float(gs.best_score_), "val_pr_auc": float(average_precision_score(y_va, p_va)),
                            "best_params": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in gs.best_params_.items()}}
        fitted[name] = gs.best_estimator_
    best = max(comparison, key=lambda k: comparison[k]["val_pr_auc"])
    model = fitted[best]

    thr = best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])
    p_te = model.predict_proba(X_te)[:, 1]
    meta = {"model": best, "comparison": comparison, "test_metrics": evaluate(y_te.values, p_te, thr),
            "threshold": thr, "n_train": int(len(X_tr)), "features": FEATURES}
    ref = X_te.copy().reset_index(drop=True)
    ref["churn_probability"] = p_te
    return model, meta, ref.head(4000)

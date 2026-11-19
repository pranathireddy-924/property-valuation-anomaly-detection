"""Evaluation metrics (MAE, RMSE, R² as named in the objectives, plus valuation-industry
measures such as the share of predictions within 10% / 20% of the sale price)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(price_true, price_pred) -> dict:
    t = np.asarray(price_true, dtype="float64")
    p = np.asarray(price_pred, dtype="float64")
    ape = np.abs(p - t) / t
    return {
        "n": len(t),
        "MAE": mean_absolute_error(t, p),
        "RMSE": float(np.sqrt(mean_squared_error(t, p))),
        "R2": r2_score(t, p),
        "R2_log": r2_score(np.log(t), np.log(p)),
        "MAPE_%": 100 * ape.mean(),
        "MdAPE_%": 100 * np.median(ape),
        "within_10%": 100 * (ape <= 0.10).mean(),
        "within_20%": 100 * (ape <= 0.20).mean(),
    }


def metrics_by_group(price_true, price_pred, groups: pd.Series, min_n: int = 30) -> pd.DataFrame:
    frame = pd.DataFrame({"t": np.asarray(price_true), "p": np.asarray(price_pred), "g": np.asarray(groups)})
    rows = []
    for g, part in frame.groupby("g", observed=True):
        if len(part) >= min_n:
            rows.append({"group": g, **regression_metrics(part["t"], part["p"])})
    return pd.DataFrame(rows)


def to_price(log_pred) -> np.ndarray:
    return np.exp(np.asarray(log_pred, dtype="float64"))

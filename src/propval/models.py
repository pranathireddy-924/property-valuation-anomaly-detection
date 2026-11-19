"""Time-based split and the three valuation models.

* Naive local baseline : smoothed mean price of the postcode district x property type
* Linear Regression    : the baseline named in the proposal
* XGBoost Regression   : the main model (native categorical support, early stopping)

All models predict log(price); predictions are converted back to pounds for evaluation.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config

log = logging.getLogger(__name__)

# High-cardinality codes are left out of the linear model: their information is
# already carried by the local price encodings (te_lad etc.).
LINEAR_EXCLUDE = {"lad", "pc_area"}


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Chronological split: train on older sales, validate and test on later ones."""
    train_end, val_end = pd.Timestamp(cfg.train_end), pd.Timestamp(cfg.val_end)
    parts = {
        "train": df.loc[df["date"] <= train_end],
        "val": df.loc[(df["date"] > train_end) & (df["date"] <= val_end)],
        "test": df.loc[df["date"] > val_end],
    }
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"The {name} period is empty - check train_end/val_end and years")
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


# ---------------------------------------------------------------------------
# Linear Regression baseline
# ---------------------------------------------------------------------------
def build_linear_model(X: pd.DataFrame) -> Pipeline:
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype) and c not in LINEAR_EXCLUDE]
    nums = [c for c in X.columns if c not in cats and c not in LINEAR_EXCLUDE]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)), cats),
    ])
    return Pipeline([("prep", pre), ("model", LinearRegression())])


def _linear_frame(X: pd.DataFrame) -> pd.DataFrame:
    """sklearn imputers do not accept pandas categoricals - pass them as plain strings."""
    out = X.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object).where(out[c].notna(), "missing")
    return out


def fit_linear(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    t0 = time.time()
    model = build_linear_model(X)
    model.fit(_linear_frame(X), y)
    log.info("Linear Regression fitted in %.1fs", time.time() - t0)
    return model


def predict_linear(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict(_linear_frame(X))


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def make_xgb(cfg: Config, params: dict, learning_rate: float, n_estimators: int | None = None) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=n_estimators or cfg.xgb_n_estimators,
        learning_rate=learning_rate,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        early_stopping_rounds=cfg.xgb_early_stopping,
        eval_metric="rmse",
        n_jobs=-1,
        random_state=cfg.random_state,
        **params,
    )


def fit_xgb(cfg: Config, Xtr, ytr, Xva, yva, params: dict | None = None,
            learning_rate: float | None = None, n_estimators: int | None = None) -> xgb.XGBRegressor:
    t0 = time.time()
    model = make_xgb(cfg, params or cfg.xgb_params, learning_rate or cfg.xgb_learning_rate, n_estimators)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    log.info("XGBoost (%d features) fitted in %.1fs, best iteration %d, val RMSE(log) %.4f",
             Xtr.shape[1], time.time() - t0, model.best_iteration, model.best_score)
    return model


def predict_xgb(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X, iteration_range=(0, model.best_iteration + 1))


PARAM_SPACE = {
    "max_depth": [6, 7, 8, 9, 10],
    "min_child_weight": [5, 10, 20, 50, 100],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    "gamma": [0.0, 0.01, 0.05],
}


def tune_xgb(cfg: Config, Xtr, ytr, Xva, yva) -> tuple[dict, pd.DataFrame]:
    """Random search on a training subsample, scored on the validation period.

    The default parameters are always evaluated as the first candidate, so tuning
    can only keep or improve on them.
    """
    rng = np.random.default_rng(cfg.random_state)
    if len(Xtr) > cfg.tune_sample_rows:
        idx = rng.choice(len(Xtr), cfg.tune_sample_rows, replace=False)
        Xs, ys = Xtr.iloc[idx], ytr.iloc[idx]
    else:
        Xs, ys = Xtr, ytr

    candidates = [dict(cfg.xgb_params)]
    for _ in range(cfg.tune_iterations - 1):
        candidates.append({k: v[rng.integers(len(v))] for k, v in PARAM_SPACE.items()})

    rows = []
    for i, params in enumerate(candidates, 1):
        m = fit_xgb(cfg, Xs, ys, Xva, yva, params=params, learning_rate=0.1, n_estimators=2000)
        rows.append({"candidate": i, **params, "best_iteration": m.best_iteration, "val_rmse_log": m.best_score})
        log.info("Tuning %d/%d: val RMSE(log) %.4f  %s", i, len(candidates), m.best_score, params)
    results = pd.DataFrame(rows).sort_values("val_rmse_log").reset_index(drop=True)
    best = {k: results.loc[0, k] for k in PARAM_SPACE}
    best = {k: (int(v) if k in ("max_depth", "min_child_weight") else float(v)) for k, v in best.items()}
    return best, results

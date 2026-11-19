"""SHAP explanations for the XGBoost model.

XGBoost computes exact TreeSHAP values natively (pred_contribs=True), which also
works with its categorical splits. Values are in log-price units, so a SHAP value
phi corresponds to a multiplicative effect of exp(phi) on the predicted price -
reported as a percentage, which is easier for property professionals to read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import FEATURE_GROUPS, PROPERTY_TYPE_NAMES, REGION_NAMES

FEATURE_LABELS = {
    "property_type": "Property type", "old_new": "New build", "duration": "Tenure",
    "sale_year": "Sale year", "sale_month": "Sale month", "months_since_start": "Months since Jan of first year",
    "region": "Region", "lad": "Local authority", "pc_area": "Postcode area",
    "lat": "Latitude", "long": "Longitude", "ruc21": "Rural/urban class", "imd_pct": "Deprivation percentile",
    "te_sector": "Local price level: postcode sector", "te_district": "Local price level: postcode district",
    "te_lsoa": "Local price level: LSOA", "te_msoa": "Local price level: MSOA",
    "te_lad": "Local price level: local authority", "te_district_type": "Local price level: district x type",
    "log_n_sector": "Sales volume in sector",
    "hpi_type_price_log": "HPI average price (type, LA)", "hpi_avg_price_log": "HPI average price (LA)",
    "hpi_12m_change": "HPI 12-month change",
}


def shap_values(model: xgb.XGBRegressor, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (SHAP matrix n x features, base value per row)."""
    booster = model.get_booster()
    dm = xgb.DMatrix(X, enable_categorical=True)
    contribs = booster.predict(dm, pred_contribs=True, iteration_range=(0, model.best_iteration + 1))
    return contribs[:, :-1], contribs[:, -1]


def global_importance(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(values).mean(axis=0)})
    imp["label"] = imp["feature"].map(FEATURE_LABELS).fillna(imp["feature"])
    imp["share_%"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def group_importance(importance: pd.DataFrame) -> pd.DataFrame:
    lookup = {f: g for g, feats in FEATURE_GROUPS.items() for f in feats}
    out = importance.assign(group=importance["feature"].map(lookup))
    return (out.groupby("group", as_index=False)[["mean_abs_shap", "share_%"]].sum()
            .sort_values("share_%", ascending=False).reset_index(drop=True))


def explain_row(values_row: np.ndarray, X_row: pd.Series, top_n: int = 8) -> pd.DataFrame:
    """Top contributions for one prediction, as % effect on the predicted price."""
    df = pd.DataFrame({
        "feature": X_row.index,
        "label": [FEATURE_LABELS.get(f, f) for f in X_row.index],
        "value": [_display_value(f, v) for f, v in X_row.items()],
        "shap": values_row,
    })
    df["effect_%"] = 100 * (np.exp(df["shap"]) - 1)
    df = df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(top_n)
    return df.reset_index(drop=True)


CODE_NAMES = {
    "property_type": PROPERTY_TYPE_NAMES,
    "duration": {"F": "Freehold", "L": "Leasehold"},
    "old_new": {"Y": "Yes", "N": "No"},
    "region": REGION_NAMES,
}


def _display_value(feature: str, value) -> str:
    if pd.isna(value):
        return "missing"
    if feature in CODE_NAMES:
        return CODE_NAMES[feature].get(str(value), str(value))
    if feature.startswith("te_") or feature in ("hpi_type_price_log", "hpi_avg_price_log"):
        return f"£{np.exp(value):,.0f}"
    if feature == "imd_pct":
        n = int(round(100 * value))
        suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix} percentile (100 = least deprived)"
    if feature == "hpi_12m_change":
        return f"{value:+.1f}%"
    if feature == "log_n_sector":
        return f"{np.expm1(value):,.0f} sales"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)

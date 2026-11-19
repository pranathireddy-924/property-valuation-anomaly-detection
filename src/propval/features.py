"""Spatial and temporal feature engineering.

The FeatureBuilder is *fitted on the training period only* and then applied to the
validation/test periods and to single properties in the prototype app, so no
information from the future leaks into the model.

Feature families
----------------
Property  : property type, new build flag, tenure
Temporal  : sale year, sale month, months since start of the study window
Spatial   : region, local authority, postcode area, latitude/longitude, rural-urban
            class, deprivation percentile and *local price patterns* - smoothed mean
            log price of earlier sales in the same postcode sector / district /
            LSOA / MSOA / local authority (target encoding)
Market    : UK House Price Index level for the local authority and property type,
            lagged by the HPI publication delay
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import ALL_FEATURES, CATEGORICAL_FEATURES, Config
from .data_loading import HPI_TYPE_COLUMNS

log = logging.getLogger(__name__)

# name -> columns forming the grouping key
TE_SPECS: dict[str, list[str]] = {
    "te_sector": ["pc_sector"],
    "te_district": ["pc_district"],
    "te_lsoa": ["lsoa21"],
    "te_msoa": ["msoa21"],
    "te_lad": ["lad"],
    "te_district_type": ["pc_district", "property_type"],
}


def _group_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    key = df[cols[0]].astype("string").fillna("__NA__")
    for c in cols[1:]:
        key = key + "|" + df[c].astype("string").fillna("__NA__")
    return key


def smoothed_mean(sum_: np.ndarray, count: np.ndarray, prior: float, m: float) -> np.ndarray:
    """Bayesian-smoothed mean: small groups are pulled towards the overall mean."""
    return (sum_ + prior * m) / (count + m)


class FeatureBuilder:
    def __init__(self, cfg: Config):
        self.smoothing = cfg.te_smoothing
        self.n_folds = cfg.te_folds
        self.seed = cfg.random_state
        self.hpi_lag = cfg.hpi_lag_months
        self.start_month_idx = cfg.years[0] * 12
        self.feature_names = list(ALL_FEATURES)
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, hpi: pd.DataFrame) -> "FeatureBuilder":
        y = train["log_price"]
        self.prior_ = float(y.mean())
        self.te_stats_ = {}
        for name, cols in TE_SPECS.items():
            self.te_stats_[name] = y.groupby(_group_key(train, cols)).agg(["sum", "count"])
        self.sector_counts_ = train.groupby("pc_sector").size()
        self.categories_ = {
            c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES
        }
        self.hpi_ = hpi.set_index(["area_code", "month_idx"]).sort_index()
        self.hpi_min_month_ = int(hpi["month_idx"].min())
        self.hpi_max_month_ = int(hpi["month_idx"].max())
        self.fitted = True
        return self

    def fit_transform(self, train: pd.DataFrame, hpi: pd.DataFrame) -> pd.DataFrame:
        """Fit, then encode the training rows *out-of-fold* so a sale never sees its own price."""
        self.fit(train, hpi)
        X = self._base_features(train)
        rng = np.random.default_rng(self.seed)
        folds = rng.integers(0, self.n_folds, len(train))
        y = train["log_price"].to_numpy()
        for name, cols in TE_SPECS.items():
            X[name] = self._oof_encode(_group_key(train, cols).to_numpy(), y, folds)
        return X[self.feature_names]

    def _oof_encode(self, key: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame({"k": key, "f": folds, "y": y})
        total = frame.groupby("k")["y"].agg(["sum", "count"])
        per_fold = frame.groupby(["k", "f"])["y"].agg(["sum", "count"])
        tot = total.reindex(frame["k"]).to_numpy()
        own = per_fold.reindex(pd.MultiIndex.from_arrays([frame["k"], frame["f"]])).to_numpy()
        out_sum, out_cnt = tot[:, 0] - own[:, 0], tot[:, 1] - own[:, 1]
        return smoothed_mean(out_sum, out_cnt, self.prior_, self.smoothing).astype("float32")

    # ------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureBuilder must be fitted first")
        X = self._base_features(df)
        for name, cols in TE_SPECS.items():
            stats = self.te_stats_[name].reindex(_group_key(df, cols)).to_numpy()
            s, c = np.nan_to_num(stats[:, 0]), np.nan_to_num(stats[:, 1])
            X[name] = smoothed_mean(s, c, self.prior_, self.smoothing).astype("float32")
        return X[self.feature_names]

    # ------------------------------------------------------------- shared parts
    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for c in CATEGORICAL_FEATURES:
            vals = df[c].astype("string")
            vals = vals.where(vals.isin(self.categories_[c]))   # unseen category -> missing
            X[c] = pd.Categorical(vals.astype(object), categories=self.categories_[c])
        X["sale_year"] = df["sale_year"].astype("float32")
        X["sale_month"] = df["sale_month"].astype("float32")
        X["months_since_start"] = (df["month_idx"] - self.start_month_idx).astype("float32")
        X["lat"] = df["lat"].astype("float32")
        X["long"] = df["long"].astype("float32")
        X["imd_pct"] = df["imd_pct"].astype("float32")
        X["log_n_sector"] = np.log1p(
            self.sector_counts_.reindex(df["pc_sector"]).fillna(0).to_numpy()).astype("float32")
        X = X.join(self._hpi_features(df))
        return X

    def _hpi_lookup(self, area: pd.Series, month: np.ndarray) -> pd.DataFrame:
        idx = pd.MultiIndex.from_arrays([area.astype("string").fillna("__NA__").to_numpy(), month])
        return self.hpi_.reindex(idx).reset_index(drop=True)

    def _hpi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """HPI level for the sale's local authority (falls back to region), lagged by the
        publication delay so the model only uses figures available at the time of sale."""
        ref = np.clip(df["month_idx"].to_numpy() - self.hpi_lag, self.hpi_min_month_, self.hpi_max_month_)
        lad = self._hpi_lookup(df["lad"], ref)
        reg = self._hpi_lookup(df["region"], ref)
        h = lad.fillna(reg)

        ptype = df["property_type"].astype("string").to_numpy()
        type_price = np.full(len(df), np.nan)
        for code, col in HPI_TYPE_COLUMNS.items():
            m = ptype == code
            type_price[m] = h[col].to_numpy()[m]
        type_price = np.where(np.isnan(type_price), h["avg_price"].to_numpy(), type_price)

        return pd.DataFrame({
            "hpi_type_price_log": np.log(type_price).astype("float32"),
            "hpi_avg_price_log": np.log(h["avg_price"].to_numpy()).astype("float32"),
            "hpi_12m_change": h["change_12m"].to_numpy().astype("float32"),
        }, index=df.index)

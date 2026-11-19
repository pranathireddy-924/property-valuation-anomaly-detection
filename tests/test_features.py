"""Unit tests - feature engineering (test cases UT-07 to UT-12)."""
import numpy as np
import pandas as pd
import pytest

from propval.config import ALL_FEATURES, Config
from propval.data_loading import add_derived_columns
from propval.features import FeatureBuilder, smoothed_mean


def _frame(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    defaults = {"old_new": "N", "duration": "F", "lad": "E06000001", "region": "E12000001",
                "lsoa21": "L1", "msoa21": "M1", "lat": 53.0, "long": -1.5, "ruc21": "UN1", "imd_pct": 0.5}
    for k, v in defaults.items():
        if k not in df:
            df[k] = v
    df = add_derived_columns(df)
    if "price" in df:
        df["log_price"] = np.log(df["price"].astype(float))
    return df


@pytest.fixture
def hpi():
    rows = []
    for m in range(2021 * 12, 2025 * 12):
        rows.append({"area_code": "E06000001", "month_idx": m, "avg_price": 100_000 + 1000 * (m - 2021 * 12),
                     "change_12m": 5.0, "DetachedPrice": 300_000.0, "SemiDetachedPrice": 200_000.0,
                     "TerracedPrice": 150_000.0, "FlatPrice": 120_000.0})
    return pd.DataFrame(rows)


def test_ut07_smoothed_mean():
    # no data -> prior; lots of data -> close to the group mean
    assert smoothed_mean(np.array([0.0]), np.array([0.0]), 12.0, 20)[0] == pytest.approx(12.0)
    assert smoothed_mean(np.array([10_000.0]), np.array([1000.0]), 12.0, 20)[0] == pytest.approx(10.039, abs=1e-3)


def test_ut08_out_of_fold_encoding_never_uses_own_price(hpi):
    """Each sale is alone in its sector, so out-of-fold its encoding must equal the prior."""
    train = _frame([{"postcode": f"A{i} 1AA", "property_type": "D", "date": "2022-05-01", "price": 100_000 * (i + 1)}
                    for i in range(10)])
    cfg = Config()
    fb = FeatureBuilder(cfg)
    X = fb.fit_transform(train, hpi)
    assert np.allclose(X["te_sector"], fb.prior_, atol=1e-5)


def test_ut09_unseen_area_falls_back_to_prior(hpi):
    train = _frame([{"postcode": "A1 1AA", "property_type": "D", "date": "2022-05-01", "price": 200_000}] * 5)
    fb = FeatureBuilder(Config()).fit(train, hpi)
    new = _frame([{"postcode": "Z9 9ZZ", "property_type": "D", "date": "2026-01-01"}])
    X = fb.transform(new)
    assert X["te_sector"].iloc[0] == pytest.approx(fb.prior_)
    assert X["log_n_sector"].iloc[0] == 0


def test_ut10_hpi_is_lagged(hpi):
    cfg = Config()
    cfg.hpi_lag_months = 2
    train = _frame([{"postcode": "A1 1AA", "property_type": "D", "date": "2022-05-01", "price": 200_000}] * 3)
    fb = FeatureBuilder(cfg).fit(train, hpi)
    X = fb.transform(_frame([{"postcode": "A1 1AA", "property_type": "D", "date": "2023-06-15"}]))
    expected_month = 2023 * 12 + 6 - 1 - 2                       # April 2023
    expected = 100_000 + 1000 * (expected_month - 2021 * 12)
    assert np.exp(X["hpi_avg_price_log"].iloc[0]) == pytest.approx(expected, rel=1e-4)
    assert np.exp(X["hpi_type_price_log"].iloc[0]) == pytest.approx(300_000, rel=1e-4)  # detached column


def test_ut11_unseen_category_becomes_missing(hpi):
    train = _frame([{"postcode": "A1 1AA", "property_type": "D", "date": "2022-05-01", "price": 200_000}] * 3)
    fb = FeatureBuilder(Config()).fit(train, hpi)
    X = fb.transform(_frame([{"postcode": "A1 1AA", "property_type": "D", "date": "2023-01-01", "region": "E12000009"}]))
    assert pd.isna(X["region"].iloc[0])
    assert list(X["region"].cat.categories) == ["E12000001"]


def test_ut12_feature_matrix_has_all_features(hpi):
    train = _frame([{"postcode": "A1 1AA", "property_type": t, "date": "2022-05-01", "price": 200_000} for t in "DSTF"])
    X = FeatureBuilder(Config()).fit_transform(train, hpi)
    assert list(X.columns) == ALL_FEATURES
    assert X["months_since_start"].iloc[0] == 4     # May 2022 vs Jan 2022

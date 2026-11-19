"""Unit tests - split, models, metrics and anomaly scoring (test cases UT-13 to UT-19)."""
import numpy as np
import pandas as pd
import pytest

from propval.anomaly import ResidualAnomalyDetector
from propval.config import Config
from propval.evaluation import regression_metrics
from propval.models import fit_linear, fit_xgb, predict_linear, predict_xgb, time_split


def test_ut13_time_split_is_chronological():
    df = pd.DataFrame({"date": pd.date_range("2022-01-01", "2026-06-30", freq="7D")})
    s = time_split(df, Config())
    assert s["train"]["date"].max() < s["val"]["date"].min()
    assert s["val"]["date"].max() < s["test"]["date"].min()
    assert sum(len(v) for v in s.values()) == len(df)


def test_ut14_time_split_rejects_empty_period():
    df = pd.DataFrame({"date": pd.date_range("2022-01-01", "2023-06-30", freq="7D")})
    with pytest.raises(ValueError):
        time_split(df, Config())


def test_ut15_metrics_known_values():
    m = regression_metrics([100, 200, 400], [110, 180, 400])
    assert m["MAE"] == pytest.approx(10.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 400) / 3))
    assert m["MdAPE_%"] == pytest.approx(10.0)
    assert m["within_10%"] == pytest.approx(100.0)


def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": pd.Categorical(rng.choice(["x", "y", "z"], n))})
    y = pd.Series(12 + 0.5 * X["a"] + np.where(X["b"] > 0, 0.4, 0) + np.where(X["c"] == "x", 0.3, 0)
                  + rng.normal(0, 0.05, n))
    return X, y


def test_ut16_xgboost_beats_linear_on_nonlinear_signal():
    X, y = _toy()
    Xv, yv = _toy(seed=1)
    cfg = Config()
    cfg.xgb_n_estimators, cfg.xgb_early_stopping = 300, 20
    lin = fit_linear(X, y)
    xg = fit_xgb(cfg, X, y, Xv, yv, params={"max_depth": 4}, learning_rate=0.1)
    rmse = lambda p: np.sqrt(np.mean((yv - p) ** 2))  # noqa: E731
    assert rmse(predict_xgb(xg, Xv)) < rmse(predict_linear(lin, Xv))


def test_ut17_robust_z_flags_injected_outliers():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.1, 5000)
    det = ResidualAnomalyDetector(3.0).fit(r, ["D"] * 5000)
    z = det.score([0.0, 0.8, -0.8], ["D", "D", "D"])
    assert abs(z[0]) < 0.2
    assert list(det.label(z)) == ["Normal", "Unusually high", "Unusually low"]
    assert det.label(det.score(r, ["D"] * 5000)).tolist().count("Normal") / 5000 > 0.99


def test_ut18_robust_statistics_resist_contamination():
    """5% gross errors in the calibration set must not inflate the threshold much."""
    rng = np.random.default_rng(1)
    clean = rng.normal(0, 0.1, 9500)
    dirty = np.concatenate([clean, rng.normal(0, 2.0, 500)])
    s_clean = ResidualAnomalyDetector().fit(clean, ["D"] * len(clean)).stats_["D"][1]
    s_dirty = ResidualAnomalyDetector().fit(dirty, ["D"] * len(dirty)).stats_["D"][1]
    assert s_dirty / s_clean < 1.15


def test_ut19_unknown_group_uses_global_statistics():
    det = ResidualAnomalyDetector().fit(np.random.default_rng(2).normal(0, 0.1, 1000), ["D"] * 1000)
    assert np.isfinite(det.score([0.5], ["F"])).all()

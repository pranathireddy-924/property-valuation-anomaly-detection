"""Integration and system tests - the whole pipeline on synthetic data
(test cases IT-01 to IT-04 and ST-01 to ST-04)."""
import numpy as np
import pandas as pd
import pytest

from propval.predict import PropertyValuer


def test_it01_pipeline_writes_all_evidence(pipeline_run):
    cfg = pipeline_run["cfg"]
    for name in ["01_cleaning_report.csv", "05_model_comparison.csv", "09_feature_ablation.csv",
                 "11_unusual_transactions_test.csv", "14_anomaly_check_category_b.csv",
                 "15_shap_global_importance.csv", "17_case_study_explanations.csv"]:
        assert (cfg.tables_dir / name).exists(), name
    assert len(list(cfg.figures_dir.glob("*.png"))) >= 12
    assert (cfg.models_dir / "valuer_bundle.joblib").exists()
    assert (cfg.output_dir / "RESULTS.md").exists()


def test_it02_cleaning_report_matches_injected_bad_rows(pipeline_run):
    rep = pd.read_csv(pipeline_run["cfg"].tables_dir / "01_cleaning_report.csv").set_index("step")
    assert rep.loc["Postcode match (ONSPD)", "rows_removed"] == 1   # ZZ9 9ZZ


def test_it03_unusual_transactions_file_has_no_addresses(pipeline_run):
    flagged = pd.read_csv(pipeline_run["cfg"].tables_dir / "11_unusual_transactions_test.csv")
    assert not {"postcode", "paon", "saon", "street"} & set(flagged.columns)


def test_it04_shap_values_add_up_to_prediction(pipeline_run):
    """SHAP property: base value + sum of contributions = model output (log price)."""
    import joblib
    from propval.explain import shap_values
    from propval.models import predict_xgb
    cfg = pipeline_run["cfg"]
    bundle = joblib.load(cfg.models_dir / "valuer_bundle.joblib")
    valuer = PropertyValuer.load(cfg.models_dir)
    row = valuer.lookup.index[0]
    from propval.data_loading import add_derived_columns
    df = pd.DataFrame([{"postcode": row, "property_type": "S", "old_new": "N", "duration": "F",
                        "date": pd.Timestamp("2024-03-01")}])
    df = add_derived_columns(pd.concat([df, valuer.lookup.loc[[row]].reset_index(drop=True)], axis=1))
    X = bundle["builder"].transform(df)
    vals, base = shap_values(bundle["model"], X)
    assert vals.sum() + base[0] == pytest.approx(predict_xgb(bundle["model"], X)[0], abs=1e-3)


def test_st01_xgboost_outperforms_baselines(pipeline_run):
    cmp = pipeline_run["result"]["comparison"]
    test = cmp[cmp["period"] == "test"].set_index("model")
    assert test.loc["XGBoost", "MAE"] < test.loc["Linear Regression", "MAE"]
    assert test.loc["XGBoost", "MAE"] < test.loc["Naive local median", "MAE"]
    assert test.loc["XGBoost", "R2_log"] > 0.8


def test_st02_injected_anomalies_are_detected(pipeline_run):
    scored = pipeline_run["result"]["scored_test"]
    inj = scored[scored["transaction_id"].isin(pipeline_run["injected_ids"])]
    assert len(inj) == 20
    recall = (inj["assessment"] != "Normal").mean()
    false_alarm = (scored.loc[~scored["transaction_id"].isin(pipeline_run["injected_ids"]), "assessment"]
                   != "Normal").mean()
    assert recall >= 0.9
    assert false_alarm < 0.03
    # direction is right: x4 prices are 'Unusually high', /4 prices 'Unusually low'
    assert (inj.loc[inj["difference_%"] > 0, "assessment"] == "Unusually high").all()


def test_st03_category_b_sales_flagged_more_often(pipeline_run):
    rates = pipeline_run["result"]["anomaly_rates"].set_index("group")["flag_rate_%"]
    assert rates.iloc[1] > rates.iloc[0]


def test_st04_prototype_valuer(pipeline_run):
    valuer = PropertyValuer.load(pipeline_run["cfg"].models_dir)
    normal = valuer.value("bb1 2ab", "D", False, "F", "2024-06-01")
    assert normal["range_low"] < normal["predicted_price"] < normal["range_high"]
    assert len(normal["explanation"]) == 8
    high = valuer.value("BB1 2AB", "D", False, "F", "2024-06-01", actual_price=normal["predicted_price"] * 4)
    assert high["assessment"] == "Unusually high"
    ok = valuer.value("BB1 2AB", "D", False, "F", "2024-06-01", actual_price=normal["predicted_price"])
    assert ok["assessment"] == "Normal"
    with pytest.raises(ValueError):
        valuer.value("ZZ9 9ZZ", "D", False, "F", "2024-06-01")
    with pytest.raises(ValueError):
        valuer.value("BB1 2AB", "X", False, "F", "2024-06-01")

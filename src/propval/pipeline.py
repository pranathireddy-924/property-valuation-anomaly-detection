"""End-to-end pipeline: data -> features -> models -> evaluation -> anomalies -> SHAP.

Every stage writes its evidence (tables as CSV, charts as PNG) to outputs/ so the
results can be dropped straight into the dissertation.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import plots
from .anomaly import LABEL_NORMAL, ResidualAnomalyDetector
from .config import ABLATION_SETS, PROPERTY_TYPE_NAMES, REGION_NAMES, Config
from .data_loading import load_base_dataset, load_hpi
from .evaluation import metrics_by_group, regression_metrics, to_price
from .dashboard_data import build_la_names, prepare as prepare_dashboard
from .explain import (FEATURE_LABELS, explain_row, global_importance, group_importance, set_la_names,
                      shap_values)
from .features import FeatureBuilder
from .models import fit_linear, fit_xgb, predict_linear, predict_xgb, time_split, tune_xgb

log = logging.getLogger("propval")


def _setup_logging(cfg: Config) -> None:
    cfg.make_dirs()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in (logging.StreamHandler(), logging.FileHandler(Path(cfg.output_dir) / "pipeline.log", mode="w")):
        h.setFormatter(fmt)
        root.addHandler(h)


class Stage:
    """Context manager that logs how long each pipeline stage took."""
    timings: dict[str, float] = {}

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        log.info("=== %s ===", self.name)
        self.t0 = time.time()

    def __exit__(self, *exc):
        Stage.timings[self.name] = round(time.time() - self.t0, 1)
        log.info("--- %s done in %.1fs", self.name, Stage.timings[self.name])


def _md_table(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def run(cfg: Config, use_cache: bool = True) -> dict:
    _setup_logging(cfg)
    Stage.timings = {}
    T, F = cfg.tables_dir, cfg.figures_dir
    t_start = time.time()
    (Path(cfg.output_dir) / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))

    # ------------------------------------------------------------------ 1. data
    with Stage("1. Load and clean data"):
        df, lookup, cleaning = load_base_dataset(cfg, use_cache=use_cache)
        hpi = load_hpi(cfg.hpi_file)
        cleaning.to_csv(T / "01_cleaning_report.csv", index=False)
        standard = df.loc[df["is_standard"]].reset_index(drop=True)
        category_b = df.loc[~df["is_standard"]].reset_index(drop=True)
        splits = time_split(standard, cfg)
        log.info("Standard (cat. A) sales: %s | category B: %s", f"{len(standard):,}", f"{len(category_b):,}")

        summary = pd.DataFrame([{
            "split": name, "sales": len(p), "from": p["date"].min().date(), "to": p["date"].max().date(),
            "median_price": float(p["price"].median()), "mean_price": float(p["price"].mean()),
            **{f"share_{PROPERTY_TYPE_NAMES[k]}_%": 100 * float((p["property_type"] == k).mean()) for k in PROPERTY_TYPE_NAMES},
            "new_build_%": 100 * float((p["old_new"] == "Y").mean()),
            "leasehold_%": 100 * float((p["duration"] == "L").mean()),
        } for name, p in splits.items()])
        summary.to_csv(T / "02_data_summary.csv", index=False)
        plots.price_distribution(standard, F / "fig01_price_distribution.png")
        plots.monthly_trend(standard, cfg.train_end, cfg.val_end, F / "fig02_monthly_trend_and_split.png")

    # ------------------------------------------------------------- 2. features
    with Stage("2. Spatial and temporal feature engineering"):
        fb = FeatureBuilder(cfg)
        Xtr = fb.fit_transform(splits["train"], hpi)
        Xva, Xte = fb.transform(splits["val"]), fb.transform(splits["test"])
        ytr, yva, yte = (splits[k]["log_price"] for k in ("train", "val", "test"))
        missing = pd.DataFrame({
            "feature": Xtr.columns, "label": [FEATURE_LABELS.get(c, c) for c in Xtr.columns],
            "missing_train_%": 100 * Xtr.isna().mean().to_numpy(),
            "missing_test_%": 100 * Xte.isna().mean().to_numpy(),
        })
        missing.to_csv(T / "03_feature_missingness.csv", index=False)

    # ---------------------------------------------------------------- 3. models
    preds_val, preds_test = {}, {}
    with Stage("3a. Naive local baseline"):
        preds_val["Naive local median"] = Xva["te_district_type"].to_numpy()
        preds_test["Naive local median"] = Xte["te_district_type"].to_numpy()

    with Stage("3b. Linear Regression baseline"):
        linear = fit_linear(Xtr, ytr)
        preds_val["Linear Regression"] = predict_linear(linear, Xva)
        preds_test["Linear Regression"] = predict_linear(linear, Xte)

    with Stage("3c. XGBoost tuning and training"):
        params = dict(cfg.xgb_params)
        if cfg.tune:
            params, tuning = tune_xgb(cfg, Xtr, ytr, Xva, yva)
            tuning.to_csv(T / "04_xgb_tuning_results.csv", index=False)
        (T / "04_xgb_best_params.json").write_text(json.dumps(params, indent=2))
        model = fit_xgb(cfg, Xtr, ytr, Xva, yva, params=params)
        preds_val["XGBoost"] = predict_xgb(model, Xva)
        preds_test["XGBoost"] = predict_xgb(model, Xte)

    with Stage("4. Evaluation"):
        rows = []
        for period, preds, y in (("validation", preds_val, yva), ("test", preds_test, yte)):
            for name, p in preds.items():
                rows.append({"period": period, "model": name, **regression_metrics(np.exp(y), to_price(p))})
        comparison = pd.DataFrame(rows)
        comparison.to_csv(T / "05_model_comparison.csv", index=False)
        test_cmp = comparison.loc[comparison["period"] == "test"].reset_index(drop=True)
        plots.model_comparison(test_cmp, F / "fig03_model_comparison.png")

        test = splits["test"]
        actual, xgb_price = test["price"].to_numpy(), to_price(preds_test["XGBoost"])
        by_type = metrics_by_group(actual, xgb_price, test["property_type"].map(PROPERTY_TYPE_NAMES))
        by_region = metrics_by_group(actual, xgb_price, test["region"].map(REGION_NAMES).fillna(test["region"]))
        by_type.to_csv(T / "06_xgb_metrics_by_property_type.csv", index=False)
        by_region.to_csv(T / "07_xgb_metrics_by_region.csv", index=False)
        plots.predicted_vs_actual(actual, xgb_price, F / "fig04_predicted_vs_actual_xgb.png",
                                  "XGBoost: predicted vs actual price (test period)")
        plots.predicted_vs_actual(actual, to_price(preds_test["Linear Regression"]),
                                  F / "fig05_predicted_vs_actual_linear.png",
                                  "Linear Regression: predicted vs actual price (test period)")
        plots.error_by_group(by_region, F / "fig06_error_by_region.png", "XGBoost error by region")
        plots.error_by_group(by_type, F / "fig07_error_by_property_type.png", "XGBoost error by property type")

        gain = model.get_booster().get_score(importance_type="gain")
        gain_df = (pd.DataFrame({"feature": list(gain), "gain": list(gain.values())})
                   .assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
                   .sort_values("gain", ascending=False))
        gain_df["share_%"] = 100 * gain_df["gain"] / gain_df["gain"].sum()
        gain_df.to_csv(T / "08_xgb_feature_importance_gain.csv", index=False)

    # ------------------------------------------------------------- 5. ablation
    ablation = None
    if cfg.run_ablation:
        with Stage("5. Feature ablation (hypothesis test)"):
            rows = []
            for name, cols in ABLATION_SETS.items():
                m = fit_xgb(cfg, Xtr[cols], ytr, Xva[cols], yva, params=params,
                            learning_rate=cfg.ablation_learning_rate)
                p = predict_xgb(m, Xte[cols])
                rows.append({"feature_set": name, "n_features": len(cols),
                             **regression_metrics(test["price"], to_price(p))})
            ablation = pd.DataFrame(rows)
            ablation.to_csv(T / "09_feature_ablation.csv", index=False)
            plots.ablation_chart(ablation, F / "fig08_feature_ablation.png")

    # ---------------------------------------------------------- 6. anomalies
    with Stage("6. Price anomaly detection"):
        detector = ResidualAnomalyDetector(cfg.anomaly_z_threshold)
        detector.fit(yva.to_numpy() - preds_val["XGBoost"], splits["val"]["property_type"])
        detector.stats_table().to_csv(T / "10_anomaly_thresholds_by_type.csv", index=False)

        resid = yte.to_numpy() - preds_test["XGBoost"]
        z = detector.score(resid, test["property_type"])
        labels = detector.label(z)
        scored = pd.DataFrame({
            "transaction_id": test["transaction_id"],
            "date": test["date"].dt.date,
            "postcode_district": test["pc_district"],   # full postcode/address withheld (ethics)
            "region": test["region"].map(REGION_NAMES),
            "property_type": test["property_type"].map(PROPERTY_TYPE_NAMES),
            "new_build": test["old_new"].eq("Y"),
            "tenure": test["duration"].map({"F": "Freehold", "L": "Leasehold"}),
            "actual_price": test["price"],
            "predicted_price": xgb_price.round(0),
            "difference_%": 100 * (test["price"] / xgb_price - 1),
            "anomaly_score": z,
            "assessment": labels,
        })
        scored.to_parquet(cfg.cache_dir / "test_predictions.parquet", index=False)
        flagged = scored.loc[scored["assessment"] != LABEL_NORMAL].reindex(
            scored.loc[scored["assessment"] != LABEL_NORMAL, "anomaly_score"].abs().sort_values(ascending=False).index)
        flagged.to_csv(T / "11_unusual_transactions_test.csv", index=False)

        anomaly_summary = (scored.groupby(["property_type", "assessment"]).size().unstack(fill_value=0))
        anomaly_summary["flag_rate_%"] = 100 * (1 - anomaly_summary.get(LABEL_NORMAL, 0) / anomaly_summary.sum(axis=1))
        anomaly_summary.reset_index().to_csv(T / "12_anomaly_summary_by_type.csv", index=False)
        region_rates = scored.groupby("region")["assessment"].apply(lambda s: 100 * (s != LABEL_NORMAL).mean())
        region_rates.rename("flag_rate_%").reset_index().to_csv(T / "13_anomaly_rate_by_region.csv", index=False)
        plots.residual_distribution(z, cfg.anomaly_z_threshold, F / "fig09_anomaly_score_distribution.png")

        # Weak external check: category B sales (repossessions, buy-to-let and company
        # transfers - excluded from training) should look unusual more often.
        b_test = category_b.loc[category_b["date"] > pd.Timestamp(cfg.val_end)].reset_index(drop=True)
        rates = [{"group": "Standard sales (cat. A)", "sales": len(scored),
                  "flag_rate_%": 100 * float((labels != LABEL_NORMAL).mean()),
                  "median_abs_diff_%": float(np.median(np.abs(scored["difference_%"])))}]
        if len(b_test):
            pb = predict_xgb(model, fb.transform(b_test))
            zb = detector.score(b_test["log_price"].to_numpy() - pb, b_test["property_type"])
            rates.append({"group": "Additional sales (cat. B)", "sales": len(b_test),
                          "flag_rate_%": 100 * float((np.abs(zb) > cfg.anomaly_z_threshold).mean()),
                          "median_abs_diff_%": float(np.median(np.abs(100 * (b_test["price"] / to_price(pb) - 1))))})
        rates = pd.DataFrame(rates)
        rates.to_csv(T / "14_anomaly_check_category_b.csv", index=False)
        plots.anomaly_rates(rates, F / "fig10_anomaly_rate_category_a_vs_b.png")

    # -------------------------------------------------------------- 7. SHAP
    shap_groups = None
    if cfg.run_shap:
        with Stage("7. SHAP explanations"):
            rng = np.random.default_rng(cfg.random_state)
            idx = rng.choice(len(Xte), min(cfg.shap_sample_rows, len(Xte)), replace=False)
            Xs = Xte.iloc[idx]
            values, _ = shap_values(model, Xs)
            imp = global_importance(values, list(Xs.columns))
            imp.to_csv(T / "15_shap_global_importance.csv", index=False)
            shap_groups = group_importance(imp)
            shap_groups.to_csv(T / "16_shap_importance_by_feature_group.csv", index=False)
            plots.shap_bar(imp, F / "fig11_shap_global_importance.png")
            plots.shap_beeswarm(values, Xs, [FEATURE_LABELS.get(c, c) for c in Xs.columns],
                                F / "fig12_shap_beeswarm.png")

            try:   # readable local authority names in the case-study charts
                la = build_la_names(cfg)
                set_la_names(dict(zip(la["lad"], la["la_name"])))
            except FileNotFoundError:
                log.warning("Local authority names not found - codes will be shown")
            # Case studies for the report and the professional questionnaire.
            cases = _pick_cases(scored, cfg.n_case_studies, rng)
            case_rows = []
            for n, i in enumerate(cases, 1):
                v, _ = shap_values(model, Xte.iloc[[i]])
                contrib = explain_row(v[0], Xte.iloc[i])
                r = scored.iloc[i]
                header = (f"Case {n}: {r['property_type']}, {r['tenure']}, {r['region']} ({r['postcode_district']}), "
                          f"{r['date']}\nActual £{r['actual_price']:,.0f} vs predicted £{r['predicted_price']:,.0f} "
                          f"({r['difference_%']:+.0f}%) - {r['assessment']}, score {r['anomaly_score']:+.1f}")
                plots.case_explanation(contrib, header, F / f"fig13_case_{n}_explanation.png")
                for _, c in contrib.iterrows():
                    case_rows.append({"case": n, "assessment": r["assessment"], "actual_price": r["actual_price"],
                                      "predicted_price": r["predicted_price"], "anomaly_score": r["anomaly_score"],
                                      "factor": c["label"], "value": c["value"], "effect_%": c["effect_%"]})
            pd.DataFrame(case_rows).to_csv(T / "17_case_study_explanations.csv", index=False)

    # ---------------------------------------------------------- 8. save model
    with Stage("8. Save prototype model"):
        metrics = test_cmp.set_index("model").to_dict(orient="index")
        joblib.dump({"model": model, "builder": fb, "detector": detector, "metrics": metrics,
                     "config": cfg.to_dict()}, cfg.models_dir / "valuer_bundle.joblib")
        model.save_model(cfg.models_dir / "xgb_model.json")
        joblib.dump(linear, cfg.models_dir / "linear_model.joblib")
        lookup_src = cfg.cache_dir / "postcode_lookup.parquet"
        if lookup_src.exists():
            shutil.copy(lookup_src, cfg.models_dir / "postcode_lookup.parquet")
        else:
            lookup.to_parquet(cfg.models_dir / "postcode_lookup.parquet", index=False)

    with Stage("9. Prepare dashboard data"):
        prepare_dashboard(cfg, df)

    total = round(time.time() - t_start, 1)
    _write_results_md(cfg, cleaning, summary, comparison, ablation, rates, shap_groups, Stage.timings, total)
    log.info("Pipeline finished in %.1f min. Outputs in %s", total / 60, cfg.output_dir)
    return {"comparison": comparison, "ablation": ablation, "anomaly_rates": rates, "detector": detector,
            "scored_test": scored, "shap_groups": shap_groups}


def _pick_cases(scored: pd.DataFrame, n: int, rng) -> list[int]:
    """Choose a balanced set of example sales: unusually high, unusually low and normal."""
    picks = []
    per = max(1, n // 3)
    for label in ("Unusually high", "Unusually low"):
        pool = scored.index[scored["assessment"] == label].to_numpy()
        # moderately unusual examples are more realistic for review than the most extreme
        pool = pool[np.abs(scored.loc[pool, "anomaly_score"].to_numpy()) < 8] if len(pool) > per * 5 else pool
        picks += list(rng.choice(pool, min(per, len(pool)), replace=False)) if len(pool) else []
    normal = scored.index[scored["assessment"] == LABEL_NORMAL].to_numpy()
    picks += list(rng.choice(normal, max(0, n - len(picks)), replace=False))
    return [int(i) for i in picks]


def _write_results_md(cfg, cleaning, summary, comparison, ablation, rates, shap_groups, timings, total) -> None:
    keep = ["model", "MAE", "RMSE", "R2", "R2_log", "MdAPE_%", "within_10%", "within_20%"]
    parts = [
        "# Results summary (auto-generated by run_pipeline.py)\n",
        f"Data: HM Land Registry Price Paid {cfg.years[0]}-{cfg.years[-1]}, "
        f"region = {cfg.region or 'England & Wales'}, sample = {cfg.sample_frac or 'full data'}. "
        f"Split: train <= {cfg.train_end} < validation <= {cfg.val_end} < test.\n",
        "## Data cleaning\n", _md_table(cleaning[["step", "rows_before", "rows_removed", "rows_after"]]),
        "\n## Data split\n", _md_table(summary[["split", "sales", "from", "to", "median_price"]], "{:,.0f}"),
        "\n## Model comparison - test period\n",
        _md_table(comparison.loc[comparison["period"] == "test", keep]),
        "\n## Model comparison - validation period\n",
        _md_table(comparison.loc[comparison["period"] == "validation", keep]),
    ]
    if ablation is not None:
        parts += ["\n## Feature ablation (XGBoost, test period)\n",
                  _md_table(ablation[["feature_set", "n_features", "MAE", "R2_log", "MdAPE_%", "within_10%"]])]
    parts += ["\n## Anomaly detection\n", _md_table(rates)]
    if shap_groups is not None:
        parts += ["\n## SHAP importance by feature group\n", _md_table(shap_groups)]
    parts += ["\n## Run time (seconds)\n",
              _md_table(pd.DataFrame({"stage": list(timings), "seconds": list(timings.values())}), "{:,.1f}"),
              f"\nTotal: {total / 60:.1f} minutes\n"]
    (Path(cfg.output_dir) / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")

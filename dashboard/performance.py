"""Model Performance: Objective 3 (XGBoost vs Linear Regression) and Objective 2 (feature test)."""
import json

import pandas as pd
import streamlit as st

from dashboard.common import text_table, TABLES, callout, figure, gbp, page_header, table

METRIC_HELP = pd.DataFrame([
    ("MAE", "Mean absolute error", "Average size of the error in pounds. Lower is better."),
    ("RMSE", "Root mean squared error", "Like MAE but penalises large errors more heavily. Lower is better."),
    ("R²", "Coefficient of determination", "Share of the variation in sale prices the model explains (0-1). Higher is better."),
    ("MdAPE", "Median absolute percentage error", "Typical error as a % of the sale price; half of sales are closer than this. Lower is better."),
    ("Within 10% / 20%", "Hit rate", "Share of sales predicted within 10% / 20% of the actual price. Higher is better."),
], columns=["Metric", "Full name", "How to read it"])


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Model": df["model"], "MAE": df["MAE"].map(gbp), "RMSE": df["RMSE"].map(gbp), "R²": df["R2"].map("{:.3f}".format),
        "R² (log price)": df["R2_log"].map("{:.3f}".format), "MdAPE": df["MdAPE_%"].map("{:.1f}%".format),
        "Within 10%": df["within_10%"].map("{:.1f}%".format), "Within 20%": df["within_20%"].map("{:.1f}%".format),
    })


def render() -> None:
    page_header("Model Performance",
                "How accurate are the models on sales they have never seen? All results are for "
                "<b>January-September 2026</b>. The models were trained on 2022-2024 and tuned on 2025. "
                "Three models are compared: a <b>naive local average</b> (the average price of the same property "
                "type in the same postcode district), the <b>Linear Regression</b> baseline, and <b>XGBoost</b>, "
                "the main model.",
                "Objective 3 (XGBoost vs Linear Regression using MAE, RMSE and R²) and Objective 2 (do spatial "
                "and temporal features help?)")

    with st.expander("What do the metrics mean?"):
        text_table(METRIC_HELP)

    cmp_ = table("05_model_comparison.csv")
    if cmp_ is None:
        st.info("Run the pipeline to produce results.")
        return
    test = cmp_[cmp_["period"] == "test"].reset_index(drop=True)
    val = cmp_[cmp_["period"] == "validation"].reset_index(drop=True)
    t = test.set_index("model")
    xgb, lr = t.loc["XGBoost"], t.loc["Linear Regression"]

    st.header("Objective 3: XGBoost compared with Linear Regression")
    text_table(_fmt(test))
    st.markdown('<p class="caption-fig">Table 1: Accuracy on the 2026 test period '
                f'({int(xgb["n"]):,} sales).</p>', unsafe_allow_html=True)
    callout(f"<b>Finding.</b> XGBoost outperforms Linear Regression on every metric. It reduces the average "
            f"error (MAE) by <b>{100 * (1 - xgb['MAE'] / lr['MAE']):.1f}%</b> ({gbp(lr['MAE'])} → {gbp(xgb['MAE'])}) "
            f"and RMSE by <b>{100 * (1 - xgb['RMSE'] / lr['RMSE']):.1f}%</b>, and raises R² from {lr['R2']:.3f} to "
            f"<b>{xgb['R2']:.3f}</b>. The typical error falls from {lr['MdAPE_%']:.1f}% to {xgb['MdAPE_%']:.1f}%, and "
            f"{xgb['within_20%']:.0f}% of sales are predicted within 20%. Objective 3 is therefore met. The remaining "
            "error mainly reflects information the data does not contain (floor area, bedrooms, condition).")
    with st.expander("Validation-period (2025) results, used for tuning"):
        text_table(_fmt(val))
        st.write("Results on 2025 and 2026 are very similar, which suggests the model is not overfitted to one period.")

    figure("fig03_model_comparison.png", "Comparison of the three models on the 2026 test period.")
    c = st.columns(2)
    with c[0]:
        figure("fig04_predicted_vs_actual_xgb.png", "XGBoost: predicted vs actual sale price (each dot is a sale; the red line is a perfect prediction).")
    with c[1]:
        figure("fig05_predicted_vs_actual_linear.png", "Linear Regression: predicted vs actual sale price.")

    st.subheader("Accuracy by region and property type (XGBoost)")
    c = st.columns(2)
    for col, name, label in ((c[0], "07_xgb_metrics_by_region.csv", "Region"),
                             (c[1], "06_xgb_metrics_by_property_type.csv", "Property type")):
        d = table(name)
        if d is not None:
            d = d.sort_values("MdAPE_%")
            col.dataframe(pd.DataFrame({label: d["group"], "Sales": d["n"].map("{:,}".format),
                                        "MdAPE": d["MdAPE_%"].map("{:.1f}%".format), "MAE": d["MAE"].map(gbp),
                                        "Within 20%": d["within_20%"].map("{:.0f}%".format)}))
    reg = table("07_xgb_metrics_by_region.csv")
    if reg is not None:
        best, worst = reg.loc[reg["MdAPE_%"].idxmin()], reg.loc[reg["MdAPE_%"].idxmax()]
        st.write(f"Accuracy is best in **{best['group']}** ({best['MdAPE_%']:.1f}% typical error) and weakest in "
                 f"**{worst['group']}** ({worst['MdAPE_%']:.1f}%), where prices vary more within small areas.")

    st.header("Objective 2: do spatial and temporal features improve accuracy?")
    st.write("To test this, XGBoost was retrained four times, adding one group of features at a time (an "
             "*ablation study*). If a group matters, adding it should reduce the error.")
    ab = table("09_feature_ablation.csv")
    if ab is not None:
        text_table(pd.DataFrame({
            "Feature set": ab["feature_set"], "Features": ab["n_features"], "MAE": ab["MAE"].map(gbp),
            "R² (log)": ab["R2_log"].map("{:.3f}".format), "MdAPE": ab["MdAPE_%"].map("{:.1f}%".format),
            "Within 10%": ab["within_10%"].map("{:.1f}%".format)}))
        st.markdown('<p class="caption-fig">Table 2: Feature ablation on the 2026 test period.</p>',
                    unsafe_allow_html=True)
        a = ab.set_index("feature_set")["MdAPE_%"]
        callout(f"<b>Finding.</b> <b>Spatial features are decisive:</b> adding them cuts the typical error from "
                f"{a.iloc[0]:.1f}% to {a.iloc[2]:.1f}%. Temporal features on their own add nothing "
                f"({a.iloc[0]:.1f}% → {a.iloc[1]:.1f}%), and the full model ({a.iloc[3]:.1f}%) performs about the "
                "same as spatial-only. A likely reason is that national prices were broadly flat over 2022-2026 "
                "(median about £290,000 throughout), and tree models cannot extrapolate a time trend into a future "
                "test period. The hypothesis is therefore <b>supported for spatial features but not for temporal "
                "features</b> in this period.")
        figure("fig08_feature_ablation.png", "Effect of adding feature groups on XGBoost accuracy.")

    p = TABLES / "04_xgb_best_params.json"
    if p.exists():
        with st.expander("Model settings (hyper-parameter tuning)"):
            tun = table("04_xgb_tuning_results.csv")
            rng = (f" (validation RMSE on log price {tun['val_rmse_log'].min():.4f}-{tun['val_rmse_log'].max():.4f})"
                   if tun is not None else "")
            st.write(f"{len(tun) if tun is not None else 'Several'} candidate settings were compared on the 2025 "
                     f"validation period by random search. Differences were small{rng}, which indicates that accuracy "
                     "is limited by the available information rather than by model tuning. Chosen settings:")
            st.json(json.loads(p.read_text()))

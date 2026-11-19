"""Explainability: Objective 5 (SHAP)."""
import altair as alt
import pandas as pd
import streamlit as st

from dashboard.common import callout, explanation_chart, figure, gbp, la_names, page_header, table


def render() -> None:
    page_header("Explainability (SHAP)",
                "A price estimate is more useful, and easier to challenge, when the reasons are visible. "
                "<b>SHAP</b> (SHapley Additive exPlanations; Lundberg & Lee, 2017) divides each prediction into "
                "contributions from each input. The contributions add up exactly to the model's output, so the "
                "explanation is faithful to what the model actually did.",
                "Objective 5: SHAP explanations showing the main factors influencing price predictions")

    with st.expander("How SHAP works, in brief"):
        st.markdown(
            "SHAP comes from cooperative game theory: each feature is a 'player' and the prediction is the 'payout'. "
            "A feature's SHAP value is its average contribution across all possible combinations of features. For "
            "tree models such as XGBoost these values can be computed **exactly** (TreeSHAP). Because the model "
            "predicts the *logarithm* of price, each contribution multiplies the price, so contributions are "
            "shown here as **percentage effects** on the estimate.")

    st.header("What drives the model overall?")
    groups = table("16_shap_importance_by_feature_group.csv")
    imp = table("15_shap_global_importance.csv")
    c = st.columns(2)
    if groups is not None:
        with c[0]:
            st.altair_chart(alt.Chart(groups).mark_bar(color="#1f3a5f").encode(
                x=alt.X("share_%:Q", title="Share of total influence (%)"), y=alt.Y("group:N", sort="-x", title=None),
                tooltip=["group", alt.Tooltip("share_%:Q", format=".1f")]).properties(height=200, title="By feature group"),
                width="stretch")
    if imp is not None:
        with c[1]:
            top = imp.head(12)
            st.altair_chart(alt.Chart(top).mark_bar(color="#2b6cb0").encode(
                x=alt.X("share_%:Q", title="Share of total influence (%)"), y=alt.Y("label:N", sort="-x", title=None,
                                                                                   axis=alt.Axis(labelLimit=300)),
                tooltip=["label", alt.Tooltip("share_%:Q", format=".1f")]).properties(height=360, title="Top 12 features"),
                width="stretch")
    st.markdown('<p class="caption-fig">Figure: mean absolute SHAP value as a share of the total, '
                '5,000 randomly selected 2026 sales.</p>', unsafe_allow_html=True)
    if groups is not None:
        g = groups.set_index("group")["share_%"]
        callout(f"<b>Finding.</b> Location dominates: spatial features account for <b>{g.get('Spatial', 0):.0f}%</b> "
                f"of the model's decisions. The most influential is the average price of earlier sales of the same "
                f"property type in the same postcode district. The UK House Price Index contributes "
                f"{g.get('Market index (UK HPI)', 0):.0f}%, property details (type, tenure, new build) "
                f"{g.get('Property', 0):.0f}%, and the date of sale only {g.get('Temporal', 0):.0f}%. This matches the "
                "ablation result on the Model Performance page.")

    figure("fig12_shap_beeswarm.png", "SHAP summary: each dot is one sale. Its position shows how much the "
           "feature moved that prediction (right = higher price); its colour shows the feature's value (red = high, "
           "blue = low). For example, red dots on the right for local price levels mean that more expensive areas "
           "raise the estimate.", width=820)

    st.header("Worked case studies")
    st.write("Six 2026 sales (two unusually high, two unusually low, two normal) were selected at random.")
    cases = table("17_case_study_explanations.csv")
    if cases is None:
        return
    names = la_names()
    ids = sorted(cases["case"].unique())
    first = cases.groupby("case").first()
    case = st.radio("Case", ids, horizontal=True,
                    format_func=lambda i: f"Case {i}: {first.loc[i, 'assessment']}")
    d = cases[cases["case"] == case].copy()
    d["value"] = [names.get(v, v) if f == "Local authority" else v for f, v in zip(d["factor"], d["value"])]
    r = d.iloc[0]
    m = st.columns(4)
    m[0].metric("Actual price", gbp(r["actual_price"]))
    m[1].metric("Predicted price", gbp(r["predicted_price"]))
    m[2].metric("Difference", f"{100 * (r['actual_price'] / r['predicted_price'] - 1):+.0f}%")
    m[3].metric("Assessment", r["assessment"], f"score {r['anomaly_score']:+.1f}", delta_color="off")
    expl = pd.DataFrame({"feature": "", "label": d["factor"], "value": d["value"], "effect_%": d["effect_%"]})
    st.altair_chart(explanation_chart(expl), width="stretch")
    st.caption("Bars show the factors behind the predicted price. A large gap between the actual and predicted "
               "price is what the anomaly score measures.")

"""Prototype decision-support app (the dissertation deliverable).

    streamlit run app.py

Tabs
  1. Value a property      - predicted price, typical range, anomaly check, SHAP reasons
  2. Unusual transactions  - test-period sales flagged for review (no addresses shown)
  3. Model performance     - model comparison, ablation, charts
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from propval.config import PROPERTY_TYPE_NAMES  # noqa: E402
from propval.predict import PropertyValuer  # noqa: E402

OUT = ROOT / "outputs"
st.set_page_config(page_title="Property Valuation & Anomaly Check", layout="wide")


@st.cache_resource(show_spinner="Loading model and postcode directory...")
def load_valuer() -> PropertyValuer:
    return PropertyValuer.load(OUT / "models")


@st.cache_data
def load_table(name: str) -> pd.DataFrame | None:
    path = OUT / "tables" / name
    return pd.read_csv(path) if path.exists() else None


st.title("Explainable AI property valuation and price anomaly check")
st.caption("England & Wales, trained on HM Land Registry Price Paid Data. Decision support only: an "
           "'unusual' result means the price differs strongly from similar sales and may need human "
           "review. It does not mean fraud or an incorrect price.")

if not (OUT / "models" / "valuer_bundle.joblib").exists():
    st.error("No trained model found. Run `python run_pipeline.py` first.")
    st.stop()

valuer = load_valuer()
tab1, tab2, tab3 = st.tabs(["Value a property", "Unusual transactions", "Model performance"])

# --------------------------------------------------------------------- tab 1
with tab1:
    with st.form("valuation"):
        c1, c2, c3 = st.columns(3)
        postcode = c1.text_input("Postcode", "S1 2BJ")
        ptype = c1.selectbox("Property type", list(PROPERTY_TYPE_NAMES), format_func=PROPERTY_TYPE_NAMES.get)
        tenure = c2.selectbox("Tenure", ["F", "L"], format_func={"F": "Freehold", "L": "Leasehold"}.get)
        new_build = c2.checkbox("New build")
        sale_date = c3.date_input("Sale date", date(2026, 6, 1), min_value=date(2022, 1, 1))
        actual = c3.number_input("Actual sale price (£, optional)", min_value=0, step=5000, value=0)
        submitted = st.form_submit_button("Value property", type="primary")

    if submitted:
        try:
            r = valuer.value(postcode, ptype, new_build, tenure, pd.Timestamp(sale_date), actual or None)
        except ValueError as exc:
            st.error(str(exc))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted price", f"£{r['predicted_price']:,.0f}")
            m2.metric("Typical range (80%)", f"£{r['range_low']:,.0f} - £{r['range_high']:,.0f}")
            if "actual_price" in r:
                m3.metric("Actual vs predicted", f"{r['difference_%']:+.1f}%",
                          f"{r['assessment']} (score {r['anomaly_score']:+.1f})",
                          delta_color="off" if r["assessment"] == "Normal" else "inverse")
                if r["assessment"] != "Normal":
                    st.warning(f"**{r['assessment']}**: this price is far from what similar sales suggest. "
                               "Check the details (e.g. condition, size, sale circumstances) before relying on it.")
                else:
                    st.success("The price is within the normal range for similar sales.")
            st.subheader("Why this prediction? (SHAP)")
            st.caption(f"Starting point: an average sale (£{r['baseline_price']:,.0f}). Each factor moves the "
                       "prediction up or down by the percentage shown.")
            expl = r["explanation"][["label", "value", "effect_%"]].rename(
                columns={"label": "Factor", "value": "Value for this property", "effect_%": "Effect on price (%)"})
            st.bar_chart(expl.set_index("Factor")["Effect on price (%)"], horizontal=True)
            st.dataframe(expl.style.format({"Effect on price (%)": "{:+.1f}%"}), hide_index=True,
                         use_container_width=True)

# --------------------------------------------------------------------- tab 2
with tab2:
    flagged = load_table("11_unusual_transactions_test.csv")
    if flagged is None:
        st.info("Run the pipeline to generate the list of unusual transactions.")
    else:
        st.write(f"{len(flagged):,} test-period sales flagged for review. Addresses are not shown; "
                 "only the postcode district is given.")
        c1, c2, c3 = st.columns(3)
        regions = c1.multiselect("Region", sorted(flagged["region"].dropna().unique()))
        types = c2.multiselect("Property type", sorted(flagged["property_type"].unique()))
        direction = c3.multiselect("Assessment", ["Unusually high", "Unusually low"])
        view = flagged
        if regions:
            view = view[view["region"].isin(regions)]
        if types:
            view = view[view["property_type"].isin(types)]
        if direction:
            view = view[view["assessment"].isin(direction)]
        st.dataframe(view.head(2000).style.format({
            "actual_price": "£{:,.0f}", "predicted_price": "£{:,.0f}",
            "difference_%": "{:+.0f}%", "anomaly_score": "{:+.1f}"}), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------- tab 3
with tab3:
    cmp_ = load_table("05_model_comparison.csv")
    if cmp_ is not None:
        st.subheader("Model comparison (test period)")
        st.dataframe(cmp_[cmp_["period"] == "test"].drop(columns="period").style.format(precision=3, thousands=","),
                     hide_index=True, use_container_width=True)
    abl = load_table("09_feature_ablation.csv")
    if abl is not None:
        st.subheader("Do spatial and temporal features help? (feature ablation)")
        st.dataframe(abl.style.format(precision=3, thousands=","), hide_index=True, use_container_width=True)
    figs = sorted((OUT / "figures").glob("fig*.png"))
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]):
            col.image(str(f), caption=f.stem, use_container_width=True)

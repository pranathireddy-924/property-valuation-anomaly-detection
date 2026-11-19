"""Unusual Transactions: Objective 4 (price anomaly score)."""
import pandas as pd
import streamlit as st

from dashboard.common import (text_table, app_data_ready, callout, explanation_chart, explanation_table, figure, flagged_rows,
                              gbp, la_names, page_header, table, valuer)
from propval.config import PROPERTY_TYPE_NAMES


@st.cache_data(show_spinner=False)
def _flagged() -> pd.DataFrame | None:
    f = table("11_unusual_transactions_test.csv")
    if f is None:
        return None
    if app_data_ready():
        lad = flagged_rows()[["transaction_id", "lad"]]
        f = f.merge(lad, on="transaction_id", how="left")
        f["local_authority"] = f["lad"].map(la_names()).fillna(f["lad"])
    else:
        f["local_authority"] = ""
    return f


def render() -> None:
    page_header("Unusual Transactions",
                "A sale is flagged as <b>unusual</b> when its actual price is much higher or lower than the price "
                "the model expects for that kind of property, in that place, at that time. Flagged sales are "
                "candidates for <b>human review</b>. They are not judged to be wrong or fraudulent.",
                "Objective 4: a price anomaly score showing transactions whose actual price is unusually "
                "different from the predicted price")

    st.header("How the anomaly score works")
    c = st.columns([3, 2])
    with c[0]:
        st.markdown(
            "1. For each sale, compare the **actual price** with the **predicted price**: "
            "*residual = log(actual) - log(predicted)*.\n"
            "2. Standardise the residual against what is normal for that property type, using the **median** "
            "and **median absolute deviation (MAD)** of 2025 validation sales. These robust statistics are not "
            "distorted by the very outliers being searched for.\n"
            "3. *Anomaly score = (residual - median) / (1.4826 × MAD)*. It reads like a z-score: 0 is a typical "
            "sale, ±1 a common difference, and beyond **±3** is unusual.\n"
            "4. Scores above +3 are labelled **unusually high**, below -3 **unusually low**.")
    th = table("10_anomaly_thresholds_by_type.csv")
    if th is not None:
        with c[1]:
            st.write("**What ±3 means in practice** (a sale is flagged if its price is):")
            text_table(pd.DataFrame({
                "Property type": th["group"].map(PROPERTY_TYPE_NAMES).fillna(th["group"]),
                "Above the estimate by": th["flag_if_price_above_%"].map("more than {:.0f}%".format),
                "Below the estimate by": (-th["flag_if_price_below_%"]).map("more than {:.0f}%".format)}))

    st.header("Does the score identify genuinely unusual sales?")
    rates = table("14_anomaly_check_category_b.csv")
    if rates is not None:
        r = rates.set_index("group")
        a, b = r.iloc[0], r.iloc[1]
        m = st.columns(3)
        m[0].metric("Standard sales flagged", f"{a['flag_rate_%']:.1f}%", help=f"{int(a['sales']):,} sales in 2026")
        m[1].metric("Non-standard sales flagged", f"{b['flag_rate_%']:.1f}%", help=f"{int(b['sales']):,} sales in 2026")
        m[2].metric("Ratio", f"{b['flag_rate_%'] / a['flag_rate_%']:.1f}×")
        callout("There is no official list of 'wrong' prices to test against, so two checks were used. "
                f"<b>(1) An external check on real data:</b> Land Registry category B sales (repossessions, "
                f"buy-to-let and company transfers) were never shown to the model, and they often happen at "
                f"non-market prices. They are flagged <b>{b['flag_rate_%'] / a['flag_rate_%']:.1f} times as often</b> "
                f"as standard sales ({b['flag_rate_%']:.1f}% vs {a['flag_rate_%']:.1f}%). <b>(2) A controlled "
                "test:</b> in the automated tests, 20 deliberately distorted prices (×4 or ÷4) were planted in "
                "synthetic data. The score detected at least 90% of them with under 3% false alarms, in the "
                "correct direction.")
    c = st.columns(2)
    with c[0]:
        figure("fig10_anomaly_rate_category_a_vs_b.png", "Share of 2026 sales flagged: standard vs non-standard sales.")
    with c[1]:
        figure("fig09_anomaly_score_distribution.png", "Distribution of anomaly scores for 2026 sales; dashed lines mark ±3.")

    st.header("Flagged 2026 sales")
    f = _flagged()
    if f is None:
        st.info("Run the pipeline to generate flagged sales.")
        return
    st.write(f"**{len(f):,}** of the 2026 test sales were flagged. Only the postcode district is shown; full "
             "addresses are never displayed. Use the filters, then select a sale below to see why it was flagged.")
    c = st.columns(4)
    regions = c[0].multiselect("Region", sorted(f["region"].dropna().unique()))
    view = f[f["region"].isin(regions)] if regions else f
    las = c[1].multiselect("Local authority", sorted(view["local_authority"].dropna().unique()))
    types = c[2].multiselect("Property type", sorted(f["property_type"].unique()))
    direction = c[3].multiselect("Assessment", ["Unusually high", "Unusually low"])
    if las:
        view = view[view["local_authority"].isin(las)]
    if types:
        view = view[view["property_type"].isin(types)]
    if direction:
        view = view[view["assessment"].isin(direction)]

    show = pd.DataFrame({
        "Date": view["date"], "District": view["postcode_district"], "Local authority": view["local_authority"],
        "Region": view["region"], "Type": view["property_type"], "Tenure": view["tenure"],
        "New build": view["new_build"].map({True: "Yes", False: "No"}),
        "Actual price": view["actual_price"].map(gbp), "Predicted price": view["predicted_price"].map(gbp),
        "Difference": view["difference_%"].map("{:+.0f}%".format), "Score": view["anomaly_score"].map("{:+.1f}".format),
        "Assessment": view["assessment"]})
    st.caption(f"{len(view):,} sales match the filters (showing up to 1,000, most unusual first).")
    st.dataframe(show.head(1000), hide_index=True, width="stretch", height=330)

    st.subheader("Explain a flagged sale")
    if not app_data_ready():
        st.info("Explanations of individual sales are available after the pipeline has been run (Run Pipeline page).")
        return
    options = view.head(500)
    if options.empty:
        return
    labels = {tid: f"{d} · {dist} · {lt} · {t} · actual {gbp(a)} vs predicted {gbp(p)} ({s:+.1f})"
              for tid, d, dist, lt, t, a, p, s in zip(options["transaction_id"], options["date"],
                                                      options["postcode_district"], options["local_authority"],
                                                      options["property_type"], options["actual_price"],
                                                      options["predicted_price"], options["anomaly_score"])}
    tid = st.selectbox("Select a sale", list(labels), format_func=labels.get)
    row = flagged_rows()
    row = row[row["transaction_id"] == tid].reset_index(drop=True)
    if row.empty:
        return
    res = valuer().evaluate_row(row, float(row.loc[0, "price"]))
    st.write(f"The model expected about **{gbp(res['predicted_price'])}** "
             f"(typical range {gbp(res['range_low'])} - {gbp(res['range_high'])}). The recorded price was "
             f"**{gbp(res['actual_price'])}** ({res['difference_%']:+.0f}%), giving a score of "
             f"**{res['anomaly_score']:+.1f}**. The factors behind the *expected* price were:")
    st.altair_chart(explanation_chart(res["explanation"]), width="stretch")
    text_table(explanation_table(res["explanation"], res["property_type"]))

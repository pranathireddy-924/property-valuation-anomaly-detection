"""Valuation Tool: the working prototype (estimate, anomaly check, explanation)."""
import re
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.common import (text_table, TENURE_NAMES, TYPE_FROM_NAME, callout, district_prices, explanation_chart,
                              explanation_table, gbp, la_name, page_header, postcodes, table, valuer)
from propval.config import PROPERTY_TYPE_NAMES

EXAMPLE_LAS = [  # (local authority, property type, tenure, description)
    ("Sheffield", "S", "F", "Semi-detached house in Sheffield"),
    ("Westminster", "F", "L", "Leasehold flat in central London"),
    ("Manchester", "T", "F", "Terraced house in Manchester"),
    ("Cardiff", "D", "F", "Detached house in Cardiff"),
    ("Cornwall", "D", "F", "Detached house in rural Cornwall"),
]
DATA_END = date(2026, 9, 30)


def _natural_key(district: str):
    """Sort S2 before S10 (not alphabetically)."""
    m = re.match(r"([A-Z]+)(\d+)([A-Z]?)", district or "")
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (district, 0, "")


@st.cache_data(show_spinner=False)
def _examples() -> list[dict]:
    pcs = postcodes()
    out = []
    for la, ptype, tenure, desc in EXAMPLE_LAS:
        sub = pcs[pcs["la_name"] == la]
        if len(sub):
            top = sub.sort_values("n_sales", ascending=False).iloc[0]
            out.append({"postcode": top["postcode"], "ptype": ptype, "tenure": tenure, "desc": desc,
                        "region": top["region_name"], "la": la, "district": top["pc_district"]})
    return out


def _set_location(pc_row) -> None:
    st.session_state["loc_mode"] = "Browse by area"
    st.session_state["sel_region"] = pc_row["region_name"]
    st.session_state["sel_la"] = pc_row["la_name"]
    st.session_state["sel_district"] = pc_row["pc_district"]
    st.session_state["sel_pc"] = pc_row["postcode"]


def _random_postcode() -> None:
    pcs = postcodes()
    row = pcs.sample(1, weights=pcs["n_sales"], random_state=np.random.randint(1_000_000)).iloc[0]
    _set_location(row)
    st.session_state.pop("result", None)


def _load_example(ex: dict) -> None:
    pcs = postcodes()
    _set_location(pcs[pcs["postcode"] == ex["postcode"]].iloc[0])
    st.session_state["ptype"] = PROPERTY_TYPE_NAMES[ex["ptype"]]
    st.session_state["tenure"] = TENURE_NAMES[ex["tenure"]]
    st.session_state.pop("result", None)


def _choose_location() -> str | None:
    pcs = postcodes()
    mode = st.radio("How would you like to choose the location?", ["Browse by area", "Type a postcode"],
                    horizontal=True, key="loc_mode")
    if mode == "Type a postcode":
        pc = st.text_input("Postcode (England or Wales)", key="pc_text", placeholder="e.g. S10 2TN",
                           help="Any full postcode in England or Wales, e.g. S10 2TN, M14 5RQ, CF14 3NA. "
                                "Spaces and capital letters do not matter.")
        return pc.strip() or None

    c = st.columns(4)
    regions = sorted(pcs["region_name"].unique())
    region = c[0].selectbox("1. Region", regions, key="sel_region")
    sub = pcs[pcs["region_name"] == region]
    la = c[1].selectbox("2. Local authority", sorted(sub["la_name"].unique()), key="sel_la")
    sub = sub[sub["la_name"] == la]
    districts = sorted(sub["pc_district"].unique(), key=_natural_key)
    district = c[2].selectbox("3. Postcode district", districts, key="sel_district",
                              help="The first half of a postcode, covering a group of streets or a small town.")
    sub = sub[sub["pc_district"] == district].sort_values("n_sales", ascending=False)
    labels = dict(zip(sub["postcode"], [f"{p}  ({n} recorded sale{'s' if n != 1 else ''})"
                                        for p, n in zip(sub["postcode"], sub["n_sales"])]))
    return c[3].selectbox("4. Postcode", list(labels), format_func=labels.get, key="sel_pc",
                          help="Only postcodes with at least one recorded sale 2022-2026 are listed.")


def _score_chart(z: float, threshold: float):
    zc = float(np.clip(z, -8, 8))
    bands = pd.DataFrame([
        {"from": -8, "to": -threshold, "zone": "Unusually low"},
        {"from": -threshold, "to": threshold, "zone": "Normal range"},
        {"from": threshold, "to": 8, "zone": "Unusually high"}])
    base = alt.Chart(bands).mark_rect(opacity=0.35).encode(
        x=alt.X("from:Q", scale=alt.Scale(domain=[-8, 8]), title="Anomaly score (robust z)"), x2="to:Q",
        color=alt.Color("zone:N", scale=alt.Scale(domain=["Unusually low", "Normal range", "Unusually high"],
                                                  range=["#c53030", "#a0aec0", "#c53030"]), legend=None))
    marker = alt.Chart(pd.DataFrame({"z": [zc]})).mark_rule(color="#1f3a5f", strokeWidth=4).encode(x="z:Q")
    label = alt.Chart(pd.DataFrame({"z": [zc], "t": [f"This sale: {z:+.1f}"]})).mark_text(
        dy=-28, color="#1f3a5f", fontWeight="bold").encode(x="z:Q", text="t:N")
    return (base + marker + label).properties(height=70)


def _local_context(result: dict) -> None:
    dp = district_prices()
    d = dp[(dp["pc_district"] == result["pc_district"]) & (dp["property_type"] == result["property_type_code"])]
    st.subheader("Recorded sales nearby")
    if d.empty:
        st.write(f"No standard sales of this property type were recorded in district {result['pc_district']} in 2022-2026.")
        return
    d = d.sort_values("sale_year")
    st.write(f"Median price of **{result['property_type'].lower()}** sales recorded in postcode district "
             f"**{result['pc_district']}** by year (HM Land Registry). These are actual sale prices, shown for context:")
    t = pd.DataFrame({"Year": d["sale_year"].astype(str), "Median price": d["median_price"].map(gbp),
                      "Number of sales": d["n_sales"]})
    text_table(t.set_index("Year").T.reset_index(names=""))


def _flag_rate_text() -> str:
    rates = table("14_anomaly_check_category_b.csv")
    if rates is None:
        return ""
    return f"only {rates.iloc[0]['flag_rate_%']:.1f}% of standard 2026 sales differed this much from the model."


def _show_result(r: dict, threshold: float) -> None:
    st.divider()
    st.header("Result")
    st.markdown(f"**{r['postcode']}** · postcode district **{r['pc_district']}** · **{la_name(r['lad'])}** · "
                f"{r['region']} · {r['property_type']}")
    m = st.columns(4)
    m[0].metric("Estimated sale price", gbp(r["predicted_price"]))
    rng_help = "Typical range (80%): 8 in 10 comparable 2025 sales fell within this range of the model's estimate."
    m[1].metric("Typical range: lower", gbp(r["range_low"]), help=rng_help)
    m[2].metric("Typical range: upper", gbp(r["range_high"]), help=rng_help)
    m[3].metric("Average sale (starting point)", gbp(r["baseline_price"]),
                help="The model's average prediction before any property-specific factors are considered.")
    st.write(f"The model expects a property like this to sell for about **{gbp(r['predicted_price'])}**. Because "
             "the data contains no size, bedroom or condition information, individual homes can reasonably sell "
             f"anywhere from **{gbp(r['range_low'])}** to **{gbp(r['range_high'])}**.")

    if "actual_price" in r:
        st.subheader("Anomaly check")
        diff, z = r["difference_%"], r["anomaly_score"]
        a = st.columns(3)
        a[0].metric("Actual price", gbp(r["actual_price"]))
        a[1].metric("Difference from estimate", f"{diff:+.1f}%")
        a[2].metric("Assessment", r["assessment"], f"score {z:+.1f}", delta_color="off")
        st.altair_chart(_score_chart(z, threshold), width="stretch")
        if r["assessment"] == "Normal":
            callout(f"The actual price is {abs(diff):.0f}% {'above' if diff > 0 else 'below'} the estimate. Its "
                    f"anomaly score of {z:+.1f} is inside the normal range (±{threshold:g}), so the price is "
                    "consistent with comparable sales.")
        else:
            callout(f"The actual price is {abs(diff):.0f}% {'above' if diff > 0 else 'below'} the estimate. Its "
                    f"anomaly score of {z:+.1f} is beyond ±{threshold:g}: {_flag_rate_text()} "
                    "<b>Suggested action:</b> review the transaction, e.g. property size and "
                    "condition, extensions, land, or special sale circumstances. This is not evidence of "
                    "wrongdoing.", warn=True)

    st.subheader("Why the model gave this estimate")
    st.write("Each bar shows how much one factor moved the estimate away from the average sale "
             f"({gbp(r['baseline_price'])}). Green raises the estimate, red lowers it. Values are SHAP "
             "contributions converted to percentages.")
    st.altair_chart(explanation_chart(r["explanation"]), width="stretch")
    text_table(explanation_table(r["explanation"], r["property_type"]))
    _local_context(r)


def render() -> None:
    page_header("Valuation Tool",
                "Try the prototype. Choose a location and describe a property: the model estimates its "
                "expected sale price, explains the estimate and, if you enter an actual or proposed price, "
                "checks whether that price is unusual for comparable sales. Every input can be changed, so "
                "you can compare areas, property types and prices.",
                "the project deliverable; Objectives 3, 4 and 5 applied to a single property")

    with st.expander("Not sure where to start? Load an example", expanded=False):
        cols = st.columns(len(_examples()) or 1)
        for col, ex in zip(cols, _examples()):
            col.button(ex["desc"], on_click=_load_example, args=(ex,), width="stretch",
                       help=f"{ex['postcode']} ({ex['la']})")

    st.subheader("Step 1. Location")
    st.button("🎲 Pick a random postcode", on_click=_random_postcode,
              help="Chooses a real postcode with recorded sales anywhere in England & Wales")
    pc = _choose_location()

    st.subheader("Step 2. Property details")
    c = st.columns(4)
    ptype_name = c[0].selectbox("Property type", list(PROPERTY_TYPE_NAMES.values()), key="ptype")
    tenure_name = c[1].selectbox("Tenure", list(TENURE_NAMES.values()), key="tenure",
                                 help="Most houses are freehold; most flats are leasehold.")
    sale_date = c[2].date_input("Date of sale (or valuation)", date(2026, 6, 1), min_value=date(2022, 1, 1),
                                max_value=date(2027, 12, 31), format="DD/MM/YYYY")
    new_build = c[3].checkbox("New build", help="Tick if the property is being sold for the first time.")
    actual = st.number_input("Step 3 (optional). Actual or proposed sale price, £ - enter a price to check "
                             "whether it is unusual; leave at 0 for an estimate only", min_value=0,
                             max_value=20_000_000, step=5_000, value=0)
    if sale_date > DATA_END:
        st.caption(f"Note: the data runs to {DATA_END:%B %Y}. For later dates the latest market index is used, "
                   "so treat the estimate as being at current market levels.")

    inputs = (pc, ptype_name, tenure_name, sale_date, new_build, actual)
    if st.button("Estimate price", type="primary", disabled=not pc):
        try:
            with st.spinner("Calculating..."):
                st.session_state["result"] = valuer().value(
                    pc, TYPE_FROM_NAME[ptype_name], new_build,
                    {v: k for k, v in TENURE_NAMES.items()}[tenure_name], pd.Timestamp(sale_date), actual or None)
                st.session_state["result_inputs"] = inputs
        except ValueError as exc:
            st.session_state.pop("result", None)
            st.error(f"{exc}. Please check the postcode, or use 'Browse by area' to pick one from the list.")

    if "result" in st.session_state:
        if st.session_state.get("result_inputs") == inputs:
            _show_result(st.session_state["result"], valuer().detector.threshold)
        else:
            st.info("The inputs have changed. Click **Estimate price** to update the result.")

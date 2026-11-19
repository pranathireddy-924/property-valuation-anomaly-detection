"""Overview: what the project is, why it matters, aims, objectives and headline results."""
import pandas as pd
import streamlit as st

from dashboard.common import text_table, AUTHOR, COURSE, SUPERVISOR, TITLE, callout, gbp, table


def _results():
    cmp_ = table("05_model_comparison.csv")
    test = cmp_[cmp_["period"] == "test"].set_index("model") if cmp_ is not None else None
    return {
        "clean": table("01_cleaning_report.csv"), "split": table("02_data_summary.csv"), "test": test,
        "ablation": table("09_feature_ablation.csv"), "rates": table("14_anomaly_check_category_b.csv"),
        "groups": table("16_shap_importance_by_feature_group.csv"),
    }


def render() -> None:
    st.title(TITLE)
    st.markdown(f'<p class="subtitle">{AUTHOR} · {COURSE} · {SUPERVISOR}</p>', unsafe_allow_html=True)

    callout("This dashboard presents the research prototype and results of an MSc dissertation. It estimates "
            "what a home in England or Wales would be expected to sell for, compares that estimate with the "
            "actual sale price to <b>flag unusual transactions for human review</b>, and explains every estimate "
            "with <b>SHAP</b> so the reasons are transparent. It uses official open government data: HM Land "
            "Registry Price Paid Data, the ONS Postcode Directory and the UK House Price Index.")

    r = _results()
    if r["test"] is not None and r["split"] is not None:
        xgb, lr = r["test"].loc["XGBoost"], r["test"].loc["Linear Regression"]
        n_sales = int(r["split"]["sales"].sum())
        flag = r["rates"].iloc[0]["flag_rate_%"] if r["rates"] is not None else float("nan")
        c = st.columns(4)
        c[0].metric("Sales analysed", f"{n_sales/1e6:.2f} million", help="Standard residential sales, Jan 2022 - Sep 2026, after cleaning")
        c[1].metric("Typical error (XGBoost)", f"{xgb['MdAPE_%']:.1f}%",
                    f"{xgb['MdAPE_%'] - lr['MdAPE_%']:+.1f} pts vs Linear Regression", delta_color="inverse",
                    help="Median absolute percentage error on 2026 sales the model never saw")
        c[2].metric("R² (XGBoost)", f"{xgb['R2']:.3f}", f"{xgb['R2'] - lr['R2']:+.3f} vs Linear Regression",
                    help="Share of price variation explained, on the 2026 test sales")
        c[3].metric("Sales flagged as unusual", f"{flag:.1f}%", help="2026 test sales with |anomaly score| > 3")

    st.header("1. The problem")
    st.markdown(
        "House prices in England and Wales vary with **location, property type, tenure and time of sale**. "
        "It is therefore hard for a business to judge whether a given transaction price is *normal* for "
        "that kind of property in that place and at that time, or whether it is *unusual* and deserves a "
        "closer look. Many simple price models give a single number without saying why, and they do not "
        "separate the question *'what should it sell for?'* from *'is this sale unusual?'*.")

    st.header("2. Research question and aim")
    st.markdown('<p class="rq"><b>Research question.</b> How accurately can an explainable AI model estimate '
                'property transaction prices and identify unusual sale prices using spatial and temporal '
                'housing market data?</p>', unsafe_allow_html=True)
    st.markdown("**Aim.** To develop an explainable AI-based property valuation and price anomaly detection "
                "model using official housing transaction data from England and Wales.")

    st.header("3. Objectives and what was achieved")
    rows = []
    if r["clean"] is not None:
        raw = int(r["clean"].iloc[0]["rows_after"])
        final = int(r["clean"].iloc[-1]["rows_after"])
        rows.append(("1. A cleaned, reusable transaction dataset for selected years and regions",
                     f"{raw:,} raw records → {final:,} cleaned and linked to geography (2022-2026); any year range or region can be selected.",
                     "Data & Method"))
    if r["ablation"] is not None:
        a = r["ablation"].set_index("feature_set")["MdAPE_%"]
        rows.append(("2. Spatial and temporal features (postcode area, type, sale month/year, local price patterns)",
                     f"23 features built. Adding spatial features cut the typical error from {a.iloc[0]:.1f}% to {a.iloc[2]:.1f}%; "
                     f"temporal and index features added little on top ({a.iloc[3]:.1f}%).",
                     "Model Performance"))
    if r["test"] is not None:
        rows.append(("3. XGBoost outperforms a Linear Regression baseline (MAE, RMSE, R²)",
                     f"Achieved on unseen 2026 sales: MAE {gbp(xgb['MAE'])} vs {gbp(lr['MAE'])}, "
                     f"RMSE {gbp(xgb['RMSE'])} vs {gbp(lr['RMSE'])}, R² {xgb['R2']:.3f} vs {lr['R2']:.3f}.",
                     "Model Performance"))
    if r["rates"] is not None:
        ra = r["rates"].set_index("group")["flag_rate_%"]
        rows.append(("4. A price anomaly score showing unusually high or low sales",
                     f"{ra.iloc[0]:.1f}% of standard sales flagged; non-standard sales (repossessions, company "
                     f"transfers) flagged {ra.iloc[1]:.1f}%, about {ra.iloc[1]/ra.iloc[0]:.1f}× as often, which supports the score's validity.",
                     "Unusual Transactions"))
    if r["groups"] is not None:
        g = r["groups"].set_index("group")["share_%"]
        rows.append(("5. SHAP explanations of the main factors behind predictions",
                     f"Every estimate is explained. Location explains {g.get('Spatial', 0):.0f}% of the model's "
                     f"decisions, the House Price Index {g.get('Market index (UK HPI)', 0):.0f}%, property details "
                     f"{g.get('Property', 0):.0f}%.",
                     "Explainability"))
    text_table(pd.DataFrame({"Objective": [x[0] for x in rows], "Outcome": [x[1] for x in rows], "See page": [x[2] for x in rows]}))
    st.caption("Professional evaluation (a short questionnaire with 5-8 property professionals) is in progress; "
               "results will appear on the Ethics & Evaluation page.")

    st.header("4. Who benefits and how")
    b = st.columns(3)
    with b[0].container(border=True):
        st.subheader("PropTech firms")
        st.write("A transparent, data-driven estimate of expected sale price that can be built into "
                 "automated valuation products, with the reasons attached.")
    with b[1].container(border=True):
        st.subheader("Estate agents")
        st.write("A quick check of whether an agreed or asking price is in line with comparable local "
                 "sales, and an explanation they can share with clients.")
    with b[2].container(border=True):
        st.subheader("Mortgage & valuation teams")
        st.write("A screening tool that highlights unusually high or low transactions so that professional "
                 "valuers can prioritise their manual review.")
    callout("<b>Decision support, not a decision.</b> An 'unusual' flag means the price differs strongly from "
            "similar sales. It does <b>not</b> indicate fraud or an incorrect price. Property condition, size and "
            "sale circumstances are not in the data and must be checked by a professional.", warn=True)

    st.header("5. How to use this dashboard")
    st.markdown(
        "| Page | What you will find |\n|---|---|\n"
        "| **Data & Method** | The datasets, how they were cleaned, the research design and the meaning of postcode and area codes |\n"
        "| **Valuation Tool** | Try the model: choose any area or postcode and property details, get an estimate, an anomaly check and the reasons |\n"
        "| **Model Performance** | Objective 3: XGBoost vs Linear Regression, plus the feature test for Objective 2 |\n"
        "| **Unusual Transactions** | Objective 4: how sales are flagged, the evidence the method works, and every flagged 2026 sale with its explanation |\n"
        "| **Explainability** | Objective 5: what drives the model overall and worked case studies |\n"
        "| **Ethics & Evaluation** | Ethical safeguards, limitations, professional feedback and references |")

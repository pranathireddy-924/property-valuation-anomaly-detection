"""Data & Method: datasets, research design, cleaning, split, features and geography codes."""
import pandas as pd
import streamlit as st

from dashboard.common import text_table, callout, figure, gbp, page_header, table

PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#f3f6fa", color="#1f3a5f", fontname="Helvetica", fontsize=11];
  edge [color="#4a5568"];
  pp [label="HM Land Registry\\nPrice Paid Data\\n(4.3m sales)"]; on [label="ONS Postcode\\nDirectory"]; hpi [label="UK House\\nPrice Index"];
  clean [label="1. Clean & link\\nto geography"]; split [label="2. Time-based split\\ntrain / validate / test"];
  feat [label="3. Spatial & temporal\\nfeatures"]; models [label="4. Linear Regression\\nvs XGBoost"];
  eval [label="5. Evaluate\\nMAE · RMSE · R²"]; anom [label="6. Anomaly score\\n(unusual sales)"]; shap [label="7. SHAP\\nexplanations"];
  app [label="Prototype\\n(this dashboard)", fillcolor="#1f3a5f", fontcolor="white"];
  pp -> clean; on -> clean; clean -> split -> feat; hpi -> feat; feat -> models -> eval;
  models -> anom; models -> shap; eval -> app; anom -> app; shap -> app;
}
"""


def render() -> None:
    page_header("Data & Method",
                "How the data was obtained and prepared, how the study was designed, and what each "
                "feature and area code means. This page addresses <b>Objective 1</b> (a cleaned, reusable "
                "dataset) and explains the inputs for <b>Objective 2</b>.")

    st.header("Datasets")
    ogl = "[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)"
    st.markdown(
        "| Dataset | Publisher | Role | Period used | Licence |\n|---|---|---|---|---|\n"
        "| [HM Land Registry Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) "
        "| HM Land Registry | Main dataset: every residential sale registered in England & Wales (price, date, postcode, "
        f"type, tenure, new build) | Jan 2022 - Sep 2026 | {ogl} |\n"
        "| [ONS Postcode Directory (Aug 2026)](https://www.ons.gov.uk/methodology/geography/geographicalproducts/postcodeproducts) "
        "| Office for National Statistics | Links each postcode to local authority, neighbourhood (LSOA/MSOA), region, "
        f"coordinates, rural-urban class and deprivation | Current release | {ogl} |\n"
        "| [UK House Price Index (Sep 2026)](https://landregistry.data.gov.uk/app/ukhpi/) | HM Land Registry / ONS "
        f"| Official monthly average prices by local authority and property type (market level over time) | 2020 - Sep 2026 | {ogl} |")
    st.caption("Why these sources: they are official, cover every registered sale (not a sample or a survey), "
               "are free to reuse, and are updated monthly, so the approach can be repeated by a business.")

    st.header("Research design")
    st.markdown(
        "| Element | Choice | Reason |\n|---|---|---|\n"
        "| Philosophy | Pragmatism | The aim is a practical, useful tool; accuracy and usefulness are both assessed |\n"
        "| Approach | Deductive | Tests stated expectations (e.g. XGBoost beats Linear Regression; spatial and temporal features improve accuracy) |\n"
        "| Methods | Mixed | Quantitative modelling plus a short professional questionnaire |\n"
        "| Strategy | Case study | Recent England & Wales transactions, 2022-2026 |\n"
        "| Time horizon | Longitudinal, time-based split | Models are trained on older sales and tested on later ones, as in real use |")

    st.header("Processing pipeline")
    st.graphviz_chart(PIPELINE_DOT)
    st.markdown('<p class="caption-fig">Diagram 1: End-to-end processing pipeline from raw data to prototype.</p>',
                unsafe_allow_html=True)

    st.header("Data cleaning")
    clean = table("01_cleaning_report.csv")
    if clean is not None:
        text_table(clean.rename(columns={"step": "Step", "rows_before": "Rows before", "rows_removed": "Removed",
                                           "rows_after": "Rows after", "reason": "Reason"}))
        st.markdown(
            "Only **standard market sales** (Land Registry category A) are used to train the models. "
            "Category B sales (repossessions, buy-to-let and transfers to companies) are kept aside and used "
            "later as an independent check of the anomaly score. Address fields (house number, flat, street) "
            "are **never loaded**, in line with the ethics approval.")

    st.header("Time-based split")
    split = table("02_data_summary.csv")
    if split is not None:
        s = split[["split", "sales", "from", "to", "median_price"]].copy()
        s["split"] = s["split"].map({"train": "Training", "val": "Validation", "test": "Test"})
        s["sales"] = s["sales"].map("{:,}".format)
        s["median_price"] = s["median_price"].map(gbp)
        text_table(s.rename(columns={"split": "Period", "sales": "Sales", "from": "From", "to": "To",
                                       "median_price": "Median price"}))
    st.markdown("The models learn from **2022-2024**, settings are tuned on **2025**, and all reported results "
                "are on **2026** sales that the models never saw. A random split was rejected because it would "
                "let the model learn from future market conditions.")
    figure("fig02_monthly_trend_and_split.png", "Monthly median price and sales volume, with the training, "
           "validation and test periods. The spike in March 2025 is the rush to complete before the stamp duty change on 1 April 2025.")
    figure("fig01_price_distribution.png", "Distribution of sale prices by property type (log scale).")

    st.header("Features used by the models")
    text_table(pd.DataFrame([
        ("Property", "Property type", "Detached, semi-detached, terraced, flat/maisonette"),
        ("Property", "Tenure", "Freehold or leasehold"),
        ("Property", "New build", "Whether the property was newly built"),
        ("Temporal", "Sale year, sale month, months since start", "When the sale happened (seasonality and trend)"),
        ("Spatial", "Region, local authority, postcode area", "Administrative location"),
        ("Spatial", "Latitude, longitude", "Exact position of the postcode"),
        ("Spatial", "Rural-urban class, deprivation percentile", "Character of the neighbourhood (ONS)"),
        ("Spatial: local price patterns", "Local price level for postcode sector, district, LSOA, MSOA, local authority, and district × type",
         "Average (smoothed) price of earlier sales in that area, using training-period sales only"),
        ("Spatial", "Sales volume in sector", "How active the local market is"),
        ("Market index", "HPI average price (LA and type), HPI 12-month change",
         "Official price level for the area, lagged 2 months to match publication delay"),
    ], columns=["Group", "Feature", "Meaning"]))
    callout("<b>Avoiding data leakage.</b> Local price levels for training sales are computed 'out-of-fold' (a "
            "sale never sees its own price), and for later periods only training-period prices are used. "
            "Index values are taken from two months before the sale, when they would actually have been published.")

    st.header("Understanding postcodes and area codes")
    st.markdown("The dashboard refers to several geographical levels. Using the example postcode **S10 2TN** (Sheffield):")
    text_table(pd.DataFrame([
        ("Postcode area", "S", "The letters at the start: a large area around a main town (S = Sheffield). 106 postcode areas have sales in England & Wales."),
        ("Postcode district", "S10", "The first half of the postcode: a group of streets or a small town. 2,285 districts have sales in the data."),
        ("Postcode sector", "S10 2", "District plus the first digit of the second half. 8,236 sectors have sales in the data."),
        ("Full postcode (unit)", "S10 2TN", "About 15 addresses on one street. Not shown in results, for privacy."),
        ("LSOA", "E01008068", "Lower Layer Super Output Area: a census neighbourhood of about 1,500 people (ONS)."),
        ("MSOA", "E02001636", "Middle Layer Super Output Area: about 8,000 people, several LSOAs."),
        ("Local authority", "Sheffield (E08000039)", "The council area. 318 in England & Wales."),
        ("Region", "Yorkshire and The Humber", "Nine English regions plus Wales."),
    ], columns=["Level", "Example", "Meaning"]))

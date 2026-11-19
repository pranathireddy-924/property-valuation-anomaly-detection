"""Shared loaders, styling and wording helpers for the dashboard pages."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from propval.config import PROPERTY_TYPE_NAMES, REGION_NAMES
from propval.predict import PropertyValuer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TABLES, FIGURES, APP_DATA = OUT / "tables", OUT / "figures", OUT / "app_data"

TITLE = "AI-Driven Property Valuation and Anomaly Detection Using Spatial and Temporal Housing Market Data"
AUTHOR = "Pranathi Nooka (C5055924)"
COURSE = "MSc Data Science and Artificial Intelligence · Sheffield Hallam University"
SUPERVISOR = "Supervisor: Royce Copley"

TENURE_NAMES = {"F": "Freehold", "L": "Leasehold"}
TYPE_FROM_NAME = {v: k for k, v in PROPERTY_TYPE_NAMES.items()}


# --------------------------------------------------------------------------- style
def inject_css() -> None:
    st.markdown("""
    <style>
      h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: #1f3a5f; }
      h1 { font-size: 2.0rem !important; line-height: 1.25; }
      .subtitle { color: #4a5568; font-size: 0.95rem; margin-top: -0.6rem; }
      .caption-fig { color: #4a5568; font-size: 0.85rem; font-style: italic; margin-top: -0.4rem; }
      .callout { background: #f3f6fa; border-left: 4px solid #1f3a5f; padding: 0.8rem 1rem;
                 border-radius: 4px; margin: 0.4rem 0 1rem 0; }
      .callout-warn { background: #fff8eb; border-left: 4px solid #c05621; padding: 0.8rem 1rem;
                      border-radius: 4px; margin: 0.4rem 0 1rem 0; }
      .rq { font-family: Georgia, serif; font-size: 1.15rem; color: #1a202c; }
      div[data-testid="stMetricValue"] { color: #1f3a5f; }
    </style>""", unsafe_allow_html=True)


def page_header(title: str, purpose: str, objective: str | None = None) -> None:
    st.title(title)
    if objective:
        st.markdown(f'<p class="subtitle"><b>Addresses:</b> {objective}</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="callout">{purpose}</div>', unsafe_allow_html=True)


def callout(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="{"callout-warn" if warn else "callout"}">{text}</div>', unsafe_allow_html=True)


_fig_counter = {"n": 0}


def reset_figures() -> None:
    _fig_counter["n"] = 0


def figure(file: str, caption: str, width: str = "stretch") -> None:
    """Show a saved chart with a numbered academic caption."""
    path = FIGURES / file
    if not path.exists():
        st.info(f"Chart {file} not found - run the pipeline first.")
        return
    _fig_counter["n"] += 1
    st.image(str(path), width=width)
    st.markdown(f'<p class="caption-fig">Figure {_fig_counter["n"]}: {caption}</p>', unsafe_allow_html=True)


def text_table(df: pd.DataFrame) -> None:
    """Static table that wraps long text (st.dataframe truncates it)."""
    st.table(df.style.hide(axis="index"))


def gbp(v: float) -> str:
    return f"£{v:,.0f}"


# --------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading the trained model and postcode directory (about 10 seconds)...")
def valuer() -> PropertyValuer:
    return PropertyValuer.load(OUT / "models")


def model_ready() -> bool:
    return (OUT / "models" / "valuer_bundle.joblib").exists()


@st.cache_data
def table(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def run_config() -> dict:
    p = OUT / "run_config.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def la_names() -> dict[str, str]:
    p = APP_DATA / "la_names.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["lad"], df["la_name"]))


@st.cache_data(show_spinner=False)
def postcodes() -> pd.DataFrame:
    df = pd.read_parquet(APP_DATA / "postcodes_with_sales.parquet")
    names = la_names()
    df["la_name"] = df["lad"].map(names).fillna(df["lad"])
    df["region_name"] = df["region"].map(REGION_NAMES).fillna(df["region"])
    return df


@st.cache_data(show_spinner=False)
def district_prices() -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / "district_prices.parquet")


@st.cache_data(show_spinner=False)
def flagged_rows() -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / "flagged_rows.parquet")


def app_data_ready() -> bool:
    return (APP_DATA / "postcodes_with_sales.parquet").exists()


def la_name(code: str) -> str:
    return la_names().get(code, code)


# --------------------------------------------------------------------------- wording
LEVEL_WORDS = {
    "te_sector": "postcode sector", "te_district": "postcode district", "te_lsoa": "neighbourhood (LSOA)",
    "te_msoa": "wider neighbourhood (MSOA)", "te_lad": "local authority", "te_district_type": "postcode district",
}


def describe_factor(feature: str, label: str, value: str, effect: float, ptype: str) -> str:
    """One plain-English sentence per SHAP factor."""
    move = "raises" if effect > 0 else "lowers"
    amount = f"**{abs(effect):.1f}%**"
    if feature == "te_district_type":
        return f"Earlier {ptype.lower()} sales in this {LEVEL_WORDS[feature]} averaged {value}; this {move} the estimate by {amount}."
    if feature in LEVEL_WORDS:
        return f"Earlier sales in this {LEVEL_WORDS[feature]} averaged {value}; this {move} the estimate by {amount}."
    if feature == "hpi_type_price_log":
        return f"The official UK House Price Index puts this property type's average in the local authority at {value}; this {move} the estimate by {amount}."
    if feature == "hpi_avg_price_log":
        return f"The UK House Price Index average for the local authority is {value}; this {move} the estimate by {amount}."
    if feature == "hpi_12m_change":
        return f"Local prices changed {value} over the previous year (HPI); this {move} the estimate by {amount}."
    if feature == "imd_pct":
        return (f"On the official deprivation index the neighbourhood is at the {value} (0 = most deprived, "
                f"100 = least deprived); this {move} the estimate by {amount}.")
    if feature == "duration":
        return f"Tenure is {value.lower()}; this {move} the estimate by {amount}."
    if feature == "property_type":
        return f"Property type is {value.lower()}; this {move} the estimate by {amount}."
    if feature == "old_new":
        return f"New build: {value.lower()}; this {move} the estimate by {amount}."
    if feature == "log_n_sector":
        return f"The postcode sector had {value} in the training period (market activity); this {move} the estimate by {amount}."
    if feature in ("lat", "long"):
        return f"Exact location ({'latitude' if feature == 'lat' else 'longitude'} {value}) {move} the estimate by {amount}."
    return f"{label}: {value}. This {move} the estimate by {amount}."


def explanation_table(expl: pd.DataFrame, ptype: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Factor": expl["label"],
        "Value for this property": expl["value"],
        "Effect on estimate": expl["effect_%"].map(lambda v: f"{v:+.1f}%"),
        "What it means": [describe_factor(f, lbl, v, e, ptype) for f, lbl, v, e in
                          zip(expl["feature"], expl["label"], expl["value"], expl["effect_%"])],
    })


def explanation_chart(expl: pd.DataFrame):
    import altair as alt
    d = expl.assign(factor=[f"{l} = {v}" for l, v in zip(expl["label"], expl["value"])],
                    direction=["Raises estimate" if e > 0 else "Lowers estimate" for e in expl["effect_%"]])
    return (alt.Chart(d).mark_bar().encode(
        x=alt.X("effect_%:Q", title="Effect on predicted price (%)"),
        y=alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises estimate", "Lowers estimate"],
                                                         range=["#2f855a", "#c53030"]), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["factor", alt.Tooltip("effect_%:Q", format="+.1f")],
    ).properties(height=34 * len(d) + 40))

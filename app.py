"""Research dashboard and prototype for the dissertation
'AI-Driven Property Valuation and Anomaly Detection Using Spatial and Temporal Housing Market Data'.

    streamlit run app.py

Requires the trained model and results:  python run_pipeline.py  (then  python prepare_dashboard.py
if the dashboard lookup files are missing).
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dashboard import anomalies, data_method, ethics, explainability, overview, performance, valuation  # noqa: E402
from dashboard.common import AUTHOR, app_data_ready, inject_css, model_ready, reset_figures  # noqa: E402

st.set_page_config(page_title="AI Property Valuation & Anomaly Detection", page_icon="🏠", layout="wide")
inject_css()


def _wrap(fn):
    def page():
        reset_figures()
        if fn is not overview.render and not model_ready():
            st.error("No trained model found. Run `python run_pipeline.py` first.")
            st.stop()
        if fn in (valuation.render,) and not app_data_ready():
            st.error("Dashboard lookup files are missing. Run `python prepare_dashboard.py`.")
            st.stop()
        fn()
    page.__name__ = fn.__module__.split(".")[-1]
    return page


pages = {
    "Project": [
        st.Page(_wrap(overview.render), title="Overview", icon="🏠", default=True),
        st.Page(_wrap(data_method.render), title="Data & Method", icon="🗂️", url_path="data-method"),
    ],
    "Prototype": [
        st.Page(_wrap(valuation.render), title="Valuation Tool", icon="🧮", url_path="valuation"),
    ],
    "Results": [
        st.Page(_wrap(performance.render), title="Model Performance", icon="📈", url_path="performance"),
        st.Page(_wrap(anomalies.render), title="Unusual Transactions", icon="🚩", url_path="unusual-transactions"),
        st.Page(_wrap(explainability.render), title="Explainability", icon="🔍", url_path="explainability"),
        st.Page(_wrap(ethics.render), title="Ethics & Evaluation", icon="⚖️", url_path="ethics-evaluation"),
    ],
}

with st.sidebar:
    st.markdown("### AI Property Valuation & Anomaly Detection")
    st.caption(f"MSc dissertation prototype · {AUTHOR} · Sheffield Hallam University")

nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption("Data: HM Land Registry Price Paid Data, ONS Postcode Directory and UK House Price Index "
               "(Open Government Licence v3.0). England & Wales, Jan 2022 - Sep 2026.")
    st.caption("Decision support only. 'Unusual' does not mean fraudulent or incorrect.")

nav.run()

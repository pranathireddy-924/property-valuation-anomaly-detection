"""Ethics, limitations, data attribution and references."""
import pandas as pd
import streamlit as st

from dashboard.common import text_table, OUT, page_header

REFERENCES = [
    "Baur, K., Rosenfelder, M., & Lutz, B. (2023). Automated real estate valuation with machine learning models using property descriptions. *Expert Systems with Applications, 213*, 119147. https://doi.org/10.1016/j.eswa.2022.119147",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). https://doi.org/10.1145/2939672.2939785",
    "HM Land Registry. (2026). *Price Paid Data*. GOV.UK. https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads",
    "HM Land Registry. (2026). *UK House Price Index*. https://landregistry.data.gov.uk/app/ukhpi/",
    "Kee, T., & Ho, W. K. O. (2025). eXplainable machine learning for real estate: XGBoost and Shapley values in price prediction. *Civil Engineering Journal, 11*(5), 2116-2133. https://doi.org/10.28991/CEJ-2025-011-05-022",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (Vol. 30). https://doi.org/10.5555/3295222.3295230",
    "Neves, F. T., Aparicio, M., & Neto, M. de C. (2024). The impacts of open data and eXplainable AI on real estate price predictions in smart cities. *Applied Sciences, 14*(5), 2209. https://doi.org/10.3390/app14052209",
    "Noh, Y., & Park, C. Y. (2026). Explainable machine learning for urban housing prices: Policy relevant diagnostics across three cities. *Cities, 176*, 107266. https://doi.org/10.1016/j.cities.2026.107266",
    "Office for National Statistics. (2026). *Postcode products*. https://www.ons.gov.uk/methodology/geography/geographicalproducts/postcodeproducts",
    "Soltani, A., Heydari, M., Aghaei, F., & Pettit, C. J. (2022). Housing price prediction incorporating spatio-temporal dependency into machine learning algorithms. *Cities, 131*, 103941. https://doi.org/10.1016/j.cities.2022.103941",
    "The National Archives. (n.d.). *Open Government Licence for public sector information, version 3.0*. https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
]


def render() -> None:
    page_header("Ethics & Limitations",
                "How the project protects people and presents results responsibly, what its limitations are, "
                "and the sources it builds on.")

    st.header("Ethical safeguards")
    st.markdown(
        "- **Approved research.** The project was reviewed under the University's low-risk ethics procedure (UREC2).\n"
        "- **Public, licensed data.** All datasets are official open data reused under the Open Government Licence v3.0.\n"
        "- **Privacy by design.** House numbers, flat numbers and street names are never loaded. Results show only "
        "the postcode district (e.g. S10), never an individual address.\n"
        "- **Careful language.** Flagged sales are described as *unusual transactions*, never as fraud or incorrect prices.\n"
        "- **Human in the loop.** The tool supports professional judgement and does not replace it.")

    st.header("Limitations")
    st.markdown(
        "- **Missing property details.** Price Paid Data has no floor area, bedrooms, condition or plot size, so two "
        "homes in the same street can legitimately differ in price. This sets a ceiling on accuracy (typical error "
        "about 15%).\n"
        "- **Market period.** 2022-2026 was a broadly flat market. The models may behave differently in a rapidly "
        "rising or falling market and should be retrained regularly.\n"
        "- **Registration delay.** Recent months are incomplete, because sales appear in the data some weeks after completion.\n"
        "- **Anomalies are statistical.** A high score shows that a price is unusual compared with similar sales, not why it is unusual.\n"
        "- **Scope.** England and Wales only. Standard residential sales (houses and flats) between £10,000 and £10 million.")
    summary = OUT / "questionnaire" / "18_questionnaire_summary.csv"
    if summary.exists():   # shown only once questionnaire results have been analysed
        st.header("Professional evaluation")
        s = pd.read_csv(summary)
        text_table(pd.DataFrame({"Statement": s["statement"], "Responses": s["n"],
                                   "Agree or strongly agree": s["agree_%"].map("{:.0f}%".format),
                                   "Median (1-5)": s["median"]}))
        fig = OUT / "questionnaire" / "fig14_questionnaire_feedback.png"
        if fig.exists():
            st.image(str(fig), width="stretch")
        com = OUT / "questionnaire" / "20_questionnaire_comments_anonymised.csv"
        if com.exists():
            with st.expander("Anonymised comments"):
                text_table(pd.read_csv(com))

    st.header("Data attribution")
    st.caption("Contains HM Land Registry data © Crown copyright and database right 2026. This data is licensed "
               "under the Open Government Licence v3.0. Source: Office for National Statistics, licensed under the "
               "Open Government Licence v3.0. Contains OS data © Crown copyright and database right 2026.")

    st.header("References")
    for ref in REFERENCES:
        st.markdown(f"- {ref}")

# AI-Driven Property Valuation and Anomaly Detection Using Spatial and Temporal Housing Market Data

MSc Data Science and AI dissertation code: Pranathi Nooka (C5055924), Sheffield Hallam University.

**Research question:** How accurately can an explainable AI model estimate property transaction prices and identify unusual sale prices using spatial and temporal housing market data?

This repository contains the full Python pipeline and a prototype app:

1. Cleans HM Land Registry Price Paid Data (England & Wales, 2022–2026) and links each sale to ONS geography.
2. Builds spatial and temporal features, including local price patterns and the UK House Price Index.
3. Trains a **Linear Regression** baseline and an **XGBoost** model on a time-based split, then compares them with MAE, RMSE and R².
4. Runs a **feature ablation** to test whether spatial and temporal features improve accuracy.
5. Flags **unusual transactions** where the actual price is far from the predicted price (robust z-score).
6. Explains predictions with **SHAP**.
7. Provides a **Streamlit research dashboard**. It sets out the problem, aims, objectives and beneficiaries, explains the data and method, and presents the results for each objective. It also includes an interactive valuation tool: browse Region → Local authority → Postcode district → Postcode, or type any postcode; get an estimate, an anomaly check and a plain-English SHAP explanation.

## Folder layout

```
Pranathi Dissertation/
├── datasets/                      raw data (see datasets/README.md for sources and licence)
│   ├── price_paid/pp-2022.csv ... pp-2026.csv
│   ├── onspd/Data/ONSPD_AUG_2026_UK.csv
│   └── uk_hpi/UK-HPI-full-file-2026-09.csv
└── Pranathi Dissertation Code/    <- this folder
    ├── run_pipeline.py            run everything
    ├── app.py                     Streamlit research dashboard and prototype
    ├── dashboard/                 dashboard pages (overview, data & method, valuation tool,
    │                              performance, unusual transactions, explainability, ethics)
    ├── prepare_dashboard.py       builds dashboard lookups (LA names, postcodes, local prices)
    ├── analyse_questionnaire.py   summarises professional feedback (counts, % agree, chart)
    ├── value_property.py          command-line valuation of one property
    ├── src/propval/               the package (config, data_loading, features, models,
    │                              evaluation, anomaly, explain, plots, pipeline, predict)
    ├── tests/                     30 unit / integration / system tests (pytest)
    ├── docs/DESIGN.md             architecture, design decisions, test plan
    ├── docs/test_results.txt      latest test run
    └── outputs/                   generated tables, figures, RESULTS.md, trained model
```

## Setup (Windows, Python 3.12+)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python run_pipeline.py --quick          # 5% sample, ~2 min: checks everything works
python run_pipeline.py                  # full England & Wales run with tuning, ablation, SHAP
python run_pipeline.py --region E12000003   # one region only (Yorkshire and The Humber)
python -m pytest                        # run the test suite
python prepare_dashboard.py             # dashboard lookups (run automatically by the pipeline)
streamlit run app.py                    # open the dashboard in the browser
python analyse_questionnaire.py --template           # blank sheet for feedback answers (1-5)
python analyse_questionnaire.py questionnaire/responses.csv  # counts, % agree, chart
```

Value one property from the command line:

```bash
python value_property.py --postcode "S1 2BJ" --type F --tenure L --date 2026-06-15 --price 180000
```

The first full run reads the raw CSVs, which takes about 1–2 minutes. It caches a joined Parquet file in `outputs/cache/`, so later runs start faster. Use `--no-cache` to rebuild it. All settings (years, split dates, thresholds, model parameters) are in `src/propval/config.py`.

## Method summary

| Step | Detail |
|---|---|
| Data | Price Paid standard (category A) sales of houses and flats, £10k–£10m, joined to the ONS Postcode Directory |
| Split | Train 2022–2024 · validation 2025 · test Jan–Sep 2026 (chronological) |
| Target | log(price); errors reported in £ and % |
| Property features | Property type, new build, tenure |
| Temporal features | Sale year, sale month, months since start |
| Spatial features | Region, local authority, postcode area, latitude/longitude, rural-urban class, deprivation percentile, local price levels (out-of-fold smoothed target encoding for postcode sector, district, LSOA, MSOA, LA, district × type), sector sales volume |
| Market features | UK HPI price for the local authority and property type, plus 12-month change (lagged 2 months) |
| Models | Naive local median; Linear Regression; XGBoost (tuned by random search on the validation period, early stopping) |
| Anomaly score | Robust z = (residual − median) / (1.4826 × MAD), per property type, calibrated on validation; \|z\| > 3 = unusual |
| Validation of anomalies | 20 injected anomalies in tests; real category B sales (repossessions etc.) should be flagged more often |
| Explainability | Exact TreeSHAP: global importance, beeswarm, importance by feature group, 6 case studies |

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `RESULTS.md` | Key results in one page (auto-generated) |
| `tables/01_cleaning_report.csv` | Rows removed by each cleaning rule |
| `tables/02_data_summary.csv` | Sales, dates and price mix per split |
| `tables/05_model_comparison.csv` | MAE, RMSE, R², MdAPE, % within 10/20%, for validation and test |
| `tables/06_`, `07_` | XGBoost error by property type and by region |
| `tables/09_feature_ablation.csv` | Hypothesis test: property → + temporal → + spatial → full |
| `tables/11_unusual_transactions_test.csv` | Flagged sales (postcode district only, no addresses) |
| `tables/14_anomaly_check_category_b.csv` | Flag rate for standard vs category B sales |
| `tables/15_`–`17_` | SHAP importance, by group, case-study explanations |
| `figures/fig01`–`fig13` | All charts for the report and presentation |
| `models/` | Trained model bundle used by the app |

## Ethics and limitations

- Data is public and licensed under the Open Government Licence v3.0. Address fields are never loaded, and outputs show only the postcode district.
- A flagged sale is an **unusual transaction for human review**. It is not evidence of fraud or an incorrect price.
- Price Paid has no floor area, bedroom count or condition. The model estimates the expected price from location, type, tenure, timing and market level only, and this limits accuracy for individual homes.
- 2026 contains sales registered up to September 2026. Recent months are incomplete because of registration delays.

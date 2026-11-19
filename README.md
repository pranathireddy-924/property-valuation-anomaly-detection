# Property Valuation and Anomaly Detection

MSc Data Science and AI dissertation project, Sheffield Hallam University.
Pranathi Nooka (C5055924).

Full title: *AI-Driven Property Valuation and Anomaly Detection Using Spatial and Temporal Housing Market Data*

## What this project does

The code predicts the sale price of a house or flat in England and Wales. It then compares the predicted price with the real sale price. If the real price is much higher or much lower than expected, the sale is marked as unusual so that a person can check it.

It also shows the reasons behind each prediction using SHAP.

Main steps:

1. Clean the HM Land Registry sales data (2022 to 2026) and add location details from the ONS Postcode Directory.
2. Create features such as postcode area, property type, sale month and local average prices.
3. Train two models, Linear Regression and XGBoost, and compare them using MAE, RMSE and R².
4. Check if location and time features improve the results.
5. Give each sale an anomaly score and mark the unusual ones.
6. Explain the predictions with SHAP.
7. Show everything in a Streamlit dashboard, where you can also try your own postcode and price.

## Data used

All data is public and free to use under the Open Government Licence v3.0.

- HM Land Registry Price Paid Data, 2022 to 2026
- ONS Postcode Directory, August 2026
- UK House Price Index, September 2026

The data files are not included in this repository because they are very large. Download them and place them in a `datasets` folder next to this folder:

```
Pranathi Dissertation/
├── datasets/
│   ├── price_paid/pp-2022.csv ... pp-2026.csv
│   ├── onspd/Data/ONSPD_AUG_2026_UK.csv
│   └── uk_hpi/UK-HPI-full-file-2026-09.csv
└── Pranathi Dissertation Code/   (this folder)
```

## Folders and files

| Name | What it is |
|---|---|
| `run_pipeline.py` | Runs all the steps |
| `app.py` | Starts the dashboard |
| `dashboard/` | The dashboard pages |
| `value_property.py` | Values one property from the command line |
| `prepare_dashboard.py` | Builds the lookup files the dashboard needs |
| `analyse_questionnaire.py` | Summarises the questionnaire answers |
| `src/propval/` | The main code |
| `tests/` | 30 tests |
| `docs/` | Design notes and test results |
| `outputs/` | Tables, charts and results created by the pipeline |

## How to set up

You need Python 3.12 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

```bash
python run_pipeline.py --quick     # quick test on a small sample (about 2 minutes)
python run_pipeline.py             # full run on all of England and Wales
python -m pytest                   # run the tests
streamlit run app.py               # open the dashboard
```

Value one property:

```bash
python value_property.py --postcode "S1 2BJ" --type F --tenure L --date 2026-06-15 --price 180000
```

Questionnaire:

```bash
python analyse_questionnaire.py --template                     # make a blank answer sheet
python analyse_questionnaire.py questionnaire/responses.csv    # summarise the answers
```

All settings are in `src/propval/config.py`.

## How the data is split

- Training: 2022 to 2024
- Validation: 2025
- Testing: January to September 2026

The models are tested only on sales they have never seen.

## Results

The main results are saved in `outputs/RESULTS.md`. Tables are in `outputs/tables/` and charts are in `outputs/figures/`.

## Ethics and limits

- No addresses are used. Results show only the postcode district.
- An unusual sale does not mean fraud or a wrong price. It only means the price is very different from similar sales and should be checked by a person.
- The data has no floor area, number of bedrooms or condition, so the predictions for single homes are not exact.
- Sales from late 2026 may be missing because they take time to be registered.

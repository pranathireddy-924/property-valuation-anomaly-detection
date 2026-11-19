# Results summary

Data: HM Land Registry Price Paid 2022-2026, region = England & Wales, sample = full data. Split: train <= 2024-12-31 < validation <= 2025-12-31 < test.

## Data cleaning

| step | rows_before | rows_removed | rows_after |
|---|---|---|---|
| Raw records | 4,321,643 | 0 | 4,321,643 |
| Record status | 4,321,643 | 0 | 4,321,643 |
| Duplicates | 4,321,643 | 0 | 4,321,643 |
| Missing postcode | 4,321,643 | 11,094 | 4,310,549 |
| Property type 'Other' | 4,310,549 | 223,662 | 4,086,887 |
| Price range | 4,086,887 | 801 | 4,086,086 |
| Postcode match (ONSPD) | 4,086,086 | 149 | 4,085,937 |

## Data split

| split | sales | from | to | median_price |
|---|---|---|---|---|
| train | 2,382,103 | 2022-01-01 | 2024-12-31 | 290,000 |
| val | 822,349 | 2025-01-01 | 2025-12-31 | 299,950 |
| test | 392,314 | 2026-01-01 | 2026-09-30 | 290,000 |

## Model comparison - test period

| model | MAE | RMSE | R2 | R2_log | MdAPE_% | within_10% | within_20% |
|---|---|---|---|---|---|---|---|
| Naive local median | 90,617.123 | 194,103.794 | 0.551 | 0.693 | 18.063 | 29.606 | 54.153 |
| Linear Regression | 84,959.204 | 178,231.328 | 0.621 | 0.733 | 17.173 | 30.955 | 56.531 |
| XGBoost | 78,226.814 | 164,126.264 | 0.679 | 0.772 | 15.591 | 33.936 | 60.658 |

## Model comparison - validation period

| model | MAE | RMSE | R2 | R2_log | MdAPE_% | within_10% | within_20% |
|---|---|---|---|---|---|---|---|
| Naive local median | 97,645.672 | 222,506.733 | 0.526 | 0.698 | 18.294 | 29.254 | 53.643 |
| Linear Regression | 91,341.616 | 203,527.962 | 0.603 | 0.740 | 17.386 | 30.726 | 55.910 |
| XGBoost | 82,971.395 | 187,127.454 | 0.665 | 0.783 | 15.428 | 34.217 | 60.919 |

## Feature ablation (XGBoost, test period)

| feature_set | n_features | MAE | R2_log | MdAPE_% | within_10% |
|---|---|---|---|---|---|
| A. Property only | 3 | 147,928.475 | 0.171 | 33.492 | 15.699 |
| B. Property + temporal | 6 | 148,062.076 | 0.170 | 33.581 | 15.725 |
| C. Property + spatial | 17 | 77,629.442 | 0.774 | 15.566 | 33.788 |
| D. Full (property + temporal + spatial + HPI) | 23 | 78,541.043 | 0.771 | 15.639 | 33.818 |

## Anomaly detection

| group | sales | flag_rate_% | median_abs_diff_% |
|---|---|---|---|
| Standard sales (cat. A) | 392,314 | 2.878 | 15.782 |
| Additional sales (cat. B) | 50,998 | 8.838 | 21.828 |

## SHAP importance by feature group

| group | mean_abs_shap | share_% |
|---|---|---|
| Spatial | 0.511 | 68.588 |
| Market index (UK HPI) | 0.108 | 14.467 |
| Property | 0.106 | 14.258 |
| Temporal | 0.020 | 2.687 |

## Run time (seconds)

| stage | seconds |
|---|---|
| 1. Load and clean data | 116.1 |
| 2. Spatial and temporal feature engineering | 39.6 |
| 3a. Naive local baseline | 0.0 |
| 3b. Linear Regression baseline | 25.3 |
| 3c. XGBoost tuning and training | 2,203.9 |
| 4. Evaluation | 5.9 |
| 5. Feature ablation (hypothesis test) | 2,173.8 |
| 6. Price anomaly detection | 10.1 |
| 7. SHAP explanations | 161.6 |
| 8. Save prototype model | 8.9 |

Total: 79.1 minutes

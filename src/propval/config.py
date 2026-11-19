"""Central configuration for the property valuation and anomaly detection pipeline.

Every tunable choice (years, split dates, thresholds, model settings) lives here so
that experiments are reproducible and documented in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# <dissertation folder>/Pranathi Dissertation Code/src/propval/config.py
CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = CODE_ROOT.parent / "datasets"


def display_path(path) -> str:
    """Path relative to the project folder, so logs and saved settings never show a user's own folders."""
    path = Path(path).resolve()
    for base in (CODE_ROOT, CODE_ROOT.parent):
        try:
            return path.relative_to(base).as_posix() or "."
        except ValueError:
            continue
    return path.name

# Column layout of the (header-less) HM Land Registry Price Paid CSV files.
PRICE_PAID_COLUMNS = [
    "transaction_id", "price", "date_of_transfer", "postcode", "property_type",
    "old_new", "duration", "paon", "saon", "street", "locality", "town_city",
    "district", "county", "ppd_category", "record_status",
]
# Address-level fields are dropped on load (privacy by design, see ethics form).
PRICE_PAID_KEEP = [
    "transaction_id", "price", "date_of_transfer", "postcode", "property_type",
    "old_new", "duration", "ppd_category", "record_status",
]

PROPERTY_TYPE_NAMES = {"D": "Detached", "S": "Semi-detached", "T": "Terraced", "F": "Flat/Maisonette"}

REGION_NAMES = {
    "E12000001": "North East",
    "E12000002": "North West",
    "E12000003": "Yorkshire and The Humber",
    "E12000004": "East Midlands",
    "E12000005": "West Midlands",
    "E12000006": "East of England",
    "E12000007": "London",
    "E12000008": "South East",
    "E12000009": "South West",
    "W92000004": "Wales",
}

# ---------------------------------------------------------------------------
# Feature groups. The ablation study adds these groups step by step to test the
# hypothesis that spatial and temporal features improve valuation accuracy.
# ---------------------------------------------------------------------------
PROPERTY_FEATURES = ["property_type", "old_new", "duration"]
TEMPORAL_FEATURES = ["sale_year", "sale_month", "months_since_start"]
SPATIAL_FEATURES = [
    "region", "lad", "pc_area", "lat", "long", "ruc21", "imd_pct",
    "te_sector", "te_district", "te_lsoa", "te_msoa", "te_lad", "te_district_type",
    "log_n_sector",
]
MARKET_FEATURES = ["hpi_type_price_log", "hpi_avg_price_log", "hpi_12m_change"]

CATEGORICAL_FEATURES = ["property_type", "old_new", "duration", "region", "lad", "pc_area", "ruc21"]

FEATURE_GROUPS = {
    "Property": PROPERTY_FEATURES,
    "Temporal": TEMPORAL_FEATURES,
    "Spatial": SPATIAL_FEATURES,
    "Market index (UK HPI)": MARKET_FEATURES,
}

ABLATION_SETS = {
    "A. Property only": PROPERTY_FEATURES,
    "B. Property + temporal": PROPERTY_FEATURES + TEMPORAL_FEATURES,
    "C. Property + spatial": PROPERTY_FEATURES + SPATIAL_FEATURES,
    "D. Full (property + temporal + spatial + HPI)": PROPERTY_FEATURES + TEMPORAL_FEATURES
    + SPATIAL_FEATURES + MARKET_FEATURES,
}
ALL_FEATURES = ABLATION_SETS["D. Full (property + temporal + spatial + HPI)"]


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = CODE_ROOT / "outputs"

    # ---- data scope ----------------------------------------------------------
    years: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
    region: str | None = None          # e.g. "E12000003"; None = all England & Wales
    sample_frac: float | None = None   # e.g. 0.05 for a quick run
    min_price: int = 10_000            # below this: almost always non-market transfers
    max_price: int = 10_000_000        # above this: very rare, mostly non-standard sales

    # ---- time-based split (no random split: avoids mixing market periods) -------
    train_end: str = "2024-12-31"      # train  = Jan 2022 - Dec 2024
    val_end: str = "2025-12-31"        # valid. = Jan 2025 - Dec 2025, test = 2026

    # ---- feature engineering ------------------------------------------------
    te_smoothing: float = 20.0         # pseudo-count for smoothed target encoding
    te_folds: int = 5                  # out-of-fold encoding on the training set
    hpi_lag_months: int = 2            # UK HPI is published ~2 months after the month

    # ---- models -----------------------------------------------------------------
    random_state: int = 42
    tune: bool = True
    tune_iterations: int = 8
    tune_sample_rows: int = 600_000
    xgb_n_estimators: int = 4000
    xgb_learning_rate: float = 0.05
    xgb_early_stopping: int = 100
    xgb_params: dict = field(default_factory=lambda: {
        "max_depth": 8,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    })
    run_ablation: bool = True
    ablation_learning_rate: float = 0.1

    # ---- anomaly detection --------------------------------------------------
    anomaly_z_threshold: float = 3.0   # robust z-score on log residuals

    # ---- explainability -------------------------------------------------------
    run_shap: bool = True
    shap_sample_rows: int = 5000
    n_case_studies: int = 6

    # ---- derived paths ----------------------------------------------------------
    @property
    def price_paid_dir(self) -> Path:
        return Path(self.data_dir) / "price_paid"

    @property
    def onspd_file(self) -> Path:
        files = sorted((Path(self.data_dir) / "onspd" / "Data").glob("ONSPD_*_UK.csv"))
        if not files:
            raise FileNotFoundError(f"No ONSPD_*_UK.csv found in {Path(self.data_dir) / 'onspd' / 'Data'}")
        return files[-1]

    @property
    def hpi_file(self) -> Path:
        files = sorted((Path(self.data_dir) / "uk_hpi").glob("UK-HPI-full-file-*.csv"))
        if not files:
            raise FileNotFoundError(f"No UK-HPI-full-file-*.csv found in {Path(self.data_dir) / 'uk_hpi'}")
        return files[-1]

    @property
    def figures_dir(self) -> Path:
        return Path(self.output_dir) / "figures"

    @property
    def tables_dir(self) -> Path:
        return Path(self.output_dir) / "tables"

    @property
    def models_dir(self) -> Path:
        return Path(self.output_dir) / "models"

    @property
    def cache_dir(self) -> Path:
        return Path(self.output_dir) / "cache"

    def make_dirs(self) -> None:
        for d in (self.figures_dir, self.tables_dir, self.models_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = display_path(self.data_dir)
        d["output_dir"] = display_path(self.output_dir)
        return d

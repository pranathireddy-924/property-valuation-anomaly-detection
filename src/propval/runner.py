"""Helpers for running the pipeline from the dashboard.

The dashboard starts `run_pipeline.py` as a separate process, so a long run does not
block the web page. These functions check the data folder, build the command and
read the progress from the log file.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import CODE_ROOT

STAGES = [
    "1. Load and clean data",
    "2. Spatial and temporal feature engineering",
    "3a. Naive local baseline",
    "3b. Linear Regression baseline",
    "3c. XGBoost tuning and training",
    "4. Evaluation",
    "5. Feature ablation (hypothesis test)",
    "6. Price anomaly detection",
    "7. SHAP explanations",
    "8. Save prototype model",
    "9. Prepare dashboard data",
]

PRICE_PAID_RE = re.compile(r"^pp-(\d{4})\.csv$", re.I)
HPI_RE = re.compile(r"^UK-HPI-full-file-.*\.csv$", re.I)
ONSPD_RE = re.compile(r"^ONSPD_.*_UK\.csv$", re.I)


@dataclass
class DataCheck:
    price_paid_years: list[int] = field(default_factory=list)
    onspd_file: Path | None = None
    hpi_file: Path | None = None

    @property
    def ready(self) -> bool:
        return bool(self.price_paid_years) and self.onspd_file is not None and self.hpi_file is not None


def check_data_dir(data_dir: Path) -> DataCheck:
    """Find the files the pipeline needs inside a data folder."""
    data_dir = Path(data_dir)
    out = DataCheck()
    pp = data_dir / "price_paid"
    if pp.is_dir():
        out.price_paid_years = sorted(int(m.group(1)) for f in pp.iterdir() if (m := PRICE_PAID_RE.match(f.name)))
    onspd = sorted((data_dir / "onspd" / "Data").glob("ONSPD_*_UK.csv")) if (data_dir / "onspd" / "Data").is_dir() else []
    out.onspd_file = onspd[-1] if onspd else None
    hpi = sorted((data_dir / "uk_hpi").glob("UK-HPI-full-file-*.csv")) if (data_dir / "uk_hpi").is_dir() else []
    out.hpi_file = hpi[-1] if hpi else None
    return out


def target_folder(data_dir: Path, file_name: str) -> Path | None:
    """Where an uploaded file belongs inside the data folder (None if the name is not recognised)."""
    data_dir = Path(data_dir)
    if PRICE_PAID_RE.match(file_name):
        return data_dir / "price_paid"
    if HPI_RE.match(file_name):
        return data_dir / "uk_hpi"
    if ONSPD_RE.match(file_name):
        return data_dir / "onspd" / "Data"
    return None


@dataclass
class RunSettings:
    data_dir: Path
    output_dir: Path
    years: list[int]
    train_end: str
    val_end: str
    region: str | None = None
    sample_frac: float | None = None
    tune: bool = True
    ablation: bool = True
    shap: bool = True
    threshold: float = 3.0
    learning_rate: float | None = None
    use_cache: bool = True


def validate(s: RunSettings) -> list[str]:
    """Return a list of problems with the settings (empty list = fine)."""
    problems = []
    if not s.years:
        problems.append("Choose at least one Price Paid year.")
    if s.train_end >= s.val_end:
        problems.append("The training period must end before the validation period ends.")
    if s.years:
        if int(s.train_end[:4]) < min(s.years):
            problems.append("The training period ends before the first chosen year.")
        if s.val_end >= f"{max(s.years)}-12-31":
            problems.append("No test period is left: the validation period must end before the last chosen year ends.")
    if s.sample_frac is not None and not 0 < s.sample_frac <= 1:
        problems.append("The sample size must be between 1% and 100%.")
    return problems


def build_command(s: RunSettings) -> list[str]:
    """Command line that runs the pipeline with these settings."""
    cmd = [sys.executable, "-u", str(CODE_ROOT / "run_pipeline.py"),
           "--data-dir", str(s.data_dir), "--output-dir", str(s.output_dir),
           "--years", *map(str, s.years), "--train-end", s.train_end, "--val-end", s.val_end,
           "--threshold", str(s.threshold)]
    if s.region:
        cmd += ["--region", s.region]
    if s.sample_frac and s.sample_frac < 1:
        cmd += ["--sample-frac", str(s.sample_frac)]
    if s.learning_rate:
        cmd += ["--learning-rate", str(s.learning_rate)]
    if not s.tune:
        cmd.append("--no-tune")
    if not s.ablation:
        cmd.append("--no-ablation")
    if not s.shap:
        cmd.append("--no-shap")
    if not s.use_cache:
        cmd.append("--no-cache")
    return cmd


def read_progress(log_text: str) -> dict:
    """Which stages have started and finished, and whether the run failed or finished."""
    started = re.findall(r"=== (.+?) ===", log_text)
    done = re.findall(r"--- (.+?) done in ([\d.]+)s", log_text)
    return {
        "started": started,
        "done": {name: float(sec) for name, sec in done},
        "current": started[-1] if started and started[-1] not in dict(done) else None,
        "finished": "Pipeline finished" in log_text,
        "failed": "Traceback (most recent call last)" in log_text,
    }

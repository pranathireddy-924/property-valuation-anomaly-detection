"""Run the full dissertation pipeline.

Examples
--------
    python run_pipeline.py                      # full England & Wales run (tuning, ablation, SHAP)
    python run_pipeline.py --quick              # 5% sample, no tuning - checks everything works in minutes
    python run_pipeline.py --region E12000003   # Yorkshire and The Humber only
    python run_pipeline.py --no-cache           # re-read the raw CSV files
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from propval.config import REGION_NAMES, Config  # noqa: E402
from propval.pipeline import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="5%% sample, no tuning, fewer trees")
    p.add_argument("--sample-frac", type=float, help="Use a random fraction of the sales")
    p.add_argument("--region", choices=list(REGION_NAMES), help="Restrict to one region code")
    p.add_argument("--years", type=int, nargs="+", help="Price Paid years to use (default 2022-2026)")
    p.add_argument("--train-end", help="Last date of the training period (YYYY-MM-DD)")
    p.add_argument("--val-end", help="Last date of the validation period (YYYY-MM-DD)")
    p.add_argument("--no-tune", action="store_true", help="Skip the hyper-parameter search")
    p.add_argument("--no-ablation", action="store_true", help="Skip the feature ablation study")
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP explanations")
    p.add_argument("--threshold", type=float, help="Anomaly |z| threshold (default 3)")
    p.add_argument("--data-dir", help="Folder holding price_paid/, onspd/ and uk_hpi/")
    p.add_argument("--output-dir", help="Where to write outputs")
    p.add_argument("--no-cache", action="store_true", help="Ignore cached Parquet files")
    a = p.parse_args()

    cfg = Config()
    if a.quick:
        cfg.sample_frac, cfg.tune, cfg.xgb_learning_rate, cfg.output_dir = 0.05, False, 0.1, cfg.output_dir.parent / "outputs_quick"
    if a.sample_frac:
        cfg.sample_frac = a.sample_frac
    if a.region:
        cfg.region = a.region
    if a.years:
        cfg.years = tuple(sorted(a.years))
    if a.train_end:
        cfg.train_end = a.train_end
    if a.val_end:
        cfg.val_end = a.val_end
    if a.no_tune:
        cfg.tune = False
    if a.no_ablation:
        cfg.run_ablation = False
    if a.no_shap:
        cfg.run_shap = False
    if a.threshold:
        cfg.anomaly_z_threshold = a.threshold
    if a.data_dir:
        cfg.data_dir = Path(a.data_dir)
    if a.output_dir:
        cfg.output_dir = Path(a.output_dir)
    run(cfg, use_cache=not a.no_cache)


if __name__ == "__main__":
    main()

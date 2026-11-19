"""Small lookup files that make the dashboard understandable and explorable.

* la_names.csv              local authority code -> name (e.g. E08000039 -> Sheffield)
* postcodes_with_sales      postcodes that had real sales, with their area, so users can
                            browse Region -> Local authority -> District -> Postcode
* district_prices           median price and number of sales per postcode district,
                            property type and year (context for a valuation)
* flagged_rows              model inputs for the flagged test-period sales, so any
                            flagged sale can be explained on demand

Built from the cached base dataset, so it takes about a minute and does not
require re-training:  python prepare_dashboard.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Config, display_path

log = logging.getLogger(__name__)


def app_data_dir(cfg: Config) -> Path:
    d = Path(cfg.output_dir) / "app_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_la_names(cfg: Config) -> pd.DataFrame:
    files = sorted((Path(cfg.data_dir) / "onspd" / "Documents").glob("LAD Local Authority District names and codes*.csv"))
    if not files:
        raise FileNotFoundError("Local authority names file not found in datasets/onspd/Documents")
    raw = pd.read_csv(files[-1], encoding="utf-8-sig")
    code = next(c for c in raw.columns if c.upper().endswith("CD"))
    name = next(c for c in raw.columns if c.upper().endswith("NM"))
    return raw[[code, name]].set_axis(["lad", "la_name"], axis=1).drop_duplicates("lad")


def prepare(cfg: Config, base: pd.DataFrame | None = None) -> dict[str, Path]:
    out = app_data_dir(cfg)
    if base is None:
        key = f"base_{cfg.years[0]}_{cfg.years[-1]}_{cfg.region or 'EW'}_{cfg.sample_frac or 'all'}"
        base = pd.read_parquet(cfg.cache_dir / f"{key}.parquet")
    std = base.loc[base["is_standard"]]
    paths = {}

    try:
        la = build_la_names(cfg)
        paths["la_names"] = out / "la_names.csv"
        la.to_csv(paths["la_names"], index=False)
    except FileNotFoundError:
        log.warning("Local authority names file missing - dashboard will show codes")

    pcs = (std.groupby("postcode", observed=True)
           .agg(pc_district=("pc_district", "first"), lad=("lad", "first"), region=("region", "first"),
                n_sales=("price", "size"))
           .reset_index())
    paths["postcodes"] = out / "postcodes_with_sales.parquet"
    pcs.to_parquet(paths["postcodes"], index=False)

    dp = (std.groupby(["pc_district", "property_type", "sale_year"], observed=True)["price"]
          .agg(median_price="median", n_sales="size").reset_index())
    paths["district_prices"] = out / "district_prices.parquet"
    dp.to_parquet(paths["district_prices"], index=False)

    flagged_csv = cfg.tables_dir / "11_unusual_transactions_test.csv"
    if flagged_csv.exists():
        ids = set(pd.read_csv(flagged_csv, usecols=["transaction_id"])["transaction_id"])
        rows = std.loc[std["transaction_id"].isin(ids)]
        paths["flagged_rows"] = out / "flagged_rows.parquet"
        rows.to_parquet(paths["flagged_rows"], index=False)
    log.info("Dashboard data written to %s", display_path(out))
    return paths

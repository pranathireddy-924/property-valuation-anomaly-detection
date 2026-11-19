"""Shared fixtures: a small *synthetic* copy of the three datasets in their real raw
formats, so the whole pipeline can be tested quickly and without the 2 GB of real data.

The synthetic prices follow a known rule (type x location x time trend x noise), and
known anomalies are injected into the test period so detection can be measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from propval.config import PRICE_PAID_COLUMNS, Config  # noqa: E402

TYPE_BASE = {"D": 420_000, "S": 260_000, "T": 210_000, "F": 170_000}
DISTRICTS = {  # district -> (lad, region, country, location multiplier)
    "AA1": ("E06000001", "E12000001", "E92000001", 0.6),
    "AA2": ("E06000001", "E12000001", "E92000001", 0.8),
    "BB1": ("E09000001", "E12000007", "E92000001", 2.2),
    "BB2": ("E09000001", "E12000007", "E92000001", 1.7),
    "CC1": ("W06000001", "W99999999", "W92000004", 0.7),
    "CC2": ("W06000001", "W99999999", "W92000004", 0.9),
}
INJECTED_PREFIX = "{INJECTED"


def _postcodes() -> pd.DataFrame:
    rows, rng = [], np.random.default_rng(0)
    for d, (lad, rgn, ctry, mult) in DISTRICTS.items():
        for sector in range(1, 4):
            for unit in ("AB", "CD", "EF", "GH", "JK"):
                rows.append({
                    "pcds": f"{d} {sector}{unit}", "doterm": "", "lad26cd": lad, "rgn26cd": rgn, "ctry26cd": ctry,
                    "lsoa21cd": f"{lad[0]}0100{d}{sector}", "msoa21cd": f"{lad[0]}0200{d}",
                    "lat": 53.0 + rng.normal(0, 0.1), "long": -1.5 + rng.normal(0, 0.1),
                    "ruc21ind": "UN1" if mult > 1 else "RSN1", "imd25ind": int(rng.integers(1, 30000)),
                    "sector_mult": 1 + 0.1 * (sector - 2),
                })
    pcs = pd.DataFrame(rows)
    extra = pd.DataFrame([
        {**rows[0], "pcds": "AB1 0AA", "ctry26cd": "S92000003", "rgn26cd": "S99999999", "lad26cd": "S12000033"},
        {**rows[0], "pcds": "AA1 9ZZ", "lat": 99.999999, "long": 0.0},
    ])
    return pd.concat([pcs, extra], ignore_index=True)


def make_synthetic_data(root: Path, n_per_year: int = 3000) -> dict:
    rng = np.random.default_rng(42)
    pcs = _postcodes()
    (root / "onspd" / "Data").mkdir(parents=True)
    (root / "price_paid").mkdir()
    (root / "uk_hpi").mkdir()
    onspd = pcs.drop(columns="sector_mult").assign(oa21cd="X")  # extra column, as in the real file
    onspd.to_csv(root / "onspd" / "Data" / "ONSPD_AUG_2026_UK.csv", index=False)

    # ---- UK HPI: monthly prices per LA / region with a 0.4% monthly trend
    hpi_rows = []
    areas = {lad: m for (lad, _, _, m) in DISTRICTS.values()} | {"E12000001": 0.7, "E12000007": 1.9, "W92000004": 0.8}
    for month in pd.date_range("2020-01-01", "2024-12-01", freq="MS"):
        t = (month.year - 2022) * 12 + month.month - 1
        for code, m in areas.items():
            level = 1.004 ** t * m
            hpi_rows.append({"Date": month.strftime("%d/%m/%Y"), "RegionName": code, "AreaCode": code,
                             "AveragePrice": 250_000 * level, "12m%Change": 4.9,
                             **{col: TYPE_BASE[k] * level for k, col in
                                zip("DSTF", ["DetachedPrice", "SemiDetachedPrice", "TerracedPrice", "FlatPrice"])}})
    pd.DataFrame(hpi_rows).to_csv(root / "uk_hpi" / "UK-HPI-full-file-2024-12.csv", index=False)

    # ---- Price Paid: header-less, all 16 columns, quoted like the real files
    valid = pcs.loc[pcs["ctry26cd"].isin(["E92000001", "W92000004"]) & pcs["lat"].lt(90)].reset_index(drop=True)
    injected = []
    for year in (2022, 2023, 2024):
        n = n_per_year
        p = valid.sample(n, replace=True, random_state=year).reset_index(drop=True)
        ptype = rng.choice(list("DSTF"), n)
        days = rng.integers(0, 365, n)
        date = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(days, unit="D")
        t = (date.year - 2022) * 12 + date.month - 1
        loc = p["pcds"].str.split(" ").str[0].map({d: v[3] for d, v in DISTRICTS.items()}).to_numpy()
        # non-linear interaction: flats are relatively cheaper outside the expensive districts
        inter = np.where((ptype == "F") & (loc < 1), 0.75, 1.0)
        new = rng.random(n) < 0.1
        price = (np.array([TYPE_BASE[x] for x in ptype]) * loc * p["sector_mult"].to_numpy() * inter
                 * 1.004 ** np.asarray(t) * np.where(new, 1.15, 1.0) * np.exp(rng.normal(0, 0.08, n)))
        cat = np.where(rng.random(n) < 0.05, "B", "A")
        price = np.where(cat == "B", price * 0.55, price)
        ids = [f"{{SYN-{year}-{i:06d}}}" for i in range(n)]
        if year == 2024:  # inject 20 clear anomalies into the test period
            for j in range(20):
                ids[j] = f"{INJECTED_PREFIX}-{j:02d}}}"
                price[j] = price[j] * (4 if j % 2 == 0 else 0.25)
                cat[j] = "A"
                injected.append(ids[j])
        df = pd.DataFrame({
            "transaction_id": ids, "price": np.round(price).astype(int),
            "date_of_transfer": date.strftime("%Y-%m-%d 00:00"), "postcode": p["pcds"],
            "property_type": ptype, "old_new": np.where(new, "Y", "N"),
            "duration": np.where(ptype == "F", "L", "F"), "paon": "12", "saon": "", "street": "HIGH STREET",
            "locality": "", "town_city": "TOWN", "district": "DISTRICT", "county": "COUNTY",
            "ppd_category": cat, "record_status": "A",
        })
        if year == 2022:  # rows each cleaning rule must remove
            bad = df.iloc[:5].copy()
            bad["transaction_id"] = [f"{{BAD-{i}}}" for i in range(5)]
            bad.loc[bad.index[0], "property_type"] = "O"
            bad.loc[bad.index[1], "price"] = 500
            bad.loc[bad.index[2], "postcode"] = ""
            bad.loc[bad.index[3], "postcode"] = "ZZ9 9ZZ"      # not in ONSPD
            bad.loc[bad.index[4], "record_status"] = "D"
            df = pd.concat([df, bad, df.iloc[[10]]], ignore_index=True)  # + one duplicate
        df[PRICE_PAID_COLUMNS].to_csv(root / "price_paid" / f"pp-{year}.csv", header=False, index=False,
                                       quoting=1)
    return {"injected_ids": injected}


def synthetic_config(data_dir: Path, out_dir: Path) -> Config:
    cfg = Config(data_dir=data_dir, output_dir=out_dir)
    cfg.years = (2022, 2023, 2024)
    cfg.train_end, cfg.val_end = "2023-06-30", "2023-12-31"
    cfg.te_folds = 3
    cfg.tune_iterations, cfg.tune_sample_rows = 2, 2000
    cfg.xgb_n_estimators, cfg.xgb_learning_rate, cfg.xgb_early_stopping = 400, 0.1, 30
    cfg.xgb_params = {**cfg.xgb_params, "max_depth": 5, "min_child_weight": 5}
    cfg.shap_sample_rows, cfg.n_case_studies = 300, 3
    return cfg


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("data")
    info = make_synthetic_data(root)
    return {"data_dir": root, **info}


@pytest.fixture(scope="session")
def pipeline_run(synthetic, tmp_path_factory) -> dict:
    from propval.pipeline import run
    cfg = synthetic_config(synthetic["data_dir"], tmp_path_factory.mktemp("outputs"))
    result = run(cfg, use_cache=False)
    return {"cfg": cfg, "result": result, **synthetic}

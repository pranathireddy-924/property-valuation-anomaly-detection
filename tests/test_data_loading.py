"""Unit tests - data loading and cleaning (test cases UT-01 to UT-06)."""
import numpy as np
import pandas as pd

from propval.config import Config
from propval.data_loading import (clean_price_paid, load_hpi, load_postcode_lookup, normalise_postcode,
                                  postcode_parts, read_price_paid_files)


def test_ut01_normalise_postcode():
    s = pd.Series(["s1 2bj", "SW1A1AA", " e1   2fq ", "XX", None])
    out = normalise_postcode(s).tolist()
    assert out[:3] == ["S1 2BJ", "SW1A 1AA", "E1 2FQ"]
    assert pd.isna(out[3]) and pd.isna(out[4])


def test_ut02_postcode_parts():
    parts = postcode_parts(pd.Series(["SW1A 1AA", "S1 2BJ"]))
    assert parts["pc_area"].tolist() == ["SW", "S"]
    assert parts["pc_district"].tolist() == ["SW1A", "S1"]
    assert parts["pc_sector"].tolist() == ["SW1A 1", "S1 2"]


def test_ut03_cleaning_rules_remove_invalid_rows(synthetic):
    cfg = Config(data_dir=synthetic["data_dir"])
    cfg.years = (2022, 2023, 2024)
    raw = read_price_paid_files(cfg)
    clean, report = clean_price_paid(raw, cfg)
    removed = {r["step"]: r["rows_removed"] for r in report}
    assert removed["Record status"] == 1
    assert removed["Duplicates"] == 1
    assert removed["Missing postcode"] == 1
    assert removed["Property type 'Other'"] == 1
    assert removed["Price range"] == 1
    assert set(clean["property_type"]) <= {"D", "S", "T", "F"}
    assert clean["price"].between(cfg.min_price, cfg.max_price).all()


def test_ut04_address_fields_are_not_loaded(synthetic):
    """Privacy by design: house number, flat and street are never read into memory."""
    cfg = Config(data_dir=synthetic["data_dir"])
    cfg.years = (2022,)
    raw = read_price_paid_files(cfg)
    assert not {"paon", "saon", "street", "locality", "town_city"} & set(raw.columns)


def test_ut05_postcode_lookup(synthetic):
    path = synthetic["data_dir"] / "onspd" / "Data" / "ONSPD_AUG_2026_UK.csv"
    lk = load_postcode_lookup(path)
    assert "AB1 0AA" not in set(lk["pcds"])                      # Scotland removed
    assert set(lk.loc[lk["country"] == "W92000004", "region"]) == {"W92000004"}  # Wales region fixed
    assert lk.loc[lk["pcds"] == "AA1 9ZZ", "lat"].isna().all()   # 99.999999 = no grid ref
    assert lk["imd_pct"].dropna().between(0, 1).all()


def test_ut06_hpi_month_index(synthetic):
    hpi = load_hpi(next((synthetic["data_dir"] / "uk_hpi").glob("*.csv")))
    assert hpi["month_idx"].min() == 2020 * 12          # January 2020
    assert not hpi.duplicated(["area_code", "month_idx"]).any()
    assert np.isfinite(hpi["avg_price"]).all()

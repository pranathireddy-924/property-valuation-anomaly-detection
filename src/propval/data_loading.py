"""Loading and cleaning of the three secondary datasets.

* HM Land Registry Price Paid Data (main dataset, one CSV per year, no header row)
* ONS Postcode Directory (postcode -> statistical geography lookup)
* UK House Price Index full file (monthly local-authority price levels)

Every cleaning rule records how many rows it removed, so the data-quality steps can
be reported in the dissertation (Risk R1/R2 in the risk register).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, PRICE_PAID_COLUMNS, PRICE_PAID_KEEP

log = logging.getLogger(__name__)

ENGLAND, WALES = "E92000001", "W92000004"


# ---------------------------------------------------------------------------
# Postcode helpers
# ---------------------------------------------------------------------------
def normalise_postcode(pc: pd.Series) -> pd.Series:
    """Upper-case and reduce to the ONS 'pcds' format: outward + single space + inward."""
    s = pc.astype("string").str.upper().str.replace(r"\s+", "", regex=True)
    s = s.where(s.str.len().between(5, 7))
    return s.str[:-3] + " " + s.str[-3:]


def postcode_parts(pcds: pd.Series) -> pd.DataFrame:
    """Split a normalised postcode into area (e.g. 'S'), district ('S1') and sector ('S1 2')."""
    outward = pcds.str.split(" ").str[0]
    return pd.DataFrame({
        "pc_area": outward.str.extract(r"^([A-Z]{1,2})", expand=False),
        "pc_district": outward,
        "pc_sector": outward + " " + pcds.str.split(" ").str[1].str[0],
    }, index=pcds.index)


# ---------------------------------------------------------------------------
# HM Land Registry Price Paid Data
# ---------------------------------------------------------------------------
def read_price_paid_files(cfg: Config) -> pd.DataFrame:
    frames = []
    for year in cfg.years:
        path = Path(cfg.price_paid_dir) / f"pp-{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing Price Paid file: {path}")
        log.info("Reading %s", path.name)
        frames.append(pd.read_csv(
            path, header=None, names=PRICE_PAID_COLUMNS, usecols=PRICE_PAID_KEEP,
            dtype={c: "string" for c in PRICE_PAID_KEEP if c != "price"} | {"price": "int64"},
        ))
    return pd.concat(frames, ignore_index=True)


def clean_price_paid(raw: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the documented cleaning rules and return (clean data, cleaning report).

    Category B transactions (repossessions, buy-to-let, transfers to companies) are
    *kept* here with a flag: they are excluded from model training but used later as
    a weak external check of the anomaly detector.
    """
    report: list[dict] = []
    df = raw

    def step(name: str, mask: pd.Series | None, reason: str) -> None:
        nonlocal df
        before = len(df)
        if mask is not None:
            df = df.loc[mask]
        report.append({"step": name, "rows_before": before, "rows_removed": before - len(df),
                       "rows_after": len(df), "reason": reason})

    step("Raw records", None, f"Price Paid yearly files {cfg.years[0]}-{cfg.years[-1]}")
    step("Record status", df["record_status"].eq("A"), "Keep additions only (drop changes/deletions)")
    step("Duplicates", ~df["transaction_id"].duplicated(), "Remove duplicated transaction IDs")

    df = df.assign(postcode=normalise_postcode(df["postcode"]))
    step("Missing postcode", df["postcode"].notna(), "Postcode needed for spatial features")
    step("Property type 'Other'", df["property_type"].isin(["D", "S", "T", "F"]),
         "Type O is non-residential / mixed use and not comparable")
    step("Price range", df["price"].between(cfg.min_price, cfg.max_price),
         f"Keep £{cfg.min_price:,} - £{cfg.max_price:,} (sanity limits)")

    df = df.assign(
        date=pd.to_datetime(df["date_of_transfer"].str[:10], format="%Y-%m-%d"),
        is_standard=df["ppd_category"].eq("A"),
    ).drop(columns=["date_of_transfer", "record_status"])
    df = df.loc[df["date"].dt.year.isin(cfg.years)]

    if cfg.sample_frac:
        before = len(df)
        df = df.sample(frac=cfg.sample_frac, random_state=cfg.random_state)
        report.append({"step": "Random sample", "rows_before": before, "rows_removed": before - len(df),
                       "rows_after": len(df), "reason": f"sample_frac={cfg.sample_frac} (quick run)"})
    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# ONS Postcode Directory
# ---------------------------------------------------------------------------
def _find_col(columns: list[str], pattern: str) -> str:
    """ONSPD column names carry a year suffix (lad26cd, lad27cd ...) - find the current one."""
    matches = [c for c in columns if re.fullmatch(pattern, c)]
    if not matches:
        raise KeyError(f"No ONSPD column matching {pattern!r}")
    return sorted(matches)[-1]


def load_postcode_lookup(path: Path) -> pd.DataFrame:
    """Slim England & Wales postcode lookup with the spatial attributes used as features."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = {
        "pcds": "pcds",
        _find_col(header, r"lad\d\dcd"): "lad",
        _find_col(header, r"rgn\d\dcd"): "region",
        _find_col(header, r"ctry\d\dcd"): "country",
        "lsoa21cd": "lsoa21",
        "msoa21cd": "msoa21",
        "lat": "lat",
        "long": "long",
        "ruc21ind": "ruc21",
        _find_col(header, r"imd\d\dind"): "imd_rank",
    }
    log.info("Reading ONS Postcode Directory %s", Path(path).name)
    lk = pd.read_csv(path, usecols=list(cols), dtype="string").rename(columns=cols)
    lk = lk.loc[lk["country"].isin([ENGLAND, WALES])].copy()

    # Wales has no English region code in ONSPD (W99999999) - use the country code.
    lk["region"] = lk["region"].mask(lk["country"].eq(WALES), WALES)
    lk["lat"] = pd.to_numeric(lk["lat"], errors="coerce")
    lk["long"] = pd.to_numeric(lk["long"], errors="coerce")
    no_grid = lk["lat"].gt(90)               # ONSPD uses 99.999999 for 'no grid reference'
    lk.loc[no_grid, ["lat", "long"]] = np.nan

    # Deprivation rank: England and Wales use separate rankings, so convert to a
    # within-country percentile (0 = most deprived, 1 = least deprived).
    rank = pd.to_numeric(lk["imd_rank"], errors="coerce").replace(0, np.nan)
    lk["imd_pct"] = rank / rank.groupby(lk["country"]).transform("max")
    lk["ruc21"] = lk["ruc21"].replace({"": pd.NA})

    lk = lk.drop(columns=["imd_rank"]).drop_duplicates("pcds").reset_index(drop=True)
    return lk


# ---------------------------------------------------------------------------
# UK House Price Index
# ---------------------------------------------------------------------------
HPI_TYPE_COLUMNS = {"D": "DetachedPrice", "S": "SemiDetachedPrice", "T": "TerracedPrice", "F": "FlatPrice"}


def load_hpi(path: Path) -> pd.DataFrame:
    """Monthly HPI levels per area code, keyed by an integer month index (year*12 + month-1)."""
    cols = ["Date", "AreaCode", "AveragePrice", "12m%Change", *HPI_TYPE_COLUMNS.values()]
    log.info("Reading UK HPI %s", Path(path).name)
    hpi = pd.read_csv(path, usecols=cols)
    date = pd.to_datetime(hpi["Date"], format="%d/%m/%Y")
    hpi = hpi.assign(month_idx=date.dt.year * 12 + date.dt.month - 1).drop(columns="Date")
    hpi = hpi.rename(columns={"AreaCode": "area_code", "AveragePrice": "avg_price", "12m%Change": "change_12m"})
    return hpi.drop_duplicates(["area_code", "month_idx"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Joining transactions to geography
# ---------------------------------------------------------------------------
def attach_geography(df: pd.DataFrame, lookup: pd.DataFrame, cfg: Config,
                     report: list[dict]) -> pd.DataFrame:
    """Join each sale to its ONSPD geography and add postcode-derived and date features."""
    before = len(df)
    out = df.merge(lookup, left_on="postcode", right_on="pcds", how="inner").drop(columns="pcds")
    report.append({"step": "Postcode match (ONSPD)", "rows_before": before, "rows_removed": before - len(out),
                   "rows_after": len(out), "reason": "Remove sales whose postcode is not in ONSPD (Risk R2)"})
    if cfg.region:
        before = len(out)
        out = out.loc[out["region"].eq(cfg.region)]
        report.append({"step": "Region filter", "rows_before": before, "rows_removed": before - len(out),
                       "rows_after": len(out), "reason": f"Keep region {cfg.region}"})

    out = add_derived_columns(out.reset_index(drop=True))
    out["log_price"] = np.log(out["price"].astype("float64"))
    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Postcode parts and calendar fields (shared by the pipeline and the prototype app)."""
    out = pd.concat([df, postcode_parts(df["postcode"])], axis=1)
    out["sale_year"] = out["date"].dt.year.astype("int16")
    out["sale_month"] = out["date"].dt.month.astype("int8")
    out["month_idx"] = (out["sale_year"].astype("int32") * 12 + out["sale_month"] - 1).astype("int32")
    return out


def load_base_dataset(cfg: Config, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (clean transactions with geography, postcode lookup, cleaning report).

    The joined dataset is cached as Parquet so later runs skip the slow CSV parsing.
    """
    cfg.make_dirs()
    key = f"base_{cfg.years[0]}_{cfg.years[-1]}_{cfg.region or 'EW'}_{cfg.sample_frac or 'all'}"
    base_path = cfg.cache_dir / f"{key}.parquet"
    report_path = cfg.cache_dir / f"{key}_cleaning.csv"
    lookup_path = cfg.cache_dir / "postcode_lookup.parquet"

    if lookup_path.exists() and use_cache:
        lookup = pd.read_parquet(lookup_path)
    else:
        lookup = load_postcode_lookup(cfg.onspd_file)
        lookup.to_parquet(lookup_path, index=False)

    if use_cache and base_path.exists() and report_path.exists():
        log.info("Using cached base dataset %s", base_path.name)
        return pd.read_parquet(base_path), lookup, pd.read_csv(report_path)

    raw = read_price_paid_files(cfg)
    df, report = clean_price_paid(raw, cfg)
    del raw
    df = attach_geography(df, lookup, cfg, report)
    report_df = pd.DataFrame(report)
    df.to_parquet(base_path, index=False)
    report_df.to_csv(report_path, index=False)
    return df, lookup, report_df

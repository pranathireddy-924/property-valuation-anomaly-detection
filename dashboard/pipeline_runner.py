"""Run Pipeline: run the full pipeline from the browser, on the current data or on new data."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.common import OUT, ROOT, callout, gbp, page_header, text_table
from propval.config import DEFAULT_DATA_DIR, REGION_NAMES
from propval.runner import STAGES, RunSettings, build_command, check_data_dir, read_progress, target_folder, validate

RUNS_DIR = ROOT / "outputs_runs"
# One pipeline run at a time for the whole app. A module-level dict survives page reruns
# (and is not wiped when the dashboard caches are cleared).
JOB: dict = {}

STAGE_INFO = {
    "1. Load and clean data": "Read the Price Paid files, apply the cleaning rules and link each sale to its postcode area.",
    "2. Spatial and temporal feature engineering": "Build the 23 features: property, time of sale, location and local price levels, House Price Index.",
    "3a. Naive local baseline": "Simple benchmark: average price of the same property type in the same postcode district.",
    "3b. Linear Regression baseline": "Train the Linear Regression model.",
    "3c. XGBoost tuning and training": "Try several XGBoost settings on the validation period, then train the final model.",
    "4. Evaluation": "Score all models on the test period (MAE, RMSE, R² and more).",
    "5. Feature ablation (hypothesis test)": "Retrain with one feature group at a time to see which groups matter.",
    "6. Price anomaly detection": "Give every test sale an anomaly score and flag the unusual ones.",
    "7. SHAP explanations": "Work out which factors drive the predictions.",
    "8. Save prototype model": "Save the trained model used by the Valuation Tool.",
    "9. Prepare dashboard data": "Build the lookup files used by the dashboard pages.",
}


def _running() -> bool:
    return bool(JOB) and JOB["proc"].poll() is None


def _free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


# --------------------------------------------------------------------------- step 1: data
def _data_section() -> tuple[Path, object]:
    st.header("Step 1. Data")
    st.write("The pipeline needs three files from official sources. Put them in a data folder with this layout. "
             "To use newer data (for example a new year of sales or a newer postcode directory), add the files "
             "here; the newest postcode directory and House Price Index file are picked automatically.")
    text_table(pd.DataFrame([
        ("price_paid/pp-YYYY.csv", "HM Land Registry Price Paid Data, one file per year",
         "gov.uk/government/statistical-data-sets/price-paid-data-downloads"),
        ("onspd/Data/ONSPD_<MONTH>_<YEAR>_UK.csv", "ONS Postcode Directory (postcode to area lookup)",
         "ons.gov.uk/methodology/geography/geographicalproducts/postcodeproducts"),
        ("uk_hpi/UK-HPI-full-file-<YYYY-MM>.csv", "UK House Price Index full file",
         "gov.uk/government/collections/uk-house-price-index-reports"),
    ], columns=["File (inside the data folder)", "What it is", "Download from"]))

    data_dir = Path(st.text_input("Data folder", str(DEFAULT_DATA_DIR),
                                  help="Folder that contains price_paid, onspd and uk_hpi"))
    check = check_data_dir(data_dir)
    rows = [
        ("Price Paid", ", ".join(map(str, check.price_paid_years)) if check.price_paid_years else "Not found"),
        ("Postcode Directory", check.onspd_file.name if check.onspd_file else "Not found"),
        ("House Price Index", check.hpi_file.name if check.hpi_file else "Not found"),
    ]
    text_table(pd.DataFrame(rows, columns=["Dataset", "Found in the data folder"]))
    if not check.ready:
        callout("Some files are missing. Add them to the data folder (or upload them below) before running.", warn=True)

    with st.expander("Upload new data files"):
        st.write("Upload Price Paid files (pp-YYYY.csv) or a House Price Index full file. They are saved into the "
                 "right sub-folder automatically. The Postcode Directory is too large to upload here (about 1.5 GB): "
                 "copy it into onspd/Data yourself.")
        files = st.file_uploader("Data files", type="csv", accept_multiple_files=True)
        if files and st.button("Save uploaded files"):
            for f in files:
                folder = target_folder(data_dir, f.name)
                if folder is None:
                    st.error(f"{f.name}: name not recognised. Expected pp-YYYY.csv or UK-HPI-full-file-YYYY-MM.csv.")
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f.name).write_bytes(f.getbuffer())
                st.success(f"Saved {f.name} to {folder}")
            st.rerun()
    return data_dir, check


# --------------------------------------------------------------------------- step 2: settings
def _settings_section(data_dir: Path, check) -> RunSettings | None:
    st.header("Step 2. Settings")
    years_found = check.price_paid_years or [2022, 2023, 2024, 2025, 2026]
    c = st.columns(2)
    run_type = c[0].radio("Type of run", ["Quick check", "Full run"], horizontal=True,
                          help="Quick check: 5% sample, no tuning, faster training (a few minutes). "
                               "Full run: all sales with tuning (can take several hours).")
    region_name = c[1].selectbox("Area", ["England & Wales (all)"] + list(REGION_NAMES.values()))
    region = {v: k for k, v in REGION_NAMES.items()}.get(region_name)

    years = st.multiselect("Price Paid years to use", years_found, default=years_found)
    ys = sorted(years) or years_found
    c = st.columns(2)
    train_end = c[0].date_input("Training period ends", date(ys[-3] if len(ys) >= 3 else ys[0], 12, 31),
                                format="DD/MM/YYYY", help="Models learn from sales up to this date.")
    val_end = c[1].date_input("Validation period ends", date(ys[-2] if len(ys) >= 2 else ys[0], 12, 31),
                              format="DD/MM/YYYY",
                              help="Sales up to this date are used for tuning. Later sales form the test period.")
    st.caption(f"Train: {ys[0]} to {train_end:%d %b %Y} · Validation: to {val_end:%d %b %Y} · "
               f"Test: after {val_end:%d %b %Y} to the end of {ys[-1]}")

    quick = run_type == "Quick check"
    with st.expander("Advanced settings"):
        a = st.columns(3)
        sample_pct = a[0].number_input("Sample of sales (%)", 1, 100, 5 if quick else 100)
        threshold = a[1].number_input("Anomaly threshold (score)", 2.0, 6.0, 3.0, 0.5,
                                      help="Sales with an anomaly score beyond ± this value are flagged.")
        use_cache = a[2].checkbox("Reuse cached data", True,
                                  help="Untick after replacing a data file with the same name, so it is read again.")
        b = st.columns(3)
        tune = b[0].checkbox("Tune XGBoost settings", not quick)
        ablation = b[1].checkbox("Feature ablation test", True)
        shap = b[2].checkbox("SHAP explanations", True)

    st.header("Step 3. Where to save the results")
    target = st.radio("Results folder", ["New folder (keeps the current dashboard results)",
                                         "Replace the dashboard results"],
                      help="With a new folder you can look at the results first, then decide to use them.")
    if target.startswith("New"):
        output_dir = RUNS_DIR / f"run_{datetime.now():%Y%m%d_%H%M}"
    else:
        output_dir = OUT
    st.caption(f"Results will be saved in: `{output_dir}`")

    s = RunSettings(data_dir=data_dir, output_dir=output_dir, years=sorted(years),
                    train_end=f"{train_end:%Y-%m-%d}", val_end=f"{val_end:%Y-%m-%d}", region=region,
                    sample_frac=sample_pct / 100 if sample_pct < 100 else None, tune=tune, ablation=ablation,
                    shap=shap, threshold=float(threshold), learning_rate=0.1 if quick else None, use_cache=use_cache)
    return s


# --------------------------------------------------------------------------- step 4: run
def _start(s: RunSettings) -> None:
    Path(s.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(s.output_dir) / "pipeline_console.log"
    log_file = open(log_path, "w", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(build_command(s), stdout=log_file, stderr=subprocess.STDOUT, cwd=str(ROOT),
                            creationflags=flags, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    JOB.clear()
    JOB.update(proc=proc, log_path=log_path, log_file=log_file, settings=s, started=time.time())


def _status_table(prog: dict, s: RunSettings) -> pd.DataFrame:
    rows = []
    for name in STAGES:
        if name in prog["done"]:
            status = f"Done ({prog['done'][name]:.0f} s)"
        elif name == prog["current"]:
            status = "Failed" if prog["failed"] else "Running..."
        elif (name.startswith("5") and not s.ablation) or (name.startswith("7") and not s.shap):
            status = "Skipped"
        else:
            status = "Waiting"
        rows.append((name.replace(". ", ": ", 1), STAGE_INFO[name], status))   # "1: ..." avoids a markdown list
    return pd.DataFrame(rows, columns=["Stage", "What it does", "Status"])


@st.fragment(run_every=3)
def _live_status() -> None:
    if not JOB:
        return
    s: RunSettings = JOB["settings"]
    text = JOB["log_path"].read_text(encoding="utf-8", errors="replace") if JOB["log_path"].exists() else ""
    prog = read_progress(text)
    running = _running()
    code = JOB["proc"].poll()
    if not running and "ended" not in JOB:
        JOB["ended"] = time.time()
    mins = (JOB.get("ended", time.time()) - JOB["started"]) / 60

    done_n = len(prog["done"])
    st.progress(min(done_n / len(STAGES), 1.0), text=f"{done_n} of {len(STAGES)} stages complete · {mins:.1f} min")
    text_table(_status_table(prog, s))
    with st.expander("Log (last 40 lines)"):
        st.code("\n".join(text.splitlines()[-40:]) or "Starting...", language="text")

    if running:
        if st.button("Stop the run"):
            JOB["proc"].terminate()
            st.rerun()
        return
    JOB["log_file"].close()
    if not JOB.get("seen_end"):   # first refresh after the run ended: reload the whole page once
        JOB["seen_end"] = True
        if code == 0 and Path(s.output_dir).resolve() == OUT.resolve():
            st.cache_data.clear()
            st.cache_resource.clear()
        st.rerun(scope="app")
    if code == 0 and prog["finished"]:
        st.success(f"Finished in {mins:.1f} minutes. Results saved in `{s.output_dir}`")
        _show_run_results(Path(s.output_dir))
    else:
        st.error("The run stopped before it finished. The last lines of the log above show the reason.")


def _show_run_results(out: Path) -> None:
    cmp_ = out / "tables" / "05_model_comparison.csv"
    if cmp_.exists():
        d = pd.read_csv(cmp_)
        d = d[d["period"] == "test"]
        st.subheader("Results of this run (test period)")
        text_table(pd.DataFrame({"Model": d["model"], "Sales": d["n"].map("{:,}".format), "MAE": d["MAE"].map(gbp),
                                 "RMSE": d["RMSE"].map(gbp), "R²": d["R2"].map("{:.3f}".format),
                                 "Typical error (MdAPE)": d["MdAPE_%"].map("{:.1f}%".format)}))
    rates = out / "tables" / "14_anomaly_check_category_b.csv"
    if rates.exists():
        r = pd.read_csv(rates)
        st.write(f"Sales flagged as unusual: **{r.iloc[0]['flag_rate_%']:.1f}%** of standard sales.")
    if Path(out).resolve() != OUT.resolve():
        st.write("These results are in a separate folder. To show them on all dashboard pages, copy them over "
                 "the current results:")
        if st.button("Use these results in the dashboard", type="primary"):
            _use_results(out)
    else:
        st.info("The dashboard pages now show the new results.")


def _use_results(src: Path) -> None:
    for name in ("tables", "figures", "models", "app_data"):
        if (src / name).exists():
            shutil.rmtree(OUT / name, ignore_errors=True)
            shutil.copytree(src / name, OUT / name)
    for name in ("RESULTS.md", "run_config.json", "pipeline.log"):
        if (src / name).exists():
            shutil.copy(src / name, OUT / name)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Done. All dashboard pages now show the results of this run.")


# --------------------------------------------------------------------------- page
def render() -> None:
    page_header("Run Pipeline",
                "Run the whole pipeline again from the browser: clean the data, build the features, train and "
                "compare the models, score unusual sales, explain the model and refresh the dashboard. Use it "
                "with newer data (for example next year's sales) or with different settings, such as one "
                "region or other split dates. The steps are the same as in the Data & Method page.",
                "reusability of the approach (Objective 1: a reusable dataset for selected years and regions)")

    if _running():
        st.header("Run in progress")
        _live_status()
        return

    data_dir, check = _data_section()
    s = _settings_section(data_dir, check)

    st.header("Step 4. Run")
    problems = validate(s)
    if not check.ready:
        problems.append("Some data files are missing (see Step 1).")
    need = 1.0 if s.sample_frac else 3.0
    free = _free_gb(s.output_dir)
    if free < need:
        problems.append(f"Only {free:.1f} GB of disk space is free; this run needs about {need:.0f} GB.")
    for p in problems:
        st.error(p)
    with st.expander("Command that will be run"):
        st.code(" ".join(f'"{c}"' if " " in c else c for c in build_command(s)), language="bash")
    if st.button("Run pipeline", type="primary", disabled=bool(problems)):
        _start(s)
        st.rerun()

    if JOB:   # a finished run from earlier in this session
        st.header("Last run")
        _live_status()

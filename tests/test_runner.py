"""Unit tests for the dashboard pipeline runner helpers (UT-23 to UT-26)."""
from pathlib import Path

from propval.runner import RunSettings, build_command, check_data_dir, read_progress, target_folder, validate


def _settings(tmp_path, **kw):
    base = dict(data_dir=tmp_path, output_dir=tmp_path / "out", years=[2022, 2023, 2024, 2025, 2026],
                train_end="2024-12-31", val_end="2025-12-31")
    base.update(kw)
    return RunSettings(**base)


def test_ut23_check_data_dir_finds_files(tmp_path):
    (tmp_path / "price_paid").mkdir()
    for y in (2023, 2022):
        (tmp_path / "price_paid" / f"pp-{y}.csv").write_text("")
    (tmp_path / "onspd" / "Data").mkdir(parents=True)
    (tmp_path / "onspd" / "Data" / "ONSPD_AUG_2026_UK.csv").write_text("")
    check = check_data_dir(tmp_path)
    assert check.price_paid_years == [2022, 2023]
    assert check.onspd_file.name == "ONSPD_AUG_2026_UK.csv"
    assert check.hpi_file is None and not check.ready


def test_ut24_uploaded_files_go_to_the_right_folder(tmp_path):
    assert target_folder(tmp_path, "pp-2027.csv") == tmp_path / "price_paid"
    assert target_folder(tmp_path, "UK-HPI-full-file-2027-03.csv") == tmp_path / "uk_hpi"
    assert target_folder(tmp_path, "ONSPD_FEB_2027_UK.csv") == tmp_path / "onspd" / "Data"
    assert target_folder(tmp_path, "notes.csv") is None


def test_ut25_settings_are_validated_and_turned_into_a_command(tmp_path):
    assert validate(_settings(tmp_path)) == []
    assert validate(_settings(tmp_path, train_end="2026-01-31"))                 # train ends after validation
    assert validate(_settings(tmp_path, val_end="2026-12-31"))                   # no test period left
    cmd = build_command(_settings(tmp_path, region="W92000004", sample_frac=0.05, tune=False, shap=False))
    assert Path(cmd[2]).name == "run_pipeline.py"
    for part in ("--region", "W92000004", "--sample-frac", "0.05", "--no-tune", "--no-shap"):
        assert part in cmd
    assert "--no-ablation" not in cmd


def test_ut26_progress_is_read_from_the_log():
    log = ("=== 1. Load and clean data ===\n--- 1. Load and clean data done in 12.5s\n"
           "=== 2. Spatial and temporal feature engineering ===\n")
    p = read_progress(log)
    assert p["done"] == {"1. Load and clean data": 12.5}
    assert p["current"] == "2. Spatial and temporal feature engineering"
    assert not p["finished"] and not p["failed"]
    assert read_progress(log + "Traceback (most recent call last):\n")["failed"]

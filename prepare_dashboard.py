"""Build the dashboard lookup files from the cached data (no re-training needed).

    python prepare_dashboard.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from propval.config import Config  # noqa: E402
from propval.dashboard_data import prepare  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name, path in prepare(Config()).items():
        print(f"{name:16s} {path}")

"""Value one property from the command line with the trained model.

    python value_property.py --postcode "S1 2BJ" --type F --tenure L --date 2026-06-15 --price 180000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from propval.predict import main  # noqa: E402

if __name__ == "__main__":
    main()

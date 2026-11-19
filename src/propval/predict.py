"""Prototype valuation service: value one property and explain the result.

Used by the Streamlit app (app.py) and the command line:

    python value_property.py --postcode "S1 2BJ" --type F --tenure L --date 2026-06-15 --price 180000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import CODE_ROOT, PROPERTY_TYPE_NAMES, REGION_NAMES
from .data_loading import add_derived_columns, normalise_postcode
from .explain import explain_row, set_la_names, shap_values
from .models import predict_xgb

DEFAULT_MODEL_DIR = CODE_ROOT / "outputs" / "models"


class PropertyValuer:
    def __init__(self, bundle: dict, lookup: pd.DataFrame):
        self.model = bundle["model"]
        self.builder = bundle["builder"]
        self.detector = bundle["detector"]
        self.metrics = bundle.get("metrics", {})
        self.lookup = lookup.set_index("pcds")

    @classmethod
    def load(cls, model_dir: Path = DEFAULT_MODEL_DIR) -> "PropertyValuer":
        model_dir = Path(model_dir)
        bundle = joblib.load(model_dir / "valuer_bundle.joblib")
        lookup = pd.read_parquet(model_dir / "postcode_lookup.parquet")
        la_file = model_dir.parent / "app_data" / "la_names.csv"
        if la_file.exists():   # show "Sheffield" rather than "E08000039" in explanations
            la = pd.read_csv(la_file)
            set_la_names(dict(zip(la["lad"], la["la_name"])))
        return cls(bundle, lookup)

    def value(self, postcode: str, property_type: str, new_build: bool, tenure: str,
              sale_date: str | pd.Timestamp, actual_price: float | None = None, top_n: int = 8) -> dict:
        if property_type not in PROPERTY_TYPE_NAMES:
            raise ValueError(f"property_type must be one of {list(PROPERTY_TYPE_NAMES)}")
        if tenure not in ("F", "L"):
            raise ValueError("tenure must be 'F' (freehold) or 'L' (leasehold)")
        pc = normalise_postcode(pd.Series([postcode])).iloc[0]
        if pd.isna(pc) or pc not in self.lookup.index:
            raise ValueError(f"Postcode {postcode!r} was not found in the England & Wales postcode directory")

        row = pd.DataFrame([{
            "postcode": pc, "property_type": property_type, "old_new": "Y" if new_build else "N",
            "duration": tenure, "date": pd.Timestamp(sale_date),
        }])
        geo = self.lookup.loc[[pc]].reset_index(drop=True)
        row = add_derived_columns(pd.concat([row, geo], axis=1))
        return self.evaluate_row(row, actual_price, top_n)

    def evaluate_row(self, row: pd.DataFrame, actual_price: float | None = None, top_n: int = 8) -> dict:
        """Predict, explain and (optionally) score one prepared sale - also used to explain
        flagged sales from the test period in the dashboard."""
        property_type = str(row.loc[row.index[0], "property_type"])
        X = self.builder.transform(row)
        pred_log = float(predict_xgb(self.model, X)[0])
        values, base = shap_values(self.model, X)
        q_lo, q_hi = self.detector.range_quantiles_
        r0 = row.iloc[0]
        result = {
            "postcode": r0["postcode"],
            "pc_district": r0["pc_district"],
            "lad": r0["lad"],
            "region": REGION_NAMES.get(r0["region"], r0["region"]),
            "property_type": PROPERTY_TYPE_NAMES[property_type],
            "property_type_code": property_type,
            "predicted_price": float(np.exp(pred_log)),
            "range_low": float(np.exp(pred_log + q_lo)),
            "range_high": float(np.exp(pred_log + q_hi)),
            "baseline_price": float(np.exp(base[0])),
            "explanation": explain_row(values[0], X.iloc[0], top_n=top_n),
        }
        if actual_price:
            resid = np.log(actual_price) - pred_log
            z = float(self.detector.score([resid], [property_type])[0])
            result.update({
                "actual_price": float(actual_price),
                "difference_%": 100 * (actual_price / result["predicted_price"] - 1),
                "anomaly_score": z,
                "assessment": str(self.detector.label([z])[0]),
            })
        return result


def _format(result: dict) -> str:
    lines = [
        f"Postcode {result['postcode']} ({result['region']}), {result['property_type']}",
        f"Predicted price : £{result['predicted_price']:,.0f}  "
        f"(typical range £{result['range_low']:,.0f} - £{result['range_high']:,.0f})",
    ]
    if "actual_price" in result:
        lines += [
            f"Actual price    : £{result['actual_price']:,.0f}  ({result['difference_%']:+.1f}% vs predicted)",
            f"Anomaly score   : {result['anomaly_score']:+.2f}  ->  {result['assessment']}",
        ]
    lines.append("Main factors (SHAP, effect on predicted price):")
    for _, r in result["explanation"].iterrows():
        lines.append(f"  {r['effect_%']:+7.1f}%  {r['label']} = {r['value']}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Value one property with the trained model")
    p.add_argument("--postcode", required=True)
    p.add_argument("--type", required=True, choices=list(PROPERTY_TYPE_NAMES), help="D/S/T/F")
    p.add_argument("--tenure", default="F", choices=["F", "L"])
    p.add_argument("--new-build", action="store_true")
    p.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--price", type=float, help="Actual sale price, to get an anomaly score")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    a = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # show £ correctly in the Windows console
    valuer = PropertyValuer.load(Path(a.model_dir))
    print(_format(valuer.value(a.postcode, a.type, a.new_build, a.tenure, a.date, a.price)))


if __name__ == "__main__":
    main()

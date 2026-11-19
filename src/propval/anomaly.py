"""Price anomaly detection from prediction error.

residual  r = log(actual price) - log(predicted price)
score     z = (r - median) / (1.4826 * MAD)     robust z-score per property type

The median and MAD are calibrated on the *validation* period (not the test period),
and the robust statistics stop a few extreme sales from hiding the others.
A sale with |z| above the threshold is labelled an *unusual transaction*: it is a
prompt for human review, never a claim of fraud or an incorrect price (Risk R5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826  # makes the MAD comparable to a standard deviation for normal data
LABEL_HIGH, LABEL_LOW, LABEL_NORMAL = "Unusually high", "Unusually low", "Normal"


class ResidualAnomalyDetector:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.stats_: dict[str, tuple[float, float]] = {}

    def fit(self, residuals, groups) -> "ResidualAnomalyDetector":
        frame = pd.DataFrame({"r": np.asarray(residuals, dtype="float64"), "g": np.asarray(groups).astype(str)})
        self.global_ = self._robust(frame["r"].to_numpy())
        for g, part in frame.groupby("g"):
            if len(part) >= 30:
                self.stats_[g] = self._robust(part["r"].to_numpy())
        # 10th / 90th residual percentiles give a plausible price range in the app.
        self.range_quantiles_ = tuple(np.quantile(frame["r"], [0.10, 0.90]))
        return self

    @staticmethod
    def _robust(r: np.ndarray) -> tuple[float, float]:
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med))) * MAD_SCALE
        return med, max(mad, 1e-6)

    def score(self, residuals, groups) -> np.ndarray:
        r = np.asarray(residuals, dtype="float64")
        g = np.asarray(groups).astype(str)
        med = np.array([self.stats_.get(x, self.global_)[0] for x in g])
        mad = np.array([self.stats_.get(x, self.global_)[1] for x in g])
        return (r - med) / mad

    def label(self, z) -> np.ndarray:
        z = np.asarray(z)
        return np.where(z > self.threshold, LABEL_HIGH, np.where(z < -self.threshold, LABEL_LOW, LABEL_NORMAL))

    def stats_table(self) -> pd.DataFrame:
        rows = [{"group": g, "median_residual": m, "robust_sd": s,
                 "flag_if_price_above_%": 100 * (np.exp(m + self.threshold * s) - 1),
                 "flag_if_price_below_%": 100 * (np.exp(m - self.threshold * s) - 1)}
                for g, (m, s) in sorted(self.stats_.items())]
        return pd.DataFrame(rows)

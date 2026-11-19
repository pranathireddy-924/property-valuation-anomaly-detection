"""Charts for the dissertation report and presentation (saved as 200-dpi PNGs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import PROPERTY_TYPE_NAMES  # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN = "#2b6cb0", "#dd6b20", "#718096", "#c53030", "#2f855a"
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def price_distribution(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(np.log10(10_000), np.log10(10_000_000), 80)
    for code, name in PROPERTY_TYPE_NAMES.items():
        vals = np.log10(df.loc[df["property_type"] == code, "price"])
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.6, label=name)
    ticks = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]
    ax.set_xticks(np.log10(ticks), [f"£{t/1000:,.0f}k" if t < 1e6 else f"£{t/1e6:,.0f}m" for t in ticks])
    ax.set(title="Sale price distribution by property type (log scale)", xlabel="Sale price", ylabel="Number of sales")
    ax.legend()
    return _save(fig, path)


def monthly_trend(df: pd.DataFrame, train_end: str, val_end: str, path: Path) -> Path:
    m = df.groupby(df["date"].dt.to_period("M")).agg(median=("price", "median"), n=("price", "size"))
    x = m.index.to_timestamp()
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(x, m["median"], color=BLUE, linewidth=2, label="Median price")
    ax1.set_ylabel("Median sale price (£)", color=BLUE)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"£{v/1000:,.0f}k"))
    ax2 = ax1.twinx()
    ax2.bar(x, m["n"], width=20, color=GREY, alpha=0.3, label="Sales volume")
    ax2.set_ylabel("Sales per month", color=GREY)
    ax2.grid(False)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    for d, lbl in ((train_end, "train | validation"), (val_end, "validation | test")):
        ax1.axvline(pd.Timestamp(d), color=RED, linestyle="--", linewidth=1)
        ax1.text(pd.Timestamp(d), ax1.get_ylim()[1], f" {lbl}", color=RED, va="top", fontsize=8)
    ax1.set_title("Monthly median price and sales volume, with the time-based split")
    return _save(fig, path)


def model_comparison(results: pd.DataFrame, path: Path) -> Path:
    """results: rows = model, columns include MAE, RMSE, R2, MdAPE_% (test set)."""
    metrics = [("MAE", "MAE (£)"), ("RMSE", "RMSE (£)"), ("R2", "R² (price)"), ("MdAPE_%", "Median abs. % error")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    colors = [GREY, ORANGE, BLUE][: len(results)] + [GREEN] * max(0, len(results) - 3)
    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(results["model"], results[col], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, results[col]):
            ax.annotate(f"{v:,.3f}" if col == "R2" else f"{v:,.0f}" if col != "MdAPE_%" else f"{v:.1f}%",
                        (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Model comparison on the test period", fontweight="bold")
    return _save(fig, path)


def ablation_chart(ablation: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in ((axes[0], "MdAPE_%", "Median absolute % error (lower is better)"),
                           (axes[1], "R2_log", "R² on log price (higher is better)")):
        bars = ax.barh(ablation["feature_set"], ablation[col], color=[GREY, ORANGE, ORANGE, BLUE][: len(ablation)])
        ax.invert_yaxis()
        ax.set_title(title)
        for b, v in zip(bars, ablation[col]):
            ax.annotate(f" {v:.3f}" if col == "R2_log" else f" {v:.1f}%", (b.get_width(), b.get_y() + b.get_height() / 2),
                        va="center", fontsize=8)
    fig.suptitle("Feature ablation: XGBoost with increasing feature groups (test period)", fontweight="bold")
    return _save(fig, path)


def predicted_vs_actual(actual, predicted, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    a, p = np.log10(actual), np.log10(predicted)
    hb = ax.hexbin(a, p, gridsize=90, bins="log", cmap="Blues", mincnt=1)
    lo, hi = np.percentile(a, [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], color=RED, linewidth=1, label="Perfect prediction")
    ticks = [3e4, 1e5, 3e5, 1e6, 3e6]
    labels = [f"£{t/1000:,.0f}k" if t < 1e6 else f"£{t/1e6:,.0f}m" for t in ticks]
    ax.set_xticks(np.log10(ticks), labels)
    ax.set_yticks(np.log10(ticks), labels)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual sale price", ylabel="Predicted price", title=title)
    fig.colorbar(hb, ax=ax, label="Number of sales (log)")
    ax.legend(loc="upper left")
    return _save(fig, path)


def residual_distribution(z: np.ndarray, threshold: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    zc = np.clip(z, -10, 10)
    ax.hist(zc, bins=200, color=BLUE, alpha=0.8)
    ax.set_yscale("log")
    for t in (-threshold, threshold):
        ax.axvline(t, color=RED, linestyle="--")
    share = 100 * (np.abs(z) > threshold).mean()
    ax.set(title=f"Robust anomaly score on the test period ({share:.1f}% flagged at |z| > {threshold:g})",
           xlabel="Robust z-score of log prediction error (clipped at ±10)", ylabel="Number of sales (log scale)")
    return _save(fig, path)


def error_by_group(table: pd.DataFrame, path: Path, title: str) -> Path:
    t = table.sort_values("MdAPE_%")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(t) + 1.5))
    bars = ax.barh(t["group"].astype(str), t["MdAPE_%"], color=BLUE)
    for b, v in zip(bars, t["MdAPE_%"]):
        ax.annotate(f" {v:.1f}%", (b.get_width(), b.get_y() + b.get_height() / 2), va="center", fontsize=8)
    ax.set(title=title, xlabel="Median absolute % error (XGBoost, test period)")
    return _save(fig, path)


def anomaly_rates(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(table["group"], table["flag_rate_%"], color=[BLUE, ORANGE][: len(table)])
    for b, v in zip(bars, table["flag_rate_%"]):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom")
    ax.set(title="Share of sales flagged as unusual (test period)", ylabel="% flagged")
    return _save(fig, path)


def shap_bar(importance: pd.DataFrame, path: Path, top_n: int = 15) -> Path:
    t = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(t) + 1.5))
    ax.barh(t["label"], t["mean_abs_shap"], color=BLUE)
    ax.set(title="Global feature importance (mean |SHAP value|, log-price units)", xlabel="Mean |SHAP value|")
    return _save(fig, path)


def shap_beeswarm(values: np.ndarray, X: pd.DataFrame, labels: list[str], path: Path) -> Path | None:
    try:
        import shap
    except ImportError:
        return None
    data = X.copy()
    for c in data.columns:
        if isinstance(data[c].dtype, pd.CategoricalDtype):
            data[c] = data[c].cat.codes.replace(-1, np.nan)
    expl = shap.Explanation(values=values, data=data.to_numpy(dtype="float64"), feature_names=labels)
    plt.figure()
    shap.plots.beeswarm(expl, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6.5)
    plt.title("SHAP summary: how each feature pushes predictions up or down", fontweight="bold")
    return _save(fig, path)


def case_explanation(contrib: pd.DataFrame, header: str, path: Path) -> Path:
    """Horizontal 'waterfall-style' bar chart of the top SHAP effects for one sale."""
    t = contrib.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(t) + 2))
    colors = [GREEN if v > 0 else RED for v in t["effect_%"]]
    labels = [f"{lbl} = {val}" for lbl, val in zip(t["label"], t["value"])]
    bars = ax.barh(labels, t["effect_%"], color=colors)
    for b, v in zip(bars, t["effect_%"]):
        ax.annotate(f" {v:+.1f}% " if v >= 0 else f" {v:+.1f}% ", (b.get_width(), b.get_y() + b.get_height() / 2),
                    va="center", fontsize=8, ha="left" if v >= 0 else "right")
    lo, hi = min(0.0, t["effect_%"].min()), max(0.0, t["effect_%"].max())
    pad = 0.18 * (hi - lo or 1)
    ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)   # room for the value labels
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Effect on predicted price compared with an average sale (%)")
    ax.set_title(header, fontsize=10)
    return _save(fig, path)

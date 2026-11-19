"""Summary analysis of the professional feedback questionnaire (evaluation / UAT).

As stated in the ethics form: closed questions are summarised with counts and
percentages; open-text comments are anonymised and listed for thematic review.
Participants are coded P1, P2 ... - no names or company names are stored.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LIKERT = {1: "Strongly disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly agree"}
QUESTIONS = {
    "q1_prediction_clear": "The predicted price and range are easy to understand",
    "q2_anomaly_useful": "The 'unusual transaction' flag would help me decide what to review",
    "q3_shap_understandable": "The SHAP explanation shows clearly why the price was predicted",
    "q4_factors_sensible": "The main factors shown match my professional experience",
    "q5_supports_review": "The tool could support (not replace) human valuation review",
    "q6_would_use": "I would use a tool like this in my work",
}
_REDACT = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email removed]"),
    (re.compile(r"(\+44\s?|0)\d[\d\s]{8,12}\d"), "[phone removed]"),
    (re.compile(r"https?://\S+"), "[link removed]"),
    (re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I), "[postcode removed]"),
]


def write_template(path: Path) -> Path:
    """Blank response sheet: one row per participant, answers 1-5."""
    cols = ["participant_id", "role", *QUESTIONS, "comments"]
    pd.DataFrame(columns=cols).to_csv(path, index=False)
    return path


def anonymise(text) -> str:
    if pd.isna(text):
        return ""
    out = str(text)
    for pattern, repl in _REDACT:
        out = pattern.sub(repl, out)
    return out


def summarise(responses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, wording in QUESTIONS.items():
        answers = pd.to_numeric(responses[q], errors="coerce").dropna().astype(int)
        answers = answers[answers.between(1, 5)]
        n = len(answers)
        row = {"question": q, "statement": wording, "n": n}
        for k, label in LIKERT.items():
            row[f"{label} (n)"] = int((answers == k).sum())
        row["agree_%"] = 100 * (answers >= 4).mean() if n else float("nan")
        row["disagree_%"] = 100 * (answers <= 2).mean() if n else float("nan")
        row["median"] = float(answers.median()) if n else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, path: Path) -> Path:
    colors = ["#c53030", "#ed8936", "#a0aec0", "#68d391", "#2f855a"]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(summary) + 1.8))
    left = pd.Series(0.0, index=summary.index)
    for (k, label), c in zip(LIKERT.items(), colors):
        share = 100 * summary[f"{label} (n)"] / summary["n"].where(summary["n"] > 0, 1)
        ax.barh(summary["statement"], share, left=left, color=c, label=label)
        left += share
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="% of participants",
           title=f"Professional feedback on the prototype (n = {int(summary['n'].max())})")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18 if len(summary) > 3 else -0.3), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def analyse(responses_csv: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(responses_csv)
    df["participant_id"] = [f"P{i}" for i in range(1, len(df) + 1)]   # never keep real identifiers
    summary = summarise(df)
    summary.to_csv(out_dir / "18_questionnaire_summary.csv", index=False)
    roles = df["role"].fillna("Not stated").value_counts().rename_axis("role").reset_index(name="participants")
    roles.to_csv(out_dir / "19_questionnaire_roles.csv", index=False)
    comments = pd.DataFrame({"participant": df["participant_id"], "comment": df["comments"].map(anonymise)})
    comments = comments[comments["comment"].str.strip() != ""]
    comments.to_csv(out_dir / "20_questionnaire_comments_anonymised.csv", index=False)
    fig = plot_summary(summary, out_dir / "fig14_questionnaire_feedback.png")
    return {"summary": summary, "roles": roles, "comments": comments, "figure": fig}

"""Unit tests - questionnaire summary analysis (test cases UT-20 to UT-22)."""
import pandas as pd

from propval.questionnaire import QUESTIONS, analyse, anonymise, summarise


def _responses():
    return pd.DataFrame({
        "participant_id": ["Jane", "Bob", "x", "y"], "role": ["Estate agent", "Mortgage adviser", "Estate agent", None],
        **{q: [5, 4, 2, 4] for q in QUESTIONS},
        "comments": ["Useful, email me at jane@agency.co.uk", "", None, "Check S1 2BJ and call 07700 900123"],
    })


def test_ut20_counts_and_percentages():
    s = summarise(_responses()).set_index("question")
    row = s.loc["q1_prediction_clear"]
    assert row["n"] == 4 and row["Agree (n)"] == 2 and row["Strongly agree (n)"] == 1
    assert row["agree_%"] == 75.0 and row["disagree_%"] == 25.0


def test_ut21_comments_are_anonymised():
    out = anonymise("email jane@agency.co.uk, call 07700 900123, see S1 2BJ")
    assert "jane@" not in out and "07700" not in out and "S1 2BJ" not in out


def test_ut22_analysis_outputs_use_codes_not_names(tmp_path):
    csv = tmp_path / "r.csv"
    _responses().to_csv(csv, index=False)
    res = analyse(csv, tmp_path / "out")
    assert list(res["comments"]["participant"]) == ["P1", "P4"]
    assert (tmp_path / "out" / "fig14_questionnaire_feedback.png").exists()
    assert "Jane" not in (tmp_path / "out" / "20_questionnaire_comments_anonymised.csv").read_text()

"""Summarise the professional feedback questionnaire.

    python analyse_questionnaire.py --template                 # create questionnaire/responses.csv to fill in
    python analyse_questionnaire.py questionnaire/responses.csv

Answers are 1-5 (1 = strongly disagree, 5 = strongly agree). Outputs go to
outputs/questionnaire/: counts and percentages per question, roles, anonymised
comments and a stacked bar chart (fig14).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from propval.questionnaire import analyse, write_template  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("responses", nargs="?", help="CSV of responses")
    p.add_argument("--template", action="store_true", help="Write a blank response sheet")
    p.add_argument("--out", default=str(ROOT / "outputs" / "questionnaire"))
    a = p.parse_args()
    if a.template:
        (ROOT / "questionnaire").mkdir(exist_ok=True)
        print("Template written to", write_template(ROOT / "questionnaire" / "responses.csv"))
    elif a.responses:
        res = analyse(Path(a.responses), Path(a.out))
        print(res["summary"][["statement", "n", "agree_%", "median"]].to_string(index=False))
        print("\nOutputs written to", a.out)
    else:
        p.print_help()

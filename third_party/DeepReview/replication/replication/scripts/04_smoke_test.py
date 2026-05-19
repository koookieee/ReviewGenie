"""
Smoke test: synthesize a fake prediction file using each paper's mean human
score (i.e. an oracle that always predicts the correct average). Run evalate.py
on it to verify the pipeline works end-to-end.

Expected: near-zero MSE/MAE, ~1.0 Spearman, ~1.0 pairwise accuracy, decision
accuracy near 1.0.
"""

import json
from pathlib import Path
import statistics

EVAL_SET = Path(__file__).resolve().parents[1] / "data" / "eval_set_iclr2024.json"
OUT_PATH = Path(__file__).resolve().parents[1] / "predictions" / "oracle_iclr2024.json"


def parse_leading_int(s):
    """Parse '3 good' -> 3 etc."""
    s = str(s).strip()
    n = ""
    for ch in s:
        if ch.isdigit():
            n += ch
        else:
            break
    return int(n) if n else None


def boxed_review(rating, sound, pres, contrib, decision):
    """Build a `\\boxed_review{}` block matching the format evalate.py expects."""
    return (
        "\\boxed_review{\n"
        f"## Summary:\n\noracle\n"
        f"## Rating:\n\n{rating}\n"
        f"## Soundness:\n\n{sound}\n"
        f"## Presentation:\n\n{pres}\n"
        f"## Contribution:\n\n{contrib}\n"
        f"## Strengths:\n\nN/A\n"
        f"## Weaknesses:\n\nN/A\n"
        f"## Suggestions:\n\nN/A\n"
        f"## Questions:\n\nN/A\n"
        f"## Confidence:\n\n3\n"
        f"## Decision:\n\n{decision}\n"
        "}\n"
    )


def main():
    with open(EVAL_SET, encoding="utf-8") as f:
        records = json.load(f)

    out = []
    for rec in records:
        ratings = [int(r["content"]["rating"][0]) for r in rec["review"]]
        sounds = [parse_leading_int(r["content"]["soundness"]) for r in rec["review"]]
        press = [parse_leading_int(r["content"]["presentation"]) for r in rec["review"]]
        contribs = [parse_leading_int(r["content"]["contribution"]) for r in rec["review"]]

        avg_rating = statistics.mean(ratings)
        avg_sound = statistics.mean(sounds)
        avg_pres = statistics.mean(press)
        avg_contrib = statistics.mean(contribs)

        item = dict(rec)
        item["pred_standard_mode"] = boxed_review(
            f"{avg_rating:.2f}",
            f"{avg_sound:.2f}",
            f"{avg_pres:.2f}",
            f"{avg_contrib:.2f}",
            rec["decision"],
        )
        out.append(item)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {len(out)} oracle predictions to {OUT_PATH}")


if __name__ == "__main__":
    main()

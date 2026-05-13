"""
Convert the raw HF test CSVs into the JSON schema expected by evalate.py.

Output JSON schema (matches Researcher/evaluate/DeepReview/sample.json):
[
  {
    "id": "...",
    "title": "...",                    # not in CSV; left as ""
    "paper_context": "...",            # extracted from inputs[1]['content']
    "decision": "Accept" | "Reject",
    "review": [                        # the human reviewer list (= reviewer_comments)
      {"id": "...", "rating": int, "content": {...}}, ...
    ],
    # No pred_*_mode fields here; those get filled by inference scripts.
  },
  ...
]

We emit one file per ICLR year so they can be evaluated independently
(matching Tables 1 and 2 of the paper).
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

CSV_TO_YEAR = {
    "test_2024.csv": "iclr2024",
    "test_2025.csv": "iclr2025",
}


def csv_to_eval_records(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        inputs = json.loads(row["inputs"])
        # inputs[0] is the system prompt (varies; not needed for eval)
        # inputs[1] is the user message which contains the paper text
        paper_context = inputs[1]["content"]

        reviewer_comments = json.loads(row["reviewer_comments"])

        records.append({
            "id": row["id"],
            "title": "",  # CSV does not include title separately
            "paper_context": paper_context,
            "decision": row["decision"],            # "Accept" / "Reject"
            "review": reviewer_comments,            # list of dicts with content.{rating,soundness,...}
        })
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for csv_name, year_tag in CSV_TO_YEAR.items():
        records = csv_to_eval_records(DATA_DIR / csv_name)
        out_path = OUT_DIR / f"eval_set_{year_tag}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()

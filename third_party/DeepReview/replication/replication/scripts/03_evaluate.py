"""
Thin wrapper around Researcher/evaluate/DeepReview/evalate.py.

Why we wrap: evalate.py imports its own functions and is hardcoded to read
'sample.json'. We import it as a module and call evaluate_deep_reviewer() on
arbitrary prediction files, so we can run it on any year/model/mode combo.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_EVALATE_CANDIDATES = [
    # Local repo clone alongside replication/
    Path(__file__).resolve().parents[2] / "Researcher" / "evaluate" / "DeepReview" / "evalate.py",
    # Script lives next to evalate.py (remote layout)
    Path(__file__).resolve().parent / "evalate.py",
    # git clone left in /tmp (common on remote)
    Path("/tmp/Researcher/evaluate/DeepReview/evalate.py"),
]
REPO_EVAL_PATH = next((p for p in _EVALATE_CANDIDATES if p.is_file()), _EVALATE_CANDIDATES[0])


def load_evalate_module():
    spec = importlib.util.spec_from_file_location("evalate", REPO_EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evalate"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path,
                    help="Path to a predictions JSON produced by 02_run_inference.py")
    ap.add_argument("--mode", default="standard", choices=["fast", "standard", "best"],
                    help="Which pred_<mode>_mode key to evaluate")
    ap.add_argument("--out-md", type=Path, help="Optional markdown output path")
    args = ap.parse_args()

    evalate = load_evalate_module()
    results = evalate.evaluate_deep_reviewer(str(args.predictions), mode=args.mode)
    md = evalate.create_markdown_table(results)

    print(f"\n=== Results: {args.predictions.name} (mode={args.mode}) ===")
    print(md)

    if args.out_md:
        args.out_md.write_text(f"# {args.predictions.name} (mode={args.mode})\n\n{md}\n")
        print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()

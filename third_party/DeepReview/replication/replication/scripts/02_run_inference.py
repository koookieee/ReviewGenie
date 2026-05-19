"""
Run a reviewer model over an eval set and emit predictions in the format
required by evalate.py.

The output JSON appends `pred_<mode>_mode` keys to each record. evalate.py
reads exactly one of `pred_fast_mode`, `pred_standard_mode`, or `pred_best_mode`.

Default behavior: writes `pred_standard_mode` (you can override with --mode-key).

Currently supports two backends:
  --backend agentreviewer   <- placeholder for YOUR AgentReviewer; wire it up
  --backend openai          <- OpenAI-compatible API (Claude/GPT/Gemini/etc via proxy)

Add new backends by implementing `generate_review(paper_context: str) -> str`
and registering it in BACKENDS.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

from prompts import GENERIC_REVIEWER_SYSTEM_PROMPT, generic_messages

# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------


def make_openai_backend(model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY") -> Callable[[str], str]:
    """Generic OpenAI-compatible chat completion backend. Works for OpenAI,
    Anthropic via proxy, Gemini via the OpenAI-compatible endpoint, DeepSeek, etc."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url) if base_url else OpenAI(api_key=os.environ[api_key_env])

    def _gen(paper_context: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=generic_messages(paper_context),
            temperature=0.4,
            max_tokens=16384,
        )
        return resp.choices[0].message.content

    return _gen


def make_agentreviewer_backend() -> Callable[[str], str]:
    """STUB. Replace this with the entry point of your AgentReviewer.

    Your AgentReviewer must accept the paper text and return a single string
    containing a `\\boxed_review{...}` block formatted per prompts.py's
    GENERIC_REVIEWER_SYSTEM_PROMPT.
    """
    def _gen(paper_context: str) -> str:
        raise NotImplementedError(
            "Wire up your AgentReviewer here. Import its main entry point and "
            "return its raw output text. The output MUST contain a "
            "\\boxed_review{...} block per prompts.py."
        )

    return _gen


BACKENDS = {
    "agentreviewer": make_agentreviewer_backend,
    "openai": lambda: make_openai_backend(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    ),
}


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def run(eval_set_path: Path, out_path: Path, generate: Callable[[str], str],
        mode_key: str = "pred_standard_mode", limit: int | None = None,
        resume: bool = True) -> None:
    with open(eval_set_path, encoding="utf-8") as f:
        records = json.load(f)
    if limit:
        records = records[:limit]

    # Resume support: load existing predictions if any
    existing = {}
    if resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for r in json.load(f):
                existing[r["id"]] = r

    out = []
    for i, rec in enumerate(records):
        if rec["id"] in existing and mode_key in existing[rec["id"]]:
            out.append(existing[rec["id"]])
            continue
        merged = dict(existing.get(rec["id"], rec))
        try:
            t0 = time.time()
            review_text = generate(rec["paper_context"])
            merged[mode_key] = review_text
            print(f"[{i+1}/{len(records)}] id={rec['id']} ok ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[{i+1}/{len(records)}] id={rec['id']} ERROR: {e}", file=sys.stderr)
            merged[mode_key] = ""  # eval will skip this row
        out.append(merged)

        # Checkpoint every 10 papers
        if (i + 1) % 10 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {len(out)} records to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True, type=Path,
                    help="Path to eval_set_iclr2024.json or eval_set_iclr2025.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=list(BACKENDS.keys()))
    ap.add_argument("--mode-key", default="pred_standard_mode",
                    choices=["pred_fast_mode", "pred_standard_mode", "pred_best_mode"])
    ap.add_argument("--limit", type=int, default=None, help="Cap papers for smoke test")
    args = ap.parse_args()

    generate = BACKENDS[args.backend]()
    run(args.eval_set, args.out, generate, mode_key=args.mode_key, limit=args.limit)


if __name__ == "__main__":
    main()

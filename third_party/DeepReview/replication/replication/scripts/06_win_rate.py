"""
06_win_rate.py — Pairwise win-rate evaluation: Our Harness vs DeepReviewer-14B

Uses the exact SYSTEM_PROMPT and judge logic from the official
Researcher/evaluate/DeepReview/win_rate_evaluate.py.

Our reviews come from a predictions JSON (pred_standard_mode key).
DeepReviewer-14B reviews come from the HF dataset outputs column
(outputs[-1]['content'] = their final Best Mode review with \\boxed_review{}).

Usage (on remote):
    # ICLR 2024 — our DeepSeek Pro run vs DeepReviewer-14B
    python3 06_win_rate.py \
        --our-predictions /root/pass_at_k/predictions_deepreview_2024_good.json \
        --split 2024 \
        --out /root/pass_at_k/win_rate_2024_pro.jsonl \
        --max-concurrent 8

    # ICLR 2025 — our DeepSeek Flash run vs DeepReviewer-14B (after run completes)
    python3 06_win_rate.py \
        --our-predictions /root/pass_at_k/predictions_deepreview_2025.json \
        --split 2025 \
        --out /root/pass_at_k/win_rate_2025_flash.jsonl \
        --max-concurrent 8

    # Print results from existing output file
    python3 06_win_rate.py --print-results /root/pass_at_k/win_rate_2024_pro.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Exact SYSTEM_PROMPT from win_rate_evaluate.py — do not modify
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
---***---
SYSTEM PROMPT
---***---

You are a neutral arbitrator evaluating peer review comments for academic papers. Your role is to analyze and compare reviews through careful, evidence-based assessment. Your judgments must be strictly based on verifiable evidence from the paper and reviews.

For each evaluation, you must:

1. Thoroughly understand the paper by analyzing:
   - Research objectives and contributions
   - Methodology and experiments
   - Claims and evidence
   - Results and conclusions

2. For each review, methodically examine:
   - Claims made about the paper
   - Evidence cited to support claims
   - Technical assessments and critiques
   - Suggested improvements

3. Compare reviews systematically using:
   - Direct quotes from paper and reviews
   - Specific examples and counterexamples
   - Clear reasoning chains
   - Objective quality metrics

You will evaluate reviews based on these key aspects:

**Technical Accuracy**
- Are claims consistent with paper content?
- Is evidence properly interpreted?
- Are technical assessments valid?
- Are critiques well-supported?

**Constructive Value**
- How actionable is the feedback?
- Are suggestions specific and feasible?
- Is criticism balanced with strengths?
- Would authors understand how to improve?

**Analytical Depth**
- How thoroughly are key aspects examined?
- Is analysis appropriately detailed?
- Are important elements addressed?
- Is assessment comprehensive?

**Communication Clarity**
- Are points clearly articulated?
- Is feedback specific and concrete?
- Is reasoning well-explained?
- Are examples effectively used?

For each aspect and overall judgment, you must:
1. Provide specific evidence from source materials
2. Quote directly from paper and reviews
3. Explain your reasoning in detail
4. Consider alternative interpretations

**Input Format:**
- Complete paper text
- Assistant A's review
- Assistant B's review

**Output Format:**

For each aspect:

```
**[Aspect Name] - Evidence Analysis:**
- From Assistant A:
  [Direct quotes and specific examples]
  [Detailed analysis of evidence]
- From Assistant B:
  [Direct quotes and specific examples]
  [Detailed analysis of evidence]
- Comparative Assessment:
  [Evidence-based comparison]
  [Clear reasoning chain]

**[Aspect Name] - Judgment:**
**Evidence-Based Reason:** [Detailed justification citing specific evidence]
**Better Assistant:** [A or B or Tie]
- If Tie: Explain why both reviews are equally strong on this aspect
```

Conclude with:

```
**Comprehensive Analysis:**
[Synthesis of evidence across aspects]
[Analysis of relative strengths]
[Discussion of key differences or similarities]

**Overall Judgment:**
**Evidence-Based Reason:** [Detailed justification synthesizing key evidence]
**Better Assistant:** [A or B or Tie]
- If Overall Tie: Explain why both reviews are comparable in overall quality
```

Key Requirements:
1. Base all judgments on concrete evidence
2. Quote directly from source materials
3. Provide detailed reasoning chains
4. Maintain neutral arbitrator perspective
5. Judge Tie when evidence shows equal strength
6. Always justify Tie decisions with specific evidence

When judging Tie:
- Ensure both reviews demonstrate similar levels of quality
- Provide explicit evidence showing comparable strengths
- Explain why differences are not significant enough to favor one over the other
- Consider both quantity and quality of evidence

Begin analysis after receiving complete materials. Take time to examine evidence thoroughly and provide detailed, justified assessments.
"""

JUDGE_MODEL = "gemini-3.1-pro-preview"

_BOXED_OPENER = "\\boxed_review{"


def extract_review_content(pred_context) -> str:
    """Exact logic from win_rate_evaluate.py:ReviewProcessor.extract_review_content."""
    try:
        return pred_context.split(_BOXED_OPENER)[-1].split('\n}')[0]
    except Exception:
        if isinstance(pred_context, dict) and 'output' in pred_context:
            return pred_context['output'].split(_BOXED_OPENER)[-1].split('\n}')[0]
        return pred_context


# ---------------------------------------------------------------------------
# Load DeepReviewer-14B outputs from HF dataset
# ---------------------------------------------------------------------------

def load_deepreviewer_outputs(split: int) -> dict[str, dict]:
    """Load DeepReviewer-14B Best Mode reviews + paper contexts from HF.

    Returns dict: paper_id -> {review: str, paper_context: str}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    if split == 2024:
        ds = load_dataset("WestlakeNLP/DeepReview-13K", split="test[:652]")
    else:
        ds = load_dataset("WestlakeNLP/DeepReview-13K", split="test[652:]")

    result = {}
    for row in ds:
        outputs = json.loads(row["outputs"]) if isinstance(row["outputs"], str) else row["outputs"]
        inputs = json.loads(row["inputs"]) if isinstance(row["inputs"], str) else row["inputs"]
        paper_context = inputs[1]["content"] if len(inputs) > 1 else ""
        # Final assistant message contains the \boxed_review{}
        final_review = outputs[-1]["content"] if outputs else ""
        result[row["id"]] = {
            "review": extract_review_content(final_review),
            "paper_context": paper_context,
        }
    print(f"Loaded {len(result)} DeepReviewer-14B reviews (split={split})")
    return result


# ---------------------------------------------------------------------------
# Load Stanford reviewer outputs
# ---------------------------------------------------------------------------

def load_stanford_data(
    reviews_dir: Path,
    data_dir: Path,
    ocr_dir: Path | None = None,
) -> dict[str, dict]:
    """Load Stanford Reviewer outputs + paper bodies.

    Returns dict: paper_id -> {review: str, paper_context: str}
    """
    result = {}
    for f in sorted(reviews_dir.glob("*_review.json")):
        paper_id = f.stem.replace("_review", "")
        try:
            d = json.loads(f.read_text())
            review = d.get("content", "")
            if not review:
                continue
        except Exception:
            continue

        # Paper body: prefer OCR markdown, fall back to LaTeX
        paper_context = ""
        if ocr_dir:
            ocr = ocr_dir / f"{paper_id}.md"
            if ocr.is_file():
                paper_context = ocr.read_text(encoding="utf-8", errors="replace")
        if not paper_context:
            latex = data_dir / paper_id / "latex" / "template.tex"
            if latex.is_file():
                paper_context = latex.read_text(encoding="utf-8", errors="replace")

        if paper_context:
            result[paper_id] = {"review": review, "paper_context": paper_context}

    print(f"Loaded {len(result)} Stanford reviews from {reviews_dir}")
    return result


# ---------------------------------------------------------------------------
# Load our reviews from pass@K trajectories directly
# ---------------------------------------------------------------------------

def load_our_trajectory_reviews(results_dir: Path) -> dict[str, str]:
    """Load our reviews directly from trajectory.json files.

    Returns dict: paper_id -> review_text
    """
    import re
    result = {}
    review_markers = ("### Summary", "### Strengths", "### Weaknesses", "### Scores", "**Scores**")
    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        for attempt_dir in sorted(paper_dir.iterdir()):
            if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt_"):
                continue
            traj = attempt_dir / "trajectory.json"
            if not traj.is_file():
                continue
            try:
                td = json.loads(traj.read_text())
                msgs = [s["message"] for s in td.get("steps", [])
                        if s.get("source") == "agent" and s.get("message") and len(s["message"]) > 200]
                if not msgs:
                    continue
                review = next((m for m in reversed(msgs) if any(mk in m for mk in review_markers)), msgs[-1])
                if review:
                    result[paper_dir.name] = review
                    break
            except Exception:
                continue
    print(f"Loaded {len(result)} of our trajectory reviews from {results_dir}")
    return result


# ---------------------------------------------------------------------------
# Load our predictions
# ---------------------------------------------------------------------------

def load_our_predictions(path: Path) -> dict[str, str]:
    """Load our reviews from predictions JSON. Returns paper_id -> review_content."""
    preds = json.loads(path.read_text())
    result = {}
    for p in preds:
        pred_text = p.get("pred_standard_mode", "")
        if pred_text:
            result[p["id"]] = extract_review_content(pred_text)
    print(f"Loaded {len(result)} of our predictions from {path.name}")
    return result


# ---------------------------------------------------------------------------
# Build comparison pairs
# ---------------------------------------------------------------------------

def build_comparison_data(
    our_reviews: dict[str, str],
    deepreviewer: dict[str, dict],
    split: int,
) -> list[dict]:
    common = set(our_reviews.keys()) & set(deepreviewer.keys())
    # Filter out empties
    common = {pid for pid in common if our_reviews[pid].strip() and deepreviewer[pid]["review"].strip()}
    print(f"Matched papers: {len(common)}")
    data = []
    for pid in sorted(common):
        data.append({
            "id": pid,
            "year": split,
            "paper_context": deepreviewer[pid]["paper_context"],
            "our_review": our_reviews[pid],
            "deepreviewer_review": deepreviewer[pid]["review"],
        })
    return data


# ---------------------------------------------------------------------------
# Judge one paper (async)
# ---------------------------------------------------------------------------

async def judge_one(item: dict, client, sem: asyncio.Semaphore) -> dict:
    from google.genai import types as genai_types

    # Random ordering to remove position bias (exact same as win_rate_evaluate.py)
    if random.randint(0, 1):
        content = (
            '# Paper:\n' + item['paper_context'] +
            '\n\n---***---\n---***---\n---***---\n' +
            '#Assistant A:\n' + item['our_review'] +
            '\n\n---***---\n---***---\n---***---\n' +
            '#Assistant B:\n' + item['deepreviewer_review']
        )
        ordering = 'A'  # our review is A
    else:
        content = (
            '# Paper:\n' + item['paper_context'] +
            '\n\n---***---\n---***---\n---***---\n' +
            '#Assistant A:\n' + item['deepreviewer_review'] +
            '\n\n---***---\n---***---\n---***---\n' +
            '#Assistant B:\n' + item['our_review']
        )
        ordering = 'B'  # our review is B

    async with sem:
        try:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=JUDGE_MODEL,
                contents=content,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=8192,
                    thinking_config=genai_types.ThinkingConfig(
                        thinking_level=genai_types.ThinkingLevel.MEDIUM,
                    ),
                ),
            )
            result_text = resp.text
        except Exception as e:
            result_text = f"ERROR: {e}"

    return {
        "id": item["id"],
        "year": item["year"],
        "v.s.": ordering,
        "result": result_text,
    }


# ---------------------------------------------------------------------------
# Parse results (exact same logic as win_rate_evaluate.py:print_result)
# ---------------------------------------------------------------------------

def get_verdict(line: str) -> str:
    for l in line.split('\n'):
        if 'Better Assistant' in l:
            l = l.replace('Better', '').replace('Assistant', '')
            if 'Tie' in l:
                return 'Tie'
            elif 'B' in l:
                return 'B'
            else:
                return 'A'
    return ''


def parse_and_print_results(data: list[dict]) -> None:
    cats = {
        'Overall Judgment': [],
        'Technical Accuracy': [],
        'Constructive Value': [],
        'Analytical Depth': [],
        'Communication Clarity': [],
    }

    for item in data:
        if "ERROR" in item.get("result", ""):
            continue
        ours = item['v.s.']
        for para in item['result'].split('\n\n'):
            if 'Better Assistant' not in para:
                continue
            verdict = get_verdict(para)
            if not verdict:
                continue
            outcome = 'win' if verdict == ours else ('tie' if verdict == 'Tie' else 'lose')
            for cat in cats:
                if cat in para:
                    cats[cat].append(outcome)
                    break

    print(f"\n{'='*55}")
    print(f"WIN-RATE: Our Harness vs DeepReviewer-14B (Best Mode)")
    print(f"{'='*55}")
    print(f"{'Category':<25} {'Win':>7} {'Tie':>7} {'Lose':>7} {'N':>5}")
    print(f"{'-'*55}")
    for cat, outcomes in cats.items():
        if not outcomes:
            print(f"{cat:<25} {'N/A':>7}")
            continue
        n = len(outcomes)
        win = outcomes.count('win') / n
        tie = outcomes.count('tie') / n
        lose = outcomes.count('lose') / n
        print(f"{cat:<25} {win:>7.1%} {tie:>7.1%} {lose:>7.1%} {n:>5}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    if args.print_results:
        data = [json.loads(l) for l in Path(args.print_results).read_text().splitlines() if l.strip()]
        parse_and_print_results(data)
        return

    from google import genai
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        sys.exit("GEMINI_API_KEY not set")

    client = genai.Client(api_key=gemini_key)

    our_reviews = load_our_predictions(args.our_predictions) if args.our_predictions else {}

    if args.vs == "stanford":
        if not args.our_trajectory_dir or not args.stanford_reviews_dir or not args.data_dir:
            sys.exit("--vs stanford requires --our-trajectory-dir, --stanford-reviews-dir, --data-dir")
        our_reviews = load_our_trajectory_reviews(Path(args.our_trajectory_dir))
        opponent = load_stanford_data(
            reviews_dir=Path(args.stanford_reviews_dir),
            data_dir=Path(args.data_dir),
            ocr_dir=Path(args.ocr_dir) if args.ocr_dir else None,
        )
        # Build pairs using opponent format
        common = set(our_reviews.keys()) & set(opponent.keys())
        common = {pid for pid in common if our_reviews[pid].strip() and opponent[pid]["review"].strip()}
        print(f"Matched papers: {len(common)}")
        pairs = [{"id": pid, "year": 0, "paper_context": opponent[pid]["paper_context"],
                  "our_review": our_reviews[pid], "deepreviewer_review": opponent[pid]["review"]}
                 for pid in sorted(common)]
    else:
        deepreviewer = load_deepreviewer_outputs(args.split)
        pairs = build_comparison_data(our_reviews, deepreviewer, args.split)

    # Resume: skip already-done paper IDs
    out_path = Path(args.out)
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_ids)} already done")

    remaining = [p for p in pairs if p["id"] not in done_ids]
    print(f"To evaluate: {len(remaining)} papers")

    sem = asyncio.Semaphore(args.max_concurrent)
    tasks = [judge_one(item, client, sem) for item in remaining]

    completed = []
    with open(out_path, 'a', encoding='utf-8') as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            completed.append(result)
            if len(completed) % 10 == 0:
                print(f"  judged {len(completed)+len(done_ids)}/{len(pairs)}")

    # Print final results
    all_results = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    parse_and_print_results(all_results)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--our-predictions", type=Path, default=None,
                   help="Predictions JSON (for --vs deepreviewer)")
    p.add_argument("--split", type=int, default=2024, choices=[2024, 2025])
    p.add_argument("--vs", default="deepreviewer", choices=["deepreviewer", "stanford"],
                   help="Which baseline to compare against")
    # Stanford-mode args
    p.add_argument("--our-trajectory-dir", type=str, default=None,
                   help="pass@K results dir with trajectory.json files (for --vs stanford)")
    p.add_argument("--stanford-reviews-dir", type=str, default=None,
                   help="Dir with <paper_id>_review.json files")
    p.add_argument("--data-dir", type=str, default=None,
                   help="pass_at_k_reviewed data dir (for paper body fallback)")
    p.add_argument("--ocr-dir", type=str, default=None,
                   help="OCR markdown dir (preferred paper body)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--print-results", type=str, default=None,
                   help="Just parse and print an existing .jsonl output file")
    args = p.parse_args()

    if not args.print_results and not args.out:
        p.error("--out is required unless --print-results is used")

    from dotenv import load_dotenv
    for env_candidate in [Path("/root/.env"), Path(__file__).resolve().parent.parent.parent.parent / ".env"]:
        if env_candidate.is_file():
            load_dotenv(env_candidate, override=False)
            break

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
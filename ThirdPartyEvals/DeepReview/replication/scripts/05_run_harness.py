"""
05_run_harness.py — Run our E2B/Harbor agent on the DeepReview benchmark.

Reads the DeepReview eval set (from HuggingFace WestlakeNLP/DeepReview-13K or
local eval_set_iclr<year>.json), builds a temporary task directory layout that
our benchmark_pass_at_k.py understands, runs it, then post-processes each
resulting trajectory into a \boxed_review{...} predictions JSON that
03_evaluate.py / evalate.py can score.

DOES NOT modify any harness code. Calls benchmark_pass_at_k.py as a subprocess.

Usage (on remote):
    # ICLR 2024 split, first 10 papers (smoke test)
    python3 05_run_harness.py \
        --split 2024 --limit 10 \
        --results-dir /root/pass_at_k/results_deepreview_2024_10 \
        --trials-dir  /root/pass_at_k/trials_deepreview_2024 \
        --out          /root/pass_at_k/predictions_deepreview_2024.json \
        --max-concurrent 8

    # Full 2024 split
    python3 05_run_harness.py \
        --split 2024 \
        --results-dir /root/pass_at_k/results_deepreview_2024 \
        --trials-dir  /root/pass_at_k/trials_deepreview_2024 \
        --out          /root/pass_at_k/predictions_deepreview_2024.json \
        --max-concurrent 20

    # Evaluate afterwards with evalate.py:
    python3 03_evaluate.py \
        --predictions /root/pass_at_k/predictions_deepreview_2024.json \
        --mode standard
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — relative to this script so it works both locally and on remote.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPLICATION_DIR = SCRIPT_DIR.parent
LOCAL_EVAL_SET = {
    2024: REPLICATION_DIR / "data" / "eval_set_iclr2024.json",
    2025: REPLICATION_DIR / "data" / "eval_set_iclr2025.json",
}

# benchmark_pass_at_k.py can live either at /root/pass_at_k/ (remote) or
# at the project root pass_at_k/ (local clone).
def _parent_safe(p: Path, n: int) -> Path | None:
    """Return p.parents[n] if it exists, else None."""
    try:
        return p.parents[n]
    except IndexError:
        return None


_here = Path(__file__).resolve()
_SEARCH_PATHS = [
    Path("/root/pass_at_k/benchmark_pass_at_k.py"),
    _here.parent / "benchmark_pass_at_k.py",
    *([(_parent_safe(_here, 4) / "pass_at_k" / "benchmark_pass_at_k.py")]
      if _parent_safe(_here, 4) else []),
]
BENCHMARK_SCRIPT = next((p for p in _SEARCH_PATHS if p.is_file()), None)

# DeepReview-specific instruction template — adapted output format.
_INSTR_SEARCH = [
    Path("/root/prompts/deepreview_instruction_template.md"),
    *([(_parent_safe(_here, 4) / "prompts" / "deepreview_instruction_template.md")]
      if _parent_safe(_here, 4) else []),
]
DEEPREVIEW_INSTRUCTION = next((p for p in _INSTR_SEARCH if p.is_file()), None)

# task.toml / Dockerfile — identical to our existing task dirs.
TASK_TOML = """\
schema_version = "1.1"

[metadata]

[verifier]
timeout_sec = 600.0

[agent]

[environment]
build_timeout_sec = 600.0
cpus = 2
memory_mb = 2048
storage_mb = 2048
gpus = 0
allow_internet = true
mcp_servers = []

[verifier.env]
[environment.env]
[solution.env]
"""

DOCKERFILE = """\
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
"""


# ---------------------------------------------------------------------------
# Load eval records
# ---------------------------------------------------------------------------

def load_eval_records(split: int, limit: int | None) -> list[dict]:
    """Load records from local JSON (preferred) or HF dataset."""
    local = LOCAL_EVAL_SET.get(split)
    if local and local.is_file():
        with open(local, encoding="utf-8") as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records from {local}")
    else:
        print(f"Local eval set not found, downloading from HuggingFace...")
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("Install datasets: pip install datasets")
        # The HF test split contains both years (year column is unreliable — always 2025).
        # We match the local CSV split: test[:652] = ICLR 2024, test[652:] = ICLR 2025.
        if split == 2024:
            ds = load_dataset("WestlakeNLP/DeepReview-13K", split="test[:652]")
        else:
            ds = load_dataset("WestlakeNLP/DeepReview-13K", split="test[652:]")

        records = []
        for row in ds:
            inputs = json.loads(row["inputs"]) if isinstance(row["inputs"], str) else row["inputs"]
            # inputs[1]["content"] is the raw LaTeX paper text
            paper_context = inputs[1]["content"] if len(inputs) > 1 else ""
            reviewer_comments = (
                json.loads(row["reviewer_comments"])
                if isinstance(row["reviewer_comments"], str)
                else row["reviewer_comments"]
            )
            # evalate.py does int(r['content']['rating'][0]) — expects string like "5: ..."
            # 2025 records store these as plain ints; normalize to strings so eval works.
            for r in reviewer_comments:
                for field in ("rating", "soundness", "presentation", "contribution", "confidence"):
                    val = r.get("content", {}).get(field)
                    if isinstance(val, int):
                        r["content"][field] = str(val)
            records.append({
                "id": row["id"],
                "title": "",
                "paper_context": paper_context,
                "decision": row["decision"],
                "review": reviewer_comments,
            })
        print(f"Downloaded {len(records)} records from HuggingFace (split={split})")

    if limit:
        records = records[:limit]
        print(f"Limited to {limit} records")
    return records


# ---------------------------------------------------------------------------
# Build task directory for a single paper
# ---------------------------------------------------------------------------

def build_task_dir(record: dict, base_dir: Path) -> Path:
    """
    Create a Harbor-compatible task directory for one DeepReview paper.

    Layout:
        <base_dir>/<paper_id>/
            task.toml
            task_metadata.json
            instruction.md          ← DeepReview-specific prompt
            environment/Dockerfile
            latex/template.tex      ← raw LaTeX from paper_context
    """
    paper_id = record["id"]
    task_dir = base_dir / paper_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # task.toml
    (task_dir / "task.toml").write_text(TASK_TOML)

    # environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(DOCKERFILE)

    # latex/template.tex  — the raw paper LaTeX from HF inputs[1]['content']
    latex_dir = task_dir / "latex"
    latex_dir.mkdir(exist_ok=True)
    (latex_dir / "template.tex").write_text(record["paper_context"], encoding="utf-8")

    # task_metadata.json
    # human_reviews must be populated so benchmark_pass_at_k.py's ≥2-review filter passes.
    # We format each DeepReview reviewer_comment into a markdown string matching our harness
    # convention. The agentic judge uses this for calibration scoring.
    human_reviews = []
    for r in record.get("review", []):
        c = r.get("content", {})
        parts = ["**Scores:**"]
        for field in ("rating", "soundness", "presentation", "contribution", "confidence"):
            val = c.get(field, "")
            if val:
                parts.append(f"- {field.capitalize()}: {val}")
        for field in ("summary", "strengths", "weaknesses", "questions", "suggestions"):
            val = c.get(field, "") or c.get("weakness", "")
            if val and field != "weakness":
                parts.append(f"\n**{field.capitalize()}:**\n{val}")
        human_reviews.append("\n".join(parts))

    metadata = {
        "paper_id": paper_id,
        "title": record.get("title", ""),
        "abstract": "",
        "published": "",
        "human_reviews": human_reviews,
    }
    (task_dir / "task_metadata.json").write_text(json.dumps(metadata, indent=2))

    # instruction.md — use DeepReview-specific template if available, else standard
    if DEEPREVIEW_INSTRUCTION and DEEPREVIEW_INSTRUCTION.is_file():
        shutil.copy2(DEEPREVIEW_INSTRUCTION, task_dir / "instruction.md")
    else:
        # Fallback: minimal inline instruction (should not normally happen)
        fallback = (
            "Read the paper at /app/latex/template.tex carefully.\n"
            "Output ONLY a \\boxed_review{...} block with sections:\n"
            "Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses, "
            "Suggestions, Questions, Confidence, Rating, Decision.\n"
            "Numeric fields must be plain numbers (e.g. 3 or 3.5, not '3 good').\n"
        )
        (task_dir / "instruction.md").write_text(fallback)

    return task_dir


# ---------------------------------------------------------------------------
# Run benchmark_pass_at_k.py as subprocess
# ---------------------------------------------------------------------------

def run_benchmark(
    data_dir: Path,
    results_dir: Path,
    trials_dir: Path,
    max_concurrent: int,
    k: int = 1,
    extra_env: dict | None = None,
    skip_judge: bool = False,
) -> int:
    if BENCHMARK_SCRIPT is None:
        sys.exit("Could not find benchmark_pass_at_k.py — check paths in 05_run_harness.py")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # Load .env if present alongside the benchmark script
    env_file = BENCHMARK_SCRIPT.parent.parent / ".env"
    if env_file.is_file():
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
        env.update({k: v for k, v in os.environ.items() if k not in env})

    if skip_judge:
        # Set a dummy key so the assert passes but every judge API call fails
        # fast and is caught. Trajectories are still saved; we post-process them.
        env["GEMINI_API_KEY"] = "disabled-deepreview-generation-only"

    cmd = [
        sys.executable, str(BENCHMARK_SCRIPT),
        "--data-dir", str(data_dir),
        "--results-dir", str(results_dir),
        "--trials-dir", str(trials_dir),
        "--k", str(k),
        "--max-concurrent", str(max_concurrent),
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


# ---------------------------------------------------------------------------
# Extract review from trajectory and wrap in \boxed_review{}
# ---------------------------------------------------------------------------

_REVIEW_MARKERS = (
    r"\boxed_review{",
    "### Summary", "### Strengths", "### Weaknesses", "### Scores",
    "**Scores**", "**Overall**",
)

# The literal opener string — a backslash followed by boxed_review{
_BOXED_OPENER = "\\boxed_review{"

# Fields evalate.py reads from the \boxed_review block
_NUMERIC_FIELDS = {"Rating", "Soundness", "Presentation", "Contribution", "Confidence"}
_TEXT_FIELDS = {"Summary", "Strengths", "Weaknesses", "Suggestions", "Questions", "Decision"}
_ALL_FIELDS = _NUMERIC_FIELDS | _TEXT_FIELDS


def _extract_boxed_review(text: str) -> str | None:
    """Return the \boxed_review{...} string if already present.

    Uses the same split logic as evalate.py:get_pred() so the result is
    guaranteed to parse correctly.
    """
    if _BOXED_OPENER not in text:
        return None
    inner = text.split(_BOXED_OPENER)[-1].split("\n}")[0]
    return _BOXED_OPENER + inner + "\n}"


def _parse_native_review(text: str) -> dict | None:
    """
    Parse our native review format (### Summary, ### Scores, etc.)
    into a field dict so we can rebuild a \boxed_review{} block.
    Returns None if the text doesn't look like a structured review.
    """
    fields: dict[str, str] = {}

    # Extract scores block: looks for lines like "**Rating**: X/10" or "- **Overall**: X/10"
    score_patterns = {
        "Rating": r"\*{0,2}Overall\*{0,2}[:\s]+(\d+(?:\.\d+)?)\s*/\s*10",
        "Soundness": r"\*{0,2}Soundness\*{0,2}[:\s]+(\d+(?:\.\d+)?)\s*/\s*4",
        "Presentation": r"\*{0,2}Presentation\*{0,2}[:\s]+(\d+(?:\.\d+)?)\s*/\s*4",
        "Contribution": r"\*{0,2}Contribution\*{0,2}[:\s]+(\d+(?:\.\d+)?)\s*/\s*4",
        "Confidence": r"\*{0,2}Confidence\*{0,2}[:\s]+(\d+(?:\.\d+)?)\s*/\s*5",
    }
    for field, pat in score_patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            fields[field] = m.group(1)

    # Decision
    m = re.search(r"\*{0,2}Decision\*{0,2}[:\s]+(Accept|Reject)", text, re.IGNORECASE)
    if m:
        fields["Decision"] = m.group(1).capitalize()
    elif "Rating" in fields:
        # Infer from rating
        fields["Decision"] = "Accept" if float(fields["Rating"]) >= 6 else "Reject"

    # Section bodies — extract between ### headers
    section_re = re.compile(r"^###\s+(\w[\w &]+)\s*$", re.MULTILINE)
    matches = list(section_re.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Map our section names to DeepReview fields
        mapping = {
            "Summary": "Summary",
            "Strengths": "Strengths",
            "Weaknesses": "Weaknesses",
            "Questions": "Questions",
            "Limitations": "Suggestions",  # best match
        }
        if name in mapping:
            fields[mapping[name]] = body

    if not fields.get("Rating"):
        return None
    return fields


def _build_boxed_review(fields: dict) -> str:
    """Assemble a \boxed_review{...} block from a field dict."""
    lines = ["\\boxed_review{"]
    ordered = [
        "Summary", "Soundness", "Presentation", "Contribution",
        "Strengths", "Weaknesses", "Suggestions", "Questions",
        "Confidence", "Rating", "Decision",
    ]
    for f in ordered:
        val = fields.get(f, "N/A")
        lines.append(f"## {f}:\n\n{val}\n")
    lines.append("}")
    return "\n".join(lines)


def extract_review_from_trajectory(trajectory_path: Path) -> str:
    """
    Extract the final review from a trajectory.json and return it as a
    \boxed_review{...} string ready for evalate.py.

    Priority:
    1. Agent already output a \boxed_review{} block → use as-is.
    2. Agent output our native format (### Summary / ### Scores) → convert.
    3. Return empty string (row will be skipped by evalate.py).
    """
    try:
        data = json.loads(trajectory_path.read_text())
    except Exception as e:
        print(f"  WARN: could not read {trajectory_path}: {e}")
        return ""

    agent_msgs = [
        step["message"]
        for step in data.get("steps", [])
        if step.get("source") == "agent"
        and step.get("message")
        and len(step["message"]) > 200
    ]
    if not agent_msgs:
        return ""

    # Find the last message that looks like a structured review
    for msg in reversed(agent_msgs):
        if any(marker in msg for marker in _REVIEW_MARKERS):
            # Already has \boxed_review{}?
            boxed = _extract_boxed_review(msg)
            if boxed:
                return boxed
            # Try to convert from native format
            fields = _parse_native_review(msg)
            if fields:
                return _build_boxed_review(fields)

    return ""


# ---------------------------------------------------------------------------
# Build predictions JSON from results dir
# ---------------------------------------------------------------------------

def build_predictions(
    records: list[dict],
    results_dir: Path,
    mode_key: str = "pred_standard_mode",
) -> list[dict]:
    """
    Walk results_dir for completed trajectories, extract reviews, merge back
    into the eval records, and return the full predictions list.
    """
    # Index original records by id
    by_id = {r["id"]: dict(r) for r in records}

    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        paper_id = paper_dir.name
        if paper_id not in by_id:
            continue

        # Look for attempt_0/trajectory.json (k=1 run)
        traj = None
        for attempt_dir in sorted(paper_dir.iterdir()):
            if attempt_dir.is_dir() and attempt_dir.name.startswith("attempt_"):
                candidate = attempt_dir / "trajectory.json"
                if candidate.is_file():
                    traj = candidate
                    break

        if traj is None:
            print(f"  WARN: no trajectory for {paper_id}")
            by_id[paper_id][mode_key] = ""
            continue

        review_text = extract_review_from_trajectory(traj)
        if not review_text:
            print(f"  WARN: could not extract review from {traj}")
        by_id[paper_id][mode_key] = review_text

    return list(by_id.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run our E2B/Harbor agent on DeepReview benchmark")
    ap.add_argument("--split", type=int, default=2024, choices=[2024, 2025],
                    help="Which ICLR year split to evaluate")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of papers (for smoke tests)")
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="Where benchmark_pass_at_k.py writes per-paper results")
    ap.add_argument("--trials-dir", required=True, type=Path,
                    help="Where Harbor trial artefacts are stored")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output predictions JSON path (fed to 03_evaluate.py)")
    ap.add_argument("--max-concurrent", type=int, default=10,
                    help="Concurrent E2B sandboxes")
    ap.add_argument("--task-data-dir", type=Path, default=None,
                    help="Override temp dir for task dirs (default: auto tempdir, cleaned up after)")
    ap.add_argument("--skip-run", action="store_true",
                    help="Skip running the benchmark; only post-process existing results")
    ap.add_argument("--skip-judge", action="store_true",
                    help="Skip our agentic judge (Gemini scoring). Use this for DeepReview "
                         "generation-only runs — scoring is done by evalate.py afterwards.")
    ap.add_argument("--mode-key", default="pred_standard_mode",
                    choices=["pred_fast_mode", "pred_standard_mode", "pred_best_mode"],
                    help="Key to write into predictions JSON (evalate.py reads this)")
    args = ap.parse_args()

    print(f"=== DeepReview Harness Runner ===")
    print(f"  Split: ICLR {args.split}")
    print(f"  Benchmark script: {BENCHMARK_SCRIPT}")
    print(f"  Instruction template: {DEEPREVIEW_INSTRUCTION}")
    print()

    records = load_eval_records(args.split, args.limit)

    # Build task dirs
    cleanup_task_dir = False
    if args.task_data_dir:
        task_base = args.task_data_dir
        task_base.mkdir(parents=True, exist_ok=True)
    else:
        task_base = Path(tempfile.mkdtemp(prefix="deepreview_tasks_"))
        cleanup_task_dir = True

    print(f"Building task dirs in {task_base} ...")
    for rec in records:
        build_task_dir(rec, task_base)
    print(f"  Built {len(records)} task dirs")

    # Run benchmark
    if not args.skip_run:
        rc = run_benchmark(
            data_dir=task_base,
            results_dir=args.results_dir,
            trials_dir=args.trials_dir,
            max_concurrent=args.max_concurrent,
            k=1,
            skip_judge=args.skip_judge,
        )
        if rc != 0:
            print(f"WARNING: benchmark_pass_at_k.py exited with code {rc}")
    else:
        print("--skip-run: skipping benchmark execution, post-processing existing results only")

    # Post-process trajectories → predictions JSON
    print(f"\nPost-processing trajectories from {args.results_dir} ...")
    predictions = build_predictions(records, args.results_dir, mode_key=args.mode_key)

    filled = sum(1 for p in predictions if p.get(args.mode_key))
    print(f"  Extracted reviews: {filled}/{len(predictions)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"  Wrote {args.out}")

    if cleanup_task_dir:
        shutil.rmtree(task_base, ignore_errors=True)

    print(f"\nNext: evaluate with:")
    print(f"  python3 {SCRIPT_DIR}/03_evaluate.py --predictions {args.out} --mode standard")


if __name__ == "__main__":
    main()
"""calibrate_judge.py — measure LLM-judge / human agreement on PeerJudge criteria.

Inputs: a JSON file where each record is one model-written review that BOTH a
human annotator and the LLM judge have scored against the same paper + human
reference reviews. Use `--sample` to auto-score a set of trajectories with the
current LLM judge, then hand-edit the file to add `manual_scores`.

Record shape (one per line in JSONL, or a top-level list in JSON):

    {
      "review_id": "1612.00472_attempt0",
      "paper_title": "...",
      "paper_abstract": "...",
      "human_reviews": ["...", "...", "..."],
      "model_review": "...",
      "judge_scores": {                  # from LLM judge
        "comprehension": {"score": 1},
        "substance_and_specificity": {"score": 1},
        "insight": {"score": 0.5},
        "issue_overlap": {"score": 0.7},
        "fabrication": {"score": 1.0},
        "calibration_pairwise": {"score": 0.5}
      },
      "manual_scores": {                 # filled in by human grader
        "comprehension": 1,
        ...
      }
    }

Metrics reported per criterion:
  - Binary (comprehension, substance): Cohen's κ, % agreement.
  - Continuous (insight, issue_overlap, fabrication,
    calibration_pairwise): Pearson r, MAE, Spearman ρ.

Usage:
    # Step 1: bootstrap a calibration file from an existing results dir.
    python calibrate_judge.py bootstrap \
        --results-dir /root/pass_at_k/results_v2 \
        --sample 20 \
        --out calibration.jsonl

    # Step 2: open calibration.jsonl, fill in `manual_scores` for each entry.

    # Step 3: compute agreement metrics.
    python calibrate_judge.py analyze --in calibration.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any


# Binary criteria: scored 0 or 1.
BINARY_CRITERIA = ["comprehension", "substance_and_specificity"]

# Continuous criteria: scored in [0, 1].
CONTINUOUS_CRITERIA = [
    "insight",
    "issue_overlap",
    "fabrication",
    "calibration_pairwise",
]

ALL_CRITERIA = BINARY_CRITERIA + CONTINUOUS_CRITERIA


# ---------------------------------------------------------------------------
# Loading / normalization
# ---------------------------------------------------------------------------

def _load_records(path: Path) -> list[dict]:
    """Accept either .jsonl or a JSON array."""
    text = path.read_text()
    # Try JSONL first
    records: list[dict] = []
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_score(node: Any) -> float | None:
    if node is None:
        return None
    if isinstance(node, dict):
        v = node.get("score")
    else:
        v = node
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _paired_scores(records: list[dict], criterion: str) -> tuple[list[float], list[float]]:
    """Return (judge_values, manual_values) paired across records where BOTH exist."""
    j_vals, m_vals = [], []
    for r in records:
        jn = (r.get("judge_scores") or {}).get(criterion)
        mn = (r.get("manual_scores") or {}).get(criterion)
        j = _extract_score(jn)
        m = _extract_score(mn)
        if j is None or m is None:
            continue
        j_vals.append(j)
        m_vals.append(m)
    return j_vals, m_vals


# ---------------------------------------------------------------------------
# Statistics (zero deps — no scipy/sklearn required)
# ---------------------------------------------------------------------------

def cohen_kappa(a: list[float], b: list[float]) -> float:
    """Cohen's κ for binary raters. Treats any value > 0.5 as 1, else 0."""
    if not a:
        return float("nan")
    a_bin = [1 if x > 0.5 else 0 for x in a]
    b_bin = [1 if x > 0.5 else 0 for x in b]
    n = len(a_bin)
    po = sum(x == y for x, y in zip(a_bin, b_bin)) / n
    pa1 = sum(a_bin) / n
    pb1 = sum(b_bin) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def pearson_r(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return float("nan")
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return float("nan")
    return num / (da * db)


def spearman_rho(a: list[float], b: list[float]) -> float:
    """Rank-correlation Spearman ρ via Pearson on ranks."""
    if len(a) < 2:
        return float("nan")

    def _ranks(xs: list[float]) -> list[float]:
        indexed = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[indexed[j + 1]] == xs[indexed[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    return pearson_r(_ranks(a), _ranks(b))


def mae(a: list[float], b: list[float]) -> float:
    if not a:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def pct_agreement(a: list[float], b: list[float], tol: float = 0.01) -> float:
    if not a:
        return float("nan")
    return sum(abs(x - y) <= tol for x, y in zip(a, b)) / len(a)


# ---------------------------------------------------------------------------
# Subcommand: bootstrap
# ---------------------------------------------------------------------------

def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Scan a results directory, pick N attempts stratified by reward, and emit
    a calibration JSONL with judge_scores pre-filled and manual_scores=null."""
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    # Walk results/paper_id/attempt_N/
    attempts: list[dict] = []
    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        for attempt_dir in sorted(paper_dir.iterdir()):
            rj = attempt_dir / "result.json"
            if not rj.is_file():
                continue
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            if r.get("status") != "success":
                continue
            attempts.append({"paper_id": paper_dir.name, "attempt_dir": attempt_dir, "result": r})

    if not attempts:
        print("error: no successful attempts found", file=sys.stderr)
        sys.exit(2)

    # Stratified sample across reward bins for coverage of quality bands.
    attempts.sort(key=lambda a: a["result"].get("reward", 0.0))
    n = min(args.sample, len(attempts))
    if n >= len(attempts):
        picked = attempts
    else:
        # Pick evenly-spaced indices to span reward distribution.
        step = len(attempts) / n
        idx = [int(i * step) for i in range(n)]
        picked = [attempts[i] for i in idx]
        # Add a little randomness within stratum
        random.Random(args.seed).shuffle(picked)

    out_path = Path(args.out)
    written = 0
    with out_path.open("w") as f:
        for a in picked:
            paper_id = a["paper_id"]
            result = a["result"]
            attempt_dir = a["attempt_dir"]
            traj = attempt_dir / "trajectory.json"
            model_review = ""
            if traj.is_file():
                try:
                    tdata = json.loads(traj.read_text())
                    for step in reversed(tdata.get("steps", []) or []):
                        if step.get("source") == "agent" and len(step.get("message", "") or "") > 200:
                            model_review = step["message"]
                            break
                except Exception:
                    pass

            # Paper metadata (title, abstract, human reviews) — look up via
            # the data dir if --data-dir is provided.
            meta: dict = {}
            if args.data_dir:
                meta_path = Path(args.data_dir) / paper_id / "task_metadata.json"
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text())
                    except Exception:
                        pass

            rec = {
                "review_id": f"{paper_id}_attempt{result.get('attempt', 0)}",
                "paper_id": paper_id,
                "paper_title": meta.get("title", ""),
                "paper_abstract": meta.get("abstract", ""),
                "human_reviews": meta.get("human_reviews", []),
                "model_review": model_review,
                "judge_scores": result.get("scores", {}),
                "manual_scores": {k: None for k in ALL_CRITERIA},
                "notes": "",
            }
            f.write(json.dumps(rec) + "\n")
            written += 1

    print(f"wrote {written} records to {out_path}")
    print(f"next: open {out_path} and fill in each record's `manual_scores`, then run: "
          f"python calibrate_judge.py analyze --in {out_path}")


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    records = _load_records(Path(args.in_path))
    n_total = len(records)
    n_with_manual = sum(
        1 for r in records
        if any(_extract_score((r.get("manual_scores") or {}).get(c)) is not None for c in ALL_CRITERIA)
    )
    print(f"loaded {n_total} records ({n_with_manual} with any manual scores)\n")

    rows = []
    for crit in ALL_CRITERIA:
        j, m = _paired_scores(records, crit)
        if not j:
            rows.append((crit, 0, "n/a", "n/a", "n/a", "n/a", "n/a"))
            continue
        if crit in BINARY_CRITERIA:
            kappa = cohen_kappa(j, m)
            agree = pct_agreement(j, m, tol=0.5)
            rows.append((crit, len(j), f"{kappa:+.3f}", f"{agree*100:.1f}%", "—", "—", "—"))
        else:
            r_pear = pearson_r(j, m)
            rho = spearman_rho(j, m)
            err = mae(j, m)
            rows.append((crit, len(j), "—", "—", f"{r_pear:+.3f}", f"{rho:+.3f}", f"{err:.3f}"))

    hdr = ("criterion", "n", "κ", "agree", "pearson", "spearman", "MAE")
    widths = [max(len(str(r[i])) for r in ([hdr] + rows)) for i in range(len(hdr))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*hdr))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))

    # Agreement on overall reward (if available)
    j_rewards, m_rewards = [], []
    for r in records:
        jr = (r.get("judge_scores") or {}).get("reward")
        # Manual reward: compute from manual_scores with the same weights
        manual = r.get("manual_scores") or {}
        if jr is None or any(manual.get(c) is None for c in ALL_CRITERIA):
            continue
        j_rewards.append(float(jr))
        m_rewards.append(
            0.05 * float(manual["comprehension"])
            + 0.05 * float(manual["substance_and_specificity"])
            + 0.15 * float(manual["insight"])
            + 0.25 * float(manual["issue_overlap"])
            + 0.20 * float(manual["fabrication"])
            + 0.25 * float(manual["calibration_pairwise"])
        )
    if j_rewards:
        print(f"\noverall reward (n={len(j_rewards)}):  "
              f"pearson={pearson_r(j_rewards, m_rewards):+.3f}  "
              f"spearman={spearman_rho(j_rewards, m_rewards):+.3f}  "
              f"MAE={mae(j_rewards, m_rewards):.3f}")

    # Top disagreements
    print("\n=== top disagreements (by |judge - manual| on reward) ===")
    diffs = []
    for r in records:
        jr = (r.get("judge_scores") or {}).get("reward")
        if jr is None:
            continue
        manual = r.get("manual_scores") or {}
        if any(manual.get(c) is None for c in ALL_CRITERIA):
            continue
        mr = (
            0.05 * float(manual["comprehension"])
            + 0.05 * float(manual["substance_and_specificity"])
            + 0.15 * float(manual["insight"])
            + 0.25 * float(manual["issue_overlap"])
            + 0.20 * float(manual["fabrication"])
            + 0.25 * float(manual["calibration_pairwise"])
        )
        diffs.append((abs(float(jr) - mr), r.get("review_id", "?"), float(jr), mr))
    diffs.sort(reverse=True)
    for delta, rid, jr, mr in diffs[:5]:
        print(f"  {rid}:  judge={jr:.3f}  manual={mr:.3f}  Δ={delta:+.3f}")

    # Criterion-level target thresholds
    print("\n=== target thresholds (ICLR-grade evidence) ===")
    print("  binary criteria: Cohen's κ ≥ 0.60")
    print("  continuous criteria: Pearson r ≥ 0.70 AND MAE ≤ 0.15")
    print("  overall reward: MAE ≤ 0.10")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="PeerJudge calibration harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="Build a calibration JSONL from an existing results dir")
    b.add_argument("--results-dir", required=True)
    b.add_argument("--data-dir", help="Paper task dir (for title/abstract/human_reviews lookup)")
    b.add_argument("--sample", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="calibration.jsonl")
    b.set_defaults(fn=cmd_bootstrap)

    a = sub.add_parser("analyze", help="Compute judge/human agreement metrics")
    a.add_argument("--in", dest="in_path", required=True)
    a.set_defaults(fn=cmd_analyze)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
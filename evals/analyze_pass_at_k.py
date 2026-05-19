"""
analyze_pass_at_k.py — Compute pass@K metrics from benchmark results

Implements the unbiased pass@K estimator from Chen et al. (2021) "Evaluating
Large Language Models Trained on Code" (Codex paper):

    pass@k = 1 - C(n-c, k) / C(n, k)

where n = total attempts, c = number of correct attempts, k = samples drawn.

Also computes:
  - Per-paper reward statistics
  - Distribution of rewards across attempts
  - Criterion-level breakdown
  - Bootstrap confidence intervals

Usage:
    python analyze_pass_at_k.py --results-dir /root/pass_at_k/results
    python analyze_pass_at_k.py --results-dir /root/pass_at_k/results --threshold 0.5
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.

    n: total number of samples per task
    c: number of correct samples (reward >= threshold)
    k: number of samples to draw

    Returns: probability that at least one of k samples is correct
    """
    if n - c < k:
        return 1.0
    # Use log-space to avoid overflow: 1 - exp(log(C(n-c,k)) - log(C(n,k)))
    # C(n-c,k)/C(n,k) = prod_{i=0}^{k-1} (n-c-i)/(n-i)
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_ratio)


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all per-attempt result.json files, grouped by paper_id."""
    results_by_paper = defaultdict(list)

    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        # Skip summary files
        if paper_dir.name.startswith("pass_at_k_summary"):
            continue

        paper_id = paper_dir.name

        # Look for attempt_*/result.json
        for attempt_dir in sorted(paper_dir.iterdir()):
            if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt_"):
                continue
            result_file = attempt_dir / "result.json"
            if result_file.exists():
                try:
                    result = json.loads(result_file.read_text())
                    results_by_paper[paper_id].append(result)
                except Exception as e:
                    print(f"  WARNING: failed to load {result_file}: {e}")

        # Also check aggregated.json as fallback
        agg_file = paper_dir / "aggregated.json"
        if agg_file.exists() and paper_id not in results_by_paper:
            try:
                agg = json.loads(agg_file.read_text())
                for i, attempt in enumerate(agg.get("attempts", [])):
                    results_by_paper[paper_id].append(attempt)
            except Exception:
                pass

    return dict(results_by_paper)


def analyze(results_dir: Path, threshold: float, k_values: list[int]):
    """Run the full pass@K analysis."""
    results_by_paper = load_results(results_dir)

    if not results_by_paper:
        print("ERROR: No results found!")
        sys.exit(1)

    num_papers = len(results_by_paper)
    all_rewards = []
    per_paper_stats = []

    print(f"\n{'='*70}")
    print(f"PASS@K ANALYSIS")
    print(f"{'='*70}")
    print(f"  Results dir: {results_dir}")
    print(f"  Papers: {num_papers}")
    print(f"  Threshold: {threshold}")
    print(f"  K values: {k_values}")
    print()

    # Per-paper statistics
    for paper_id in sorted(results_by_paper.keys()):
        attempts = results_by_paper[paper_id]
        rewards = [a.get("reward", 0.0) for a in attempts]
        all_rewards.extend(rewards)

        n = len(rewards)
        c = sum(1 for r in rewards if r >= threshold)

        stats = {
            "paper_id": paper_id,
            "n": n,
            "c": c,
            "rewards": rewards,
            "mean_reward": np.mean(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "std_reward": np.std(rewards),
        }

        # Compute pass@k for each k
        for k in k_values:
            if k <= n:
                stats[f"pass@{k}"] = pass_at_k(n, c, k)
            else:
                stats[f"pass@{k}"] = None

        per_paper_stats.append(stats)

    # Aggregate pass@k across all papers
    print(f"{'='*70}")
    print(f"PASS@K RESULTS (threshold={threshold})")
    print(f"{'='*70}")
    print()

    for k in k_values:
        key = f"pass@{k}"
        values = [s[key] for s in per_paper_stats if s[key] is not None]
        if values:
            mean_pass = np.mean(values)
            std_pass = np.std(values)
            # Bootstrap CI
            bootstrap_means = []
            rng = np.random.default_rng(42)
            for _ in range(10000):
                sample = rng.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))
            ci_low = np.percentile(bootstrap_means, 2.5)
            ci_high = np.percentile(bootstrap_means, 97.5)

            print(f"  pass@{k}: {mean_pass:.4f} (std={std_pass:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}])")
        else:
            print(f"  pass@{k}: N/A (not enough attempts)")

    # Overall reward statistics
    print()
    print(f"{'='*70}")
    print(f"REWARD STATISTICS")
    print(f"{'='*70}")
    print(f"  Total attempts: {len(all_rewards)}")
    print(f"  Mean reward: {np.mean(all_rewards):.4f}")
    print(f"  Std reward: {np.std(all_rewards):.4f}")
    print(f"  Median reward: {np.median(all_rewards):.4f}")
    print(f"  Min reward: {min(all_rewards):.4f}")
    print(f"  Max reward: {max(all_rewards):.4f}")
    print()

    # Reward distribution
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist, _ = np.histogram(all_rewards, bins=bins)
    print(f"  Reward distribution:")
    for i in range(len(bins) - 1):
        bar = "#" * hist[i]
        print(f"    [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:4d} {bar}")

    # Criterion-level breakdown
    print()
    print(f"{'='*70}")
    print(f"CRITERION BREAKDOWN (across all attempts)")
    print(f"{'='*70}")

    criteria = ["comprehension", "substance_and_specificity", "insight", "issue_overlap", "calibration"]
    for criterion in criteria:
        scores = []
        for paper_id, attempts in results_by_paper.items():
            for a in attempts:
                s = a.get("scores", {})
                if isinstance(s, dict) and criterion in s:
                    val = s[criterion]
                    if isinstance(val, dict):
                        scores.append(float(val.get("score", 0)))
                    else:
                        scores.append(float(val))
        if scores:
            print(f"  {criterion}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, n={len(scores)}")
        else:
            print(f"  {criterion}: no data")

    # Per-paper detail table
    print()
    print(f"{'='*70}")
    print(f"PER-PAPER DETAIL (sorted by max reward)")
    print(f"{'='*70}")
    print(f"  {'Paper ID':<25} {'n':>3} {'c':>3} {'Mean':>6} {'Max':>6} {'Std':>6} {'pass@1':>7} {'pass@4':>7}")
    print(f"  {'-'*25} {'---':>3} {'---':>3} {'------':>6} {'------':>6} {'------':>6} {'-------':>7} {'-------':>7}")

    sorted_stats = sorted(per_paper_stats, key=lambda s: s["max_reward"], reverse=True)
    for s in sorted_stats:
        p1 = f"{s.get('pass@1', 0):.3f}" if s.get("pass@1") is not None else "N/A"
        p4 = f"{s.get('pass@4', 0):.3f}" if s.get("pass@4") is not None else "N/A"
        print(
            f"  {s['paper_id']:<25} {s['n']:>3} {s['c']:>3} "
            f"{s['mean_reward']:>6.3f} {s['max_reward']:>6.3f} {s['std_reward']:>6.3f} "
            f"{p1:>7} {p4:>7}"
        )

    # Save full analysis as JSON
    analysis = {
        "threshold": threshold,
        "k_values": k_values,
        "num_papers": num_papers,
        "total_attempts": len(all_rewards),
        "pass_at_k": {},
        "reward_stats": {
            "mean": float(np.mean(all_rewards)),
            "std": float(np.std(all_rewards)),
            "median": float(np.median(all_rewards)),
            "min": float(min(all_rewards)),
            "max": float(max(all_rewards)),
        },
        "per_paper": per_paper_stats,
    }

    for k in k_values:
        key = f"pass@{k}"
        values = [s[key] for s in per_paper_stats if s[key] is not None]
        if values:
            analysis["pass_at_k"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    analysis_path = results_dir / "pass_at_k_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, default=float))
    print(f"\nFull analysis saved to: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pass@K results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="Reward threshold for 'pass'")
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 2, 3, 4],
        help="K values to compute pass@K for",
    )
    args = parser.parse_args()

    analyze(Path(args.results_dir), args.threshold, args.k_values)


if __name__ == "__main__":
    main()
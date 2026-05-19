"""run_experiment.py — end-to-end orchestrator for multi-model pass@K experiments.

Wraps: dataset health check → provider switch → stale-sandbox cleanup →
benchmark launch → cross-model comparison.

Example:
    python pass_at_k/run_experiment.py \\
        --models deepinfra-qwen3p6-35b-a3b deepinfra-glm-5p1 \\
        --data-dir /root/data/pass_at_k_papers_reviewed \\
        --n-papers 5 \\
        --k 4 \\
        --max-concurrent 40 \\
        --exp-root /root/pass_at_k/exp \\
        --name multi-n5k4

    # Explicit paper list
    python pass_at_k/run_experiment.py \\
        --models deepinfra-qwen3p6-35b-a3b \\
        --data-dir /root/data/pass_at_k_papers_reviewed \\
        --tasks 1612.00472 1703.05698 1705.07136 \\
        --k 4 --max-concurrent 40 \\
        --name qwen-smoke

Layout produced:
    <exp-root>/<name>/
        scope.json                          # picked paper IDs + meta
        <model-slug>/
            results/  trials/  log.txt
        comparison.json
        comparison.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_providers(yaml_path: Path) -> dict:
    """Tiny YAML reader — same flat-scalar subset as switch_provider.sh."""
    text = yaml_path.read_text()
    entries: dict[str, dict] = {}
    current = None
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_\-]+):\s*$", stripped)
        if m:
            current = m.group(1)
            entries[current] = {}
            continue
        if current:
            m = re.match(r"^\s+([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", stripped)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if val == "|":
                    entries[current][key] = "<multiline>"
                else:
                    entries[current][key] = val.strip("\"'")
    return entries


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model.split("/")[-1]).strip("-").lower() or "model"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def pick_papers(data_dir: Path, n: int | None, explicit: list[str] | None) -> list[str]:
    if explicit:
        # Validate each
        for pid in explicit:
            if not (data_dir / pid / "task_metadata.json").is_file():
                sys.exit(f"error: {pid} not found in {data_dir}")
        return explicit
    if n is None:
        sys.exit("error: provide --n-papers or --tasks")
    available = sorted(p.name for p in data_dir.iterdir()
                       if p.is_dir() and (p / "task_metadata.json").is_file())
    if len(available) < n:
        sys.exit(f"error: only {len(available)} papers in {data_dir}, need {n}")
    return available[:n]


def dataset_health_check(data_dir: Path, paper_ids: list[str], min_chars: int = 500) -> list[str]:
    """Return paper IDs whose human_reviews look broken (<500 chars each)."""
    broken = []
    for pid in paper_ids:
        meta_path = data_dir / pid / "task_metadata.json"
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            broken.append(pid)
            continue
        hr = meta.get("human_reviews") or []
        if not hr or any(len(str(r)) < min_chars for r in hr):
            broken.append(pid)
    return broken


def regen_reviews_for(data_dir: Path, paper_ids: list[str]) -> None:
    """Call regen_human_reviews.py for a specific set of IDs."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "regen_human_reviews.py"),
        "--data-dir", str(data_dir),
        "--papers", *paper_ids,
    ]
    print(f"==> regenerating reviews for {len(paper_ids)} papers:")
    for line in _stream(cmd):
        print(f"    {line}")


def switch_provider(entry_name: str) -> dict:
    """Shell out to switch_provider.sh <entry>. Returns parsed upstream config."""
    script = SCRIPT_DIR / "switch_provider.sh"
    if not script.is_file():
        sys.exit(f"error: {script} missing")
    cmd = ["bash", str(script), entry_name]
    print(f"==> switching provider to {entry_name}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        print(f"    {line}")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"switch_provider.sh failed with code {proc.returncode}")
    # After switch, .env contains the new PROXY_* — read them
    env_path = Path(os.environ.get("PROXY_ENV_FILE", "/root/.env"))
    upstream: dict = {}
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            m = re.match(r"^(PROXY_[A-Z_]+)=(.*)$", line.strip())
            if m:
                upstream[m.group(1)] = m.group(2).strip().strip("\"'")
    return upstream


def kill_stale_sandboxes() -> int:
    """Clean up leftover E2B boxes between model runs."""
    env_path = Path(os.environ.get("PROXY_ENV_FILE", "/root/.env"))
    e2b_key = ""
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            if line.startswith("E2B_API_KEY="):
                e2b_key = line.split("=", 1)[1].strip().strip("\"'")
                break
    if not e2b_key:
        e2b_key = os.environ.get("E2B_API_KEY", "")
    if not e2b_key:
        print("    (no E2B_API_KEY, skipping cleanup)")
        return 0
    env = dict(os.environ)
    env["E2B_API_KEY"] = e2b_key
    src = (
        "from e2b import Sandbox\n"
        "p = Sandbox.list(); k = 0\n"
        "try:\n"
        "    while True:\n"
        "        items = p.next_items()\n"
        "        if not items: break\n"
        "        for s in items: Sandbox.kill(s.sandbox_id); k += 1\n"
        "except Exception: pass\n"
        "print(k)\n"
    )
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env)
    try:
        return int(proc.stdout.strip() or 0)
    except ValueError:
        return 0


def launch_benchmark(
    *,
    model_entry: str,
    upstream: dict,
    data_dir: Path,
    results_dir: Path,
    trials_dir: Path,
    log_path: Path,
    paper_ids: list[str],
    k: int,
    max_concurrent: int,
    benchmark_cwd: Path,
) -> None:
    """Run benchmark_pass_at_k.py as a subprocess and stream its log."""
    results_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # Propagate proxy config to benchmark (for model label, trial name)
    for k_env, v_env in upstream.items():
        env[k_env] = v_env

    cmd = [
        sys.executable, "-u", str(benchmark_cwd / "benchmark_pass_at_k.py"),
        "--data-dir", str(data_dir),
        "--results-dir", str(results_dir),
        "--trials-dir", str(trials_dir),
        "--k", str(k),
        "--max-concurrent", str(max_concurrent),
        "--tasks", *paper_ids,
    ]
    start = time.time()
    print(f"==> launching benchmark for {model_entry}")
    print(f"    results: {results_dir}")
    print(f"    log:     {log_path}")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=str(benchmark_cwd), env=env)
    # Poll until done, printing a heartbeat
    while proc.poll() is None:
        time.sleep(20)
        done = sum(1 for _ in results_dir.rglob("result.json"))
        print(f"    [{int(time.time()-start)}s] completed attempts: {done} / {len(paper_ids)*k}")
    rc = proc.wait()
    if rc != 0:
        sys.exit(f"benchmark failed with code {rc} — see {log_path}")
    print(f"==> benchmark done for {model_entry} in {int(time.time()-start)}s")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Standard HumanEval pass@k unbiased estimator.

    n: total attempts per problem, c: num correct among them, k: pass@k.
    """
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    import math
    return 1 - (math.comb(n - c, k) / math.comb(n, k))


def aggregate_model(results_dir: Path, threshold: float = 0.7) -> dict:
    """Aggregate per-model results under the new 7-criterion PeerJudge rubric."""
    attempts_by_paper: dict[str, list[dict]] = {}
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
            attempts_by_paper.setdefault(paper_dir.name, []).append(r)

    papers = sorted(attempts_by_paper.keys())
    total_attempts = sum(len(v) for v in attempts_by_paper.values())
    all_rewards = [a.get("reward", 0.0) for v in attempts_by_paper.values() for a in v]

    # pass@k for various k (only valid when >= k attempts per paper)
    per_paper_k = {}
    all_ks = {len(v) for v in attempts_by_paper.values() if v}
    for k in sorted(all_ks):
        # Treat "pass" = reward >= threshold
        hits = 0
        n_papers = 0
        for pid, attempts in attempts_by_paper.items():
            if len(attempts) < k:
                continue
            n = len(attempts)
            c = sum(1 for a in attempts if a.get("reward", 0.0) >= threshold)
            hits += _pass_at_k_estimator(n, c, k)
            n_papers += 1
        if n_papers:
            per_paper_k[f"pass@{k}"] = hits / n_papers

    # Per-criterion averages (new rubric)
    crit_keys = [
        "comprehension", "substance_and_specificity", "insight",
        "issue_overlap", "missed_weakness", "fabrication", "calibration_pairwise",
    ]
    crit_sums = {k: 0.0 for k in crit_keys}
    crit_n = {k: 0 for k in crit_keys}
    for v in attempts_by_paper.values():
        for a in v:
            scores = a.get("scores") or {}
            for ck in crit_keys:
                node = scores.get(ck)
                if node is None:
                    continue
                val = node.get("score") if isinstance(node, dict) else node
                try:
                    crit_sums[ck] += float(val)
                    crit_n[ck] += 1
                except (TypeError, ValueError):
                    continue
    crit_avg = {ck: (crit_sums[ck] / crit_n[ck] if crit_n[ck] else None) for ck in crit_keys}

    return {
        "papers": papers,
        "n_papers": len(papers),
        "total_attempts": total_attempts,
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "median_reward": sorted(all_rewards)[len(all_rewards)//2] if all_rewards else 0.0,
        "max_reward": max(all_rewards) if all_rewards else 0.0,
        "min_reward": min(all_rewards) if all_rewards else 0.0,
        "pass_at_k": per_paper_k,
        "criteria_avg": crit_avg,
        "threshold": threshold,
    }


def comparison_markdown(exp_name: str, model_results: dict[str, dict], scope: dict) -> str:
    lines = [f"# Experiment: {exp_name}", ""]
    lines.append(f"- **Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- **Papers (n={len(scope['paper_ids'])}):** {', '.join(scope['paper_ids'])}")
    lines.append(f"- **K:** {scope['k']}")
    lines.append(f"- **Pass threshold:** reward ≥ {scope['threshold']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    hdr = ["model", "mean_r", "median_r", "min_r", "max_r"]
    # Collect pass@k keys across models
    pass_keys = sorted({k for r in model_results.values() for k in r.get("pass_at_k", {})})
    hdr += pass_keys
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for m, r in model_results.items():
        row = [m, f"{r['mean_reward']:.3f}", f"{r['median_reward']:.3f}",
               f"{r['min_reward']:.3f}", f"{r['max_reward']:.3f}"]
        for pk in pass_keys:
            v = r["pass_at_k"].get(pk)
            row.append(f"{v:.3f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Per-criterion means (PeerJudge)")
    lines.append("")
    crit_keys = ["comprehension", "substance_and_specificity", "insight",
                 "issue_overlap", "missed_weakness", "fabrication", "calibration_pairwise"]
    short = {k: k[:14] for k in crit_keys}
    hdr = ["model"] + [short[k] for k in crit_keys]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for m, r in model_results.items():
        row = [m]
        for ck in crit_keys:
            v = r["criteria_avg"].get(ck)
            row.append(f"{v:.3f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream(cmd: list[str]):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Orchestrate a multi-model pass@K experiment end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Entries from providers.yaml (e.g. deepinfra-qwen3p6-35b-a3b)")
    p.add_argument("--providers-yaml", default=str(SCRIPT_DIR / "providers.yaml"))
    p.add_argument("--data-dir", required=True, help="Prepared task directory")
    p.add_argument("--benchmark-cwd", default="/root/benchmark",
                   help="Dir where benchmark_pass_at_k.py lives (needs prompts/, skills/ alongside)")

    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--n-papers", type=int, help="Pick first N papers from data-dir")
    scope.add_argument("--tasks", nargs="+", help="Explicit paper IDs")

    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-concurrent", type=int, default=40)
    p.add_argument("--exp-root", default="/root/pass_at_k/exp")
    p.add_argument("--name", required=True, help="Experiment name (becomes exp-root/<name>/)")
    p.add_argument("--pass-threshold", type=float, default=0.7)
    p.add_argument("--skip-health-check", action="store_true",
                   help="Skip dataset regen step")
    args = p.parse_args()

    # Validate models
    providers_yaml = Path(args.providers_yaml)
    if not providers_yaml.is_file():
        sys.exit(f"error: providers.yaml not found at {providers_yaml}")
    providers = load_providers(providers_yaml)
    unknown = [m for m in args.models if m not in providers]
    if unknown:
        sys.exit(f"error: unknown provider(s): {unknown}. Known: {list(providers)}")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"error: data-dir not found: {data_dir}")

    paper_ids = pick_papers(data_dir, args.n_papers, args.tasks)

    # Dataset health check → auto-regen any broken reviews
    if not args.skip_health_check:
        broken = dataset_health_check(data_dir, paper_ids)
        if broken:
            print(f"==> {len(broken)} papers have broken/short human_reviews, regenerating:")
            regen_reviews_for(data_dir, broken)
        else:
            print("==> dataset health check: all papers have full human reviews")

    # Prepare exp dir
    exp_dir = Path(args.exp_root) / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    scope_manifest = {
        "name": args.name,
        "paper_ids": paper_ids,
        "k": args.k,
        "max_concurrent": args.max_concurrent,
        "models": args.models,
        "threshold": args.pass_threshold,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (exp_dir / "scope.json").write_text(json.dumps(scope_manifest, indent=2))
    print(f"==> experiment dir: {exp_dir}")
    print(f"    papers: {paper_ids}")
    print(f"    models: {args.models}")
    print(f"    k={args.k}, max_concurrent={args.max_concurrent}")

    # Run each model
    model_results: dict[str, dict] = {}
    for entry in args.models:
        model_id = providers[entry].get("model", entry)
        slug = model_slug(model_id)
        print(f"\n{'='*60}\n  model: {entry}  ({model_id})\n{'='*60}")

        upstream = switch_provider(entry)

        killed = kill_stale_sandboxes()
        print(f"==> killed {killed} stale sandboxes")

        model_dir = exp_dir / slug
        launch_benchmark(
            model_entry=entry,
            upstream=upstream,
            data_dir=data_dir,
            results_dir=model_dir / "results",
            trials_dir=model_dir / "trials",
            log_path=model_dir / "log.txt",
            paper_ids=paper_ids,
            k=args.k,
            max_concurrent=args.max_concurrent,
            benchmark_cwd=Path(args.benchmark_cwd),
        )

        agg = aggregate_model(model_dir / "results", threshold=args.pass_threshold)
        (model_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
        model_results[entry] = agg

    # Cross-model comparison
    (exp_dir / "comparison.json").write_text(json.dumps({
        "scope": scope_manifest,
        "models": model_results,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    md = comparison_markdown(args.name, model_results, scope_manifest)
    (exp_dir / "comparison.md").write_text(md)
    print("\n" + md)
    print(f"\n==> experiment complete: {exp_dir}")


if __name__ == "__main__":
    main()
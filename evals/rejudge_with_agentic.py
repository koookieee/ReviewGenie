"""rejudge_with_agentic.py — re-score existing pass@K trajectories with the agentic judge.

Walks `--results-dir` looking for `<paper_id>/attempt_<i>/trajectory.json`,
runs `score_review` from benchmark_pass_at_k.py, and writes a new
`result.json` (copying through metadata fields like duration, tokens,
trial_name from the original) to `--out-dir`. The original results are
never modified.

Usage:
    python rejudge_with_agentic.py \
        --results-dir /root/pass_at_k/results_minimax_100 \
        --data-dir /root/data/pass_at_k_reviewed \
        --out-dir /root/pass_at_k/results_minimax_100_rejudged \
        --max-concurrent 12
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_pass_at_k import score_review  # noqa: E402


async def _rejudge_one(*, paper_id: str, attempt_idx: int,
                       trajectory_path: Path, task_dir: Path,
                       out_dir: Path, original_result: dict,
                       judge_api_key: str, judge_model: str,
                       judge_base_url: str, sem: asyncio.Semaphore) -> dict:
    out_path = out_dir / paper_id / f"attempt_{attempt_idx}" / "result.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("status") == "rejudged":
                logger.info(f"skip {paper_id}/attempt_{attempt_idx} — already rejudged")
                return existing
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with sem:
        t0 = time.monotonic()
        scores = await score_review(
            trajectory_path=trajectory_path,
            task_dir=task_dir,
            judge_api_key=judge_api_key,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
        )
        elapsed = time.monotonic() - t0

    new_result = {
        "paper_id": paper_id,
        "attempt": attempt_idx,
        "trial_name": original_result.get("trial_name"),
        "status": "rejudged",
        "rejudge_duration_sec": round(elapsed, 2),
        "scores": scores,
        "reward": scores.get("reward", 0.0),
        "model": original_result.get("model"),
        "judge_model": judge_model,
        "judge_kind": "agentic_grep_paper_v1",
        "tokens": original_result.get("tokens", {}),
        "original_reward": original_result.get("reward"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(new_result, indent=2))

    tel = scores.get("_judge_telemetry", {}) or {}
    logger.info(
        f"{paper_id}/attempt_{attempt_idx}: "
        f"reward {original_result.get('reward', 0):.3f} -> {new_result['reward']:.3f} "
        f"(steps={tel.get('steps')} grep={tel.get('tool_calls', {}).get('grep_paper', 0)} "
        f"t={elapsed:.1f}s)"
    )
    return new_result


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv(SCRIPT_DIR.parent / ".env")
    judge_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not judge_api_key:
        raise SystemExit("GEMINI_API_KEY missing")
    judge_model = os.environ.get("JUDGE_MODEL", "gemini-3.1-pro-preview")
    judge_base_url = os.environ.get(
        "JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover (paper_id, attempt_idx, trajectory_path, original_result_path) tuples.
    work: list[tuple[str, int, Path, dict]] = []
    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        for attempt_dir in sorted(paper_dir.iterdir()):
            if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt_"):
                continue
            traj = attempt_dir / "trajectory.json"
            res = attempt_dir / "result.json"
            if not traj.exists() or not res.exists():
                continue
            try:
                idx = int(attempt_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            try:
                orig = json.loads(res.read_text())
            except Exception:
                orig = {}
            work.append((paper_dir.name, idx, traj, orig))

    if args.limit:
        work = work[:args.limit]
    logger.info(f"rejudging {len(work)} trajectories with concurrency {args.max_concurrent}")

    sem = asyncio.Semaphore(args.max_concurrent)
    coros = [
        _rejudge_one(
            paper_id=pid, attempt_idx=idx, trajectory_path=traj,
            task_dir=data_dir / pid, out_dir=out_dir, original_result=orig,
            judge_api_key=judge_api_key, judge_model=judge_model,
            judge_base_url=judge_base_url, sem=sem,
        ) for pid, idx, traj, orig in work
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    ok = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "rejudged")
    failed = len(results) - ok
    logger.info(f"done: {ok} ok / {failed} failed")

    rewards = [r.get("reward", 0.0) for r in results if isinstance(r, dict)]
    if rewards:
        mean = sum(rewards) / len(rewards)
        logger.info(f"new mean reward: {mean:.3f} (over {len(rewards)} attempts)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-concurrent", type=int, default=12)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, rejudge only the first N attempts (for smoke tests)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
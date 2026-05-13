"""rejudge_stanford.py — re-score Stanford Reviewer outputs with the agentic judge.

Stanford's outputs live at /root/Stanford_Reviewer/reviews/<paper_id>_review.json
(review text in `content`) and the paper metadata + body live in our existing
/root/data/pass_at_k_reviewed/<paper_id>/ tree. We feed the review string +
paper metadata to AgenticJudge.score() and write a result.json per paper.

Usage:
    python rejudge_stanford.py \
        --reviews-dir /root/Stanford_Reviewer/reviews \
        --data-dir /root/data/pass_at_k_reviewed \
        --out-dir /root/Stanford_Reviewer/results_agentic \
        --max-concurrent 12
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from agentic_judge import AgenticJudge, AgenticJudgeConfig
from benchmark_pass_at_k import _load_task_metadata


def _load_paper_body(paper_id: str, markdown_dir: Path | None, metadata: dict) -> str:
    """Return paper body text — identical logic to benchmark_pass_at_k.score_review."""
    paper_body = ""
    if markdown_dir is not None:
        md_path = markdown_dir / f"{paper_id}.md"
        if md_path.is_file():
            try:
                text = md_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > 500:
                    paper_body = text
            except Exception:
                pass
    if not paper_body:
        paper_body = metadata.get("paper_body", "")
    ref_m = re.search(r"^#+\s*(References|Bibliography)\s*$", paper_body, re.MULTILINE | re.IGNORECASE)
    if not ref_m:
        ref_m = re.search(r"^\*{0,2}(References|Bibliography)\*{0,2}\s*$", paper_body, re.MULTILINE | re.IGNORECASE)
    if ref_m:
        paper_body = paper_body[:ref_m.start()].rstrip()
    return paper_body


async def _rejudge_one(*, paper_id: str, review_text: str, task_dir: Path,
                       markdown_dir: Path | None, out_dir: Path,
                       judge: AgenticJudge, sem: asyncio.Semaphore) -> dict:
    out_path = out_dir / paper_id / "result.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("status") == "rejudged":
                logger.info(f"skip {paper_id} — already rejudged")
                return existing
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _load_task_metadata(task_dir)
    if not metadata["title"]:
        logger.warning(f"{paper_id}: no title — skipping")
        return {"paper_id": paper_id, "status": "skipped_no_title", "reward": 0.0}

    human_reviews_text = "\n\n---\n\n".join(
        f"**Human Review {i+1}:**\n\n{r}"
        for i, r in enumerate(metadata["human_reviews"])
    )
    paper_body = _load_paper_body(paper_id, markdown_dir, metadata)

    async with sem:
        t0 = time.monotonic()
        scores = await judge.score(
            title=metadata["title"],
            abstract=metadata["abstract"],
            paper_body=paper_body,
            human_reviews_text=human_reviews_text,
            model_review=review_text,
        )
        elapsed = time.monotonic() - t0

    # Option A weighting — drop comp/sub/ins from the score.
    def _s(key: str) -> float:
        node = scores.get(key, {})
        if isinstance(node, dict):
            return float(node.get("score", 0) or 0)
        return float(node or 0)
    overlap = _s("issue_overlap")
    fabrication = _s("fabrication")
    calibration = _s("calibration_pairwise")
    reward = (overlap + fabrication + calibration) / 3.0
    scores["reward"] = reward

    result = {
        "paper_id": paper_id,
        "status": "rejudged",
        "rejudge_duration_sec": round(elapsed, 2),
        "scores": scores,
        "reward": reward,
        "criterion_scores": {
            "issue_overlap": overlap,
            "fabrication": fabrication,
            "calibration_pairwise": calibration,
        },
        "model": "stanford_reviewer",
        "judge_kind": "agentic_grep_paper_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(result, indent=2))

    tel = scores.get("_judge_telemetry", {}) or {}
    logger.info(
        f"{paper_id}: reward={result['reward']:.3f} "
        f"(steps={tel.get('steps')} grep={tel.get('tool_calls', {}).get('grep_paper', 0)} "
        f"read={tel.get('tool_calls', {}).get('read_paper', 0)} t={elapsed:.1f}s)"
    )
    return result


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv(SCRIPT_DIR.parent / ".env")
    judge_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not judge_api_key:
        raise SystemExit("GEMINI_API_KEY missing")
    judge_model = os.environ.get("JUDGE_MODEL", "gemini-3.1-pro-preview")

    reviews_dir = Path(args.reviews_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir = Path(args.markdown_dir) if args.markdown_dir else None
    if markdown_dir:
        logger.info(f"using OCR markdown from {markdown_dir}")

    # Single shared judge across all workers.
    prompt_template = (SCRIPT_DIR.parent / "prompts" / "llm_judge_instruction.md").read_text()
    judge = AgenticJudge(
        api_key=judge_api_key,
        prompt_template=prompt_template,
        config=AgenticJudgeConfig(model=judge_model),
    )

    work: list[tuple[str, str]] = []
    for f in sorted(reviews_dir.glob("*_review.json")):
        paper_id = f.stem.replace("_review", "")
        if not (data_dir / paper_id).is_dir():
            logger.warning(f"{paper_id}: no task dir at {data_dir} — skipping")
            continue
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            logger.warning(f"{paper_id}: cannot parse {f}: {e}")
            continue
        review_text = d.get("content") or ""
        if len(review_text) < 100:
            logger.warning(f"{paper_id}: review too short — skipping")
            continue
        work.append((paper_id, review_text))

    if args.limit:
        work = work[:args.limit]
    logger.info(f"rejudging {len(work)} Stanford reviews with concurrency {args.max_concurrent}")

    sem = asyncio.Semaphore(args.max_concurrent)
    coros = [
        _rejudge_one(
            paper_id=pid, review_text=text, task_dir=data_dir / pid,
            markdown_dir=markdown_dir, out_dir=out_dir, judge=judge, sem=sem,
        ) for pid, text in work
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    ok = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "rejudged")
    failed = len(results) - ok
    rewards = [r.get("reward", 0.0) for r in results if isinstance(r, dict) and r.get("status") == "rejudged"]
    logger.info(f"done: {ok} ok / {failed} failed")
    if rewards:
        mean = sum(rewards) / len(rewards)
        logger.info(f"new mean reward (6c default formula): {mean:.3f} (over {len(rewards)} reviews)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reviews-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--markdown-dir", default=None,
                   help="Directory of OCR markdown files (<paper_id>.md). "
                        "Preferred over task-dir latex when present.")
    p.add_argument("--max-concurrent", type=int, default=12)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
"""
Paper Reviewer Benchmark — Opus 4.6 via Harbor + E2B

Runs the same paper review tasks used in training, but with Claude Opus 4.6
(real Anthropic API) to establish SOTA baseline performance.

For each task:
1. Creates a Harbor Trial with Claude Code agent on E2B
2. Uploads task files (LaTeX, search skill, search API URL) to the sandbox
3. Runs the agent (Opus 4.6 reviews the paper)
4. Extracts ATIF trajectory from Harbor's output
5. Scores the review using the LLM judge (Gemini 3 Flash)
6. Saves trajectory + scores to results/

Usage:
    python benchmark.py                         # run all tasks
    python benchmark.py --tasks 2603.10165v1    # run specific task(s)
    python benchmark.py --max-tasks 5           # limit number of tasks
"""

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial


# ---------------------------------------------------------------------------
# Task file upload — same approach as PaperReviewerTrial in the training code
# ---------------------------------------------------------------------------

_TASK_SKIP_DIRS = {"environment", ".git", "__pycache__"}


class BenchmarkTrial(Trial):
    """Trial subclass that uploads task content to E2B sandboxes.

    E2B's Template.from_dockerfile has no build context, so task files
    (latex/, skills, etc.) must be uploaded after environment setup.
    """

    # Set by the runner before calling .run()
    search_api_url: str = ""

    async def _setup_environment(self) -> None:
        await super()._setup_environment()

        task_dir = Path(self.config.task.path)
        if not task_dir.is_dir():
            logger.warning(f"Task dir {task_dir} not found, skipping file upload")
            return

        workdir = self._environment._workdir or "/app"

        # Upload task content (everything except environment/)
        for item in sorted(task_dir.iterdir()):
            if item.name in _TASK_SKIP_DIRS:
                continue
            target = f"/{workdir.strip('/')}/{item.name}"
            try:
                if item.is_file():
                    await self._environment.upload_file(item, target)
                elif item.is_dir():
                    if item.name == "latex":
                        # Trim template.tex to Conclusion before uploading
                        tex_file = item / "template.tex"
                        if tex_file.is_file():
                            content = tex_file.read_text(errors="replace")
                            m = re.search(r"^(# Conclusion|\\section\{Conclusion\})", content, re.MULTILINE | re.IGNORECASE)
                            if m:
                                next_section = re.search(r"^# (?!Conclusion)", content[m.start():], re.MULTILINE)
                                end = m.start() + next_section.start() if next_section else len(content)
                                tex_file.write_text(content[:end])
                                logger.info(f"Trimmed template.tex to Conclusion: {len(content):,} -> {end:,} chars")
                    await self._environment.upload_dir(item, target)
            except Exception as e:
                logger.warning(f"Failed to upload {item.name} to {target}: {e}")
        logger.info(f"Uploaded task content from {task_dir} to {workdir}")

        # Overwrite instruction.md with our canonical template (has correct paths)
        instruction_template = Path(__file__).parent / "prompts" / "paper_reviewer_instruction_template.md"
        if instruction_template.is_file():
            target = f"/{workdir.strip('/')}/instruction.md"
            try:
                await self._environment.upload_file(instruction_template, target)
                logger.info(f"Uploaded instruction template to {target}")
            except Exception as e:
                logger.warning(f"Failed to upload instruction template: {e}")

        # Upload search skill
        skill_file = Path(__file__).parent / "skills" / "search-papers" / "SKILL.md"
        if skill_file.is_file():
            skill_target = "/root/.claude/skills/search-papers/SKILL.md"
            try:
                await self._environment.upload_file(skill_file, skill_target)
                logger.info(f"Uploaded search skill to {skill_target}")
            except Exception as e:
                logger.warning(f"Failed to upload search skill: {e}")
            # Fallback copy to workdir
            fallback = f"/{workdir.strip('/')}/search-papers-skill.md"
            try:
                await self._environment.upload_file(skill_file, fallback)
            except Exception:
                pass

        # Write search API URL
        if self.search_api_url:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(self.search_api_url)
                tmp_path = f.name
            target = f"/{workdir.strip('/')}/search_api_url.txt"
            try:
                await self._environment.upload_file(Path(tmp_path), target)
                logger.info(f"Uploaded search API URL to {target}")
            except Exception as e:
                logger.warning(f"Failed to upload search API URL: {e}")
            finally:
                os.unlink(tmp_path)



# ---------------------------------------------------------------------------
# LLM Judge — scores the review output
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_PATH = Path(__file__).parent / "prompts" / "llm_judge_instruction.md"


def _load_judge_prompt() -> str:
    return _JUDGE_PROMPT_PATH.read_text()


def _extract_last_review(trajectory_path: Path) -> str:
    """Extract the last agent message from the ATIF trajectory."""
    data = json.loads(trajectory_path.read_text())
    for step in reversed(data.get("steps", [])):
        if step.get("source") == "agent" and step.get("message"):
            msg = step["message"]
            if len(msg) > 200:  # Skip short tool messages
                return msg
    return ""


def _load_task_metadata(task_dir: Path) -> dict:
    metadata = {"title": "", "abstract": "", "human_reviews": []}

    meta_path = task_dir / "task_metadata.json"
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text())
            metadata["title"] = data.get("title", "")
            metadata["abstract"] = data.get("abstract", "")
            metadata["human_reviews"] = data.get("human_reviews", [])
            return metadata
        except Exception:
            pass

    # Fallback: parse template.tex (may be markdown or LaTeX)
    tex_path = task_dir / "latex" / "template.tex"
    if tex_path.is_file():
        tex = tex_path.read_text(errors="replace")
        # Markdown title: first # heading
        m = re.search(r"^# (.+)$", tex, re.MULTILINE)
        if m:
            metadata["title"] = m.group(1).strip()
        else:
            # LaTeX title fallback
            m = re.search(r"\\title\{([^}]+)\}", tex)
            if m:
                metadata["title"] = m.group(1).strip()
        # Markdown abstract: ## Abstract section
        m = re.search(r"^## Abstract\s*\n(.*?)(?=^#|\Z)", tex, re.MULTILINE | re.DOTALL)
        if m:
            metadata["abstract"] = m.group(1).strip()
        else:
            # LaTeX abstract fallback
            m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
            if m:
                metadata["abstract"] = m.group(1).strip()

    return metadata


async def score_review(
    trajectory_path: Path,
    task_dir: Path,
    judge_api_key: str,
    judge_model: str = "gemini-3-flash-preview",
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
) -> dict:
    """Score a review using the LLM judge. Returns scores dict."""
    from openai import AsyncOpenAI

    review = _extract_last_review(trajectory_path)
    if not review or len(review) < 100:
        logger.warning("Review too short or empty, returning zero scores")
        return {"error": "review_too_short", "reward": 0.0}

    metadata = _load_task_metadata(task_dir)
    if not metadata["title"]:
        logger.warning(f"No title found in {task_dir}")
        return {"error": "no_title", "reward": 0.0}

    # Format human reviews
    if metadata["human_reviews"]:
        human_reviews_text = "\n\n---\n\n".join(
            f"**Human Review {i+1}:**\n\n{r}"
            for i, r in enumerate(metadata["human_reviews"])
        )
    else:
        human_reviews_text = "(No human reviews available for this paper.)"

    prompt = _load_judge_prompt()
    prompt = prompt.replace("{title}", metadata["title"])
    prompt = prompt.replace("{abstract}", metadata["abstract"])
    prompt = prompt.replace("{human_reviews}", human_reviews_text)
    prompt = prompt.replace("{model_review}", review)

    client = AsyncOpenAI(api_key=judge_api_key, base_url=judge_base_url)

    try:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16384,
        )
        judge_output = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"LLM judge API call failed: {e}")
        return {"error": str(e), "reward": 0.0}

    # Parse JSON
    try:
        json_match = re.search(r"\{[\s\S]*\}", judge_output)
        if not json_match:
            return {"error": "no_json_in_judge_output", "raw": judge_output, "reward": 0.0}
        scores = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return {"error": f"json_parse: {e}", "raw": judge_output, "reward": 0.0}

    # Compute composite reward
    comprehension = float(scores.get("comprehension", {}).get("score", 0))
    substance = float(scores.get("substance_and_specificity", {}).get("score", 0))
    insight = float(scores.get("insight", {}).get("score", 0))
    issue_overlap = float(scores.get("issue_overlap", {}).get("score", 0))
    calibration = float(scores.get("calibration", {}).get("score", 0))

    if metadata["human_reviews"]:
        reward = (
            0.20 * comprehension
            + 0.25 * substance
            + 0.25 * insight
            + 0.20 * issue_overlap
            + 0.10 * calibration
        )
    else:
        reward = 0.30 * comprehension + 0.35 * substance + 0.35 * insight

    scores["reward"] = reward
    logger.info(
        f"Judge scores: comprehension={comprehension}, substance={substance}, "
        f"insight={insight}, issue_overlap={issue_overlap}, calibration={calibration} "
        f"→ reward={reward:.3f}"
    )
    return scores


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def build_trial_config(
    task_path: str,
    trials_dir: str,
    trial_name: str,
    agent_timeout: int = 2400,
    max_turns: int = 200,
) -> dict:
    """Build a TrialConfig dict for a benchmark trial."""
    return {
        "trial_name": trial_name,
        "trials_dir": trials_dir,
        "task": {"path": task_path},
        "agent": {
            "name": "claude-code",
            "override_timeout_sec": agent_timeout,
            "kwargs": {
                "max_turns": max_turns,
                "reasoning_effort": "high",
            },
            "env": {
                "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", ""),
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            },
        },
        "environment": {
            "type": "e2b",
            "force_build": False,
            "override_cpus": 2,
            "override_memory_mb": 2048,
            "override_storage_mb": 2048,
            "suppress_override_warnings": True,
            "kwargs": {
                "auto_stop_interval_mins": 45,
            },
        },
        "verifier": {"disable": True},
    }


async def run_single_task(
    task_dir: Path,
    trials_dir: Path,
    results_dir: Path,
    search_api_url: str,
    judge_api_key: str,
    judge_model: str,
    judge_base_url: str,
) -> dict:
    """Run a single benchmark trial and score it."""
    task_name = task_dir.name
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trial_name = f"opus-benchmark-{task_name}-{int(time.time())}"
    result_dir = results_dir / f"{run_ts}_{task_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting benchmark for task: {task_name}")

    # Keep task dir instruction.md in sync with our canonical template
    # (Harbor reads instruction from the task dir on disk BEFORE environment setup)
    instruction_template = Path(__file__).parent / "prompts" / "paper_reviewer_instruction_template.md"
    if instruction_template.is_file():
        shutil.copy2(instruction_template, task_dir / "instruction.md")

    # Build trial config
    config_dict = build_trial_config(
        task_path=str(task_dir),
        trials_dir=str(trials_dir),
        trial_name=trial_name,
    )
    trial_config = TrialConfig.model_validate(config_dict)

    # Create trial with file upload support
    trial = await BenchmarkTrial.create(trial_config)
    trial.search_api_url = search_api_url

    # Run the trial
    start_time = time.time()
    try:
        trial_result = await trial.run()
    except Exception as e:
        logger.error(f"Trial failed for {task_name}: {e}")
        error_result = {
            "task": task_name,
            "status": "error",
            "error": str(e),
            "duration_sec": time.time() - start_time,
        }
        (result_dir / "result.json").write_text(json.dumps(error_result, indent=2))
        return error_result

    duration = time.time() - start_time
    logger.info(f"Trial completed for {task_name} in {duration:.0f}s")

    # Copy trajectory from trial output
    trial_dir = trials_dir / trial_name
    trajectory_path = None
    for traj_file in trial_dir.rglob("trajectory.json"):
        trajectory_path = traj_file
        break

    if trajectory_path and trajectory_path.exists():
        dest = result_dir / "trajectory.json"
        shutil.copy2(trajectory_path, dest)
        trajectory_path = dest
        logger.info(f"Saved trajectory to {dest}")
    else:
        logger.warning(f"No trajectory found for {task_name}")

    # Also copy raw session JSONL if available
    for jsonl_file in trial_dir.rglob("*.jsonl"):
        dest = result_dir / "session.jsonl"
        shutil.copy2(jsonl_file, dest)
        break

    # Score the review
    scores = {}
    if trajectory_path and trajectory_path.exists():
        scores = await score_review(
            trajectory_path=trajectory_path,
            task_dir=task_dir,
            judge_api_key=judge_api_key,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
        )

    # Build result
    exception_type = None
    if trial_result.exception_info:
        exception_type = trial_result.exception_info.exception_type

    result = {
        "task": task_name,
        "trial_name": trial_name,
        "status": "success" if not exception_type else "error",
        "exception_type": exception_type,
        "duration_sec": duration,
        "scores": scores,
        "reward": scores.get("reward", 0.0),
        "agent_info": {
            "name": "claude-code",
            "model": "qwen/qwen3.5-397b-a17b",
        },
        "tokens": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Extract token usage from trial result
    if trial_result.agent_result:
        result["tokens"] = {
            "input": trial_result.agent_result.n_input_tokens,
            "output": trial_result.agent_result.n_output_tokens,
            "cache": trial_result.agent_result.n_cache_tokens,
            "cost_usd": trial_result.agent_result.cost_usd,
        }

    # Save result
    (result_dir / "result.json").write_text(json.dumps(result, indent=2))
    logger.info(
        f"Task {task_name}: reward={result['reward']:.3f}, "
        f"duration={duration:.0f}s, status={result['status']}"
    )

    return result


async def run_benchmark(
    data_dir: Path,
    trials_dir: Path,
    results_dir: Path,
    search_api_url: str,
    judge_api_key: str,
    judge_model: str,
    judge_base_url: str,
    task_filter: list[str] | None = None,
    max_tasks: int | None = None,
    max_concurrent: int = 2,
):
    """Run the full benchmark across all tasks."""
    # Discover tasks
    task_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    ])

    if task_filter:
        task_dirs = [d for d in task_dirs if d.name in task_filter]

    if max_tasks:
        task_dirs = task_dirs[:max_tasks]

    logger.info(f"Found {len(task_dirs)} tasks to benchmark")

    results_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(task_dir):
        async with semaphore:
            return await run_single_task(
                task_dir=task_dir,
                trials_dir=trials_dir,
                results_dir=results_dir,
                search_api_url=search_api_url,
                judge_api_key=judge_api_key,
                judge_model=judge_model,
                judge_base_url=judge_base_url,
            )

    tasks = [run_with_semaphore(td) for td in task_dirs]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful = []
    failed = []
    for r in all_results:
        if isinstance(r, Exception):
            failed.append({"error": str(r)})
        elif r.get("status") == "success":
            successful.append(r)
        else:
            failed.append(r)

    # Compute summary
    rewards = [r["reward"] for r in successful if "reward" in r]
    summary = {
        "model": "qwen/qwen3.5-397b-a17b",
        "agent": "claude-code",
        "total_tasks": len(task_dirs),
        "successful": len(successful),
        "failed": len(failed),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "rewards": rewards,
        "total_duration_sec": sum(
            r.get("duration_sec", 0)
            for r in all_results
            if isinstance(r, dict)
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [r for r in all_results if isinstance(r, dict)],
    }

    # Save summary with timestamp so runs don't overwrite each other
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = results_dir / f"benchmark_summary_{run_ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        f"\n{'='*60}\n"
        f"BENCHMARK COMPLETE\n"
        f"  Model: Claude Opus 4.6\n"
        f"  Tasks: {summary['successful']}/{summary['total_tasks']} successful\n"
        f"  Mean reward: {summary['mean_reward']:.3f}\n"
        f"  Total time: {summary['total_duration_sec']:.0f}s\n"
        f"  Results: {summary_path}\n"
        f"{'='*60}"
    )

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Paper Reviewer Benchmark — Opus 4.6")
    parser.add_argument("--config", type=str, default="config/benchmark.yaml")
    parser.add_argument("--tasks", nargs="*", help="Specific task IDs to run")
    parser.add_argument("--max-tasks", type=int, help="Max number of tasks")
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()

    # Load .env
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Load config
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Resolve settings from env + config
    data_dir = Path(cfg.get("data_dir", os.environ.get("DATA_DIR", "/root/data/harbor/PaperReviews")))
    results_dir = Path(cfg.get("results_dir", "/root/benchmark/results"))
    trials_dir = Path(cfg.get("trials_dir", "/root/benchmark/trials"))

    search_api_url = cfg.get("search_api_url", os.environ.get("SEARCH_PUBLIC_URL", ""))
    judge_api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("LLM_JUDGE_API_KEY", ""))
    judge_model = cfg.get("judge", {}).get("model", "gemini-3-flash-preview")
    judge_base_url = cfg.get("judge", {}).get(
        "base_url", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Validate
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY must be set"
    assert os.environ.get("E2B_API_KEY"), "E2B_API_KEY must be set"
    assert judge_api_key, "GEMINI_API_KEY must be set for LLM judge"

    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Search API: {search_api_url}")

    asyncio.run(
        run_benchmark(
            data_dir=data_dir,
            trials_dir=trials_dir,
            results_dir=results_dir,
            search_api_url=search_api_url,
            judge_api_key=judge_api_key,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            task_filter=args.tasks,
            max_tasks=args.max_tasks,
            max_concurrent=args.max_concurrent,
        )
    )


if __name__ == "__main__":
    main()

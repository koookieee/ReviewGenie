#!/usr/bin/env python3
"""score_stanford_reviews.py — score all Stanford Reviewer proxy reviews with PeerJudge.

Reads review JSONs from /root/Stanford_Reviewer/reviews_proxy/
Reads paper metadata (title, abstract, human_reviews, latex) from /root/data/pass_at_k_reviewed/
Writes scored output to /root/Stanford_Reviewer/scores/ as <paper_id>_score.json

Runs all 115 papers in parallel (asyncio + semaphore to stay within rate limits).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REVIEWS_DIR = Path("/root/Stanford_Reviewer/reviews_proxy")
DATA_DIR = Path("/root/data/pass_at_k_reviewed")
SCORES_DIR = Path("/root/Stanford_Reviewer/scores")
JUDGE_PROMPT_PATH = Path("/root/prompts/llm_judge_instruction.md")

JUDGE_MODEL = "gemini-3.1-pro-preview"
JUDGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Limit concurrent Gemini calls to avoid rate limits
CONCURRENCY = 10

WEIGHTS = {
    "comprehension": 0.05,
    "substance_and_specificity": 0.05,
    "insight": 0.15,
    "issue_overlap": 0.25,
    "fabrication": 0.20,
    "calibration_pairwise": 0.25,
}


# ---------------------------------------------------------------------------
# LaTeX helpers (mirrors benchmark_pass_at_k.py)
# ---------------------------------------------------------------------------

def _find_latex_entry(latex_dir: Path) -> Path:
    """Find the main .tex file in a latex directory."""
    candidates = ["main.tex", "template.tex", "paper.tex"]
    for c in candidates:
        p = latex_dir / c
        if p.is_file():
            return p
    tex_files = list(latex_dir.glob("*.tex"))
    if tex_files:
        return tex_files[0]
    return latex_dir / "template.tex"


def _latex_to_markdown(tex_path: Path) -> str:
    """Very light LaTeX → Markdown conversion (same approach as benchmark)."""
    text = tex_path.read_text(errors="replace")

    # Remove comments
    text = re.sub(r"(?m)%.*$", "", text)

    # Section headings
    text = re.sub(r"\\section\*?\{([^}]+)\}", r"# \1", text)
    text = re.sub(r"\\subsection\*?\{([^}]+)\}", r"## \1", text)
    text = re.sub(r"\\subsubsection\*?\{([^}]+)\}", r"### \1", text)

    # Title / abstract
    text = re.sub(r"\\title\{([^}]+)\}", r"# \1", text)
    text = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", r"## Abstract\n\1", text, flags=re.DOTALL)

    # Bold / italic
    text = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", text)

    # Equations — strip display math
    text = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", "[equation]", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", "[equation]", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$\n]+?\$", "[eq]", text)

    # Remove common environments we don't need
    for env in ["figure", "table", "algorithm", "lstlisting", "verbatim", "tikzpicture"]:
        text = re.sub(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", f"[{env}]", text, flags=re.DOTALL)

    # Remove bibliography
    text = re.sub(r"\\begin\{thebibliography\}.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\\bibliography\{[^}]+\}", "", text)

    # Strip remaining LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"\{|\}", "", text)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _trim_to_conclusion(paper_body: str) -> str:
    """Trim paper body to end at the Conclusion section (inclusive)."""
    # Match the start of References / Bibliography / Acknowledgements after Conclusion
    # Look for a top-level section that follows conclusion
    cut_patterns = [
        r"\n# (?:References|Bibliography|Acknowledgements?|Appendix|Supplementary)",
        r"\n## (?:References|Bibliography|Acknowledgements?)",
    ]
    for pat in cut_patterns:
        m = re.search(pat, paper_body, re.IGNORECASE)
        if m:
            return paper_body[:m.start()].strip()
    return paper_body


def _load_task_metadata(task_dir: Path) -> dict:
    metadata = {"title": "", "abstract": "", "human_reviews": [], "paper_body": ""}
    meta_path = task_dir / "task_metadata.json"
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text())
            metadata["title"] = data.get("title", "")
            metadata["abstract"] = data.get("abstract", "")
            metadata["human_reviews"] = data.get("human_reviews", [])
        except Exception:
            pass

    latex_dir = task_dir / "latex"
    tex_file = _find_latex_entry(latex_dir) if latex_dir.is_dir() else latex_dir / "template.tex"
    if tex_file.is_file():
        try:
            paper_body = _latex_to_markdown(tex_file)
            metadata["paper_body"] = _trim_to_conclusion(paper_body)
        except Exception as e:
            logger.warning(f"Could not load paper body for {task_dir.name}: {e}")
            try:
                raw = tex_file.read_text(errors="replace")
                metadata["paper_body"] = _trim_to_conclusion(raw)
            except Exception:
                pass

    return metadata


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def _parse_scores(text: str):
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None, "no_json_in_judge_output"
    try:
        return json.loads(m.group()), ""
    except json.JSONDecodeError as e:
        return None, f"json_parse: {e}"


async def score_one(
    paper_id: str,
    review_json: dict,
    judge_api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    from openai import AsyncOpenAI

    model_review = review_json.get("content", "")
    if not model_review or len(model_review) < 100:
        return {"paper_id": paper_id, "error": "review_too_short", "reward": 0.0}

    task_dir = DATA_DIR / paper_id
    if not task_dir.is_dir():
        return {"paper_id": paper_id, "error": "no_task_dir", "reward": 0.0}

    metadata = _load_task_metadata(task_dir)
    if not metadata["title"]:
        return {"paper_id": paper_id, "error": "no_title", "reward": 0.0}

    human_reviews_text = "\n\n---\n\n".join(
        f"**Human Review {i+1}:**\n\n{r}"
        for i, r in enumerate(metadata["human_reviews"])
    )

    paper_body = metadata.get("paper_body", "")
    if len(paper_body) > 60000:
        paper_body = paper_body[:60000] + "\n\n[... paper truncated for length ...]"

    prompt_template = JUDGE_PROMPT_PATH.read_text()
    prompt = prompt_template.replace("{title}", metadata["title"])
    prompt = prompt.replace("{abstract}", metadata["abstract"])
    prompt = prompt.replace("{paper_body}", paper_body or "(Paper body unavailable.)")
    prompt = prompt.replace("{human_reviews}", human_reviews_text)
    prompt = prompt.replace("{model_review}", model_review)

    client = AsyncOpenAI(api_key=judge_api_key, base_url=JUDGE_BASE_URL)

    async def _call_judge(force_json_mode: bool, extra_hint: str = "") -> str:
        kwargs: dict = {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt + extra_hint}],
            "temperature": 0.0,
            "max_tokens": 16384,
        }
        if force_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    scores = None
    last_raw = ""
    last_err = ""

    async with semaphore:
        for attempt_i in range(2):
            try:
                if attempt_i == 0:
                    judge_output = await _call_judge(force_json_mode=False)
                else:
                    logger.warning(f"[{paper_id}] judge retry after: {last_err[:80]}")
                    judge_output = await _call_judge(
                        force_json_mode=True,
                        extra_hint=(
                            "\n\nIMPORTANT ON RETRY: your previous response had malformed JSON. "
                            "Output a single valid JSON object matching the schema above. "
                            "Escape all quotes inside strings. No trailing commas. "
                            "No text outside the JSON object."
                        ),
                    )
            except Exception as e:
                logger.warning(f"[{paper_id}] API error: {e}")
                return {"paper_id": paper_id, "error": str(e), "reward": 0.0}

            last_raw = judge_output
            scores, last_err = _parse_scores(judge_output)
            if scores is not None:
                break

    if scores is None:
        return {"paper_id": paper_id, "error": last_err, "raw": last_raw, "reward": 0.0}

    def _s(key: str) -> float:
        node = scores.get(key, {})
        if isinstance(node, dict):
            return float(node.get("score", 0) or 0)
        return float(node or 0)

    comprehension = _s("comprehension")
    substance = _s("substance_and_specificity")
    insight = _s("insight")
    issue_overlap = _s("issue_overlap")
    fabrication = _s("fabrication")
    calibration = _s("calibration_pairwise")

    reward = (
        0.05 * comprehension
        + 0.05 * substance
        + 0.15 * insight
        + 0.25 * issue_overlap
        + 0.20 * fabrication
        + 0.25 * calibration
    )

    scores["reward"] = reward
    scores["paper_id"] = paper_id
    scores["title"] = review_json.get("title", metadata["title"])

    logger.info(
        f"[{paper_id}] comp={comprehension} sub={substance} ins={insight} "
        f"overlap={issue_overlap} fab={fabrication} cal={calibration} -> reward={reward:.3f}"
    )
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    judge_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not judge_api_key:
        # Try loading from .env
        env_path = Path("/root/.env")
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    judge_api_key = line.split("=", 1)[1].strip()
                    break
    if not judge_api_key:
        logger.error("GEMINI_API_KEY not found in environment or /root/.env")
        sys.exit(1)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # Load all review JSONs
    review_files = sorted(REVIEWS_DIR.glob("*_review.json"))
    logger.info(f"Found {len(review_files)} review files")

    # Skip already scored
    to_score = []
    for rf in review_files:
        paper_id = rf.stem.replace("_review", "")
        out_path = SCORES_DIR / f"{paper_id}_score.json"
        if out_path.exists():
            logger.info(f"[{paper_id}] already scored, skipping")
            continue
        review_json = json.loads(rf.read_text())
        to_score.append((paper_id, review_json, out_path))

    logger.info(f"Scoring {len(to_score)} papers (skipping {len(review_files) - len(to_score)} already done)")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def score_and_save(paper_id: str, review_json: dict, out_path: Path) -> dict:
        result = await score_one(paper_id, review_json, judge_api_key, semaphore)
        out_path.write_text(json.dumps(result, indent=2))
        return result

    tasks = [
        score_and_save(paper_id, review_json, out_path)
        for paper_id, review_json, out_path in to_score
    ]
    results = await asyncio.gather(*tasks)

    # Summary
    successful = [r for r in results if "error" not in r or r.get("reward", 0) > 0]
    rewards = [r["reward"] for r in results if "reward" in r and r.get("reward", 0) > 0]
    errors = [r for r in results if r.get("reward", 0) == 0]

    print(f"\n{'='*60}")
    print(f"Scored: {len(successful)}/{len(results)}")
    if rewards:
        print(f"Reward  mean={sum(rewards)/len(rewards):.3f}  "
              f"min={min(rewards):.3f}  max={max(rewards):.3f}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e.get('paper_id')}: {e.get('error', '?')}")

    # Write summary JSON
    summary_path = SCORES_DIR / "summary.json"
    all_scores = []
    for sf in sorted(SCORES_DIR.glob("*_score.json")):
        try:
            all_scores.append(json.loads(sf.read_text()))
        except Exception:
            pass
    summary_path.write_text(json.dumps(all_scores, indent=2))
    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())

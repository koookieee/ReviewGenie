"""
eval_literature_search.py — Evaluate literature search quality from final review text.

Core idea: measure how RELEVANT the papers each agent found are to the paper under review.
Uses the Gemini embedding index (same embeddings the search API uses) to compute
cosine similarity between the paper-under-review and each agent-discussed paper.

Primary metric:
  MRS (Mean Relevance Score) — mean cosine similarity between paper-under-review
  and agent-cited papers. Higher = agent found more relevant papers.

Secondary metric (when human reviews available):
  HCR (Human-Citation Recall) — fraction of human-flagged papers the agent also found.

Usage:
  python eval_literature_search.py build-ground-truth \
    --data-dir /root/data/pass_at_k_reviewed \
    --search-api-url http://localhost:8081 \
    --gemini-api-key $GEMINI_API_KEY \
    --out /root/pass_at_k/literature_ground_truth.json

  python eval_literature_search.py extract \
    --name deepseek --results-dir /root/pass_at_k/results_deepseek_95 \
    --search-api-url http://localhost:8081 \
    --gemini-api-key $GEMINI_API_KEY \
    --out /root/pass_at_k/extracted_deepseek.json

  python eval_literature_search.py score \
    --ground-truth /root/pass_at_k/literature_ground_truth.json \
    --extracted deepseek=/root/pass_at_k/extracted_deepseek.json \
    --extracted stanford=/root/pass_at_k/extracted_stanford.json \
    --gemini-api-key $GEMINI_API_KEY \
    --gemini-index-dir /workspace/gemini_index \
    --out /root/pass_at_k/literature_search_results.json

  python eval_literature_search.py report \
    --results /root/pass_at_k/literature_search_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")

# Citation tags: short alphanumeric strings like "JYW20", "US21", "XCJ+21", "KRNJ20"
# These are not resolvable via search API — they're internal reference keys
CITE_TAG_RE = re.compile(r"^[A-Z][A-Za-z+]{1,7}[0-9]{2}[a-z]?$")


# ---------------------------------------------------------------------------
# Search API client (for extract + ground truth)
# ---------------------------------------------------------------------------

def _api_post(url: str, path: str, payload: dict, timeout: int = 120) -> dict:
    full_url = url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        full_url, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"  [warn] API {path}: {e}", file=sys.stderr)
        return {}


def search_papers(queries: list[str], search_url: str, max_results: int = 5) -> list[dict]:
    resp = _api_post(search_url, "/batch_search", {
        "queries": queries, "max_results": max_results, "sort_by": "importance",
    })
    papers = resp.get("papers", [])
    return [
        {k: p.get(k, "") for k in ["arxiv_id", "title", "abstract", "year", "citation_count"]}
        for p in papers
    ]


# ---------------------------------------------------------------------------
# Gemini embedding index (for relevance scoring)
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """Wraps the LanceDB Gemini index for fast pairwise similarity computation."""

    def __init__(self, gemini_api_key: str, index_dir: str):
        from arxiv_search_kit import ArxivClient
        self._client = ArxivClient(
            embedding="gemini",
            gemini_api_key=gemini_api_key,
            index_dir=index_dir,
        )
        self._store = self._client._store
        self._cache: dict[str, np.ndarray | None] = {}

    def get_vector(self, arxiv_id: str) -> np.ndarray | None:
        if arxiv_id in self._cache:
            return self._cache[arxiv_id]
        vec = self._store.get_paper_vector(arxiv_id)
        self._cache[arxiv_id] = vec
        return vec

    def similarity(self, id1: str, id2: str) -> float | None:
        v1 = self.get_vector(id1)
        v2 = self.get_vector(id2)
        if v1 is None or v2 is None:
            return None
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ---------------------------------------------------------------------------
# LLM-based paper extraction from review text
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are analyzing a peer review to identify every paper it discusses as prior work or related literature. Your job is to extract a structured list — do NOT add papers the review doesn't mention, and do NOT omit papers it does mention.

For each paper the review discusses:
1. Find its arXiv ID if explicitly mentioned (e.g., "1704.07813")
2. Find its title if identifiable ("Unsupervised Learning of Depth and Ego-Motion from Video")
3. Find author+year if given ("Zhou et al. 2017" or "Zhou et al. (2017)")
4. If the review describes a paper by topic only (e.g., "unsupervised optical flow methods"), capture the topic description

Also capture what CLAIM the review makes about each paper — what does the review say this paper did, found, or contributed?

Return a JSON object with a single key "papers_discussed" containing a list of objects:
  {
    "papers_discussed": [
      {
        "arxiv_id": "1704.07813",           // null if not mentioned
        "title": "Unsupervised Learning...", // null if not identifiable
        "author_year": "Zhou et al. 2017",   // null if not given
        "topic_description": null,           // non-null only if paper is described by topic
        "claim": "showed that view synthesis can supervise depth and pose learning"
      }
    ]
  }

Rules:
- Include a paper even if only ONE of arxiv_id/title/author_year/topic_description is identifiable
- Prefer arXiv ID > title > author_year > topic_description as the primary identifier
- The "claim" field should capture what the review says THIS paper did — be specific
- If the same paper is discussed multiple times, include it once with the most specific identifier
- Skip self-references to the paper being reviewed
- Return ONLY the JSON object, no preamble

## Review Text

{review_text}"""


def _extract_papers_llm(review_text: str, api_key: str, model: str = "gemini-3.1-pro-preview") -> list[dict]:
    """Use Gemini to extract discussed papers from review text."""
    from google import genai
    from google.genai import types as genai_types

    if len(review_text) < 200:
        return []

    text = review_text if len(review_text) < 30000 else review_text[:30000]

    client = genai.Client(api_key=api_key)
    prompt = EXTRACTION_PROMPT.replace("{review_text}", text)

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        print(f"    [warn] LLM extraction failed: {e}", file=sys.stderr)
        return []

    cand = (resp.candidates or [None])[0]
    if not cand or not cand.content:
        return []

    text_out = ""
    for part in (cand.content.parts or []):
        if getattr(part, "text", None):
            text_out += part.text

    try:
        data = json.loads(text_out)
        return data.get("papers_discussed", [])
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text_out)
        if m:
            try:
                return json.loads(m.group()).get("papers_discussed", [])
            except json.JSONDecodeError:
                pass
        print(f"    [warn] LLM extraction produced invalid JSON: {text_out[:200]}", file=sys.stderr)
        return []


def _resolve_extracted(extracted: list[dict], search_url: str) -> list[str]:
    """Resolve extracted paper descriptions to arXiv IDs via the search API.

    Resolution priority:
      1. arXiv ID — use directly, no search needed
      2. Full title — search by title (most reliable text query)
      3. Claim (what the paper did) — much better than author-year string
      4. Author-year + claim combined
    """
    resolved: list[str] = []

    for paper in extracted:
        aid = paper.get("arxiv_id")
        if aid and ARXIV_ID_RE.match(str(aid)):
            resolved.append(aid)
            continue

        title = paper.get("title")
        if title and len(str(title)) > 10:
            results = search_papers([str(title)], search_url, max_results=1)
            if results:
                resolved.append(results[0]["arxiv_id"])
                continue

        # Use the claim as the search query — it describes what the paper did
        claim = paper.get("claim", "")
        author_year = paper.get("author_year", "")
        ay_str = str(author_year).strip() if author_year else ""

        if claim and len(str(claim)) > 15:
            # Combine claim with year for better accuracy
            year_match = re.search(r"(\d{4})", ay_str)
            query = str(claim)
            if year_match:
                query += f" {year_match.group(1)}"
            results = search_papers([query], search_url, max_results=1)
            if results:
                resolved.append(results[0]["arxiv_id"])
                continue

        # Fall back to author-year (skip citation tags)
        if ay_str and len(ay_str) > 3 and not CITE_TAG_RE.match(ay_str):
            results = search_papers([ay_str], search_url, max_results=1)
            if results:
                resolved.append(results[0]["arxiv_id"])
                continue

        topic = paper.get("topic_description")
        if topic and len(str(topic)) > 10:
            results = search_papers([str(topic)], search_url, max_results=1)
            if results:
                resolved.append(results[0]["arxiv_id"])
                continue

    return list(set(resolved))


# ---------------------------------------------------------------------------
# Review text extraction (per source type)
# ---------------------------------------------------------------------------

def _get_review_text_deepseek(results_dir: Path) -> dict[str, str]:
    reviews: dict[str, str] = {}
    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        pid = paper_dir.name
        for attempt_dir in sorted(paper_dir.iterdir()):
            if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt_"):
                continue
            traj = attempt_dir / "trajectory.json"
            if not traj.is_file():
                continue
            try:
                data = json.loads(traj.read_text())
                steps = data.get("steps", [])
                agent_msgs = [
                    s["message"] for s in steps
                    if s.get("source") == "agent" and s.get("message") and len(s["message"]) > 200
                ]
                review_markers = ("### Summary", "### Strengths", "### Weaknesses",
                                  "### Scores", "**Scores**")
                review = ""
                for msg in reversed(agent_msgs):
                    if any(m in msg for m in review_markers):
                        review = msg
                        break
                if not review and agent_msgs:
                    review = agent_msgs[-1]
                if review:
                    reviews[pid] = review
                    break
            except Exception:
                pass
    return reviews


def _get_review_text_stanford(reviews_dir: Path) -> dict[str, str]:
    reviews: dict[str, str] = {}
    for f in sorted(reviews_dir.glob("*_review.json")):
        paper_id = f.stem.replace("_review", "")
        try:
            d = json.loads(f.read_text())
            content = d.get("content", "")
            if len(content) > 200:
                reviews[paper_id] = content
        except Exception:
            pass
    return reviews


def _get_review_text_deepreviewer(base_dir: Path) -> dict[str, str]:
    reviews: dict[str, str] = {}
    jobs_dir = base_dir / "data" / "jobs"
    if not jobs_dir.is_dir():
        return reviews
    for job_dir in sorted(jobs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        report = job_dir / "final_report.md"
        if not report.is_file():
            continue
        content = report.read_text(errors="replace")
        if len(content) < 200:
            continue
        job_json = job_dir / "job.json"
        paper_id = job_dir.name
        if job_json.is_file():
            try:
                paper_id = json.loads(job_json.read_text()).get("paper_id", job_dir.name)
            except Exception:
                pass
        reviews[paper_id] = content
    return reviews


def _get_review_text_agentreview(base_dir: Path) -> dict[str, str]:
    reviews: dict[str, str] = {}
    all115 = base_dir / "paper_review" / "ALL115" / "gpt-4o" / "BASELINE"
    if not all115.is_dir():
        return reviews
    for paper_dir in sorted(all115.iterdir()):
        if not paper_dir.is_dir():
            continue
        pid = paper_dir.name
        review_file = paper_dir / f"{pid}.json"
        if not review_file.is_file():
            continue
        try:
            d = json.loads(review_file.read_text())
            review = d.get("review") or d.get("content") or ""
            if isinstance(review, dict):
                review = json.dumps(review)
            if len(str(review)) > 200:
                reviews[pid] = str(review)
        except Exception:
            pass
    return reviews


# ---------------------------------------------------------------------------
# Ground truth builder (Tier B: human-reviewed papers, for HCR when available)
# ---------------------------------------------------------------------------

def build_ground_truth(
    data_dir: Path,
    search_api_url: str,
    gemini_api_key: str,
    out_path: Path,
    max_papers: int | None = None,
) -> dict:
    """Extract papers discussed in human reviews (Tier B) for HCR metric."""
    task_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "task_metadata.json").exists()
    ])
    if max_papers:
        task_dirs = task_dirs[:max_papers]

    ground_truth: dict[str, dict] = {}

    for i, td in enumerate(task_dirs):
        paper_id = td.name
        print(f"[{i+1}/{len(task_dirs)}] {paper_id} ...", end=" ", flush=True)

        gt: dict[str, Any] = {"paper_id": paper_id, "tier_b_ids": []}

        meta_path = td / "task_metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            human_reviews = meta.get("human_reviews", [])
            all_human_text = "\n\n---REVIEW---\n\n".join(
                str(r) for r in human_reviews if isinstance(r, str)
            )
            if len(all_human_text) > 200:
                extracted = _extract_papers_llm(all_human_text, gemini_api_key)
                gt["tier_b_extracted"] = extracted
                gt["tier_b_ids"] = _resolve_extracted(extracted, search_api_url)

        print(f"TierB={len(gt['tier_b_ids'])}")
        ground_truth[paper_id] = gt
        time.sleep(0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ground_truth, indent=2))
    print(f"\nGround truth saved to {out_path} ({len(ground_truth)} papers)")
    return ground_truth


# ---------------------------------------------------------------------------
# Extractor (per baseline)
# ---------------------------------------------------------------------------

def _extract_arxiv_ids_regex(review_text: str, own_paper_id: str) -> list[str]:
    """Extract arXiv IDs directly from review text using regex.

    For agents that explicitly write arXiv:XXXX.XXXX or XXXX.XXXXX in their reviews.
    Much more reliable than LLM extraction — catches exactly what the agent cited.
    """
    ids = set()
    # Match "arXiv:XXXX.XXXXX" or "arxiv:XXXX.XXXXX"
    for m in re.finditer(r'(?:arxiv:\s*)?(\d{4}\.\d{4,5})', review_text, re.IGNORECASE):
        ids.add(m.group(1))
    # Remove the paper's own ID
    own = own_paper_id.split("v")[0]
    ids.discard(own)
    return sorted(ids)


def extract_papers(
    name: str,
    results_dir: Path | None,
    reviews_dir: Path | None,
    deepreviewer_dir: Path | None,
    agentreview_dir: Path | None,
    search_api_url: str,
    gemini_api_key: str,
    out_path: Path,
    max_papers: int | None = None,
    mode: str = "llm",
) -> dict[str, dict]:
    """Extract papers discussed in reviews for a single baseline.

    mode: "regex" — extract arXiv IDs directly from review text (for agents that
          write explicit arXiv IDs like `arXiv:1704.07813`). No API calls needed.
          "llm" — use Gemini to extract paper descriptions and resolve via search
          API (for reviews that cite by author-year like "Zhou et al. (2017)").
    """
    if results_dir:
        reviews = _get_review_text_deepseek(results_dir)
    elif reviews_dir:
        reviews = _get_review_text_stanford(reviews_dir)
    elif deepreviewer_dir:
        reviews = _get_review_text_deepreviewer(deepreviewer_dir)
    elif agentreview_dir:
        reviews = _get_review_text_agentreview(agentreview_dir)
    else:
        print("error: must provide one of --results-dir, --reviews-dir, --deepreviewer-dir, --agentreview-dir", file=sys.stderr)
        sys.exit(2)

    print(f"Extracting papers from {name} ({mode} mode): {len(reviews)} reviews found")
    if max_papers:
        reviews = dict(sorted(reviews.items())[:max_papers])

    output: dict[str, dict] = {}
    for i, (paper_id, review_text) in enumerate(sorted(reviews.items())):
        print(f"  [{i+1}/{len(reviews)}] {paper_id} ...", end=" ", flush=True)

        if mode == "regex":
            resolved_ids = _extract_arxiv_ids_regex(review_text, paper_id)
            output[paper_id] = {
                "extracted": [], "arxiv_ids": resolved_ids, "count": len(resolved_ids),
            }
            print(f"{len(resolved_ids)} arXiv IDs found")
        else:
            extracted = _extract_papers_llm(review_text, gemini_api_key)
            resolved_ids = _resolve_extracted(extracted, search_api_url)
            output[paper_id] = {
                "extracted": extracted, "arxiv_ids": resolved_ids, "count": len(resolved_ids),
            }
            print(f"{len(extracted)} extracted -> {len(resolved_ids)} resolved")

        if (i + 1) % 10 == 0:
            time.sleep(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Saved to {out_path} ({len(output)} papers)")
    return output


# ---------------------------------------------------------------------------
# Scorer — uses Gemini embedding index for relevance scoring
# ---------------------------------------------------------------------------

def _score_one(paper_id: str, gt: dict, extracted: dict, index: EmbeddingIndex) -> dict:
    """Score one paper's discussed papers using embedding similarity.

    For each agent-discussed paper, compute cosine similarity to the paper
    under review using the Gemini embedding index.

    Returns:
      similarities: list of cosine similarities (one per agent paper in the index)
      tier_b_overlap: agent papers also found by human reviewers
      hcr: Human-Citation Recall (when Tier B available)
      mrs: Mean Relevance Score — mean cosine similarity
    """
    agent_ids = [aid for aid in extracted.get("arxiv_ids", [])
                 if ARXIV_ID_RE.match(aid)]
    tier_b = set(gt.get("tier_b_ids", []))

    # Compute similarities
    sims: list[float] = []
    sim_details: list[dict] = []
    for aid in agent_ids:
        sim = index.similarity(paper_id, aid)
        if sim is not None:
            sims.append(sim)
            sim_details.append({"arxiv_id": aid, "similarity": round(sim, 4)})

    sim_details.sort(key=lambda x: x["similarity"], reverse=True)

    # HCR: did the agent find human-flagged papers?
    agent_set = set(agent_ids)
    in_b = agent_set & tier_b
    hcr = len(in_b) / len(tier_b) if tier_b else None

    return {
        "paper_id": paper_id,
        "agent_count": len(agent_ids),
        "in_index_count": len(sims),
        "not_in_index_count": len(agent_ids) - len(sims),
        "tier_b_size": len(tier_b),
        "in_b": sorted(in_b),
        "in_b_missed": sorted(tier_b - agent_set),
        "similarities": sim_details,
        "mean_similarity": round(float(np.mean(sims)), 4) if sims else None,
        "median_similarity": round(float(np.median(sims)), 4) if sims else None,
        "max_similarity": round(float(np.max(sims)), 4) if sims else None,
        "min_similarity": round(float(np.min(sims)), 4) if sims else None,
        "hcr": round(hcr, 4) if hcr is not None else None,
    }


def score(
    ground_truth_path: Path,
    extracted_paths: dict[str, Path],
    gemini_api_key: str,
    gemini_index_dir: str,
    out_path: Path,
) -> dict:
    """Score all baselines against ground truth using embedding similarity."""
    print("Loading Gemini embedding index ...", end=" ", flush=True)
    index = EmbeddingIndex(gemini_api_key=gemini_api_key, index_dir=gemini_index_dir)
    print("done.")

    gt_all = json.loads(ground_truth_path.read_text())

    extracted_all: dict[str, dict] = {}
    for name, path in extracted_paths.items():
        extracted_all[name] = json.loads(path.read_text())

    # Shared papers
    shared = set(gt_all.keys())
    for data in extracted_all.values():
        shared &= set(data.keys())
    print(f"Ground truth: {len(gt_all)} papers")
    for name, data in extracted_all.items():
        print(f"  {name}: {len(data)} papers")
    print(f"  Shared: {len(shared)} papers")

    results: dict[str, Any] = {
        "per_baseline": {},
        "per_paper": [],
    }

    for paper_id in sorted(shared):
        paper_gt = gt_all.get(paper_id, {})
        paper_result = {
            "paper_id": paper_id,
            "ground_truth": {"tier_b_count": len(paper_gt.get("tier_b_ids", []))},
            "baselines": {},
        }
        for bname, bdata in extracted_all.items():
            if paper_id in bdata:
                paper_result["baselines"][bname] = _score_one(
                    paper_id, paper_gt, bdata[paper_id], index
                )
        results["per_paper"].append(paper_result)

    # Per-baseline aggregates
    for bname in extracted_all:
        scores = []
        for pp in results["per_paper"]:
            if bname in pp["baselines"]:
                s = pp["baselines"][bname]
                if s.get("mean_similarity") is not None:
                    scores.append(s)

        if not scores:
            continue

        n = len(scores)

        def _safe_mean(field):
            vals = [s[field] for s in scores if s.get(field) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        # Distribution of similarities across all agent papers
        all_sims = []
        for s in scores:
            for d in s.get("similarities", []):
                all_sims.append(d["similarity"])

        # Thresholds for relevance categories
        high_rel = sum(1 for x in all_sims if x >= 0.85)
        med_rel = sum(1 for x in all_sims if 0.75 <= x < 0.85)
        low_rel = sum(1 for x in all_sims if x < 0.75)

        results["per_baseline"][bname] = {
            "num_papers": n,
            "mrs": _safe_mean("mean_similarity"),
            "median_of_means": round(float(np.median([s["mean_similarity"] for s in scores])), 4),
            "mean_max_similarity": _safe_mean("max_similarity"),
            "mean_papers_discussed": round(sum(s["agent_count"] for s in scores) / n, 2),
            "mean_in_index": round(sum(s["in_index_count"] for s in scores) / n, 2),
            "mean_not_in_index": round(sum(s["not_in_index_count"] for s in scores) / n, 2),
            "hcr": _safe_mean("hcr"),
            "total_similarities": len(all_sims),
            "high_relevance_pct": round(high_rel / len(all_sims) * 100, 1) if all_sims else 0,
            "medium_relevance_pct": round(med_rel / len(all_sims) * 100, 1) if all_sims else 0,
            "low_relevance_pct": round(low_rel / len(all_sims) * 100, 1) if all_sims else 0,
        }

    # Paired comparison (first two baselines)
    names = list(extracted_all.keys())
    if len(names) >= 2:
        a, b = names[0], names[1]
        wins_a = defaultdict(int)
        wins_b = defaultdict(int)
        ties_count = defaultdict(int)
        for pp in results["per_paper"]:
            sa = pp["baselines"].get(a, {})
            sb = pp["baselines"].get(b, {})
            for metric in ["mean_similarity", "hcr"]:
                va = sa.get(metric)
                vb = sb.get(metric)
                if va is None or vb is None:
                    continue
                if va > vb:
                    wins_a[metric] += 1
                elif vb > va:
                    wins_b[metric] += 1
                else:
                    ties_count[metric] += 1
        results["paired_comparison"] = {
            "num_shared": len(shared),
            f"{a}_wins": dict(wins_a),
            f"{b}_wins": dict(wins_b),
            "ties": dict(ties_count),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

def report(results_path: Path) -> None:
    data = json.loads(results_path.read_text())

    print(f"\n{'='*80}")
    print("LITERATURE SEARCH QUALITY EVALUATION")
    print(f"{'='*80}")
    print()
    print("MRS (Mean Relevance Score) = mean cosine similarity between paper-under-review")
    print("  and agent-cited papers, using the Gemini embedding index (same embeddings")
    print("  the search API uses). Higher = agent found more relevant papers.")
    print()
    print("HCR (Human-Citation Recall) = fraction of human-flagged papers the agent also found.")
    print("  Only available when human reviews exist for the paper.")
    print()

    # Summary table
    header = f"{'Baseline':<25} {'Papers':>7} {'MRS':>8} {'MedMRS':>8} {'MaxSim':>8} {'HighRel%':>9} {'MedRel%':>9} {'LowRel%':>8} {'HCR':>8} {'Disc':>6} {'InIdx':>6}"
    print(header)
    print("-" * len(header))
    for name, stats in data.get("per_baseline", {}).items():
        print(
            f"{name:<25} {stats['num_papers']:>7} "
            f"{stats['mrs'] or 'N/A':>8} "
            f"{stats['median_of_means'] or 'N/A':>8} "
            f"{stats['mean_max_similarity'] or 'N/A':>8} "
            f"{stats['high_relevance_pct']:>8.1f}% "
            f"{stats['medium_relevance_pct']:>8.1f}% "
            f"{stats['low_relevance_pct']:>8.1f}% "
            f"{stats['hcr'] or 'N/A':>8} "
            f"{stats['mean_papers_discussed']:>6.1f} "
            f"{stats['mean_in_index']:>6.1f}"
        )
    print()

    # Paired comparison
    pc = data.get("paired_comparison", {})
    if pc:
        names = list(data.get("per_baseline", {}).keys())
        if len(names) >= 2:
            a, b = names[0], names[1]
            print(f"{'='*80}")
            print(f"HEAD-TO-HEAD ({pc.get('num_shared', '?')} shared papers)")
            print(f"{'='*80}")
            for metric, label in [("mean_similarity", "MRS"), ("hcr", "HCR")]:
                aw = pc.get(f"{a}_wins", {}).get(metric, 0)
                bw = pc.get(f"{b}_wins", {}).get(metric, 0)
                tw = pc.get("ties", {}).get(metric, 0)
                print(f"  {label}: {a}={aw}  {b}={bw}  ties={tw}")
            print()

    # Per-paper detail
    papers = data.get("per_paper", [])
    if papers:
        # Top/bottom by MRS
        ranked = []
        for pp in papers:
            mrss = [
                b.get("mean_similarity") for b in pp.get("baselines", {}).values()
                if b.get("mean_similarity") is not None
            ]
            avg = sum(mrss) / len(mrss) if mrss else -1
            ranked.append((pp["paper_id"], avg, pp))
        ranked.sort(key=lambda x: x[1], reverse=True)

        print(f"{'='*80}")
        print("TOP 5 PAPERS (highest avg MRS — agent found most relevant papers)")
        print(f"{'='*80}")
        for pid, avg_mrs, pp in ranked[:5]:
            gt = pp["ground_truth"]
            blines = pp.get("baselines", {})
            parts = []
            for n, b in blines.items():
                m = b.get("mean_similarity")
                h = b.get("hcr")
                parts.append(f"{n} MRS={m:.4f} HCR={h}" if m else f"{n}=N/A")
            print(f"  {pid}: avg_MRS={avg_mrs:.4f}  ({', '.join(parts)})  "
                  f"TierB={gt['tier_b_count']}")

        print(f"\n{'='*80}")
        print("BOTTOM 5 PAPERS (lowest avg MRS — search found less relevant papers)")
        print(f"{'='*80}")
        for pid, avg_mrs, pp in ranked[-5:]:
            gt = pp["ground_truth"]
            blines = pp.get("baselines", {})
            parts = []
            for n, b in blines.items():
                m = b.get("mean_similarity")
                parts.append(f"{n} MRS={m:.4f}" if m else f"{n}=N/A")
            print(f"  {pid}: avg_MRS={avg_mrs:.4f}  ({', '.join(parts)})  "
                  f"TierB={gt['tier_b_count']}")

        # Best individual discovery per baseline
        print(f"\n{'='*80}")
        print("TOP DISCOVERIES (highest individual similarity scores per baseline)")
        print(f"{'='*80}")
        for bname in data.get("per_baseline", {}):
            top = []
            for pp in papers:
                b = pp["baselines"].get(bname, {})
                for d in b.get("similarities", []):
                    top.append((d["similarity"], d["arxiv_id"], pp["paper_id"]))
            top.sort(reverse=True)
            print(f"  {bname}:")
            for sim, aid, pid in top[:5]:
                print(f"    paper={pid}  cited={aid}  similarity={sim:.4f}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Literature search quality evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build-ground-truth
    bgt = sub.add_parser("build-ground-truth")
    bgt.add_argument("--data-dir", type=str, required=True)
    bgt.add_argument("--search-api-url", type=str, required=True)
    bgt.add_argument("--gemini-api-key", type=str, required=True)
    bgt.add_argument("--out", type=str, required=True)
    bgt.add_argument("--max-papers", type=int, default=None)

    # extract
    ext = sub.add_parser("extract")
    ext.add_argument("--name", type=str, required=True)
    ext.add_argument("--results-dir", type=str, default=None)
    ext.add_argument("--reviews-dir", type=str, default=None)
    ext.add_argument("--deepreviewer-dir", type=str, default=None)
    ext.add_argument("--agentreview-dir", type=str, default=None)
    ext.add_argument("--search-api-url", type=str, required=True)
    ext.add_argument("--gemini-api-key", type=str, required=True)
    ext.add_argument("--out", type=str, required=True)
    ext.add_argument("--max-papers", type=int, default=None)
    ext.add_argument("--mode", type=str, default="llm", choices=["llm", "regex"])

    # score
    sco = sub.add_parser("score")
    sco.add_argument("--ground-truth", type=str, required=True)
    sco.add_argument("--extracted", type=str, action="append", required=True,
                     help="name=path pairs, e.g. --extracted deepseek=FILE --extracted stanford=FILE")
    sco.add_argument("--gemini-api-key", type=str, required=True)
    sco.add_argument("--gemini-index-dir", type=str, required=True)
    sco.add_argument("--out", type=str, required=True)

    # report
    rep = sub.add_parser("report")
    rep.add_argument("--results", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "build-ground-truth":
        build_ground_truth(
            data_dir=Path(args.data_dir),
            search_api_url=args.search_api_url,
            gemini_api_key=args.gemini_api_key,
            out_path=Path(args.out),
            max_papers=args.max_papers,
        )
    elif args.cmd == "extract":
        extract_papers(
            name=args.name,
            results_dir=Path(args.results_dir) if args.results_dir else None,
            reviews_dir=Path(args.reviews_dir) if args.reviews_dir else None,
            deepreviewer_dir=Path(args.deepreviewer_dir) if args.deepreviewer_dir else None,
            agentreview_dir=Path(args.agentreview_dir) if args.agentreview_dir else None,
            search_api_url=args.search_api_url,
            gemini_api_key=args.gemini_api_key,
            out_path=Path(args.out),
            max_papers=args.max_papers,
            mode=args.mode,
        )
    elif args.cmd == "score":
        extracted_paths = {}
        for e in args.extracted:
            name, path = e.split("=", 1)
            extracted_paths[name] = Path(path)
        score(
            ground_truth_path=Path(args.ground_truth),
            extracted_paths=extracted_paths,
            gemini_api_key=args.gemini_api_key,
            gemini_index_dir=args.gemini_index_dir,
            out_path=Path(args.out),
        )
    elif args.cmd == "report":
        report(results_path=Path(args.results))


if __name__ == "__main__":
    main()

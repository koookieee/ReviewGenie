"""
fetch_papers_with_reviews.py — Download 100 papers that have human reviews in OpenReview.

Strategy:
  1. Load ArXiv-150K metadata.parquet  →  5,243 papers overlap with openreview-filtered-reviews
  2. Match by paper_title (case-insensitive, exact)
  3. Prefer papers with 2-4 reviews; sample 100 deterministically (seed=42)
  4. Download LaTeX from arXiv e-print
  5. Write task_metadata.json with human_reviews populated

Usage:
    python fetch_papers_with_reviews.py --output-dir /root/data/pass_at_k_papers_reviewed
"""

import argparse
import asyncio
import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path

import aiohttp
import pandas as pd
from huggingface_hub import hf_hub_download

ARXIV_REPO_ID = "Vidushee/ArXiv-Papers-150K"
REVIEWS_REPO_ID = "Vidushee/openreview-filtered-reviews"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{paper_id}"
MAX_CONCURRENT = 20


def load_metadata(cache_dir: str) -> pd.DataFrame:
    print("Downloading ArXiv-150K metadata.parquet ...")
    path = hf_hub_download(
        repo_id=ARXIV_REPO_ID, filename="metadata.parquet",
        repo_type="dataset", cache_dir=cache_dir,
    )
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} papers")
    return df


def load_reviews(cache_dir: str) -> pd.DataFrame:
    print("Downloading filtered_all.csv from openreview-filtered-reviews ...")
    path = hf_hub_download(
        repo_id=REVIEWS_REPO_ID, filename="data/filtered_all.csv",
        repo_type="dataset", cache_dir=cache_dir,
    )
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(df)} reviews, all passed_filter=True")
    return df


def select_papers(meta_df: pd.DataFrame, reviews_df: pd.DataFrame,
                  num_papers: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """
    Returns (sampled_meta_df, reviews_by_title_lower).
    Prefers papers with 2-4 reviews for richer signal.
    """
    meta_df = meta_df.copy()
    reviews_df = reviews_df.copy()

    meta_df["title_lower"] = meta_df["title"].str.strip().str.lower()
    reviews_df["title_lower"] = reviews_df["paper_title"].str.strip().str.lower()

    # Count reviews per title
    review_counts = reviews_df.groupby("title_lower").size().reset_index(name="review_count")

    # Merge to find overlap
    matched = meta_df.merge(review_counts, on="title_lower", how="inner")
    print(f"  ArXiv papers with matching reviews: {len(matched)}")
    print(f"  Review count distribution: {matched['review_count'].value_counts().sort_index().to_dict()}")

    # Filter: prefer 2-4 reviews (good signal, not too noisy)
    preferred = matched[matched["review_count"].between(2, 4)]
    rest = matched[~matched["review_count"].between(2, 4)]
    print(f"  Preferred (2-4 reviews): {len(preferred)}, rest: {len(rest)}")

    # Sample: fill from preferred first, then rest
    rng = pd.Series(range(len(preferred))).sample(frac=1, random_state=seed).index
    preferred_shuffled = preferred.iloc[rng].reset_index(drop=True)

    rng2 = pd.Series(range(len(rest))).sample(frac=1, random_state=seed).index
    rest_shuffled = rest.iloc[rng2].reset_index(drop=True)

    pool = pd.concat([preferred_shuffled, rest_shuffled], ignore_index=True)

    # Take 30% extra to cover download failures
    n_oversample = min(int(num_papers * 1.3), len(pool))
    sampled = pool.head(n_oversample)
    print(f"  Oversampled {n_oversample} papers (target: {num_papers})")

    # Build reviews lookup: title_lower -> list of formatted review strings
    reviews_lookup: dict[str, list[str]] = {}
    for title_lower, group in reviews_df[reviews_df["title_lower"].isin(sampled["title_lower"])].groupby("title_lower"):
        reviews_lookup[title_lower] = [
            _format_review(row.to_dict())
            for _, row in group.iterrows()
        ]

    return sampled, reviews_lookup


def _val(v) -> str:
    """Return empty string for NaN/None, else str."""
    import math
    if v is None:
        return ""
    try:
        if math.isnan(float(v)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def _format_review(row: dict) -> str:
    """
    Convert a raw review row dict into a readable markdown string.

    The judge prompt uses human_reviews as plain text via:
        f"**Human Review {i+1}:**\\n\\n{r}"
    so each element of human_reviews must be a string, not a dict.

    We include all scoring fields (rating, confidence, soundness, contribution,
    presentation) so the judge can evaluate calibration properly.
    """
    parts = []

    # --- Scores block (needed for Calibration criterion) ---
    scores = {}
    for field in ["rating", "confidence", "soundness", "contribution", "presentation"]:
        v = _val(row.get(field))
        if v:
            scores[field.capitalize()] = v
    if scores:
        parts.append("**Scores:**")
        for k, v in scores.items():
            parts.append(f"- {k}: {v}")
        parts.append("")

    # --- Main review body: try structured fields first, fall back to 'review' ---
    # Some reviews use structured fields (summary, strengths, weaknesses, questions)
    # Others use a single 'review' blob. Use whichever has content.
    # Use _val per-field: `row.get(x) or row.get(y)` is buggy because
    # pandas NaN is truthy, so the short-circuit returns NaN instead of falling
    # through to the next candidate.
    def _first_val(*keys: str) -> str:
        for k in keys:
            v = _val(row.get(k))
            if v:
                return v
        return ""

    summary = _first_val("summary", "summary_of_the_paper")
    strengths = _val(row.get("strengths"))
    weaknesses = _val(row.get("weaknesses"))
    sw = _first_val("strength_and_weaknesses", "strengths_and_weaknesses")
    questions = _val(row.get("questions"))
    main_text = _first_val("main_review", "review", "summary_of_the_review")

    if summary:
        parts.append(f"**Summary:**\n{summary}\n")
    if strengths:
        parts.append(f"**Strengths:**\n{strengths}\n")
    if weaknesses:
        parts.append(f"**Weaknesses:**\n{weaknesses}\n")
    if sw and not (strengths or weaknesses):
        # Fallback: combined strengths/weaknesses field
        parts.append(f"**Strengths and Weaknesses:**\n{sw}\n")
    if questions:
        parts.append(f"**Questions:**\n{questions}\n")
    if main_text and not (summary or strengths or weaknesses or sw):
        # No structured fields — use the raw review blob
        parts.append(main_text)
    elif main_text and (summary or strengths or weaknesses or sw):
        # Structured fields present but also a main body — append if it adds new content
        # (some datasets have both a blob and structured fields that are subsets)
        if len(main_text) > 100 and main_text not in "".join(parts):
            parts.append(f"**Full Review:**\n{main_text}\n")

    return "\n".join(parts).strip()


def extract_source(data: bytes, latex_dir: Path) -> bool:
    latex_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("r:gz", "r:*"):
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as tar:
                tar.extractall(path=str(latex_dir), filter="data")
            return True
        except (tarfile.TarError, EOFError):
            pass
    try:
        text = gzip.decompress(data).decode("utf-8", errors="replace")
        if "\\documentclass" in text or "\\begin{document}" in text:
            (latex_dir / "main.tex").write_text(text)
            return True
    except Exception:
        pass
    try:
        text = data.decode("utf-8", errors="replace")
        if "\\documentclass" in text or "\\begin{document}" in text:
            (latex_dir / "main.tex").write_text(text)
            return True
    except Exception:
        pass
    return False


def find_main_tex(latex_dir: Path) -> Path | None:
    tex_files = list(latex_dir.rglob("*.tex"))
    if not tex_files:
        return None
    for tf in tex_files:
        try:
            content = tf.read_text(errors="replace")
            if "\\documentclass" in content and "\\begin{document}" in content:
                return tf
        except Exception:
            continue
    for name in ["main.tex", "paper.tex", "manuscript.tex", "article.tex"]:
        for tf in tex_files:
            if tf.name.lower() == name:
                return tf
    return max(tex_files, key=lambda f: f.stat().st_size)


def setup_task_dir(paper_dir: Path, row: dict, reviews: list[dict]) -> bool:
    latex_dir = paper_dir / "latex"
    main_tex = find_main_tex(latex_dir)
    if main_tex is None:
        return False

    template_tex = latex_dir / "template.tex"
    if main_tex != template_tex:
        shutil.copy2(main_tex, template_tex)

    metadata = {
        "paper_id": row.get("paper_id", ""),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "authors": row.get("authors", ""),
        "categories": row.get("categories", ""),
        "primary_category": row.get("primary_category", ""),
        "published": str(row.get("published", "")),
        "human_reviews": reviews,
    }
    (paper_dir / "task_metadata.json").write_text(json.dumps(metadata, indent=2))
    (paper_dir / "instruction.md").write_text("Review the paper at latex/template.tex\n")
    return True


async def fetch_one(
    session: aiohttp.ClientSession,
    row: dict,
    reviews: list[dict],
    output_dir: Path,
    sem: asyncio.Semaphore,
) -> tuple[str, bool]:
    paper_id = row["paper_id"]
    paper_dir = output_dir / paper_id
    latex_dir = paper_dir / "latex"

    # If latex is already downloaded, re-write metadata (ensures correct string format)
    if (paper_dir / "task_metadata.json").exists() and (latex_dir / "template.tex").exists():
        if setup_task_dir(paper_dir, row, reviews):
            return paper_id, True

    base_id = paper_id.split("v")[0] if "v" in paper_id else paper_id
    url = ARXIV_EPRINT_URL.format(paper_id=base_id)

    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    print(f"  HTTP {resp.status} for {paper_id}")
                    return paper_id, False
                data = await resp.read()
        except Exception as e:
            print(f"  Error fetching {paper_id}: {e}")
            return paper_id, False

    if not data or not extract_source(data, latex_dir):
        shutil.rmtree(paper_dir, ignore_errors=True)
        return paper_id, False

    if not setup_task_dir(paper_dir, row, reviews):
        shutil.rmtree(paper_dir, ignore_errors=True)
        return paper_id, False

    return paper_id, True


async def fetch_all(sampled: pd.DataFrame, reviews_lookup: dict,
                    output_dir: Path, target: int) -> tuple[list, list]:
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    headers = {"User-Agent": "PaperReviewBenchmark/1.0 (research)"}
    conn = aiohttp.TCPConnector(limit=MAX_CONCURRENT)

    async with aiohttp.ClientSession(headers=headers, connector=conn) as session:
        tasks = []
        for _, row in sampled.iterrows():
            row_dict = row.to_dict()
            title_lower = row_dict.get("title_lower", "")
            reviews = reviews_lookup.get(title_lower, [])
            tasks.append(fetch_one(session, row_dict, reviews, output_dir, sem))
        results = await asyncio.gather(*tasks)

    ok = [pid for pid, success in results if success]
    fail = [pid for pid, success in results if not success]
    print(f"  Downloaded: {len(ok)} OK, {len(fail)} failed")

    if len(ok) > target:
        ok_sorted = sorted(ok)
        to_remove = ok_sorted[target:]
        for pid in to_remove:
            shutil.rmtree(output_dir / pid, ignore_errors=True)
        ok = ok_sorted[:target]
        print(f"  Trimmed to {target} papers")

    return ok, fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-papers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="/tmp/hf_cache")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df = load_metadata(args.cache_dir)
    reviews_df = load_reviews(args.cache_dir)

    sampled, reviews_lookup = select_papers(meta_df, reviews_df, args.num_papers, args.seed)

    # Save manifest
    cols = [c for c in ["paper_id", "title", "primary_category", "review_count"] if c in sampled.columns]
    (output_dir / "manifest.json").write_text(
        json.dumps(sampled[cols].to_dict(orient="records"), indent=2)
    )
    print(f"  Manifest saved with {len(sampled)} candidates")

    ok, fail = asyncio.run(fetch_all(sampled, reviews_lookup, output_dir, args.num_papers))

    # Verify reviews are populated
    task_dirs = sorted([
        d.name for d in output_dir.iterdir()
        if d.is_dir() and (d / "task_metadata.json").exists()
    ])
    with_reviews = 0
    for td in task_dirs:
        meta = json.loads((output_dir / td / "task_metadata.json").read_text())
        if meta.get("human_reviews"):
            with_reviews += 1

    (output_dir / "final_manifest.json").write_text(json.dumps(task_dirs, indent=2))

    print(f"\n{'='*60}")
    print(f"DONE: {len(task_dirs)} papers ready (target: {args.num_papers})")
    print(f"  Papers with human reviews: {with_reviews}/{len(task_dirs)}")
    if fail:
        (output_dir / "failed_papers.json").write_text(json.dumps(fail, indent=2))
        print(f"  Failed downloads: {len(fail)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
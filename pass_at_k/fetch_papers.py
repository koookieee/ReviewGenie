"""
fetch_papers.py — Download 100 random papers from Vidushee/ArXiv-Papers-150K

Parallel downloads from arXiv e-print. Fetches metadata from HuggingFace,
samples 100 CS papers, downloads all LaTeX sources concurrently (~1 min).

Usage:
    python fetch_papers.py --output-dir /root/data/pass_at_k_papers
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

REPO_ID = "Vidushee/ArXiv-Papers-150K"
METADATA_FILE = "metadata.parquet"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{paper_id}"
MAX_CONCURRENT = 20  # parallel downloads


def download_metadata(cache_dir: str) -> pd.DataFrame:
    print("Downloading metadata.parquet from HuggingFace...")
    path = hf_hub_download(
        repo_id=REPO_ID, filename=METADATA_FILE,
        repo_type="dataset", cache_dir=cache_dir,
    )
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} papers")
    return df


def sample_papers(df: pd.DataFrame, num_papers: int, seed: int) -> pd.DataFrame:
    mask = df["abstract"].notna() & (df["abstract"].str.len() > 50)
    mask &= df["title"].notna() & (df["title"].str.len() > 10)
    mask &= df["primary_category"].str.startswith("cs.")
    pool = df[mask]
    print(f"  {len(pool)} CS papers with abstracts")
    # Oversample to cover failures
    n = min(int(num_papers * 1.3), len(pool))
    return pool.sample(n=n, random_state=seed)


def extract_source(data: bytes, latex_dir: Path) -> bool:
    """Extract LaTeX source from arXiv e-print response."""
    latex_dir.mkdir(parents=True, exist_ok=True)

    # Try tar.gz (most common)
    for mode in ("r:gz", "r:*"):
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as tar:
                tar.extractall(path=str(latex_dir), filter="data")
            return True
        except (tarfile.TarError, EOFError):
            pass

    # Try gzip single file
    try:
        text = gzip.decompress(data).decode("utf-8", errors="replace")
        if "\\documentclass" in text or "\\begin{document}" in text:
            (latex_dir / "main.tex").write_text(text)
            return True
    except Exception:
        pass

    # Raw tex
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


def setup_task_dir(paper_dir: Path, row: dict) -> bool:
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
        "human_reviews": [],
    }
    (paper_dir / "task_metadata.json").write_text(json.dumps(metadata, indent=2))
    (paper_dir / "instruction.md").write_text("Review the paper at latex/template.tex\n")
    return True


async def fetch_one(
    session: aiohttp.ClientSession,
    row: dict,
    output_dir: Path,
    sem: asyncio.Semaphore,
) -> tuple[str, bool]:
    paper_id = row["paper_id"]
    paper_dir = output_dir / paper_id
    latex_dir = paper_dir / "latex"

    # Skip if done
    if (paper_dir / "task_metadata.json").exists() and (latex_dir / "template.tex").exists():
        return paper_id, True

    base_id = paper_id.split("v")[0] if "v" in paper_id else paper_id
    url = ARXIV_EPRINT_URL.format(paper_id=base_id)

    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    return paper_id, False
                data = await resp.read()
        except Exception:
            return paper_id, False

    if not data or not extract_source(data, latex_dir):
        shutil.rmtree(paper_dir, ignore_errors=True)
        return paper_id, False

    if not setup_task_dir(paper_dir, row):
        shutil.rmtree(paper_dir, ignore_errors=True)
        return paper_id, False

    return paper_id, True


async def fetch_all(sampled: pd.DataFrame, output_dir: Path, target: int):
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    headers = {"User-Agent": "PaperReviewBenchmark/1.0 (research)"}
    conn = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(headers=headers, connector=conn) as session:
        rows = [row.to_dict() for _, row in sampled.iterrows()]
        tasks = [fetch_one(session, r, output_dir, sem) for r in rows]
        results = await asyncio.gather(*tasks)

    ok = [pid for pid, success in results if success]
    fail = [pid for pid, success in results if not success]
    print(f"  Downloaded: {len(ok)} OK, {len(fail)} failed")

    # If we have enough, trim extras; if not, report
    if len(ok) > target:
        # Remove extras (keep first `target` alphabetically for reproducibility)
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

    df = download_metadata(args.cache_dir)
    sampled = sample_papers(df, args.num_papers, args.seed)
    print(f"  Will try {len(sampled)} papers (target: {args.num_papers})")

    # Save manifest
    cols = [c for c in ["paper_id", "title", "primary_category"] if c in sampled.columns]
    (output_dir / "manifest.json").write_text(
        json.dumps(sampled[cols].to_dict(orient="records"), indent=2)
    )

    ok, fail = asyncio.run(fetch_all(sampled, output_dir, args.num_papers))

    # Save final list
    task_dirs = sorted([
        d.name for d in output_dir.iterdir()
        if d.is_dir() and (d / "task_metadata.json").exists()
    ])
    (output_dir / "final_manifest.json").write_text(json.dumps(task_dirs, indent=2))

    print(f"\n{'='*60}")
    print(f"DONE: {len(task_dirs)} papers ready (target: {args.num_papers})")
    if fail:
        (output_dir / "failed_papers.json").write_text(json.dumps(fail, indent=2))
        print(f"Failed: {len(fail)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
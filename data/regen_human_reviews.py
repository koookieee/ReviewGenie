"""regen_human_reviews.py — regenerate task_metadata.json.human_reviews for
papers using the NaN-safe _format_review (see RUNBOOK Part 7.2).

Use this when you suspect a paper's human_reviews got truncated to just the
Scores block. Idempotent: if reviews are already full-body, it keeps them.

Usage:
    # Regenerate specified paper IDs
    python regen_human_reviews.py --data-dir /root/data/pass_at_k_papers_reviewed \\
        --papers 1612.00472 1703.05698

    # Regenerate ALL papers in a data dir whose reviews look broken (< 500 chars each)
    python regen_human_reviews.py --data-dir /root/data/pass_at_k_papers_reviewed --auto-fix

    # Force regenerate every paper in the data dir
    python regen_human_reviews.py --data-dir /root/data/pass_at_k_papers_reviewed --all
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path


DEFAULT_CSV_GLOB = (
    "/workspace/.hf_home/hub/datasets--Vidushee--openreview-filtered-reviews/"
    "snapshots/*/data/filtered_all.csv"
)


def _load_formatter(repo_root: Path):
    """Import `_format_review` from fetch_papers_with_reviews.py in this repo."""
    path = repo_root / "fetch_papers_with_reviews.py"
    if not path.is_file():
        # Search in common locations
        for cand in [
            Path("/root/fetch_papers_with_reviews.py"),
            Path.cwd() / "fetch_papers_with_reviews.py",
        ]:
            if cand.is_file():
                path = cand
                break
    if not path.is_file():
        raise FileNotFoundError(
            "fetch_papers_with_reviews.py not found — pass --formatter-path"
        )
    spec = importlib.util.spec_from_file_location("fpwr", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod._format_review


def _load_csv(csv_glob: str):
    import pandas as pd  # lazy import

    paths = glob.glob(csv_glob)
    if not paths:
        raise FileNotFoundError(f"no CSV matches {csv_glob}")
    df = pd.read_csv(paths[0], low_memory=False)
    return df


def _match_rows(df, title: str):
    if not title:
        return df.iloc[0:0]
    norm = title.strip().lower()
    hits = df[df["paper_title"].astype(str).str.strip().str.lower() == norm]
    if len(hits):
        return hits
    # Substring fallback (first 40 chars)
    return df[
        df["paper_title"]
        .astype(str)
        .str.lower()
        .str.contains(norm[:40], regex=False, na=False)
    ]


def _looks_broken(meta: dict, min_chars: int = 500) -> bool:
    hr = meta.get("human_reviews") or []
    if not hr:
        return True
    return any(len(str(r)) < min_chars for r in hr)


def regen_one(
    paper_id: str,
    data_dir: Path,
    df,
    _format_review,
    *,
    force: bool = False,
    min_chars: int = 500,
    verbose: bool = True,
) -> dict:
    meta_path = data_dir / paper_id / "task_metadata.json"
    if not meta_path.is_file():
        return {"paper_id": paper_id, "status": "missing"}

    meta = json.loads(meta_path.read_text())
    if not force and not _looks_broken(meta, min_chars=min_chars):
        return {"paper_id": paper_id, "status": "ok_skip",
                "review_lens": [len(r) for r in meta.get("human_reviews", [])]}

    title = meta.get("title", "")
    rows = _match_rows(df, title)
    if not len(rows):
        return {"paper_id": paper_id, "status": "no_csv_match", "title": title[:80]}

    reviews = [_format_review(r.to_dict()) for _, r in rows.iterrows()]
    reviews = [r for r in reviews if len(r) > 200]  # drop obviously-empty
    if not reviews:
        return {"paper_id": paper_id, "status": "empty_after_format"}

    old_lens = [len(r) for r in meta.get("human_reviews", [])]
    meta["human_reviews"] = reviews
    meta_path.write_text(json.dumps(meta, indent=2))
    new_lens = [len(r) for r in reviews]
    if verbose:
        print(f"  {paper_id}: {old_lens} -> {new_lens}")
    return {"paper_id": paper_id, "status": "regenerated",
            "before": old_lens, "after": new_lens}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--csv-glob", default=DEFAULT_CSV_GLOB,
                   help="Glob for OpenReview filtered_all.csv (default: HF cache path)")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--papers", nargs="+", help="Specific paper IDs to regen")
    scope.add_argument("--all", action="store_true", help="Regen every paper in data-dir")
    scope.add_argument("--auto-fix", action="store_true",
                       help="Only regen papers whose reviews look broken (<500 chars each)")
    p.add_argument("--min-chars", type=int, default=500,
                   help="Any review under this length counts as broken (default 500)")
    p.add_argument("--force", action="store_true", help="Regen even if reviews look fine")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        p.error(f"data-dir not found: {data_dir}")

    # Resolve formatter + CSV
    _format_review = _load_formatter(Path(__file__).parent.parent)
    print(f"loading CSV from {args.csv_glob}", file=sys.stderr)
    df = _load_csv(args.csv_glob)
    print(f"  {len(df):,} review rows loaded", file=sys.stderr)

    # Decide scope
    all_ids = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if args.papers:
        scope_ids = args.papers
    elif args.all:
        scope_ids = all_ids
    else:
        # default / --auto-fix
        scope_ids = all_ids

    want_autofix = args.auto_fix or (not args.papers and not args.all and not args.force)

    print(f"scanning {len(scope_ids)} papers...")
    results = []
    for pid in scope_ids:
        # If --auto-fix, skip papers that look fine (unless forced)
        if want_autofix and not args.force:
            meta_path = data_dir / pid / "task_metadata.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                    if not _looks_broken(meta, min_chars=args.min_chars):
                        continue
                except Exception:
                    pass
        r = regen_one(
            pid, data_dir, df, _format_review,
            force=args.force, min_chars=args.min_chars,
        )
        results.append(r)

    # Summary
    from collections import Counter
    c = Counter(r["status"] for r in results)
    print(f"\n==> summary: {dict(c)}")
    missing = [r for r in results if r["status"] == "no_csv_match"]
    if missing:
        print(f"warning: {len(missing)} papers had no CSV match — listed in task_metadata but not in filtered_all.csv")
        for r in missing[:5]:
            print(f"  {r['paper_id']}: {r.get('title','')}")


if __name__ == "__main__":
    main()
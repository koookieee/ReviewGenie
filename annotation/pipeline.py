"""
pipeline.py — End-to-end annotation pipeline orchestrator.

Usage:
    python pipeline.py \\
        --trajectory trajectory_1612.00472_fab1.0_new.json \\
        --paper main.tex \\
        --arxiv-id 1612.00472 \\
        --title "Understanding image motion with group representations" \\
        --output usability_report.html

Phases:
    0. Download PDF + extract text layer (pymupdf, free)
    1. Parse trajectory  (free)
    2. Extract claims    (1 LLM call — Gemini Flash)
    3. Trace provenance  (1 LLM call — Gemini Flash, batched)
    4. Anchor to paper   (1 LLM call — Gemini Flash, batched, uses PDF text)
    5. Generate HTML report   (free)
    6. Annotate PDF           (1 LLM call for quote normalisation + pymupdf)

Total: 4 LLM calls per paper (3 without --annotate-pdf). Cost: <$0.02.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from trajectory_parser import parse_trajectory, summarize
from claim_extractor import ClaimExtractor, Claim
from trajectory_tracer import TrajectoryTracer, ClaimProvenance
from text_anchor import TextAnchorEngine, TextAnchor
from report_generator import ReportGenerator
from pdf_annotator import download_arxiv_pdf, annotate_pdf


def run_pipeline(
    trajectory_path: str | Path,
    paper_path: str | Path,
    api_key: str,
    arxiv_id: str = "",
    paper_title: str = "",
    output_path: str | Path = "usability_report.html",
    model: str = "gemini-3-flash-preview",
    save_intermediates: bool = True,
    annotate_pdf_flag: bool = True,
    verbose: bool = True,
) -> Path:
    """Run the full annotation pipeline and return the path to the output report."""
    t_start = time.time()

    def log(msg: str) -> None:
        if verbose:
            elapsed = time.time() - t_start
            print(f"[{elapsed:6.1f}s] {msg}", flush=True)

    out_dir = Path(output_path).parent
    pdf_path: Path | None = None
    pdf_text: str = ""

    # ──────────────────────────────────────────────
    # Phase 0: Download PDF + extract text layer
    # ──────────────────────────────────────────────
    if arxiv_id:
        log("Phase 0 — Downloading PDF and extracting text...")
        try:
            from pdftext.extraction import plain_text_output
            pdf_cache = out_dir / f"{arxiv_id}.pdf"
            pdf_path = download_arxiv_pdf(arxiv_id, pdf_cache)
            # pdftext gives clean rendered text with no LaTeX markup,
            # matching exactly what the PDF text layer contains.
            pdf_text = plain_text_output(str(pdf_path), sort=True, hyphens=False)
            log(f"  Extracted {len(pdf_text):,} chars via pdftext")
        except Exception as e:
            log(f"  WARNING: PDF extraction failed ({e}) — will use LaTeX source for anchoring")
            pdf_path = None
            pdf_text = ""

    # ──────────────────────────────────────────────
    # Phase 1: Parse trajectory
    # ──────────────────────────────────────────────
    log("Phase 1/5 — Parsing trajectory...")
    pt = parse_trajectory(trajectory_path)

    if verbose:
        summarize(pt)

    # Load paper text for context (LaTeX source used for provenance tracing)
    paper_text = pt.paper_text
    paper_file = Path(paper_path)
    if paper_file.is_file():
        paper_text = paper_file.read_text(encoding="utf-8", errors="replace")
        log(f"  Loaded LaTeX source from disk: {len(paper_text):,} chars")
    elif paper_text:
        log(f"  Using paper text from trajectory: {len(paper_text):,} chars")
    else:
        log("  WARNING: No paper text available — anchoring will be limited")

    # Use PDF text layer for anchoring if available (quotes match PDF exactly)
    anchor_text = pdf_text if pdf_text else paper_text
    if pdf_text:
        log(f"  Will use PDF text layer for anchoring ({len(anchor_text):,} chars)")

    # ──────────────────────────────────────────────
    # Phase 2: Extract claims
    # ──────────────────────────────────────────────
    log("Phase 2/5 — Extracting claims from review...")
    extractor = ClaimExtractor(api_key=api_key, model=model)
    claims = extractor.extract(pt.final_review)
    log(f"  Extracted {len(claims)} claims")

    # Print claim summary
    if verbose:
        from collections import Counter
        type_counts = Counter(c.claim_type for c in claims)
        for t, n in sorted(type_counts.items()):
            print(f"    {t:20s} {n}")

    # ──────────────────────────────────────────────
    # Phase 3: Trace provenance
    # ──────────────────────────────────────────────
    log("Phase 3/5 — Tracing provenance from trajectory...")
    tracer = TrajectoryTracer(api_key=api_key, model=model)
    provenances = tracer.trace(claims, pt)
    log(f"  Traced {len(provenances)} claim provenances")

    if verbose:
        prov_by_id = {p.claim_id: p for p in provenances}
        from collections import Counter
        conf_counts = Counter(prov_by_id[c.id].confidence for c in claims if c.id in prov_by_id)
        for conf, n in sorted(conf_counts.items()):
            print(f"    {conf:10s} {n}")

    # ──────────────────────────────────────────────
    # Phase 4: Anchor to paper text
    # ──────────────────────────────────────────────
    log("Phase 4/5 — Anchoring claims to paper text...")
    engine = TextAnchorEngine(api_key=api_key, model=model)
    anchors = engine.anchor(claims, provenances, anchor_text)
    log(f"  Anchored {len(anchors)} claims")

    if verbose:
        found = sum(1 for a in anchors if not a.not_found)
        print(f"    {found}/{len(anchors)} claims successfully anchored to paper text")

    # Save intermediates for debugging
    if save_intermediates:
        out_dir = Path(output_path).parent
        _save_intermediates(out_dir, claims, provenances, anchors, pt)
        log(f"  Saved intermediate JSON files to {out_dir}")

    # ──────────────────────────────────────────────
    # Phase 5: Generate HTML report
    # ──────────────────────────────────────────────
    log("Phase 5/5 — Generating HTML report...")
    gen = ReportGenerator()
    report_path = gen.generate(
        claims=claims,
        provenances=provenances,
        anchors=anchors,
        pt=pt,
        paper_title=paper_title or _infer_title(paper_text),
        output_path=output_path,
    )
    log(f"  HTML report: {report_path}")

    # ──────────────────────────────────────────────
    # Phase 6: Annotate PDF (optional)
    # ──────────────────────────────────────────────
    if annotate_pdf_flag and pdf_path:
        log("Phase 6 — Annotating PDF...")
        annotated_path = out_dir / f"annotated_{arxiv_id}.pdf"
        annotate_pdf(
            pdf_path=pdf_path,
            claims_path=out_dir / "claims.json",
            provenances_path=out_dir / "provenances.json",
            anchors_path=out_dir / "anchors.json",
            output_path=annotated_path,
            api_key=api_key,
            model=model,
            verbose=verbose,
        )
        log(f"  Annotated PDF: {annotated_path}")
    elif annotate_pdf_flag and not pdf_path:
        log("  Skipping PDF annotation — no arXiv ID provided (use --arxiv-id)")

    elapsed = time.time() - t_start
    log(f"Done in {elapsed:.1f}s — report: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_title(paper_text: str) -> str:
    """Try to extract paper title from LaTeX or Markdown source."""
    import re
    # LaTeX \title{...}
    m = re.search(r"\\title\{([^}]+)\}", paper_text)
    if m:
        return m.group(1).replace("\\\\", " ").replace("\\", "").strip()
    # Markdown # heading
    m = re.search(r"^# (.+)$", paper_text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return "Research Paper Review"


def _save_intermediates(
    out_dir: Path,
    claims: list[Claim],
    provenances: list[ClaimProvenance],
    anchors: list[TextAnchor],
    pt,
) -> None:
    """Save JSON intermediates for inspection and debugging."""
    from dataclasses import asdict as _asdict

    def safe_asdict(obj):
        try:
            return _asdict(obj)
        except Exception:
            return str(obj)

    (out_dir / "claims.json").write_text(
        json.dumps([safe_asdict(c) for c in claims], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "provenances.json").write_text(
        json.dumps([safe_asdict(p) for p in provenances], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "anchors.json").write_text(
        json.dumps([safe_asdict(a) for a in anchors], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "search_papers.json").write_text(
        json.dumps([safe_asdict(p) for p in pt.search_results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReviewGenie annotation pipeline — generates a usability report for paper authors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trajectory", required=True,
                        help="Path to ATIF trajectory JSON file")
    parser.add_argument("--paper", required=True,
                        help="Path to paper source file (.tex or .md)")
    parser.add_argument("--arxiv-id", default="",
                        help="arXiv paper ID (e.g. 1612.00472) — enables PDF download, text-layer anchoring, and PDF annotation")
    parser.add_argument("--title", default="",
                        help="Paper title (inferred from source if omitted)")
    parser.add_argument("--output", default="usability_report.html",
                        help="Output HTML report path")
    parser.add_argument("--model", default="gemini-3-flash-preview",
                        help="Gemini model to use (default: gemini-3-flash-preview)")
    parser.add_argument("--no-intermediates", action="store_true",
                        help="Skip saving intermediate JSON files")
    parser.add_argument("--no-pdf-annotation", action="store_true",
                        help="Skip PDF annotation phase")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        trajectory_path=args.trajectory,
        paper_path=args.paper,
        api_key=api_key,
        arxiv_id=args.arxiv_id,
        paper_title=args.title,
        output_path=args.output,
        model=args.model,
        save_intermediates=not args.no_intermediates,
        annotate_pdf_flag=not args.no_pdf_annotation,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

"""
pdf_annotator.py — Annotate the paper PDF directly with review claims.

For each claim with a found anchor:
  - Highlight the anchor_quote in the PDF (colored by claim type)
  - Add a sticky note with reviewer text, reasoning trail, confidence
  - Underline for questions, squiggle for nits
  - FreeText callout box in right margin for missing_citation claims

Downloads the PDF from arXiv if not cached locally.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
from google import genai
from google.genai import types as genai_types


# ---------------------------------------------------------------------------
# Color scheme (Nathan Lambert extended) — RGB tuples 0.0–1.0
# ---------------------------------------------------------------------------

CLAIM_COLORS: dict[str, tuple[float, float, float]] = {
    "strength":        (0.20, 0.78, 0.35),   # green
    "question":        (0.20, 0.55, 0.95),   # blue
    "nit":             (0.95, 0.28, 0.28),   # red
    "weakness":        (0.98, 0.58, 0.10),   # orange
    "methodology_gap": (0.65, 0.20, 0.85),   # purple
    "missing_citation":(0.95, 0.85, 0.10),   # yellow
    "suggestion":      (0.10, 0.82, 0.90),   # cyan
}

SEVERITY_OPACITY: dict[str, float] = {
    "critical": 1.0,
    "major":    0.85,
    "minor":    0.65,
    "nit":      0.50,
}


# ---------------------------------------------------------------------------
# LLM-based quote normaliser — convert LaTeX anchor quotes to PDF search text
# ---------------------------------------------------------------------------

_NORMALISE_PROMPT = """\
You are given a list of text excerpts that were taken from a LaTeX source file.
They may contain LaTeX markup such as \\cite{}, \\ref{}, \\textbf{}, \\texttt{},
math mode ($...$), \\footnote{}, etc.

For each excerpt, produce the "rendered" version — the plain text that would
appear in the compiled PDF — exactly as a reader would see it. Rules:
- Remove ALL LaTeX commands (\\cmd{arg} → arg, \\cmd → "")
- Replace math mode with a short readable description (e.g. "$2 \\times 10^{-5}$" → "2 × 10⁻⁵" or "2e-5")
- Resolve \\citep{key}/\\citet{key} as just the key in brackets e.g. "[key]"
- Resolve \\ref{label} as "§label" or just ""
- Strip \\footnote{...} entirely
- Keep the actual prose words intact — do not paraphrase
- Trim to the first 10-12 words if the excerpt is longer (we only need enough to uniquely locate it in the PDF)

Return ONLY a JSON object: {"results": [{"id": <same id>, "plain": "<plain text>"}]}
No preamble, no markdown fences. Start with { directly.

Excerpts:
{excerpts_json}
"""


def _normalise_quotes_with_llm(
    quotes: list[tuple[str, str]],   # list of (id, latex_quote)
    api_key: str,
    model: str = "gemini-3-flash-preview",
) -> dict[str, str]:
    """
    Use Gemini to convert LaTeX anchor quotes into plain PDF-searchable text.
    Returns mapping id → plain_text.
    """
    if not quotes:
        return {}

    client = genai.Client(api_key=api_key)
    excerpts = [{"id": qid, "latex": q} for qid, q in quotes]
    prompt = _NORMALISE_PROMPT.replace(
        "{excerpts_json}", json.dumps(excerpts, indent=2)
    )

    resp = client.models.generate_content(
        model=model,
        contents=[genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )],
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
            thinking_config=genai_types.ThinkingConfig(
                thinking_level=genai_types.ThinkingLevel.MINIMAL,
            ),
        ),
    )

    text = ""
    cand = (resp.candidates or [None])[0]
    if cand and cand.content:
        for part in (cand.content.parts or []):
            if getattr(part, "text", None):
                text += part.text

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        raw = json.loads(m.group()) if m else {}

    result: dict[str, str] = {}
    for item in raw.get("results", []):
        result[str(item.get("id", ""))] = str(item.get("plain", ""))
    return result


# ---------------------------------------------------------------------------
# Fuzzy text search — try progressively shorter substrings
# ---------------------------------------------------------------------------

def _search_text_in_pdf(
    page: fitz.Page,
    text: str,
    max_words: int = 12,
) -> list[fitz.Rect]:
    """
    Search for text on a PDF page. Tries progressively shorter prefixes
    until we get a hit or run out of words.
    Returns list of match rects (may be multiple if text wraps lines).
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    def _clean(t: str) -> str:
        # Strip LaTeX citation stubs: \citep{key}, \citet{key}, \cite{key}
        t = re.sub(r"\\cite[pt]?\{[^}]*\}", "", t)
        # Strip already-resolved bracket stubs: [key], [dapo], [opr]
        t = re.sub(r"\[\w[\w.\-]*\]", "", t)
        # Strip remaining LaTeX command wrappers but keep their text
        t = re.sub(r"\\(?:textbf|texttt|emph|textit)\{([^}]*)\}", r"\1", t)
        # Strip lone backslash commands with no args
        t = re.sub(r"\\\w+", " ", t)
        # Normalize whitespace
        t = re.sub(r"\s{2,}", " ", t).strip()
        return t

    cleaned = _clean(text)

    candidates = []
    for base in ([text, cleaned] if cleaned != text else [text]):
        words = base.split()
        candidates.append(base)
        for n in (max_words, 10, 8, 6):
            if len(words) > n:
                candidates.append(" ".join(words[:n]))

    # Deduplicate preserving order
    seen: set[str] = set()
    attempts = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen and len(c) >= 6:
            seen.add(c)
            attempts.append(c)

    for attempt in attempts:
        hits = page.search_for(attempt, quads=False)
        if hits:
            return hits

    return []


def _find_text_across_pages(
    doc: fitz.Document,
    anchor_quote: str,
    line_hint: int = 0,
) -> Optional[tuple[int, list[fitz.Rect]]]:
    """
    Search all pages for the anchor quote. Returns (page_idx, rects) or None.
    Prioritises pages near line_hint (rough heuristic: assume ~40 lines/page).
    """
    if not anchor_quote or len(anchor_quote) < 10:
        return None

    # Estimate which page to start from
    n_pages = doc.page_count
    start_page = max(0, (line_hint // 40) - 1) if line_hint > 0 else 0

    # Search starting from estimated page, then scan all pages
    search_order = list(range(start_page, n_pages)) + list(range(0, start_page))

    for pg_idx in search_order:
        page = doc[pg_idx]
        rects = _search_text_in_pdf(page, anchor_quote)
        if rects:
            return pg_idx, rects

    return None


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _clamp_rect_to_page(rect: fitz.Rect, page: fitz.Page) -> fitz.Rect:
    pb = page.rect
    return fitz.Rect(
        max(rect.x0, pb.x0),
        max(rect.y0, pb.y0),
        min(rect.x1, pb.x1),
        min(rect.y1, pb.y1),
    )


def _sticky_note_content(
    claim: dict,
    provenance: dict,
) -> str:
    """Build the text content for a sticky note annotation."""
    parts = []
    ctype = claim.get("claim_type", "").upper()
    severity = claim.get("severity", "")
    confidence = provenance.get("confidence", "")
    ev_type = provenance.get("evidence_type", "")

    parts.append(f"[{ctype}] {severity.upper()} | conf={confidence} | ev={ev_type}")
    parts.append("")
    parts.append("REVIEWER SAID:")
    review_text = claim.get("review_text", "")
    parts.append(review_text[:400] + ("..." if len(review_text) > 400 else ""))

    reasoning = provenance.get("reasoning_trail", "")
    if reasoning:
        parts.append("")
        parts.append("LLM REASONING:")
        parts.append(reasoning[:300] + ("..." if len(reasoning) > 300 else ""))

    ev_expl = provenance.get("evidence_explanation", "")
    if ev_expl:
        parts.append("")
        parts.append("EVIDENCE:")
        parts.append(ev_expl[:200])

    return "\n".join(parts)


def _citation_gap_callout(
    claim: dict,
    provenance: dict,
    anchor: dict,
    page: fitz.Page,
    ref_rect: fitz.Rect,
) -> None:
    """
    Add a FreeText callout box in the right margin for a missing_citation claim.
    """
    page_rect = page.rect
    margin_x0 = page_rect.width * 0.72
    margin_x1 = page_rect.width - 4

    # Position callout vertically aligned to the anchor rect
    box_h = 90
    y0 = max(ref_rect.y0 - 5, page_rect.y0 + 4)
    y1 = min(y0 + box_h, page_rect.y1 - 4)
    callout_rect = fitz.Rect(margin_x0, y0, margin_x1, y1)

    missing_abstract = provenance.get("missing_paper_abstract", "")
    insertion_quote = anchor.get("insertion_quote", "")
    review_text = claim.get("review_text", "")[:180]

    lines = ["MISSING CITATION"]
    if review_text:
        lines.append(review_text[:120] + ("..." if len(review_text) > 120 else ""))
    if missing_abstract:
        lines.append("")
        lines.append("Missing paper abstract:")
        lines.append(missing_abstract[:200] + ("..." if len(missing_abstract) > 200 else ""))
    if insertion_quote:
        lines.append("")
        lines.append(f'Near: "{insertion_quote[:80]}"')

    content = "\n".join(lines)

    annot = page.add_freetext_annot(
        rect=callout_rect,
        text=content,
        fontsize=5,
        fontname="helv",
        text_color=(0, 0, 0),
        fill_color=(1.0, 0.97, 0.7),   # pale yellow
        border_color=(0.8, 0.7, 0.0),
    )
    annot.set_opacity(0.92)
    annot.update()


# ---------------------------------------------------------------------------
# Main annotator
# ---------------------------------------------------------------------------

def annotate_pdf(
    pdf_path: str | Path,
    claims_path: str | Path,
    provenances_path: str | Path,
    anchors_path: str | Path,
    output_path: str | Path,
    api_key: str = "",
    model: str = "gemini-2.0-flash",
    verbose: bool = True,
) -> Path:
    """
    Annotate a PDF with review claims and save the result.
    Returns the path to the annotated PDF.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # Load data
    claims_data = json.loads(Path(claims_path).read_text())
    prov_data = json.loads(Path(provenances_path).read_text())
    anchors_data = json.loads(Path(anchors_path).read_text())

    claims_by_id = {c["id"]: c for c in claims_data}
    prov_by_id = {p["claim_id"]: p for p in prov_data}
    anchors_by_id = {a["claim_id"]: a for a in anchors_data}

    # ── LLM normalisation pass: convert LaTeX quotes → plain PDF text ────
    if api_key:
        log("Normalising anchor quotes via LLM...")
        to_normalise: list[tuple[str, str]] = []
        for claim_id, anchor in anchors_by_id.items():
            claim = claims_by_id.get(claim_id, {})
            claim_type = claim.get("claim_type", "")
            if claim_type == "missing_citation":
                q = anchor.get("insertion_quote", "") or anchor.get("anchor_quote", "")
            else:
                q = anchor.get("anchor_quote", "")
            if q and not anchor.get("not_found", True):
                to_normalise.append((claim_id, q))

        plain_quotes = _normalise_quotes_with_llm(to_normalise, api_key, model)
        log(f"  Normalised {len(plain_quotes)} quotes")

        # Patch anchors_by_id with plain versions
        for claim_id, plain in plain_quotes.items():
            if plain and claim_id in anchors_by_id:
                claim = claims_by_id.get(claim_id, {})
                claim_type = claim.get("claim_type", "")
                if claim_type == "missing_citation":
                    anchors_by_id[claim_id]["_plain_insertion_quote"] = plain
                else:
                    anchors_by_id[claim_id]["_plain_anchor_quote"] = plain
    else:
        log("WARNING: No API key — skipping LLM normalisation (LaTeX quotes may not match)")

    doc = fitz.open(str(pdf_path))
    log(f"Opened PDF: {pdf_path} ({doc.page_count} pages)")

    annotated = 0
    skipped = 0
    citation_gaps = 0

    for claim_id, anchor in anchors_by_id.items():
        claim = claims_by_id.get(claim_id, {})
        provenance = prov_by_id.get(claim_id, {})
        claim_type = claim.get("claim_type", "weakness")
        severity = claim.get("severity", "minor")

        color = CLAIM_COLORS.get(claim_type, (0.7, 0.7, 0.7))
        opacity = SEVERITY_OPACITY.get(severity, 0.65)

        # ── Handle missing_citation: use insertion_quote as the anchor ──
        # Prefer LLM-normalised (LaTeX→plain) versions if available
        if claim_type == "missing_citation":
            anchor_quote = (
                anchor.get("_plain_insertion_quote")
                or anchor.get("insertion_quote", "")
                or anchor.get("_plain_anchor_quote")
                or anchor.get("anchor_quote", "")
            )
        else:
            anchor_quote = (
                anchor.get("_plain_anchor_quote")
                or anchor.get("anchor_quote", "")
            )

        not_found = anchor.get("not_found", True)
        line_hint = anchor.get("anchor_line_approx", 0)

        if not_found or not anchor_quote:
            log(f"  [{claim_id}] SKIP — not_found or empty quote")
            skipped += 1
            continue

        # Search PDF for this text
        result = _find_text_across_pages(doc, anchor_quote, line_hint=line_hint)
        if not result:
            log(f"  [{claim_id}] NOT FOUND in PDF: \"{anchor_quote[:60]}\"")
            skipped += 1
            continue

        pg_idx, rects = result
        page = doc[pg_idx]

        # ── Union rect of all match rects ─────────────────────────────────
        union_rect = rects[0]
        for r in rects[1:]:
            union_rect = union_rect | r
        union_rect = _clamp_rect_to_page(union_rect, page)

        # ── Highlight (all claim types) ───────────────────────────────────
        hl = page.add_highlight_annot(rects)
        hl.set_colors(stroke=color)
        hl.set_opacity(opacity)
        hl.update()

        # ── Underline for questions ───────────────────────────────────────
        if claim_type == "question":
            ul = page.add_underline_annot(rects)
            ul.set_colors(stroke=color)
            ul.set_opacity(0.9)
            ul.update()

        # ── Squiggle for nits ─────────────────────────────────────────────
        if claim_type == "nit":
            sq = page.add_squiggly_annot(rects)
            sq.set_colors(stroke=color)
            sq.set_opacity(0.9)
            sq.update()

        # ── Sticky note with full claim content ──────────────────────────
        note_content = _sticky_note_content(claim, provenance)
        # Place sticky icon at the start of the first rect, left margin
        icon_pt = fitz.Point(union_rect.x0, union_rect.y0)
        sticky = page.add_text_annot(
            point=icon_pt,
            text=note_content,
            icon="Note",
        )
        sticky.set_colors(stroke=color, fill=color)
        sticky.set_opacity(0.95)
        sticky.update()

        # ── Citation gap callout box ──────────────────────────────────────
        if claim_type == "missing_citation":
            _citation_gap_callout(claim, provenance, anchor, page, union_rect)
            citation_gaps += 1

        annotated += 1
        log(f"  [{claim_id}] {claim_type.upper()} p.{pg_idx+1} — \"{anchor_quote[:50]}\"")

    out = Path(output_path)
    doc.save(str(out), garbage=4, deflate=True)
    doc.close()

    log(f"\nAnnotated {annotated} claims ({skipped} skipped, {citation_gaps} citation gap callouts)")
    log(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# arXiv PDF downloader
# ---------------------------------------------------------------------------

def download_arxiv_pdf(arxiv_id: str, dest: str | Path) -> Path:
    """Download a paper PDF from arXiv. Returns local path."""
    dest = Path(dest)
    if dest.exists():
        print(f"PDF already cached: {dest}")
        return dest

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"Downloading {url} → {dest}")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ReviewGenie/1.0)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print(f"Downloaded {dest.stat().st_size:,} bytes")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate a paper PDF with review claims."
    )
    parser.add_argument("--arxiv-id", required=True,
                        help="arXiv paper ID (e.g. 1612.00472)")
    parser.add_argument("--claims", required=True,
                        help="Path to claims.json")
    parser.add_argument("--provenances", required=True,
                        help="Path to provenances.json")
    parser.add_argument("--anchors", required=True,
                        help="Path to anchors.json")
    parser.add_argument("--output", default="annotated_paper.pdf",
                        help="Output PDF path")
    parser.add_argument("--pdf", default="",
                        help="Local PDF path (skips download if given)")
    parser.add_argument("--model", default="gemini-3-flash-preview",
                        help="Gemini model for quote normalisation")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf and Path(args.pdf).exists():
        pdf_path = Path(args.pdf)
    else:
        pdf_path = download_arxiv_pdf(
            args.arxiv_id,
            out_dir / f"{args.arxiv_id}.pdf",
        )

    annotate_pdf(
        pdf_path=pdf_path,
        claims_path=args.claims,
        provenances_path=args.provenances,
        anchors_path=args.anchors,
        output_path=args.output,
        api_key=api_key,
        model=args.model,
    )

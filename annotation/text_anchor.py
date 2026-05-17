"""
text_anchor.py — Anchor each claim to the exact text span(s) in the paper.

LLM-only approach. For each claim we pass:
  - the claim text
  - the key_phrases (from claim_extractor)
  - the paper_region_hint (from trajectory_tracer)
  - the relevant section(s) of the paper source text

The LLM returns:
  - anchor_quote: the exact verbatim phrase from the paper (≤60 words)
    that the claim is ABOUT (i.e. the text the review is critiquing/praising)
  - anchor_line_approx: approximate line number in the paper source
  - anchor_context: 1-2 sentence context explaining why this span is the anchor
  - not_found: true if the claim cannot be anchored to any specific paper span
    (e.g. it's about an external missing paper, or purely subjective)

Batches all claims in ONE LLM call to keep costs minimal.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

from claim_extractor import Claim
from trajectory_tracer import ClaimProvenance


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class TextAnchor:
    claim_id: str
    anchor_quote: str          # exact verbatim text from the paper (≤60 words)
    anchor_line_approx: int    # approximate line number in paper source (1-based)
    anchor_context: str        # why this is the right anchor
    not_found: bool            # True if no paper span applies
    # For missing_citation: the sentence in the paper WHERE the citation should appear
    insertion_quote: str = ""  # verbatim paper text near where citation belongs


# ---------------------------------------------------------------------------
# Paper section extractor — avoid sending full paper every time
# ---------------------------------------------------------------------------

def _extract_relevant_sections(
    paper_text: str,
    region_hint: str,
    key_phrases: list[str],
    window_lines: int = 60,
) -> tuple[str, dict[int, str]]:
    """
    Return (section_text, line_map) where line_map maps relative line index
    to absolute line number in the paper.

    Strategy: find lines containing any key_phrase or matching the region_hint,
    then expand to a window of ±window_lines.
    """
    lines = paper_text.split("\n")
    hit_lines: set[int] = set()

    # Search for key_phrases in the paper (case-insensitive)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for phrase in key_phrases:
            if phrase.lower() in line_lower:
                hit_lines.add(i)
                break

    # Also look for region_hint keywords
    if region_hint:
        hint_words = [w.lower() for w in region_hint.split() if len(w) > 4]
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if sum(1 for w in hint_words if w in line_lower) >= 2:
                hit_lines.add(i)

    if not hit_lines:
        # No hits — return first 120 lines + last 30 (intro + conclusion)
        selected = list(range(min(120, len(lines)))) + list(range(max(0, len(lines)-30), len(lines)))
        hit_lines = set(selected)

    # Expand windows around hits
    expanded: set[int] = set()
    for h in hit_lines:
        for j in range(max(0, h - window_lines // 2), min(len(lines), h + window_lines // 2)):
            expanded.add(j)

    sorted_indices = sorted(expanded)

    # Build output with absolute line numbers (1-based)
    result_lines = []
    line_map: dict[int, int] = {}  # relative idx -> absolute line number (1-based)
    for rel_idx, abs_idx in enumerate(sorted_indices):
        abs_lineno = abs_idx + 1
        line_map[rel_idx] = abs_lineno
        result_lines.append(f"{abs_lineno:4d} | {lines[abs_idx]}")

    return "\n".join(result_lines), line_map


# ---------------------------------------------------------------------------
# Anchor prompt
# ---------------------------------------------------------------------------

_ANCHOR_PROMPT = """\
You are an expert scientific editor helping authors understand exactly which parts
of their paper a reviewer is critiquing or praising.

You will be given:
1. A list of claims from a peer review, with key phrases and region hints
2. Numbered excerpts from the paper source text

For each claim, find the EXACT verbatim text span in the paper that the claim
is ABOUT — the sentence, phrase, table cell, or equation that the reviewer read
and is commenting on.

For each claim return:
- "claim_id": the claim's id
- "anchor_quote": the EXACT verbatim text from the paper (copy word-for-word,
    ≤60 words). This must be text that ACTUALLY APPEARS in the provided paper
    excerpts. If the claim is about a table value, quote the surrounding sentence
    plus the relevant table row header (do not make up numbers).
- "anchor_line_approx": the line number shown in the excerpt (the number before |)
    where the anchor_quote appears or starts.
- "anchor_context": 1-2 sentences explaining why this is the right anchor — what
    the reviewer is specifically commenting on about this passage.
- "not_found": true ONLY if:
    (a) this claim is about an EXTERNAL paper not present in this paper's text
        (e.g. a missing citation claim where the paper SHOULD cite something but
        the anchor is in the missing paper, not this one), OR
    (b) the claim is purely subjective with no specific paper passage, OR
    (c) the relevant passage simply does not appear in the excerpts provided.
- "insertion_quote": for missing_citation claims ONLY — copy verbatim the
    sentence or passage in the paper where the missing citation SHOULD have
    been added. Empty string for other claim types.

CRITICAL RULES:
- anchor_quote must be verbatim from the paper. Do NOT paraphrase.
- For weakness/strength/question/methodology_gap: always try to find an anchor.
    These claims are about specific things in the paper.
- For missing_citation: not_found=true for the missing paper itself, but
    insertion_quote should quote WHERE in the paper the citation belongs.
- If multiple passages are relevant, pick the most specific one.
- Do not fabricate line numbers. Use only numbers that appear in the excerpts.

## Paper excerpts (line number | text):

{paper_excerpts}

## Claims to anchor:

{claims_json}

Return ONLY a valid JSON object with key "anchors" containing the list.
IMPORTANT: All string values must be valid JSON — escape any double quotes inside strings
as \\", escape newlines as \\n. No preamble, no markdown fences. Start response with { directly.
"""


# ---------------------------------------------------------------------------
# LLM-based LaTeX → plain text converter
# ---------------------------------------------------------------------------

_DELATEX_PROMPT = """\
You are given numbered lines of a LaTeX paper source file.
Convert every line to the plain text that would appear in the compiled PDF.

Rules:
- Strip ALL LaTeX commands: \\textbf{X} → X, \\emph{X} → X, \\texttt{X} → X, etc.
- Remove citation commands entirely: \\citep{key}, \\citet{key}, \\cite{key} → ""
- Remove cross-reference commands: \\ref{label}, \\label{x}, \\eqref{x} → ""
- Remove footnotes: \\footnote{...} → ""
- Keep math content readable: $2 \\times 10^{-5}$ → 2 × 10⁻⁵
- Keep section headings as plain text (strip the \\section{} wrapper)
- Remove LaTeX structural commands (\\begin{}, \\end{}, \\item, etc.) but keep their text content
- Keep ALL plain prose words exactly as written — do NOT paraphrase or summarise
- Preserve the line number prefix exactly as given: "  42 | " must stay " 42 | "
- If a line is pure LaTeX with no readable content (e.g. \\vspace{}, \\hline), output the line number prefix with empty content

Return ONLY the converted lines, one per line, preserving the " NN | " prefix format.
No JSON, no preamble, no explanation.

Lines to convert:
{lines}
"""


def _delatex_with_llm(excerpt: str, client, model: str) -> str:
    """Use Gemini to convert a LaTeX excerpt to PDF-like plain text."""
    # Cap at ~300 lines to avoid huge prompts
    lines = excerpt.split("\n")
    if len(lines) > 300:
        lines = lines[:300]
    excerpt_capped = "\n".join(lines)

    prompt = _DELATEX_PROMPT.replace("{lines}", excerpt_capped)

    resp = client.models.generate_content(
        model=model,
        contents=[genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )],
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=16384,
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
    return text.strip()


# ---------------------------------------------------------------------------
# Anchor engine
# ---------------------------------------------------------------------------

class TextAnchorEngine:
    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview"):
        if not api_key:
            raise ValueError("api_key is required")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def anchor(
        self,
        claims: list[Claim],
        provenances: list[ClaimProvenance],
        paper_text: str,
    ) -> list[TextAnchor]:
        prov_by_id = {p.claim_id: p for p in provenances}

        # Build a combined excerpt covering all claims' relevant regions
        # We collect all key_phrases and region_hints together
        all_key_phrases: list[str] = []
        all_region_hints: list[str] = []
        for c in claims:
            all_key_phrases.extend(c.key_phrases)
            prov = prov_by_id.get(c.id)
            if prov and prov.paper_region_hint:
                all_region_hints.append(prov.paper_region_hint)

        excerpt, line_map = _extract_relevant_sections(
            paper_text,
            region_hint=" ".join(all_region_hints),
            key_phrases=all_key_phrases,
            window_lines=40,
        )

        # Build claims payload for the prompt — include provenance region hints
        claims_payload = []
        for c in claims:
            prov = prov_by_id.get(c.id)
            claims_payload.append({
                "claim_id": c.id,
                "claim_type": c.claim_type,
                "review_text": c.review_text,
                "key_phrases": c.key_phrases,
                "paper_region_hint": prov.paper_region_hint if prov else "",
                "cited_papers": c.cited_papers,
            })

        # Use replace() instead of .format() — paper excerpts can contain
        # literal { } characters (LaTeX) that would break str.format().
        prompt = (
            _ANCHOR_PROMPT
            .replace("{paper_excerpts}", excerpt[:40000])
            .replace("{claims_json}", json.dumps(claims_payload, indent=2))
        )

        resp = self._client.models.generate_content(
            model=self._model,
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=16384,
                # No response_schema — prevents premature truncation.
                # JSON validity enforced via prompt instructions instead.
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

        raw = _parse_json(text)
        anchors_data = raw.get("anchors", [])

        return [
            TextAnchor(
                claim_id=str(a.get("claim_id", "")),
                anchor_quote=str(a.get("anchor_quote", "")),
                anchor_line_approx=int(a.get("anchor_line_approx", 0) or 0),
                anchor_context=str(a.get("anchor_context", "")),
                not_found=bool(a.get("not_found", False)),
                insertion_quote=str(a.get("insertion_quote", "")),
            )
            for a in anchors_data
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from trajectory_parser import parse_trajectory
    from claim_extractor import ClaimExtractor
    from trajectory_tracer import TrajectoryTracer

    api_key = os.environ.get("GEMINI_API_KEY", "")
    traj_path = sys.argv[1] if len(sys.argv) > 1 else "trajectory_1612.00472_fab1.0_new.json"
    paper_path = sys.argv[2] if len(sys.argv) > 2 else "main.tex"

    pt = parse_trajectory(traj_path)
    paper_text = open(paper_path).read()

    extractor = ClaimExtractor(api_key=api_key)
    claims = extractor.extract(pt.final_review)
    print(f"Extracted {len(claims)} claims")

    tracer = TrajectoryTracer(api_key=api_key)
    provenances = tracer.trace(claims, pt)
    print(f"Traced {len(provenances)} provenances")

    engine = TextAnchorEngine(api_key=api_key)
    anchors = engine.anchor(claims, provenances, paper_text)

    anchor_by_id = {a.claim_id: a for a in anchors}
    for c in claims:
        a = anchor_by_id.get(c.id)
        if not a:
            print(f"[{c.id}] NO ANCHOR")
            continue
        nf = "NOT_FOUND" if a.not_found else f"line~{a.anchor_line_approx}"
        print(f"[{c.id}] {c.claim_type.upper()} | {nf}")
        if not a.not_found:
            print(f"  Quote:   \"{a.anchor_quote[:120]}\"")
        if a.insertion_quote:
            print(f"  Insert:  \"{a.insertion_quote[:100]}\"")
        print(f"  Context: {a.anchor_context[:120]}")
        print()

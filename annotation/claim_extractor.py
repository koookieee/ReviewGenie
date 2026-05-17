"""
claim_extractor.py — Extract structured claims from the final review using Gemini.

One LLM call on the review text only (~3-8K chars).
Returns a list of Claim objects covering every reviewable assertion.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal

from google import genai
from google.genai import types as genai_types


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

ClaimType = Literal[
    "weakness",           # a problem the reviewer found
    "strength",           # something the reviewer praised
    "question",           # an open question posed to the authors
    "missing_citation",   # reviewer says a paper is absent from the paper
    "methodology_gap",    # experimental design / missing ablation
    "suggestion",         # constructive suggestion for improvement
    "nit",                # minor grammar / presentation issue
]

Severity = Literal["critical", "major", "minor", "nit"]


@dataclass
class Claim:
    id: str                          # e.g. "W1", "S2", "MC3", "MG4"
    claim_type: ClaimType
    severity: Severity
    review_text: str                 # exact sentence(s) from the review
    review_section: str              # which review section it came from
    key_phrases: list[str]           # noun phrases for downstream anchoring
    cited_papers: list[str]          # arXiv IDs mentioned in this claim
    cited_paper_titles: list[str]    # human-readable titles if mentioned
    is_verifiable: bool              # can this be checked against paper text?


# ---------------------------------------------------------------------------
# Claim extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are an expert scientific editor. You will be given a peer review of a research paper.
Your task is to extract every distinct claim, critique, strength, suggestion, and question
from this review into a structured JSON list.

For each claim extract:
- "id": short unique label. Use prefix based on type:
    W = weakness, S = strength, Q = question, MC = missing_citation,
    MG = methodology_gap, SG = suggestion, N = nit
  e.g. "W1", "S1", "MC1", "MG1", "SG1", "Q1", "N1"
- "claim_type": one of: weakness | strength | question | missing_citation |
    methodology_gap | suggestion | nit
- "severity": one of: critical | major | minor | nit
- "review_text": the exact sentence(s) from the review that express this claim.
    Copy verbatim — do NOT paraphrase.
- "review_section": the review section heading this appears under
    (e.g. "Weaknesses", "Strengths", "Suggestions for Improvement", etc.)
- "key_phrases": list of 2-5 specific noun phrases that could be used to find
    the relevant passage in the paper. Focus on: method names, table/figure
    numbers, section numbers, equation names, metric names, baseline names,
    dataset names. Do NOT include generic words like "the paper" or "the authors".
- "cited_papers": list of arXiv IDs mentioned in this claim (e.g. ["1704.07813"]).
    Empty list if none.
- "cited_paper_titles": list of paper titles/author names mentioned
    (e.g. ["SfMLearner", "Zhou et al. 2017"]). Empty list if none.
- "is_verifiable": true if this claim can be checked against the paper body
    (e.g. numerical claim, comparative claim, citation presence/absence).
    false if it is purely subjective opinion.

Rules:
- Split compound claims into separate entries. One concern = one claim.
- For missing_citation claims: key_phrases = what topic/section should cite it,
  cited_papers = the missing paper's arXiv ID if given.
- For methodology_gap claims: key_phrases = the specific experiment or baseline missing.
- Do NOT merge claims. It's better to have 30 specific claims than 10 vague ones.
- Ignore score/rating lines (e.g. "Soundness: 2/4") — those are metadata, not claims.
- Include ALL suggestions, questions, strengths, weaknesses — be exhaustive.

Return ONLY a valid JSON object with a single key "claims" containing the list.
IMPORTANT: All string values must be valid JSON strings — escape any double quotes inside
strings as \\", escape newlines as \\n, and do not use unescaped control characters.
No preamble, no explanation, no markdown fences. Start your response with { directly.

## Review to analyse:

{review_text}
"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ClaimExtractor:
    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview"):
        if not api_key:
            raise ValueError("api_key is required")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def extract(self, review_text: str) -> list[Claim]:
        # Use replace instead of .format() to avoid issues with { } in review text
        prompt = _EXTRACTION_PROMPT.replace("{review_text}", review_text)

        resp = self._client.models.generate_content(
            model=self._model,
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=16384,
                # No response_schema — it causes premature truncation on long outputs.
                # We instruct the model to produce valid JSON in the prompt instead.
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
        claims_data = raw.get("claims", [])
        return [_dict_to_claim(d) for d in claims_data]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first JSON object
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())
        raise


def _dict_to_claim(d: dict) -> Claim:
    return Claim(
        id=str(d.get("id", "")),
        claim_type=d.get("claim_type", "weakness"),
        severity=d.get("severity", "minor"),
        review_text=str(d.get("review_text", "")),
        review_section=str(d.get("review_section", "")),
        key_phrases=[str(x) for x in d.get("key_phrases", [])],
        cited_papers=[str(x) for x in d.get("cited_papers", [])],
        cited_paper_titles=[str(x) for x in d.get("cited_paper_titles", [])],
        is_verifiable=bool(d.get("is_verifiable", False)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from trajectory_parser import parse_trajectory

    api_key = os.environ.get("GEMINI_API_KEY", "")
    traj_path = sys.argv[1] if len(sys.argv) > 1 else "trajectory_1612.00472_fab1.0_new.json"

    pt = parse_trajectory(traj_path)
    extractor = ClaimExtractor(api_key=api_key)
    claims = extractor.extract(pt.final_review)

    print(f"Extracted {len(claims)} claims:\n")
    for c in claims:
        vmark = "✓" if c.is_verifiable else "~"
        print(f"[{c.id}] {c.claim_type.upper()} ({c.severity}) {vmark}")
        print(f"  Section: {c.review_section}")
        print(f"  Text:    {c.review_text[:120]}...")
        if c.cited_papers:
            print(f"  ArXiv:   {c.cited_papers}")
        if c.cited_paper_titles:
            print(f"  Titles:  {c.cited_paper_titles}")
        print(f"  Keys:    {c.key_phrases}")
        print()

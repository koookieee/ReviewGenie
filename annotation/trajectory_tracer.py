"""
trajectory_tracer.py — Link each claim to its provenance in the trajectory.

Uses LLM only (no grep/regex). For each claim, sends the claim + the full
reasoning log + relevant tool-call observations to Gemini and asks it to
identify:
  - Which reasoning steps generated this claim
  - What search evidence supported it
  - What parts of the paper were being read when it was formed
  - Confidence level based on evidence quality

Batches ALL claims in one LLM call per paper to minimise token cost.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal

from google import genai
from google.genai import types as genai_types

from claim_extractor import Claim
from trajectory_parser import ParsedTrajectory, ReasoningEntry, ToolCall, SearchPaper


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

EvidenceType = Literal[
    "llm_confirmed_absent",   # LLM confirmed the thing is not in the paper
    "llm_confirmed_present",  # LLM confirmed the thing is in the paper
    "search_found",           # found via search API
    "read_section",           # derived from a Read tool call
    "reasoning_only",         # came from LLM reasoning without specific evidence
]


@dataclass
class ClaimProvenance:
    claim_id: str
    reasoning_trail: str          # the LLM's thinking text that led to this claim
    reasoning_step_ids: list[int] # which reasoning steps contributed
    evidence_type: EvidenceType
    confidence: Literal["high", "medium", "low"]
    supporting_search_papers: list[SearchPaper]   # papers from search that back this
    paper_region_hint: str        # hint about which part of the paper is relevant
    # For missing_citation claims: the abstract of the missing paper from search
    missing_paper_abstract: str = ""
    # LLM explanation of how the evidence maps to the claim
    evidence_explanation: str = ""


# ---------------------------------------------------------------------------
# Context builder — what we send to the LLM
# ---------------------------------------------------------------------------

def _build_reasoning_context(pt: ParsedTrajectory, max_chars: int = 30000) -> str:
    """Build a concise reasoning log string to include in the prompt."""
    lines = []
    for r in pt.reasoning_log:
        lines.append(f"[STEP {r.step_id} REASONING]\n{r.content}")
    full = "\n\n".join(lines)
    if len(full) > max_chars:
        # Truncate keeping first and last chunks (usually most informative)
        half = max_chars // 2
        full = full[:half] + "\n\n...[truncated]...\n\n" + full[-half:]
    return full


def _build_search_context(pt: ParsedTrajectory, max_papers: int = 40) -> str:
    """Build a concise list of all papers found during the review process."""
    lines = ["Papers found during literature search:"]
    for p in pt.search_results[:max_papers]:
        lines.append(
            f"  [{p.arxiv_id}] (step {p.found_at_step}, {p.citation_count} cites) "
            f"{p.title}\n  Abstract: {p.abstract[:300]}..."
        )
    return "\n".join(lines)


def _build_tool_call_context(pt: ParsedTrajectory) -> str:
    """Summarise all Bash/Read/Grep tool calls and their key results."""
    lines = ["Tool calls made during the review:"]
    for tc in pt.tool_calls:
        if tc.function_name in ("Read", "Bash", "Grep"):
            args_summary = _summarise_args(tc)
            result_preview = tc.result[:400].replace("\n", " ")
            lines.append(
                f"  [Step {tc.step_id}] {tc.function_name}({args_summary})\n"
                f"    Result: {result_preview}"
            )
    return "\n".join(lines)


def _summarise_args(tc: ToolCall) -> str:
    if tc.function_name == "Read":
        return tc.arguments.get("file_path", "?")
    if tc.function_name == "Bash":
        cmd = tc.arguments.get("command", "")
        return cmd[:100].replace("\n", " ")
    if tc.function_name == "Grep":
        return f"pattern={tc.arguments.get('pattern','?')}"
    return str(tc.arguments)[:80]


# ---------------------------------------------------------------------------
# Tracer prompt
# ---------------------------------------------------------------------------

_TRACER_PROMPT = """\
You are a research assistant helping trace the provenance of a paper reviewer's claims
back to the evidence the reviewer gathered during the review process.

You will be given:
1. A list of claims extracted from the final review
2. The reviewer's full reasoning log (internal thinking at each step)
3. All papers found during the literature search
4. A log of all tool calls (file reads, searches, verifications)

For each claim, identify:
- Which specific reasoning steps generated this claim (by step_id)
- A "reasoning_trail": the most relevant excerpt(s) from the reasoning log that
  led to this specific claim. Quote the actual reasoning text. Be precise.
- The "evidence_type": how the claim was established:
    "llm_confirmed_absent"  — the reviewer explicitly checked and found something
                              absent from the paper
    "llm_confirmed_present" — the reviewer verified something IS in the paper
    "search_found"          — the claim came from search results about external papers
    "read_section"          — the claim came from reading a specific section of the paper
    "reasoning_only"        — the claim came from the LLM's own reasoning without
                              a specific external verification step
- The "confidence":
    "high"   — the claim has direct evidence (verified against paper, confirmed by search)
    "medium" — the claim is plausible based on partial evidence
    "low"    — the claim is based on reasoning alone, no direct verification
- "supporting_arxiv_ids": list of arXiv IDs of search papers that directly support
  this claim (empty list if none)
- "paper_region_hint": a brief hint about which part of the paper this claim concerns
  (e.g. "Section 4.2, Table 2 (KITTI odometry results)" or "Related Work section,
  citation list"). Be specific. Empty string if the claim is about an external paper.
- "evidence_explanation": 1-2 sentences explaining how the evidence maps to the claim.
- "missing_paper_abstract": if this is a missing_citation claim and the missing paper
  was found in the search results, copy its abstract here. Otherwise empty string.

## Claims to trace:

{claims_json}

## Reviewer's Reasoning Log:

{reasoning_context}

## Papers Found in Search:

{search_context}

## Tool Calls Log:

{tool_calls_context}

Return ONLY a valid JSON object with key "provenances" containing a list, one entry per claim.
Each entry must have exactly these fields:
  claim_id, reasoning_step_ids (list of ints), reasoning_trail (string),
  evidence_type (string), confidence (string), supporting_arxiv_ids (list of strings),
  paper_region_hint (string), evidence_explanation (string), missing_paper_abstract (string)

IMPORTANT: All string values must be valid JSON — escape any double quotes inside strings
as \\", escape newlines as \\n. No preamble, no markdown fences. Start response with { directly.
"""


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class TrajectoryTracer:
    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview"):
        if not api_key:
            raise ValueError("api_key is required")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def trace(
        self,
        claims: list[Claim],
        pt: ParsedTrajectory,
    ) -> list[ClaimProvenance]:
        claims_json = json.dumps(
            [
                {
                    "id": c.id,
                    "claim_type": c.claim_type,
                    "severity": c.severity,
                    "review_text": c.review_text,
                    "key_phrases": c.key_phrases,
                    "cited_papers": c.cited_papers,
                    "cited_paper_titles": c.cited_paper_titles,
                }
                for c in claims
            ],
            indent=2,
        )

        # Use replace() instead of .format() — the reasoning/tool contexts can
        # contain literal { } characters that would break str.format().
        prompt = (
            _TRACER_PROMPT
            .replace("{claims_json}", claims_json)
            .replace("{reasoning_context}", _build_reasoning_context(pt))
            .replace("{search_context}", _build_search_context(pt))
            .replace("{tool_calls_context}", _build_tool_call_context(pt))
        )

        resp = self._client.models.generate_content(
            model=self._model,
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=32768,
                # No response_schema — prevents premature truncation on long outputs.
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
        provenance_data = raw.get("provenances", [])

        # Build lookup by claim_id for quick access to search papers
        search_by_id: dict[str, SearchPaper] = {p.arxiv_id: p for p in pt.search_results}

        provenances: list[ClaimProvenance] = []
        for pd_item in provenance_data:
            cid = pd_item.get("claim_id", "")
            arxiv_ids = [str(x) for x in pd_item.get("supporting_arxiv_ids", [])]
            supporting = [search_by_id[aid] for aid in arxiv_ids if aid in search_by_id]

            provenances.append(ClaimProvenance(
                claim_id=cid,
                reasoning_trail=str(pd_item.get("reasoning_trail", "")),
                reasoning_step_ids=[int(x) for x in pd_item.get("reasoning_step_ids", [])],
                evidence_type=pd_item.get("evidence_type", "reasoning_only"),
                confidence=pd_item.get("confidence", "low"),
                supporting_search_papers=supporting,
                paper_region_hint=str(pd_item.get("paper_region_hint", "")),
                missing_paper_abstract=str(pd_item.get("missing_paper_abstract", "")),
                evidence_explanation=str(pd_item.get("evidence_explanation", "")),
            ))

        return provenances


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

    api_key = os.environ.get("GEMINI_API_KEY", "")
    traj_path = sys.argv[1] if len(sys.argv) > 1 else "trajectory_1612.00472_fab1.0_new.json"

    pt = parse_trajectory(traj_path)
    extractor = ClaimExtractor(api_key=api_key)
    claims = extractor.extract(pt.final_review)
    print(f"Extracted {len(claims)} claims, tracing provenance...")

    tracer = TrajectoryTracer(api_key=api_key)
    provenances = tracer.trace(claims, pt)

    prov_by_id = {p.claim_id: p for p in provenances}
    for c in claims:
        prov = prov_by_id.get(c.id)
        if not prov:
            print(f"[{c.id}] NO PROVENANCE FOUND")
            continue
        print(f"[{c.id}] {c.claim_type.upper()} | conf={prov.confidence} | ev={prov.evidence_type}")
        print(f"  Region:    {prov.paper_region_hint}")
        print(f"  Evidence:  {prov.evidence_explanation}")
        if prov.supporting_search_papers:
            for sp in prov.supporting_search_papers:
                print(f"  Search:    [{sp.arxiv_id}] {sp.title[:60]}")
        print(f"  Reasoning: {prov.reasoning_trail[:200]}...")
        print()

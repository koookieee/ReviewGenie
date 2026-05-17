"""
trajectory_parser.py — Parse ATIF-v1.2 trajectory JSON into structured data.

Extracts:
- paper_text: the full paper source the agent read (from Read tool results)
- reasoning_log: list of (step_id, reasoning_content) — the LLM's thinking
- search_results: all papers found via search API (arxiv_id, title, abstract)
- tool_calls: structured list of every tool call + observation
- final_review: the last long agent message (the written review)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningEntry:
    step_id: int
    content: str
    timestamp: str = ""


@dataclass
class ToolCall:
    step_id: int
    function_name: str
    arguments: dict
    result: str       # raw text of the observation
    timestamp: str = ""


@dataclass
class SearchPaper:
    arxiv_id: str
    title: str
    abstract: str
    citation_count: int = 0
    found_at_step: int = 0


@dataclass
class ParsedTrajectory:
    paper_text: str                          # full paper source text (LaTeX/Markdown)
    paper_cutoff: str                        # YYYY-MM from paper_cutoff.txt
    reasoning_log: list[ReasoningEntry]      # LLM thinking across all steps
    search_results: list[SearchPaper]        # all papers found via search
    tool_calls: list[ToolCall]               # every tool call + result
    final_review: str                        # the written review
    session_id: str = ""
    model_name: str = ""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_trajectory(trajectory_path: str | Path) -> ParsedTrajectory:
    """Parse an ATIF-v1.2 trajectory JSON file into a ParsedTrajectory."""
    path = Path(trajectory_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    session_id = data.get("session_id", "")
    model_name = data.get("agent", {}).get("model_name", "")
    steps = data.get("steps", [])

    reasoning_log: list[ReasoningEntry] = []
    tool_calls_list: list[ToolCall] = []
    search_results: list[SearchPaper] = []
    seen_arxiv_ids: set[str] = set()

    paper_text = ""
    paper_cutoff = ""
    final_review = ""

    for step in steps:
        step_id = step.get("step_id", 0)
        timestamp = step.get("timestamp", "")
        source = step.get("source", "")

        # --- Collect reasoning ---
        reas = step.get("reasoning_content", "").strip()
        if reas:
            reasoning_log.append(ReasoningEntry(
                step_id=step_id,
                content=reas,
                timestamp=timestamp,
            ))

        # --- Build observation lookup: tool_call_id -> result text ---
        obs_lookup: dict[str, str] = {}
        observation = step.get("observation", {})
        for r in observation.get("results", []):
            cid = r.get("source_call_id", "")
            content = r.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            obs_lookup[cid] = str(content)

        # --- Process each tool call in this step ---
        for tc in step.get("tool_calls", []):
            fn = tc.get("function_name", "")
            args = tc.get("arguments", {})
            cid = tc.get("tool_call_id", "")
            result_text = obs_lookup.get(cid, "")

            tool_calls_list.append(ToolCall(
                step_id=step_id,
                function_name=fn,
                arguments=args,
                result=result_text,
                timestamp=timestamp,
            ))

            # --- Extract paper text from Read calls ---
            if fn == "Read":
                fpath = args.get("file_path", "")
                if "template.tex" in fpath or "main.tex" in fpath:
                    # Prefer the longest read (full paper, not a partial offset read)
                    if len(result_text) > len(paper_text):
                        paper_text = _strip_line_numbers(result_text)
                elif "paper_cutoff.txt" in fpath:
                    cleaned = _strip_line_numbers(result_text)
                    paper_cutoff = cleaned.strip().split("\n")[0].strip()
                    # Strip metadata lines
                    paper_cutoff = re.sub(r"\[metadata\].*", "", paper_cutoff).strip()

            # --- Extract search papers from Bash tool outputs ---
            elif fn == "Bash":
                _extract_search_papers(result_text, step_id, search_results, seen_arxiv_ids)

        # --- Capture final review: last agent message longer than 500 chars ---
        if source == "agent":
            msg = step.get("message", "").strip()
            if len(msg) > 500:
                # Prefer messages with review structure markers
                review_markers = (
                    "### Summary", "### Strengths", "### Weaknesses",
                    "### Scores", "**Scores**", "### Novelty"
                )
                if any(m in msg for m in review_markers):
                    final_review = msg
                elif not final_review:
                    final_review = msg

    # If no structured review found, fall back to last long agent message
    if not final_review:
        for step in reversed(steps):
            if step.get("source") == "agent":
                msg = step.get("message", "").strip()
                if len(msg) > 500:
                    final_review = msg
                    break

    # Strip any preamble like "Now I have all the information I need..."
    final_review = _strip_review_preamble(final_review)

    return ParsedTrajectory(
        paper_text=paper_text,
        paper_cutoff=paper_cutoff,
        reasoning_log=reasoning_log,
        search_results=search_results,
        tool_calls=tool_calls_list,
        final_review=final_review,
        session_id=session_id,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_line_numbers(text: str) -> str:
    """Remove cat -n style line number prefixes like '  42\t'."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Match leading whitespace + digits + tab
        m = re.match(r"^\s*\d+\t(.*)", line)
        if m:
            cleaned.append(m.group(1))
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def _extract_search_papers(
    result_text: str,
    step_id: int,
    search_results: list[SearchPaper],
    seen: set[str],
) -> None:
    """Extract papers from search API JSON responses embedded in Bash output."""
    # Search API returns JSON blobs with a "papers" array
    # Find all JSON objects in the result text
    for json_str in _find_json_objects(result_text):
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        papers_list = obj.get("papers", [])
        # Also handle {"results": [...]} shape from query endpoint
        if not papers_list and "results" in obj:
            results = obj["results"]
            if isinstance(results, dict):
                # {"results": {"arxiv_id": "answer text"}} — no paper metadata
                pass
            elif isinstance(results, list):
                papers_list = results

        for paper in papers_list:
            if not isinstance(paper, dict):
                continue
            arxiv_id = str(paper.get("arxiv_id", "")).strip()
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            search_results.append(SearchPaper(
                arxiv_id=arxiv_id,
                title=str(paper.get("title", "")).strip(),
                abstract=str(paper.get("abstract", "")).strip(),
                citation_count=int(paper.get("citation_count", 0) or 0),
                found_at_step=step_id,
            ))


def _find_json_objects(text: str) -> list[str]:
    """Find top-level JSON object strings in arbitrary text."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            in_str = False
            escape = False
            j = i
            while j < len(text):
                c = text[j]
                if escape:
                    escape = False
                elif c == "\\" and in_str:
                    escape = True
                elif c == '"':
                    in_str = not in_str
                elif not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:j+1]
                            if len(candidate) > 50:
                                results.append(candidate)
                            i = j
                            break
                j += 1
        i += 1
    return results


def _strip_review_preamble(review: str) -> str:
    """Remove non-review preamble text before the first ### heading."""
    # Find first markdown heading that looks like the review start
    markers = ["### Summary", "### Novelty", "### Strengths", "### Weaknesses"]
    for marker in markers:
        idx = review.find(marker)
        if idx > 0:
            return review[idx:].strip()
    return review.strip()


# ---------------------------------------------------------------------------
# Quick sanity print
# ---------------------------------------------------------------------------

def summarize(pt: ParsedTrajectory) -> None:
    print(f"Session:        {pt.session_id}")
    print(f"Model:          {pt.model_name}")
    print(f"Paper cutoff:   {pt.paper_cutoff}")
    print(f"Paper text:     {len(pt.paper_text):,} chars")
    print(f"Reasoning log:  {len(pt.reasoning_log)} entries")
    print(f"Tool calls:     {len(pt.tool_calls)} total")
    print(f"Search papers:  {len(pt.search_results)} unique papers found")
    print(f"Final review:   {len(pt.final_review):,} chars")
    print()
    print("Tool call breakdown:")
    from collections import Counter
    counts = Counter(tc.function_name for tc in pt.tool_calls)
    for name, n in counts.most_common():
        print(f"  {name:20s} {n}")
    print()
    print("Search papers found (first 10):")
    for p in pt.search_results[:10]:
        print(f"  [{p.arxiv_id}] step={p.found_at_step} cites={p.citation_count} — {p.title[:70]}")
    print()
    print("Review preview (first 300 chars):")
    print(pt.final_review[:300])


if __name__ == "__main__":
    import sys
    traj_path = sys.argv[1] if len(sys.argv) > 1 else "trajectory_1612.00472_fab1.0_new.json"
    pt = parse_trajectory(traj_path)
    summarize(pt)

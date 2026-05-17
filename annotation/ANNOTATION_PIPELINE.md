# ReviewGenie Annotation Pipeline

Transforms a raw review trajectory (ATIF-v1.2 JSON) + the paper source into two usability artifacts:

1. **HTML report** — triage panel, per-claim cards with reviewer text, reasoning trail, paper anchor, citation gap diff view, rebuttal stubs
2. **Annotated PDF** — the original arXiv paper with colored highlights, margin sticky notes, underlines, squiggles, and citation gap callout boxes written directly into the PDF

**Cost:** ~4 Gemini Flash calls per paper, < $0.02, ~70 seconds end-to-end.

---

## Architecture Overview

```
trajectory JSON + paper .tex
         │
         ▼
  Phase 0: Download PDF → pdftext extraction (clean plain text)
         │
         ▼
  Phase 1: Parse trajectory (free)
     ├─ reasoning_log (LLM thinking)
     ├─ search_results (papers found via search API)
     ├─ tool_calls (Read/Bash/Grep + observations)
     └─ final_review (the written review text)
         │
         ▼
  Phase 2: ClaimExtractor — 1 Gemini call on review text
     └─ list[Claim]: id, claim_type, severity, review_text,
                     key_phrases, cited_papers
         │
         ▼
  Phase 3: TrajectoryTracer — 1 Gemini call, batched over all claims
     └─ list[ClaimProvenance]: reasoning_trail, evidence_type,
                                confidence, supporting_search_papers,
                                paper_region_hint
         │
         ▼
  Phase 4: TextAnchorEngine — 1 Gemini call, batched over all claims
     │       input: pdftext output (clean, no LaTeX markup)
     └─ list[TextAnchor]: anchor_quote (verbatim PDF text),
                           anchor_line_approx, not_found,
                           insertion_quote (for missing_citation)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
  Phase 5: ReportGenerator           Phase 6: PDF Annotator
  → usability_report.html            → annotated_{arxiv_id}.pdf
  (always generated)                 (requires --arxiv-id)
```

---

## Modules

### `trajectory_parser.py`
Parses ATIF-v1.2 trajectory JSON into a `ParsedTrajectory` dataclass.

**Key logic:**
- Strips `cat -n` line number prefixes from Read tool results (`_strip_line_numbers`)
- Extracts search papers from Bash tool JSON outputs via balanced-brace JSON parser (`_find_json_objects`)
- Identifies the final review as the last agent message containing `### Summary` / `### Strengths` / `### Weaknesses` markers
- Strips pre-review preamble before the first `###` heading

**Output dataclasses:**
```python
ParsedTrajectory:
    paper_text: str          # LaTeX/Markdown source from Read calls
    paper_cutoff: str        # YYYY-MM from paper_cutoff.txt
    reasoning_log: list[ReasoningEntry]   # LLM thinking at each step
    search_results: list[SearchPaper]     # papers found via search API
    tool_calls: list[ToolCall]            # every tool call + result
    final_review: str                     # the written review
```

---

### `claim_extractor.py`
Extracts every distinct claim from the final review text in a single Gemini call.

**Claim types:** `weakness`, `strength`, `question`, `missing_citation`, `methodology_gap`, `suggestion`, `nit`

**Severity:** `critical`, `major`, `minor`, `nit`

**ID prefix convention:**
| Prefix | Type |
|--------|------|
| `W` | weakness |
| `S` | strength |
| `Q` | question |
| `MC` | missing_citation |
| `MG` | methodology_gap |
| `SG` | suggestion |
| `N` | nit |

**Key design decisions:**
- Uses `.replace("{review_text}", ...)` instead of `.format()` — review text often contains LaTeX `{` `}` which break `str.format()`
- No `response_schema` — enforces JSON via prompt instructions instead (schema caused premature truncation in Gemini)
- `ThinkingLevel.MINIMAL` (not `NONE` — that enum value does not exist in google-genai 2.3.0)

---

### `trajectory_tracer.py`
Links each claim to its provenance in the trajectory in a single batched Gemini call.

**Evidence types:**
| Value | Meaning |
|-------|---------|
| `llm_confirmed_absent` | Reviewer explicitly checked and found something absent |
| `llm_confirmed_present` | Reviewer verified something IS in the paper |
| `search_found` | Claim came from search results about external papers |
| `read_section` | Claim derived from a Read tool call |
| `reasoning_only` | Came from LLM reasoning without specific verification |

**Confidence:** `high` (direct evidence) / `medium` (partial evidence) / `low` (reasoning only)

**Context sent to LLM:**
- Reasoning log (first + last 15K chars, max 30K total)
- All search papers found (first 40, with abstracts)
- Tool call log (Bash/Read/Grep only, 400-char result previews)

**Output per claim:** `reasoning_trail`, `reasoning_step_ids`, `evidence_type`, `confidence`, `supporting_search_papers`, `paper_region_hint`, `missing_paper_abstract`, `evidence_explanation`

---

### `text_anchor.py`
Anchors each claim to its exact verbatim text span in the paper in a single batched Gemini call.

**Critical design:** Takes `pdftext`-extracted plain text as input (not the raw `.tex` source). This ensures the LLM copies text that `page.search_for()` can find verbatim in the PDF.

**Section extraction heuristic (`_extract_relevant_sections`):**
1. Searches all lines for `key_phrases` matches (case-insensitive)
2. Expands ±40 line windows around hits
3. Falls back to first 120 + last 30 lines if no hits
4. Outputs numbered `{lineno} | {text}` format

**Output per claim:**
```python
TextAnchor:
    anchor_quote: str          # verbatim PDF text (≤60 words)
    anchor_line_approx: int    # approximate line in paper source
    anchor_context: str        # why this is the right anchor
    not_found: bool            # True if no paper span applies
    insertion_quote: str       # for missing_citation: where in paper it should appear
```

**`not_found = True` when:**
- Claim is about an external paper not present in this paper's text
- Claim is purely subjective
- Relevant passage not in the provided excerpts

---

### `pdf_annotator.py`
Annotates the arXiv PDF with all four annotation types.

**Color scheme (Nathan Lambert extended):**
| Claim type | Color |
|------------|-------|
| strength | green |
| question | blue |
| nit | red |
| weakness | orange |
| methodology_gap | purple |
| missing_citation | yellow |
| suggestion | cyan |

**Annotation types applied per claim:**
1. **Highlight** — colored by claim type, opacity scaled by severity (critical=1.0 → nit=0.5)
2. **Sticky note** — reviewer text + LLM reasoning trail + confidence + evidence type
3. **Underline** — additionally applied for `question` claims
4. **Squiggle** — additionally applied for `nit` claims
5. **FreeText callout box** — right margin, for `missing_citation` claims only; contains missing paper abstract + insertion_quote

**LLM normalisation pass:** Before searching, sends all anchor quotes to Gemini to resolve any remaining LaTeX artifacts into plain text.

**Search strategy (`_search_text_in_pdf`):** Tries progressively shorter prefixes (full quote → 12 words → 10 → 8 → 6 words) until a match is found. Strips residual citation markup from each attempt.

**PDF search:** Uses `fitz.page.search_for()` (pymupdf). Works because anchor quotes come from `pdftext` output which is consistent with the PDF text layer.

---

### `pipeline.py`
End-to-end orchestrator for all 6 phases.

**CLI usage:**
```bash
python pipeline.py \
  --trajectory results/trajectory_1612.00472_fab1.0_new.json \
  --paper Test_Papers/arXiv-1612.00472v2/main.tex \
  --arxiv-id 1612.00472 \
  --output output/usability_report.html \
  --model gemini-3-flash-preview
```

**All flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--trajectory` | required | ATIF-v1.2 trajectory JSON |
| `--paper` | required | Paper `.tex` or `.md` source |
| `--arxiv-id` | `""` | arXiv ID — enables Phase 0 (PDF download) and Phase 6 (PDF annotation) |
| `--title` | inferred | Paper title (auto-extracted from `\title{}` or `# heading`) |
| `--output` | `usability_report.html` | HTML report output path |
| `--model` | `gemini-3-flash-preview` | Gemini model for all LLM calls |
| `--no-intermediates` | off | Skip saving intermediate JSONs |
| `--no-pdf-annotation` | off | Skip Phase 6 PDF annotation |
| `--quiet` | off | Suppress progress output |

**Intermediate files saved** (in output directory):
- `claims.json` — extracted claims
- `provenances.json` — traced provenances
- `anchors.json` — text anchors
- `search_papers.json` — all papers found during the review
- `{arxiv_id}.pdf` — downloaded PDF (cached)
- `annotated_{arxiv_id}.pdf` — annotated PDF output

**Without `--arxiv-id`:** Phases 0 and 6 are skipped; only the HTML report is generated, with anchoring using the `.tex` source (lower match rate due to LaTeX markup).

---

### `report_generator.py`
Generates the HTML usability report from all pipeline outputs.

**Sections:**
1. **Triage panel** — claims bucketed by confidence (high/medium/low), anchor links to detail cards
2. **Weaknesses + Methodology gaps** — per-claim cards with reviewer quote, paper anchor blockquote, LLM reasoning trail (yellow box), supporting papers list
3. **Citation gap diff view** — 2-column grid: left = paper passage (where citation should appear), right = missing paper abstract from search results
4. **Strengths, Questions, Suggestions, Nits**
5. **Rebuttal stubs** — pre-drafted response templates per weakness/methodology gap

---

## End-to-End Example

```bash
# On the remote machine (ssh -p 54871 root@171.226.34.64)
cd /root/annotation_module

# Paper 1: 1612.00472 — "Understanding image motion with group representations"
GEMINI_API_KEY=... python3 pipeline.py \
  --trajectory trajectory_1612.00472_fab1.0_new.json \
  --paper main.tex \
  --arxiv-id 1612.00472 \
  --output run_1612/report.html \
  --model gemini-3-flash-preview

# Paper 2: 2602.15849 — IntelliAsk RLVR paper
GEMINI_API_KEY=... python3 pipeline.py \
  --trajectory trajectory_2602.15849_skip_judge.json \
  --paper run_2602.15849/main.tex \
  --arxiv-id 2602.15849 \
  --output run_2602.15849/usability_report.html \
  --model gemini-3-flash-preview
```

**Outputs for paper 2602.15849:**
- `run_2602.15849/usability_report.html` — HTML report (30 claims, 28/30 high confidence)
- `run_2602.15849/annotated_2602.15849.pdf` — annotated PDF (22/30 claims placed)
- Runtime: ~68 seconds

---

## Benchmark Results (Test Papers)

| Paper | Claims | High conf | Anchored (HTML) | Annotated (PDF) | Time |
|-------|--------|-----------|-----------------|-----------------|------|
| 1612.00472 | 22 | 12/22 | 20/22 | — | ~54s |
| 2602.15849 | 30 | 28/30 | 25/30 | 22/30 | ~68s |

**Why 2602.15849 has higher confidence:** The trajectory for that paper has 13 Read calls (vs 4 for 1612.00472), giving the trajectory tracer richer explicit evidence trails.

**Why some claims are not_found in PDF:** The anchor LLM sets `not_found=True` for claims that are about external papers (e.g. missing citations are about a paper that isn't in this paper's body), purely subjective opinions, or where the relevant passage didn't appear in the extracted excerpt window.

---

## Key Design Decisions

### Why pdftext instead of the .tex source for anchoring
The `.tex` source contains LaTeX markup (`\citep{}`, `\textbf{}`, `$...$`, `\ref{}`). If the anchor LLM copies from the LaTeX source, the quote will contain markup that `page.search_for()` cannot find in the compiled PDF.

`pdftext` extracts the rendered text layer from the PDF with hyphenation resolved — exactly what appears on screen. Anchor quotes copied from this input match the PDF text layer verbatim.

### Why no `response_schema` in Gemini calls
Enforcing a `response_schema` in `gemini-3-flash-preview` caused premature output truncation at the token boundary for long outputs (>30 claims). All three LLM phases instead use prompt-level JSON instructions ("start with `{` directly, escape `"` as `\"`") and higher `max_output_tokens`.

### Why `.replace()` instead of `.format()` for prompts
Paper text and review text routinely contain literal `{` `}` characters (LaTeX, math, code). `str.format()` treats these as format placeholders and raises `ValueError: unmatched '{'`. All prompt assembly uses `.replace("{placeholder}", value)` chains.

### Why ThinkingLevel.MINIMAL
`ThinkingLevel.NONE` does not exist in `google-genai 2.3.0`. Available values: `HIGH`, `MEDIUM`, `LOW`, `MINIMAL`. Using `MINIMAL` keeps latency low while still allowing brief chain-of-thought before the JSON output.

---

## Dependencies

```bash
pip install google-genai pymupdf pdftext
```

| Package | Purpose |
|---------|---------|
| `google-genai` | Gemini API client |
| `pymupdf` (fitz) | PDF annotation (highlight, sticky note, underline, squiggle, freetext) |
| `pdftext` | Clean PDF text extraction (no LaTeX, hyphen-resolved) — used in Phase 0 |
| `requests` / `urllib` | arXiv PDF download |

**Environment variable required:**
```bash
export GEMINI_API_KEY=your_key_here
```

---

## File Structure

```
annotation/
├── pipeline.py              # Orchestrator — run this
├── trajectory_parser.py     # Phase 1: parse ATIF-v1.2 JSON
├── claim_extractor.py       # Phase 2: extract claims from review
├── trajectory_tracer.py     # Phase 3: trace claim provenance
├── text_anchor.py           # Phase 4: anchor claims to paper text
├── report_generator.py      # Phase 5: generate HTML report
├── pdf_annotator.py         # Phase 6: annotate PDF
└── ANNOTATION_PIPELINE.md   # this file
```

---

## Quick Start

```bash
# HTML only (no arxiv ID needed)
python pipeline.py --trajectory traj.json --paper main.tex --output report.html

# HTML + annotated PDF
python pipeline.py --trajectory traj.json --paper main.tex --arxiv-id 2602.15849 --output report.html
```

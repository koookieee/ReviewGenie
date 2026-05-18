---
name: paper-tools
description: Tools for reading and verifying content in the paper under review. Section reader, table extractor, figure finder, citation checker. Use these instead of grep or raw file reads.
argument-hint: "[tool name or question]"
---

# Paper Tools — Read, Verify, and Cross-Check the Paper

You have four local CLIs for targeted paper access. Use them instead of reading the whole file or running raw greps.

All tools operate on `/app/latex/template.tex` (the paper under review).

---

## 1. `read_section` — Read a named section

```bash
/app/read_section --list                        # see all section headings
/app/read_section "Introduction"                # read the Introduction
/app/read_section "Experiments" --subsections   # read Experiments + all subsections
/app/read_section "Related Work"
/app/read_section "Conclusion"
```

Section names are case-insensitive substring matches. If you want just one subsection, name it exactly. Use `--subsections` to pull the parent section and everything nested under it.

---

## 2. `read_table` — Extract a table by number or caption

```bash
/app/read_table --list                          # see all tables with captions
/app/read_table 1                               # Table 1
/app/read_table 2                               # Table 2
/app/read_table "accuracy"                      # first table whose caption contains "accuracy"
/app/read_table "ablation"
```

Use this to read exact numbers before writing any quantitative claim. Never transcribe numbers from memory.

---

## 3. `read_figure` — Find a figure by number or caption

```bash
/app/read_figure --list                         # see all figures with captions
/app/read_figure 1                              # Figure 1
/app/read_figure "architecture"                 # first figure with "architecture" in caption
/app/read_figure "comparison"
```

Returns the figure environment (LaTeX) or the image reference with surrounding context (Markdown). If the figure content was lost in PDF conversion, the caption alone helps you verify whether the reviewer's description is accurate.

---

## 4. `/app/search` — Search 928K+ arXiv CS/stat papers

See the `/search-papers` skill for full documentation. Quick reference:

```bash
/app/search batch "core topic" "method name" "related technique" --max 6 --sort importance
/app/search related <arxiv_id> --max 6
/app/search query <id1> <id2> --q what are the key contributions
/app/search query --pair <id1> "question 1" --pair <id2> "question 2"
```

The CLI auto-applies a temporal cutoff from `/app/paper_cutoff.txt` (submission month − 3 months). You will see `[search] applying --before YYYY-MM` on stderr.

---

## Workflow guidance

- **Start** by listing sections (`read_section --list`) and tables (`read_table --list`) to orient yourself.
- **Before writing any number**, read the table or figure it comes from.
- **Before claiming a paper is uncited**, run `grep -i "authorname" /app/latex/template.tex`.
- **Before claiming X outperforms Y**, read the table row for both methods.
- **After finding an interesting search result**, use `search query <id> --q "..."` to get a targeted summary rather than relying on the abstract alone.

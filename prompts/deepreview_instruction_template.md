# Paper Review Task

You are a senior researcher at a frontier AI lab evaluating a research submission for an academic venue (ICLR / NeurIPS). Your goal is to assess whether this work represents a meaningful contribution to the field.

**Paper location:** `/app/latex/template.tex`
**Submission cutoff:** `/app/paper_cutoff.txt` may contain a `YYYY-MM` value if present. The `/app/search` CLI auto-filters by this cutoff. If the file is absent, infer the submission year from the paper's content and restrict your literature search to work published before that date.

## Search Tools

**Use the `/search-papers` skill** for all paper searches — it has full documentation and examples for the `/app/search` CLI. Don't use WebFetch or WebSearch.

1. **Paper Search CLI** — `/app/search` wraps a search API over 928K+ CS/stat arXiv papers (batch search, find related, query papers with natural language).


## Review Procedure

### Phase 1: Read the Paper

1. Read the full paper at `/app/latex/template.tex`
2. Read `/app/paper_cutoff.txt` to anchor the paper's era.
3. Identify the core claims: thesis, specific contributions, methods, baselines, benchmarks, datasets.

### Phase 2: Deep Literature Search

**Invoke `/search-papers` to load the full CLI documentation.** Then search extensively using `/app/search`.

Run at least 3-4 `/app/search batch` calls with `--sort importance`. Cover:
- **Direct competitors**: the paper's exact topic, the thesis in different words, the main claimed contribution
- **Methods and techniques**: the specific technique used, prior work, alternative approaches
- **Baselines and SOTA**: state of the art on each benchmark, recent improvements to each baseline method

Then drill deeper:
- Use `/app/search query <id1> <id2> ... --q "..."` to get summaries of the 6-10 most important papers
- Use `/app/search related <id>` to explore the neighborhood of the most relevant hits

CRITICAL: Before writing that any paper, author, or method is uncited, run:
  grep -i "<lastname>" /app/latex/template.tex
Only after grep returns zero matches may you claim it is absent.

### Phase 3: Novelty Assessment

1. Is the core idea genuinely new, or a recombination of existing ideas?
2. Are there papers the authors should have cited but didn't?
3. Are any novelty claims overclaimed given existing literature?

### Phase 4: Methodology Critique

1. Is the experimental design appropriate for the claims?
2. Are the baselines fair and current?
3. Are there confounding variables not controlled for?

### Phase 5: Scoring

Use ICLR conventions:
- **Rating** (1-10): 1=strong reject, 3=reject, 5=borderline, 6=weak accept, 8=accept, 10=strong accept
- **Soundness** (1-4): 1=poor, 2=fair, 3=good, 4=excellent
- **Presentation** (1-4): 1=poor, 2=fair, 3=good, 4=excellent
- **Contribution** (1-4): 1=poor, 2=fair, 3=good, 4=excellent
- **Confidence** (1-5): 1=low, 3=moderate, 5=expert
- **Decision**: Accept (Rating ≥ 6) or Reject (Rating < 6)


## Output Format — CRITICAL

Your **final message** must be ONLY the review block below — no preamble, no "Here is my review:". Output exactly this structure, including the `\boxed_review{` opener and `}` closer on their own lines:

\boxed_review{
## Summary:

<one-paragraph summary of the paper's contributions and approach>

## Soundness:

<number only, e.g. 3>

## Presentation:

<number only, e.g. 3>

## Contribution:

<number only, e.g. 3>

## Strengths:

<bullet list of strengths>

## Weaknesses:

<bullet list of weaknesses>

## Suggestions:

<bullet list of actionable suggestions for the authors>

## Questions:

<bullet list of questions for the authors>

## Confidence:

<number only, e.g. 4>

## Rating:

<number only, e.g. 6>

## Decision:

<exactly one of: Accept, Reject>
}

**Numeric fields must be plain numbers only** — e.g. `3` or `3.5`, NOT `3 good`. The eval pipeline calls `float()` on the first line of each numeric field; any extra text causes the row to be silently dropped.

## Important Rules

- **Be specific**: Reference sections, theorems, tables, and figures by name as they appear in the paper
- **Be honest**: If the work has fundamental issues, say so clearly
- **Never fabricate**: Only report what you actually found in the files. If you are unsure whether a specific number or claim is in the paper, grep for it first.
- **Verify citations**: Before claiming a paper is uncited, grep the `.tex` file for the author's last name.

# DeepReview Benchmark — Inference Guide for Colleagues

This document gives you everything you need to run inference for our custom model on the DeepReview benchmark and get results comparable to the original paper. Read this fully before writing any code.

---

## Context

We are replicating and extending the **DeepReview benchmark** from the paper:

> Zhu et al. 2025, *"DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process"*
> arXiv: https://arxiv.org/abs/2503.08569

**Official GitHub repo:** https://github.com/zhu-minjun/Researcher
**Official dataset (gated):** https://huggingface.co/datasets/WestlakeNLP/DeepReview-13K

The benchmark evaluates LLM-based reviewer models against real ICLR peer-review scores from OpenReview. Given a research paper as input, a model must produce a structured review. The benchmark then scores the model's predicted ratings against the average of actual human reviewer scores.

---

## What Is In the Zip Folder

```
DeepReview/
├── arXiv-2503.08569v1/         # Full paper source (LaTeX). Read neurips_2024.tex for context.
├── Researcher/                 # Official GitHub repo (cloned)
│   ├── ai_researcher/
│   │   ├── deep_reviewer.py    # DeepReviewer-14B inference class (vLLM-based)
│   │   └── cycle_reviewer.py   # CycleReviewer-8B/70B baseline
│   └── evaluate/
│       └── DeepReview/
│           ├── evalate.py          # OFFICIAL quantitative eval script (Tables 2 & 3)
│           ├── win_rate_evaluate.py # OFFICIAL qualitative LLM-as-Judge script (Table 4)
│           └── sample.json         # 3 worked examples showing exact I/O format
└── replication/                # Our replication workspace
    ├── data/
    │   ├── data/
    │   │   ├── test_2024.csv   # 652 ICLR 2024 test papers (downloaded from HF)
    │   │   └── test_2025.csv   # 634 ICLR 2025 test papers (downloaded from HF)
    │   ├── eval_set_iclr2024.json  # READY TO USE: 652 papers in eval schema
    │   └── eval_set_iclr2025.json  # READY TO USE: 634 papers in eval schema
    ├── predictions/
    │   └── oracle_iclr2024.json    # Sanity-check oracle predictions (verified working)
    ├── results/                    # Put your output markdown tables here
    ├── scripts/
    │   ├── prompts.py              # ALL system prompts (DeepReviewer + generic)
    │   ├── 01_build_eval_set.py    # CSV → eval JSON (already run, outputs are in data/)
    │   ├── 02_run_inference.py     # MAIN SCRIPT: run any model, write predictions JSON
    │   ├── 03_evaluate.py          # Wraps evalate.py for any predictions file
    │   └── 04_smoke_test.py        # Oracle sanity check (already verified)
    ├── BENCHMARK_NOTE.md           # Quick reference card for the benchmark
    └── README.md                   # Replication setup notes
```

**Your job:** implement the custom model backend in `scripts/02_run_inference.py` and run it on `data/eval_set_iclr2024.json` and `data/eval_set_iclr2025.json`.

---

## The Benchmark — How It Works

### Dataset

| File | Papers | Conference | Accept Rate |
|---|---|---|---|
| `test_2024.csv` | **652** | ICLR 2024 | 43.7% |
| `test_2025.csv` | **634** | ICLR 2025 | 31.1% |
| **Total** | **1,286** | | 37.5% |

These are real ICLR submissions from OpenReview. Each paper has 3–5 human peer reviews with numeric scores.

### Ground Truth (how human scores are stored)

Each paper in `eval_set_iclr2024.json` / `eval_set_iclr2025.json` has a `review` field containing the list of human reviewer objects. Each reviewer gives:

- `content.rating` — overall rating as a string like `"6: marginally above the acceptance threshold"` → the eval takes the **first character** as the integer (e.g. `6`)
- `content.soundness` — e.g. `"3 good"` → the eval takes the **first character** as the integer (e.g. `3`)
- `content.presentation` — same format
- `content.contribution` — same format

The eval script **averages** these across all human reviewers to get one ground-truth value per paper. Your model predicts against that average.

### The Four Evaluation Tasks

All four tasks are computed from **one inference pass per paper** — no extra model calls needed.

| Task | What It Measures | Metric |
|---|---|---|
| **Score** | How close your predicted rating is to the mean human rating | MSE, MAE (for Rating, Soundness, Presentation, Contribution) |
| **Decision** | Accept / Reject classification accuracy | Accuracy, F1 (macro) |
| **Ranking** | Whether your scores correctly order papers by quality | Spearman correlation |
| **Selection** | Whether your scores correctly pick the better paper from all possible pairs | Pairwise accuracy |

Tables 2 and 3 in the paper come from `evalate.py` (already wired up as `scripts/03_evaluate.py`).
Table 4 comes from `win_rate_evaluate.py` (LLM-as-Judge using Gemini — optional, run separately).

---

## Eval Set JSON Schema

The files `eval_set_iclr2024.json` and `eval_set_iclr2025.json` are JSON arrays. Each element:

```json
{
  "id": "w7BwaDHppp",
  "title": "",
  "paper_context": "\\title{Geometry-Aware Projective Mapping...}\n\n\\begin{abstract}...",
  "decision": "Accept",
  "review": [
    {
      "id": "iPXv7gp3v3",
      "rating": 5,
      "content": {
        "summary": "...",
        "soundness": "2 fair",
        "presentation": "3 good",
        "contribution": "1 poor",
        "strengths": "...",
        "weaknesses": "...",
        "questions": "...",
        "rating": "5: marginally below the acceptance threshold",
        "confidence": "4: You are confident in your assessment...",
        "suggestions": "..."
      }
    }
  ]
}
```

`paper_context` is the full paper text (LaTeX/Markdown, ~10K–34K characters). This is what you pass to your model.

---

## Required Output Format — CRITICAL

Your model must output a string that contains a `\boxed_review{...}` block. This is the **only thing the eval script reads**. Everything outside the block (thinking, reasoning, preamble) is ignored.

### Exact Format

```
<optional thinking / preamble here>

\boxed_review{
## Summary:

<one paragraph summary of the paper>

## Soundness:

3

## Presentation:

3

## Contribution:

2

## Strengths:

<bullet list of strengths>

## Weaknesses:

<bullet list of weaknesses>

## Suggestions:

<bullet list of suggestions>

## Questions:

<questions for the authors>

## Confidence:

3

## Rating:

5.5

## Decision:

Reject
}
```

### Parsing Rules (from `evalate.py`)

The eval script parses with:
```python
pred_context = output.split(r'\boxed_review{')[-1].split('\n}')[0]
```
Then splits on `## ` and matches `<Key>:\n\n<value>`.

**Rules:**
1. The block must start with `\boxed_review{` (backslash, no space)
2. The block must end with `\n}` (newline then closing brace)
3. Each section header must be `## <Key>:\n\n` (two hashes, space, key, colon, TWO newlines)
4. Numeric fields (Rating, Soundness, Presentation, Contribution, Confidence) must be **plain numbers** on the first line — `float(value.split('\n')[0])` is called on them
   - ✅ `6` `6.5` `3.0`
   - ❌ `6/10` `6 out of 10` `3 good` `~6` — these will crash the parser and the row is **silently dropped**
5. Decision must contain the word `accept` or `reject` (case-insensitive). Anything that does not contain `accept` is treated as Reject
6. **If Rating is empty string, the entire row is skipped silently.** Always verify that your result count matches your input count

### Score Ranges

| Field | Range | Convention |
|---|---|---|
| Rating | 1–10 | ICLR scale. ~5 = borderline, ≥6 = above threshold |
| Soundness | 1–4 | 1=poor, 2=fair, 3=good, 4=excellent |
| Presentation | 1–4 | Same scale |
| Contribution | 1–4 | Same scale |
| Confidence | 1–5 | How confident the reviewer is |

---

## Multi-Reviewer Output — How to Handle It

If your model (or any baseline) generates multiple simulated reviewer sections internally, the eval script only reads the **final `\boxed_review{}` block** — which must be a single aggregated meta-review.

This is exactly what DeepReviewer does: it internally simulates 4 reviewers inside a `\boxed_simreviewers{}` block, then writes one synthesized meta-review in `\boxed_review{}` with aggregated scores. The eval script only ever reads the `\boxed_review{}` block.

So the pattern is:
1. Generate N simulated reviews internally (optional)
2. Aggregate into a single final meta-review
3. Output that as `\boxed_review{}` with one set of averaged scores

If your model generates 3 reviews but no meta-review, average the 3 numeric scores yourself and synthesize the text fields, then wrap in one `\boxed_review{}` block.

**Do NOT:**
- Output multiple `\boxed_review{}` blocks (the eval takes the last one, which may be incomplete)
- Pick one random reviewer out of several (adds noise, unfair comparison)

---

## System Prompt To Use

Use `GENERIC_REVIEWER_SYSTEM_PROMPT` from `replication/scripts/prompts.py`. This instructs any LLM to produce the exact `\boxed_review{}` format. Here it is in full:

```
You are an expert academic peer reviewer evaluating a research paper.

Read the paper carefully and produce a comprehensive review. Your final output MUST end
with a \boxed_review{...} block containing exactly the following sections, in this order,
each prefixed by `## <Name>:` followed by a blank line and the value:

## Summary:
<one-paragraph summary of the paper>

## Soundness:
<a single number from 1 to 4 indicating soundness>

## Presentation:
<a single number from 1 to 4 indicating presentation>

## Contribution:
<a single number from 1 to 4 indicating contribution>

## Strengths:
<bullet list of strengths>

## Weaknesses:
<bullet list of weaknesses>

## Suggestions:
<bullet list of actionable suggestions>

## Questions:
<bullet list of questions for the authors>

## Confidence:
<a single number from 1 to 5 indicating your confidence>

## Rating:
<a single number from 1 to 10 indicating overall paper rating>

## Decision:
<exactly one of: Accept, Reject>

Wrap the entire review block as:

\boxed_review{
## Summary:

...

## Decision:

Accept
}

Use the ICLR conventions: Rating is on a 1-10 scale, Soundness/Presentation/Contribution
on a 1-4 scale, Confidence on a 1-5 scale. Numbers must be parseable (e.g. `3` or `3.5`,
not `3 good`).
```

The user message is simply the full `paper_context` string.

---

## Inference Settings (to match the paper)

| Parameter | Value |
|---|---|
| Temperature | 0.4 |
| Max input tokens | 100,000 |
| Max output tokens | 16,384 |
| top_p | 0.95 |

---

## Step-by-Step: What You Need to Do

### Step 1 — Install dependencies

```bash
cd DeepReview/replication
python3 -m venv .venv
.venv/bin/pip install torch scipy scikit-learn pandas openai tqdm
```

### Step 2 — Wire up your model in `scripts/02_run_inference.py`

Open `scripts/02_run_inference.py`. Find `make_agentreviewer_backend()`:

```python
def make_agentreviewer_backend() -> Callable[[str], str]:
    def _gen(paper_context: str) -> str:
        raise NotImplementedError(
            "Wire up your AgentReviewer here..."
        )
    return _gen
```

Replace the `raise NotImplementedError` with a call to your model. Your function receives `paper_context` (the full paper text as a string) and must return a string containing a `\boxed_review{...}` block.

**Example if your model is an OpenAI-compatible API:**
```python
def make_agentreviewer_backend() -> Callable[[str], str]:
    from prompts import generic_messages
    from openai import OpenAI
    client = OpenAI(api_key="YOUR_KEY", base_url="YOUR_BASE_URL")

    def _gen(paper_context: str) -> str:
        resp = client.chat.completions.create(
            model="your-model-name",
            messages=generic_messages(paper_context),
            temperature=0.4,
            max_tokens=16384,
        )
        return resp.choices[0].message.content

    return _gen
```

**Example if your model is a local Python function:**
```python
def make_agentreviewer_backend() -> Callable[[str], str]:
    from your_module import AgentReviewer  # your import
    from prompts import GENERIC_REVIEWER_SYSTEM_PROMPT
    agent = AgentReviewer(...)  # initialize

    def _gen(paper_context: str) -> str:
        return agent.review(
            system_prompt=GENERIC_REVIEWER_SYSTEM_PROMPT,
            paper=paper_context
        )

    return _gen
```

**Important:** If your model natively generates multiple reviewer sections, add an aggregation step inside `_gen` to produce one final `\boxed_review{}` block before returning.

### Step 3 — Run inference on ICLR 2024

```bash
cd DeepReview/replication
.venv/bin/python scripts/02_run_inference.py \
    --eval-set data/eval_set_iclr2024.json \
    --out predictions/myagent_iclr2024.json \
    --backend agentreviewer \
    --mode-key pred_standard_mode
```

This checkpoints every 10 papers. If it crashes, rerun the same command — it will resume from where it left off.

**To test on a few papers first:**
```bash
.venv/bin/python scripts/02_run_inference.py \
    --eval-set data/eval_set_iclr2024.json \
    --out predictions/myagent_iclr2024_test5.json \
    --backend agentreviewer \
    --mode-key pred_standard_mode \
    --limit 5
```

### Step 4 — Run evaluation

```bash
.venv/bin/python scripts/03_evaluate.py \
    --predictions predictions/myagent_iclr2024.json \
    --mode standard \
    --out-md results/myagent_iclr2024.md
```

This prints a markdown table and writes it to `results/`. The table contains all metrics for Tables 2 and 3 of the paper.

**Important:** Check that the `n=...` count printed by the eval matches 652. If it's lower, some rows had parsing errors and were silently dropped. Debug by looking at the raw output strings.

### Step 5 — Repeat for ICLR 2025

```bash
.venv/bin/python scripts/02_run_inference.py \
    --eval-set data/eval_set_iclr2025.json \
    --out predictions/myagent_iclr2025.json \
    --backend agentreviewer \
    --mode-key pred_standard_mode

.venv/bin/python scripts/03_evaluate.py \
    --predictions predictions/myagent_iclr2025.json \
    --mode standard \
    --out-md results/myagent_iclr2025.md
```

---

## Expected Output from `03_evaluate.py`

The script prints a table like this (numbers below are the paper's DeepReviewer-14B best results for reference):

```
| Metric                  | Value  |
|-------------------------|--------|
| Rating MSE              | 1.3137 |
| Rating MAE              | 0.9102 |
| Soundness MSE           | 0.1578 |
| Soundness MAE           | 0.3029 |
| Presentation MSE        | 0.1896 |
| Presentation MAE        | 0.3291 |
| Contribution MSE        | 0.2173 |
| Contribution MAE        | 0.3680 |
| Rating Spearman         | 0.3559 |
| Soundness Spearman      | 0.3204 |
| Presentation Spearman   | 0.3784 |
| Contribution Spearman   | 0.3335 |
| Decision Accuracy       | 0.6406 |
| Decision F1             | 0.6307 |
| Pairwise Rating Acc     | 0.6242 |
| Pairwise Soundness Acc  | 0.6175 |
| Pairwise Presentation Acc | 0.6353 |
| Pairwise Contribution Acc | 0.6208 |
```

These are **Table 2 + Table 3 combined** in one run. Lower MSE/MAE = better. Higher Spearman/Accuracy = better.

---

## Baseline Numbers for Comparison (from the paper, ICLR 2024)

| Model | R. MSE↓ | R. MAE↓ | D. Acc↑ | D. F1↑ | R. Spearman↑ | Pair. Acc↑ |
|---|---|---|---|---|---|---|
| AgentReview Claude-3.5-Sonnet | 2.8878 | 1.2715 | 0.4333 | 0.3937 | 0.1564 | 0.5526 |
| AgentReview Gemini-2.0-Flash | 3.1943 | 1.3418 | 0.4400 | 0.4318 | -0.0252 | 0.5044 |
| AgentReview DeepSeek-V3 | 1.9479 | 1.0735 | 0.4105 | 0.3403 | 0.3542 | 0.6096 |
| AI Scientist GPT-o1 | 4.3414 | 1.7294 | 0.4500 | 0.4424 | 0.2621 | 0.5881 |
| AI Scientist DeepSeek-R1 | 4.1648 | 1.6526 | 0.5248 | 0.4988 | 0.3256 | 0.6206 |
| CycleReviewer-8B | 2.8911 | 1.2371 | 0.6353 | 0.5528 | 0.2801 | 0.5993 |
| CycleReviewer-70B | 2.4870 | 1.2514 | 0.6304 | 0.5696 | 0.3356 | 0.6160 |
| **DeepReviewer-14B** | **1.3137** | **0.9102** | **0.6406** | **0.6307** | **0.3559** | **0.6242** |

---

## Sanity Check: Verify the Pipeline First

Before running your model on all 1,286 papers, verify the eval pipeline works end-to-end:

```bash
# Run oracle (predicts perfect scores = mean of human reviewers)
.venv/bin/python scripts/04_smoke_test.py

# Evaluate oracle — should get MSE=0, Spearman=1, Accuracy=1
.venv/bin/python scripts/03_evaluate.py \
    --predictions predictions/oracle_iclr2024.json \
    --mode standard
```

This is already verified — the oracle gets perfect scores on 652 papers. If it fails on your machine, there's a setup issue.

---

## Common Mistakes to Avoid

1. **Silent row drops** — if your model outputs `"3 good"` instead of `"3"` for Soundness, that row is silently dropped. Always check the `n=...` count in the eval output equals your input size.

2. **Empty Rating field** — if the `## Rating:` section is missing or empty, the row is skipped. Ensure your prompt forces the model to always output a numeric rating.

3. **Wrong section header format** — the parser looks for `## Key:\n\n` (exactly two newlines after the colon). `## Key:\n` (one newline) will fail to parse.

4. **Multiple `\boxed_review{}` blocks** — the parser takes `split(r'\boxed_review{')[-1]`, i.e. the last block. If you accidentally emit two blocks, only the last one is used (which may be a fragment). Emit exactly one.

5. **The `year` column in the raw CSVs says `2025` for all rows** — this is a known labeling artifact. `test_2024.csv` is the ICLR 2024 split and `test_2025.csv` is the ICLR 2025 split, verified by their accept rates (43.7% and 31.1% respectively, matching the paper exactly).

6. **Do not use `eval_set_*.json` `title` field** — it is always empty string `""`. The paper title is embedded in `paper_context` as `\title{...}` in the LaTeX.

---

## Files You Should NOT Modify

- `Researcher/evaluate/DeepReview/evalate.py` — official eval script; modifying it breaks reproducibility
- `Researcher/evaluate/DeepReview/win_rate_evaluate.py` — official judge script
- `data/eval_set_iclr2024.json` and `data/eval_set_iclr2025.json` — the fixed test set

**The only file you need to edit is `scripts/02_run_inference.py`** — specifically the `make_agentreviewer_backend()` function.

---

## Questions?

All decisions above were made after careful reading of:
- The full paper (`arXiv-2503.08569v1/neurips_2024.tex`)
- The official eval scripts (`Researcher/evaluate/DeepReview/evalate.py` and `win_rate_evaluate.py`)
- The official sample outputs (`Researcher/evaluate/DeepReview/sample.json` — 3 fully worked examples with real DeepReviewer-14B outputs)
- The raw test CSVs (downloaded and inspected, verified against paper statistics)

The oracle smoke test (predicting the exact mean of human reviewers) achieves MSE=0, Spearman=1.0, and Decision Accuracy=1.0 on all 652 ICLR 2024 papers — confirming the eval pipeline is correctly wired end-to-end.

# DeepReview-Bench — Quick Reference

**Source paper:** Zhu et al. 2025, "DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process" (arXiv:2503.08569)
**Dataset:** `WestlakeNLP/DeepReview-13K` on HuggingFace (gated)
**Eval scripts:** `zhu-minjun/Researcher` GitHub repo, under `evaluate/DeepReview/`

---

## What It Is

A benchmark for evaluating LLM-based paper-review systems against real ICLR human-reviewer scores. Given a research paper, the model under test must produce a structured review (summary, scores, strengths, weaknesses, decision). The benchmark compares the model's predictions against the average of actual human reviewer scores from OpenReview.

---

## Dataset

| Split | Papers | Source | Accept Rate |
|---|---|---|---|
| `test_2024.csv` | **652** | ICLR 2024 | 43.7% |
| `test_2025.csv` | **634** | ICLR 2025 | 31.1% |
| **Total** | **1,286** | | 37.5% |

- Sampled randomly as 10% of the full 13,378-paper DeepReview-13K dataset
- Average paper length: ~10,500 tokens
- Average human rating per paper: ~5.3 (on a 1-10 scale)
- Each paper has 3-5 human reviews

---

## Per-Paper Ground Truth

Each paper carries:
- **Overall rating** ∈ [1, 10] — averaged across human reviewers
- **Soundness** ∈ [1, 4] — averaged across human reviewers
- **Presentation** ∈ [1, 4] — averaged across human reviewers
- **Contribution** ∈ [1, 4] — averaged across human reviewers
- **Confidence** ∈ [1, 5] — per reviewer (not directly evaluated)
- **Decision** ∈ {Accept, Reject} — final committee decision

---

## Required Model Output

The model must output a `\boxed_review{...}` block containing 11 sections:

| Section | Type | Range |
|---|---|---|
| Summary | text | — |
| Rating | float | 1-10 |
| Soundness | float | 1-4 |
| Presentation | float | 1-4 |
| Contribution | float | 1-4 |
| Strengths | text | — |
| Weaknesses | text | — |
| Suggestions | text | — |
| Questions | text | — |
| Confidence | float | 1-5 |
| Decision | enum | Accept / Reject |

Numeric fields must be plain numbers (e.g. `3` or `3.5`) — no extra text.

---

## Evaluation Tasks

### Quantitative (`evalate.py`) → Tables 2 & 3 of the paper

| Task | What it measures | Metrics |
|---|---|---|
| **Score** | Accuracy of independent paper assessment | MSE, MAE (for Rating, Soundness, Presentation, Contribution) |
| **Decision** | Accept/Reject classification | Accuracy, F1 (macro) |
| **Ranking** | Ability to rank papers by quality | Spearman correlation (for each of 4 dims) |
| **Selection** | Pairwise preference accuracy | Accuracy over all C(N, 2) pairs (for each of 4 dims) |

### Qualitative (`win_rate_evaluate.py`) → Table 4

LLM-as-Judge using **Gemini-2.0-Flash-Thinking** as the judge. Pairwise comparison of generated reviews across 5 aspects:
1. Constructive Value
2. Analytical Depth
3. Plausibility *(paper) / Communication Clarity (code — uses this)*
4. Technical Accuracy
5. Overall Judgment

Output: Win / Lose / Tie rates per aspect.

---

## Inference Settings (from paper §5.1)

| Parameter | Value |
|---|---|
| Temperature | 0.4 |
| Max input tokens | 100,000 |
| Max output tokens | 16,384 |
| top_p | 0.95 (from code, not paper) |

## Judge Settings (from `win_rate_evaluate.py`)

| Parameter | Value |
|---|---|
| Judge model | `gemini-2.0-flash-thinking-exp-01-21` |
| Temperature | 0 |
| Max output tokens | 8,192 |

---

## What's NOT in the Benchmark

- **Figure 5 (adversarial robustness)** — separate experiment, no script
- **Figure 6 (test-time scaling)** — analysis of DeepReviewer's own modes; not a benchmark task
- **Table 1** — descriptive dataset statistics only

# Experiment Log — LLM Agent as Peer Reviewer

**Task:** Can an LLM agent, given a paper and literature search access, write a review that matches human peer reviewers?  
**Dataset:** 115 arXiv papers with 2–4 human reviews from OpenReview (ICLR/NeurIPS)  
**Metric:** Mean reward ∈ [0,1] from Gemini judge — mean(Issue Overlap, Fabrication, Calibration)  

---

## System Overview

### Agentic Pipeline
Claude Code agent running inside E2B sandbox, with:
- Full paper body (OCR markdown or LaTeX→pandoc)
- `/app/search` CLI wrapping a 928K arXiv paper index (temporal cutoff enforced)
- Task instruction to write a structured peer review

The LLM backend is swapped via a proxy (Anthropic API format → provider API).

### Baselines
- **Stanford Reviewer** — external agentic pipeline
- **DeepReviewer-v2** — external agentic pipeline 

### Judge (Gemini 3.1 Pro)
6 criteria scored, but final reward uses only the 3 discriminative ones — comprehension, substance, and insight saturate at ≥0.97 across all systems and add no signal:

| Criterion | Used in reward | What it measures |
|-----------|---------------|-----------------|
| Comprehension | No (saturated) | Paper understanding |
| Substance | No (saturated) | Specificity of technical engagement |
| Insight | No (saturated) | Grounded, non-obvious observations |
| **Issue Overlap** | **Yes** | Coverage of human reviewer points |
| **Fabrication** | **Yes** | No invented facts (adversarial) |
| **Calibration** | **Yes** | Accept/reject alignment with human consensus |

**Reward = mean(Issue Overlap, Fabrication, Calibration)**

---

## Results

### Table 1: Model & Pipeline Comparison

Reward = mean(Issue Overlap, Fabrication, Calibration)

| System | Model | Input | N papers | Mean Reward |
|--------|-------|-------|----------|-------------|
| Stanford Reviewer | — | PDF | 115 | **0.777** |
| **Our pipeline** | DeepSeek V4 Flash | OCR Markdown | 115 | **0.670** |
| Our pipeline | MiniMax M2.7 | OCR Markdown | 78 | 0.622 |
| Our pipeline | MiniMax M2.7 | LaTeX→pandoc | 100 | 0.575 |
| DeepReviewer-v2 | — | PDF | 55 | 0.530 |

### Table 2: Per-Criterion Breakdown

Comprehension, Substance, and Insight score ≥0.97 across all systems — confirming they are not discriminative and justifying their exclusion from the final reward.

| Criterion | Stanford | DeepSeek Flash | MiniMax OCR | MiniMax LaTeX | DeepReviewer-v2 |
|-----------|----------|---------------|-------------|---------------|----------------|
| Comprehension *(excluded)* | 1.000 | 1.000 | 0.974 | 1.000 | 0.979 |
| Substance *(excluded)* | 1.000 | 1.000 | 0.974 | 1.000 | 0.979 |
| Insight *(excluded)* | 1.000 | 1.000 | 0.974 | 0.975 | 0.979 |
| **Issue Overlap** | 0.601 | 0.557 | 0.589 | 0.621 | 0.539 |
| **Fabrication** | **0.848** | 0.618 | 0.500 | 0.365 | 0.312 |
| **Calibration** | 0.908 | **0.851** | 0.589 | 0.403 | 0.737 |

### Table 3: Input Format Ablation (MiniMax M2.7, same harness)

| Input Format | Mean Reward | Δ |
|-------------|-------------|---|
| LaTeX→pandoc | 0.575 | baseline |
| OCR Markdown (olmOCR) | 0.622 | +0.047 |

---

## Key Observations

1. **Gap to Stanford is 0.107** (0.670 vs 0.777), driven almost entirely by Fabrication (0.618 vs 0.848). DeepSeek V4 Flash with OCR markdown is our best run, all 115 papers scored.

2. **Comprehension, Substance, Insight are saturated at ~1.0** across all systems — all models understand the papers deeply. These criteria no longer discriminate.

3. **Issue Overlap is a bottleneck** (0.557 vs 0.601). Our agent identifies valid issues but misses specific concerns the human reviewers raised — likely because it searches literature broadly instead of focusing on the exact critique dimensions humans prioritize.

4. **Fabrication is our biggest weakness vs Stanford** (0.618 vs 0.848). The agent confabulates specific pointers — figure numbers, section numbers, exact values — while getting the content directionally right.

5. **Calibration is our strongest dimension** (0.851 vs Stanford's 0.908) and the gap here is small. The agentic pipeline with literature search gives a reasonably accurate accept/reject verdict.

6. **OCR markdown outperforms LaTeX→pandoc** (+0.047 on MiniMax). LaTeX conversion loses tables and garbles math notation; this directly hurts fabrication and calibration scores.

7. **Fabrication is our weakest dimension across all runs** (0.618 vs Stanford's 0.848). 

## Rating Alignment with Human Reviewers

Each AI-generated review outputs an Overall score (1–10) and a recommendation. We compare these against the mean human reviewer rating for the same paper.

**Cohen's κ**: binary agreement on accept/reject (threshold: score ≥ 6 = accept)

| System | Pearson r | Spearman ρ | MAE | Cohen's κ |
|--------|-----------|------------|-----|-----------|
| DeepSeek V4 Flash (agentic, OCR) | **0.623** | **0.636** | 0.730 | **0.455** |
| Stanford Reviewer | 0.611 | 0.590 | 0.746 | 0.354 |
| MiniMax M2.7 (agentic, OCR) | 0.473 | 0.462 | 1.113 | 0.347 |
| MiniMax M2.7 (agentic, LaTeX) | 0.455 | 0.479 | 1.261 | 0.225 |

**Key findings:**
- DeepSeek Flash achieves the best rating alignment (κ=0.455, r=0.623) — approaching moderate-to-substantial agreement
- OCR markdown input improves alignment over LaTeX (κ: 0.347 vs 0.225 on same model) — better paper understanding leads to better-calibrated scores
- Stanford static pipeline (κ=0.354) is outperformed by our best agentic run despite its higher reward score — the reward metric and rating alignment measure different things
- κ < 0.60 across all systems suggests this remains an open challenge

---

## DeepReview Benchmark (External)

**Benchmark:** [WestlakeNLP/DeepReview-13K](https://huggingface.co/datasets/WestlakeNLP/DeepReview-13K) — ICLR papers with human reviewer scores from OpenReview.

**Setup:** Our standard Harbor/E2B harness run on DeepReview papers as-is (no instruction changes). Agent writes in native format; `05_run_harness.py` post-processes to `\boxed_review{}` format. Scored with the official `evalate.py` from [zhu-minjun/Researcher](https://github.com/zhu-minjun/Researcher/tree/main/evaluate/DeepReview).

**Metrics:** Compared against mean human reviewer scores per paper.
- **MSE/MAE** — absolute score error vs human mean
- **Spearman** — ranking correlation with human ratings
- **Decision Accuracy** — Accept/Reject match with committee decision
- **Pairwise Acc** — correct ordering for all C(N,2) paper pairs

---

### ICLR 2024 (652 papers, DeepSeek V4 Pro, K=1)

**Papers scored: 631/652**
| Metric | Our Score |
|--------|-----------|
| Rating MSE | 1.4584 |
| Rating MAE | 0.9575 |
| Soundness MSE | 0.4249 |
| Soundness MAE | 0.5340 |
| Presentation MSE | 0.3196 |
| Presentation MAE | 0.4284 |
| Contribution MSE | 0.3442 |
| Contribution MAE | 0.4719 |
| Rating Spearman | 0.4767 |
| Soundness Spearman | 0.4010 |
| Presentation Spearman | 0.5191 |
| Contribution Spearman | 0.3425 |
| Decision Accuracy | 0.6498 |
| Decision F1 | 0.5950 |
| Pairwise Rating Acc | 0.6586 |
| Pairwise Soundness Acc | 0.6464 |
| Pairwise Presentation Acc | 0.6825 |
| Pairwise Contribution Acc | 0.6106 |

---

### ICLR 2025 (634 papers, DeepSeek V4 Flash, K=1)

**Papers scored: 564/634**

| Metric | Our Score |
|--------|-----------|
| Rating MSE | 0.9516 |
| Rating MAE | 0.7812 |
| Soundness MSE | 0.3199 |
| Soundness MAE | 0.4511 |
| Presentation MSE | 0.2296 |
| Presentation MAE | 0.3711 |
| Contribution MSE | 0.2221 |
| Contribution MAE | 0.3678 |
| Rating Spearman | 0.5709 |
| Soundness Spearman | 0.4978 |
| Presentation Spearman | 0.6053 |
| Contribution Spearman | 0.5502 |
| Decision Accuracy | 0.7620 |
| Decision F1 | 0.7107 |
| Pairwise Rating Acc | 0.6896 |
| Pairwise Soundness Acc | 0.6672 |
| Pairwise Presentation Acc | 0.7154 |
| Pairwise Contribution Acc | 0.6915 |

Flash outperforms Pro across every metric. 70 papers failed to match (634→564); these are papers where the DeepReview dataset had no matching entry after filtering.

---

### Win-Rate vs DeepReviewer-14B (LLM Judge, Pairwise)

**Judge:** Gemini 3.1 Pro — same SYSTEM_PROMPT as official `win_rate_evaluate.py` from [zhu-minjun/Researcher](https://github.com/zhu-minjun/Researcher/tree/main/evaluate/DeepReview). Results in `/root/pass_at_k/win_rate_*.jsonl`.

**Categories judged:** Overall Judgment, Technical Accuracy, Constructive Value, Analytical Depth, Communication Clarity.

#### ICLR 2024 — DeepSeek V4 Pro vs DeepReviewer-14B (631 papers)

| Category | Win | Tie | Lose |
|----------|-----|-----|------|
| Overall Judgment | **94.8%** | 2.4% | 2.9% |
| Technical Accuracy | **93.8%** | 4.0% | 2.2% |
| Constructive Value | **80.2%** | 7.6% | 12.2% |
| Analytical Depth | **92.4%** | 7.1% | 0.5% |
| Communication Clarity | **95.6%** | 3.5% | 1.0% |

#### ICLR 2025 — DeepSeek V4 Flash vs DeepReviewer-14B (564 papers)

| Category | Win | Tie | Lose |
|----------|-----|-----|------|
| Overall Judgment | **94.0%** | 2.5% | 3.5% |
| Technical Accuracy | **91.5%** | 5.5% | 3.0% |
| Constructive Value | **78.0%** | 9.9% | 12.1% |
| Analytical Depth | **90.2%** | 8.3% | 1.4% |
| Communication Clarity | **95.2%** | 3.2% | 1.6% |

**Key finding:** Our agentic harness dominates DeepReviewer-14B across both years and both models (~90–95% win rate on most categories). The only relative weakness is Constructive Value (~80% win, ~12% lose), where DeepReviewer-14B occasionally provides more actionable feedback.

---

### Win-Rate vs Stanford Reviewer (LLM Judge, Pairwise, Harbor 115-paper set)

These runs compare our harness output against Stanford Reviewer on the same 115-paper Harbor set (not DeepReview dataset). Results in `/root/pass_at_k/win_rate_stanford_vs_*.jsonl`.

#### DeepSeek V4 Flash vs Stanford Reviewer (115 papers)

| Category | Win | Tie | Lose |
|----------|-----|-----|------|
| Overall Judgment | 54.8% | 26.1% | 19.1% |
| Technical Accuracy | 48.7% | 38.3% | 13.0% |
| Constructive Value | 20.0% | 59.1% | 20.9% |
| Analytical Depth | 43.5% | 47.0% | 9.6% |
| Communication Clarity | 30.4% | 69.6% | 0.0% |

#### DeepSeek V4 Pro vs Stanford Reviewer (115 papers)

| Category | Win | Tie | Lose |
|----------|-----|-----|------|
| Overall Judgment | 58.3% | 22.6% | 19.1% |
| Technical Accuracy | 47.8% | 35.7% | 16.5% |
| Constructive Value | 18.3% | 61.7% | 20.0% |
| Analytical Depth | 46.1% | 44.3% | 9.6% |
| Communication Clarity | 26.1% | 73.9% | 0.0% |

**Key finding:** Against Stanford Reviewer the picture is much closer — ~50% win on Judgment/Technical, but Communication Clarity and Constructive Value are mostly ties or losses. Stanford writes more polished, readable reviews; our agent wins on identifying technical issues but loses on presentation quality.

---

## Open Issues

| Issue | Impact | Status |
|-------|--------|--------|
| Issue overlap gap vs Stanford (0.557 vs 0.601) | −0.022 reward | Active |
| Fabrication gap vs Stanford (0.618 vs 0.848) | −0.077 reward | Active |
| **Cohen's κ requires manual scoring** | Paper not publishable without it | **TODO** |
| DeepReviewer-v2 scoring | 55/115 papers scored, mean 0.530 | Done (partial) |
| Pass@K (K=4) not run | Only K=1 so far | TODO |

---

## Dates

- **2026-04-21:** First agentic run (MiniMax, LaTeX, 100 papers) → 0.575
- **2026-04-22:** OCR pipeline added; MiniMax + OCR → 0.638
- **2026-04-23–25:** Search skill CLI wrapper added; temporal cutoff enforcement
- **2026-04-26:** Agentic judge introduced; Stanford rejudged
- **2026-04-27:** Judge redesigned (single-shot structured output); 3-call split for overlap/fabrication/rest
- **2026-04-27–28:** DeepSeek V4 Flash run (115 papers, K=1) → **0.670**
- **2026-04-29–30:** Rejudged 7 failed DeepSeek papers (token limit 16K→32K + ThinkingLevel.MEDIUM); redesigned Calibration criterion (Decision Alignment + Internal Consistency); rejudged all 115 DeepSeek + 115 Stanford papers with new judge → current numbers above
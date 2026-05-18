# Experiment Log — LLM Agent as Peer Reviewer

**Task:** Can an LLM agent, given a paper and literature search access, write a review that matches human peer reviewers?  
**Dataset:** 115 arXiv papers with 2–4 human reviews from OpenReview (ICLR/NeurIPS)  
**Metric:** Mean reward ∈ \[0,1\] from Gemini judge — mean(Issue Overlap, Fabrication, Calibration)

---

## **System Overview**

### **Agentic Pipeline**

Claude Code agent running inside E2B sandbox, with:

- Full paper body (OCR markdown or LaTeX→pandoc)  
- `/app/search` CLI wrapping a 928K arXiv paper index (temporal cutoff enforced)  
- Task instruction to write a structured peer review

The LLM backend is swapped via a proxy (Anthropic API format → provider API).

### **Baselines**

- **Stanford Reviewer** — external agentic pipeline  
- **DeepReviewer-v2** — external agentic pipeline

### **Judge (Gemini 3.1 Pro)**

6 criteria scored, but final reward uses only the 3 discriminative ones — comprehension, substance, and insight saturate at ≥0.97 across all systems and add no signal:

| Criterion | Used in reward | What it measures |
| :---- | :---- | :---- |
| Comprehension | No (saturated) | Paper understanding |
| Substance | No (saturated) | Specificity of technical engagement |
| Insight | No (saturated) | Grounded, non-obvious observations |
| **Issue Overlap** | **Yes** | Coverage of human reviewer points |
| **Fabrication** | **Yes** | No invented facts (adversarial) |
| **Calibration** | **Yes** | Accept/reject alignment with human consensus |

---

## **Results**

### **Table 1: Model & Pipeline Comparison**

Reward \= mean(Issue Overlap, Fabrication, Calibration)

| System | Model | Input | Mean Reward |
| :---- | :---- | :---- | :---- |
| Stanford Reviewer | — | PDF | **0.777** |
| **Our pipeline** | DeepSeek V4 Flash | OCR Markdown | **0.690** |
| Our pipeline | MiniMax M2.7 | OCR Markdown | 0.575 |
| Our pipeline | MiniMax M2.7 | LaTeX→pandoc | 0.463 |
| DeepReviewer-v2 | Stepfun | PDF | 0.530 |

### **Table 2: Per-Criterion Breakdown**

Comprehension, Substance, and Insight score ≥0.97 across all systems — confirming they are not discriminative and justifying their exclusion from the final reward.

| Criterion | Stanford | DeepSeek Flash | MiniMax OCR | MiniMax LaTeX | DeepReviewer-v2 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Comprehension *(excluded)* | 1.000 | 1.000 | 0.974 | 1.000 | 0.979 |
| Substance *(excluded)* | 1.000 | 1.000 | 0.974 | 1.000 | 0.979 |
| Insight *(excluded)* | 1.000 | 1.000 | 0.974 | 0.975 | 0.979 |
| **Issue Overlap** | 0.601 | 0.557 | 0.589 | 0.621 | 0.539 |
| **Fabrication** | **0.848** | 0.618 | 0.500 | 0.365 | 0.312 |
| **Calibration** | 0.908 | **0.851** | 0.589 | 0.403 | 0.737 |

### **Table 3: Input Format Ablation (MiniMax M2.7, same harness)**

| Input Format | Mean Reward | Δ |
| :---- | :---- | :---- |
| LaTeX→pandoc | 0.463 | baseline |
| OCR Markdown (olmOCR) | 0.575 | \+0.112 |

---

## **Key Observations**

1. **Gap to Stanford is 0.107** (0.670 vs 0.777), driven almost entirely by Fabrication (0.618 vs 0.848). DeepSeek V4 Flash with OCR markdown is our best run, all 115 papers scored.  
     
2. **Comprehension, Substance, Insight are saturated at \~1.0** across all systems — all models understand the papers deeply. These criteria no longer discriminate.  
     
3. **Issue Overlap is a bottleneck** (0.557 vs 0.601). Our agent identifies valid issues but misses specific concerns the human reviewers raised, maybe because it searches literature broadly instead of focusing on the exact critique dimensions humans prioritize.  
     
4. **Fabrication is our biggest weakness vs Stanford** (0.618 vs 0.848). The agent confabulates specific pointers — figure numbers, section numbers, exact values — while getting the content directionally right.  
     
5. **Calibration is our strongest dimension** (0.851 vs Stanford's 0.908) and the gap here is small. The agentic pipeline with literature search gives a reasonably accurate accept/reject verdict.  
     
6. **OCR markdown outperforms LaTeX→pandoc** (+0.047 on MiniMax). LaTeX conversion loses tables and garbles math notation; this directly hurts fabrication and calibration scores.

7. **Fabrication is our weakest dimension across all runs** (0.618 vs Stanford's 0.848).  
   

---

## **Rating Alignment with Human Reviewers**

Each AI-generated review outputs an Overall score (1–10) and a recommendation. We compare these against the mean human reviewer rating for the same paper.

**Cohen's κ**: binary agreement on accept/reject (threshold: score ≥ 6 \= accept)

| System | Pearson r | Spearman ρ | MAE | Cohen's κ |
| :---- | :---- | :---- | :---- | :---- |
| Our Harness \+ DeepSeek V4 Flash (OCR) | **0.623** | **0.636** | 0.730 | **0.455** |
| Our Harness \+ Deepseek V4 Pro (OCR) |  |  |  |  |
| Stanford Reviewer | 0.611 | 0.590 | 0.746 | 0.354 |
| MiniMax M2.7 (agentic, OCR) | 0.473 | 0.462 | 1.113 | 0.347 |
| MiniMax M2.7 (agentic, LaTeX) | 0.455 | 0.479 | 1.261 | 0.225 |

**Key findings:**

- DeepSeek Flash achieves the best rating alignment (κ=0.455, r=0.623) — approaching moderate-to-substantial agreement  
- OCR markdown input improves alignment over LaTeX (κ: 0.347 vs 0.225 on same model) — better paper understanding leads to better-calibrated scores  
- Stanford static pipeline (κ=0.354) is outperformed by our best agentic run despite its higher reward score — the reward metric and rating alignment measure different things  
- κ \< 0.60 across all systems suggests this remains an open challenge

---

## **Open Issues**

| Issue | Impact | Status |
| :---- | :---- | :---- |
| Issue overlap gap vs Stanford (0.557 vs 0.601) | −0.022 reward | Active |
| Fabrication gap vs Stanford (0.618 vs 0.848) | −0.077 reward | Active |
| **Cohen's κ requires manual scoring** | Paper not publishable without it | **TODO** |
| DeepReviewer-v2 scoring | 55/115 papers scored, mean 0.530 | Done (partial) |
| Pass@K (K=4) not run | Only K=1 so far | TODO |

---

## **Dates**

- **2026-04-21:** First agentic run (MiniMax, LaTeX, 100 papers) → 0.575  
- **2026-04-22:** OCR pipeline added; MiniMax \+ OCR → 0.638  
- **2026-04-23–25:** Search skill CLI wrapper added; temporal cutoff enforcement  
- **2026-04-26:** Agentic judge introduced; Stanford rejudged  
- **2026-04-27:** Judge redesigned (single-shot structured output); 3-call split for overlap/fabrication/rest  
- **2026-04-27–28:** DeepSeek V4 Flash run (115 papers, K=1) → **0.670**  
- **2026-04-29–30:** Rejudged 7 failed DeepSeek papers (token limit 16K→32K \+ ThinkingLevel.MEDIUM); redesigned Calibration criterion (Decision Alignment \+ Internal Consistency); rejudged all 115 DeepSeek \+ 115 Stanford papers with new judge → current numbers above

FINAL Literature Search Papers Score (What’s the quality of literature search/papers found by the the reviewing pipelines)

|  | DeepSeek | Stanford |
| :---- | :---: | :---: |
| Papers scored | **95** | 95 |
| **MRS** | **0.8486** | 0.8041 |
| High (\>0.85) | **50.0%** | 29.1% |
| Medium (0.75–0.85) | 43.2% | **57.8%** |
| Low (\<0.75) | **6.8%** | 13.2% |
| Papers discussed | 3.3 | 5.9 |

**EXTERNAL BENCHMARK EVALS: [Agentic Reviewer](https://docs.google.com/document/d/1HFD2pFZJmd9LiA2GtLID5zGxuhv_TZBYafuPW_4fez8/edit?tab=t.zam0zz3q5lt2) [alex.shengzhi@gmail.com](mailto:alex.shengzhi@gmail.com)**
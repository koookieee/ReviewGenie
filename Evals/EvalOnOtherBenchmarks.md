## **DeepReview Benchmark (External)**

**Benchmark:** [WestlakeNLP/DeepReview-13K](https://huggingface.co/datasets/WestlakeNLP/DeepReview-13K) — ICLR papers with human reviewer scores from OpenReview.

**Setup:** Our standard Harbor/E2B harness run on DeepReview papers as-is (no instruction changes). Agent writes in native format; `05_run_harness.py` post-processes to `\boxed_review{}` format. Scored with the official `evalate.py` from [zhu-minjun/Researcher](https://github.com/zhu-minjun/Researcher/tree/main/evaluate/DeepReview).

**Metrics:** Compared against mean human reviewer scores per paper.

- **MSE/MAE** — absolute score error vs human mean  
- **Spearman** — ranking correlation with human ratings  
- **Decision Accuracy** — Accept/Reject match with committee decision  
- **Pairwise Acc** — correct ordering for all C(N,2) paper pairs


  
---

### **ICLR 2024 (652 papers, DeepSeek V4 Pro, K=1)**

| Metric | Our Score |
| :---- | :---- |
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
| Pairwise Rating Accuracy | 0.6586 |
| Pairwise Soundness Accuracy | 0.6464 |
| Pairwise Presentation Accuracy | 0.6825 |
| Pairwise Contribution Accuracy | 0.6106 |

---

### **ICLR 2024 (652 papers, DeepSeek V4 Flash, K=1)**

| Metric | Our Score |
| :---- | :---- |
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

### **Win-Rate vs DeepReviewer-14B (LLM Judge, Pairwise)**

**Judge:** Gemini 3.1 Pro — same SYSTEM\_PROMPT as official `win_rate_evaluate.py` from [zhu-minjun/Researcher](https://github.com/zhu-minjun/Researcher/tree/main/evaluate/DeepReview). Results in `/root/pass_at_k/win_rate_*.jsonl`.

**Categories judged:** Overall Judgment, Technical Accuracy, Constructive Value, Analytical Depth, Communication Clarity.

#### **ICLR 2024 — DeepSeek V4 Pro vs DeepReviewer-14B**

| Category | Win | Tie | Lose |
| :---- | :---- | :---- | :---- |
| Overall Judgment | **94.8%** | 2.4% | 2.9% |
| Technical Accuracy | **93.8%** | 4.0% | 2.2% |
| Constructive Value | **80.2%** | 7.6% | 12.2% |
| Analytical Depth | **92.4%** | 7.1% | 0.5% |
| Communication Clarity | **95.6%** | 3.5% | 1.0% |

#### 

#### 

#### **ICLR 2024 — DeepSeek V4 Flash vs DeepReviewer-14B** 

| Category | Win | Tie | Lose |
| :---- | :---- | :---- | :---- |
| Overall Judgment | **94.0%** | 2.5% | 3.5% |
| Technical Accuracy | **91.5%** | 5.5% | 3.0% |
| Constructive Value | **78.0%** | 9.9% | 12.1% |
| Analytical Depth | **90.2%** | 8.3% | 1.4% |
| Communication Clarity | **95.2%** | 3.2% | 1.6% |

---

### **Win-Rate vs Stanford Reviewer (LLM Judge, Pairwise, Harbor 115-paper set)**

These runs compare our harness output against Stanford Reviewer on the **same 115-paper Harbor set** (not DeepReview dataset). Results in `/root/pass_at_k/win_rate_stanford_vs_*.jsonl`.

#### **DeepSeek V4 Flash vs Stanford Reviewer (115 papers)**

| Category | Win | Tie | Lose |
| :---- | :---- | :---- | :---- |
| Overall Judgment | 54.8% | 26.1% | 19.1% |
| Technical Accuracy | 48.7% | 38.3% | 13.0% |
| Constructive Value | 20.0% | 59.1% | 20.9% |
| Analytical Depth | 43.5% | 47.0% | 9.6% |
| Communication Clarity | 30.4% | 69.6% | 0.0% |

#### **DeepSeek V4 Pro vs Stanford Reviewer (115 papers)**

| Category | Win | Tie | Lose |
| :---- | :---- | :---- | :---- |
| Overall Judgment | 58.3% | 22.6% | 19.1% |
| Technical Accuracy | 47.8% | 35.7% | 16.5% |
| Constructive Value | 18.3% | 61.7% | 20.0% |
| Analytical Depth | 46.1% | 44.3% | 9.6% |
| Communication Clarity | 26.1% | 73.9% | 0.0% |

**Key finding:** Against Stanford Reviewer the picture is much closer — \~50% win on Judgment/Technical, but Communication Clarity and Constructive Value are mostly ties or losses.  
---

# **BELOW IS WIN RATE OF SCHOLAR PEER(FROM GOOGLE) vs Others:**  

**What we can still improve: WE CAN MAKE THE REVIEW FROM OUR Harness to be more constructive (that is the review should also provide feedback on how we a paper can be improved, which Stanford Reviewer does and that’s what we almost tie on Constructive Value )**


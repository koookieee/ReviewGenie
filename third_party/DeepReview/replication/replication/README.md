# DeepReview Benchmark Replication

This folder contains everything needed to reproduce the DeepReview benchmark on your own model and on baselines, using the released `WestlakeNLP/DeepReview-13K` test splits and the official eval scripts at `../Researcher/evaluate/DeepReview/`.

## Layout

```
replication/
├── data/
│   ├── data/                          # raw HF download (test_2024.csv, test_2025.csv)
│   ├── eval_set_iclr2024.json         # 652 papers, eval-ready schema
│   └── eval_set_iclr2025.json         # 634 papers, eval-ready schema
├── scripts/
│   ├── prompts.py                     # canonical system prompts
│   ├── 01_build_eval_set.py           # CSV → JSON in evalate.py schema
│   ├── 02_run_inference.py            # runs your model + writes pred_*_mode
│   ├── 03_evaluate.py                 # wraps the official evalate.py
│   └── 04_smoke_test.py               # oracle sanity check (verified working)
├── predictions/                       # one JSON per (model, year)
└── results/                           # markdown tables of metrics
```

## What I Verified About the Test Set

| Property | Value | Match with paper? |
|---|---|---|
| `test_2024.csv` rows | 652 | ✅ paper says 652 |
| `test_2025.csv` rows | 634 | ✅ paper says 634 |
| `test_2024.csv` accept rate | 43.7% | ✅ exact |
| `test_2025.csv` accept rate | 31.1% | ✅ exact |
| ID overlap between files | 0 | ✅ disjoint splits |

**Caveat:** the `year` column inside both CSVs reads `2025` for every row, but by filename and by accept-rate match, `test_2024.csv` IS the ICLR 2024 split. Don't trust the column.

## CSV Schema (raw)

| Column | Type | Notes |
|---|---|---|
| `inputs` | JSON list | `[{role:"system",content:...}, {role:"user",content:<paper_text>}]`. The system prompt is DeepReviewer-specific Best Mode and **not used by the eval**. |
| `outputs` | JSON list | DeepReviewer-14B's own generated review (multi-turn for Best Mode). Only useful as a reference. |
| `year` | int | Always 2025. **Ignore — use filename instead.** |
| `id` | str | OpenReview submission ID |
| `mode` | str | Always `best` (relates to which DeepReviewer mode produced `outputs`). Ignore. |
| `rating` | JSON list[int] | Per-reviewer overall ratings. |
| `decision` | str | `Accept` / `Reject` |
| `reviewer_comments` | JSON list | Full human reviews. **This is the ground truth source.** Each entry has `content.{rating, soundness, presentation, contribution, summary, strengths, weaknesses, ...}` where the numeric fields are strings like `"3 good"`. |

## Eval JSON Schema (what `evalate.py` reads)

After `01_build_eval_set.py`, each record is:
```json
{
  "id": "...",
  "title": "",
  "paper_context": "<full paper LaTeX/text>",
  "decision": "Accept" | "Reject",
  "review": [...same as reviewer_comments...]
}
```

Your inference step appends `pred_<mode>_mode` keys per record:
```json
"pred_standard_mode": "<arbitrary thinking text>\\boxed_review{\n## Summary:\n\n...\n## Rating:\n\n6.5\n...\n## Decision:\n\nAccept\n}"
```

`evalate.py` parses the `\boxed_review{...}` block and looks for `## <Field>:\n\n<value>` for: Summary, Rating, Soundness, Presentation, Contribution, Strengths, Weaknesses, Suggestions, Questions, Confidence, Decision.

## Required Output Format (CRITICAL)

The `\boxed_review{}` block must contain numeric values that parse cleanly:

- ✅ `## Rating:\n\n6.5`
- ✅ `## Soundness:\n\n3`
- ❌ `## Soundness:\n\n3 good` (eval calls `float()` on first line and crashes; row silently dropped)
- Decision must contain the substring `accept` or `reject` (case-insensitive)

If your model's output doesn't conform, `evalate.py` silently skips the row. **Always check that the row count in the printed results matches your dataset size.** I patched `02_run_inference.py` to log errors but the eval script itself swallows them.

## Prompts

### For DeepReviewer-14B
Verbatim from `Researcher/ai_researcher/deep_reviewer.py`. See `scripts/prompts.py:deepreviewer_system_prompt(mode, reviewer_num)`. Three modes: Fast, Standard, Best. Paper main results use `reviewer_num=4`.

### For your AgentReviewer / API baselines
The paper does not specify a single canonical prompt for non-DeepReviewer models — it just says they use the AI Scientist or AgentReview scaffolds. To stay reproducible, I provide `GENERIC_REVIEWER_SYSTEM_PROMPT` in `scripts/prompts.py` that instructs any LLM to emit the required `\boxed_review{}` format. Use this for:
- Your AgentReviewer (wrap its output through this prompt OR post-process its native output into the schema)
- API baselines (Claude / GPT / Gemini / DeepSeek) when not using AI Scientist / AgentReview scaffolds

### For the Gemini judge (LLM-as-Judge)
Verbatim in `Researcher/evaluate/DeepReview/win_rate_evaluate.py`, the `SYSTEM_PROMPT` constant. Don't modify — reproducibility depends on it.

## Inference Settings

From the paper §5.1 and `deep_reviewer.py`:

| Setting | Value | Source |
|---|---|---|
| Temperature | 0.4 | paper + code |
| top_p | 0.95 | code only (not in paper) |
| Max input tokens | 100K | paper |
| Max output tokens | 16,384 | paper (code uses 35K for DeepReviewer) |
| Judge model | `gemini-2.0-flash-thinking-exp-01-21` | code |
| Judge temperature | 0 | code |

## End-to-End Flow

```bash
# 0) deps (one-time)
python3 -m venv .venv && .venv/bin/pip install torch scipy scikit-learn pandas openai tqdm

# 1) build eval-ready JSON from the CSVs (already done, but rerun if needed)
.venv/bin/python scripts/01_build_eval_set.py

# 2) sanity check the eval pipeline with an oracle (already verified — gets MSE=0, Spearman=1)
.venv/bin/python scripts/04_smoke_test.py
.venv/bin/python scripts/03_evaluate.py --predictions predictions/oracle_iclr2024.json --mode standard

# 3) run YOUR AgentReviewer (after wiring up make_agentreviewer_backend in scripts/02_run_inference.py)
.venv/bin/python scripts/02_run_inference.py \
    --eval-set data/eval_set_iclr2024.json \
    --out predictions/myagent_iclr2024.json \
    --backend agentreviewer \
    --mode-key pred_standard_mode

# 4) evaluate
.venv/bin/python scripts/03_evaluate.py \
    --predictions predictions/myagent_iclr2024.json \
    --mode standard \
    --out-md results/myagent_iclr2024.md

# 5) repeat for ICLR 2025
# 6) repeat for each baseline you want to compare against
# 7) (optional) win-rate evaluation — adapt Researcher/evaluate/DeepReview/win_rate_evaluate.py
```

## What's Verified

✅ Test set downloaded and row counts / accept rates match the paper exactly
✅ Schema converter produces the right format
✅ Oracle smoke test produces perfect metrics — eval pipeline works
✅ Generic prompt produces parseable `\boxed_review{}` output (format spec)

## What Still Needs You

🟡 Wire your AgentReviewer entry point into `make_agentreviewer_backend()` in `scripts/02_run_inference.py`
🟡 Decide which baselines to run (each costs API budget or GPU time — see bottlenecks doc in main DeepReview/ folder)
🟡 If you want LLM-as-Judge: adapt `win_rate_evaluate.py` (it's hardcoded to compare DeepReviewer's own modes; the `prepare_comparison_data` function needs ~10 lines changed to take two arbitrary prediction files)

## Security Note

You shared an HF token in chat. **Rotate it now** at https://huggingface.co/settings/tokens — once any token is in a logged conversation, treat it as compromised.

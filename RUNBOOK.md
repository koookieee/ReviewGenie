# HarborTrajectoryGen — Pass@K Benchmark Runbook

Complete record of issues encountered, fixes applied, and how to run the pass@K benchmark end-to-end.

---

## Overview

The goal is to run a pass@K evaluation: for each of 100 arXiv papers with human reviews, run K=4 independent paper review attempts using **Kimi K2.5** (via Fireworks), score each with a **Gemini LLM judge**, and compute pass@K statistics.

**Stack:**
- **Harbor** — trial execution framework
- **E2B** — cloud sandboxes where Claude Code + Kimi K2.5 runs
- **stream_proxy.py** — translates Anthropic API calls from Claude Code → Fireworks (Kimi K2.5)
- **Gemini 3.1 Pro** — LLM judge that scores each review
- **Paper Search API** — 928K arXiv paper search running on the machine (port 8081)

**Machine:** vast.ai instance at ssh -p 47139 root@171.226.36.255 -L 8080:localhost:8080

---

## Part 1: Dataset Preparation

### Problem: No papers with human reviews

The original `fetch_papers.py` downloaded 100 random CS papers from `Vidushee/ArXiv-Papers-150K` but `human_reviews` was empty (`[]`) for all of them. The LLM judge needs human reviews to evaluate `issue_overlap` and `calibration` criteria — without them it uses a degraded 3-criterion scoring formula.

### Fix: `fetch_papers_with_reviews.py`

**Strategy:**
1. Load `Vidushee/ArXiv-Papers-150K` metadata.parquet (150K papers)
2. Load `Vidushee/openreview-filtered-reviews` filtered_all.csv (96K reviews, all `passed_filter=True`)
3. Match by `paper_title` (case-insensitive exact match) → **5,243 papers** overlap
4. Prefer papers with 2–4 reviews (best signal-to-noise)
5. Download LaTeX from arXiv e-print, convert, set up task directories
6. Populate `human_reviews` in `task_metadata.json` as **formatted markdown strings**

**Result:** 100 papers at `/root/data/pass_at_k_papers_reviewed/`
- 21 papers with 2 reviews, 44 with 3, 35 with 4 — all from ICLR/NeurIPS via OpenReview

### Problem: `human_reviews` stored as dicts, not strings

The benchmark scoring code does:
```python
f"**Human Review {i+1}:**\n\n{r}"
for i, r in enumerate(metadata["human_reviews"])
```
If `r` is a dict, `str(r)` produces raw Python dict syntax — unreadable for the judge.

**Fix:** `_format_review()` in `fetch_papers_with_reviews.py` converts each review dict into clean markdown:
```
**Scores:**
- Rating: 7: Good paper, accept
- Confidence: 3: ...

**Summary:**
...

**Strengths:**
...
```
This also preserves `rating`, `soundness`, `contribution`, `presentation` fields needed for the **Calibration** criterion.

### Problem: Missing Harbor task files

The new dataset was missing `task.toml` and `environment/Dockerfile` which Harbor requires for every task directory.

**Fix:** Added both to all 100 task directories:
- `task.toml` — Harbor schema v1.1, `build_timeout_sec=600`, `allow_internet=true`
- `environment/Dockerfile` — minimal `ubuntu:22.04` with `curl git`, `WORKDIR /app`

All 100 Dockerfiles are **identical** (same MD5), which means E2B builds one shared template and reuses it across all runs.

---

## Part 2: Infrastructure Setup

### Fireworks API Key

The `stream_proxy.py` was running with an invalid key. Valid key is in `.env` as `FIREWORKS_API_KEY`.

### Wrong Fireworks model name

The proxy had `FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2p5-turbo"` which returned 404. Correct model ID confirmed via the Fireworks models API:

```
accounts/fireworks/routers/kimi-k2p5-turbo
```

**Fix in `stream_proxy.py`:**
```python
FIREWORKS_MODEL = "accounts/fireworks/routers/kimi-k2p5-turbo"
```

### Port mapping (vast.ai)

vast.ai maps internal ports to public ports via env vars:
```
VAST_TCP_PORT_8861=17217   # stream_proxy
VAST_TCP_PORT_8081=17869   # search API
VAST_TCP_PORT_8282=17263   # trajectory viewer
```

E2B sandboxes are remote cloud boxes — they reach the machine via **public IP + mapped port**, not `localhost`. So `ANTHROPIC_BASE_URL` must be `http://142.127.68.223:17217` (not `localhost:8861`).

### Starting services on the remote machine

```bash
# 1. Stream proxy (Anthropic → Fireworks translator)
python3 /root/stream_proxy.py $FIREWORKS_API_KEY 8861 > /root/proxy.log 2>&1 &

# 2. Paper search API
GEMINI_API_KEY=<key> nohup python3 /root/paper_reviewer/search_api.py \
  --port 8081 --gemini-index-dir /workspace/gemini_index > /root/search_api.log 2>&1 &

# 3. Trajectory viewer
cd /root/trajectory-viewer && \
  RESULTS_DIR=/root/pass_at_k/results_v2 python3 serve.py 8282 > server.log 2>&1 &
```

### `.env` for `benchmark_pass_at_k.py`

The script loads `PROJECT_DIR/.env` where `PROJECT_DIR = Path(__file__).parent.parent`. Since the script lives at `/root/benchmark/benchmark_pass_at_k.py`, `PROJECT_DIR = /root` and `.env` must be at `/root/.env`:

```env
ANTHROPIC_BASE_URL=http://142.127.68.223:17217
ANTHROPIC_API_KEY=dummy-key
ANTHROPIC_AUTH_TOKEN=dummy-key
E2B_API_KEY=<your-e2b-key>
GEMINI_API_KEY=<your-gemini-key>
SEARCH_PUBLIC_URL=http://142.127.68.223:17869
```

The script also expects `prompts/` and `skills/` under `PROJECT_DIR`. On the remote these were symlinked:
```bash
ln -s /root/benchmark/prompts /root/prompts
ln -s /root/benchmark/skills /root/skills
```

---

## Part 3: Benchmark Code Fixes

### Fix 1: LaTeX → Markdown conversion

**Problem:** The paper was uploaded as raw LaTeX (`\documentclass{article}...`). The agent tried to read it as text and got garbled content. The instruction says to read `latex/template.tex` expecting clean readable text.

**Fix in `_setup_environment`:** Before uploading, convert using `latex_to_markdown.py` (pandoc-based), write to a temp file, upload that as `latex/template.tex`. The source file on disk is **never mutated**.

```python
md_content = _latex_to_markdown(tex_file)   # calls latex_to_markdown.py --stdout
md_content = _trim_to_conclusion(md_content) # cuts off after Conclusion section
# write to tmp file, upload to /app/latex/template.tex
```

### Fix 2: `force_build=False`

**Problem:** `force_build=True` caused Harbor to rebuild the E2B Docker template from scratch on every trial. With 50 concurrent runs all rebuilding simultaneously, E2B consistently failed with `TimeoutException: The sandbox was not found`.

**Fix:** Set `force_build=False`. Since all 100 task directories share an identical `environment/Dockerfile` (same MD5), they all hash to the same E2B template name. The template is built once on the first run, then reused by all subsequent runs — sandbox startup drops from ~2 minutes to ~3 seconds.

```python
"environment": {
    "type": "e2b",
    "force_build": False,   # was True — caused mass sandbox failures at high concurrency
    ...
}
```

### Fix 3: Skill uploaded to all possible paths

**Problem:** Claude Code inside E2B looks for skills at `~/.claude/skills/`. The home directory inside E2B might be `/root`, `/home/user`, or the workdir. Only uploading to `/root/.claude/skills/` meant the skill was missing if home was elsewhere.

**Fix:** Upload to all three locations:
```python
for skill_target in [
    "/root/.claude/skills/search-papers/SKILL.md",
    "/home/user/.claude/skills/search-papers/SKILL.md",
    f"/{workdir.strip('/')}/.claude/skills/search-papers/SKILL.md",
]:
    await self._environment.upload_file(skill_file, skill_target)
```

### Fix 4: Skip already-completed attempts

**Problem:** Re-running the benchmark after a partial run would re-execute successful attempts, wasting time and money.

**Fix:** At the start of `run_single_attempt`, check if `result.json` exists and has `status=success` with `reward > 0`:
```python
existing_result = result_dir / "result.json"
if existing_result.exists():
    existing = json.loads(existing_result.read_text())
    if existing.get("status") == "success" and existing.get("reward", 0) > 0:
        return existing  # skip
```

### Fix 5: Agent using WebSearch instead of search API

**Problem:** The agent loaded the `/search-papers` skill correctly but then used `WebSearch` (59 times!) instead of `curl` calls to the local search API. `WebSearch` is a built-in Claude Code tool that works without any API key — it's executed by the Claude Code binary directly.

**Fix:** Made both `paper_reviewer_instruction_template.md` and `SKILL.md` explicitly prohibit WebSearch:

In `instruction_template.md`:
```
**CRITICAL: You MUST use the `/search-papers` skill (local Paper Search API) for ALL
literature searches. Do NOT use WebSearch, WebFetch, or any internet search tools —
they are disabled for this task.**
```

In `SKILL.md`:
```
**IMPORTANT: Use ONLY this local Paper Search API for all literature searches.
Do NOT use WebSearch, WebFetch, or any internet search — they are prohibited.**
```

---

## Part 4: Viewer Fix

**Problem:** The trajectory viewer (`serve.py`) only looked one level deep (`results/run_name/trajectory.json`) but pass@K results are two levels deep (`results/paper_id/attempt_N/trajectory.json`).

**Fix in `serve.py`:** Added nested path traversal:
```python
# Flat layout: results/run_name/trajectory.json
traj = top_dir / "trajectory.json"
if traj.is_file(): ...

# Nested layout: results/paper_id/attempt_N/trajectory.json
for attempt_dir in sorted(top_dir.iterdir()):
    traj = attempt_dir / "trajectory.json"
    if traj.is_file():
        label = f"{top_dir.name}/{attempt_dir.name}"
        url = f"/results/{top_dir.name}/{attempt_dir.name}/trajectory.json"
```

Also added a matching route handler:
```python
app.router.add_get("/results/{paper}/{attempt}/{filename}", handle_nested_result_file)
```

---

## Part 5: Running the Full Benchmark

### Prerequisites

1. All services running (stream_proxy, search_api, trajectory viewer)
2. Dataset at `/root/data/pass_at_k_papers_reviewed/` (100 papers with human reviews)
3. `/root/.env` configured (see Part 7 for the new env-driven provider config)
4. Symlinks `/root/prompts` → `/root/benchmark/prompts` and `/root/skills` → `/root/benchmark/skills`
5. Provider + model wired up via `pass_at_k/switch_provider.sh` (see Part 7)

### Command

```bash
# 1. Kill any stale E2B sandboxes (see "Stopping" below).
# 2. Point the proxy at the provider+model you want:
bash /root/benchmark/pass_at_k/switch_provider.sh deepinfra-qwen3p6-35b-a3b

# 3. Source env so PROXY_MODEL is visible to the benchmark (used for labels).
set -a && source /root/.env && set +a

# 4. Launch.
cd /root/benchmark && nohup python3 -u benchmark_pass_at_k.py \
  --data-dir /root/data/pass_at_k_papers_reviewed \
  --results-dir /root/pass_at_k/results_v2 \
  --trials-dir /root/pass_at_k/trials_v2 \
  --k 4 \
  --max-concurrent 50 \
  > /root/pass_at_k_full_v2.log 2>&1 &
```

**Concurrency note:** 50 concurrent is safe because `force_build=False` means no template rebuilds. Each trial takes ~5–10 minutes (agent execution) so 400 trials / 50 concurrent ≈ **~30–40 minutes** total.

### Monitoring

```bash
# Progress
tail -f /root/pass_at_k_full_v2.log

# Proxy activity (confirms model is running)
tail -f /root/proxy.log

# Completed attempts
find /root/pass_at_k/results_v2 -name 'result.json' | wc -l

# View trajectories
# http://142.127.68.223:17263/
```

### Stopping and killing E2B sandboxes

```bash
# Kill benchmark process
pkill -f benchmark_pass_at_k

# Kill all E2B sandboxes
python3 -c "
import os
os.environ['E2B_API_KEY'] = '<key>'
from e2b import Sandbox
p = Sandbox.list()
count = 0
while True:
    items = p.next_items()
    if not items: break
    for s in items:
        Sandbox.kill(s.sandbox_id)
        count += 1
print(f'Killed {count} sandboxes')
"
```

---

## Part 6: Running on a New Machine

When moving to a new vast.ai instance:

1. **Check port mappings:**
   ```bash
   env | grep VAST_TCP_PORT
   ```

2. **Update `/root/.env`** with new `ANTHROPIC_BASE_URL` and `SEARCH_PUBLIC_URL` using the new mapped ports.

3. **Restart all three services** (stream_proxy, search_api, trajectory viewer) with the new ports.

4. **Re-run is safe** — the skip logic in `run_single_attempt` skips already-successful attempts, so you can resume from where you left off.

5. **Dataset** at `/root/data/pass_at_k_papers_reviewed/` needs to be present. Either rsync it or re-run `fetch_papers_with_reviews.py`.

---

## File Reference

| File | Purpose |
|------|---------|
| `pass_at_k/benchmark_pass_at_k.py` | Main benchmark runner. Model label, trial-name prefix, and `result.json.model` are all derived from `$PROXY_MODEL` — no hardcoded names |
| `pass_at_k/stream_proxy.py` | Anthropic→OpenAI proxy. Fully env-driven (`PROXY_BASE_URL`, `PROXY_MODEL`, `PROXY_API_KEY`, `PROXY_MAX_TOKENS_CAP`). None-safe tool_calls for DeepInfra streaming |
| `pass_at_k/providers.yaml` | Catalog of vetted provider+model configs (Fireworks+MiniMax, Fireworks+Kimi, DeepInfra+Qwen) |
| `pass_at_k/switch_provider.sh` | One-shot provider switcher — updates .env, restarts proxy, verifies E2E non-streaming + streaming before returning |
| `pass_at_k/calibrate_judge.py` | PeerJudge calibration harness (bootstrap + analyze subcommands; computes Cohen's κ / Pearson r / MAE vs manual scores) |
| `pass_at_k/fetch_papers_with_reviews.py` | Download 100 papers with human reviews. Includes `_format_review` with the NaN-safe `_first_val` helper (see Part 7) |
| `pass_at_k/analyze_pass_at_k.py` | Compute pass@K statistics from results |
| `latex_to_markdown.py` | LaTeX→Markdown converter (pandoc-based) |
| `viewer/serve.py` | Trajectory viewer server (supports nested paper/attempt paths + `/evaltests` route) |
| `prompts/paper_reviewer_instruction_template.md` | Agent task instructions. Points the agent at `/app/search` CLI (not curl) |
| `prompts/llm_judge_instruction.md` | PeerJudge — 7-criterion adversarial rubric with groundedness, fabrication, missed-weakness, and pairwise calibration (see Part 7) |
| `skills/search-papers/SKILL.md` | Skill doc for the `/app/search` CLI wrapper |
| `skills/search-papers/search` | Python CLI wrapper for the Paper Search API (three subcommands: `batch`, `related`, `query`) |
| `.env` | All API keys + `PROXY_*` config (consumed by stream_proxy + benchmark + switch_provider) |

---

## Part 7: Changes from the April 18, 2026 overhaul

This section documents the major rework: search-tool CLI wrapper, dataset-formatter
fix, provider-agnostic proxy, and the new PeerJudge rubric. Everything below is
already applied on remote — this is the mental model.

### 7.1 `/app/search` CLI replaces curl-in-skill

**Problem:** Agents ignored the skill's curl examples and invented endpoints
(`/search_papers`, `/api/help`) with wrong field names (`query`, `top_k`).
Root cause was not the model — it was that skill content arrives as
`role:"tool"` via stream_proxy, which MiniMax/Kimi weight lower than
`role:"user"`. Any serious REST prior overrides the skill.

**Fix:** `skills/search-papers/search` is a tiny Python CLI the agent calls
by absolute path. The skill (and `instruction.md`) document ONLY the CLI;
there is no way to get the endpoint or field names wrong.

```bash
# Find papers (same question or multi-query)
/app/search batch "query 1" "query 2" "query 3" --max 6 --sort importance
#                                               optional: --categories cs.LG cs.CV --year 2024

# Find papers related to one
/app/search related 1706.03762 --max 6

# Ask ONE question across one or many papers
/app/search query 1706.03762 2010.11929 --q what are the key contributions

# Ask DIFFERENT questions per paper (parallel fan-out)
/app/search query \
  --pair 1706.03762 "what is the attention mechanism" \
  --pair 2010.11929 "how are patches tokenized"
```

CLI behavior:
- `batch` accepts both positional queries and a `--queries` alias (prevents
  "unrecognized arguments" failures from models that copy the wrong syntax).
- `query --q` is `nargs="+"` (greedy), so quoting is optional. It MUST come
  last. Unquoted multi-word questions work.
- `query --pair` is the per-paper-question escape hatch; QUESTION must be quoted.

The CLI is uploaded into each E2B sandbox at `/app/search` in `_setup_environment`
alongside the skill. Source of truth is `skills/search-papers/search` —
do NOT hand-edit the remote copy.

### 7.2 Dataset formatter fix — the pandas-NaN-is-truthy bug

**Problem:** `task_metadata.json.human_reviews` for some papers contained only
the `**Scores:**` header (~150 chars), not the full review body (1–2K chars).
This capped `issue_overlap` at 0.0 because the judge had nothing to compare
against.

**Root cause in `_format_review`:**

```python
main_text = _val(row.get("main_review") or row.get("review"))
```

`row.get("main_review")` returns `float('nan')` for rows where the field is
empty in pandas. **NaN is truthy in Python**, so `or` short-circuits and
returns NaN instead of falling through to `row.get("review")`. `_val(nan)`
correctly returns `""` — but by then the real body has been skipped.

**Fix:** added `_first_val(*keys)` helper that applies `_val` to each candidate
and returns the first non-empty string. Used for every OR-style field lookup
in the formatter.

To regenerate metadata for existing papers after the fix:

```bash
python3 << 'PY'
import pandas as pd, json, sys
sys.path.insert(0, '/root/benchmark/pass_at_k')
import fetch_papers_with_reviews as m
df = pd.read_csv('/workspace/.hf_home/hub/datasets--Vidushee--openreview-filtered-reviews/snapshots/*/data/filtered_all.csv', low_memory=False)
for paper_dir in Path('/root/data/pass_at_k_papers_reviewed').iterdir():
    meta = json.loads((paper_dir / 'task_metadata.json').read_text())
    title = meta.get('title', '')
    hits = df[df['paper_title'].astype(str).str.lower().str.contains(title.lower()[:30], na=False)]
    if not len(hits): continue
    meta['human_reviews'] = [m._format_review(r.to_dict()) for _, r in hits.iterrows()]
    (paper_dir / 'task_metadata.json').write_text(json.dumps(meta, indent=2))
PY
```

### 7.3 Provider-agnostic stream proxy

**Before:** `stream_proxy.py` had `FIREWORKS_URL` / `FIREWORKS_MODEL` hardcoded,
API key as argv[1], `max_tokens` capped at 16384.

**After:** all config is env-driven and `benchmark_pass_at_k.py` reads
`PROXY_MODEL` to label `result.json` and trial names.

```
PROXY_BASE_URL        — e.g. https://api.deepinfra.com/v1/openai
PROXY_MODEL           — e.g. Qwen/Qwen3.6-35B-A3B
PROXY_API_KEY         — upstream provider key
PROXY_MAX_TOKENS_CAP  — hard ceiling (default 32768)
```

**Additional proxy bug fixes:**
- Streaming `delta.get("tool_calls", [])` → `delta.get("tool_calls") or []`.
  DeepInfra emits `{"tool_calls": null}` in stream deltas; the old code crashed
  with `TypeError: 'NoneType' object is not iterable`.
- `max_tokens` cap raised from 16384 → 32768. Reasoning models like
  `Qwen3.6-35B-A3B` burn 1–2K tokens on `reasoning_content` (stripped by the
  proxy) before emitting final `content`, so 16K is tight for long reviews.

### 7.4 One-shot provider switching

Three named entries in `pass_at_k/providers.yaml`:

- `fireworks-minimax-m2p7` — MiniMax-M2.7 via Fireworks
- `fireworks-kimi-k2p5` — Kimi K2.5 via Fireworks (original paper baseline)
- `deepinfra-qwen3p6-35b-a3b` — Qwen3.6 35B A3B via DeepInfra (reasoning model)

Switching:

```bash
# List available
bash /root/benchmark/pass_at_k/switch_provider.sh --list

# Show what's running right now
bash /root/benchmark/pass_at_k/switch_provider.sh --status

# Switch (updates .env, kills old proxy, launches new, runs non-streaming +
# streaming E2E ping before exiting 0)
bash /root/benchmark/pass_at_k/switch_provider.sh deepinfra-qwen3p6-35b-a3b
```

The script reads the provider's `api_key_env` field from `/root/.env`
(e.g. `DEEPINFRA_API_KEY`), writes it as `PROXY_API_KEY`, and relaunches.
Exits non-zero if either verification ping fails — so if the script returns
0, the proxy is good to use.

**After switching, source the env** so `benchmark_pass_at_k.py` picks up
`PROXY_MODEL` for result labels:

```bash
set -a && source /root/.env && set +a
```

### 7.5 Claude Code version pin

We pin Claude Code to `2.1.101` (last release before April 11, 2026) via
`agent.kwargs.version` in `build_trial_config`. This rules out post-April-11
CC changes as a variable. If you want to roll forward, edit the version
string in `benchmark_pass_at_k.py::build_trial_config`.

### 7.6 PeerJudge — new 7-criterion rubric (adversarial)

`prompts/llm_judge_instruction.md` is rewritten. It replaces the old
5-criterion rubric with an adversarial 7-criterion one.

Criteria and weights (computed in `score_review`):

| # | Criterion | Type | Weight |
|---|-----------|------|--------|
| 1 | Comprehension | binary {0, 1} | 0.10 |
| 2 | Substance & Specificity | binary {0, 1} | 0.10 |
| 3 | Insight (groundedness-enforced) | {0.0, 0.5, 1.0} | 0.15 |
| 4 | Issue Overlap | [0.0, 1.0] | 0.15 |
| 5 | Missed-Weakness (adversarial) | [0.0, 1.0] | 0.15 |
| 6 | Fabrication check (adversarial) | {0.0, 0.5, 1.0} | 0.15 |
| 7 | Pairwise Calibration | [0.0, 1.0] | 0.20 |

Key design choices:

- **Groundedness over citation formatting.** Criterion 3 requires each
  counted observation to be traceable to paper content — by name OR by
  paraphrase. Judge must emit an `insight_observations` list with
  `(observation, grounds_in, evidence)` for each. Previous "must cite by
  section number" was biased toward empirical/figure-heavy papers;
  theoretical reviews now score fairly.
- **Two adversarial criteria** (missed-weakness, fabrication) create
  downward pressure on the reward ceiling. Old rubric had none.
- **Pairwise calibration** replaces absolute ±2 tolerance. Judge makes
  per-dimension Worse/Equal/Better verdicts against the human consensus
  reasoning (LMSys-Arena-style).
- **Handles variable N human reviews (1–4).** If N=1, use that reviewer
  as ground truth. If N≥2, weight convergent points (raised by ≥2
  reviewers) 2× versus single-reviewer points.
- **Every score requires evidence.** The judge must emit `justification`
  + `evidence` fields. Scores without evidence → 0.

Degraded mode (no human reviews): weights renormalize across
Comprehension + Substance + Insight + Fabrication only (overlap,
missed-weakness, pairwise-calibration all require humans).

### 7.7 Calibration harness — PeerJudge validation

`pass_at_k/calibrate_judge.py` measures judge-vs-human agreement on the 7
criteria. Without this, the rubric is unpublishable — ICLR reviewers will
ask for κ numbers first.

```bash
# Step 1: pick 20 stratified-by-reward reviews from an existing results dir.
python3 pass_at_k/calibrate_judge.py bootstrap \
  --results-dir /root/pass_at_k/results_v2 \
  --data-dir /root/data/pass_at_k_papers_reviewed \
  --sample 20 \
  --out /root/calibration.jsonl

# Step 2: open /root/calibration.jsonl, for each of the 20 records fill in
# `manual_scores` by hand on all 7 criteria. This is YOUR judgement of how
# each model review should score. Takes ~1-2 hours.

# Step 3: compute agreement metrics.
python3 pass_at_k/calibrate_judge.py analyze --in /root/calibration.jsonl
```

Report emits per-criterion Cohen's κ (binary) and Pearson r + Spearman ρ
+ MAE (continuous), overall reward correlation, and top-5 disagreement
cases for qualitative iteration.

**Target thresholds (ICLR-grade):**
- Binary criteria: Cohen's κ ≥ 0.60
- Continuous criteria: Pearson r ≥ 0.70 AND MAE ≤ 0.15
- Overall reward: MAE ≤ 0.10

If any criterion misses, iterate the rubric prompt and re-run analyze.
The 20 reviews stay fixed; only the judge prompt changes between iterations.

### 7.8 Stopping E2B sandboxes (updated)

The SDK's `next_items()` raises `Exception("No more items to fetch")` as
its end-of-pagination signal (ugly, but documented). Always wrap in
try/except:

```bash
ssh -p 17909 root@142.127.68.223 "set -a; source /root/.env; set +a; python3 -c '
from e2b import Sandbox
p = Sandbox.list()
killed = 0
try:
    while True:
        items = p.next_items()
        if not items: break
        for s in items:
            Sandbox.kill(s.sandbox_id); killed += 1
except Exception: pass
print(f\"killed: {killed}\")
'"
```

---

## Part 8: OCR Markdown Pipeline + Agentic Judge

### Overview

Two major additions on top of Part 7:

1. **OCR markdown input** — replace LaTeX→pandoc conversion with olmOCR-produced
   markdown, giving the agent richer, more accurate paper text.
2. **Agentic judge** — replace single-shot Gemini judge with a tool-use loop that
   greps the paper body to verify claims before scoring fabrication.

---

### 8.1 OCR markdown with olmOCR

**Why:** LaTeX→pandoc conversion drops tables, garbles math, and produces
inconsistent structure. olmOCR (via DeepInfra) produces clean markdown directly
from PDFs.

**Where OCR files live:**
```
/root/data/ocr_markdown/<paper_id>.md   ← canonical location for all baselines
```

Do NOT store them under a specific baseline's folder (e.g. `Stanford_Reviewer/`).

**Running olmOCR on new PDFs:**
```bash
olmocr /root/olmocr_papers \
  --server https://api.deepinfra.com/v1/openai \
  --api_key <DEEPINFRA_API_KEY> \
  --model allenai/olmOCR-2-7B-1025 \
  --workers 1 --max_concurrent_requests 200 \
  --markdown \
  --pdfs /root/Stanford_Reviewer/*.pdf
```

Output lands at `/root/olmocr_papers/markdown/root/Stanford_Reviewer/<paper_id>.md`.
Move to canonical location after:
```bash
mv /root/olmocr_papers/markdown/root/Stanford_Reviewer /root/data/ocr_markdown
```

**Known limitation:** olmOCR sometimes drops or garbles numbers inside HTML table
cells (`<td><b>10.91</b></td>`). Numbers are present but the agentic judge must
grep with flexible patterns to find them (see 8.3).

---

### 8.2 Running the benchmark with OCR markdown input

Pass `--markdown-dir` to `benchmark_pass_at_k.py`. It is optional — papers
without a matching `.md` file fall back to the latex→pandoc path automatically.

```bash
# Current machine: 142.117.93.98, SSH port 35622
# Port mappings: 8861→35721 (stream_proxy), 8081→35746 (search API)

cd /root/pass_at_k && nohup python3 benchmark_pass_at_k.py \
  --data-dir /root/data/pass_at_k_reviewed \
  --markdown-dir /root/data/ocr_markdown \
  --results-dir /root/pass_at_k/results_<model>_<run> \
  --trials-dir /root/pass_at_k/trials_<model>_<run> \
  --k 1 --max-concurrent 8 --max-tasks 100 \
  > /root/pass_at_k/run_<model>.log 2>&1 &
```

**Priority order for paper body inside the sandbox:**
1. `--markdown-dir/<paper_id>.md` (OCR) — if file exists and >500 chars
2. `latex/template.tex` pre-converted markdown — if exists and not raw LaTeX
3. pandoc conversion of the raw `.tex` source

The sandbox always sees the paper at `/app/latex/template.tex` regardless of source.

---

### 8.3 Switching to a new model

**Option A — MiniMax or any provider with an Anthropic-compatible endpoint:**

Set `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` in `.env` to point directly
at the provider. No stream_proxy needed.

```env
ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
ANTHROPIC_API_KEY=<minimax_key>
```

**Option B — Any OpenAI-compatible provider (DeepInfra, Fireworks, etc.):**

Start the stream_proxy with the target model, then point the benchmark at it
via the **public** port (E2B sandboxes can't reach localhost):

```bash
# Start proxy
PROXY_BASE_URL="https://api.deepinfra.com/v1/openai" \
PROXY_MODEL="moonshotai/Kimi-K2.6" \
PROXY_API_KEY="<deepinfra_key>" \
PROXY_MAX_TOKENS_CAP="32768" \
nohup python3 /root/stream_proxy.py <key> 8861 > /root/proxy.log 2>&1 &

# Test proxy before launching (critical — don't skip)
curl -s -X POST http://localhost:8861/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-3-5-sonnet-20241022","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}'
# Must return a valid Anthropic-format JSON response before proceeding.

# Launch benchmark using PUBLIC port (35721 maps to internal 8861)
ANTHROPIC_BASE_URL="http://142.117.93.98:35721" \
ANTHROPIC_API_KEY="<key>" \
nohup python3 benchmark_pass_at_k.py \
  --data-dir /root/data/pass_at_k_reviewed \
  --markdown-dir /root/data/ocr_markdown \
  --results-dir /root/pass_at_k/results_<model>_10 \
  --trials-dir /root/pass_at_k/trials_<model> \
  --k 1 --max-concurrent 4 --max-tasks 10 \
  > /root/pass_at_k/run_<model>.log 2>&1 &
```

**Port mapping reference (142.117.93.98):**
```
35721 → 8861   stream_proxy
35746 → 8081   search API
35622 → 22     SSH
```

**Model suitability notes:**
- `DeepSeek-V4-Flash` — too slow (~2 min/turn on long contexts). Avoid.
- `moonshotai/Kimi-K2.6` — gets 429 rate limits on DeepInfra under load.
- `MiniMax-M2.7` via `api.minimax.io/anthropic` — works well, ~4 min/paper.

---

### 8.4 Agentic judge (`agentic_judge.py`)

**Why:** Single-shot Gemini judge fails at fabrication checking — it can't hunt
for specific numbers or citations in a long paper body in one pass.

**How it works:**
- Tool-use loop using native `google-genai` SDK (NOT the OpenAI-compat shim).
- Two tools: `grep_paper(pattern)` — regex search with context windows;
  `read_paper()` — full paper body re-read fallback.
- Hard caps: 40 steps, 480s wall-clock. If budget exhausted → forced JSON-mode
  final call.
- Uses `gemini-3.1-pro-preview`. Config via `AgenticJudgeConfig`.

**Why native SDK (not OpenAI shim):** Gemini 3.1 Pro emits `thought_signature`
blobs on tool-call turns that MUST be round-tripped back. The OpenAI shim drops
them, breaking the next turn. The native SDK preserves them automatically.

**Rubric (Option A, 3-criterion):**
```
reward = 0.30 × issue_overlap + 0.30 × fabrication + 0.40 × calibration_pairwise
```
comp/sub/ins are still scored but excluded from reward (all score ~1.0, no signal).

---

### 8.5 Rejudging existing outputs with the agentic judge

**Stanford Reviewer outputs:**
```bash
cd /root/pass_at_k && python3 rejudge_stanford.py \
  --reviews-dir /root/Stanford_Reviewer/reviews_proxy \
  --data-dir /root/data/pass_at_k_reviewed \
  --markdown-dir /root/data/ocr_markdown \
  --out-dir /root/Stanford_Reviewer/results_agentic_ocr \
  --max-concurrent 12
```

**MiniMax/any-model trajectories:**
```bash
python3 rejudge_with_agentic.py \
  --results-dir /root/pass_at_k/results_minimax_ocr_100 \
  --data-dir /root/data/pass_at_k_reviewed \
  --markdown-dir /root/data/ocr_markdown \
  --out-dir /root/pass_at_k/results_minimax_ocr_100_rejudged \
  --max-concurrent 12
```

**Monitoring rejudge:**
```bash
grep "reward=" /root/pass_at_k/stanford_rejudge.log | wc -l  # completed
grep "reward=" /root/pass_at_k/stanford_rejudge.log | tail -5  # recent
```

---

### 8.6 Known judge failure modes and fixes applied

**Problem 1 — Judge marks verified claims as fabrications**
The judge called `grep_paper` once, got no match on bare text, and declared
a claim unverified — even when the text was present in LaTeX (`\( T = 3000 \)`)
or HTML table cells (`<td><b>10.91</b></td>`).

**Fix in `prompts/llm_judge_instruction.md`:** Mandatory 4-step escalating grep
before any `unverified` ruling: (1) prose form, (2) loose pattern, (3) LaTeX/HTML
form, (4) `read_paper()`. Must quote the matching snippet in `note`. A single
failed grep is never enough to mark `unverified`.

**Problem 2 — Agent reviewer claims "paper didn't cite X" when it did**
Agent finds papers via `/app/search` (external), then assumes they're uncited
without checking `template.tex`.

**Fix in `prompts/paper_reviewer_instruction_template.md`:**
```
CRITICAL: Before writing that any paper, author, or method is uncited, run:
  grep -i "<lastname>" /app/latex/template.tex
Only after grep returns zero matches may you claim it is absent.
```

**Problem 3 — Agent mischaracterizes paper's framing**
Agent infers how a paper frames a comparison from search results rather than
re-reading the actual passage.

**Fix in reviewer instruction:** Added rule to re-read the relevant passage in
`template.tex` before characterizing framing (e.g. "authors dismiss X").

---

### 8.8 Results dir naming caveat

`results_deepseek_v4_pro_100` is **misnamed** — the dir was created with `v4_pro` in the path but the model that actually ran was `deepseek-v4-flash`. Always verify the actual model from the result files, not the dir name:

```bash
python3 -c "
import json
d = json.loads(open('results_deepseek_v4_pro_100/1906.00820/aggregated.json').read())
print('model:', d['attempts'][0]['model'])
"
# output: model: deepseek-v4-flash
```

The `model` field in `aggregated.json` and `result.json` is derived from `$PROXY_MODEL` at run time — it is the ground truth. Dir names are unreliable.

---

### 8.7 Results summary (as of 2026-04-27)

| Run | Papers | Mean reward | Notes |
|-----|--------|-------------|-------|
| MiniMax (no OCR, old judge) | 100 | 0.410 | baseline |
| Stanford Reviewer (old judge) | 115 | 0.668 | static reviewer |
| MiniMax OCR + agentic judge | 77/100 | 0.625 | in progress |
| Stanford OCR + agentic judge | 115 | 0.773 | complete |

Stanford wins 23/32 paired papers, mean gap +0.13. Main MiniMax weaknesses:
fabrication hallucinations (agent invents missing citations / wrong numbers)
and occasional short/empty reviews (model hits output budget before finishing).
# Harbor Infrastructure — Search + Review API + AI-Scientist v3

End-to-end setup for running the HarborTrajectoryGen review pipeline (search API + review API + AI-Scientist v3) on a remote machine, fronted by Cloudflare tunnels.

Stack:
- **Search API** (`arxiv-search-kit`) — 928K-paper arXiv retrieval, port `8081`
- **Review API** (`HarborTrajectoryGen_v2/review_api.py`) — paper review pipeline backed by Harbor + E2B, port `8082`, async submit + poll
- **AI-Scientist v3** — autonomous research agent that drives literature search → experiments → paper → external review through both APIs
- **Cloudflare Tunnel** — exposes both APIs on `*.eigenlabs.online` (use **dashed** hostnames; underscored hostnames break Python's strict SSL hostname check)

Remote: `root@65.108.33.106` (current). Past machines: `171.226.34.64`, `115.73.216.179`, `116.106.197.74`.

Source of truth: `gdrive_vastai:VastAIWorksapce/VastAI5060_60GB/root/HarborTrajectoryGen_v2/` and `.../ai-scientist-v3/` on Google Drive (rclone). Files in [patches/](patches/) are the gold copies with the dead vast.ai IPs swapped for our Cloudflare hostnames and a few correctness fixes (see [Fixes Applied](#fixes-applied)).

---

## Public Endpoints

Use the **dashed** form. Underscored hostnames work via curl but break Python (SSL hostname mismatch).

| Service | Public URL | Local port |
|---|---|---|
| Search API | `https://search-api.eigenlabs.online` | `localhost:8081` |
| Review API | `https://review-api.eigenlabs.online` | `localhost:8082` |

Health checks:
```bash
curl https://search-api.eigenlabs.online/health
curl https://review-api.eigenlabs.online/health
```

---

## Required Versions (DO NOT BUMP)

| Package | Version | Why |
|---|---|---|
| `harbor` | `0.7.0` | Gold's `PassAtKTrial._setup_environment` overrides a method that was renamed in `0.8+`. On `0.9.0` the override silently no-ops, so `/app/latex/template.tex`, `/app/search`, `/app/paper_cutoff.txt` never reach the sandbox and every review fails with "files not present". |

```bash
/root/venv/bin/pip install "harbor==0.7.0"
```

---

## Initial Setup (cold remote)

1. **Pull gold trees from rclone** (local machine, has rclone configured):
   ```bash
   mkdir -p /tmp/gold
   rclone copy gdrive_vastai:VastAIWorksapce/VastAI5060_60GB/root/HarborTrajectoryGen_v2 /tmp/gold/HarborTrajectoryGen_v2 \
       --exclude '__pycache__/**' --exclude 'data/**' --exclude '.git/**' --transfers 8 --progress
   rclone copy gdrive_vastai:VastAIWorksapce/VastAI5060_60GB/root/ai-scientist-v3 /tmp/gold/ai-scientist-v3 \
       --exclude '__pycache__/**' --exclude 'jobs/**' --exclude '.git/**' --transfers 8 --progress
   ```

2. **Apply the patches** in [patches/](patches/) over the gold (they are already URL-corrected and bug-fixed):
   ```bash
   cp harbor_infra/patches/review_api.py        /tmp/gold/HarborTrajectoryGen_v2/review_api.py
   cp harbor_infra/patches/search_cli.py        /tmp/gold/HarborTrajectoryGen_v2/skills/search-papers/search
   cp harbor_infra/patches/search_cli.py        /tmp/gold/ai-scientist-v3/.claude/skills/search-papers/search
   cp harbor_infra/patches/submit_for_review.sh /tmp/gold/ai-scientist-v3/scripts/submit_for_review.sh
   cp harbor_infra/patches/run.sh               /tmp/gold/ai-scientist-v3/run.sh
   chmod +x /tmp/gold/HarborTrajectoryGen_v2/skills/search-papers/search \
            /tmp/gold/ai-scientist-v3/.claude/skills/search-papers/search \
            /tmp/gold/ai-scientist-v3/run.sh \
            /tmp/gold/ai-scientist-v3/scripts/submit_for_review.sh
   ```

3. **rsync to remote**:
   ```bash
   rsync -az --delete --exclude '__pycache__' --exclude '.git' --exclude 'data' --exclude 'jobs' \
       /tmp/gold/HarborTrajectoryGen_v2/ root@65.108.33.106:/root/HarborTrajectoryGen_v2/
   rsync -az --delete --exclude '__pycache__' --exclude '.git' --exclude 'jobs' \
       /tmp/gold/ai-scientist-v3/ root@65.108.33.106:/root/ai-scientist-v3/
   ```

4. **Install Harbor 0.7.0 on remote**:
   ```bash
   ssh root@65.108.33.106 '/root/venv/bin/pip install "harbor==0.7.0"'
   ```

5. **Place the Cloudflare tunnel config** ([configs/cloudflared_config.yml](configs/cloudflared_config.yml)) at `/root/.cloudflared/config.yml` on the remote. The dashed and underscored hostnames are both routed (the underscored ones are kept for back-compat, but Python clients MUST use dashed).
   ```bash
   scp harbor_infra/configs/cloudflared_config.yml root@65.108.33.106:/root/.cloudflared/config.yml
   ```

   If hostnames don't yet exist as DNS routes:
   ```bash
   ssh root@65.108.33.106 'cloudflared tunnel route dns search_api search-api.eigenlabs.online'
   ssh root@65.108.33.106 'cloudflared tunnel route dns search_api review-api.eigenlabs.online'
   ```

6. **Start everything** with [scripts/start_remote.sh](scripts/start_remote.sh):
   ```bash
   scp harbor_infra/scripts/start_remote.sh root@65.108.33.106:/tmp/
   ssh root@65.108.33.106 'GEMINI_API_KEY=<your-key> bash /tmp/start_remote.sh'
   ```

---

## Day-to-Day Operations

### Start / restart everything
```bash
ssh root@65.108.33.106 'GEMINI_API_KEY=<key> bash /tmp/start_remote.sh'
```

### Restart only review API (after editing `review_api.py`)
```bash
scp harbor_infra/scripts/restart_review.sh root@65.108.33.106:/tmp/
ssh root@65.108.33.106 'bash /tmp/restart_review.sh'
```

### Restart only Cloudflare tunnel
```bash
scp harbor_infra/scripts/restart_cf.sh root@65.108.33.106:/tmp/
ssh root@65.108.33.106 'bash /tmp/restart_cf.sh'
```

### Tail logs
```bash
ssh root@65.108.33.106 'tail -f /root/search_api.log'
ssh root@65.108.33.106 'tail -f /root/review_api.log'
ssh root@65.108.33.106 'tail -f /root/cloudflared.log'
```

### Health checks
```bash
curl https://search-api.eigenlabs.online/health
curl https://review-api.eigenlabs.online/health
```

---

## Search API Usage

```bash
curl -X POST https://search-api.eigenlabs.online/batch_search \
  -H 'Content-Type: application/json' \
  -d '{"queries":["transformer attention"],"max_results":2,"date_to":"2023-12-31","sort_by":"importance"}'
```

The `search` CLI wrapper baked into Harbor sandboxes (`/app/search`) reads `/app/search_api_url.txt` (set by `run.sh`) and `/app/paper_cutoff.txt` (set by `benchmark_pass_at_k.py` when reviewing). It now sets `User-Agent: Mozilla/5.0 (compatible; HarborSearchCLI/1.0)` because Cloudflare returns **403 / error 1010** for the default `Python-urllib/X.Y` UA.

---

## Review API Usage (async submit + poll)

The review API is **async**. The old synchronous `POST /review` is kept for back-compat but Cloudflare's 100s tunnel timeout will kill anything using it — use submit + poll.

### 1. Submit
```bash
curl -X POST https://review-api.eigenlabs.online/review/start \
  -H 'Content-Type: application/json' \
  -d '{"latex_content":"...","title":"...","abstract":"..."}'
# → {"job_id":"abc123def456","status":"pending"}
```

### 2. Poll
```bash
curl https://review-api.eigenlabs.online/review/status/abc123def456
# → {"job_id":"...","status":"running","review_text":"","error":"",...}
# → eventually: {"status":"success","review_text":"### Summary..."}
```

Statuses: `pending` → `running` → `success` | `error` | `timeout` (`not_found` if `job_id` is unknown).

The agent-side `submit_for_review.sh` ([patches/submit_for_review.sh](patches/submit_for_review.sh)) already does submit + 15s polling automatically when `REVIEWER_MODE=api-external` (the default).

### Tunable env vars (inside sandbox or `.env`)
| Var | Default | Purpose |
|---|---|---|
| `REVIEWER_MODE` | `api-external` | mode: `api-external` / `subagent` / `ensemble` |
| `REVIEW_API_URL` | `https://review-api.eigenlabs.online` | base URL of review API |
| `REVIEW_POLL_INTERVAL` | `15` | seconds between polls |
| `REVIEW_POLL_MAX_SEC` | `2700` | client-side cap (45 min) |

---

## AI-Scientist v3 — Run a Job

```bash
ssh root@65.108.33.106
cd /root/ai-scientist-v3
export PATH="/root/venv/bin:$PATH"          # so `harbor` resolves
./run.sh ideas/idea_cpu_calibration.json \
    --model deepseek-v4-pro \
    --env e2b \
    --timeout 14400 \
    --use-upstream-agent
```

Use `--model deepseek-v4-pro` (or `deepseek-v4-flash`) — these are the only models accepted by the proxy in `.env`. Don't pass `anthropic/claude-opus-4-6` directly, the proxy will 400.

The runtime flow inside the sandbox:
1. Agent reads idea, calls `/app/search batch …` → search API tunnel → 928K papers.
2. Agent runs experiments, writes `latex/template.tex`.
3. Agent runs `bash scripts/submit_for_review.sh latex/template.tex`.
4. With `REVIEWER_MODE=api-external` (default), the script POSTs to `/review/start`, gets a `job_id`, polls `/review/status/{job_id}` every 15s.
5. Review text is saved to `submissions/v{N}_{ts}/reviewer_communications/response.md`.

Outputs land in `/root/ai-scientist-v3/jobs/<idea>__<timestamp>/`.

### Launch one job (helper)
[scripts/launch_one.sh](scripts/launch_one.sh) — drops a job in the background with proper PATH and writes to `/root/ai_scientist_<idea>.log`.

---

## Fixes Applied (vs gold)

These are baked into [patches/](patches/) — apply on top of the rclone gold.

1. **Cloudflare URLs** — replaced dead `http://171.226.34.64:54321` (review) and `http://171.226.34.64:54735` (search) with our tunnels in `Dockerfile.cpu`, `Dockerfile.gpu`, `run.sh`, `submit_for_review.sh`, `.claude/CLAUDE.md`, `.env`.
2. **Dashed hostnames** — `*_api.eigenlabs.online` → `*-api.eigenlabs.online`. Cloudflare's wildcard cert `*.eigenlabs.online` covers both, but Python's `ssl` module treats hostnames with `_` as invalid (ssl certificate hostname mismatch error). curl is lenient. **Use dashed everywhere in code; underscored remain as DNS aliases for back-compat.**
3. **Default reviewer mode = `api-external`** — `run.sh` forces `REVIEWER_MODE=api-external` and `REVIEW_API_URL=https://review-api.eigenlabs.online` into the sandbox `.env`. CLAUDE.md updated to say `api-external` is the default (not `ensemble`).
4. **`review_api.py` async endpoints** — added:
   - `POST /review/start` → `{job_id, status}` immediately
   - `GET /review/status/{job_id}` → current state
   - Old `POST /review` kept (synchronous) but only usable from same-machine clients
5. **`submit_for_review.sh` api-external branch** — now does submit + poll instead of one long blocking POST. Avoids Cloudflare's 100s tunnel timeout.
6. **`search` CLI User-Agent** — added `Mozilla/5.0 (compatible; HarborSearchCLI/1.0)`. Without this, Cloudflare returns 1010 to the agent's `urllib` requests.
7. **Harbor pinned to `0.7.0`** — gold's `_setup_environment` override is a no-op on `0.8+`. (Not a code patch — install-time pin.)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Agent inside sandbox says `/app/latex/template.tex does not exist` | Harbor too new (≥0.8) — `PassAtKTrial._setup_environment` no-ops | `pip install "harbor==0.7.0"` |
| Search CLI returns 403 / "error code 1010" | Default `Python-urllib` UA blocked by Cloudflare | Use [patches/search_cli.py](patches/search_cli.py) (sets Mozilla UA) |
| `SSL: CERTIFICATE_VERIFY_FAILED ... Hostname mismatch` from Python | Underscored hostname (`search_api...`) — Python's strict hostname check rejects `_` | Switch URL to dashed form (`search-api...`) |
| Review API call times out at 100s through tunnel (HTTP 524) | Cloudflare tunnel's 100s read timeout | Use async `/review/start` + poll, not synchronous `/review` |
| `Response 404` lines in review log | Harmless — claude-code installer's status probe | Ignore |
| `harbor: command not found` in `run.sh` | Venv not on PATH | `export PATH="/root/venv/bin:$PATH"` before `./run.sh` |
| Agent: `API Error: 400 The supported API model names are deepseek-v4-pro or deepseek-v4-flash` | Wrong `--model` flag | Use `--model deepseek-v4-pro` |
| `nohup: failed to run command './run.sh': Permission denied` | Lost exec bit during rsync | `chmod +x /root/ai-scientist-v3/run.sh /root/ai-scientist-v3/scripts/*.sh` |

---

## Files in This Directory

```
harbor_infra/
├── RUNBOOK.md                     # this file
├── patches/
│   ├── review_api.py              # async submit + poll + sync /review for backcompat
│   ├── search_cli.py              # search CLI with Mozilla UA (deploy as `search` in skills/search-papers/)
│   ├── submit_for_review.sh       # api-external mode now polls instead of blocking
│   └── run.sh                     # ai-scientist-v3 launcher with api-external defaults
├── configs/
│   └── cloudflared_config.yml     # tunnel ingress: dashed + underscored hostnames
└── scripts/
    ├── start_remote.sh            # cold start: search API + review API + tunnel
    ├── restart_review.sh          # restart only review API
    ├── restart_cf.sh              # restart only cloudflared tunnel
    └── launch_one.sh              # launch one ai-scientist run with the right env
```

Source of truth for the rest of the gold trees stays in Google Drive — only the deltas live here.

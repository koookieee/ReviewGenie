# ReviewGenie self-host

`docker compose up` and you have:
- a **Search API** (928K-paper arXiv search via Gemini embeddings)
- a **Review API** (paper-review pipeline backed by E2B sandboxes + Claude Code)

Then `git clone` our **AI Scientist v3 fork** and point it at both — three env vars, one command.

---

## Prerequisites

| Need | Get it from |
|---|---|
| Docker + docker-compose v2 | [docs.docker.com](https://docs.docker.com/get-docker/) |
| Gemini API key | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| E2B API key | [e2b.dev/dashboard](https://e2b.dev/dashboard) |
| Anthropic-compatible LLM key | Anthropic, or DeepSeek's anthropic-compatible proxy |
| ngrok or cloudflared (only if you want sandboxes to use your local search) | [ngrok.com](https://ngrok.com) / `brew install cloudflared` |

You can skip the last row if you're fine with E2B sandboxes calling our **hosted** Search API instead of your local one (default). The Review API stays local either way — it only needs to be reachable from your AI Scientist client.

---

## Step 1 — Start the APIs

```bash
git clone https://github.com/koookieee/ReviewGenie.git
cd ReviewGenie/selfhost

cp .env.example .env
# Edit .env — fill GEMINI_API_KEY, E2B_API_KEY, ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL if not vanilla Anthropic)

docker compose up -d
```

First start downloads the ~10 GB arXiv index from HuggingFace into a named volume (5–15 min, **once**). Subsequent starts are instant.

Sanity check:
```bash
curl http://localhost:8081/health   # search-api  → {"status":"ok",...}
curl http://localhost:8082/health   # review-api  → {"status":"ok"}
```

That's the APIs. **Done.** You can stop here and use them directly via curl/code.

---

## Step 2 (optional) — Expose your local search-api publicly

Skip this if you want sandboxes to use our hosted search-api (default in step 3).

E2B sandboxes run on E2B's network, **not** your docker-compose network. They need a public URL to reach your local search-api. Pick one:

```bash
# Cloudflare Quick Tunnel (no signup, throwaway URL):
docker run --rm -d --network host cloudflare/cloudflared:latest \
    tunnel --url http://localhost:8081
docker logs <container> 2>&1 | grep trycloudflare.com   # → your URL

# Or ngrok:
ngrok http 8081
```

Save the resulting URL — you'll set it as `SEARCH_PUBLIC_URL` in step 3.

---

## Step 3 — Run AI Scientist v3 against your APIs

```bash
git clone -b harbor-apis https://github.com/koookieee/ai-scientist-v3.git
cd ai-scientist-v3
chmod +x run.sh scripts/*.sh

cat > .env <<EOF
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic     # or https://api.anthropic.com
ANTHROPIC_API_KEY=<your-key>
ANTHROPIC_AUTH_TOKEN=<your-key>
E2B_API_KEY=<your-key>
GEMINI_API_KEY=<your-key>

# Where review jobs go. Local docker by default; hosted endpoint if you skipped step 1.
REVIEW_API_URL=http://localhost:8082

# Where in-sandbox /app/search calls go.
# - If you did step 2: SEARCH_PUBLIC_URL=<your-tunnel-url>
# - If you skipped:    SEARCH_PUBLIC_URL=https://search-api.eigenlabs.online
SEARCH_PUBLIC_URL=https://search-api.eigenlabs.online
EOF

# pip-install harbor (pinned 0.7.0 in pyproject.toml) into a venv
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

./run.sh ideas/idea_lww_memory_conflict_resolution.json --model deepseek-v4-pro --env e2b --use-upstream-agent
```

Defaults are already wired — `REVIEWER_MODE=api-external`, our search CLI baked into the sandbox image, etc. The fork's `harbor-apis` branch is what you want; `main` is upstream and not wired up.

Job results: `ai-scientist-v3/jobs/<idea>__<timestamp>/`

---

## Zero-infra mode (skip steps 1 & 2)

If you don't want to run anything locally, point AI Scientist at our hosted endpoints directly. Skip everything above except step 3, and use these:

```bash
REVIEW_API_URL=https://review-api.eigenlabs.online
SEARCH_PUBLIC_URL=https://search-api.eigenlabs.online
```

Hosted endpoints have a small per-IP rate limit and we won't keep them up forever. For real workloads, run your own.

---

## Direct API usage (no AI Scientist)

### Search

```bash
curl -X POST http://localhost:8081/batch_search \
  -H 'Content-Type: application/json' \
  -d '{"queries":["transformer attention"],"max_results":5,"sort_by":"importance"}'
```

### Review (async submit + poll)

```bash
JOB_ID=$(curl -s -X POST http://localhost:8082/review/start \
  -H 'Content-Type: application/json' \
  -d '{"latex_content":"\\documentclass{article}...","title":"...","abstract":"..."}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

# Poll (typical: 1–3 min after first cold start which builds the E2B template)
while true; do
    R=$(curl -s "http://localhost:8082/review/status/$JOB_ID")
    S=$(echo "$R" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    echo "$S"
    case "$S" in success|error|timeout) echo "$R" | python3 -m json.tool; break;; esac
    sleep 15
done
```

---

## Common pitfalls

| Symptom | Fix |
|---|---|
| `harbor: command not found` when running `./run.sh` | Activate the venv: `source .venv/bin/activate` |
| Review API call returns HTTP 524 through a tunnel | Cloudflare's 100 s read limit. Use the async `/review/start` + poll, never the synchronous `/review`. |
| `SSL: CERTIFICATE_VERIFY_FAILED ... Hostname mismatch` from Python | Use the **dashed** hostnames (`search-api.eigenlabs.online`), not underscored. Python's strict SSL rejects underscores. |
| Search returns `error code 1010` (Cloudflare) | The default `urllib` UA is bot-blocked. Our search CLI sets a Mozilla UA. If you call from custom code, set `User-Agent: Mozilla/5.0 (compatible; HarborSearchCLI/1.0)`. |
| First review takes 5–10 min | E2B template is being built. Subsequent reviews reuse it (~1–2 min). |
| Review agent says "files not present in /app" | Wrong harbor version. We pin `harbor==0.7.0` in pyproject. `harbor>=0.8` silently breaks file uploads. |

---

## Where things live

```
selfhost/
├── docker-compose.yml         # both services + healthchecks + named volume
├── .env.example
├── search-api/
│   ├── Dockerfile             # python:3.12-slim + arxiv-search-kit==0.2.4 + torch CPU
│   ├── search_api.py          # bundled (the wheel ships only the library)
│   └── entrypoint.sh          # downloads or reuses the index
└── review-api/
    ├── Dockerfile             # python:3.12-slim + harbor==0.7.0 + pandoc
    ├── requirements.txt       # versions pinned to the verified production deployment
    ├── review_api.py          # async POST /review/start + GET /review/status/{id}
    ├── benchmark_pass_at_k.py
    ├── latex_to_markdown.py
    ├── prompts/               # reviewer instruction template
    └── skills/search-papers/  # SKILL.md + search CLI uploaded into each sandbox
```

The AI Scientist fork (`koookieee/ai-scientist-v3:harbor-apis`) adds:
- `REVIEWER_MODE=api-external` branch in `submit_for_review.sh`
- `/app/search` CLI in `.claude/skills/search-papers/`
- Defensive `set -u` guards in `run.sh` for non-GitLab runs
- harbor pin to 0.7.0

See [the fork's diff vs upstream](https://github.com/koookieee/ai-scientist-v3/compare/main...harbor-apis) for the exact changes.

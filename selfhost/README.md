# ReviewGenie self-host

Two paper-review APIs and an AI Scientist v3 fork wired to use them.

- **Search API** — 928K-paper arXiv search (Gemini embeddings)
- **Review API** — async paper review backed by E2B sandboxes + Claude Code
- **AI Scientist v3 fork** — autonomous research agent that uses both: [`koookieee/ai-scientist-v3:harbor-apis`](https://github.com/koookieee/ai-scientist-v3/tree/harbor-apis)

---

## Path A — Zero infra (~5 min)

Use the hosted endpoints. No Docker, no tunnels.

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

REVIEW_API_URL=https://review-api.eigenlabs.online
SEARCH_PUBLIC_URL=https://search-api.eigenlabs.online
EOF

python3 -m venv .venv && . .venv/bin/activate
pip install -e .

./run.sh ideas/idea_lww_memory_conflict_resolution.json --model deepseek-v4-pro --env e2b --use-upstream-agent
```

Job results land in `ai-scientist-v3/jobs/<idea>__<timestamp>/`.

Hosted endpoints have a small per-IP rate limit and aren't a permanent service. For real workloads use Path B.

### Or just call the APIs directly

```bash
# Search
curl -X POST https://search-api.eigenlabs.online/batch_search \
  -H 'Content-Type: application/json' \
  -d '{"queries":["transformer attention"],"max_results":5,"sort_by":"importance"}'

# Review — async submit + poll
JOB_ID=$(curl -s -X POST https://review-api.eigenlabs.online/review/start \
  -H 'Content-Type: application/json' \
  -d '{"latex_content":"\\documentclass{article}...","title":"...","abstract":"..."}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

while true; do
    R=$(curl -s "https://review-api.eigenlabs.online/review/status/$JOB_ID")
    S=$(echo "$R" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    echo "$S"
    case "$S" in success|error|timeout) echo "$R" | python3 -m json.tool; break;; esac
    sleep 15
done
```

First review on a cold E2B template takes 5–10 min; later ones are 1–3 min.

---

## Path B — Self-host both APIs

For when you don't want our endpoints in the loop. Roughly 30 min including the index download.

### How the pieces talk

`./run.sh` launches the AI Scientist agent inside an E2B sandbox. From inside that sandbox, the agent calls **both** APIs over the public internet:

```
your laptop
   │ ./run.sh
   ▼
   harbor ──► E2B sandbox (research agent)
                  │ /app/search   ──► SEARCH_PUBLIC_URL  (must be public)
                  │ submit_review ──► REVIEW_API_URL     (must be public)
                                            │ (review-api spawns its own E2B sandbox per request)
                                            ▼
                                          Reviewer agent ──► SEARCH_PUBLIC_URL
```

`localhost:8082` does not work — that's the sandbox's own loopback. Both URLs must resolve from the public internet.

### Prerequisites

| Need | Get it from |
|---|---|
| Docker + docker-compose v2 | [docs.docker.com](https://docs.docker.com/get-docker/) |
| Gemini API key | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| E2B API key | [e2b.dev/dashboard](https://e2b.dev/dashboard) |
| Anthropic-compatible LLM key | Anthropic, or DeepSeek's anthropic-compatible proxy |
| ngrok or cloudflared | [ngrok.com](https://ngrok.com) / `brew install cloudflared` |

### Step 1 — Start the APIs

```bash
git clone https://github.com/koookieee/ReviewGenie.git
cd ReviewGenie/selfhost

cp .env.example .env
# Fill GEMINI_API_KEY, E2B_API_KEY, ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL if not vanilla Anthropic)

docker compose up -d
```

First start downloads the ~10 GB arXiv index from HuggingFace into a named volume (5–15 min, **once**). Subsequent starts are instant.

```bash
curl http://localhost:8081/health   # search-api  → {"status":"ok",...}
curl http://localhost:8082/health   # review-api  → {"status":"ok"}
```

### Step 2 — Expose both APIs publicly

```bash
# Search API tunnel
docker run --rm -d --name cf-search --network host cloudflare/cloudflared:latest \
    tunnel --url http://localhost:8081
docker logs cf-search 2>&1 | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1

# Review API tunnel
docker run --rm -d --name cf-review --network host cloudflare/cloudflared:latest \
    tunnel --url http://localhost:8082
docker logs cf-review 2>&1 | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1
```

Or with ngrok: `ngrok http 8081` and `ngrok http 8082` in separate terminals.

Save both URLs.

### Step 3 — Run AI Scientist v3 against your tunnels

Same as **Path A**, but use your saved tunnel URLs for `REVIEW_API_URL` and `SEARCH_PUBLIC_URL` instead of the hosted ones.

```bash
git clone -b harbor-apis https://github.com/koookieee/ai-scientist-v3.git
cd ai-scientist-v3
chmod +x run.sh scripts/*.sh

cat > .env <<EOF
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_API_KEY=<your-key>
ANTHROPIC_AUTH_TOKEN=<your-key>
E2B_API_KEY=<your-key>
GEMINI_API_KEY=<your-key>

REVIEW_API_URL=<your-cf-review-url>
SEARCH_PUBLIC_URL=<your-cf-search-url>
EOF

python3 -m venv .venv && . .venv/bin/activate
pip install -e .

./run.sh ideas/idea_lww_memory_conflict_resolution.json --model deepseek-v4-pro --env e2b --use-upstream-agent
```

Defaults inside the fork (`REVIEWER_MODE=api-external`, our `/app/search` CLI baked into the sandbox image, harbor pinned to 0.7.0) are already wired — you only set the env vars above.

---

## Common pitfalls

| Symptom | Fix |
|---|---|
| `harbor: command not found` when running `./run.sh` | Activate the venv: `source .venv/bin/activate` |
| Review API call returns HTTP 524 through a tunnel | Cloudflare's 100 s read limit. Always use async `/review/start` + poll, never synchronous `/review`. |
| `SSL: CERTIFICATE_VERIFY_FAILED ... Hostname mismatch` from Python | Use **dashed** hostnames (`search-api.eigenlabs.online`), not underscored. Python's strict SSL rejects underscores. |
| Search returns `error code 1010` (Cloudflare) | Default `urllib` UA is bot-blocked. Our CLI sets a Mozilla UA; if you call from custom code, set `User-Agent: Mozilla/5.0 (compatible; HarborSearchCLI/1.0)`. |
| First review takes 5–10 min | E2B template is being built. Subsequent reviews reuse it (~1–2 min). |
| Reviewer agent says "files not present in /app" | Wrong harbor version. Pin `harbor==0.7.0`. `harbor>=0.8` silently breaks file uploads. |

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

The AI Scientist fork ([`koookieee/ai-scientist-v3:harbor-apis`](https://github.com/koookieee/ai-scientist-v3/tree/harbor-apis)) adds:
- `REVIEWER_MODE=api-external` branch in `submit_for_review.sh` (default in this fork)
- `/app/search` CLI in `.claude/skills/search-papers/`
- Defensive `set -u` guards in `run.sh` for non-GitLab runs
- harbor pin to 0.7.0

See [the fork's diff vs upstream](https://github.com/koookieee/ai-scientist-v3/compare/main...harbor-apis) for the exact changes.

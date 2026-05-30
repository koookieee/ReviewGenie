# ReviewGenie — self-host

`docker compose up` and you have a working Search API + Review API. AI Scientist v3 (separate repo) is one `git clone && ./run.sh` away.

## Quick start

```bash
git clone https://github.com/koookieee/ReviewGenie.git
cd ReviewGenie/selfhost
cp .env.example .env
# fill in GEMINI_API_KEY, E2B_API_KEY, ANTHROPIC_API_KEY (see .env.example for details)
docker compose up -d
```

First start downloads the ~10 GB arXiv index from HuggingFace (5–15 min, one time).
After that, restarts are instant.

```bash
# Sanity check
curl http://localhost:8081/health    # search-api
curl http://localhost:8082/health    # review-api
```

## What you get

| Service | Port | What it does |
|---|---|---|
| `search-api` | 8081 | 928K arXiv papers, semantic search via Gemini embeddings |
| `review-api` | 8082 | Spawns an E2B sandbox running Claude Code; returns a paper review |

Both APIs are internal to your machine by default. The Review API's E2B sandboxes call back into the Search API — they need a **public URL** for it (E2B sandboxes don't share your docker network). The default in `docker-compose.yml` points sandboxes at our hosted Search API (`https://search-api.eigenlabs.online`) so you don't have to deal with tunneling. To use your own local Search API from sandboxes, expose it via Cloudflare Tunnel or ngrok and set `SEARCH_PUBLIC_URL` accordingly — see [.env.example](.env.example).

## API usage

### Search API
```bash
curl -X POST http://localhost:8081/batch_search \
  -H 'Content-Type: application/json' \
  -d '{
    "queries": ["transformer attention mechanisms"],
    "max_results": 5,
    "sort_by": "importance"
  }'
```

### Review API (async submit + poll)
```bash
# 1. Submit
JOB_ID=$(curl -s -X POST http://localhost:8082/review/start \
    -H 'Content-Type: application/json' \
    -d '{"latex_content": "...your tex...", "title": "...", "abstract": "..."}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

# 2. Poll (5–10 min for first review on cold E2B template, ~1–2 min after)
while true; do
    R=$(curl -s "http://localhost:8082/review/status/$JOB_ID")
    S=$(echo "$R" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    echo "status=$S"
    case "$S" in success|error|timeout) echo "$R" | python3 -m json.tool; break;; esac
    sleep 15
done
```

## Wiring AI Scientist v3

```bash
# Clone the upstream
git clone https://github.com/findalexli/ai-scientist-v3.git
cd ai-scientist-v3

# Point it at your local APIs (or our hosted ones — same shape, just different URLs)
cat >> .env <<EOF
SEARCH_PUBLIC_URL=http://localhost:8081
REVIEWER_MODE=api-external
REVIEW_API_URL=http://localhost:8082
EOF

./run.sh ideas/idea_cpu_calibration.json --model deepseek-v4-pro --env e2b --use-upstream-agent
```

The patches in [`../harbor_infra/patches/`](../harbor_infra/) (search CLI Mozilla User-Agent, async submit/poll in `submit_for_review.sh`, `api-external` reviewer-mode default) are needed on top of upstream `findalexli/ai-scientist-v3`. See [`../harbor_infra/README.md`](../harbor_infra/README.md) for the apply-on-top instructions.

## Use our hosted endpoints (zero setup)

If you don't want to run anything locally:

| | Hosted URL |
|---|---|
| Search | `https://search-api.eigenlabs.online` |
| Review | `https://review-api.eigenlabs.online` |

Same protocol, same payloads. Hosted instance has a small per-IP rate limit; for sustained use, run your own.

## Layout

```
selfhost/
├── docker-compose.yml        # both services, healthchecks, named volume for the index
├── .env.example              # all required + optional env vars, documented
├── search-api/
│   ├── Dockerfile            # python:3.12-slim + arxiv-search-kit==0.2.4
│   ├── search_api.py         # bundled (the wheel ships only the library)
│   └── entrypoint.sh         # downloads index from HF on first run; runs server
└── review-api/
    ├── Dockerfile            # python:3.12-slim + harbor==0.7.0 + pandoc
    ├── requirements.txt      # versions pinned to verified production deployment
    ├── review_api.py         # async /review/start + /review/status/{id}
    ├── benchmark_pass_at_k.py
    ├── latex_to_markdown.py
    ├── prompts/              # reviewer instruction template
    └── skills/search-papers/ # SKILL.md + search CLI uploaded into each sandbox
```

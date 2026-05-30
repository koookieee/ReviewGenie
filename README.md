# ReviewGenie

Open-source paper-review pipeline. Three components:

| Component | What | Where |
|---|---|---|
| **Search API** | 928K-paper arXiv semantic search (Gemini embeddings) | [`selfhost/search-api/`](selfhost/search-api/) |
| **Review API** | Async paper review via E2B sandbox + Claude Code | [`selfhost/review-api/`](selfhost/review-api/) |
| **AI Scientist v3 integration** | Autonomous research agent wired to both APIs | [`koookieee/ai-scientist-v3:harbor-apis`](https://github.com/koookieee/ai-scientist-v3/tree/harbor-apis) (fork) |

## Quick start (3 minutes)

Self-host both APIs locally, then point AI Scientist at them — or skip the self-host and use our hosted endpoints.

**→ See [`selfhost/README.md`](selfhost/README.md) for the full guide.**

## Hosted endpoints (zero infra)

```
Search API:  https://search-api.eigenlabs.online
Review API:  https://review-api.eigenlabs.online
```

Same protocol as the self-hosted version. Small per-IP rate limit; for sustained use, run your own.

## Repo layout

```
ReviewGenie/
├── README.md              ← this file
├── selfhost/              ← Docker stack (search-api + review-api), one-stop guide
│   ├── README.md          ← user-facing 3-step guide
│   ├── docker-compose.yml
│   ├── .env.example
│   ├── search-api/
│   └── review-api/
└── harbor_infra/          ← ops notes for our hosted deployment (internal)
    └── README.md
```

## License

MIT.

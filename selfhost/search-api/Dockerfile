FROM python:3.12-slim

# System deps for arxiv-search-kit (lancedb needs libgomp; httpx needs SSL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install arxiv-search-kit + the deps not declared in its wheel:
#   - aiohttp: search_api.py uses it
#   - huggingface_hub: index download + lancedb
#   - google-genai: /query_paper handler
#   - torch (CPU wheel) + transformers + adapters: arxiv_search_kit.index.embedder
#                        unconditionally imports them at module load even when using
#                        Gemini embeddings. CPU wheel keeps the image small (~200MB
#                        vs ~3GB for CUDA).
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.12.0" && \
    pip install --no-cache-dir \
        "arxiv-search-kit==0.2.4" \
        "aiohttp==3.13.5" \
        "huggingface_hub>=0.20" \
        "google-genai==2.7.0" \
        "transformers==4.57.6" \
        "adapters==1.3.0"

# Bundle the API server explicitly (the wheel ships only the library).
COPY search_api.py /app/search_api.py

# The index lives in a volume so the ~10 GB download persists across restarts.
ENV ARXIV_SEARCH_KIT_CACHE=/data/arxiv_search_kit \
    HF_HOME=/data/hf

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8081

# /health is reachable as soon as the server binds; long start_period covers first-run index download.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=900s \
    CMD curl -fsS http://localhost:8081/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]

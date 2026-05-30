#!/bin/bash
# Start search API, review API, and cloudflared tunnel on remote
set +e

# Verify search index exists
if [ ! -d /root/.cache/arxiv_search_kit/arxiv_gemini_index ]; then
    echo "ERROR: search index missing at /root/.cache/arxiv_search_kit/arxiv_gemini_index"
    exit 1
fi

# Verify venv
if [ ! -x /root/venv/bin/python3 ]; then
    echo "ERROR: /root/venv/bin/python3 missing"
    exit 1
fi

# Start search API
cd /root/arxiv-search-kit
nohup /root/venv/bin/python3 -u search_api.py --port 8081 \
    --gemini-api-key $GEMINI_API_KEY \
    > /root/search_api.log 2>&1 &
SEARCH_PID=$!
echo "search_api PID=$SEARCH_PID"

# Start review API
cd /root/HarborTrajectoryGen_v2
nohup /root/venv/bin/python3 -u review_api.py --port 8082 \
    > /root/review_api.log 2>&1 &
REVIEW_PID=$!
echo "review_api PID=$REVIEW_PID"

# Start cloudflared tunnel
nohup cloudflared tunnel run search_api > /root/cloudflared.log 2>&1 &
TUNNEL_PID=$!
echo "cloudflared PID=$TUNNEL_PID"

sleep 6

echo "=== process check ==="
pgrep -fa search_api | head -3
pgrep -fa review_api | head -3
pgrep -fa cloudflared | head -3

echo "=== local health ==="
curl -s -m 5 http://localhost:8081/health; echo
curl -s -m 5 http://localhost:8082/health; echo

exit 0

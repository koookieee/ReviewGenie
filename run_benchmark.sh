#!/bin/bash
# Run Paper Reviewer Benchmark with Claude Opus 4.6
#
# Usage:
#   bash run_benchmark.sh                          # all tasks
#   bash run_benchmark.sh --max-tasks 3            # first 3 tasks
#   bash run_benchmark.sh --tasks 2603.10165v1     # specific task
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Install deps if needed
pip install -q pyyaml python-dotenv loguru openai 2>/dev/null

echo "=== Paper Reviewer Benchmark ==="
echo "  Model: Claude Opus 4.6 (default in Claude Code)"
echo "  Environment: E2B"
echo "  Search API: $SEARCH_PUBLIC_URL"
echo ""

python3 "$SCRIPT_DIR/benchmark.py" "$@"

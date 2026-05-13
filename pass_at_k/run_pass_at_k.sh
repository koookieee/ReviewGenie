#!/bin/bash
# run_pass_at_k.sh — Full pass@K experiment on remote machine
#
# Orchestrates:
#   1. Install dependencies
#   2. Fetch 100 random papers from HuggingFace
#   3. Run K=4 review attempts per paper using Kimi K2.5 via Fireworks
#   4. Analyze pass@K results
#
# Usage:
#   bash run_pass_at_k.sh                    # full pipeline
#   bash run_pass_at_k.sh --skip-fetch       # skip paper download (if already done)
#   bash run_pass_at_k.sh --analyze-only     # only run analysis
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Paths
DATA_DIR="/root/data/pass_at_k_papers"
RESULTS_DIR="/root/pass_at_k/results"
TRIALS_DIR="/root/pass_at_k/trials"

# Parse arguments
SKIP_FETCH=false
ANALYZE_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-fetch) SKIP_FETCH=true ;;
        --analyze-only) ANALYZE_ONLY=true ;;
    esac
done

# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------

# Load .env from project root
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Load pass@K specific .env (overrides)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Override for Kimi via Fireworks
export ANTHROPIC_BASE_URL="https://api.fireworks.ai/inference/v1"
export ANTHROPIC_API_KEY="$FIREWORKS_API_KEY"

# Judge model
export JUDGE_MODEL="gemini-3.1-pro-preview"

echo "=============================================="
echo " PASS@K EXPERIMENT"
echo "=============================================="
echo "  Model: Kimi K2.5 (Fireworks)"
echo "  Judge: $JUDGE_MODEL"
echo "  K: 4"
echo "  Papers: 100"
echo "  Data dir: $DATA_DIR"
echo "  Results: $RESULTS_DIR"
echo "  ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"
echo ""

# -----------------------------------------------------------------------
# Validate environment
# -----------------------------------------------------------------------

if [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$FIREWORKS_API_KEY" ]; then
    echo "ERROR: FIREWORKS_API_KEY must be set"
    echo "  Set it in $SCRIPT_DIR/.env as FIREWORKS_API_KEY=fw_..."
    exit 1
fi

if [ -z "$E2B_API_KEY" ]; then
    echo "ERROR: E2B_API_KEY must be set"
    exit 1
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY must be set"
    exit 1
fi

# -----------------------------------------------------------------------
# Install dependencies
# -----------------------------------------------------------------------

echo "Installing dependencies..."
pip install -q pyyaml python-dotenv loguru openai pandas pyarrow huggingface_hub numpy 2>/dev/null
echo "Done."
echo ""

# -----------------------------------------------------------------------
# Step 1: Fetch papers
# -----------------------------------------------------------------------

if [ "$ANALYZE_ONLY" = true ]; then
    echo "Skipping to analysis..."
elif [ "$SKIP_FETCH" = true ]; then
    echo "Skipping paper fetch (--skip-fetch)"
    echo ""
else
    echo "Step 1: Fetching 100 random papers..."
    python3 "$SCRIPT_DIR/fetch_papers.py" \
        --output-dir "$DATA_DIR" \
        --num-papers 100 \
        --seed 42 \
        --converter "$PROJECT_DIR/latex_to_markdown.py"
    echo ""
fi

# -----------------------------------------------------------------------
# Step 2: Run benchmark
# -----------------------------------------------------------------------

if [ "$ANALYZE_ONLY" = false ]; then
    echo "Step 2: Running pass@K benchmark (K=4, 100 papers = 400 trials)..."
    python3 "$SCRIPT_DIR/benchmark_pass_at_k.py" \
        --data-dir "$DATA_DIR" \
        --results-dir "$RESULTS_DIR" \
        --trials-dir "$TRIALS_DIR" \
        --k 4 \
        --max-concurrent 4
    echo ""
fi

# -----------------------------------------------------------------------
# Step 3: Analyze results
# -----------------------------------------------------------------------

echo "Step 3: Analyzing pass@K results..."
python3 "$SCRIPT_DIR/analyze_pass_at_k.py" \
    --results-dir "$RESULTS_DIR" \
    --threshold 0.5 \
    --k-values 1 2 3 4

echo ""
echo "=============================================="
echo " EXPERIMENT COMPLETE"
echo "=============================================="
echo "  Results: $RESULTS_DIR"
echo "  Analysis: $RESULTS_DIR/pass_at_k_analysis.json"
echo ""
echo "To copy results locally:"
echo "  scp -P 17909 -r root@142.127.68.223:$RESULTS_DIR ./pass_at_k_results/"
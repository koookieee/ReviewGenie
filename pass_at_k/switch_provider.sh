#!/usr/bin/env bash
# switch_provider.sh — one-shot provider switcher for the pass@K stream_proxy.
#
# Usage:
#   switch_provider.sh <entry-name>            # switch to a named provider
#   switch_provider.sh --list                  # list available entries
#   switch_provider.sh --status                # show current running config
#
# What it does:
#   1. Reads pass_at_k/providers.yaml and finds the named entry
#   2. Looks up the API key from the entry's api_key_env (required in /root/.env)
#   3. Writes PROXY_BASE_URL / PROXY_MODEL / PROXY_API_KEY / PROXY_MAX_TOKENS_CAP
#      into /root/.env (idempotent — updates existing keys, appends new)
#   4. Kills the old stream_proxy and relaunches with the new env
#   5. Runs a non-streaming + streaming end-to-end ping against the proxy
#      to verify the upstream works before returning success.
#
# Exits 0 only when E2E verification passes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDERS_YAML="$SCRIPT_DIR/providers.yaml"
ENV_FILE="${PROXY_ENV_FILE:-/root/.env}"
PROXY_SCRIPT="${PROXY_SCRIPT:-/root/stream_proxy.py}"
PROXY_PORT="${PROXY_PORT:-8861}"

# ---------------------------------------------------------------------------
# Helpers

die() { echo "error: $*" >&2; exit 1; }

# Parse providers.yaml using python (no yaml dependency needed — tiny subset).
yaml_entries() {
    PROVIDERS_YAML="$PROVIDERS_YAML" python3 - <<'PY'
import re, sys, os
path = os.environ["PROVIDERS_YAML"]
text = open(path).read()
for m in re.finditer(r"^([a-zA-Z0-9_\-]+):\s*$", text, re.MULTILINE):
    print(m.group(1))
PY
}

yaml_get() {
    # yaml_get <entry> <field>
    PROVIDERS_YAML="$PROVIDERS_YAML" ENTRY="$1" FIELD="$2" python3 - <<'PY'
import os, re, sys
path = os.environ["PROVIDERS_YAML"]
entry = os.environ["ENTRY"]
field = os.environ["FIELD"]
text = open(path).read()

# Tiny YAML walker: find `entry:` then read indented `field:` lines until next top-level key.
lines = text.splitlines()
in_entry = False
value = None
for i, line in enumerate(lines):
    stripped = line.rstrip()
    if re.match(rf"^{re.escape(entry)}:\s*$", stripped):
        in_entry = True
        continue
    if in_entry:
        # End of entry = a line that starts at column 0 and is not blank / comment
        if stripped and not line.startswith(" ") and not line.startswith("#"):
            break
        m = re.match(rf"^\s+{re.escape(field)}:\s*(.*)$", line)
        if m:
            val = m.group(1).strip()
            if val == "|":
                # Multiline block scalar: collect following more-indented lines
                buf = []
                base_indent = len(line) - len(line.lstrip())
                for j in range(i + 1, len(lines)):
                    nl = lines[j]
                    if nl.strip() == "":
                        buf.append("")
                        continue
                    ni = len(nl) - len(nl.lstrip())
                    if ni <= base_indent:
                        break
                    buf.append(nl[base_indent + 2:])
                value = "\n".join(buf).rstrip()
            else:
                value = val.strip().strip('"').strip("'")
            break
if value is None:
    sys.exit(f"field not found: {entry}.{field}")
print(value)
PY
}

cmd_list() {
    echo "Available providers (from $PROVIDERS_YAML):"
    for e in $(yaml_entries); do
        model=$(yaml_get "$e" model 2>/dev/null || echo "?")
        echo "  $e  →  $model"
    done
}

cmd_status() {
    local pid
    pid=$(pgrep -f 'stream_proxy.py' || true)
    if [ -z "$pid" ]; then
        echo "stream_proxy is NOT running"
        return 0
    fi
    echo "stream_proxy PID=$pid"
    tr '\0' '\n' </proc/"$pid"/environ 2>/dev/null | grep -E '^PROXY_' || echo "(env not visible)"
    ps -p "$pid" -o args= 2>/dev/null | head -c 300; echo
}

# Upsert KEY=VALUE into an env file (replace if exists, append if not)
upsert_env() {
    local file="$1" key="$2" value="$3"
    if grep -qE "^${key}=" "$file" 2>/dev/null; then
        # Escape characters for sed replacement
        local escaped
        escaped=$(printf '%s\n' "$value" | sed 's/[\/&]/\\&/g')
        sed -i "s|^${key}=.*|${key}=${escaped}|" "$file"
    else
        printf '%s=%s\n' "$key" "$value" >>"$file"
    fi
}

# ---------------------------------------------------------------------------
# Main

[ ! -f "$PROVIDERS_YAML" ] && die "providers.yaml not found at $PROVIDERS_YAML"

case "${1:-}" in
    ""|-h|--help)
        sed -n '2,19p' "$0"
        exit 0
        ;;
    --list) cmd_list; exit 0 ;;
    --status) cmd_status; exit 0 ;;
esac

ENTRY="$1"
export PROVIDERS_YAML

# Validate entry exists
if ! yaml_entries | grep -qx "$ENTRY"; then
    echo "error: '$ENTRY' not in providers.yaml" >&2
    cmd_list
    exit 2
fi

BASE_URL=$(yaml_get "$ENTRY" base_url)
MODEL=$(yaml_get "$ENTRY" model)
API_KEY_ENV=$(yaml_get "$ENTRY" api_key_env)
MAX_TOKENS_CAP=$(yaml_get "$ENTRY" max_tokens_cap)

# Pull API key: prefer already-exported env, fall back to reading from .env file.
API_KEY="${!API_KEY_ENV:-}"
if [ -z "$API_KEY" ] && [ -f "$ENV_FILE" ]; then
    API_KEY=$(grep -E "^${API_KEY_ENV}=" "$ENV_FILE" | head -1 | cut -d= -f2- || true)
    API_KEY="${API_KEY%\"}"; API_KEY="${API_KEY#\"}"
fi
[ -z "$API_KEY" ] && die "API key env var '$API_KEY_ENV' is empty. Set it in $ENV_FILE or export it."

echo "==> switching proxy to: $ENTRY"
echo "    base_url  = $BASE_URL"
echo "    model     = $MODEL"
echo "    key_env   = $API_KEY_ENV (len=${#API_KEY})"
echo "    cap       = $MAX_TOKENS_CAP"

# 1. Persist config into .env (idempotent)
touch "$ENV_FILE"
upsert_env "$ENV_FILE" PROXY_BASE_URL "$BASE_URL"
upsert_env "$ENV_FILE" PROXY_MODEL "$MODEL"
upsert_env "$ENV_FILE" PROXY_API_KEY "$API_KEY"
upsert_env "$ENV_FILE" PROXY_MAX_TOKENS_CAP "$MAX_TOKENS_CAP"
echo "==> wrote config to $ENV_FILE"

# 2. Kill old proxy (if any)
old_pids=$(pgrep -f 'stream_proxy.py' || true)
if [ -n "$old_pids" ]; then
    echo "==> killing old proxy: $old_pids"
    kill $old_pids 2>/dev/null || true
    sleep 2
    remaining=$(pgrep -f 'stream_proxy.py' || true)
    [ -n "$remaining" ] && kill -9 $remaining 2>/dev/null || true
fi

# 3. Relaunch proxy
echo "==> launching new proxy on :$PROXY_PORT"
PROXY_BASE_URL="$BASE_URL" \
PROXY_MODEL="$MODEL" \
PROXY_API_KEY="$API_KEY" \
PROXY_MAX_TOKENS_CAP="$MAX_TOKENS_CAP" \
    nohup python3 "$PROXY_SCRIPT" "$API_KEY" "$PROXY_PORT" \
        > /root/proxy.log 2>&1 </dev/null &
disown
sleep 3

# 4. End-to-end verification
pid=$(pgrep -f 'stream_proxy.py' || true)
[ -z "$pid" ] && { echo; tail -40 /root/proxy.log; die "proxy did not start"; }
echo "==> proxy PID=$pid"

ping_url="http://localhost:$PROXY_PORT/v1/messages"
body='{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"reply exactly: OK"}],"max_tokens":2048}'
echo "==> verifying non-streaming..."
resp=$(curl -sS --max-time 60 -X POST "$ping_url" -H 'Content-Type: application/json' -H 'x-api-key: dummy' -d "$body" || true)
if ! echo "$resp" | grep -q '"role": "assistant"'; then
    echo "--- response ---"; echo "$resp" | head -c 600; echo
    echo "--- proxy log ---"; tail -30 /root/proxy.log
    die "non-streaming verification failed"
fi
echo "    OK"

echo "==> verifying streaming..."
stream_body='{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"reply exactly: OK"}],"max_tokens":2048,"stream":true}'
stream_resp=$(curl -sSN --max-time 60 -X POST "$ping_url" -H 'Content-Type: application/json' -H 'x-api-key: dummy' -d "$stream_body" || true)
if ! echo "$stream_resp" | grep -q 'event: message_stop'; then
    echo "--- response ---"; echo "$stream_resp" | head -c 600; echo
    echo "--- proxy log ---"; tail -30 /root/proxy.log
    die "streaming verification failed"
fi
echo "    OK"

echo "==> provider switch complete: $ENTRY"
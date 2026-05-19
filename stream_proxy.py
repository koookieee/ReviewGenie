"""
Anthropic-to-OpenAI proxy for Claude Code → Fireworks (Kimi K2.5).

Claude Code speaks Anthropic Messages API (/v1/messages).
Fireworks speaks OpenAI Chat Completions API (/v1/chat/completions).

This proxy:
1. Receives Anthropic Messages requests from Claude Code (via Harbor/E2B)
2. Converts to OpenAI Chat Completions format (including tools)
3. Sends to Fireworks with the Fireworks API key
4. Converts the OpenAI response back to Anthropic SSE streaming format

Based on AgentReviewer's stream_proxy.py, adapted for Fireworks.

Usage:
    python stream_proxy.py <fireworks_api_key> [port]
    python stream_proxy.py fw_7gHSHHxrTBXzJHxhPiSiWq 8861
"""

import json
import os
import sys
import uuid
from aiohttp import web, ClientSession

# All configuration is env-driven so switching providers is a single .env edit:
#   PROXY_BASE_URL        — e.g. https://api.deepinfra.com/v1/openai
#   PROXY_MODEL           — e.g. Qwen/Qwen3.6-35B-A3B
#   PROXY_API_KEY         — provider API key (argv[1] still accepted for compat)
#   PROXY_MAX_TOKENS_CAP  — hard cap, default 32768
FIREWORKS_URL = os.environ.get("PROXY_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_MODEL = os.environ.get("PROXY_MODEL", "accounts/fireworks/models/minimax-m2p7")
FIREWORKS_API_KEY = (
    sys.argv[1] if len(sys.argv) > 1 else
    os.environ.get("PROXY_API_KEY") or os.environ.get("FIREWORKS_API_KEY", "")
)
MAX_TOKENS_CAP = int(os.environ.get("PROXY_MAX_TOKENS_CAP", "32768"))


def anthropic_tools_to_openai(tools: list) -> list:
    """Convert Anthropic tool definitions to OpenAI format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return openai_tools


def anthropic_messages_to_openai(messages: list) -> list:
    """Convert Anthropic messages to OpenAI format."""
    openai_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            openai_msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            tool_calls = []
            tool_results = []

            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif btype == "tool_result":
                    tool_result_content = block.get("content", "")
                    if isinstance(tool_result_content, list):
                        tool_result_content = "\n".join(
                            b.get("text", "") for b in tool_result_content if isinstance(b, dict)
                        )
                    tool_results.append({
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(tool_result_content),
                    })

            if role == "assistant":
                m = {"role": "assistant"}
                if text_parts:
                    m["content"] = "\n".join(text_parts)
                if tool_calls:
                    m["tool_calls"] = tool_calls
                    if not text_parts:
                        m["content"] = None
                openai_msgs.append(m)
            elif tool_results:
                for tr in tool_results:
                    openai_msgs.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": tr["content"],
                    })
            else:
                openai_msgs.append({"role": role, "content": "\n".join(text_parts)})
    return openai_msgs


def openai_response_to_anthropic(data: dict, model: str) -> dict:
    """Convert OpenAI Chat Completions response to Anthropic Messages format."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = data.get("usage", {})

    content_blocks = []

    # Text content (skip thinking/reasoning which Kimi puts in reasoning_content)
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool calls
    for tc in message.get("tool_calls", []) or []:
        func = tc.get("function", {})
        try:
            inp = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            inp = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": func.get("name", ""),
            "input": inp,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    finish = choice.get("finish_reason", "end_turn")
    stop_reason = "tool_use" if finish == "tool_calls" else "end_turn" if finish == "stop" else finish

    return {
        "id": data.get("id", f"msg_{uuid.uuid4().hex[:12]}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    try:
        body_json = json.loads(body) if body else {}
    except json.JSONDecodeError:
        body_json = {}

    is_streaming = body_json.get("stream", False)
    model = body_json.get("model", "")
    anthropic_tools = body_json.get("tools", [])
    anthropic_messages = body_json.get("messages", [])
    anthropic_system = body_json.get("system", "")

    # Convert Anthropic → OpenAI
    max_tokens = min(body_json.get("max_tokens", 8192), MAX_TOKENS_CAP)
    # Fireworks requires stream=true for max_tokens > 4096
    use_streaming = max_tokens > 4096

    # Build OpenAI messages, prepending system prompt if present
    openai_msgs = []
    if anthropic_system:
        # Anthropic system can be string or list of content blocks
        if isinstance(anthropic_system, str):
            openai_msgs.append({"role": "system", "content": anthropic_system})
        elif isinstance(anthropic_system, list):
            sys_text = "\n".join(
                b.get("text", "") for b in anthropic_system if isinstance(b, dict) and b.get("type") == "text"
            )
            if sys_text:
                openai_msgs.append({"role": "system", "content": sys_text})
    openai_msgs.extend(anthropic_messages_to_openai(anthropic_messages))

    openai_request = {
        "model": FIREWORKS_MODEL,
        "messages": openai_msgs,
        "max_tokens": max_tokens,
        "temperature": body_json.get("temperature", 1.0),
        "stream": use_streaming,
    }

    if anthropic_tools:
        openai_request["tools"] = anthropic_tools_to_openai(anthropic_tools)
        openai_request["tool_choice"] = "auto"

    for key in ("top_p", "top_k"):
        if key in body_json:
            openai_request[key] = body_json[key]

    num_tools = len(anthropic_tools)
    print(f"[proxy] model={model} tools={num_tools} msgs={len(anthropic_messages)} stream={is_streaming}", flush=True)

    url = f"{FIREWORKS_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    }

    async with ClientSession() as session:
        async with session.post(url, json=openai_request, headers=headers) as resp:
            if resp.status != 200:
                resp_body = await resp.read()
                print(f"[proxy] Fireworks error: {resp.status} {resp_body[:500]}", flush=True)
                return web.Response(body=resp_body, status=resp.status,
                                    content_type="application/json")

            if use_streaming:
                # Collect streamed SSE chunks into a complete response
                content_text = ""
                tool_calls_map = {}  # index -> {id, name, arguments}
                finish_reason = "stop"
                usage = {}

                async for line in resp.content:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if "usage" in chunk:
                        usage = chunk["usage"]

                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        if delta.get("content"):
                            content_text += delta["content"]
                        for tc in (delta.get("tool_calls") or []):
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": "",
                                }
                            if tc.get("id"):
                                tool_calls_map[idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                tool_calls_map[idx]["name"] = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                tool_calls_map[idx]["arguments"] += tc["function"]["arguments"]

                # Reconstruct a standard OpenAI response
                tc_list = []
                for idx in sorted(tool_calls_map.keys()):
                    tc = tool_calls_map[idx]
                    tc_list.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    })

                data = {
                    "choices": [{"message": {
                        "content": content_text or None,
                        "tool_calls": tc_list or None,
                    }, "finish_reason": finish_reason}],
                    "usage": usage,
                }
            else:
                resp_body = await resp.read()
                try:
                    data = json.loads(resp_body)
                except json.JSONDecodeError:
                    return web.Response(body=resp_body, status=resp.status,
                                        content_type=resp.content_type)

            # Log response summary
            choice = data.get("choices", [{}])[0] if "choices" in data else {}
            msg = choice.get("message", {}) if isinstance(choice, dict) else {}
            tool_calls = msg.get("tool_calls") or []
            finish = choice.get("finish_reason", "?")
            content_len = len(msg.get("content", "") or "")
            print(f"[proxy] response: finish={finish} content={content_len}chars tool_calls={len(tool_calls)}", flush=True)
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    print(f"[proxy]   tool_call: {fn.get('name', '?')}({fn.get('arguments', '')[:100]})", flush=True)

            if "error" in data:
                print(f"[proxy] Fireworks error: {json.dumps(data)[:500]}", flush=True)
                return web.Response(body=json.dumps(data).encode(), status=400,
                                    content_type="application/json")

            # Convert OpenAI response → Anthropic format
            anthropic_resp = openai_response_to_anthropic(data, model)

            if not is_streaming:
                return web.Response(
                    body=json.dumps(anthropic_resp).encode(),
                    status=200,
                    content_type="application/json",
                )

            # Convert to SSE stream
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
            )
            await response.prepare(request)

            # message_start
            msg_start = {
                "type": "message_start",
                "message": {
                    **anthropic_resp,
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": anthropic_resp["usage"]["input_tokens"],
                        "output_tokens": 0,
                    },
                },
            }
            await response.write(f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode())

            # content blocks
            for idx, block in enumerate(anthropic_resp.get("content", [])):
                block_type = block.get("type", "text")

                if block_type == "text":
                    start_evt = {"type": "content_block_start", "index": idx,
                                 "content_block": {"type": "text", "text": ""}}
                    await response.write(f"event: content_block_start\ndata: {json.dumps(start_evt)}\n\n".encode())

                    text = block.get("text", "")
                    if text:
                        delta_evt = {"type": "content_block_delta", "index": idx,
                                     "delta": {"type": "text_delta", "text": text}}
                        await response.write(f"event: content_block_delta\ndata: {json.dumps(delta_evt)}\n\n".encode())

                    stop_evt = {"type": "content_block_stop", "index": idx}
                    await response.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode())

                elif block_type == "tool_use":
                    start_evt = {"type": "content_block_start", "index": idx,
                                 "content_block": {"type": "tool_use", "id": block.get("id", ""),
                                                   "name": block.get("name", ""), "input": {}}}
                    await response.write(f"event: content_block_start\ndata: {json.dumps(start_evt)}\n\n".encode())

                    inp = json.dumps(block.get("input", {}))
                    delta_evt = {"type": "content_block_delta", "index": idx,
                                 "delta": {"type": "input_json_delta", "partial_json": inp}}
                    await response.write(f"event: content_block_delta\ndata: {json.dumps(delta_evt)}\n\n".encode())

                    stop_evt = {"type": "content_block_stop", "index": idx}
                    await response.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode())

            # message_delta
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": anthropic_resp.get("stop_reason", "end_turn"),
                          "stop_sequence": None},
                "usage": {"output_tokens": anthropic_resp["usage"]["output_tokens"]},
            }
            await response.write(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode())

            await response.write(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode())

            await response.write_eof()
            return response


async def count_tokens_handler(request: web.Request) -> web.Response:
    """Stub for /v1/messages/count_tokens — Claude Code calls this."""
    body = await request.json()
    # Return a rough estimate
    msgs = body.get("messages", [])
    total_chars = sum(len(json.dumps(m)) for m in msgs)
    est_tokens = total_chars // 4
    return web.json_response({"input_tokens": est_tokens})


async def models_handler(request: web.Request) -> web.Response:
    """Return available models — report as Claude models so Claude Code accepts them."""
    return web.json_response({
        "data": [
            {"id": "claude-sonnet-4-6-20250514", "object": "model"},
            {"id": "claude-sonnet-4-5-20241022", "object": "model"},
        ]
    })


async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


app = web.Application()
app.router.add_post("/v1/messages", proxy_handler)
app.router.add_post("/v1/messages/count_tokens", count_tokens_handler)
app.router.add_get("/v1/models", models_handler)
app.router.add_head("/", health_handler)
app.router.add_get("/health", health_handler)

if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8861
    print(
        f"Anthropic→OpenAI proxy on :{port}  "
        f"upstream={FIREWORKS_URL}  model={FIREWORKS_MODEL}  "
        f"cap={MAX_TOKENS_CAP}"
    )
    web.run_app(app, port=port, print=None)
#!/usr/bin/env bash
# Quick coherence + throughput smoke against a running vLLM-Moet server.
# usage: smoke.sh [port] [model-name]
set -euo pipefail
PORT="${1:-8000}"
MODEL="${2:-deepseek-v4-flash}"

echo "=== completion (greedy, 128 tok) ==="
t0=$(date +%s.%N)
RESP=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 128,
       \"messages\": [{\"role\": \"user\", \"content\":
       \"What is 17 * 23? Then name the capital of Australia and write one haiku about GPUs.\"}]}")
t1=$(date +%s.%N)
echo "$RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
c = r['choices'][0]['message']['content']
u = r.get('usage', {})
print(c)
print('---')
print('completion tokens:', u.get('completion_tokens'))
"
python3 -c "print(f'wall: {$t1-$t0:.1f}s')"

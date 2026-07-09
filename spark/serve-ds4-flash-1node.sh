#!/usr/bin/env bash
# DeepSeek-V4-Flash (159B) on ONE DGX Spark (GB10, 121 GiB unified).
# 2-bit experts (~72.7 GiB) + FP8 dense, fully resident — the upstream "96 GB
# card" config with breathing room. Bring-up model for the GB10 port.
#
# usage: serve-ds4-flash-1node.sh [--eager] [extra vllm args...]
set -euo pipefail

VENV="$HOME/venvs/vllm-moet"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$HOME/models/hf/DeepSeek-V4-Flash"

export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0          # delta tier keeps a pinned host store —
                                       # double-dips unified memory; keep 0 on Spark
export VLLM_MOE_W2_CUBIT_DIR="$REPO/kernels/cubins-sm120"

ARGS=(
  --served-model-name deepseek-v4-flash --trust-remote-code
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 24576
  # unified memory: vLLM's "GPU total" is system RAM — leave headroom for the OS
  --gpu-memory-utilization 0.80
  --max-num-batched-tokens 1024 --max-num-seqs 4
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl
  --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'
  --port 8000
)

if [[ "${1:-}" == "--eager" ]]; then
  shift
  ARGS+=(--enforce-eager)
else
  ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}')
fi

exec "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" "$@"

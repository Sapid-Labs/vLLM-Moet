#!/usr/bin/env bash
# DISCRIMINATOR EXPERIMENT for the GLM MTP-under-PP corruption:
# DeepSeek-V4-Flash + MTP k=2 + PP2 across both Sparks.
# Upstream validated exactly this combination bit-exact on a single host
# (4x RTX 5090). If it is ALSO corrupt on our two-node Ray setup, the bug is
# in the generic multi-node draft-token plumbing; if it is clean, the bug is
# GLM-drafter-specific.
set -euo pipefail

VENV="$HOME/venvs/vllm-moet"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$HOME/models/hf/DeepSeek-V4-Flash"

export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0
export VLLM_MOE_W2_CUBIT_DIR="$REPO/kernels/cubins-sm120"
export VLLM_MOE_W2_PREPACKED_DIR="$MODEL/moe_w2_planes"
export VLLM_MOE_W2_PLANES_MMAP=1
export VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
export MALLOC_MMAP_THRESHOLD_=65536

export VLLM_HOST_IP=192.168.100.1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export RAY_memory_monitor_refresh_ms=0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1
export NCCL_IB_GID_INDEX=3

# 43 main layers + drafter on last rank
# partition left to default split (an explicit one also reaches the 1-layer
# drafter model config and trips len(partitions) != pp_size)

python3 "$REPO/spark/purge-cache.py" "$HOME/models/hf/DeepSeek-V4-Flash" >/dev/null || true
ssh 192.168.100.2 "python3 ~/Dev/vLLM-Moet/spark/purge-cache.py ~/models/hf/DeepSeek-V4-Flash" >/dev/null || true

ARGS=(
  --served-model-name ds4-pp2 --trust-remote-code
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 8192
  --gpu-memory-utilization 0.60
  --kv-cache-memory-bytes 2147483648
  --max-num-batched-tokens 512 --max-num-seqs 2
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl
  --host 0.0.0.0 --port 8000
)

if [[ "${1:-}" == "--no-mtp" ]]; then
  shift
else
  ARGS+=(--speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}')
fi

if [[ "${1:-}" == "--eager" ]]; then
  shift
  ARGS+=(--enforce-eager)
else
  ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}')
fi

exec systemd-run --user --scope --collect \
  -p MemoryMax=90G -p MemorySwapMax=0 \
  "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" "$@"

#!/usr/bin/env bash
# GLM-5.2 (753B) across TWO DGX Sparks — pipeline parallel over the 200G
# fabric. From zai-org/GLM-5.2-FP8 with PREPACKED 2-bit planes
# (spark/prepack_planes.py output on BOTH nodes at $MODEL/moe_w2_planes).
#
# Per rank ≈ 95 GiB pageable-host planes (read via ATS) + ~13 GiB cuda dense
# + KV. Bring the Ray cluster up first: spark/start-ray-cluster.sh
#
# usage: serve-glm52-pp2.sh [--eager] [extra vllm args...]
set -euo pipefail

VENV="$HOME/venvs/vllm-moet"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$HOME/models/hf/GLM-5.2-FP8"

export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0          # pinned host store double-dips unified mem
export VLLM_MOE_W2_CUBIT_DIR="$REPO/kernels/cubins-sm120"
export VLLM_MOE_W2_PREPACKED_DIR="$MODEL/moe_w2_planes"
export VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
export MALLOC_MMAP_THRESHOLD_=65536

# fabric plumbing (mirrors the proven hy3 PP2 run; peers of these env vars
# for rank 1 are set on the raylet by start-ray-cluster.sh)
export VLLM_HOST_IP=192.168.100.1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export RAY_memory_monitor_refresh_ms=0
export NCCL_IB_DISABLE=1   # PP boundary traffic is one 6144-vector/token; TCP is fine

# 78 layers: rank0 carries embeddings + 3 dense layers; give rank1 one more
# MoE layer only if memory tilts. Start even.
# export VLLM_PP_LAYER_PARTITION="39,39"

ARGS=(
  --served-model-name glm-5.2 --trust-remote-code
  --distributed-executor-backend ray --pipeline-parallel-size 2
  --kv-cache-dtype fp8 --max-model-len 8192
  # KV budget = util x total(121.7) - measured-used (system-wide incl. host
  # planes ~95 + cuda ~13 + base). Tune from the "Available KV cache memory"
  # log line; 0.93 targets ~4-6 GiB KV per rank.
  --gpu-memory-utilization 0.93
  --max-num-batched-tokens 1024 --max-num-seqs 2
  --host 0.0.0.0 --port 8000
)

if [[ "${1:-}" == "--eager" ]]; then
  shift
  ARGS+=(--enforce-eager)
else
  ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}')
fi

exec systemd-run --user --scope --collect \
  -p MemoryMax=112G -p MemorySwapMax=0 \
  "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" "$@"

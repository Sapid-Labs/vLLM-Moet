#!/usr/bin/env bash
# GLM-5.2 (753B) across TWO DGX Sparks — pipeline parallel over the 200G RoCE
# fabric. From zai-org/GLM-5.2-FP8 (dense already FP8; experts fp8->2bit at load
# via the patch's Fp8MoEMethod hook, K=6144/K=2048 cubins).
#
# Per rank ~95 GiB 2-bit experts + ~12 GiB FP8 dense + overhead ~= 111 GiB of
# ~114 usable. Run on the head node (spark-05cc, 192.168.100.1) with a Ray
# worker already up on the peer (spark-c84b, 192.168.100.2):
#
#   peer:  ~/venvs/vllm-moet/bin/ray start --address=192.168.100.1:6379
#   head:  ~/venvs/vllm-moet/bin/ray start --head --node-ip-address=192.168.100.1 --port=6379
#
# usage: serve-glm52-pp2.sh [--eager] [extra vllm args...]
set -euo pipefail

VENV="$HOME/venvs/vllm-moet"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$HOME/models/hf/GLM-5.2-FP8"

export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0
export VLLM_MOE_W2_CUBIT_DIR="$REPO/kernels/cubins-sm120"

# fabric plumbing (mirrors the proven hy3 PP2 run on this cluster)
export VLLM_HOST_IP=192.168.100.1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export RAY_memory_monitor_refresh_ms=0
# PP boundary traffic is tiny (one 6144-wide vector/token); socket transport is
# fine and avoids RoCE GID pinning headaches. Flip to RoCE later if it matters:
#   NCCL_IB_DISABLE=0 NCCL_IB_HCA=rocep1s0f1 NCCL_IB_GID_INDEX=3 (.1) / 5 (.2)
export NCCL_IB_DISABLE=1

# 78 layers: rank0 gets embeddings + 3 dense layers, so give rank1 the extra
# MoE layer if memory tilts — tune with VLLM_PP_LAYER_PARTITION="40,38" etc.

ARGS=(
  --served-model-name glm-5.2 --trust-remote-code
  --distributed-executor-backend ray --pipeline-parallel-size 2
  --kv-cache-dtype fp8 --max-model-len 16384
  --gpu-memory-utilization 0.92
  --max-num-batched-tokens 1024 --max-num-seqs 2
  --host 0.0.0.0 --port 8000
)

if [[ "${1:-}" == "--eager" ]]; then
  shift
  ARGS+=(--enforce-eager)
else
  ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}')
fi

exec "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" "$@"

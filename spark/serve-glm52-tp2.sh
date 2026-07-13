#!/usr/bin/env bash
# GLM-5.2 (753B) across TWO DGX Sparks — TENSOR parallel over the 200G
# fabric (replaces PP2's pipeline bubble with ~156 tiny allreduces/token;
# measured 20 us/op on this RoCE link => ~3 ms/token of comm vs the ~43 ms
# serial second-rank read PP2 pays; both ranks read their expert halves in
# parallel).
#
# Needs the TP2-sharded planes on BOTH nodes:
#   spark/prepack_planes.py --model ~/models/hf/GLM-5.2-FP8 \
#     --tp-rank <0 on .1 / 1 on .2> --tp-size 2
# (writes $MODEL/moe_w2_planes_tp2; w2 shards to K=1024 -> k1024 cubins).
# Bring the Ray cluster up first: spark/start-ray-cluster.sh
#
# usage: serve-glm52-tp2.sh [--eager] [extra vllm args...]
set -euo pipefail

VENV="$HOME/venvs/vllm-moet"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$HOME/models/hf/GLM-5.2-FP8"

export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0
export VLLM_MOE_W2_CUBIT_DIR="$REPO/kernels/cubins-sm120"
export VLLM_MOE_W2_PREPACKED_DIR="${VLLM_MOE_W2_PREPACKED_DIR:-$MODEL/moe_w2_planes_tp2}"
export VLLM_MOE_W2_PLANES_MMAP=1   # file-backed planes (same budget as PP2:
                                   # each rank holds HALF of EVERY layer)
export VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
export MALLOC_MMAP_THRESHOLD_=65536

# fabric plumbing — same as PP2, but now latency-critical (allreduce per
# layer). Peer-side env (GID index 5) lives on the raylet via
# start-ray-cluster.sh.
export VLLM_HOST_IP=192.168.100.1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export RAY_memory_monitor_refresh_ms=0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1
export NCCL_IB_GID_INDEX=3
# FULL_AND_PIECEWISE runs decode collectives INSIDE cuda graphs and prefill/
# piecewise collectives EAGERLY on the SAME communicator. Without graph-mixing
# support NCCL can service captured/uncaptured ops out of order on the proxy and
# deadlock (observed: both ranks stall at the same AllGather opCount, GPUs idle,
# RoCE flat). Required for FULL graphs cross-node. Overridable from the env.
export NCCL_GRAPH_MIXING_SUPPORT="${NCCL_GRAPH_MIXING_SUPPORT:-1}"

ARGS=(
  --served-model-name glm-5.2 --trust-remote-code
  --distributed-executor-backend ray --tensor-parallel-size 2
  --kv-cache-dtype fp8 --max-model-len 8192
  --gpu-memory-utilization 0.85
  --kv-cache-memory-bytes 2147483648
  --max-num-batched-tokens 512 --max-num-seqs 8
  --host 0.0.0.0 --port 8000
)

if [[ "${1:-}" == "--eager" ]]; then
  shift
  ARGS+=(--enforce-eager)
else
  ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}')
fi

# purge page cache on both nodes (startup free-memory check)
python3 "$REPO/spark/purge-cache.py" "$HOME/models/hf" || true
ssh 192.168.100.2 "python3 ~/Dev/vLLM-Moet/spark/purge-cache.py ~/models/hf" || true

exec systemd-run --user --scope --collect \
  -p MemoryMax=10G -p MemorySwapMax=0 \
  "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" "$@"

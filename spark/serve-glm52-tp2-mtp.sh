#!/usr/bin/env bash
# GLM-5.2 (753B) TP2 across both Sparks + MTP speculative decode.
#
# Same TP2 path as serve-glm52-tp2.sh (NCCL 2.30.7 via raylet fixes the FULL-
# graph replay hang; TP-sharded planes; both ranks read expert halves in
# parallel). Adds GLM's native next-n-predict drafter (config has
# num_nextn_predict_layers=1 -> k=1 is the native, best-read-economics case).
#
# WHY this might NOT help: the stack is weight-read-bound (~193 ms/token, 96%
# GPU both nodes at TP2 single-stream). A k=1 verify step processes 2 positions
# -> up to ~2x expert-weight reads for (1+acceptance)x tokens. A prior k=2 test
# under PP LOST (3x reads / 2.2x tokens -> 3.65 vs 4.9-5.5 tok/s). k=1 is the
# most favorable case and is untested under TP2 — this measures acceptance rate
# and tok/s to settle it.
#
# Needs the TP2-sharded planes on BOTH nodes and the Ray cluster up
# (start-ray-cluster.sh — it carries VLLM_NCCL_SO_PATH=~/nccl-2.30.7 + GIDs).
#
# usage: serve-glm52-tp2-mtp.sh [--eager] [--no-mtp] [extra vllm args...]
#   MTP_K env overrides num_speculative_tokens (default 1).
set -euo pipefail

VENV="$HOME/venvs/vllm-moet"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$HOME/models/hf/GLM-5.2-FP8}"

export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0
export VLLM_MOE_W2_CUBIT_DIR="$REPO/kernels/cubins-sm120"
export VLLM_MOE_W2_PREPACKED_DIR="${VLLM_MOE_W2_PREPACKED_DIR:-$MODEL/moe_w2_planes_tp2}"
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
export NCCL_GRAPH_MIXING_SUPPORT="${NCCL_GRAPH_MIXING_SUPPORT:-1}"

MTP_K="${MTP_K:-1}"

ARGS=(
  --served-model-name glm-5.2 --trust-remote-code
  --distributed-executor-backend ray --tensor-parallel-size 2
  --kv-cache-dtype fp8 --max-model-len 8192
  --gpu-memory-utilization 0.85
  --kv-cache-memory-bytes 2147483648
  --max-num-batched-tokens 512 --max-num-seqs 8
  --host 0.0.0.0 --port 8000
)

if [[ "${1:-}" == "--no-mtp" ]]; then
  shift
else
  # DRAFT_TP: run the MTP draft at a different tensor-parallel size than the
  # target (must be 1 or the target TP). DRAFT_TP=1 makes the draft forward
  # TP-free (no per-draft all-reduce) — attacks the fixed per-draft overhead.
  spec="{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP_K"
  if [[ -n "${DRAFT_TP:-}" ]]; then
    spec="$spec, \"draft_tensor_parallel_size\": $DRAFT_TP"
  fi
  spec="$spec}"
  ARGS+=(--speculative-config "$spec")
fi

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

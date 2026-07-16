#!/usr/bin/env bash
# GLM-5.2 (753B) TP2 across both Sparks + EXTERNAL DSpark speculative decode.
#
# Same TP2 plane-offload path as serve-glm52-tp2-mtp.sh, but instead of GLM's
# native MTP drafter it uses the external speculators-format draft
# RedHatAI/GLM-5.2-speculator.dspark (DSparkDraftModel, markov head). Needs the
# dspark port applied to the venv (see ~/Dev/vLLM-Moet/dspark-port/ and HANDOFF.md).
#
# Needs the TP2-sharded planes on BOTH nodes, the Ray cluster up
# (start-ray-cluster.sh), AND the speculator at the SAME path on both nodes:
#   ~/models/hf/GLM-5.2-speculator.dspark
#
# usage: serve-glm52-tp2-dspark.sh [--eager] [--no-spec] [extra vllm args...]
#   DSPARK_K env overrides num_speculative_tokens (default 3; speculator native = 7).
set -euo pipefail

VENV="$HOME/venvs/vllm-moet"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$HOME/models/hf/GLM-5.2-FP8}"
SPECULATOR="${SPECULATOR:-$HOME/models/hf/GLM-5.2-speculator.dspark}"

export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0
export VLLM_MOE_W2_CUBIT_DIR="$REPO/kernels/cubins-sm120"
export VLLM_MOE_W2_PREPACKED_DIR="${VLLM_MOE_W2_PREPACKED_DIR:-$MODEL/moe_w2_planes_tp2}"
export VLLM_MOE_W2_PLANES_MMAP=1
export VLLM_MOE_W2_FADVISE_GLOB="${VLLM_MOE_W2_FADVISE_GLOB-$MODEL/*.safetensors}"
export MALLOC_MMAP_THRESHOLD_=65536

export VLLM_HOST_IP=192.168.100.1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export RAY_memory_monitor_refresh_ms=0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_GRAPH_MIXING_SUPPORT="${NCCL_GRAPH_MIXING_SUPPORT:-1}"

DSPARK_K="${DSPARK_K:-3}"

ARGS=(
  --served-model-name glm-5.2 --trust-remote-code
  --distributed-executor-backend ray --tensor-parallel-size 2
  --kv-cache-dtype fp8 --max-model-len 8192
  --gpu-memory-utilization 0.85
  --kv-cache-memory-bytes 2147483648
  --max-num-batched-tokens 512 --max-num-seqs 8
  --host 0.0.0.0 --port 8000
)

if [[ "${1:-}" == "--no-spec" ]]; then
  shift
else
  spec="{\"method\": \"dspark\", \"model\": \"$SPECULATOR\", \"num_speculative_tokens\": $DSPARK_K"
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

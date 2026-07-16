#!/usr/bin/env bash
# GLM-5.2 TP2 "hidden-states server" for DSpark draft self-distillation.
#
# Same fast-build target as serve-glm52-tp2-dspark.sh (planes + NVFP4 + top-k4
# via env/args), but instead of a speculative drafter it runs the
# extract_hidden_states method + ExampleHiddenStatesConnector: every completion
# request (speculators' data-gen sends token ids, max_tokens=1,
# return_token_ids) makes the server write {hidden_states [seq, 6, 6144],
# token_ids} safetensors for aux layers [8,23,39,55,70]+last into
# $HIDDEN_STATES_PATH and return the file path in kv_transfer_params.
#
# Mirrors speculators/scripts/launch_vllm.py's two JSON configs; chunked
# prefill must be OFF for extraction, so max-num-batched-tokens = max-model-len.
#
# usage: HIDDEN_STATES_PATH=~/dspark-hs-cache bash spark/serve-glm52-tp2-hidden.sh [extra vllm args...]
# See spark/dspark-distill/PLAN.md.
set -euo pipefail

VENV="$HOME/venvs/vllm-moet"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$HOME/models/hf/GLM-5.2-FP8}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-$HOME/dspark-hs-cache}"
mkdir -p "$HIDDEN_STATES_PATH"

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

# 5 aux layers for the draft fc + layer 78 (= num_hidden_layers, the LAST
# layer) whose hidden states the trainer needs to compute the tv/ce targets
# via verifier_lm_head. Matches launch_vllm.py's include_last_layer behavior.
# HIDDEN_AUX_IDS overrides the aux id list (this fork collects the INPUT of
# layer i, so id i = output of layer i-1; +1 the card's ids to get outputs).
HIDDEN_AUX_IDS="${HIDDEN_AUX_IDS:-[8, 23, 39, 55, 70, 78]}"
SPEC_CONFIG="{\"method\": \"extract_hidden_states\", \"num_speculative_tokens\": 1, \"draft_model_config\": {\"hf_config\": {\"eagle_aux_hidden_state_layer_ids\": $HIDDEN_AUX_IDS}}}"
KVT_CONFIG="{\"kv_connector\": \"ExampleHiddenStatesConnector\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"shared_storage_path\": \"$HIDDEN_STATES_PATH\"}}"

ARGS=(
  --served-model-name glm-5.2 --trust-remote-code
  --distributed-executor-backend ray --tensor-parallel-size 2
  --kv-cache-dtype fp8 --max-model-len 8192
  --gpu-memory-utilization 0.85
  --kv-cache-memory-bytes 2147483648
  --max-num-batched-tokens 8192 --max-num-seqs 8
  --no-enable-chunked-prefill
  --host 0.0.0.0 --port 8000
  --speculative-config "$SPEC_CONFIG"
  --kv-transfer-config "$KVT_CONFIG"
  --enforce-eager
)

# purge page cache on both nodes (startup free-memory check)
python3 "$REPO/spark/purge-cache.py" "$HOME/models/hf" || true
ssh 192.168.100.2 "python3 ~/Dev/vLLM-Moet/spark/purge-cache.py ~/models/hf" || true

exec systemd-run --user --scope --collect \
  -p MemoryMax=10G -p MemorySwapMax=0 \
  "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" "$@"

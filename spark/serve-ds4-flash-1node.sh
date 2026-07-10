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
# prepacked 2-bit planes (spark/prepack_planes.py): load planes from disk,
# skip all in-process staging/requant
export VLLM_MOE_W2_PREPACKED_DIR="$MODEL/moe_w2_planes"
# drop consumed shard page-cache during load (cudaMalloc can't reclaim it)
export VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
# glibc: return freed medium allocations (the loader churns thousands of
# ~10MB buffers; arena free-lists otherwise retain tens of GiB)
export MALLOC_MMAP_THRESHOLD_=65536

ARGS=(
  --served-model-name deepseek-v4-flash --trust-remote-code
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 8192
  # unified memory: vLLM's KV budget = util x total(121.7) - measured-used.
  # The probe is system-wide, so the ~89 GiB in use after load (74 host
  # planes + 12 cuda dense + base) counts; 0.78 leaves ~6 GiB KV — millions
  # of MLA tokens at 656 B/token.
  --gpu-memory-utilization 0.78
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

# Containment: the engine runs in a user cgroup scope. MemoryHigh triggers
# in-cgroup reclaim before the hard cap; MemoryMax guarantees an overrun
# kills ONLY the engine — never the desktop/session (the kernel OOM killer
# took the whole box down once; not again).
exec systemd-run --user --scope --collect \
  -p MemoryMax=110G -p MemorySwapMax=0 \
  "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" "$@"

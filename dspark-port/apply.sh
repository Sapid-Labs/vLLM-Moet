#!/usr/bin/env bash
# Re-apply the dspark speculator port into a vLLM venv site-packages.
# Usage: ./apply.sh /path/to/venv/lib/python3.12/site-packages/vllm
#
# Every file under new/ is copied. Earlier revisions of this script shipped only
# a subset (it silently skipped v1/spec_decode/*.py and gpu_model_runner.py),
# which produced a venv that imported fine and then behaved like stock vLLM —
# the worst possible failure mode. Keep this list complete: if you add a file to
# new/, add it here, or let the check at the bottom catch you.
set -euo pipefail
VLLM="${1:?pass the target vllm site-packages dir}"
HERE="$(cd "$(dirname "$0")" && pwd)/new"

# --- config / registry plumbing -------------------------------------------
cp -v "$HERE"/transformers_utils/configs/speculators/algos.py "$VLLM/transformers_utils/configs/speculators/algos.py"
cp -v "$HERE"/config/speculative.py                            "$VLLM/config/speculative.py"
cp -v "$HERE"/model_executor/models/registry.py                "$VLLM/model_executor/models/registry.py"

# --- model definitions ------------------------------------------------------
# qwen3_dflash.py is REQUIRED: qwen3_dspark subclasses it, and it carries the
# quantized-drafter fixes (qkv_proj routed through quant_method; fused context-KV
# rows unpacked instead of slicing .weight). Without it a W4A16 drafter either
# raises or silently computes garbage.
cp -v "$HERE"/model_executor/models/qwen3_dflash.py "$VLLM/model_executor/models/qwen3_dflash.py"
cp -v "$HERE"/model_executor/models/qwen3_dspark.py "$VLLM/model_executor/models/qwen3_dspark.py"

# --- v1 proposer framework --------------------------------------------------
mkdir -p "$VLLM/v1/spec_decode"
cp -v "$HERE"/v1/spec_decode/dflash.py            "$VLLM/v1/spec_decode/dflash.py"
cp -v "$HERE"/v1/spec_decode/dspark.py            "$VLLM/v1/spec_decode/dspark.py"
cp -v "$HERE"/v1/spec_decode/llm_base_proposer.py "$VLLM/v1/spec_decode/llm_base_proposer.py"
cp -v "$HERE"/v1/worker/gpu_model_runner.py       "$VLLM/v1/worker/gpu_model_runner.py"

# --- speculators-format reference path --------------------------------------
mkdir -p "$VLLM/v1/worker/gpu/spec_decode/dspark"
cp -v "$HERE"/v1/worker/gpu/spec_decode/__init__.py "$VLLM/v1/worker/gpu/spec_decode/__init__.py"
cp -v "$HERE"/v1/worker/gpu/spec_decode/dspark/*.py "$VLLM/v1/worker/gpu/spec_decode/dspark/"

# --- guard: nothing under new/ left behind ----------------------------------
missing=0
while IFS= read -r src; do
    rel="${src#"$HERE"/}"
    case "$rel" in serve-*.sh) continue ;; esac   # helper scripts, not venv files
    if ! cmp -s "$src" "$VLLM/$rel"; then
        echo "NOT APPLIED: new/$rel" >&2
        missing=1
    fi
done < <(find "$HERE" -type f -name '*.py')
if [ "$missing" -ne 0 ]; then
    echo "ERROR: files under new/ were not copied — this script is out of date." >&2
    exit 1
fi

echo "dspark port applied to $VLLM"

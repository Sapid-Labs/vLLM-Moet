#!/usr/bin/env bash
# Re-apply the dspark speculator port into a vLLM venv site-packages.
# Usage: ./apply.sh /path/to/venv/lib/python3.12/site-packages/vllm
set -euo pipefail
VLLM="${1:?pass the target vllm site-packages dir}"
HERE="$(cd "$(dirname "$0")" && pwd)/new"
cp -v "$HERE"/transformers_utils/configs/speculators/algos.py "$VLLM/transformers_utils/configs/speculators/algos.py"
cp -v "$HERE"/config/speculative.py "$VLLM/config/speculative.py"
cp -v "$HERE"/model_executor/models/registry.py "$VLLM/model_executor/models/registry.py"
cp -v "$HERE"/model_executor/models/qwen3_dspark.py "$VLLM/model_executor/models/qwen3_dspark.py"
mkdir -p "$VLLM/v1/worker/gpu/spec_decode/dspark"
cp -v "$HERE"/v1/worker/gpu/spec_decode/__init__.py "$VLLM/v1/worker/gpu/spec_decode/__init__.py"
cp -v "$HERE"/v1/worker/gpu/spec_decode/dspark/*.py "$VLLM/v1/worker/gpu/spec_decode/dspark/"
echo "dspark port applied to $VLLM"

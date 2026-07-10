#!/usr/bin/env bash
# Bring up the 2-Spark Ray cluster for GLM-5.2 PP2 (run on spark-05cc / .1).
#
# Encodes the hy3-era gotchas (howtospark docs/dgx-spark-notes.md):
#  - raylets must start from LOGIN shells so Ray actors inherit PATH with
#    ninja/nvcc (flashinfer JITs on first forward)
#  - GLOO_SOCKET_IFNAME + per-node VLLM_HOST_IP on the RAYLETS (actors
#    inherit env from the raylet, not from vllm serve)
#  - RAY_memory_monitor_refresh_ms=0 (it counts model weights as system RAM)
#  - raylets run under a cgroup scope so a bad rank can't take a box down
set -euo pipefail
VENV="$HOME/venvs/vllm-moet"

COMMON="RAY_memory_monitor_refresh_ms=0 NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1 \
VLLM_MOE_W2=1 VLLM_MOE_W2_DELTA_GB=0 \
VLLM_MOE_W2_CUBIT_DIR=\$HOME/Dev/vLLM-Moet/kernels/cubins-sm120 \
VLLM_MOE_W2_PREPACKED_DIR=\$HOME/models/hf/GLM-5.2-FP8/moe_w2_planes \
VLLM_MOE_W2_FADVISE_GLOB='\$HOME/models/hf/GLM-5.2-FP8/*.safetensors' \
MALLOC_MMAP_THRESHOLD_=65536"

echo "== stopping any old ray =="
bash -lc "$VENV/bin/ray stop --force" || true
ssh 192.168.100.2 "bash -lc '$VENV/bin/ray stop --force'" || true
sleep 2

echo "== head (this node, .1) =="
bash -lc "env $COMMON VLLM_HOST_IP=192.168.100.1 \
  systemd-run --user --scope --collect -p MemoryMax=112G -p MemorySwapMax=0 \
  $VENV/bin/ray start --head --node-ip-address=192.168.100.1 --port=6379 --block" &
sleep 6

echo "== worker (peer, .2) =="
ssh 192.168.100.2 "bash -lc \"env $COMMON VLLM_HOST_IP=192.168.100.2 \
  systemd-run --user --scope --collect -p MemoryMax=112G -p MemorySwapMax=0 \
  $VENV/bin/ray start --address=192.168.100.1:6379 \
  --node-ip-address=192.168.100.2 --block\"" &
sleep 8

bash -lc "$VENV/bin/ray status"

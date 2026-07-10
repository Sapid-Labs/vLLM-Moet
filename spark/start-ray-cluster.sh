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

# Node truths ONLY (fabric + memory knobs). Config-specific vars — the
# VLLM_MOE_W2_* family, spec/trace toggles — must NOT be baked here: ray
# workers apply the raylet env over driver-forwarded env (setdefault
# semantics: node-local values always win), so a stale raylet value
# silently overrides whatever the serve script exports. That bit us on
# 2026-07-10: TP2 workers loaded the PP planes dir baked into the raylet
# and computed garbage. Serve-script env reaches workers via the driver
# forwarding as long as the raylet doesn't shadow it.
COMMON="RAY_memory_monitor_refresh_ms=0 NCCL_IB_DISABLE=0 NCCL_IB_HCA=rocep1s0f1 \
NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1 \
MALLOC_MMAP_THRESHOLD_=65536"

echo "== stopping any old ray =="
bash -lc "$VENV/bin/ray stop --force" || true
ssh 192.168.100.2 "bash -lc '$VENV/bin/ray stop --force'" || true
sleep 2

echo "== head (this node, .1) =="
# via self-SSH: a FRESH login session picks up limits.d (memlock unlimited
# for RDMA registration — an inherited shell keeps the old 8MB and NCCL dies)
ssh 192.168.100.1 "bash -lc \"env $COMMON VLLM_HOST_IP=192.168.100.1 NCCL_IB_GID_INDEX=3 \
  systemd-run --user --scope --collect -p MemoryMax=102G -p MemorySwapMax=0 \
  $VENV/bin/ray start --head --node-ip-address=192.168.100.1 --port=6379 --object-store-memory=2000000000\""
sleep 6

echo "== worker (peer, .2) =="
ssh 192.168.100.2 "bash -lc \"env $COMMON VLLM_HOST_IP=192.168.100.2 NCCL_IB_GID_INDEX=5 \
  systemd-run --user --scope --collect -p MemoryMax=114G -p MemorySwapMax=0 \
  $VENV/bin/ray start --address=192.168.100.1:6379 --object-store-memory=2000000000 \
  --node-ip-address=192.168.100.2\""
sleep 8

bash -lc "$VENV/bin/ray status"

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
# NCCL_GRAPH_MIXING_SUPPORT=1: required for TP2 FULL cuda graphs cross-node.
# FULL_AND_PIECEWISE captures decode collectives in graphs but runs prefill/
# piecewise collectives eagerly on the SAME communicator; without graph-mixing
# support NCCL's proxy can service captured/uncaptured ops out of order and
# deadlock (both ranks stall at the same AllGather opCount, GPUs idle, RoCE
# flat). This is an NCCL node-truth like the NCCL_IB_* vars, so it must live on
# the raylet — actors take the raylet NCCL env, not the driver's.
# VLLM_NCCL_SO_PATH: pin the NCCL runtime pynccl loads. Torch bundles 2.28.9;
# GB10+CX7 is a new platform combo and cuda-graph collective-replay fixes land
# in NCCL regularly, so we test 2.30.7 (staged at ~/nccl-2.30.7 on both nodes).
# Must live on the raylet — actors load NCCL per the raylet env, not the driver.
NCCL_SO="$HOME/nccl-2.30.7/libnccl.so.2"
COMMON="RAY_memory_monitor_refresh_ms=0 RAY_local_fs_capacity_threshold=1 NCCL_IB_DISABLE=0 NCCL_IB_HCA=rocep1s0f1 \
NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1 \
NCCL_GRAPH_MIXING_SUPPORT=1 VLLM_NCCL_SO_PATH=$NCCL_SO MALLOC_MMAP_THRESHOLD_=65536"

# GID indexes are NOT stable across reboots/link events (2026-07-12: .2's v2
# GID moved 5→6 and NCCL died at init with "unhandled system error"). Resolve
# each node's RoCEv2 GID index for its fabric IP at start time.
gid_index() { # $1 = node ssh target, $2 = fabric ip
  ssh "$1" "for g in /sys/class/infiniband/rocep1s0f1/ports/1/gids/*; do \
    i=\${g##*/}; \
    [ \"\$(cat /sys/class/infiniband/rocep1s0f1/ports/1/gid_attrs/types/\$i 2>/dev/null)\" = 'RoCE v2' ] || continue; \
    case \$(cat \$g) in *ffff:$(printf '%02x%02x:%02x%02x' $(echo $2 | tr . ' '))) echo \$i; break;; esac; done"
}
GID1=$(gid_index 192.168.100.1 192.168.100.1)
GID2=$(gid_index 192.168.100.2 192.168.100.2)
[ -n "$GID1" ] && [ -n "$GID2" ] || { echo "FATAL: could not resolve RoCEv2 GID index (.1='$GID1' .2='$GID2')"; exit 1; }
echo "== RoCEv2 GID indexes: .1=$GID1 .2=$GID2 =="

echo "== stopping any old ray =="
bash -lc "$VENV/bin/ray stop --force" || true
ssh 192.168.100.2 "bash -lc '$VENV/bin/ray stop --force'" || true
sleep 2

echo "== head (this node, .1) =="
# via self-SSH: a FRESH login session picks up limits.d (memlock unlimited
# for RDMA registration — an inherited shell keeps the old 8MB and NCCL dies)
ssh 192.168.100.1 "bash -lc \"env $COMMON VLLM_HOST_IP=192.168.100.1 NCCL_IB_GID_INDEX=$GID1 \
  systemd-run --user --scope --collect -p MemoryMax=102G -p MemorySwapMax=0 \
  $VENV/bin/ray start --head --node-ip-address=192.168.100.1 --port=6379 --object-store-memory=2000000000\""
sleep 6

echo "== worker (peer, .2) =="
ssh 192.168.100.2 "bash -lc \"env $COMMON VLLM_HOST_IP=192.168.100.2 NCCL_IB_GID_INDEX=$GID2 \
  systemd-run --user --scope --collect -p MemoryMax=114G -p MemorySwapMax=0 \
  $VENV/bin/ray start --address=192.168.100.1:6379 --object-store-memory=2000000000 \
  --node-ip-address=192.168.100.2\""
sleep 8

bash -lc "$VENV/bin/ray status"

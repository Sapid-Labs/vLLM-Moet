#!/usr/bin/env bash
# Pre-fault the mmap'd 2-bit planes into page cache (both nodes for GLM).
# Cold planes cost the first requests seconds of NVMe faults; run this after
# the server starts (or after anything that purged the cache).
set -euo pipefail
D="${1:-$HOME/models/hf/GLM-5.2-FP8/moe_w2_planes}"
echo "warming $D (local)"; cat "$D"/*.npy > /dev/null 2>&1 || true
if [[ "${2:-}" == "--peer" ]]; then
  echo "warming peer"; ssh 192.168.100.2 "cat $D/*.npy > /dev/null" || true
fi
echo done

#!/usr/bin/env bash
# Queued decode-step profiler for the 20 tok/s goal (spark/GOAL.md).
#
# Splits the ~67 ms/token single-stream budget three ways, since FULL-cudagraph
# decode is one opaque graph (prefill_timers no-op inside capture):
#   A) M-scaling on the LIVE graph+MTP server (non-destructive) -> bandwidth-
#      bound vs latency/M-starved verdict.
#   B) eager reboot with VLLM_PREFILL_TIMERS=1 -> moe_w2 GEMM ms/token (the
#      dominant read cost) vs everything-else; + graph-vs-eager tok/s delta.
#   C) standalone TP2 allreduce at decode message size -> comm ms/token.
# Restores the shipped graph+MTP config on exit (trap).
#
# Waits for the eval battery to finish first (never co-runs — wrecks both).
# Launch detached:  nohup bash spark/profile_decode.sh > ~/profile-decode.log 2>&1 &
set -u
REPO="$HOME/Dev/vLLM-Moet"
MODEL="$HOME/models/hf/GLM-5.2-FP8"
PLANES="$MODEL/moe_w2_planes_tp2_p208"
PEER=192.168.100.2
OUT="$HOME/profile-decode-results.txt"
say() { echo "=== [$(TZ=EST5EST date +%H:%M:%S)] $* ==="; }

wait_health() { # wait_health <max_s>
  local i=0
  until curl -s -m 2 http://localhost:8000/health >/dev/null 2>&1; do
    pgrep -f 'vllm serve' >/dev/null || { echo "serve process gone during boot"; return 1; }
    sleep 5; i=$((i+5)); [ $i -ge "${1:-1200}" ] && { echo "health timeout"; return 1; }
  done
}
serve_down() {
  pkill -f 'vllm serve' 2>/dev/null; sleep 8
  pgrep -f 'vllm serve' >/dev/null && { pkill -9 -f 'vllm serve'; sleep 5; }
}
restore_shipped() {
  say "RESTORE: relaunching shipped graph+MTP server"
  serve_down
  VLLM_MOE_W2_PREPACKED_DIR="$PLANES" MTP_K=1 \
    nohup bash "$REPO/spark/serve-glm52-tp2-mtp.sh" > "$HOME/serve-mtp-restored.log" 2>&1 &
  wait_health 1200 && bash "$REPO/spark/warm-planes.sh" "$PLANES" --peer >/dev/null 2>&1
  say "RESTORE done (server on shipped config). Manual restore if needed:
    VLLM_MOE_W2_PREPACKED_DIR=$PLANES MTP_K=1 bash $REPO/spark/serve-glm52-tp2-mtp.sh"
}
trap 'restore_shipped; say "PROFILE COMPLETE"' EXIT

cd "$REPO" || exit 1
: > "$OUT"
say "PROFILE QUEUED — waiting for eval battery to finish"
# ---- gate: evals fully idle for two consecutive checks ----
idle=0
while true; do
  if pgrep -f 'run_battery.sh|lm_eval' >/dev/null; then idle=0; else idle=$((idle+1)); fi
  [ $idle -ge 2 ] && break
  sleep 90
done
say "evals idle — server is free; starting profile"

########################################################################
say "PHASE A: M-scaling on live graph+MTP server (non-destructive)" | tee -a "$OUT"
bash "$REPO/spark/warm-planes.sh" "$PLANES" --peer >/dev/null 2>&1
# settle the resident tier
python3 "$REPO/spark/demo.py" --tokens 300 >/dev/null 2>&1 || true
python3 "$REPO/spark/profiling/m_scaling.py" --tokens 256 --levels 1,2,4,8 2>&1 | tee -a "$OUT"

########################################################################
say "PHASE B: eager reboot + VLLM_PREFILL_TIMERS (moe_w2 GEMM share)" | tee -a "$OUT"
serve_down
VLLM_MOE_W2_PREPACKED_DIR="$PLANES" \
  VLLM_PREFILL_TIMERS=1 VLLM_PREFILL_TIMERS_FLUSH=75 \
  nohup bash "$REPO/spark/serve-glm52-tp2-mtp.sh" --no-mtp --eager \
  > "$HOME/profile-eager-serve.log" 2>&1 &
if wait_health 1500; then
  bash "$REPO/spark/warm-planes.sh" "$PLANES" --peer >/dev/null 2>&1
  # eager single-stream rate (no MTP): clean per-token decode
  echo "-- eager (no-MTP) decode rate --" | tee -a "$OUT"
  python3 "$REPO/spark/demo.py" --tokens 200 2>&1 | grep -E 'tok/s' | tee -a "$OUT"
  # drain the timer: run enough decode that moe_w2 flushes (75 spans/token)
  python3 "$REPO/spark/demo.py" --tokens 200 >/dev/null 2>&1 || true
  sleep 3
  echo "-- prefill-timer cumulative (eager) --" | tee -a "$OUT"
  grep 'prefill-timer' "$HOME/profile-eager-serve.log" | tail -12 | tee -a "$OUT"
  python3 - "$HOME/profile-eager-serve.log" >> "$OUT" 2>&1 <<'PY'
import re, sys
tot, cnt = {}, {}
for ln in open(sys.argv[1]):
    m = re.search(r'prefill-timer\]\s+(\S+)\s+total\s+([\d.]+) ms over\s+(\d+)', ln)
    if m: tot[m.group(1)] = float(m.group(2)); cnt[m.group(1)] = int(m.group(3))
mm = tot.get('moe_w2'); c = cnt.get('moe_w2')
if mm and c:
    per_layer = mm / c
    print(f"-- moe_w2: {mm:.0f} ms over {c} spans = {per_layer:.3f} ms/layer-call")
    print(f"-- x75 MoE layers = {per_layer*75:.1f} ms/token of moe_w2 GEMM+read (eager)")
PY
else
  echo "PHASE B: eager server failed to come up (see profile-eager-serve.log)" | tee -a "$OUT"
fi

########################################################################
say "PHASE C: standalone TP2 allreduce at decode msg size (comm)" | tee -a "$OUT"
serve_down  # free the GPU on both nodes for the probe
# resolve this node's RoCEv2 GID index for .1 (drifts across reboots)
GID=$(for g in /sys/class/infiniband/rocep1s0f1/ports/1/gids/*; do
        v=$(cat "$g" 2>/dev/null); i=$(basename "$g");
        [ "$(cat "${g%gids/*}gid_attrs/types/$i" 2>/dev/null)" = "RoCE v2" ] &&
        echo "$v" | grep -qi 'ffff:c0a8:6401' && echo "$i" && break; done)
GID=${GID:-3}
NENV="NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1 NCCL_IB_GID_INDEX=$GID NCCL_IB_DISABLE=0"
PYBIN="$HOME/venvs/vllm-moet/bin/python3.12"
( ssh "$PEER" "cd ~/Dev/vLLM-Moet && $NENV MASTER_ADDR=192.168.100.1 MASTER_PORT=29555 RANK=1 WORLD_SIZE=2 \
     ~/venvs/vllm-moet/bin/python3.12 spark/profiling/allreduce_probe.py" </dev/null 2>&1 | sed 's/^/[peer] /' ) &
env $NENV MASTER_ADDR=192.168.100.1 MASTER_PORT=29555 RANK=0 WORLD_SIZE=2 \
    "$PYBIN" "$REPO/spark/profiling/allreduce_probe.py" 2>&1 | tee -a "$OUT"
wait

say "RAW RESULTS in $OUT — restoring shipped server next"
# trap handles restore + COMPLETE marker

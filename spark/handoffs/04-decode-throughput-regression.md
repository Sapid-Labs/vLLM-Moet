# HANDOFF — decode throughput regression: dspark K=2 = 10.4 tok/s (was 21.8)

Resume-here for the investigation into why GLM-5.2 dspark K=2 single-stream decode
measures ~10.4 tok/s now vs the documented 21.8 (session 14). Written 2026-07-17
(session 17). Deep context: `MTP-DRAFT-COST.md` (the draft-cost thesis this confirms),
`RUNBOOK.md` §4b (the config), `handoffs/03-*.md` (the 21.8/15.6 baselines).

## STATUS — regression LOCALIZED to the verify per-position expert-read; NOT draft, NOT hardware. Needs kernel profiling, not more black-box serve-measure.

Symptom (measured this session, both v2 AND epoch-3 speculators, identical):
| config (fast build §4b, top-k4, greedy 300-tok, streamed) | now | documented (s14) |
| dspark K=2 | **10.4 tok/s** (pos-0 0.771, pos-1 0.561, 2.33 tok/verify) | 21.8 (0.68/0.45, 2.14) |
| no-draft floor (top-k4) | **14.0 tok/s** | 15.6 |
**dspark is BELOW the no-draft floor → the draft is currently NET-NEGATIVE** (it was
+40% at s14: 15.6→21.8; now −26%: 14.0→10.4). Acceptance is FINE (even better than
documented). No-draft only dropped ~10% (15.6→14); dspark dropped ~52%.

## THE REFRAMING (do not re-derive — from MTP-DRAFT-COST.md session 12)
The draft FORWARD is not the cost. Session 12 proved cheapening the draft (FULL
cudagraph / TP-free / breakable) gives ZERO speedup. The per-depth cost (~32 ms) is
on the VERIFY side: each spec position makes the verify process one more position,
and the verify is **bandwidth-bound on per-position MoE expert reads** (K+1 positions
≈ (K+1)× top-k expert reads). So dspark K=2 pays a 3-position verify-read tax.
→ The "76 ms draft pass" I computed is really the verify doing 3 positions of expert
reads, mis-attributed. The regression is in the **multi-position verify expert-gather**,
which hits dspark (3 positions) ~3× harder than no-draft (1 position) — fits the
10% vs 52% split exactly. And s12 shows spec-decode's margin over no-draft was always
small/fragile (MTP K1 18.5 vs K2 19.0), so a modest per-position-read regression flips
dspark net-negative.

## DONE — exhaustively RULED OUT (don't re-run these; all clean)
- **Thermal**: 170s sustained decode → throttle bitmask 0x0 both nodes, temps plateau
  68°C, SM clock steady ~2500 MHz, tok/s flat (no droop). GPU T.Limit fine.
- **Power cap**: not active during decode; ~38-40 W drawn (SW-power-cap counter shows
  ~14.8h lifetime but NOT active now).
- **Disk/plane faulting**: 0 MB NVMe read during 30s decode (both nodes) → planes fully
  resident. fadvise-DONTNEED of 300 GB hidden states changed nothing.
- **v2-specific**: v2 and epoch-3 speculators byte-identical (config.py + weight
  keys/shapes/dtypes; only cosmetic transformers_version differs). Both 10.4.
- **RoCE**: bandwidth 97.6 Gb/s (ib_write_bw), RDMA write latency **1.45 µs**
  (ib_write_lat) — healthy. Link up 200G both nodes. GID .1=3 .2=6 correct.
- **CPU C-states**: deep LPI-3 (433 µs) inflates ICMP ping (0.61→0.043 ms with cores
  spun awake) BUT disabling state2/state3 gave **ZERO decode change** (10.4→10.4).
  NCCL uses GPUDirect RDMA (kernel-bypass), not gated by CPU sleep. Governor=performance,
  2808 MHz. **C-states are latency red-herring for decode.**
- **Memory bandwidth**: torch copy (read+write) **241 GB/s** (~88% of 273 peak), sum-read
  194 GB/s → healthy. NOT a mem-clock/P-state halving.
- **cudagraphs**: FULL captured ("6 decode, FULL") + PIECEWISE ("9 mixed"). torch.compile
  loaded from cache. GPU 93-95% busy (not idle-waiting on comm).
- **NCCL version**: workers loaded `~/nccl-2.30.7/libnccl.so.2` (via VLLM_NCCL_SO_PATH,
  set by start-ray-cluster.sh). torch-bundled is 2.28.9 but not the one used.
- **Draft code**: installed `v1/spec_decode/dspark.py`, `llm_base_proposer.py`,
  `qwen3_dspark.py`, `dspark/speculator.py`+`utils.py`, `gpu_model_runner.py` are
  **byte-identical (0 diff)** to `dspark-port/new/` canonical. 07-16 19:10 mtime on
  llm_base_proposer was apply.sh re-write, not a content change.
- **Base vLLM**: installed 2026-07-09 (dist-info + core mtimes), UNCHANGED since before
  the s14 21.8 run. 0.24.0.

## NEXT — profile the verify per-position expert-read (the ONLY unturned stone)
The question: why is the multi-position verify expert-gather ~2× more expensive than
s14. Black-box serve-measure is exhausted; this needs instrumentation:
1. **Byte-count per verify position**: confirm top-k4 actually applies on the VERIFY
   path (num_experts_per_tok=4), and count expert bytes read per position. If verify
   routes k8 (override not hitting the verify/spec path), that doubles per-position
   reads. Check `moe_w2_cubit.py` gather + the hf-override plumbing into the spec verify.
2. **nvtx-scoped timer** around the verify MoE expert read (NOT torch profiler — s12
   says it self-distorts serving to ~10 tok/s and can't split draft/verify). Compare
   per-position expert-read ms now vs the s12/s14 ~32 ms/position.
3. **A/B K sweep on a clean serve**: dspark K=1 vs K=2 step-time delta = the true
   per-position tax. If Δ ≈ 32 ms it matches s14 (regression is elsewhere); if Δ ≫ 32 ms
   the per-position read regressed.
4. Suspect the expert-gather access pattern / a scatter kernel: the 241 GB/s copy test
   is sequential; the MoE gather is scatter-heavy and could be running far below peak
   specifically in the multi-position batched case.

## HOW TO RESUME
- Free worker disk BEFORE serving (was 100% full, blocked Ray init → ActorHandleNotFound
  on every launch — that error is CLEANUP NOISE, root cause was the full disk). Freed
  `~/dspark-hs-reasoning-in` (86 GB) → worker 106 GB free. Magpie cache
  `~/dspark-distill-data/prepared/hidden_states` (205 GB) still there.
- Clean Ray restart after all the thrashing: `pkill -9 EngineCore` + `ray stop --force`
  both nodes, then `spark/start-ray-cluster.sh` (force-stops both, re-pins NCCL + GID).
- Serve the baseline: RUNBOOK §4b command (dspark K=2). Measure: stream chat completion
  temp0 300-tok, decode = completion_toks/(t_end−t_first). Per-pos accept from /metrics:
  `per_pos_total{position="N"} / num_drafts_total`.
- No-draft floor: same serve `--no-spec --hf-overrides '{"num_experts_per_tok":4}'`.

## GOTCHAS / KEY FACTS
- **Machine left altered**: deep C-states (state2/state3) DISABLED on both nodes (sudo).
  Harmless but not default — revert: `echo 0 > /sys/.../cpu*/cpuidle/state{2,3}/disable`.
- **Ray ActorHandleNotFoundError on serve launch = disk full**, not a code bug. Free disk.
- **spinner keep-awake test contaminates decode** (steals CPU) — dropped dspark to 7.5.
  Use it only to measure LATENCY, never decode throughput.
- **/proc/PID/environ is an exec-time snapshot** — Ray-propagated worker env (VLLM_* flags)
  is INVISIBLE there; check the worker's loaded .so via /proc/PID/maps instead.
- **acceptance is content-dependent** (creative/prose 0.67 weak, math 0.91 strong) — see
  session-16 dspark-distill handoff; the fine-tune effort concluded NULL.

## CORRECTNESS GATES
- Baseline sanity: no-draft floor should be ~14-15.6. If it's also ~half, the regression
  is broader than the verify path.
- The fix works iff dspark K=2 climbs back above the no-draft floor (net-positive) toward
  ~20 tok/s, greedy-coherent, acceptance unchanged (~0.77/0.56 pos-0/1).

## LINKS
- `MTP-DRAFT-COST.md` (draft-cost thesis; verify-side is THE lever), `RUNBOOK.md` §4b,
  `handoffs/03-dspark-acceptance-ablation.md` (21.8/15.6 baselines).
- dspark-distill effort (concluded null): `dspark-distill/HANDOFF.md` session 16g.
- Lab notebook: `~/Dev/howtospark/models/glm-5.2.md`.

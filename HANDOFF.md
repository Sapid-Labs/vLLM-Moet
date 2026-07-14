# HANDOFF — GLM-5.2 on 2× DGX Spark (branch spark-gb10)

Short "resume here" pointer. Deep docs: `spark/NVFP4-DENSE.md` (the shipped
lever, full design/repro), `spark/GOAL.md` (20 tok/s plan), `spark/RUNBOOK.md`
(serve/measure), `spark/handoffs/02-*.md` (how we got to 15).

## STATUS (2026-07-14, session 10)

**GLM-5.2 on 2× Spark now serves ~20 tok/s single-stream (greedy)** — up from
15 — via **NVFP4 weight-only on the attention + shared-expert linears** (stacks
on top of the shipped 2-bit pruned experts + MTP). Decode is bandwidth-bound;
after 2-bit experts, attention was ~60% of per-token bytes at FP8, so 4-bit-ing
it bought ~1.33×. Progression: FP8-attn 15 → NVFP4 big-3 ~18 → full attn+shared
~20. **Quality of the NVFP4 build NOT yet evaluated** (weights rel-L1 ~0.09).

- **Server is UP right now** on the full NVFP4 build (`nvfp4_dense_overlay`),
  TP2, healthy on :8000.
- **Code pushed:** `Sapid-Labs/vLLM-Moet` branch `spark-gb10` (commit 78345be).
- **Model pushed:** `sapidlabs/GLM-5.2-NVFP4-attn-experimental` (public,
  experimental, 8.4 GB NVFP4 attention delta + honest card; upload may still be
  finishing on nohup — check `~/nvfp4-hf-upload.log`).

## NEXT (agreed plan — order matters)

**REAP → drafter → eval**, with cheap sanity checks between (full battery only at
the end; it's slow). Rationale: the drafter is *fit to* the target's output
distribution, and REAP changes which experts are active → do all target-changing
steps first, fit the drafter last against the frozen model.

1. **REAP (do first)** — replace the frequency-based expert prune (drop 48
   coldest by traffic) with saliency-based REAP at the same 208 experts, to buy
   back quality. Quality move, NOT speed (same bytes/token).
   - **Pipeline (confirmed session 11):** REAP observer → per-layer expert
     saliency → top-208-per-layer keep-list in `spark/routing/keep208.json`
     format → `spark/routing/prune_planes.py <src> <dst> <keep.json>` (row-select,
     no requant, ~15 min) on the FULL `moe_w2_planes_tp2` (still on disk, 97 GB)
     → serve with the new p208 dir. `spark/prepack_planes.py` packs all experts;
     it is `prune_planes.py` that applies the keep-list. Run prune_planes on BOTH
     nodes' tp2 planes (same keep.json; rank-agnostic row-select).
   - **CODE DONE (session 11, reap `add-glm_moe_dsa-support` @ 02b838a):** GLM-5.2
     REAP support is complete. Observer/prune registries were already there; the
     missing piece was the **disk-streaming FP8 path** — the 357B model is >> RAM
     so calibration must stream layer-by-layer from disk, but `disk_stream` had no
     GLM converter and no FP8 dequant (it cast raw float8 bytes, ignoring
     `weight_scale_inv`). Added block-wise FP8 dequant + `glm_converter` (fuses
     per-expert gate/up/down into native batched params); 4 GLM smoke tests pass
     (incl. a synthetic real-layout FP8 checkpoint matched to bf16 ref, rel-L1
     <5%); hy3 tests still green. Env: run reap with **`~/venvs/hf/bin/python`
     `PYTHONPATH=src`** (reap not pip-installed; that venv has pytest+transformers
     w/ GlmMoeDsa).
   - **NEXT CONCRETE STEP:** run the observer on `~/models/hf/GLM-5.2-FP8` via
     `python -m reap.layerwise_prune --disk_stream ... --run_observer_only true`
     (needs the GPUs → **take down the live NVFP4 server first**; mind the Ray
     GPU-release footgun). Then saliency→keep-list→prune_planes. Sanity-check
     after: ~10-20 prompts coherence or GSM8K-50.
2. **Drafter (second, last model change)** — fine-tune the MTP head (layer 78)
   against the frozen NVFP4+REAP target to raise MTP acceptance (the real lever
   for effective throughput; sampled tok/s currently varies ~13-20 with
   acceptance). Bigger lift than REAP. Sanity-check after.
3. **Full quality battery (end)** — GPQA + GSM8K + IFEval + MMLU-Pro on the final
   build. This is the gate for any "same quality" public claim. Run it ALONE
   (never co-run — session-9 crash) and resumable (`--use_cache`), see below.

## HOW TO RESUME

**Serve the shipped NVFP4 build (2× Spark, TP2):**
```bash
cd ~/Dev/vLLM-Moet; M=~/models/hf/GLM-5.2-FP8
MODEL=$M/nvfp4_dense_overlay VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208 \
VLLM_NVFP4_DENSE=1 \
VLLM_NVFP4_TARGETS="o_proj,q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,fused_qkv_a_proj,gate_proj,up_proj,gate_up_proj,down_proj" \
MTP_K=1 VLLM_ENGINE_READY_TIMEOUT_S=2400 \
nohup bash spark/serve-glm52-tp2-mtp.sh > ~/serve-nvfp4.log 2>&1 &
```
Big-3-only cut = same but `VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj` +
`MODEL=$M/nvfp4_big3_overlay`. Plain FP8-attn baseline = drop `VLLM_NVFP4_DENSE`
+ `MODEL=$M` + `VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208`.

**Measure decode:** use **greedy (temperature 0)** for a clean number (~20);
sampled (temp>0) varies with MTP acceptance. Settle 3-5 warmups first (Marlin
FP4 kernels warm slowly). One-liner protocol in session-10 transcript / GOAL.md.

**Rebuild an overlay** (e.g. after REAP changes nothing here, but for new quant):
`python spark/prepack_nvfp4_linear.py --targets <basenames> --out <dir>` on BOTH
nodes (deterministic; peer needs the packer copied — it's at `~/prepack_nvfp4_linear.py`).

## GOTCHAS / KEY FACTS (hard-won this session)

- **NVFP4 code lives in site-packages** (`~/venvs/vllm-moet/.../vllm/`), captured
  in `patch/nvfp4-dense.patch` + `spark/nvfp4_dense_hook.py` (Dockerfile applies
  both). A fresh vllm install needs them applied. The PEER node's site-packages
  still has leftover debug prints (`NVFP4HOOKDBG`) — harmless, but re-sync the
  cleaned files if you rebuild: hook, parameter.py, deepseek_v2.py, deepseek_mtp.py.
- **Why the overlay works:** vLLM globs ALL *.safetensors (not just the index),
  so the overlay's NVFP4 tensor AND the symlinked original's fp8 tensor both load.
  Handled by `NVFP4SKIP2` guard (skip fp8→uint8-param) + `NVFP4ORPHAN` guards
  (skip fp8 `weight_scale_inv` with no nvfp4 param) in BOTH main + MTP loaders,
  stacked + non-stacked paths. Don't remove these.
- **GPU release between serves:** `kill -9` on the APIServer ORPHANS the Ray
  EngineCore + RayWorkerProc (one holds ~79 GB), which squat the GPUs → next boot
  fails "Cannot provide a placement group requiring 2.0 GPUs". Kill
  `EngineCore|RayWorkerProc|vllm serve` by PID on BOTH nodes, then `ray status`
  should show `0.0/2.0 GPU`.
- **Ray corrupts across many failed boots** (`ActorHandle ... across Ray
  sessions`) → `bash spark/start-ray-cluster.sh` (resolves RoCE GIDs, restarts
  clean).
- **Fresh NVFP4 compile > 600s** default engine-ready timeout → set
  `VLLM_ENGINE_READY_TIMEOUT_S=2400` (cached after first boot; ~4 min warm).
- **GB10 has no native FP4 MMA** → weight-only Marlin FP4 (4-bit read, bf16
  compute). Correct for bandwidth-bound decode. Adds a little dequant latency
  (measured 1.2× vs predicted 1.29× for big-3).
- **Never co-run evals** (session-9: IFEval co-ran with GPQA 5.5h → aiohttp
  "Session is closed" crash). Run alone, `--use_cache <dir>` + `--log_samples`
  (resumable), `max_length=8192` matching the server + `max_gen_toks=5000`
  (avoids the 8192-context overflow 400s), `num_concurrent=4`.

## CORRECTNESS GATES

- NVFP4 serve: coherent greedy output (verified — it explains memory-bound
  inference correctly) + greedy decode ≥ ~1.3× the FP8-attn baseline.
- After REAP / drafter: cheap sanity (coherence / GSM8K-50) each; full battery
  within noise of the 15 tok/s baseline (GSM8K ≥ ~90%) at the end.

## LINKS

- `spark/NVFP4-DENSE.md`, `spark/GOAL.md`, `spark/RUNBOOK.md`
- `spark/prepack_nvfp4_linear.py`, `spark/nvfp4_dense_hook.py`, `patch/nvfp4-dense.patch`
- HF: `sapidlabs/GLM-5.2-NVFP4-attn-experimental`,
  `sapidlabs/GLM-5.2-2bit-MoE-planes-pruned208-tp2` (2-bit planes)
- REAP tooling: `~/Dev/reap` (branch `add-glm_moe_dsa-support`)
- Eval harness: `~/Dev/howtospark/evals/` (run_battery.sh + run_eval.py)
